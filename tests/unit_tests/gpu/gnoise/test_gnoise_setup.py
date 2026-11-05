# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gnoise set-up, guards and logging ownership: refusals only when enabled (T16, T21),
nothing installed when off (T20), no checkpoint state (T17), the torch pin (T18),
exactly once under the chunked loss (T22), the entry guard (T23), one owner per key
(T13), the router modes (T26)."""

import inspect
import os
import sys
import tempfile

import pytest
import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard import (
    _fsdp_collectives,
    _fsdp_param,
    _fsdp_param_group,
    _fsdp_state,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gn_harness as H  # noqa: E402

from torchtitan.optimizers.gnoise import (  # noqa: E402, F401
    GradNoiseConfig,
    refuse_unsupported,
)

pytestmark = pytest.mark.multi_gpu


def _config(**gnoise):
    import harness

    cfg = harness.trainer_config(H.gauge_config(), log_norm_freq=10)
    cfg.metrics.log_freq = 10
    cfg.training.disable_accelerator_graphs = True
    for k, v in gnoise.items():
        setattr(cfg.optim.optimizer.gnoise, k, v)
    return cfg


@pytest.mark.parametrize(
    "change",
    [
        ("parallelism", "tensor_parallel_degree", 2),
        ("parallelism", "context_parallel_degree", 2),
        ("parallelism", "pipeline_parallel_degree", 2),
        ("training", "disable_accelerator_graphs", False),
        ("parallelism", "fsdp_enable_cpu_offloading", True),
    ],
)
def test_refusals_only_when_enabled(change):
    """T16 / T21: each unsupported set-up builds with gnoise off and raises with it on."""
    section, key, value = change
    off = _config()
    setattr(getattr(off, section), key, value)
    off._resolve_norm_logging()  # no error
    on = _config(enable=True)
    setattr(getattr(on, section), key, value)
    with pytest.raises(ValueError, match="gnoise is not supported"):
        on._resolve_norm_logging()


def test_refuses_compiled_autograd(monkeypatch):
    monkeypatch.setattr(torch._dynamo.config, "compiled_autograd", True)
    _config()._resolve_norm_logging()
    with pytest.raises(ValueError, match="compiled autograd"):
        _config(enable=True)._resolve_norm_logging()


def test_knob_rules():
    with pytest.raises(ValueError, match="knobs are set"):
        GradNoiseConfig(research_freq=100).validate(log_norm_freq=10)
    with pytest.raises(ValueError, match="research_freq 0"):
        GradNoiseConfig(enable=True, research_topk=[2]).validate(log_norm_freq=10)
    with pytest.raises(ValueError, match="multiple of the E1 interval"):
        GradNoiseConfig(enable=True, research_freq=15).validate(log_norm_freq=10)
    with pytest.raises(ValueError, match="log_norm_freq > 0"):
        GradNoiseConfig(enable=True).validate(log_norm_freq=0)
    GradNoiseConfig(enable=True, research_freq=100, research_sizes=[1, 2]).validate(
        log_norm_freq=10
    )
    GradNoiseConfig().validate(log_norm_freq=0)


def test_torch_api_pin():
    """T18: the private FSDP2 members the hook relies on (torch 2.16)."""
    G = _fsdp_param_group.FSDPParamGroup
    assert list(inspect.signature(G.post_backward).parameters) == ["self", "unused"]
    assert isinstance(inspect.getattr_static(G, "is_unsharded"), property)
    src = inspect.getsource(G.post_backward)
    # the order the entry hook depends on: the no-reduce return, then the
    # gradients taken from the unsharded leaves, reshard, reduce
    red = src.index("foreach_reduce(")
    order = [
        src.index("if not self.reduce_grads:"),
        src.index("self._take_unsharded_grads_to_reduce()"),
        src.rindex("self.reshard()", 0, red),
        red,
    ]
    assert order == sorted(order), order
    take = inspect.getsource(G._take_unsharded_grads_to_reduce)
    take += inspect.getsource(G._get_unsharded_grad_to_reduce)
    # the gradient reduced is the leaf's .grad, which the hook reads
    assert "unsharded_param.grad is not None" in take
    assert "return param.unsharded_grad_data" in take
    P = _fsdp_param.FSDPParam
    assert "self.unsharded_param.grad" in inspect.getsource(P.unsharded_grad_data.fget)
    init = inspect.getsource(P.__init__) + inspect.getsource(P)
    for a in ("self.sharded_param", "self.param_dtype", "self._unsharded_param"):
        assert a in init, a
    assert "self._fsdp_param_groups" in inspect.getsource(
        _fsdp_state.FSDPState.__init__
    )
    assert list(inspect.signature(_fsdp_collectives.foreach_reduce).parameters)[:2] == [
        "fsdp_params",
        "unsharded_grads",
    ]
    # both callers look post_backward up on the instance
    assert "ctx.param_group.post_backward()" in inspect.getsource(_fsdp_param_group)
    assert "fsdp_param_group.post_backward()" in inspect.getsource(_fsdp_state)


class TestSetup(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self):
        if self.device_type == "cpu":
            H.patch_dense_attention()
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _groups(self, model):
        return [
            g
            for m in model.modules()
            if isinstance(m, FSDPModule)
            for g in m._get_fsdp_state()._fsdp_param_groups
        ]

    @with_comms
    def test_off_installs_nothing(self):
        """T20: off, every FSDP group keeps the class's post_backward and no gnoise exists."""
        with tempfile.TemporaryDirectory() as tmp:
            _, model, cont, _ = H.build(
                dict(dp_replicate=1, dp_shard=4), self._device(), tmp
            )
            assert cont.gnoise is None
            groups = self._groups(model)
            assert groups and all("post_backward" not in vars(g) for g in groups)

    @with_comms
    def test_on_wraps_every_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, model, cont, _ = H.build(
                dict(dp_replicate=1, dp_shard=4),
                self._device(),
                tmp,
                gnoise=dict(enable=True),
            )
            assert cont.gnoise is not None
            assert all("post_backward" in vars(g) for g in self._groups(model))

    @with_comms
    def test_no_checkpoint_state(self):
        """T17: after an armed step and an optimizer step the EMAs exist, and nothing of
        gnoise is in the container's state_dict (it resets at resume instead)."""
        dev = self._device()
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                dict(dp_replicate=1, dp_shard=4), dev, tmp, gnoise=dict(enable=True)
            )
            cont.gnoise.validate_layout(accum=1)
            H.run_step(
                model, pd, cfg, cont, 1, [H.batch(self.rank, 2 * H.SEQ, dev)], 8 * H.SEQ
            )
            cont.step()
            assert cont.gnoise.ema
            ema_ids = {id(t) for t in cont.gnoise.ema.values()}

            def walk(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        assert "gnoise" not in str(k)
                        walk(v)
                elif isinstance(x, (list, tuple)):
                    for v in x:
                        walk(v)
                elif torch.is_tensor(x):
                    assert id(x) not in ema_ids

            walk(cont.state_dict())

    @with_comms
    def test_exactly_once_chunked(self):
        """T22: [norm, lm_head] under the chunked loss: every covered param, the final
        norm's gain included, is recorded exactly once per microbatch. The embedding,
        which FSDP leaves sharded in backward, is read on the armed step and left
        sharded in backward again after it."""
        dev = self._device()
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                dict(dp_replicate=1, dp_shard=4),
                dev,
                tmp,
                gnoise=dict(enable=True),
                chunks=8,
            )
            gn = cont.gnoise
            gn.validate_layout(accum=2)
            mbs = [H.batch(self.rank + 10 * k, 2 * H.SEQ, dev) for k in range(2)]
            names = {pp.name for pp in gn.params}
            assert {"norm.weight", "lm_head.weight"} <= names
            assert gn.backward_unshard_off  # the embedding's group
            H.run_step(model, pd, cfg, cont, 1, mbs, 8 * 2 * H.SEQ)
            for rec in gn.records:
                assert all(rec[pp.i] == 1 for pp in gn.params), dict(rec)
            assert not any(g.unshard_in_backward for g in gn.backward_unshard_off)

    @with_comms
    def test_entry_guard(self):
        """T23: a group left sharded in backward is refused, not read. FSDP skips
        the embedding's backward unshard and gnoise unshards it on armed steps;
        with that turned off the embedding group enters sharded."""
        dev = self._device()
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                dict(dp_replicate=1, dp_shard=4), dev, tmp, gnoise=dict(enable=True)
            )
            assert cont.gnoise.backward_unshard_off  # the embedding's group
            cont.gnoise.backward_unshard_off = []
            model.tok_embeddings.set_unshard_in_backward(False)
            cont.gnoise.validate_layout(accum=1)
            with pytest.raises(
                RuntimeError, match="post_backward entered with the group sharded"
            ):
                H.run_step(
                    model,
                    pd,
                    cfg,
                    cont,
                    1,
                    [H.batch(self.rank, 2 * H.SEQ, dev)],
                    8 * H.SEQ,
                )

    @with_comms
    def test_one_owner_per_key(self):
        """T13 (CPU): local mode, the ranks' keys partition all keys; gather mode, rank 0."""
        dev = self._device()
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                dict(dp_replicate=2, dp_shard=2), dev, tmp, gnoise=dict(enable=True)
            )
            gn = cont.gnoise
            gn.validate_layout(accum=1)
            for local in (True, False):
                cont.log_metrics_locally = local
                H.run_step(
                    model,
                    pd,
                    cfg,
                    cont,
                    1,
                    [H.batch(self.rank % 2, 2 * H.SEQ, dev)],
                    8 * H.SEQ,
                )
                every = set(gn.metrics)
                mine = gn.collect()
                got = [None] * 4
                torch.distributed.all_gather_object(got, sorted(mine))
                union = [k for keys in got for k in keys]
                assert len(union) == len(set(union)), "a key is logged twice"
                assert set(union) == every, (len(set(union)), len(every))
                if not local:
                    assert len(got[0]) == len(every)

    def _router(self, aux):
        dev = self._device()
        mcfg = H.gauge_config(moe=True)
        mcfg["aux_loss"] = aux
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                dict(dp_replicate=1, dp_shard=4),
                dev,
                tmp,
                gnoise=dict(enable=True),
                model_cfg=mcfg,
            )
            gn = cont.gnoise
            gn.validate_layout(accum=1)
            m = H.run_step(
                model, pd, cfg, cont, 1, [H.batch(3, 2 * H.SEQ, dev)], 8 * H.SEQ
            )
        return {pp.name for pp in gn.params}, m

    def _router_covered(self, aux):
        names, m = self._router(aux)
        r = sorted(n for n in names if ".router." in n)
        assert r, "router not covered"
        for n in r:
            key = n.removesuffix(".weight")
            assert abs(float(m[f"gnoise_tr/{key}"])) <= 1e-3 * 2 * H.SEQ * abs(
                float(m[f"gnoise_sig/{key}"])
            )

    @with_comms
    def test_router_no_aux_loss(self):
        """T26: covered with bias balancing only (identical small batches: tr = 0)."""
        self._router_covered(None)

    @with_comms
    def test_router_left_out_with_a_balance_loss(self):
        """Upstream's balance losses couple a microbatch's tokens: no per-token sum."""
        for loss_type in ("microbatch_wise", "batch_wise"):
            names, _ = self._router(dict(coeff=1e-3, loss_type=loss_type))
            assert not any(".router." in n for n in names), loss_type
