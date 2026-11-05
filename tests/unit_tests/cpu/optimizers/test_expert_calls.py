# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An expert block (same-shape expert parameters of consecutive layers) runs
its rule in batched calls of whole layers, up to `_EXPERT_CALL_BYTES` per fp32
temporary; its rule states are views of one buffer per block. One call per
layer must give the weights and states of one call over the block, bitwise,
and the views must survive a checkpoint round trip."""

import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout, stacked_units
from torchtitan.optimizers.disco.optimizer import DiSCO

E, F, D, LAYERS = 4, 16, 8, 3


@pytest.fixture(scope="module", autouse=True)
def one_rank():
    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


def _names():
    out = []
    for i in range(LAYERS):
        out += [
            f"layers.{i}.feed_forward.routed_experts.w13.weight",
            f"layers.{i}.feed_forward.routed_experts.w2.weight",
        ]
    return out


def _disco(cap: int, device: str = "cpu"):
    g = torch.Generator().manual_seed(0)
    shapes = [(E, 2, F, D), (E, D, F)] * LAYERS
    params = [
        nn.Parameter((torch.randn(*s, generator=g) * 0.3).to(device)) for s in shapes
    ]
    layouts = {}
    for n in _names():
        layouts[n] = (
            ParamLayout("expert_batch", units=stacked_units(("w1", "w3"), F, D))
            if "w13" in n
            else ParamLayout("expert_batch")
        )
    opt = DiSCO(
        [
            {
                "params": params,
                "param_names": _names(),
                "momentum": 0.95,
                "nesterov": True,
            }
        ],
        parallelism_context=_NoParallel(),
        lr=0.03,
        rule="amphione",
        args={"compute": "svd"},
        param_layouts=layouts,
    )
    opt._EXPERT_CALL_BYTES = cap
    return opt, params


def _steps(opt, params, n=3):
    calls = []
    real = opt._run
    opt._run = lambda *a, **k: calls.append(len(a[4])) or real(*a, **k)
    for s in range(n):
        g = torch.Generator().manual_seed(100 + s)
        for p in params:
            p.grad = (0.1 * torch.randn(p.shape, generator=g)).to(p.device)
        opt.step()
    opt._run = real
    return calls


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"),
        ),
    ],
)
def test_one_call_per_layer_is_one_call_over_the_block(device):
    """Bitwise on CPU. On CUDA (the compiled path) the batched kernels depend
    on the batch count: the two agree to fp32 rounding (measured max relative
    1.1e-7 after 3 steps), so only a block above the cap changes rounding."""
    one, p1 = _disco(1 << 40, device)
    per_layer, p2 = _disco(4 * 2 * E * F * D, device)  # one layer's w13 per call
    c1, c2 = _steps(one, p1), _steps(per_layer, p2)
    assert len(c2) > len(c1)  # the cap split the block's calls
    tol = dict(rtol=0, atol=0) if device == "cpu" else dict(rtol=1e-6, atol=1e-7)
    for a, b in zip(p1, p2):
        torch.testing.assert_close(a, b, **tol)
        torch.testing.assert_close(
            one.state[a]["rule_state"], per_layer.state[b]["rule_state"], **tol
        )


def test_the_block_states_are_views_and_survive_a_reload():
    opt, params = _disco(1 << 40)
    _steps(opt, params, 2)
    big = opt._big_state[0]
    st = opt.state[params[0]]["rule_state"]
    assert st.data_ptr() == big.data_ptr()  # a view of the block's buffer
    want = {i: opt.state[p]["rule_state"].clone() for i, p in enumerate(params)}
    sd = opt.state_dict()
    for i, s in sd["state"].items():  # handed out as copies
        assert s["rule_state"].untyped_storage().nbytes() == s["rule_state"].nbytes
    opt.load_state_dict(sd)
    for i, p in enumerate(params):
        assert torch.equal(opt.state[p]["rule_state"], want[i])
    assert opt.state[params[0]]["rule_state"].data_ptr() == opt._big_state[0].data_ptr()


def test_the_owners_batch_states_round_trip_through_dcp(tmp_path):
    """The owners' per-unit states are views of one buffer per step batch; a
    real dcp.save / dcp.load of them (as the checkpointer does) restores every
    unit's state exactly, and the next step is the uninterrupted run's."""
    import torch.distributed.checkpoint as dcp

    from torchtitan.components.optim.utils import (
        get_flat_optim_state_dict,
        init_optim_state,
        load_flat_optim_state_dict,
    )

    names = ["a.weight", "b.weight", "c.weight"]

    def build():
        g = torch.Generator().manual_seed(1)
        ps = [nn.Parameter(torch.randn(16, 8, generator=g) * 0.3) for _ in names]
        return ps, DiSCO(
            [{"params": ps, "param_names": names, "momentum": 0.95, "nesterov": True}],
            parallelism_context=_NoParallel(),
            lr=0.03,
            rule="amphione",
            args={"compute": "svd"},
            param_layouts={n: ParamLayout("matrix") for n in names},
        )

    class Flat:
        def __init__(self, o):
            self.o = o

        def state_dict(self):
            init_optim_state(self.o)
            return get_flat_optim_state_dict(self.o)

        def load_state_dict(self, sd):
            init_optim_state(self.o)
            for gr in self.o.param_groups:
                for k in DiSCO._CONFIG_ONLY_GROUP_KEYS:
                    if k in gr:
                        for n in names:
                            sd.setdefault(f"param_groups.{n}.{k}", gr[k])
            load_flat_optim_state_dict(self.o, sd)

    def step(ps, o, seed):
        g = torch.Generator().manual_seed(seed)
        for p in ps:
            p.grad = 0.1 * torch.randn(p.shape, generator=g)
        o.step()

    ps, opt = build()
    assert opt._batch_state  # the views exist
    for s in (1, 2):
        step(ps, opt, s)
    dcp.save({"optimizer": Flat(opt)}, checkpoint_id=str(tmp_path))
    ps2, opt2 = build()
    with torch.no_grad():
        for a, b in zip(ps, ps2):
            b.copy_(a)
    dcp.load({"optimizer": Flat(opt2)}, checkpoint_id=str(tmp_path))
    for a, b in zip(ps, ps2):
        for k, v in opt.state[a]["rule_state"].items():
            assert torch.equal(opt2.state[b]["rule_state"][k], v)
    step(ps, opt, 3)
    step(ps2, opt2, 3)
    for a, b in zip(ps, ps2):
        assert torch.equal(a, b)
