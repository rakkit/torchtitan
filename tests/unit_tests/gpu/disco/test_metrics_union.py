# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Who logs which DiSCO metric, and are the values right.

On a norm-logging step every rank computes the metrics of the units it owns.
Two ownership modes must both reconstruct the full key set exactly once:

* gather mode (default): the storing rank holds every key, other ranks none;
* local mode (`log_metrics_locally`): the union over ranks is the full set and
  the per-rank sets are pairwise disjoint.

The full key set is derived from the parameter list alone, never from the
optimizer, and two values with a closed form are checked against the
reference: `frobenius_norm` of the update (-lr*u) and of the post-step weight.
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from reference import rules_from_container  # noqa: E402
from test_step_paths import _SameLMOReference  # noqa: E402

from torchtitan.optimizers.helpers.norm_helper import get_norms_to_log  # noqa: E402
from torchtitan.optimizers.helpers.radial_helper import (  # noqa: E402
    RADIAL_METRIC_NAMES,
)
from torchtitan.optimizers.utils import (  # noqa: E402
    remove_orig_mod_and_weight_for_p_name,
)

pytestmark = pytest.mark.multi_gpu


def aus_reference(wb: torch.Tensor, wa: torch.Tensor) -> tuple[float, float]:
    """`aus_frobenius` and `aus_rms_to_rms`: the distance between the normalised
    pre- and post-update weights, from an SVD. Both are transpose-invariant."""
    wb, wa = wb.double(), wa.double()
    fro = (wb / wb.norm() - wa / wa.norm()).norm()

    def spec(x):
        return torch.linalg.matrix_norm(x, ord=2)

    return float(fro), float(spec(wb / spec(wb) - wa / spec(wa)))


def expected_keys(model, norms: list[str]) -> set[str]:
    keys = set()
    for name, p in harness.trainable(model):
        clean = remove_orig_mod_and_weight_for_p_name(name)
        if p.numel() == 1:
            keys.add(f"scalar_param_supremum/{clean}")
            continue
        prefixes = (
            [f"ep_{e}/{clean}" for e in range(p.shape[0])] if p.ndim == 3 else [clean]
        )
        for pre in prefixes:
            for n in norms:
                keys.add(f"track_update_{n}/{pre}")
                keys.add(f"track_param_{n}/{pre}")
            for r in RADIAL_METRIC_NAMES:
                keys.add(f"track_radial_{r}/{pre}")
    return keys


class TestMetricsUnion(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, topo: dict, *, local: bool, moe: bool = True):
        device = self._device()
        pd = harness.parallel_dims(
            world_size=self.world_size, device_type=device.type, **topo
        )
        cfg = harness.trainer_config(
            harness.tiny_model_config(moe=moe),
            compute="svd",
            log_norm_freq=1,
            norms_to_log=("all",),
            communication_dtype="float32",
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        container.log_metrics_locally = local

        params = {
            n: harness.full_tensor(p).float() for n, p in harness.trainable(model)
        }
        before = {n: t.clone() for n, t in params.items()}
        ref = _SameLMOReference(params, rules_from_container(container, model))

        grads = harness.set_synthetic_grads(model, 1, device)
        container.arm_norm_computation()
        container.step()
        container.zero_grad()
        ref.step(grads)
        metrics = container.collect_metrics(1)
        mine = set(metrics.keys())
        # Scalars reach the logger as numbers: a 0-d device tensor costs a sync.
        on_device = [
            k for k, v in metrics.items() if torch.is_tensor(v) and v.numel() == 1
        ]
        opt = container.optimizers[0]
        kept = [
            (ex, key)
            for ex in (opt._ex_all, opt._ex_row, *opt._ex_vocab)
            if ex is not None
            for key in ex._ws
            if key[0] in ("w", "graw", "u")
        ]
        problems_kept = [f"logging buffers kept after the step: {kept}"] if kept else []
        if on_device:
            problems_kept.append(
                f"{len(on_device)} scalars as tensors: {on_device[:2]}"
            )
        # The embedding and the head land whole in their arrival buffer.
        problems_kept += [
            f"vocabulary unit {vx.units[0].index} is copied after arrival"
            for vx in opt._ex_vocab
            if vx.owned and not vx.in_place
        ]

        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, mine)
        expected = expected_keys(
            model, get_norms_to_log(list(cfg.optimizer.norms_to_log))
        )

        union = set().union(*gathered)
        total = sum(len(s) for s in gathered)
        problems = list(problems_kept)
        if union != expected:
            problems.append(
                f"union != expected: missing {sorted(expected - union)[:5]}, extra {sorted(union - expected)[:5]}"
            )
        if total != len(expected):
            problems.append(
                f"keys counted across ranks {total} != {len(expected)} (duplicates or drops)"
            )
        if not local:
            holders = [r for r, s in enumerate(gathered) if s]
            if holders != [0]:
                problems.append(
                    f"gather mode: ranks holding keys = {holders}, expected [0]"
                )

        # Values with a closed form, on whichever rank holds them.
        lr = cfg.optimizer.groups["default"]["lr"]
        for name, p in harness.trainable(model):
            clean = remove_orig_mod_and_weight_for_p_name(name)
            if p.numel() == 1:
                continue
            u = ref.last_update[name]
            w = ref.params[name]
            wb = before[name]
            if p.ndim == 3:
                items = [
                    (f"ep_{e}/{clean}", u[e], w[e], wb[e]) for e in range(p.shape[0])
                ]
            else:
                items = [(clean, u, w, wb)]
            for pre, u_e, w_e, wb_e in items:
                fro, r2r = aus_reference(wb_e, w_e)
                for key, want, tol in (
                    (f"track_radial_aus_frobenius/{pre}", fro, 1e-4),
                    (f"track_radial_aus_rms_to_rms/{pre}", r2r, 2e-3),
                ):
                    if key in metrics:
                        got = float(metrics[key])
                        if abs(got - want) > tol * max(abs(want), 1e-6):
                            problems.append(f"{key}: {got:.6e} vs reference {want:.6e}")
                ku, kw = (
                    f"track_update_frobenius_norm/{pre}",
                    f"track_param_frobenius_norm/{pre}",
                )
                if ku in metrics:
                    got = float(metrics[ku])
                    want = (lr * u_e).norm().item()
                    if abs(got - want) > 1e-4 * max(1.0, abs(want)):
                        problems.append(f"{ku}: {got:.6e} vs reference {want:.6e}")
                if kw in metrics:
                    got = float(metrics[kw])
                    want = w_e.norm().item()
                    if abs(got - want) > 1e-4 * max(1.0, abs(want)):
                        problems.append(f"{kw}: {got:.6e} vs reference {want:.6e}")

        ok = harness.all_ranks_agree(not problems, device)
        if problems and self.rank == 0:
            print("\n".join(problems[:20]))
        self.assertTrue(ok, f"rank {self.rank}: {problems[:3]}")

    def _check_radial_checkpoint(self, topo: dict, *, fused: bool = False):
        """A unit's radial accumulators advance on its metric owner only, and
        DCP saves one copy of a plain tensor. After a save, a wipe and a load,
        every rank must hold the owner's values -- the owner is the rank whose
        accumulators moved (replicas agree)."""
        device = self._device()
        pd = harness.parallel_dims(
            world_size=self.world_size, device_type=device.type, **topo
        )
        cfg = harness.trainer_config(
            harness.tiny_model_config(moe=True, fuse_qkv=fused, fuse_w13=fused),
            compute="svd",
            log_norm_freq=1,
            norms_to_log=("all",),
            communication_dtype="float32",
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        harness.set_synthetic_grads(model, 1, device)
        container.arm_norm_computation()
        container.step()
        container.zero_grad()
        container.collect_metrics(1)
        opt = container.optimizers[0]
        dense = [e for e in opt.entries if e.family.value != "expert"]

        def radial():
            return torch.cat(  # a fused param keeps one slot per unit
                [
                    t.float().reshape(-1)
                    for e in dense
                    for t in opt.state[e.param]["radial_state"].values()
                ]
            )

        seen = [torch.empty_like(radial()) for _ in range(self.world_size)]
        dist.all_gather(seen, radial())
        stacked = torch.stack(seen)
        want = stacked.gather(0, stacked.abs().argmax(0, keepdim=True))[0]
        path = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(path)
        pg = dist.new_group(
            backend="gloo"
        )  # async_save stages on the CPU, as the trainer's
        ret = dcp.async_save(
            container.state_dict(), checkpoint_id=path[0], process_group=pg
        )
        getattr(ret, "upload_completion", ret).result()
        for e in dense:
            for t in opt.state[e.param]["radial_state"].values():
                t.zero_()
        sd = container.state_dict()
        dcp.load(sd, checkpoint_id=path[0])
        container.load_state_dict(sd)
        got = radial()
        lost = int((got != want).sum())
        ok = harness.all_ranks_agree(lost == 0 and bool(want.any()), device)
        self.assertTrue(
            ok, f"rank {self.rank}: {lost} of {want.numel()} accumulators lost"
        )

    @with_comms
    def test_fsdp4_radial_survives_a_checkpoint(self):
        self._check_radial_checkpoint(dict(dp_replicate=1, dp_shard=4, cp=1, ep=1))

    @with_comms
    def test_fsdp4_fused_radial_survives_a_checkpoint(self):
        """A fused param's units have different owners: each slot must come
        back from its own owner, not the sum over them."""
        self._check_radial_checkpoint(
            dict(dp_replicate=1, dp_shard=4, cp=1, ep=1), fused=True
        )

    @with_comms
    def test_hsdp22_radial_survives_a_checkpoint(self):
        self._check_radial_checkpoint(dict(dp_replicate=2, dp_shard=2, cp=1, ep=1))

    @with_comms
    def test_fsdp4_gather(self):
        self._check(dict(dp_replicate=1, dp_shard=4, cp=1, ep=1), local=False)

    @with_comms
    def test_fsdp4_local(self):
        self._check(dict(dp_replicate=1, dp_shard=4, cp=1, ep=1), local=True)

    @with_comms
    def test_hsdp22_local(self):
        self._check(dict(dp_replicate=2, dp_shard=2, cp=1, ep=1), local=True)

    @with_comms
    def test_ddp4_dense_local(self):
        # Pure DDP x MoE is unsupported (see test_step_paths); dense covers the path.
        self._check(dict(dp_replicate=4, dp_shard=1, cp=1, ep=1), local=True, moe=False)
