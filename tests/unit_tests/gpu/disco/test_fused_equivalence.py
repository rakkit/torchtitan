# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused wqkv / w13 under DiSCO give the split parameters' updates and keys.

The fused model gets the split model's weights and gradients (interleaved /
stacked the way the modules lay them out); with the fp32 + svd transport the
updates must match bitwise, and a logging step must produce the same key set.
"""

import os
import sys
import tempfile

import pytest
import torch
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

pytestmark = pytest.mark.multi_gpu

TOPOLOGIES = {
    "fsdp4": dict(dp_replicate=1, dp_shard=4, cp=1, ep=1),
    "ddp4": dict(dp_replicate=4, dp_shard=1, cp=1, ep=1),
    "hsdp22": dict(dp_replicate=2, dp_shard=2, cp=1, ep=1),
}
N_HEADS, N_KV, HEAD_DIM = 2, 1, 32


def _fuse_qkv(wq, wk, wv):
    hpk = N_HEADS // N_KV
    q = wq.view(N_KV, hpk, HEAD_DIM, -1)
    k = wk.view(N_KV, 1, HEAD_DIM, -1)
    v = wv.view(N_KV, 1, HEAD_DIM, -1)
    return torch.cat([q, k, v], dim=1).reshape(-1, wq.shape[-1])


def _split_name(name: str):
    """Fused parameter name -> the split parameters it holds, and how to fuse them."""
    if name.endswith("qkv_linear.wqkv.weight"):
        base = name[: -len("wqkv.weight")]
        return [base + "wq.weight", base + "wk.weight", base + "wv.weight"], _fuse_qkv
    if name.endswith("feed_forward.w13.weight"):
        base = name[: -len("w13.weight")]
        return [base + "w1.weight", base + "w3.weight"], lambda a, b: torch.stack(
            [a, b]
        )
    return [name], lambda t: t


@torch.no_grad()
def _copy_into(fused_model, split_full: dict):
    """Load the split model's full tensors into the fused model."""
    for name, p in harness.trainable(fused_model):
        parts, fuse = _split_name(name)
        full = fuse(*[split_full[n] for n in parts]).detach()
        if isinstance(p, DTensor):
            p.to_local().copy_(
                distribute_tensor(full, p.device_mesh, p.placements).to_local()
            )
        else:
            p.copy_(full)


def _set_grads(fused_model, split_grads: dict):
    for name, p in harness.trainable(fused_model):
        parts, fuse = _split_name(name)
        g = fuse(*[split_grads[n] for n in parts]).to(p.dtype).detach()
        p.grad = (
            distribute_tensor(g, p.device_mesh, p.placements)
            if isinstance(p, DTensor)
            else g.clone()
        )


class TestFusedEquivalence(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self):
        return (
            torch.device("cpu")
            if self.device_type == "cpu"
            else torch.device(self.device_type, self.rank)
        )

    def _check(self, topo, *, pre="identity", log=False):
        device = self._device()
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        models, containers = {}, {}
        for fused in (False, True):
            cfg = harness.trainer_config(
                harness.tiny_model_config(moe=False, fuse_qkv=fused, fuse_w13=fused),
                compute="svd",
                communication_dtype="float32",
                pre=pre,
                log_norm_freq=1 if log else 0,
                **topo,
            )
            with tempfile.TemporaryDirectory() as tmp:
                models[fused], containers[fused] = harness.build_model_and_optimizer(
                    cfg, pd, device, tmp
                )
        split_full = {
            n: harness.full_tensor(p) for n, p in harness.trainable(models[False])
        }
        _copy_into(models[True], split_full)
        failures = []
        keys = {}
        for step in (1, 2):
            grads = harness.set_synthetic_grads(models[False], step, device)
            _set_grads(models[True], grads)
            for fused in (False, True):
                if log:
                    containers[fused].arm_norm_computation()
                containers[fused].step()
                containers[fused].zero_grad()
                if log:
                    keys[fused] = set(containers[fused].collect_metrics(step))
            split_full = {
                n: harness.full_tensor(p) for n, p in harness.trainable(models[False])
            }
            for name, p in harness.trainable(models[True]):
                parts, fuse = _split_name(name)
                want = fuse(*[split_full[n] for n in parts])
                got = harness.full_tensor(p)
                # A reduced (col / mat) statistic travels in a differently packed
                # all-reduce for the two models: equal to rounding, not to the bit.
                tol = 0.0 if pre == "identity" else 1e-6
                d = (got.float() - want.float()).abs()
                if d.max().item() > tol * max(1.0, want.float().abs().max().item()):
                    failures.append(
                        f"step {step} {name}: max |diff| {d.max().item():.3e}"
                    )
        if log and keys[True] != keys[False]:
            failures.append(
                f"metric keys differ: only split {sorted(keys[False] - keys[True])[:4]}, "
                f"only fused {sorted(keys[True] - keys[False])[:4]}"
            )
        ok = harness.all_ranks_agree(not failures, device)
        if failures and self.rank == 0:
            print("\n".join(failures[:12]))
        self.assertTrue(ok, failures[:3])

    @with_comms
    def test_fsdp4(self):
        self._check(TOPOLOGIES["fsdp4"])

    @with_comms
    def test_fsdp4_col_pre_and_keys(self):
        self._check(TOPOLOGIES["fsdp4"], pre="col-l2", log=True)

    @with_comms
    def test_ddp4(self):
        self._check(TOPOLOGIES["ddp4"], pre="mat-l2", log=True)

    @with_comms
    def test_hsdp22(self):
        self._check(TOPOLOGIES["hsdp22"])
