# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""amphione (the production rule) on 4 ranks under FSDP and DDP, dense and MoE
(experts, router with frozen row lengths), against its compute run unit by unit
on whole tensors in one process: the transport of the weights, the per-unit
state, the masks and the returned weights.
Runs on CPU/gloo when fewer than 4 accelerators are visible."""

import os
import sys
import tempfile

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from torchtitan.optimizers.disco.optimizer import _unit_type  # noqa: E402
from torchtitan.optimizers.disco.presets import AmphioneMap  # noqa: E402
from torchtitan.optimizers.disco.rules import StepCtx  # noqa: E402

pytestmark = pytest.mark.multi_gpu

FSDP = dict(dp_replicate=1, dp_shard=4, cp=1, ep=1)
DDP = dict(dp_replicate=4, dp_shard=1, cp=1, ep=1)
STEPS, LR, TOL = 3, 0.02, 1e-5
# (topology, MoE, amphione args); no MoE under pure DDP (not a supported setup)
CASES = {
    "fsdp4_dense": (FSDP, False, {}),
    "ddp4_dense": (DDP, False, {}),
    "fsdp4_dense_gain_beta": (FSDP, False, {"gain_beta": 0.9}),
    "fsdp4_dense_columns": (FSDP, False, {"columns": True}),
    "ddp4_dense_columns": (DDP, False, {"columns": True}),
    "fsdp4_moe": (FSDP, True, {}),
    "fsdp4_moe_frozen_router": (
        FSDP,
        True,
        {"frozen_length_units": ["moe.router.gate"]},
    ),
}


class TestAmphionePaths(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, topo, moe, args):
        device = self._device()
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        model_cfg = harness.tiny_model_config(moe=moe)
        model_cfg.layer.attention.n_kv_heads = 2  # MHA: one state shape per wqkv
        cfg = harness.trainer_config(
            model_cfg,
            compute="svd",
            communication_dtype="float32",
            rule="amphione",
            rule_args={"pre": None, **args},
            lr=LR,
            momentum=1.0,
            nesterov=False,
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        names = {id(p): n for n, p in harness.trainable(model)}
        entries = [
            e
            for opt in container.optimizers
            for e in opt.entries
            if isinstance(e.rule.compute, AmphioneMap)
        ]
        self.assertTrue(entries, "no parameter runs amphione")
        if moe:
            self.assertTrue(any("router" in names[id(e.param)] for e in entries))
            self.assertTrue(any("experts" in names[id(e.param)] for e in entries))
        frozen = set(args.get("frozen_length_units", ()))
        if frozen:
            self.assertTrue(
                any(_unit_type(e, u) in frozen for e in entries for u in e.units)
            )
        ref = {
            names[id(e.param)]: harness.full_tensor(e.param).float().clone()
            for e in entries
        }
        state = {
            (names[id(e.param)], k): torch.zeros(*e.rule.compute.state_shape(*u.shape))
            for e in entries
            for k, u in enumerate(e.units)
        }
        failures = []
        for step in range(1, STEPS + 1):
            grads = harness.set_synthetic_grads(model, step, device)
            container.step()
            container.zero_grad()
            for e in entries:
                cm = e.rule.compute
                name = names[id(e.param)]
                W2 = ref[name].reshape(-1, e.view[1])
                G2 = grads[name].float().cpu().reshape(-1, e.view[1])
                out = W2.clone()
                for k, u in enumerate(e.units):
                    rows = [(a, a + n) for a, n in u.rows.runs]
                    Wu = torch.cat([W2[a:b] for a, b in rows])
                    Gu = torch.cat([G2[a:b] for a, b in rows])
                    t = _unit_type(e, u)  # the masks from their definitions
                    ctx = StepCtx(
                        LR,
                        0.0,
                        step,
                        colmask=torch.tensor([t in cm.col_units])
                        if cm.columns
                        else None,
                        frozen=torch.tensor([t in frozen]) if frozen else None,
                    )
                    res = cm.run(Gu[None], Wu[None], state[name, k][None], ctx)
                    state[name, k] = res.state[0]
                    off = 0
                    for a, b in rows:
                        out[a:b] = res.weight[0][off : off + b - a]
                        off += b - a
                ref[name] = out.reshape(ref[name].shape)
            for e in entries:
                name = names[id(e.param)]
                got, want = harness.full_tensor(e.param).float().cpu(), ref[name]
                d = (got - want).abs().max().item()
                if d > TOL * max(1.0, want.abs().max().item()):
                    failures.append(f"step {step} {name}: max |diff| {d:.3e}")
        ok = harness.all_ranks_agree(not failures, device)
        self.assertTrue(ok, "; ".join(failures[:5]))


def _case(name):
    @with_comms
    def test(self):
        self._check(*CASES[name])

    return test


for _name in CASES:  # one test per case: each builds its meshes in fresh processes
    setattr(TestAmphionePaths, f"test_{_name}", _case(_name))
