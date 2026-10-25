# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""angular_muown (native, and the Stage-2 cells) on 4 ranks under FSDP and DDP,
against its compute run unit by unit on whole tensors in one process: the
transport of the weights, the per-unit state and the returned weights.
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

from torchtitan.optimizers.disco.presets import AngularMuownMap  # noqa: E402
from torchtitan.optimizers.disco.rules import StepCtx  # noqa: E402

pytestmark = pytest.mark.multi_gpu

TOPOLOGIES = {
    "fsdp4": dict(dp_replicate=1, dp_shard=4, cp=1, ep=1),
    "ddp4": dict(dp_replicate=4, dp_shard=1, cp=1, ep=1),
}
STEPS, LR, TOL = 3, 0.02, 1e-5
_F = {
    "scale": "one",
    "warmup": 0,
    "c": 0.0,
    "native_norm": False,
    "radius": 4.0,
    "eta": {"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0},
}
CELLS = {
    "native": {"scale": "image_spectral", "warmup": 0, "c": 0.0},
    "native_unit": {"scale": "image_spectral", "magnitude_shape": "unit"},
    "row_row_tan": {**_F, "magnitude_lr": 0.5},
    "matrix_row_off": {**_F, "sphere": "matrix", "tangent": False, "magnitude_lr": 0.5},
    "matrix_row_off_col": {
        **_F,
        "sphere": "matrix",
        "tangent": False,
        "magnitude_lr": 0.5,
        "col_scale": True,
    },
    "row_none_off": {**_F, "magnitude_shape": "none", "tangent": False},
    "matrix_unit_tan": {
        **_F,
        "sphere": "matrix",
        "magnitude_shape": "unit",
        "magnitude_lr": 0.5,
    },
}


class TestAngularMuownPaths(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, topo, cell):
        device = self._device()
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        model_cfg = harness.tiny_model_config(moe=False)
        model_cfg.layer.attention.n_kv_heads = (
            2  # MHA, as gauge: one state shape per wqkv
        )
        cfg = harness.trainer_config(
            model_cfg,
            compute="svd",
            communication_dtype="float32",
            rule="angular_muown",
            rule_args={"pre": None, **CELLS[cell]},
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
            if isinstance(e.rule.compute, AngularMuownMap)
        ]
        self.assertTrue(entries, "no parameter runs angular_muown")
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
                name = names[id(e.param)]
                W2, G2 = ref[name].reshape(-1, e.view[1]), grads[
                    name
                ].float().cpu().reshape(-1, e.view[1])
                out = W2.clone()
                for k, u in enumerate(e.units):
                    rows = [(a, a + n) for a, n in u.rows.runs]
                    Wu = torch.cat([W2[a:b] for a, b in rows])
                    Gu = torch.cat([G2[a:b] for a, b in rows])
                    res = e.rule.compute.run(
                        Gu[None], Wu[None], state[name, k][None], StepCtx(LR, 0.0, step)
                    )
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


def _case(topo, cell):
    @with_comms
    def test(self):
        self._check(TOPOLOGIES[topo], cell)

    return test


for _topo in TOPOLOGIES:  # one test per case: each builds its meshes in fresh processes
    for _cell in CELLS:
        setattr(TestAngularMuownPaths, f"test_{_topo}_{_cell}", _case(_topo, _cell))
