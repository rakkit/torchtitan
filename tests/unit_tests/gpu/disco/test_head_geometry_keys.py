# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The LM-head geometry metrics reach the storing rank, once, with real values.

`head_geometry` itself is checked against brute-force references by
`tests/unit_tests/disco_metrics/verify_head_geometry.py`. What this adds is the
distributed half: the owner of the head unit computes them, `_flush_head`
carries them to whoever stores norms, the key set is the same in gather and
local mode, and a step without a probe logs none of them.
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from torchtitan.optimizers.helpers.head_geometry import HEAD_METRIC_NAMES  # noqa: E402

pytestmark = pytest.mark.multi_gpu

TOPOLOGIES = {
    "fsdp4": dict(dp_shard=4),
    "ddp4": dict(dp_replicate=4),
}


class TestHeadGeometryKeys(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, topo: dict, *, local: bool):
        device = self._device()
        pd = harness.parallelism_context(
            world_size=self.world_size, device_type=device.type, **topo
        )
        cfg = harness.trainer_config(
            harness.tiny_model_config(moe=True),
            compute="svd",
            log_norm_freq=1,
            communication_dtype="float32",
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        container.log_metrics_locally = local
        opt = container.optimizers[0]
        problems = []
        if not opt._head_keys:
            problems.append("no LM-head unit was registered for head geometry")
        head_keys = list(opt._head_keys)

        dim = cfg.model.dim
        h = torch.randn(
            32, dim, generator=torch.Generator().manual_seed(7), dtype=torch.float32
        ).to(device)

        # A logging step with no probe: the diagnostic is off, nothing logged.
        harness.set_synthetic_grads(model, 1, device)
        container.begin_step(1, should_log=True, is_final_step=False)
        container.step()
        container.zero_grad()
        no_probe = {k for k in container.collect_metrics(1) if "head_" in k}
        if no_probe:
            problems.append(f"probe-free step logged {sorted(no_probe)[:3]}")

        # A logging step with a probe.
        harness.set_synthetic_grads(model, 2, device)
        rows = container.begin_step(
            2, should_log=True, is_final_step=False, head_input=lambda: h
        )
        if rows != container.head_diag_probes:
            problems.append(f"begin_step asked for {rows} head-input rows")
        container.step()
        container.zero_grad()
        metrics = container.collect_metrics(2)
        mine = {k for k in metrics if k.startswith("head_")}

        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, mine)
        union = set().union(*gathered)
        expected = {f"head_{m}/{k}" for k in head_keys for m in HEAD_METRIC_NAMES}
        if union != expected:
            problems.append(
                f"union != expected: missing {sorted(expected - union)[:3]}, "
                f"extra {sorted(union - expected)[:3]}"
            )
        if sum(len(s) for s in gathered) != len(expected):
            problems.append("a head key is stored by more than one rank")
        if not local and [r for r, s in enumerate(gathered) if s] != [0]:
            problems.append("gather mode: head keys are not on rank 0 alone")

        lr = cfg.optim.optimizer.groups["default"]["lr"]
        for k in head_keys:
            for name in HEAD_METRIC_NAMES:
                key = f"head_{name}/{k}"
                if key not in metrics:
                    continue
                v = float(metrics[key])
                if v != v:  # NaN
                    problems.append(f"{key} is NaN")
                if name == "eta" and abs(v - lr) > 1e-6:
                    problems.append(f"{key} = {v} != head lr {lr}")
                # gamma_H folds the probes in, so this bound is an identity.
                if name == "a_H_max" and v > 1.0 + 1e-5:
                    problems.append(f"{key} = {v} > 1")
                if name in ("gamma_H", "rho_max", "U_rms_to_inf") and not v > 0.0:
                    problems.append(f"{key} = {v}, expected > 0")

        ok = harness.all_ranks_agree(not problems, device)
        if problems and self.rank == 0:
            print("\n".join(problems[:8]))
        self.assertTrue(ok, problems[:3])

    @with_comms
    def test_fsdp4_gather(self):
        self._check(TOPOLOGIES["fsdp4"], local=False)

    @with_comms
    def test_fsdp4_local(self):
        self._check(TOPOLOGIES["fsdp4"], local=True)

    @with_comms
    def test_ddp4_gather(self):
        self._check(TOPOLOGIES["ddp4"], local=False)
