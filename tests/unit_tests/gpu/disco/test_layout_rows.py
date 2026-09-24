# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The row tables must say exactly which rows of the `[R, C]` view each rank
holds -- for this rank, checked against the data; for every peer, checked as
a partition. On a real opt_moe model under every topology the optimizer
supports."""

import os
import sys
import tempfile

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from torchtitan.optimizers.disco.layout import (  # noqa: E402
    local_rows,
    matrix_view,
    rows_of_coordinate,
)

pytestmark = pytest.mark.multi_gpu

TOPOLOGIES = {
    "fsdp4": dict(dp_replicate=1, dp_shard=4, cp=1, ep=1),
    "hsdp22": dict(dp_replicate=2, dp_shard=2, cp=1, ep=1),
    "cp2_fsdp2": dict(dp_replicate=1, dp_shard=2, cp=2, ep=1),
    "fsdp4_ep2": dict(dp_replicate=1, dp_shard=4, cp=1, ep=2),
}


def _rows_of(t: torch.Tensor, view):
    return t.reshape(-1, view[1])


class TestLayoutRows(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, topo):
        device = self._device()
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        cfg = harness.trainer_config(harness.tiny_model_config(moe=True), **topo)
        with tempfile.TemporaryDirectory() as tmp:
            model, _ = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        problems = []
        layouts = model.param_layouts()  # what the model declares, not ndim
        for name, p in harness.trainable(model):
            assert name in layouts, f"{name} is not declared"
            view = matrix_view(p.shape)
            full = _rows_of(harness.full_tensor(p), view)
            mine = local_rows(p, view)
            if isinstance(p, DTensor):
                local = _rows_of(p.to_local(), view)
                got = (
                    torch.cat([full[s : s + n] for s, n in mine.runs])
                    if mine.runs
                    else full[:0]
                )
                if got.shape != local.shape or not torch.equal(got, local):
                    problems.append(
                        f"{name}: local rows {mine.runs} do not match to_local()"
                    )
                # Every peer's rows from its coordinate alone: over the whole
                # mesh, every row must be held by the same number of ranks (the
                # replication factor) -- true for plain and strided shards alike.
                mesh = p.device_mesh
                import itertools

                count = torch.zeros(view[0], dtype=torch.int64)
                for coord in itertools.product(
                    *(range(mesh.size(a)) for a in range(mesh.ndim))
                ):
                    for s, n in rows_of_coordinate(p.shape, view, p, list(coord)).runs:
                        count[s : s + n] += 1
                if count.min() != count.max() or count.min() == 0:
                    problems.append(
                        f"{name}: rows held {count.min()}..{count.max()} times over the mesh"
                    )
            else:
                if mine.runs != ((0, view[0]),):
                    problems.append(
                        f"{name}: plain tensor should own all rows, got {mine.runs}"
                    )
        ok = harness.all_ranks_agree(not problems, device)
        if problems and self.rank == 0:
            print("\n".join(problems[:10]))
        self.assertTrue(ok, problems[:3])

    @with_comms
    def test_fsdp4(self):
        self._check(TOPOLOGIES["fsdp4"])

    @with_comms
    def test_hsdp22(self):
        self._check(TOPOLOGIES["hsdp22"])

    @with_comms
    def test_cp2_fsdp2(self):
        self._check(TOPOLOGIES["cp2_fsdp2"])

    @with_comms
    def test_fsdp4_ep2(self):
        self._check(TOPOLOGIES["fsdp4_ep2"])
