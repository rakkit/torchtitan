# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`lmo` on identical input must be bitwise identical on every rank.

The FSDP transport hands each full matrix to ONE owner rank; the result is
then scattered to every rank's shard. Ownership is therefore only free to
change (round-robin today, cost-balanced later) if the LMO does not depend on
which rank runs it. This pins that for the pure-torch backends. The triton
backends are `@triton.autotune`d and are not expected to pass.
"""

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.optimizers.disco.rules import lmo

pytestmark = pytest.mark.multi_gpu

SHAPES = [(64, 32), (32, 64), (128, 128), (5, 32, 16)]
BACKENDS = ["newtonschulz5", "polar_express", "svd"]


class TestLMORankInvariance(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_same_bits_on_every_rank(self):
        device = (
            torch.device("cpu")
            if self.device_type == "cpu"
            else torch.device(self.device_type, self.rank)
        )
        for backend in BACKENDS:
            for shape in SHAPES:
                g = torch.randn(shape, generator=torch.Generator().manual_seed(7)).to(
                    device
                )
                u = lmo(g, backend, "spectral", 5)
                gathered = [torch.empty_like(u) for _ in range(self.world_size)]
                dist.all_gather(gathered, u)
                for r, other in enumerate(gathered):
                    self.assertTrue(
                        torch.equal(u, other),
                        f"{backend} {shape}: rank {self.rank} vs rank {r} differ by "
                        f"{(u - other).abs().max().item():.3e}",
                    )
