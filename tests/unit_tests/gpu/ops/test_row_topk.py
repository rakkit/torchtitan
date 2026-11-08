# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""row_topk (the MoE routers' Triton top-k) returns K distinct indices per row,
as torch.topk does, also for rows with ties, NaN or fewer than K finite scores:
a duplicated expert id collapses in the router's routing map, so the per-expert
counts no longer match the routed rows (an EP all-to-all then fails on the split
sizes: 33 counted vs 40 routed rows in a TitanRL generator)."""

import pytest
import torch

from torchtitan.ops.row_topk import row_topk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, K = 256, 8


def _rows() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    nan_some = torch.rand(4, E)
    nan_some[:, [3, 50, 100]] = float("nan")
    few_finite = torch.full((4, E), float("-inf"))
    few_finite[:, [1, 2, 3]] = torch.tensor([3.0, 2.0, 1.0])
    return {
        "random": torch.rand(64, E),
        "ties": torch.full((4, E), 0.5),
        "nan_some": nan_some,
        "nan_all": torch.full((4, E), float("nan")),
        "few_finite": few_finite,
    }


@pytest.mark.parametrize("name", list(_rows()))
def test_row_topk_distinct_and_matches_torch(name):
    x = _rows()[name].cuda()
    values, indices = row_topk(x, K)
    want_values, want_indices = torch.topk(x, K, dim=-1)
    for row in indices.tolist():
        assert len(set(row)) == K and all(0 <= i < E for i in row)
    # the values are the row's own entries, in torch.topk's order (NaN first)
    torch.testing.assert_close(values, x.gather(1, indices), equal_nan=True)
    torch.testing.assert_close(values, want_values, equal_nan=True)
    if name == "random":  # distinct scores: the same experts in the same order
        assert torch.equal(indices, want_indices)
    if name == "few_finite":  # the finite ones first, then distinct -inf columns
        assert indices[:, :3].tolist() == [[1, 2, 3]] * 4
