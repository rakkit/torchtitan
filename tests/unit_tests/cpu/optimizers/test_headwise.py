# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Head-wise Muon (`muon` with `block_rows` / `block_cols`): each block is
its own polar factor `U V^T`, times sqrt(rows / cols) of the block."""

import pytest
import torch

from torchtitan.optimizers.disco.presets import Blocks, Orthogonalise
from torchtitan.optimizers.disco.rules import rule_of


def polar(a: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(a.double(), full_matrices=False)
    return u @ vh


def headwise(g: torch.Tensor, rows: int = 0, cols: int = 0) -> torch.Tensor:
    m, n = g.shape
    if rows:
        return torch.cat(
            [polar(b) * (rows / n) ** 0.5 for b in g.split(rows, dim=0)], dim=0
        )
    return torch.cat([polar(b) * (m / cols) ** 0.5 for b in g.split(cols, dim=1)], 1)


def _rule(**args):
    return rule_of({"rule": "muon", "args": {"compute": "svd", **args}})


@pytest.mark.parametrize(
    "shape, args",
    [((12, 8), {"block_rows": 4}), ((8, 12), {"block_cols": 4})],
)
def test_each_block_is_its_own_scaled_polar_factor(shape, args):
    torch.manual_seed(0)
    g = torch.randn(shape)
    want = headwise(g, args.get("block_rows", 0), args.get("block_cols", 0))
    got = _rule(**args).run_whole(g).update
    torch.testing.assert_close(got.double(), want, rtol=1e-5, atol=1e-5)
    whole = polar(g) * (shape[0] / shape[1]) ** 0.5
    assert not torch.allclose(got.double(), whole, atol=1e-3)  # not plain Muon


@pytest.mark.parametrize("args", [{"block_rows": 4}, {"block_cols": 4}])
def test_a_stack_matches_each_matrix(args):
    torch.manual_seed(1)
    x = torch.randn(3, 8, 12)
    rule = _rule(**args)
    got = rule.run_whole(x).update
    for b in range(3):
        torch.testing.assert_close(got[b], rule.run_whole(x[b]).update)


def test_blocks_are_checked():
    with pytest.raises(ValueError, match="exactly one"):
        Blocks(Orthogonalise("svd"), rows=4, cols=4)
    with pytest.raises(ValueError, match="do not divide"):
        Blocks(Orthogonalise("svd"), rows=5).check_unit(12, 8)
    Blocks(Orthogonalise("svd"), cols=4).check_unit(5, 8)
