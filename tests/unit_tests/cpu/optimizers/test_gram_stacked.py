# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The expert path's Gram metrics run under `vmap` over a stack of experts:
every value must be the per-expert one (a zero-update expert in the stack
included), and the pass must not fall back to per-element loops (an op
without a batching rule, e.g. an in-place `fill_diagonal_`, warns)."""

import warnings

import pytest
import torch

from torchtitan.optimizers.helpers.gram_helper import (
    calculate_gram_metrics,
    calculate_gram_metrics_stacked,
)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_stacked_gram_metrics_are_the_per_expert_ones(level):
    g = torch.Generator().manual_seed(0)
    E, m, n = 3, 8, 24
    W = torch.randn(E, m, n, generator=g) / n**0.5
    V = torch.randn(E, m, n, generator=g) * 1e-2
    W_after = W - 0.05 * V
    V[1] = 0  # a zero update in the middle of the stack
    W_after[1] = W[1]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*performance drop.*")
        stacked = calculate_gram_metrics_stacked(W, V, W_after, level=level)
    for e in range(E):
        alone = calculate_gram_metrics(W[e], V[e], W_after[e], level=level)
        assert stacked.keys() == alone.keys()
        for k, v in alone.items():
            torch.testing.assert_close(
                stacked[k][e], v, rtol=1e-4, atol=1e-6, equal_nan=True, msg=k
            )


def test_degenerate_stack_returns_nothing():
    W = torch.randn(4, 1, 16)  # one row per unit: no Gram metrics
    assert calculate_gram_metrics_stacked(W, W, W, level=2) == {}
