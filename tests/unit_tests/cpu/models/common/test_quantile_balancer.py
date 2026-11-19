# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.common.moe import estimate_expert_biases


def _per_layer_estimate(histogram_EB, expert_bias_E, top_k, num_bins):
    """The quantile estimate of one layer as it was computed per layer."""
    num_experts = histogram_EB.shape[0]
    counts_E = histogram_EB.sum(dim=-1, dtype=torch.int64)
    target_count_E = counts_E.float() * (top_k / num_experts)
    cumulative_counts_EB = histogram_EB.cumsum(dim=-1, dtype=torch.int64)
    target_rank_E = target_count_E.ceil().to(torch.int64)
    target_bin_E = (cumulative_counts_EB < target_rank_E.unsqueeze(-1)).sum(dim=-1)
    target_bin_E1 = target_bin_E.unsqueeze(-1)
    counts_in_bin_E = histogram_EB.gather(-1, target_bin_E1).squeeze(-1)
    counts_before_E = (
        cumulative_counts_EB.gather(-1, target_bin_E1).squeeze(-1) - counts_in_bin_E
    )
    fraction_E = (target_count_E - counts_before_E.float()) / counts_in_bin_E.float()
    bin_width = (expert_bias_E.max() - expert_bias_E.min() + 2.0) / num_bins
    quantile_position_E = target_bin_E.float() + fraction_E
    return (quantile_position_E - quantile_position_E.mean()) * bin_width


@pytest.mark.parametrize("max_count", [3, 5000, 100000])
def test_batched_estimate_is_the_per_layer_estimate_bitwise(max_count):
    # the balancing hook updates all layers in one batch; the bias must not change
    num_layers, num_experts, num_bins, top_k = 6, 64, 200, 8
    generator = torch.Generator().manual_seed(max_count)
    histogram_LEB = torch.randint(
        1,
        max_count,
        (num_layers, num_experts, num_bins),
        dtype=torch.int32,
        generator=generator,
    )
    expert_bias_LE = (
        torch.rand(num_layers, num_experts, generator=generator) - 0.5
    ) * 0.6
    expected_LE = torch.stack(
        [
            _per_layer_estimate(histogram_LEB[i], expert_bias_LE[i], top_k, num_bins)
            for i in range(num_layers)
        ]
    )
    actual_LE = estimate_expert_biases(
        histogram_LEB, expert_bias_LE, top_k=top_k, num_bins=num_bins
    )
    assert torch.equal(actual_LE, expected_LE)
