# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preparation for adopting fused SwiGLU, which is not adopted yet.

AdamW is the target for now: it builds a flat named-parameter list, so a fused
model needs no optimizer change. DiSCO needs a declaration first: an undeclared
rank-3+ parameter is a build error, and `expert_batch` does not yet split an
expert into w1 / w3 units.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.overrides.fused_swiglu import fused_grouped_experts, FusedGroupedExperts


def test_the_override_warns_instead_of_silently_doing_nothing(caplog):
    """A no-op override is indistinguishable from one that worked."""

    class NotExactlyGroupedExperts(GroupedExperts.Config):
        pass

    cfg = NotExactlyGroupedExperts(num_experts=2, dim=8, hidden_dim=16)
    with caplog.at_level(logging.WARNING):
        result = fused_grouped_experts(cfg)

    assert result is cfg, "a subclass config must be returned unchanged"
    assert any(
        "no-op" in r.getMessage() for r in caplog.records
    ), "skipping the override silently is what made this hard to notice"


def test_the_exact_config_is_converted():
    cfg = GroupedExperts.Config(num_experts=2, dim=8, hidden_dim=16)
    fused = fused_grouped_experts(cfg)
    assert isinstance(fused, FusedGroupedExperts.Config)


def test_the_fused_expert_weight_is_rank_4():
    """`[E, F, 2, D]`, the shape a DiSCO declaration would have to describe.

    `[2, F, D]`, rank 3, is the DENSE stacked shape (`fuse_w13`), which is
    already declared as two units.
    """
    module = FusedGroupedExperts(
        FusedGroupedExperts.Config(num_experts=2, dim=8, hidden_dim=16)
    )
    assert module.w13.ndim == 4, module.w13.shape
    assert not hasattr(module, "w1_EFD")
    assert not hasattr(module, "w3_EFD")


def test_a_flat_adamw_container_groups_every_parameter_once():
    """AdamW's branch: one group, `param_names` present for state_dict."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    container = DiSCOOptimizersContainer(
        DiSCOOptimizersContainer.Config(
            groups={"default": {"optimizer": "AdamW", "lr": 1e-3}}
        ),
        model_parts=[model],
        parallel_dims=None,
    )
    groups = container.optimizers[0].param_groups
    assert len(groups) == 1
    assert len(groups[0]["params"]) == len(list(model.parameters()))
    assert "param_names" in groups[0]
