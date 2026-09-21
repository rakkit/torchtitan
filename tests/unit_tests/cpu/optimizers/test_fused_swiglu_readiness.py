# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preparation for adopting fused SwiGLU, which is not adopted yet.

AdamW is the target for now: it builds a flat named-parameter list and never
reaches `get_param_type`, so a fused model needs no DiSCO change. DiSCO does,
because `FusedGroupedExperts.w13` is rank 4 and `ndim == 3` would miss it.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.overrides.fused_swiglu import fused_grouped_experts, FusedGroupedExperts


def test_adamw_never_reaches_the_expert_classifier():
    """Why fused SwiGLU can land under AdamW without touching DiSCO."""
    import inspect

    source = inspect.getsource(DiSCOOptimizersContainer.__init__)
    assert "create_disco_param_groups" in source
    assert "issubclass(optimizer_cls, DiSCO)" in source, (
        "the mesh-derived grouping (and with it get_param_type) must stay "
        "gated on the optimizer actually being DiSCO; AdamW takes the flat "
        "named_parameters() branch"
    )


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
    """The shape DiSCO's `ndim == 3` classifier would miss.

    Pinned so the TODO in disco.py cannot go stale again -- it previously
    documented `[2, F, D]`, rank 3, which is the DENSE stacked shape and the
    opposite failure mode.
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
        DiSCOOptimizersContainer.Config(name="AdamW", lr=1e-3),
        model_parts=[model],
        parallel_dims=None,
    )
    groups = container.optimizers[0].param_groups
    assert len(groups) == 1
    assert len(groups[0]["params"]) == len(list(model.parameters()))
    assert "param_names" in groups[0]
