# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Under PP a stage holds only its own parameters: the embedding lives on the
first stage, the head on the last. A group that matches nothing on a stage is
dropped for it there; without PP it stays an error (it catches a typo)."""

import pytest
import torch.nn as nn

from torchtitan.components.optimizer.optimizer import ParamGroupConfig
from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.container import _KIND, DiSCOOptimizersContainer


class _MiddleStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])


def _configs():
    return [
        ParamGroupConfig(pattern=_KIND + "embedding", optimizer_name="DiSCO"),
        ParamGroupConfig(pattern=r"tok_embeddings\.weight", optimizer_name="DiSCO"),
        ParamGroupConfig(pattern=".*", optimizer_name="DiSCO"),
    ]


def _layouts():
    return {"layers.0.weight": ParamLayout("matrix")}


def test_a_stage_drops_the_groups_it_has_no_parameters_for():
    out = DiSCOOptimizersContainer._by_kind(
        _configs(), _MiddleStage(), _layouts(), pp=True
    )
    assert [c.pattern for c in out] == [".*"]


def test_without_pp_an_empty_kind_group_is_an_error():
    with pytest.raises(ValueError, match="kind 'embedding'"):
        DiSCOOptimizersContainer._by_kind(_configs(), _MiddleStage(), _layouts())
