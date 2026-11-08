# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform that turns on RL routing replay."""

from typing import Literal

from torchtitan.models.base.module import Module
from torchtitan.models.common.moe import TokenChoiceTopKRouter

from .base import ModelConfigTransform

__all__ = ["RoutingReplayTransform"]


class RoutingReplayTransform(ModelConfigTransform):
    """Set every router's ``routing_replay`` (R3, https://arxiv.org/abs/2510.11370).

    Applied to both RL model configs: the generator records each token's experts,
    the trainer counts ("measure") or routes ("replay") them. Dense models are
    unchanged.
    """

    def __init__(self, mode: Literal["measure", "replay"]) -> None:
        self.mode = mode

    def transform(self, model: Module.Config) -> Module.Config:
        for _fqn, router, _parent, _attr in model.traverse(
            TokenChoiceTopKRouter.Config
        ):
            router.routing_replay = self.mode
        return model
