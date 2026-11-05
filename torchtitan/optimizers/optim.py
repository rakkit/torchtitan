# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`Optim` for DiSCO: upstream's update (clip, finite checks, step, lr
schedule, EMA) with DiSCO's container, which needs the parallelism context and
the run's output folder."""

from dataclasses import dataclass, field

import torch.nn as nn

from torchtitan.components.optim import Optim
from torchtitan.optimizers.container import DiSCOOptimizersContainer


class DiSCOOptim(Optim):
    @dataclass(kw_only=True, slots=True)
    class Config(Optim.Config):
        optimizer: DiSCOOptimizersContainer.Config = field(
            default_factory=DiSCOOptimizersContainer.Config
        )

        def __post_init__(self) -> None:
            Optim.Config.__post_init__(self)
            if self.max_norm == 0:
                # Upstream's clip scales by max_norm / (norm + 1e-6), clamped at
                # 1: max_norm 0 zeroes every gradient.
                raise ValueError(
                    "optim.max_norm = 0 zeroes every gradient; use "
                    "float('inf') for no clipping"
                )
            if self.enable_accelerator_graph:
                raise ValueError(
                    "DiSCO's step (host-side unit plan, all-to-alls) does not run "
                    "in an accelerator graph: set optim.enable_accelerator_graph "
                    "= False"
                )

    optimizers: DiSCOOptimizersContainer

    def _build_optimizers(
        self, *, model_parts: list[nn.Module], enable_accelerator_graph: bool
    ) -> DiSCOOptimizersContainer:
        assert not enable_accelerator_graph  # refused by Config
        return self.config.optimizer.build(
            model_parts=model_parts,
            parallelism_context=self.parallelism_context,
            output_dir=self.output_dir,
        )
