# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

from torchtitan.models.common import feed_forward
from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU
from torchtitan.models.common.config_utils import fused_gate_up_param_init
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.param_layout import named, ParamLayout, stacked_units
from torchtitan.protocols.module import Module

from .utils.inits import make_param_init


class FeedForward(feed_forward.FeedForward):
    """Upstream's SwiGLU FeedForward (gate and up in one `[2, F, D]` `w13`),
    built from OPT MoE's flat settings: its initialisers.

    Config takes the **final** hidden_dim; ``dim`` is stamped by the model's
    config expansion.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_dim: int
        activation_fn: BinaryActivationFn.Config = field(default_factory=SwiGLU.Config)

        w1_init_fn_type: str = "scaled_orthogonal"
        w2_init_fn_type: str = "scaled_orthogonal"
        w3_init_fn_type: str = "scaled_orthogonal"

        w1_init_std: float = 1.0
        w2_init_std: float = 1.0
        w3_init_std: float = 1.0

        # Stamped per layer by OPTMoEModel.Config expansion.
        linear_sharding_config: Any | None = None
        """ShardingConfig for `w13` and `w2`: spmd_types needs DTensor params
        before fully_shard, and this config builds its Linears itself."""
        dim: int = 0
        residual_div: float = 1.0
        init_gate_as_residual: bool = False

    def __init__(self, config: Config):
        assert config.dim > 0, (
            "FeedForward.Config.dim must be stamped by the model's config "
            "expansion before build()."
        )
        w3_div = config.residual_div if config.init_gate_as_residual else 1.0
        w13_init = fused_gate_up_param_init(
            {"weight": make_param_init(config.w1_init_fn_type, config.w1_init_std)},
            {
                "weight": make_param_init(
                    config.w3_init_fn_type, config.w3_init_std, w3_div
                )
            },
        )
        w2_init = make_param_init(
            config.w2_init_fn_type, config.w2_init_std, config.residual_div
        )
        super().__init__(
            feed_forward.FeedForward.Config(
                w13=Linear.Config(
                    in_features=config.dim,
                    out_features=config.hidden_dim,
                    num_linears=2,
                    param_init=w13_init,
                    sharding_config=config.linear_sharding_config,
                ),
                w2=Linear.Config(
                    in_features=config.hidden_dim,
                    out_features=config.dim,
                    param_init={"weight": w2_init},
                    sharding_config=config.linear_sharding_config,
                ),
                activation_fn=config.activation_fn,
            )
        )

    def param_layouts(self) -> dict:
        """`w13` is two units: rows of slot 0 (w1) and slot 1 (w3)."""
        w = self.w13.weight
        units = stacked_units(("w1", "w3"), w.shape[1], w.shape[2])
        return {named(self, w): ParamLayout("matrix", units=units)}
