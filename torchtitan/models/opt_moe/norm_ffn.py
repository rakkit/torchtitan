# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

import torch
import torch_remat as remat
from torch import nn

from torchtitan.models.common import feed_forward
from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU
from torchtitan.models.common.config_utils import fused_gate_up_param_init
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.param_layout import named, ParamLayout, stacked_units
from torchtitan.protocols.module import Module

from .utils.activations import ffn_activation
from .utils.inits import make_param_init


class FeedForward(feed_forward.FeedForward):
    """Upstream's SwiGLU FeedForward (gate and up in one `[2, F, D]` `w13`),
    built from OPT MoE's flat settings: its initialisers and the mid-norm.

    Config takes the **final** hidden_dim; ``dim`` is stamped by the model's
    config expansion.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_dim: int
        norm_everywhere: bool = False
        """Adds the mid-norm after the activation (`MidNormActivation`)."""
        norm_type: str = "np_rmsnorm"
        norm_eps: float = 1e-30
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
        row_multipliers: tuple[str, ...] = ()
        """Learned per-row output multipliers (init 1), y = m * (W x), i.e.
        W_eff = Diag(m) W (arXiv 2601.04890), on the named projections: "w1"
        (gate), "w3" (up) and/or "w2" (the block's residual write)."""
        scalar_multipliers: bool = False
        """Make each of `row_multipliers` a single learned scalar per projection
        (shape [1], W_eff = m W) instead of one per row."""
        swish_beta: bool = False
        """One learned scalar beta per layer (shape [1], init 1) on the gate
        input: act(beta * gate, up), i.e. silu(beta z) up for SwiGLU. beta
        folds into w1, so the function class is unchanged; it adds one scale
        channel for the optimizer (Ramachandran et al., Swish beta)."""

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
                activation_fn=ffn_activation(
                    config.activation_fn,
                    mid_norm=config.norm_everywhere,
                    norm_type=config.norm_type,
                    norm_eps=config.norm_eps,
                ),
            )
        )
        unknown = set(config.row_multipliers) - {"w1", "w3", "w2"}
        if unknown:
            raise ValueError(
                f"feed_forward row_multipliers takes 'w1', 'w3' and/or 'w2', got "
                f"{sorted(unknown)}"
            )

        if config.scalar_multipliers and not config.row_multipliers:
            raise ValueError("feed_forward scalar_multipliers needs row_multipliers")

        def multiplier(name: str, size: int) -> nn.Parameter | None:
            if name not in config.row_multipliers:
                return None
            return nn.Parameter(torch.empty(1 if config.scalar_multipliers else size))

        self.swish_beta = nn.Parameter(torch.empty(1)) if config.swish_beta else None
        self.w1_multiplier = multiplier("w1", config.hidden_dim)
        self.w3_multiplier = multiplier("w3", config.hidden_dim)
        self.w2_multiplier = multiplier("w2", config.dim)

    def _init_self_parameters(self) -> None:
        for m in (
            self.w1_multiplier,
            self.w3_multiplier,
            self.w2_multiplier,
            self.swish_beta,
        ):
            if m is not None:
                nn.init.ones_(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mults = (self.w1_multiplier, self.w3_multiplier, self.w2_multiplier)
        if all(m is None for m in mults) and self.swish_beta is None:
            return super().forward(x)
        # Upstream's forward, with each projection's rows scaled by its multiplier.
        gate_up_T2F = remat.region(
            self.w13,
            self.remat_region_name("w13"),
            recompute=self.remat_should_recompute("w13"),
        )(x)
        gate_TF, up_TF = gate_up_T2F.unbind(-2)
        # On the saved regions' outputs, before their first consumer (the multiply).
        remat.recompute_needs_tensor(gate_TF, up_TF)
        if self.swish_beta is not None:
            gate_TF = gate_TF * self.swish_beta
        if self.w1_multiplier is not None:
            gate_TF = gate_TF * self.w1_multiplier
        if self.w3_multiplier is not None:
            up_TF = up_TF * self.w3_multiplier
        out_TD = remat.region(
            self.w2,
            self.remat_region_name("w2"),
            recompute=self.remat_should_recompute("w2"),
        )(self.activation_fn(gate_TF, up_TF))
        remat.recompute_needs_tensor(out_TD)
        if self.w2_multiplier is not None:
            out_TD = out_TD * self.w2_multiplier
        return out_TD

    def param_layouts(self) -> dict:
        """`w13` is two units: rows of slot 0 (w1) and slot 1 (w3)."""
        w = self.w13.weight
        units = stacked_units(("w1", "w3"), w.shape[1], w.shape[2])
        return {named(self, w): ParamLayout("matrix", units=units)}
