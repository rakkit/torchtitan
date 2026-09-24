# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Identity
from torchtitan.protocols.module import Module
from .utils.activations import build_activation

from .utils.inits import make_param_init
from .utils.norms import build_norm_config


# TODO(fused-swiglu): make this eligible for `overrides/fused_swiglu.py`.
# Two things block it today (DiSCO does not: a fused w13 is declared as two
# units, `stacked_units`, with the same update as split w1/w3):
#  1. `fused_swiglu` is registered `@override(target=FeedForward.Config,
#     exact=True)` and reads `cfg.w1.param_init` as a CONFIG field, but this
#     class builds its `Linear.Config`s inside `__init__`. Lift w1/w2/w3 to
#     config fields (the `param_init` callables from `make_param_init` carry
#     over unchanged, and `_make_fused_linear_init` composes w1/w3 for the
#     fused w13 -- so the scion/orthogonal init survives fusion intact).
#  2. Stock `FusedSwiGLU.forward` is `w2(silu_and_mul(w13(x)))` with NO slot for
#     `mid_norm`. Most flavors set `norm_everywhere=True`, so fusing them
#     with the stock class would silently DELETE a norm. Those need a subclass
#     computing `w2(mid_norm(silu_and_mul(w13(x))))`; it still gets both wins
#     (w1+w3 in one GEMM, fused silu_and_mul kernel). The other 45 flavors can
#     use the stock class directly.
# Checkpoints are NOT safe any more: #4808 DELETED FusedSwiGLU's
# _split_w13_on_save / _merge_w13_on_load and moved the split to the HF adapter
# boundary, so `named_parameters()` now equals `state_dict()` (measured on
# llama3) and fusing CHANGES the native DCP layout. HF export still works, but
# it needs the `.clone()` in convert_to_hf._export_hf_weights: gate_proj/up_proj
# come out of `to_hf` as VIEWS of one storage and safetensors refuses those.
class FeedForward(Module):
    """SwiGLU feed-forward module for OPT MoE.

    Config takes the **final** hidden_dim (no internal 2/3 scaling).
    Use compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    Runtime ``dim`` is passed as a build() kwarg.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_dim: int
        norm_everywhere: bool = False
        norm_type: str = "np_rmsnorm"
        norm_eps: float = 1e-30
        activation_type: str = "silu"
        fuse_w13: bool = False
        """Gate and up in one `[2, F, D]` weight, so they run as one GEMM. DiSCO
        sees it as two units (`param_layouts`), same math as split w1/w3."""

        w1_init_fn_type: str = "scaled_orthogonal"
        w2_init_fn_type: str = "scaled_orthogonal"
        w3_init_fn_type: str = "scaled_orthogonal"

        w1_init_std: float = 1.0
        w2_init_std: float = 1.0
        w3_init_std: float = 1.0

        # Stamped per layer by OPTMoEModel.Config expansion; these used to be
        # build()/init_weights() arguments.
        linear_sharding_config: Any | None = None
        """ShardingConfig stamped on w1/w2/w3. Same reason as the
        attention config: spmd_types requires DTensor params before
        fully_shard, and this config builds its Linears rather than
        exposing them as fields."""
        norm_sharding_config: Any | None = None
        """ShardingConfig for `mid_norm`, which this config builds internally.
        Without it a PARAMETRIC `norm_type` leaves `feed_forward.mid_norm` and
        `moe.shared_experts.mid_norm` as plain tensors and `fully_shard`
        rejects them under spmd_types. Inert for the
        parameter-free `np_*` norms."""
        dim: int = 0
        residual_div: float = 1.0
        init_gate_as_residual: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dim = config.dim
        assert dim > 0, (
            "FeedForward.Config.dim must be stamped by the model's config "
            "expansion before build()."
        )
        # Configurable Linear rather than nn.Linear: every submodule has to
        # satisfy the Module protocol, and param_init then lives on the config.
        w3_residual_div = config.residual_div if config.init_gate_as_residual else 1.0
        self.fuse_w13 = config.fuse_w13
        w1_init = make_param_init(config.w1_init_fn_type, config.w1_init_std)
        w3_init = make_param_init(
            config.w3_init_fn_type, config.w3_init_std, w3_residual_div
        )
        if config.fuse_w13:
            # Slot 0 gate, slot 1 up; drawn full-shape, then written through
            # the parameter's own placement.
            def _init_w13(w):
                full = torch.empty(tuple(w.shape), dtype=w.dtype, device=w.device)
                w1_init(full[0])
                w3_init(full[1])
                if isinstance(w, DTensor):
                    w.copy_(distribute_tensor(full, w.device_mesh, w.placements))
                else:
                    w.copy_(full)

            self.w13 = Linear.Config(
                in_features=dim,
                out_features=config.hidden_dim,
                num_linears=2,
                param_init={"weight": _init_w13},
                sharding_config=config.linear_sharding_config,
            ).build()
        else:
            self.w1 = Linear.Config(
                in_features=dim,
                out_features=config.hidden_dim,
                param_init={"weight": w1_init},
                sharding_config=config.linear_sharding_config,
            ).build()
        # w2 before w3, matching the original assignment order: module
        # registration order drives init_weights()'s RNG draws, so reordering
        # these silently changes every w2/w3 weight.
        self.w2 = Linear.Config(
            in_features=config.hidden_dim,
            out_features=dim,
            param_init={
                "weight": make_param_init(
                    config.w2_init_fn_type, config.w2_init_std, config.residual_div
                )
            },
            sharding_config=config.linear_sharding_config,
        ).build()
        if not config.fuse_w13:
            self.w3 = Linear.Config(
                in_features=dim,
                out_features=config.hidden_dim,
                param_init={"weight": w3_init},
                sharding_config=config.linear_sharding_config,
            ).build()
        self.act_fn = build_activation(config.activation_type)

        if config.norm_everywhere:
            self.mid_norm = build_norm_config(
                config.norm_type,
                config.hidden_dim,
                config.norm_eps,
                sharding_config=config.norm_sharding_config,
            ).build()
        else:
            self.mid_norm = Identity.Config().build()

    def param_layouts(self) -> dict:
        """The stacked w13 is two units: rows of slot 0 (w1) and slot 1 (w3)."""
        if not self.fuse_w13:
            return {}
        from torchtitan.models.common.param_layout import (
            named,
            ParamLayout,
            stacked_units,
        )

        w = self.w13.weight
        units = stacked_units(("w1", "w3"), w.shape[1], w.shape[2])
        return {named(self, w): ParamLayout("matrix", units=units)}

    def forward(self, x):
        if self.fuse_w13:
            gate, up = self.w13(x).unbind(-2)
        else:
            gate, up = self.w1(x), self.w3(x)
        return self.w2(self.mid_norm(self.act_fn(gate) * up))
