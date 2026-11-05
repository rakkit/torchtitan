# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch_remat as remat
from torch import nn

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    GQAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata
from torchtitan.models.common.attn_gym import causal_conv1d, require_attn_gym
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.models.common.param_layout import named, ParamLayout, qkv_units


class ShortConvDocuments:
    """The key of the Canon short conv's document offsets in the attention
    metadata map (an ``LinearAttentionMetadata``, built by
    ``OPTMoEModel._get_attention_metadata`` from the positions); never built."""


def canon_short_conv(
    x_TC: torch.Tensor,
    weight_C1W: torch.Tensor,
    conv_metadata: LinearAttentionMetadata | None,
) -> torch.Tensor:
    """The Canon / Inkling short conv ``x + conv(x)`` (residual, no activation)
    on a token-flat stream: attn_gym's depthwise causal conv (``nn.Conv1d``
    orientation: ``w[..., -1]`` multiplies the current token; fp32
    accumulation), restarted at every document of ``conv_metadata`` (None: one
    document)."""
    varlen = None if conv_metadata is None else conv_metadata.varlen
    conv_1TC = causal_conv1d(
        x_TC.unsqueeze(0),
        weight_C1W.squeeze(1).to(x_TC.dtype),
        cu_seqlens=None if varlen is None else varlen.cu_seq_q,
    )
    return conv_1TC.squeeze(0) + x_TC


# Attention sinks: a copy of gpt_oss's apply_attention_sink_rescale (same math),
# compiled as its own region; opt_moe does not import gpt_oss code.
@local_compile("opt_moe_sink_rescale", batch_invariant=True)
def apply_attention_sink_rescale(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """Rescale attention output by the learned per-head sink term:
    ``out * sigmoid(lse - s)`` with ``s`` the head's sink logit, which equals the softmax
    over the keys plus one extra logit ``s`` that carries no value."""
    sinks = sinks.view(*([1] * (lse.ndim - 1)), -1)
    sink_scale = torch.sigmoid(lse - sinks).unsqueeze(-1)
    return out * sink_scale.to(out.dtype)


class GatedSWAttention(GQAttention):
    """Gated sliding-window attention: upstream's GQAttention (fused Q/K/V, wo,
    RoPE, QK norm, inner attention; full or sliding-window by the inner
    attention's type), plus three optional pieces upstream has no equivalent
    for: the element-wise output gate, the Canon short conv on q / k / v, and
    attention sinks. Partial RoPE (``rope.dim < head_dim``) and NoPE
    (``rope=None``) follow from ``rope``.

    Token-flat: inputs are ``[T, D]``, q / k / v ``[T, H, K]``; document
    boundaries travel in ``positions`` and the attention metadata.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        gate_proj: Linear.Config | None = None
        """The element-wise output gate ``sigmoid(gate_proj(x))`` over the
        attention output (None: no gate)."""
        q_conv: Conv1d.Config | None = None
        k_conv: Conv1d.Config | None = None
        v_conv: Conv1d.Config | None = None
        """The Canon short conv ``x + conv(x)`` on q, k and v (k and v together;
        q optional; None: no conv)."""
        attention_sinks: bool = False
        """A learned logit per head in the softmax denominator (gpt-oss), the
        parameter ``sinks``, initialized from ``param_init["sinks"]``."""

        def __post_init__(self) -> None:
            GQAttention.Config.__post_init__(self)
            if (self.k_conv is None) != (self.v_conv is None):
                raise ValueError("k_conv and v_conv come together")
            if self.q_conv is not None and self.k_conv is None:
                raise ValueError("q_conv needs k_conv and v_conv")
            if self.attention_sinks and "sinks" not in (self.param_init or {}):
                raise ValueError('attention sinks need param_init["sinks"]')

    def __init__(self, config: Config):
        super().__init__(config)
        # The rotated width (the RoPE cache is built at rope.dim).
        self.qk_rope_dim: int = (
            self.head_dim if config.rope is None else config.rope.dim
        )
        if self.qk_rope_dim > self.head_dim:
            raise ValueError(
                f"rope.dim ({self.qk_rope_dim}) must be <= head_dim ({self.head_dim})"
            )
        self.partial_rope = self.qk_rope_dim != self.head_dim
        self.gate_proj = (
            config.gate_proj.build() if config.gate_proj is not None else None
        )
        self.q_conv = config.q_conv.build() if config.q_conv is not None else None
        self.k_conv = config.k_conv.build() if config.k_conv is not None else None
        self.v_conv = config.v_conv.build() if config.v_conv is not None else None
        if self.k_conv is not None:
            require_attn_gym(causal_conv1d)
        self.sinks = (
            nn.Parameter(torch.empty(self.n_heads)) if config.attention_sinks else None
        )

    def param_layouts(self) -> dict:
        """The fused wqkv is three units (q / k / v rows interleaved per KV group)."""
        w = self.qkv_linear.wqkv.weight
        units = qkv_units(self.n_heads, self.n_kv_heads, self.head_dim, w.shape[-1])
        layouts = {named(self, w): ParamLayout("matrix", units=units)}
        for conv in (self.q_conv, self.k_conv, self.v_conv):
            if conv is not None:
                layouts[named(self, conv.weight)] = ParamLayout("conv")
        return layouts

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        conv_metadata: LinearAttentionMetadata | None = None,
    ) -> torch.Tensor:
        """``conv_metadata``: the short conv's document offsets (the model's
        ``ShortConvDocuments`` entry)."""
        xq, xk, xv = self.qkv_linear(x_TD)
        gate = self.gate_proj(x_TD) if self.gate_proj is not None else None
        # The q / k / v preparation reads the projection's outputs with bare ops.
        remat.recompute_needs_tensor(xq, xk, xv)
        # The Canon short conv (on q, k and v, or k and v only), the QK norm
        # (before RoPE, per Qwen3), then RoPE (partial, or none for NoPE).
        if self.k_conv is not None:
            if conv_metadata is None and positions is not None:
                # Packed documents without their offsets: the conv would run across them.
                raise ValueError(
                    "the short conv needs its document offsets (the model's "
                    "_get_attention_metadata) when positions are given"
                )
            if self.q_conv is not None:
                xq = canon_short_conv(
                    xq.flatten(-2), self.q_conv.weight, conv_metadata
                ).view_as(xq)
            xk = canon_short_conv(
                xk.flatten(-2), self.k_conv.weight, conv_metadata
            ).view_as(xk)
            xv = canon_short_conv(
                xv.flatten(-2), self.v_conv.weight, conv_metadata
            ).view_as(xv)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)
        if self.rope is not None:
            xq, xk = self._apply_rope(xq, xk, positions)
        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            xq,
            xk,
            xv,
            attention_metadata=attention_metadata,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
            out_transform=self._apply_sinks if self.sinks is not None else None,
        )
        remat.recompute_needs_tensor(output)
        if gate is not None:
            # The gate reads the gate projection's output with bare ops.
            remat.recompute_needs_tensor(gate)
        # Not in the compiled region: wo declares its own remat regions.
        return self.wo(self._flatten_attention_output(output, gate))

    def _apply_sinks(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        return apply_attention_sink_rescale(out, lse, self.sinks)

    def _apply_rope(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.partial_rope:
            return self.rope(xq_THK, xk_THK, positions)
        # Partial RoPE: rotate the leading qk_rope_dim dims, pass the rest through.
        d = self.qk_rope_dim
        xq_rot, xk_rot = self.rope(xq_THK[..., :d], xk_THK[..., :d], positions)
        return (
            torch.cat([xq_rot, xq_THK[..., d:]], dim=-1),
            torch.cat([xk_rot, xk_THK[..., d:]], dim=-1),
        )

    @local_compile("opt_moe_attention_output", batch_invariant=True)
    def _flatten_attention_output(
        self, output_THK: torch.Tensor, gate_TF: torch.Tensor | None
    ) -> torch.Tensor:
        """Flatten the heads, applying the element-wise gate (in float32, cast
        back) when configured."""
        output_TF = output_THK.contiguous().reshape(output_THK.shape[0], -1)
        if gate_TF is not None:
            output_TF = (output_TF.float() * torch.sigmoid(gate_TF.float())).to(
                output_TF.dtype
            )
        return output_TF
