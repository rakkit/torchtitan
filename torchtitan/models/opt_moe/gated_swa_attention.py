# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn

from torchtitan.distributed.local_compile import local_compile
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    GQAttention,
    local_head_split,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attention.cp_attention import CPInnerAttention, Ulysses
from torchtitan.models.common.attention.cp_linear_attention import (
    ContextParallelLinearAttentionMetadata,
    prepare_cp_linear_attention_metadata,
)
from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata
from torchtitan.models.common.attn_gym import (
    causal_conv1d,
    context_parallel_conv_history,
    ContextParallelRouting,
    require_attn_gym,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.models.common.param_layout import named, ParamLayout, qkv_units


class ShortConvDocuments(
    CPInnerAttention[LinearAttentionMetadata, ContextParallelLinearAttentionMetadata]
):
    """The key of the Canon short conv's document offsets in the attention
    metadata map (a ``LinearAttentionMetadata``, built by
    ``OPTMoEModel._get_attention_metadata`` from the positions); never built.

    A ``CPInnerAttention`` so that upstream's ``Decoder._cp_shard`` prepares the
    entry with the batch's CP token permutation, as it does for Gated DeltaNet's
    conv: the result routes each local fragment's ``W - 1`` preceding tokens,
    held by whichever rank owns them."""

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: LinearAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> ContextParallelLinearAttentionMetadata:
        return prepare_cp_linear_attention_metadata(
            attention_metadata, permutation=permutation, family="OPT MoE short conv"
        )


def canon_short_conv(
    x_TC: torch.Tensor,
    weight_C1W: torch.Tensor,
    conv_metadata: LinearAttentionMetadata | None,
) -> torch.Tensor:
    """The Canon / Inkling short conv ``x + conv(x)`` (residual, no activation)
    on a token-flat stream: attn_gym's depthwise causal conv (``nn.Conv1d``
    orientation: ``w[..., -1]`` multiplies the current token; fp32
    accumulation), restarted at every document of ``conv_metadata`` (None: one
    document). Under CP (``ContextParallelLinearAttentionMetadata``) the rank's
    tokens are fragments of the global order; each fragment starts from the
    ``W - 1`` tokens before it, exchanged across the CP ranks (upstream's
    ``context_parallel_conv_history``, differentiable)."""
    initial_state = None
    if isinstance(conv_metadata, ContextParallelLinearAttentionMetadata):
        routing = conv_metadata.cp_routing
        cu_seqlens = routing.cu_seqlens
        initial_state = context_parallel_conv_history(
            x_TC.unsqueeze(0), routing, spmd_mesh_group(MeshAxisName.CP)
        )
    elif conv_metadata is not None and conv_metadata.varlen is not None:
        cu_seqlens = conv_metadata.varlen.cu_seq_q
    else:
        cu_seqlens = None
    conv_1TC = causal_conv1d(
        x_TC.unsqueeze(0),
        weight_C1W.squeeze(1).to(x_TC.dtype),
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )
    return conv_1TC.squeeze(0) + x_TC


class CanonShortConv(Module):
    """The short conv's computation ``x + conv(x)`` (``canon_short_conv``)
    as a component without parameters (the attention owns the conv weights),
    so that an inference engine can swap in a version that keeps each
    sequence's last ``W - 1`` inputs (TitanRL's generator)."""

    needs_document_offsets = True
    """Packed documents restart the conv at the offsets of ``conv_metadata``
    (False: the implementation tracks each sequence itself)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        x_TC: torch.Tensor,
        weight_C1W: torch.Tensor,
        conv_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        return canon_short_conv(x_TC, weight_C1W, conv_metadata)


def _output_and_lse(
    out: torch.Tensor, lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return out, lse


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
    for: the element-wise output gate, the Canon short conv on k and v, and
    attention sinks. Partial RoPE (``rope.dim < head_dim``) and NoPE
    (``rope=None``) follow from ``rope``.

    Token-flat: inputs are ``[T, D]``, q / k / v ``[T, H, K]``; document
    boundaries travel in ``positions`` and the attention metadata. Under TP
    each rank holds whole KV groups (their q heads, k, v, conv channels, gate
    rows and sinks); the input is gathered once for wqkv and the gate.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        gate_proj: Linear.Config | None = None
        """The element-wise output gate ``sigmoid(gate_proj(x))`` over the
        attention output (None: no gate)."""
        k_conv: Conv1d.Config | None = None
        v_conv: Conv1d.Config | None = None
        """The Canon short conv ``x + conv(x)`` on k and v (both or None: no
        conv). Only k and v: the KV cache's inputs, so generation keeps the
        last ``W - 1`` raw k / v per sequence, never q's much wider state."""
        short_conv: CanonShortConv.Config = field(default_factory=CanonShortConv.Config)
        """The conv's computation (with the convs above)."""
        attention_sinks: bool = False
        """A learned logit per head in the softmax denominator (gpt-oss), the
        parameter ``sinks``, initialized from ``param_init["sinks"]``; not with
        ``gate_proj``."""
        sinks_after_region: bool = True
        """In training, the sink rescale runs after the inner-attention remat
        region, so the backward recomputes it instead of saving its output (one
        ``[T, H, K]`` per layer). Set by the layer builder: False under TP or
        CP, whose inner-attention wrappers take one output tensor; there the
        rescale stays the region's ``out_transform``.
        TODO: (output, lse) through the CP and TP inner-attention wrappers."""

        def __post_init__(self) -> None:
            GQAttention.Config.__post_init__(self)
            if (self.k_conv is None) != (self.v_conv is None):
                raise ValueError("k_conv and v_conv come together")
            if self.gate_proj is not None and self.attention_sinks:
                raise ValueError("the output gate and attention sinks: one or neither")
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
        self.k_conv = config.k_conv.build() if config.k_conv is not None else None
        self.v_conv = config.v_conv.build() if config.v_conv is not None else None
        self.short_conv = (
            config.short_conv.build() if config.k_conv is not None else None
        )
        if self.k_conv is not None:
            require_attn_gym(
                causal_conv1d, context_parallel_conv_history, ContextParallelRouting
            )
        self.sinks = (
            nn.Parameter(torch.empty(self.n_heads)) if config.attention_sinks else None
        )
        # Ulysses CP runs the inner attention (and the in-region sink rescale) on
        # this CP rank's contiguous heads / cp heads: the rescale takes their slice.
        self.sinks_heads_sharded = isinstance(self.inner_attention, Ulysses)
        # The sliding-window layers' attention is its own remat region, so a policy
        # can recompute it (cheap at the window size) and keep the full layers'
        # saved: "*inner_attention_swa*" selects it, "*inner_attention*" both.
        self.attention_region = (
            "inner_attention_swa"
            if getattr(config.inner_attention, "sliding_window_size", None)
            else "inner_attention"
        )
        self.sinks_after_region = config.sinks_after_region

    def param_layouts(self) -> dict:
        """The fused wqkv is three units (q / k / v rows interleaved per KV group)."""
        w = self.qkv_linear.wqkv.weight
        units = qkv_units(self.n_heads, self.n_kv_heads, self.head_dim, w.shape[-1])
        layouts = {named(self, w): ParamLayout("matrix", units=units)}
        for conv in (self.k_conv, self.v_conv):
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
        if (
            self.short_conv is not None
            and self.short_conv.needs_document_offsets
            and conv_metadata is None
            and positions is not None
        ):
            # Packed documents without their offsets: the conv would run across them.
            raise ValueError(
                "the short conv needs its document offsets (the model's "
                "_get_attention_metadata) when positions are given"
            )
        # wqkv and the gate read one input: gathered once (TP; a no-op without).
        x_TD = maybe_gather_tp_input(self, x_TD)
        # The Canon short conv on k and v, the QK norm (before RoPE, per Qwen3),
        # then RoPE (partial, or none for NoPE).
        if self.k_conv is not None:
            xq, xk, xv = self._qkv_with_kv_conv(x_TD, conv_metadata)
        else:
            xq, xk, xv = self.qkv_linear(x_TD)
        gate = (
            local_head_split(self.gate_proj(x_TD), self.head_dim, cp_shard_dim=0)
            if self.gate_proj is not None
            else None
        )
        # The q / k / v preparation reads the projection's outputs with bare ops.
        remat.recompute_needs_tensor(xq, xk, xv)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)
        if self.rope is not None:
            xq, xk = self._apply_rope(xq, xk, positions)
        # In training the sink rescale runs after the region, as bare ops that the
        # backward recomputes (like the gate): inside the region its output was
        # saved beside the attention output, one more [T, H, K] per layer. The
        # no-grad paths (validation, the RL generator, which injects its own sink
        # hook into its attention backend) keep the rescale as out_transform.
        sinks_after_region = (
            self.sinks is not None
            and self.sinks_after_region
            and torch.is_grad_enabled()
        )
        output = remat.region(
            self.inner_attention,
            self.remat_region_name(self.attention_region),
            recompute=self.remat_should_recompute(self.attention_region),
        )(
            xq,
            xk,
            xv,
            attention_metadata=attention_metadata,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
            out_transform=(
                _output_and_lse
                if sinks_after_region
                else self._apply_sinks
                if self.sinks is not None
                else None
            ),
        )
        if sinks_after_region:
            output, lse = output
            remat.recompute_needs_tensor(output, lse)
            output = self._apply_sinks(output, lse)
        else:
            remat.recompute_needs_tensor(output)
        if gate is not None:
            # The gate reads the gate projection's output with bare ops.
            remat.recompute_needs_tensor(gate)
        # Not in the compiled region: wo declares its own remat regions.
        return self.wo(self._flatten_attention_output(output, gate))

    # QKVLinear.forward's output contract: [T, H, K] sharded on heads over TP.
    @spmd.local_map(
        out_types=(
            (
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
                spmd.PartitionSpec(("dp", "cp"), "tp", None),
            ),
        )
        * 3
    )
    def _qkv_with_kv_conv(
        self,
        x_TD: torch.Tensor,
        conv_metadata: LinearAttentionMetadata | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q, k and v with the Canon short conv on k and v in ONE call: wqkv,
        then the conv over each KV group's k and v rows (adjacent in wqkv's
        output: per group its q heads, its k, its v), copied out once. The conv
        is depthwise, so this is the two per-stream convs; its weight is the k /
        v conv weights interleaved per KV group, built at forward from the two
        parameters (the state dict, the HF format and the optimizer groups stay
        per stream). In place of ``QKVLinear.forward``, with its output types;
        under TP the rank's KV groups (local shapes); returns q / k / v
        contiguous, as it does."""
        qkv_TC = self.qkv_linear.wqkv(x_TD)
        # The split and the conv read the wqkv projection output with bare ops.
        remat.recompute_needs_tensor(qkv_TC)
        T = qkv_TC.shape[0]
        K, R = self.head_dim, self.qkv_linear.heads_per_kv
        W = self.k_conv.weight.shape[-1]
        # G: this rank's KV groups (all of them without TP)
        qkv_TGRK = qkv_TC.view(T, -1, R + 2, K)
        weight_C1W = torch.stack(
            [self.k_conv.weight.view(-1, K, W), self.v_conv.weight.view(-1, K, W)],
            dim=1,
        ).view(-1, 1, W)
        kv_TC = self.short_conv(
            qkv_TGRK[:, :, R:].reshape(T, -1), weight_C1W, conv_metadata
        )
        kv_TG2K = kv_TC.view(T, -1, 2, K)
        return (
            qkv_TGRK[:, :, :R].reshape(T, -1, K),
            kv_TG2K[:, :, 0].contiguous(),
            kv_TG2K[:, :, 1].contiguous(),
        )

    def _apply_sinks(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        sinks = self.sinks
        if self.sinks_heads_sharded:
            # each head's sink gradient comes from one CP rank; FSDP's reduce over
            # the CP ranks sums the slices
            heads = lse.shape[-1]
            start = torch.distributed.get_rank(spmd_mesh_group(MeshAxisName.CP)) * heads
            sinks = sinks[start : start + heads]
        return apply_attention_sink_rescale(out, lse, sinks)

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
        self, output_THK: torch.Tensor, gate_THK: torch.Tensor | None
    ) -> torch.Tensor:
        """Flatten the heads, applying the element-wise gate (in float32, cast
        back) when configured."""
        if gate_THK is not None:
            output_THK = (output_THK.float() * torch.sigmoid(gate_THK.float())).to(
                output_THK.dtype
            )
        return output_THK.contiguous().reshape(output_THK.shape[0], -1)
