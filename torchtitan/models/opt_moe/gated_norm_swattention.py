# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch_remat as remat
from torch import nn

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    FlexInnerAttention,
    GQAttention,
    InnerAttention,
    SlidingWindowFlexInnerAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.config_utils import get_attention_config, make_gqa_config
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.models.common.rope import RoPE
from .utils.attention_sinks import apply_attention_sink_rescale
from .utils.inits import conv_init_fn, make_param_init

from .utils.norms import build_norm_config
from .utils.short_conv import causal_short_conv


@dataclass(kw_only=True, slots=True)
class AttentionSpec:
    """The flat authoring surface of an OPT attention layer: what a flavor and a
    run config set (``m.layer.attention.conv_kernel_size = 4``). The model's
    config expansion turns it into a per-layer ``GatedNormSWAttention.Config``
    with ``build_attention_config``; it is never built itself."""

    n_heads: int
    n_kv_heads: int | None = None
    head_dim: int | None = None
    qk_norm: bool = False
    gated_attention_type: str | None = None  # None (no gate) or "element-wise"
    norm_eps: float = 1e-30
    norm_type: str = "rmsnorm"
    # flex, matching upstream: masks are always per document (block causal),
    # since batches are token-flat and `positions` resets at each document.
    attn_backend: str = "flex"
    sliding_window_size: int = -1
    """The window of the sliding-window layers (the model's `swa_pattern` picks
    them; -1: none)."""

    # === Canon short conv ===
    conv_kernel_size: int = 0
    """0: no conv (the conv fields below then do nothing). Else the Canon short
    conv after the projection: x + conv(x), a depthwise causal conv of this width."""
    conv_streams: str = "qkv"
    """"qkv" (a conv on q, k and v) or "kv" (k and v only; q has no q_conv)."""
    conv_init_fn_type: str = "delay"
    """"delay", "random_opnorm" or "torch_default" (utils.inits.conv_init_fn)."""
    conv_init_gain: float = 4.0
    """The per-channel operator norm of the init (delay / random_opnorm), or the
    factor on the nn.Conv1d default draw (torch_default). 4.0: the x4 in the init
    of the production recipe (never run; the study ran 1.0 with its optimizer's
    step-1 x4)."""

    # === Attention sinks ===
    attention_sinks: bool = False
    """A learned logit per head in the softmax denominator (gpt-oss)."""
    sinks_init_fn_type: str = "zeros"
    sinks_init_std: float = 0.0

    inner_attention: InnerAttention.Config = field(
        default_factory=FlexInnerAttention.Config
    )
    """The flex block size and kernel options every layer's inner attention
    takes (its type follows `attn_backend` and the layer's SWA)."""

    wq_init_fn_type: str = "scaled_orthogonal"
    wk_init_fn_type: str = "scaled_orthogonal"
    wv_init_fn_type: str = "scaled_orthogonal"
    wo_init_fn_type: str = "scaled_orthogonal"
    w_gate_init_fn_type: str = "scaled_orthogonal"

    wq_init_std: float = 1.0
    wk_init_std: float = 1.0
    wv_init_std: float = 1.0
    wo_init_std: float = 1.0
    w_gate_init_std: float = 1.0


class GatedNormSWAttention(GQAttention):
    """Gated Norm Sliding Window Attention, built on upstream's GQAttention.

    Token-flat: inputs are ``[T, D]`` and Q/K/V are ``[T, H, K]``. There is no
    batch dimension; document boundaries travel in ``positions`` and in the flex
    ``BlockMask``.

    Q/K/V projection, output projection, RoPE, QK-norm and the inner attention
    backend all come from ``GQAttention``. Only what upstream has no equivalent
    for is added here, each an optional sub-config: the element-wise output gate,
    the Canon short conv, attention sinks; and partial RoPE / NoPE, which follow
    from ``rope``. ``build_attention_config`` builds the config from an
    ``AttentionSpec``.
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
        sinks_init: Callable[[torch.Tensor], None] | None = None
        """The init of the attention sinks, one logit per head (None: no sinks)."""

        def __post_init__(self) -> None:
            GQAttention.Config.__post_init__(self)
            if (self.k_conv is None) != (self.v_conv is None):
                raise ValueError("k_conv and v_conv come together")
            if self.q_conv is not None and self.k_conv is None:
                raise ValueError("q_conv needs k_conv and v_conv")

    def __init__(self, config: Config):
        # GQAttention builds qkv_linear, wo, rope, inner_attention, q/k norm and
        # the scaling factor; the optional pieces upstream has no equivalent for
        # are built below, in the order the parameters are initialized.
        super().__init__(config)
        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
            )
        # The rotated width: rope.dim (its cos/sin cache is built at rope.dim).
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
        self.sinks = (
            nn.Parameter(torch.empty(self.n_heads))
            if config.sinks_init is not None
            else None
        )
        self._sinks_init = config.sinks_init

    def _init_self_parameters(self) -> None:
        if self.sinks is not None:
            self._sinks_init(self.sinks)

    def _apply_sinks(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        return apply_attention_sink_rescale(out, lse, self.sinks)

    def _apply_rope(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.rope is not None
        if not self.partial_rope:
            return self.rope(xq_THK, xk_THK, positions)
        # Partial RoPE: rotate only the leading qk_rope_dim dims, pass the rest
        # through untouched.
        xq_rot, xq_pass = (
            xq_THK[..., : self.qk_rope_dim],
            xq_THK[..., self.qk_rope_dim :],
        )
        xk_rot, xk_pass = (
            xk_THK[..., : self.qk_rope_dim],
            xk_THK[..., self.qk_rope_dim :],
        )
        xq_rot, xk_rot = self.rope(xq_rot, xk_rot, positions)
        return (
            torch.cat([xq_rot, xq_pass], dim=-1),
            torch.cat([xk_rot, xk_pass], dim=-1),
        )

    def param_layouts(self) -> dict:
        """The fused wqkv is three units (q / k / v rows interleaved per KV group)."""
        from torchtitan.models.common.param_layout import named, ParamLayout, qkv_units

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
    ) -> torch.Tensor:
        xq, xk, xv = self.qkv_linear(x_TD)  # upstream's fused QKVLinear
        gate = None
        if self.gate_proj is not None:
            gate = self.gate_proj(x_TD)  # gate: [T, n_local_heads * head_dim]
        # The q / k / v preparation reads the projection's outputs with bare ops.
        remat.recompute_needs_tensor(xq, xk, xv)
        xq, xk, xv = self._prepare_qkv(xq, xk, xv, positions)

        # No sdpa branch: `_inner_attention_config` takes upstream's
        # `get_attention_config`, which RAISES for "sdpa" ("no longer supported for
        # language models; positions are always available"). Language models
        # always use block_causal masking, which sdpa cannot express. Flex and
        # varlen are natively token-flat, so there is no batch dim to add or drop.
        scale_kwargs = {"scale": self.scaling} if self.scaling is not None else {}
        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            xq,
            xk,
            xv,
            attention_metadata=attention_metadata,
            enable_gqa=self.enable_gqa,
            out_transform=self._apply_sinks if self.sinks is not None else None,
            **scale_kwargs,
        )
        remat.recompute_needs_tensor(output)
        if gate is not None:
            # The gate reads the gate projection's output with bare ops.
            remat.recompute_needs_tensor(gate)
        # Not in the compiled region: wo declares its own remat regions.
        return self.wo(self._flatten_attention_output(output, gate))

    # Limit 16: the layer kinds (conv on q / k / v or k / v only, RoPE, partial RoPE or
    # NoPE, SWA or full attention) share this region and use more graphs than the
    # default 8. As muse_spark's attention_qk region: no pointwise autotuning.
    @local_compile(
        "opt_moe_attention_qkv",
        batch_invariant=False,
        recompile_limit=16,
        options={"triton.autotune_pointwise": False},
    )
    def _prepare_qkv(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THK: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q, k and v as the inner attention reads them. Each step runs only if
        configured: the Canon short conv (on q, k and v, or k and v only), the
        QK norm (before RoPE, per Qwen3), then RoPE (partial, or none for NoPE)."""
        if self.k_conv is not None:
            if self.q_conv is not None:
                xq_THK = causal_short_conv(
                    xq_THK.flatten(-2), self.q_conv.weight, positions
                ).view_as(xq_THK)
            xk_THK = causal_short_conv(
                xk_THK.flatten(-2), self.k_conv.weight, positions
            ).view_as(xk_THK)
            xv_THK = causal_short_conv(
                xv_THK.flatten(-2), self.v_conv.weight, positions
            ).view_as(xv_THK)
        if self.q_norm is not None:
            xq_THK = self.q_norm(xq_THK)
        if self.k_norm is not None:
            xk_THK = self.k_norm(xk_THK)
        if self.rope is not None:
            xq_THK, xk_THK = self._apply_rope(xq_THK, xk_THK, positions)
        return xq_THK, xk_THK, xv_THK

    @local_compile("opt_moe_attention_output", batch_invariant=True)
    def _flatten_attention_output(
        self, output_THK: torch.Tensor, gate_TF: torch.Tensor | None
    ) -> torch.Tensor:
        """Flatten the heads, applying the element-wise gate (in float32 for
        numeric stability, cast back) when configured."""
        output_TF = output_THK.contiguous().reshape(output_THK.shape[0], -1)
        if gate_TF is not None:
            output_TF = (output_TF.float() * torch.sigmoid(gate_TF.float())).to(
                output_TF.dtype
            )
        return output_TF


def _inner_attention_config(spec: AttentionSpec, sliding_window_size: int):
    """The layer's inner attention: upstream's config for ``attn_backend``, with
    the spec's flex block size and kernel options; the sliding-window flex one on
    a sliding-window layer."""
    inner = get_attention_config(spec.attn_backend)
    if isinstance(inner, FlexInnerAttention.Config) and isinstance(
        spec.inner_attention, FlexInnerAttention.Config
    ):
        inner = dataclasses.replace(
            inner,
            block_size=spec.inner_attention.block_size,
            kernel_options=spec.inner_attention.kernel_options,
        )
    if sliding_window_size > 0:
        if not isinstance(inner, FlexInnerAttention.Config):
            raise ValueError(
                "a sliding-window layer runs flex attention (got "
                f"attn_backend={spec.attn_backend!r})"
            )
        inner = SlidingWindowFlexInnerAttention.Config(
            block_size=inner.block_size,
            kernel_options=inner.kernel_options,
            sliding_window_size=sliding_window_size,
        )
    return inner


def build_attention_config(
    spec: AttentionSpec,
    *,
    dim: int,
    rope: RoPE.Config | None,
    sliding_window_size: int,
    residual_div: float,
    enable_sp: bool,
) -> GatedNormSWAttention.Config:
    """One layer's complete, sharded attention config from the flat spec:
    upstream's GQA part (``make_gqa_config``), then the gate, the convs and the
    sinks. ``rope`` None: a NoPE layer; ``sliding_window_size`` > 0: a
    sliding-window layer; ``residual_div``: the depth-init divisor of wo and the
    gate."""
    from torchtitan.models.opt_moe.sharding import (
        attention_sinks_state_sharding,
        cover_all_norm_params,
        dense_linear_sharding_config,
    )

    if {
        (spec.wk_init_fn_type, spec.wk_init_std),
        (spec.wv_init_fn_type, spec.wv_init_std),
    } != {(spec.wq_init_fn_type, spec.wq_init_std)}:
        raise ValueError(
            "wq, wk and wv initializers must match: the fused wqkv draws "
            "q, k and v with one initializer (fused_qkv_param_init)"
        )
    if spec.gated_attention_type not in (None, "element-wise"):
        raise ValueError(
            "gated_attention_type must be None (no gate) or 'element-wise', "
            f"got {spec.gated_attention_type!r}"
        )
    n_kv = spec.n_heads if spec.n_kv_heads is None else spec.n_kv_heads
    head_dim = spec.head_dim if spec.head_dim is not None else dim // spec.n_heads
    gqa = make_gqa_config(
        dim=dim,
        n_heads=spec.n_heads,
        n_kv_heads=n_kv,
        head_dim=spec.head_dim,
        wqkv_param_init={
            "weight": make_param_init(spec.wq_init_fn_type, spec.wq_init_std)
        },
        wo_param_init={
            "weight": make_param_init(
                spec.wo_init_fn_type, spec.wo_init_std, residual_div
            )
        },
        inner_attention=_inner_attention_config(spec, sliding_window_size),
        rope=rope,
        qk_norm=(
            build_norm_config(spec.norm_type, head_dim, spec.norm_eps)
            if spec.qk_norm
            else None
        ),
        enable_sp=enable_sp,
    )
    fields = {f.name: getattr(gqa, f.name) for f in dataclasses.fields(gqa)}
    if gqa.qk_norm is not None:  # every norm type's parameters, not only `weight`
        fields["qk_norm"] = dataclasses.replace(
            gqa.qk_norm,
            sharding_config=cover_all_norm_params(gqa.qk_norm.sharding_config),
        )
    if spec.gated_attention_type == "element-wise":
        # Dense gate over the full attention output channel dimension.
        fields["gate_proj"] = Linear.Config(
            in_features=dim,
            out_features=spec.n_heads * head_dim,
            param_init={
                "weight": make_param_init(
                    spec.w_gate_init_fn_type, spec.w_gate_init_std, residual_div
                )
            },
            sharding_config=dense_linear_sharding_config(),
        )
    if spec.conv_kernel_size > 0:
        if spec.conv_streams not in ("qkv", "kv"):
            raise ValueError(
                f"conv_streams must be qkv | kv, got {spec.conv_streams!r}"
            )

        def conv(channels: int) -> Conv1d.Config:
            return Conv1d.Config(
                in_channels=channels,
                out_channels=channels,
                kernel_size=spec.conv_kernel_size,
                groups=channels,
                bias=False,
                param_init={
                    "weight": conv_init_fn(spec.conv_init_fn_type, spec.conv_init_gain)
                },
                sharding_config=dense_linear_sharding_config(),
            )

        if spec.conv_streams == "qkv":
            fields["q_conv"] = conv(spec.n_heads * head_dim)
        fields["k_conv"] = conv(n_kv * head_dim)
        fields["v_conv"] = conv(n_kv * head_dim)
    if spec.attention_sinks:
        fields["sinks_init"] = make_param_init(
            spec.sinks_init_fn_type, spec.sinks_init_std
        )
        # The sinks are a direct parameter of the attention.
        fields["sharding_config"] = attention_sinks_state_sharding(
            gqa.sharding_config
        )
    return GatedNormSWAttention.Config(**fields)
