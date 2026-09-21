# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from typing import Any

import torch

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexInnerAttention,
    InnerAttention,
    GQAttention,
)
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.linear import Linear
from .split_qkv import SplitQKVLinear
from torchtitan.models.common.nn_modules import Identity
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module
from .utils.inits import make_param_init
from .utils.norms import build_norm_config


class GatedNormSWAttention(GQAttention):
    """Gated Norm Sliding Window Attention, built on upstream's GQAttention.

    Token-flat: inputs are ``[T, D]`` and Q/K/V are ``[T, H, K]``. There is no
    batch dimension; document boundaries travel in ``positions`` and in the flex
    ``BlockMask``.

    Q/K/V projection, output projection, RoPE, QK-norm and the inner attention
    backend all come from ``GQAttention``, including its pluggable
    ``qkv_linear``. Only what upstream has no equivalent for is added here:
    V-norm, mid-norm, sigmoid gating and partial RoPE.

    The config keeps the flat, string-valued authoring surface the flavor
    registry uses (``attn_backend="flex"``, ``qk_norm=True``, ...) and
    synthesizes upstream's structured ``GQAttention.Config`` in
    ``to_gqa_config``. Upstream's field of the same name, ``qk_norm``, is a norm
    config rather than a bool, which is why the two are not merged.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        # ``dim`` used to arrive as a build kwarg from the block. Upstream builds
        # every layer config with a bare ``build()``, so the model's config
        # expansion stamps it onto each per-layer copy instead.
        dim: int = 0
        n_kv_heads: int | None = None
        head_dim: int | None = None
        qk_norm: bool = False
        mid_norm: bool = False
        v_norm: bool = False
        norm_everywhere: bool = False
        gated_attention_type: str | None = None  # "none", "head-wise", "element-wise"
        gate_only: bool = False
        norm_eps: float = 1e-30
        norm_type: str = "np_rmsnorm"
        mid_norm_position: str = "after"  # "after" or "before"
        head_wise_mid_norm: bool = False
        use_rope: bool = True
        # flex + block_causal, matching upstream. Upstream removed sdpa as a
        # language-model backend entirely (config_utils.get_attention_config
        # raises on it) because sdpa takes only a boolean is_causal and cannot
        # consume per-document positions. Batches are token-flat, so a plain
        # causal mask spans the whole concatenated stream and attends across
        # the document boundaries that `positions` resets mark, while RoPE has
        # already restarted at each one. Measured cost of flex over sdpa is
        # ~3% of step time; block_causal over causal is free (flex skips
        # fully-masked blocks).
        attn_backend: str = "flex"
        attn_mask_type: str = "block_causal"

        qk_rope_dim: int | None = None
        # Number of head dimensions to apply RoPE to.  None (default) means full
        # head_dim (standard RoPE).  Set to a smaller value for partial RoPE where
        # only the first qk_rope_dim dims are rotated and the rest pass through.
        sliding_window_size: int = -1

        # Filled in by OPTMoEModel.Config expansion from ``attn_backend`` and the
        # model-level rope/rope_of_swa configs. Declared with defaults so the
        # compact flavor registry in __init__.py does not have to spell them out.
        inner_attention: InnerAttention.Config = field(
            default_factory=FlexInnerAttention.Config
        )
        rope: RoPE.Config | None = None

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

        # Depth-init divisor for the output projections, stamped per layer by the
        # model's config expansion (was passed to init_weights(residual_div=...)).
        residual_div: float = 1.0

        linear_sharding_config: Any | None = None
        """ShardingConfig stamped onto every Linear this config synthesizes
        (wq/wk/wv or wqkv, wo, gate_proj).

        Needed because `spmd_backend="spmd_types"` requires EVERY parameter to
        already be a DTensor before `fully_shard(dp_mesh_dims=...)` runs
        ("all parameters must be DTensors on the full SPMD mesh ... Got plain
        tensor"). Upstream models express this by exposing their Linears as
        config FIELDS, so `set_*_sharding` can reach them; this config is flat
        and builds them in `to_gqa_config`, so the plan has to be threaded in.
        With TP off, every Linear takes the same Invariant placement, so one
        config for all of them is sufficient."""
        norm_sharding_config: Any | None = None
        """Sharding plan for the norms this config builds internally (qk_norm,
        v_norm, mid_norm). They are not config fields, so `sharding.py` cannot
        reach them the way it reaches `attention_norm`/`ffn_norm`; it threads
        the plan in here instead. Inert for parameter-free norm types."""

        def to_gqa_config(self) -> "GQAttention.Config":
            """Synthesize upstream's GQAttention.Config from these flat fields."""
            assert self.dim > 0, (
                "GatedNormSWAttention.Config.dim must be stamped by the model's "
                "config expansion before build()."
            )
            n_kv = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
            head_dim = (
                self.head_dim if self.head_dim is not None else self.dim // self.n_heads
            )
            wq_init = {
                "weight": make_param_init(self.wq_init_fn_type, self.wq_init_std)
            }
            wkv_init = {
                "weight": make_param_init(self.wk_init_fn_type, self.wk_init_std)
            }
            if (self.wk_init_fn_type, self.wk_init_std) != (
                self.wv_init_fn_type,
                self.wv_init_std,
            ):
                # SplitQKVLinear builds wk and wv from one shared config, so it cannot
                # express different initializers for them. Fail loudly rather
                # than silently applying wk's to both.
                raise ValueError(
                    "wk and wv initializers must match "
                    f"(got wk={self.wk_init_fn_type}/{self.wk_init_std}, "
                    f"wv={self.wv_init_fn_type}/{self.wv_init_std}): upstream's "
                    "SplitQKVLinear shares one config between them."
                )

            qkv = SplitQKVLinear.Config(
                head_dim=head_dim,
                wq=Linear.Config(
                    in_features=self.dim,
                    out_features=self.n_heads * head_dim,
                    param_init=wq_init,
                ),
                wkv=Linear.Config(
                    in_features=self.dim,
                    out_features=n_kv * head_dim,
                    param_init=wkv_init,
                ),
            )

            qk_norm = None
            if self.qk_norm or self.norm_everywhere:
                qk_norm = build_norm_config(
                    self.norm_type,
                    head_dim,
                    self.norm_eps,
                    sharding_config=self.norm_sharding_config,
                )

            if self.linear_sharding_config is not None:
                for _sub in ("wqkv", "wq", "wkv", "wk", "wv"):
                    _lin = getattr(qkv, _sub, None)
                    if _lin is not None:
                        _lin.sharding_config = self.linear_sharding_config

            return GQAttention.Config(
                n_heads=self.n_heads,
                n_kv_heads=n_kv,
                head_dim=self.head_dim,
                dim=self.dim,
                qkv_linear=qkv,
                wo=Linear.Config(
                    in_features=self.n_heads * head_dim,
                    out_features=self.dim,
                    param_init={
                        "weight": make_param_init(
                            self.wo_init_fn_type, self.wo_init_std, self.residual_div
                        )
                    },
                    sharding_config=self.linear_sharding_config,
                ),
                qk_norm=qk_norm,
                inner_attention=self.inner_attention,
                rope=self.rope,
            )

        def build_inner_attention(self) -> Module.Config:
            """Map ``attn_backend`` onto an upstream inner-attention config.

            Delegates to upstream's `get_attention_config` rather than keeping a
            private copy of the mapping. The private copy is what let this model
            drift: upstream deprecated SDPA and the "causal" mask type for
            language models in pytorch/torchtitan#3571 ("Today, text dataloader
            and chat dataloader always output `positions` for `Decoder` to
            consume, which SDPA could not handle"), but the local copy kept the
            `sdpa` case and defaulted to it, so this was the only language model
            in the repo training without document masking. Delegating also makes
            `flex_flash` selectable -- though it does not lower in this torch
            build ("BACKEND='FLASH' but flash attention not available"), so it
            is reachable, not usable, here.
            """
            return get_attention_config(self.attn_backend)

    def __init__(self, config: Config):
        # GQAttention builds qkv_linear, wo, rope, inner_attention, q/k norm and
        # the scaling factor. Only the pieces upstream has no equivalent for are
        # added below.
        super().__init__(config.to_gqa_config())
        self.config = config

        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
            )
        self.use_rope = config.use_rope
        self.qk_rope_dim: int = (
            config.qk_rope_dim if config.qk_rope_dim is not None else self.head_dim
        )
        assert (
            self.qk_rope_dim <= self.head_dim
        ), f"qk_rope_dim ({self.qk_rope_dim}) must be <= head_dim ({self.head_dim})"
        # `_apply_rope` slices Q/K to qk_rope_dim and hands them to a RoPE whose
        # cos/sin cache is sized by `rope.dim`. If the two disagree the rotation
        # silently uses the wrong frequencies (or broadcasts), so pin them.
        if config.rope is not None and config.rope.dim != self.qk_rope_dim:
            raise ValueError(
                f"rope.dim ({config.rope.dim}) must equal the rotated width "
                f"qk_rope_dim ({self.qk_rope_dim}); the RoPE cache is built at "
                "rope.dim but only qk_rope_dim head dims are rotated."
            )
        self.partial_rope = self.qk_rope_dim != self.head_dim
        if not self.use_rope:
            # GQAttention always builds a RoPE; drop it so NoPE layers carry no
            # unused cache and cannot accidentally apply it.
            self.rope = None

        self.gated_attention_type = config.gated_attention_type
        self.gate_only = config.gate_only
        self.sliding_window_size = config.sliding_window_size
        self.mid_norm_position = config.mid_norm_position
        assert self.mid_norm_position in (
            "after",
            "before",
        ), f"mid_norm_position ({self.mid_norm_position}) must be 'after' or 'before'"
        self.head_wise_mid_norm = config.head_wise_mid_norm
        self.attn_backend = config.attn_backend

        identity = Identity.Config()
        self.v_norm = identity.build()
        self.mid_norm = identity.build()
        self.gate_proj = identity.build()

        dim = config.dim
        if config.v_norm or config.norm_everywhere:
            self.v_norm = build_norm_config(
                config.norm_type,
                self.head_dim,
                config.norm_eps,
                sharding_config=config.norm_sharding_config,
            ).build()
        if config.mid_norm or config.norm_everywhere:
            mid_dim = (
                self.n_heads * self.head_dim
                if self.mid_norm_position == "after" and not self.head_wise_mid_norm
                else self.head_dim
            )
            self.mid_norm = build_norm_config(
                config.norm_type,
                mid_dim,
                config.norm_eps,
                sharding_config=config.norm_sharding_config,
            ).build()

        gate_init = {
            "weight": make_param_init(
                config.w_gate_init_fn_type, config.w_gate_init_std, config.residual_div
            )
        }
        if self.gated_attention_type == "head-wise":
            # G1-style: one gate per attention head.
            self.gate_proj = Linear.Config(
                in_features=dim,
                out_features=self.n_heads,
                param_init=gate_init,
                sharding_config=config.linear_sharding_config,
            ).build()
        elif self.gated_attention_type == "element-wise":
            # Dense gate over the full attention output channel dimension.
            self.gate_proj = Linear.Config(
                in_features=dim,
                out_features=self.n_heads * self.head_dim,
                param_init=gate_init,
                sharding_config=config.linear_sharding_config,
            ).build()

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

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        # Q/K/V projection comes from GQAttention's pluggable qkv_linear, so
        # fused and split projections are both available.
        xq, xk, xv = self.qkv_linear(x_TD)

        # QK norm before RoPE (per Qwen3); V norm has no upstream equivalent.
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)
        xv = self.v_norm(xv)

        if self.use_rope:
            xq, xk = self._apply_rope(xq, xk, positions)

        # No sdpa branch: `build_inner_attention` delegates to upstream's
        # `get_attention_config`, which RAISES for "sdpa" ("no longer supported for
        # language models; positions are always available"), and `model.py:161-168`
        # independently rejects any backend outside flex/flex_flash/varlen. Language
        # models always use block_causal masking, which sdpa cannot express. Flex and
        # varlen are natively token-flat, so there is no batch dim to add or drop.
        scale_kwargs = {"scale": self.scaling} if self.scaling is not None else {}
        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            enable_gqa=self.enable_gqa,
            **scale_kwargs,
        )
        output = output.contiguous()

        # "before" mid-norm: per-head norm applied before gating.
        # output shape here: [T, n_local_heads, head_dim]
        if self.mid_norm_position == "before" and not self.gate_only:
            output = self.mid_norm(output)

        # The string "none" is listed as a legal value for this field, but it
        # is truthy: it entered this block, computed a full fp32 sigmoid over
        # [T, D] and then matched neither branch below, so the result was
        # discarded -- no gating, plus the cost of computing one.
        if self.gated_attention_type not in (None, "none"):
            # Gate in float32 for numeric stability, then cast back.
            orig_dtype = output.dtype
            gate = torch.sigmoid(self.gate_proj(x_TD).float())
            if self.gated_attention_type == "head-wise":
                # gate: [T, n_local_heads] -> broadcast over head_dim
                output = (output.float() * gate.unsqueeze(-1)).to(orig_dtype)
            elif self.gated_attention_type == "element-wise":
                # gate: [T, n_local_heads * head_dim]
                output = (output.reshape(num_tokens, -1).float() * gate).to(orig_dtype)
            else:
                raise ValueError(
                    f"unknown gated_attention_type "
                    f"{self.gated_attention_type!r}; the gate would be "
                    "computed and silently discarded."
                )

        # "after" mid-norm: per-head or over the concatenated head output.
        if self.mid_norm_position == "after" and not self.gate_only:
            if self.head_wise_mid_norm:
                output = output.reshape(num_tokens, -1, self.head_dim)
                output = self.mid_norm(output)
                output = output.reshape(num_tokens, -1)
            else:
                output = output.reshape(num_tokens, -1)
                output = self.mid_norm(output)
        else:
            output = output.reshape(num_tokens, -1)
        return self.wo(output)
