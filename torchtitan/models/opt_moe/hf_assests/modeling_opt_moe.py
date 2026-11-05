# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF implementation of torchtitan's opt_moe (`torchtitan/models/opt_moe`).

Every op is the one the native forward runs, in the same order and precision,
so the export reproduces the trainer's forward: fp32 RoPE, parameter-free
`F.rms_norm`, bf16 grouped expert GEMMs, the fp32-output router gate, the fp32
score multiply and a deterministic scatter-add. With
`attn_implementation="torchtitan_flex"` attention is the native compiled flex
kernel too, and a prefill matches the native forward bitwise; the stock
backends (sdpa, flash, eager) differ only in the attention kernel.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import AuxRequest, flex_attention

from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_experts_implementation
from transformers.masking_utils import (
    AttentionMaskInterface,
    create_causal_mask,
    create_sliding_window_causal_mask,
    flex_attention_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    AttentionInterface,
    PreTrainedModel,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    eager_attention_forward as eager_attention_with_sinks,
)
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_opt_moe import OptMoEConfig

# ---------------------------------------------------------------- attention
# torchtitan's FlexInnerAttention compiles flex_attention with these options;
# the same options pick the same kernel, so the result is the native one.
_FLEX_OPTIONS = {
    "wrap_inductor_compiled_regions": True,
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": False,
}
_compiled_flex: Callable | None = None


def torchtitan_flex_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    s_aux: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """`[B, H, T, K]` q/k/v and a BlockMask -> `[B, T, H, K]`, as torchtitan's
    flex call (GQA without repeating k/v); `s_aux` (sinks) rescales by
    `sigmoid(lse - sink)` as torchtitan's out_transform does."""
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, options=_FLEX_OPTIONS)
    out, aux = _compiled_flex(
        query,
        key,
        value,
        block_mask=attention_mask,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
        return_aux=AuxRequest(lse=s_aux is not None),
        kernel_options={},
    )
    out = out.transpose(1, 2)
    if s_aux is not None:
        lse = aux.lse.transpose(1, 2)
        out = out * torch.sigmoid(lse - s_aux.view(1, 1, -1)).unsqueeze(-1).to(out.dtype)
    return out, None


AttentionInterface.register("torchtitan_flex", torchtitan_flex_attention)
AttentionMaskInterface.register("torchtitan_flex", flex_attention_mask)


# ---------------------------------------------------------------- norms, rope
def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """opt_moe's parameter-free RMSNorm (`nn.RMSNorm`, no affine)."""
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


def causal_short_conv(x, weight, positions):
    """`[B, T, C]` -> `silu` of a depthwise causal conv (`nn.Conv1d`
    orientation), accumulated in fp32; tap `j` only where `positions >= j`, so a
    packed document never sees the previous one (as torchtitan's)."""
    k = weight.shape[-1]
    x32 = x.float()
    w = weight.squeeze(1).float()
    if positions is None:
        positions = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
    y = x32 * w[:, k - 1]
    for j in range(1, k):
        back = F.pad(x32[:, :-j], (0, 0, j, 0))
        y = y + torch.where((positions >= j).unsqueeze(-1), back, 0.0) * w[:, k - 1 - j]
    return F.silu(y).to(x.dtype)


def _is_sliding(layer_type: str) -> bool:
    return layer_type in ("sliding_attention", "hybrid_sliding")


class OptMoERotaryEmbedding(nn.Module):
    """cos/sin of torchtitan's `CosSinRoPE`, kept in fp32: computed on the
    input's device (as the native cache is, on the trainer's device) from
    `positions * inv_freq`, which equals the cache row `outer(t, inv_freq)`."""

    def __init__(self, rope_parameters: dict, dim: int):
        super().__init__()
        if rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("opt_moe exports only the default RoPE")
        self.theta = rope_parameters["rope_theta"]
        self.dim = dim

    @torch.no_grad()
    def forward(self, x, position_ids) -> tuple[torch.Tensor, torch.Tensor]:
        dim = self.dim
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, dim, 2, device=x.device)[: (dim // 2)].float() / dim)
        )
        freqs = position_ids.to(inv_freq.dtype)[..., None] * inv_freq
        theta = torch.cat([freqs, freqs], dim=-1)
        return theta.cos(), theta.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """`[B, T, H, D]` q/k and `[B, T, D']` fp32 cos/sin: the first `D'` dims
    rotate in fp32, the rest pass through."""
    d = cos.shape[-1]
    cos, sin = cos[:, :, None], sin[:, :, None]

    def rotate(x):
        xr = x[..., :d].float()
        out = ((xr * cos) + (_rotate_half(xr) * sin)).type_as(x)
        return out if d == x.shape[-1] else torch.cat([out, x[..., d:]], dim=-1)

    return rotate(q), rotate(k)


class OptMoEAttention(nn.Module):
    def __init__(self, config: OptMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.n_heads = config.num_attention_heads
        self.num_key_value_groups = self.n_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.eps = config.rms_norm_eps
        self.use_rope = config.use_rope_layers[layer_idx]
        self.sliding_window = (
            config.sliding_window if _is_sliding(config.layer_types[layer_idx]) else None
        )
        bias = config.attention_bias
        d = config.hidden_size
        kv = config.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d, kv, bias=bias)
        self.v_proj = nn.Linear(d, kv, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=bias)
        self.gated = config.gated_attention
        if self.gated is not None:
            width = self.n_heads if self.gated == "head-wise" else self.n_heads * self.head_dim
            self.gate_proj = nn.Linear(d, width, bias=False)
        self.convs = None
        if config.conv_kernel_size > 0:
            k = config.conv_kernel_size
            self.q_conv, self.k_conv, self.v_conv = (
                nn.Conv1d(c, c, k, groups=c, bias=False)
                for c in (self.n_heads * self.head_dim, kv, kv)
            )
            self.convs = (self.q_conv, self.k_conv, self.v_conv)
        self.sinks = nn.Parameter(torch.empty(self.n_heads)) if config.attention_sinks else None

    def _short_conv(self, x, state_idx, positions, cache):
        """The q (0), k (1) or v (2) short conv. With a cache, the conv state
        carries the last `k - 1` inputs into the next call."""
        conv = self.convs[state_idx]
        if cache is None:
            return causal_short_conv(x, conv.weight, positions)
        width = conv.weight.shape[-1] - 1
        decoding = cache.layers[self.layer_idx].has_previous_state[state_idx]
        full = cache.update_conv_state(
            x.transpose(1, 2), self.layer_idx, state_idx, conv_kernel_size=width
        ).transpose(1, 2)
        if not decoding:
            return causal_short_conv(x, conv.weight, positions)
        # The cached history (zero-padded after a short prefill) then the new tokens.
        return causal_short_conv(full, conv.weight, None)[:, -x.shape[1] :]

    def _mid_norm(self, out: torch.Tensor, position: str) -> torch.Tensor:
        """`[B, T, H, D]` in; the attention mid-norm at `position`."""
        cfg = self.config
        if not cfg.attention_mid_norm or cfg.mid_norm_position != position:
            return out
        if position == "before" or cfg.head_wise_mid_norm:
            return rms_norm(out, self.eps)
        return rms_norm(out.flatten(-2), self.eps).view_as(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cfg = self.config
        B, T, _ = hidden_states.shape
        q, k, v = (proj(hidden_states) for proj in (self.q_proj, self.k_proj, self.v_proj))
        if self.convs is not None:
            positions = kwargs.get("position_ids")
            q, k, v = (
                self._short_conv(t, i, positions, past_key_values)
                for i, t in enumerate((q, k, v))
            )
        q, k, v = (t.view(B, T, -1, self.head_dim) for t in (q, k, v))
        if cfg.qk_norm:
            q, k = rms_norm(q, self.eps), rms_norm(k, self.eps)
        if cfg.v_norm:
            v = rms_norm(v, self.eps)
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        eager = eager_attention_forward
        if self.sinks is not None:
            if cfg._attn_implementation not in ("torchtitan_flex", "eager"):
                raise ValueError(
                    "attention sinks need attn_implementation torchtitan_flex or eager"
                )
            kwargs["s_aux"] = self.sinks
            eager = eager_attention_with_sinks
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            cfg._attn_implementation, eager
        )
        out, weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        out = self._mid_norm(out.contiguous(), "before")
        if self.gated is not None:
            dtype = out.dtype
            gate = torch.sigmoid(self.gate_proj(hidden_states).float())
            if self.gated == "head-wise":
                out = (out.float() * gate.unsqueeze(-1)).to(dtype)
            else:
                out = (out.flatten(-2).float() * gate).to(dtype).view_as(out)
        out = self._mid_norm(out, "after")
        return self.o_proj(out.reshape(B, T, -1)), weights


# ---------------------------------------------------------------- MLPs
class OptMoEMLP(nn.Module):
    """SwiGLU `down(silu(gate(x)) * up(x))`, optionally RMS-normed before `down`."""

    def __init__(self, config: OptMoEConfig, intermediate_size: int, mid_norm: bool):
        super().__init__()
        d = config.hidden_size
        self.gate_proj = nn.Linear(d, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d, bias=False)
        self.mid_norm = mid_norm
        self.eps = config.rms_norm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.mid_norm:
            h = rms_norm(h, self.eps)
        return self.down_proj(h)


@use_experts_implementation
class OptMoEExperts(nn.Module):
    """Routed experts as 3D tensors: `gate_up_proj` `[E, 2F, D]` (gate rows,
    then up rows) and `down_proj` `[E, D, F]` -- torchtitan's `w13` flattened
    and `w2`. This forward ("eager", the default) is torchtitan's: tokens
    sorted by expert, bf16 grouped GEMMs, the fp32 score multiply, a
    deterministic scatter-add. transformers' backends (`grouped_mm`, ...) are
    selectable and compute the same up to rounding."""

    def __init__(self, config: OptMoEConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        f, d = config.moe_intermediate_size, config.hidden_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * f, d))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, d, f))
        self.act_fn = F.silu
        self.mid_norm = config.moe_mid_norm
        self.eps = config.rms_norm_eps

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        return rms_norm(h, self.eps) if self.mid_norm else h

    def _grouped_mm(self, x, weight, offsets):
        if x.is_cuda:
            return torch._grouped_mm(x, weight.bfloat16().transpose(-2, -1), offs=offsets)
        out, start = x.new_empty(x.shape[0], weight.shape[1]), 0
        for e, end in enumerate(offsets.tolist()):
            out[start:end] = x[start:end] @ weight[e].bfloat16().T
            start = end
        return out

    def forward(self, hidden_states, top_k_index, top_k_weights):
        k = top_k_index.shape[-1]
        order = torch.argsort(top_k_index.view(-1), stable=True)
        weights = top_k_weights.view(-1)[order]
        token_idx = order // k
        routed = hidden_states[token_idx]
        counts = torch.bincount(top_k_index.view(-1), minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        h = self._apply_gate(self._grouped_mm(routed.bfloat16(), self.gate_up_proj, offsets))
        out = self._grouped_mm(h, self.down_proj, offsets).type_as(routed)
        out = (out.float() * weights.reshape(-1, 1)).to(out.dtype)
        index = token_idx.reshape(-1, 1).expand(-1, out.shape[-1])
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            return torch.zeros_like(hidden_states).scatter_add(0, index, out)
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class OptMoETopKRouter(nn.Module):
    """Scores from an fp32-output gate, top-k on the scores plus the balancing
    bias (the bias only selects), renormalised and scaled."""

    def __init__(self, config: OptMoEConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        # fp32 as in training (a buffer, never cast), whatever the load dtype.
        self.e_score_correction_bias = nn.Buffer(
            torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self.weight
        if x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            logits = torch.mm(x, w.T, out_dtype=torch.float32)
        else:
            logits = torch.mm(x.float(), w.float().T)
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        elif self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            scores = F.softplus(logits).sqrt()
        choice = scores + self.e_score_correction_bias
        top_k_index = torch.topk(choice, k=self.top_k, dim=-1, sorted=False).indices
        top_k_weights = scores.gather(dim=-1, index=top_k_index)
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, top_k_weights * self.routed_scaling_factor, top_k_index


class OptMoESparseMoeBlock(nn.Module):
    def __init__(self, config: OptMoEConfig):
        super().__init__()
        self.gate = OptMoETopKRouter(config)
        self.experts = OptMoEExperts(config)
        self.shared_experts = (
            OptMoEMLP(
                config,
                config.moe_intermediate_size * config.n_shared_experts,
                config.moe_mid_norm,
            )
            if config.n_shared_experts > 0
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        _, top_k_weights, top_k_index = self.gate(x)
        out = self.experts(x, top_k_index, top_k_weights)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out.view(shape)


# ---------------------------------------------------------------- layers
class OptMoEDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OptMoEConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = OptMoEAttention(config, layer_idx)
        if layer_idx < config.first_k_dense_replace:
            self.mlp = OptMoEMLP(config, config.intermediate_size, config.mlp_mid_norm)
        else:
            self.mlp = OptMoESparseMoeBlock(config)
        self.eps = config.rms_norm_eps
        self.identity_scale = config.residual_identity_scale
        self.block_scale = config.residual_block_scale

    def _residual(self, h: torch.Tensor, block_out: torch.Tensor) -> torch.Tensor:
        lhs = h if self.identity_scale == 1.0 else self.identity_scale * h
        rhs = block_out if self.block_scale == 1.0 else self.block_scale * block_out
        return lhs + rhs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            rms_norm(hidden_states, self.eps),
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        h = self._residual(hidden_states, attn_out)
        return self._residual(h, self.mlp(rms_norm(h, self.eps)))


@auto_docstring
class OptMoEPreTrainedModel(PreTrainedModel):
    config: OptMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OptMoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OptMoEDecoderLayer,
        "attentions": OptMoEAttention,
    }

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        # Unrequested -> this model's own (torchtitan's) forward, not grouped_mm.
        return super().get_correct_experts_implementation(requested_experts or "eager")

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, OptMoEExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, OptMoETopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, OptMoEAttention) and module.sinks is not None:
            init.normal_(module.sinks, mean=0.0, std=std)


@auto_docstring
class OptMoEModel(OptMoEPreTrainedModel):
    def __init__(self, config: OptMoEConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [OptMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.rotary_emb = OptMoERotaryEmbedding(config.rope_parameters, config.rotary_dim)
        self.rotary_emb_sliding = (
            OptMoERotaryEmbedding(config.rope_parameters_sliding, config.rotary_dim)
            if config.rope_parameters_sliding is not None
            else None
        )
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            ).unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        full = create_causal_mask(**mask_kwargs)
        sliding = (
            create_sliding_window_causal_mask(**mask_kwargs)
            if any(map(_is_sliding, self.config.layer_types))
            else None
        )

        h = inputs_embeds
        if self.config.embeddings_norm:
            h = rms_norm(h, self.config.rms_norm_eps)
        rope = self.rotary_emb(h, position_ids)
        rope_sliding = (
            self.rotary_emb_sliding(h, position_ids)
            if self.rotary_emb_sliding is not None
            else rope
        )
        for layer in self.layers:
            is_sliding = _is_sliding(layer.attention_type)
            h = layer(
                h,
                attention_mask=sliding if is_sliding else full,
                position_embeddings=rope_sliding if is_sliding else rope,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        return MoeModelOutputWithPast(
            last_hidden_state=rms_norm(h, self.config.rms_norm_eps),
            past_key_values=past_key_values,
        )


@auto_docstring
class OptMoEForCausalLM(OptMoEPreTrainedModel, GenerationMixin):
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: OptMoEConfig):
        super().__init__(config)
        self.model = OptMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        h = outputs.last_hidden_state
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, keep, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["OptMoEForCausalLM", "OptMoEModel", "OptMoEPreTrainedModel"]
