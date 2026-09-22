# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from collections.abc import Callable
from typing import Unpack

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin

from transformers.integrations import use_kernel_forward_from_hub

from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.modeling_rope_utils import dynamic_rope_update, ROPE_INIT_FUNCTIONS

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from transformers.utils import auto_docstring, can_return_tuple, TransformersKwargs

from .configuration_opt_moe import OptMoEConfig


logger = logging.getLogger(__name__)


def _get_attention_interface(attn_implementation: str) -> Callable:
    if attn_implementation in (None, "eager"):
        return eager_attention_forward
    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        try:
            return ALL_ATTENTION_FUNCTIONS.get_interface(
                attn_implementation, eager_attention_forward
            )
        except TypeError:
            return ALL_ATTENTION_FUNCTIONS.get_interface(attn_implementation)
    if isinstance(ALL_ATTENTION_FUNCTIONS, dict):
        return ALL_ATTENTION_FUNCTIONS[attn_implementation]
    return ALL_ATTENTION_FUNCTIONS[attn_implementation]


def _build_dynamic_cache(config: OptMoEConfig) -> DynamicCache:
    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


def _causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Helper: parse per-layer pattern strings ("RRRN", "SSSF") into bool lists
# ---------------------------------------------------------------------------


def _parse_pattern(
    pattern: "str | list | None",
    n_layers: int,
    true_char: str,
    false_char: str,
    default: bool = True,
) -> "list[bool]":
    """Convert a layer pattern to a list of booleans.

    Args:
        pattern: None, a string like "RRRN"/"SSSF", or a list of bool/str.
        n_layers: Number of layers.
        true_char: Character that maps to True (e.g. 'R' or 'S').
        false_char: Character that maps to False (e.g. 'N' or 'F').
        default: Default value when pattern is None.
    """
    if pattern is None:
        return [default] * n_layers
    if isinstance(pattern, (list, tuple)):
        # Keep compatibility with native parser:
        #   ['SSSF'] and ['S', 'S', 'S', 'F'] are valid string forms.
        if len(pattern) == 1 and isinstance(pattern[0], str):
            pattern = pattern[0]
        elif pattern and all(isinstance(v, str) and len(v) == 1 for v in pattern):
            pattern = "".join(pattern)
        else:
            if len(pattern) != n_layers:
                raise ValueError(
                    f"Pattern list length {len(pattern)} != n_layers {n_layers}"
                )
            if not all(isinstance(v, bool) for v in pattern):
                raise ValueError(
                    "Pattern list must be list[bool], ['PATTERN'], or list of single-character strings."
                )
            return list(pattern)
    # string
    if len(pattern) != n_layers:
        raise ValueError(f"Pattern string length {len(pattern)} != n_layers {n_layers}")
    result = []
    for c in pattern:
        if c.upper() == true_char.upper():
            result.append(True)
        elif c.upper() == false_char.upper():
            result.append(False)
        else:
            raise ValueError(
                f"Unknown character '{c}' in pattern '{pattern}'. "
                f"Expected '{true_char}' or '{false_char}'."
            )
    return result


def _normalize_gated_attention_type(value: "str | None") -> "str | None":
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized == "head-wise":
        return "head-wise"
    if normalized == "element-wise":
        return "element-wise"
    return value


def _normalize_mid_norm_position(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"after", "before"}:
        raise ValueError(
            "mid_norm_position must be either 'after' or 'before', " f"got {value!r}"
        )
    return normalized


def _get_attention_norm_everywhere(config) -> bool:
    return bool(
        getattr(
            config,
            "attention_norm_everywhere",
            getattr(config, "norm_everywhere", False),
        )
    )


def _get_ffn_norm_everywhere(config) -> bool:
    return bool(
        getattr(
            config,
            "ffn_norm_everywhere",
            getattr(config, "norm_everywhere", False),
        )
    )


def _get_moe_norm_everywhere(config) -> bool:
    return bool(
        getattr(
            config,
            "moe_norm_everywhere",
            getattr(
                config,
                "ffn_norm_everywhere",
                getattr(config, "norm_everywhere", False),
            ),
        )
    )


@use_kernel_forward_from_hub("RMSNorm")
class OptMoERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"eps={self.variance_epsilon}"


class OptMoERotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    @staticmethod
    def compute_default_rope_parameters(
        config: OptMoEConfig | None = None,
        device: "torch.device | None" = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """Compute inverse frequencies for the non-scaled/default RoPE path."""
        assert config is not None
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    def __init__(self, config: OptMoEConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # Normalize to the current transformers rope_parameters contract.
        self.config.standardize_rope_params()
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class OptMoEAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OptMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.qk_rope_dim = getattr(config, "qk_rope_dim", self.head_dim)
        if not (0 < self.qk_rope_dim <= self.head_dim):
            raise ValueError(
                f"qk_rope_dim must be in (0, head_dim], got {self.qk_rope_dim} "
                f"for head_dim={self.head_dim}."
            )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.gate_only = bool(getattr(config, "gate_only", False))
        self.mid_norm_position = _normalize_mid_norm_position(
            getattr(config, "mid_norm_position", "after")
        )
        self.head_wise_mid_norm = bool(getattr(config, "head_wise_mid_norm", False))

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        attention_norm_everywhere = _get_attention_norm_everywhere(config)
        qk_norm = config.qk_norm or attention_norm_everywhere
        if qk_norm:
            self.q_norm = OptMoERMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = OptMoERMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if attention_norm_everywhere:
            self.v_norm = OptMoERMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.v_norm = nn.Identity()

        use_mid_norm = (
            bool(getattr(config, "mid_norm", False)) or attention_norm_everywhere
        )
        if use_mid_norm:
            if self.mid_norm_position == "after" and not self.head_wise_mid_norm:
                self.mid_norm = OptMoERMSNorm(
                    config.num_attention_heads * self.head_dim,
                    config.rms_norm_eps,
                )
            else:
                self.mid_norm = OptMoERMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.mid_norm = nn.Identity()

        # Gated attention: head-wise or element-wise gate projection
        self.gated_attention_type = _normalize_gated_attention_type(
            getattr(config, "gated_attention_type", None)
        )
        self.gate_proj = nn.Identity()
        if self.gated_attention_type == "head-wise":
            self.gate_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads, bias=False
            )
        elif self.gated_attention_type == "element-wise":
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False,
            )

        # Per-layer flags — set by OptMoEModel after layer creation
        self.use_rope: bool = True
        self.sliding_window: int | None = None
        # True when this layer is a SWA layer that should use rotary_emb_swa
        # (i.e. rope_theta_swa is configured and this layer uses SWA + RoPE)
        self.use_swa_rope: bool = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states.contiguous())
        key_states = self.k_norm(key_states.contiguous())
        value_states = self.v_norm(value_states.contiguous())

        # Apply RoPE only for layers that use positional encoding
        if self.use_rope and position_embeddings is not None:
            cos, sin = position_embeddings
            if self.qk_rope_dim != self.head_dim:
                query_rot, query_pass = (
                    query_states[..., : self.qk_rope_dim],
                    query_states[..., self.qk_rope_dim :],
                )
                key_rot, key_pass = (
                    key_states[..., : self.qk_rope_dim],
                    key_states[..., self.qk_rope_dim :],
                )
                query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
                query_states = torch.cat((query_rot, query_pass), dim=-1)
                key_states = torch.cat((key_rot, key_pass), dim=-1)
            else:
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

        if past_key_values is not None:
            if self.use_rope and position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = None, None
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        use_backend_attention = self.config._attn_implementation != "eager"
        if (
            use_backend_attention
            and isinstance(attention_mask, torch.Tensor)
            and attention_mask.dim() == 4
            and self.config._attn_implementation
            in {"flash_attention_2", "flash_attention_3", "flex_attention"}
        ):
            use_backend_attention = False
        if use_backend_attention:
            attention_interface = _get_attention_interface(
                self.config._attn_implementation
            )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        if self.mid_norm_position == "before" and not self.gate_only:
            attn_output = self.mid_norm(attn_output)

        # Apply gated attention output (head-wise or element-wise)
        if self.gated_attention_type is not None:
            orig_dtype = attn_output.dtype
            gate = torch.sigmoid(self.gate_proj(hidden_states).float())
            if self.gated_attention_type == "head-wise":
                # gate: [bs, seq, n_heads] → [bs, seq, n_heads, 1] for broadcasting
                bsz, seq_len = hidden_states.shape[:2]
                n_heads = self.config.num_attention_heads
                gate = gate.unsqueeze(-1)  # [bs, seq, n_heads, 1]
                attn_output = attn_output.view(bsz, seq_len, n_heads, self.head_dim)
                attn_output = (attn_output.float() * gate).to(orig_dtype)
            else:  # element-wise
                attn_output = (attn_output.reshape(*input_shape, -1).float() * gate).to(
                    orig_dtype
                )

        if self.mid_norm_position == "after" and not self.gate_only:
            if self.head_wise_mid_norm:
                bsz, seq_len = input_shape
                n_heads = self.config.num_attention_heads
                attn_output = attn_output.reshape(bsz, seq_len, n_heads, self.head_dim)
                attn_output = self.mid_norm(attn_output)
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            else:
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.mid_norm(attn_output)
        else:
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OptMoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

        if _get_ffn_norm_everywhere(config):
            self.mid_norm = OptMoERMSNorm(self.intermediate_size, config.rms_norm_eps)
        else:
            self.mid_norm = nn.Identity()

    def forward(self, x):
        return self.down_proj(
            self.mid_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


class OptMoESharedExperts(nn.Module):
    def __init__(
        self,
        hidden_size,
        moe_intermediate_size,
        norm_everywhere,
        rms_norm_eps,
        hidden_act,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(moe_intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

        if norm_everywhere:
            self.mid_norm = OptMoERMSNorm(moe_intermediate_size, rms_norm_eps)
        else:
            self.mid_norm = nn.Identity()

    def forward(self, x):
        return self.down_proj(
            self.mid_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


# =====


class TokenChoiceTopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k, route_scale):
        super().__init__()

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Router logits are always computed in fp32 for stable routing decisions.
        scores = torch.nn.functional.linear(
            x.to(torch.float32), self.gate.weight.to(torch.float32)
        )
        scores = torch.sigmoid(scores)

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale

        return selected_experts_indices, top_scores


class OptMoEMoE(nn.Module):
    experts_parallel_enabled = False

    def __init__(
        self,
        config: OptMoEConfig,
    ):
        super().__init__()
        shared_hidden = int(config.moe_intermediate_size) * max(
            1, int(getattr(config, "n_shared_experts", 1))
        )
        moe_norm_everywhere = _get_moe_norm_everywhere(config)
        self.shared_experts = OptMoESharedExperts(
            hidden_size=config.hidden_size,
            moe_intermediate_size=shared_hidden,
            norm_everywhere=moe_norm_everywhere,
            rms_norm_eps=config.rms_norm_eps,
            hidden_act=config.hidden_act,
        )

        self.experts = nn.ModuleList(
            [
                OptMoESharedExperts(
                    hidden_size=config.hidden_size,
                    moe_intermediate_size=config.moe_intermediate_size,
                    norm_everywhere=moe_norm_everywhere,
                    rms_norm_eps=config.rms_norm_eps,
                    hidden_act=config.hidden_act,
                )
                for _ in range(config.n_total_experts)
            ]
        )

        self.router = TokenChoiceTopKRouter(
            dim=config.hidden_size,
            num_experts=config.n_total_experts,
            top_k=config.n_active_experts,
            route_scale=config.moe_scaling_factor,
        )

        self.register_buffer(
            "expert_bias", torch.zeros(config.n_total_experts, dtype=torch.float32)
        )

    def moe_infer(
        self,
        x: torch.Tensor,  # [T, D]
        top_scores: torch.Tensor,  # [T, K]
        selected_experts_indices: torch.Tensor,  # [T, K]
    ) -> torch.Tensor:  # -> [T, D]
        T, D = x.shape
        K = selected_experts_indices.size(1)
        num_experts = len(self.experts)

        # Flatten expert indices: [T, K] -> [T*K]
        sel_flat = selected_experts_indices.view(-1)  # [T*K]

        # Sort assignments by expert id so tokens for each expert are contiguous
        sort_idx = torch.argsort(sel_flat, stable=True)  # [T*K]

        # Sorted expert ids (not strictly needed but nice for sanity)
        sel_sorted = sel_flat[sort_idx]  # [T*K]

        # How many (token, top-k) assignments per expert
        tokens_per_expert = torch.bincount(
            sel_sorted,
            minlength=num_experts,
        )  # [num_experts]

        # For each sorted assignment, what original token index did it come from?
        # (since each token has K slots, its slots are contiguous in the flattened array)
        token_indices = sort_idx // K  # [T*K] in [0, T)

        # Build indices to gather the inputs in "expert-sorted" order
        token_indices_2d = token_indices.unsqueeze(-1).expand(-1, D)  # [T*K, D]

        # routed_input: each token replicated K times, grouped by expert
        routed_input = torch.gather(x, dim=0, index=token_indices_2d)  # [T*K, D]

        # Run each expert on its contiguous slice
        routed_output = torch.empty_like(routed_input)
        start = 0
        for expert_idx, n_tokens in enumerate(tokens_per_expert.tolist()):
            if n_tokens == 0:
                continue
            end = start + n_tokens

            expert_in = routed_input[start:end]  # [n_tokens, D]
            expert_out = self.experts[expert_idx](expert_in)  # [n_tokens, D]

            routed_output[start:end] = expert_out.to(routed_input.dtype)
            start = end

        # Reorder top_scores to match expert-sorted assignments
        gate_flat_sorted = top_scores.view(-1)[sort_idx]  # [T*K]

        # Apply gating
        weighted_output = routed_output * gate_flat_sorted.unsqueeze(-1)  # [T*K, D]

        # Scatter-add back to original token positions
        out = torch.zeros_like(x)
        out = out.scatter_add(
            dim=0, index=token_indices_2d, src=weighted_output.to(out.dtype)
        )

        return out  # [T, D]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [batch_size, seq_len, hidden_dim]
        """
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape

        # Flatten tokens for routing / expert computation
        x = hidden_states.view(-1, h)  # [T, H], T = bsz * seq_len

        topk_idx, topk_weight = self.router(x, self.expert_bias)

        # MoE path
        moe_out = self.moe_infer(x, topk_weight, topk_idx)  # [T, H]
        moe_out = moe_out.view(bsz, seq_len, h)

        # Add shared experts residual
        y = moe_out + self.shared_experts(identity)
        return y


def _compute_residual_scales(residual_scale: str, n_layers: int) -> tuple[float, float]:
    """Return (block_scale, identity_scale) matching native setup_residual_scale."""
    if residual_scale == "depth_scale":
        total_depth = 2 * n_layers
        return 1.0 / total_depth, (total_depth - 1) / total_depth
    return 1.0, 1.0  # "identity"


class OptMoEDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OptMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = "full_attention"

        residual_scale = getattr(config, "residual_scale", "identity")
        self.block_scale, self.identity_scale = _compute_residual_scales(
            residual_scale, config.num_hidden_layers
        )

        self.self_attn = OptMoEAttention(config=config, layer_idx=layer_idx)

        if layer_idx < config.n_dense_layers:
            self.mlp = OptMoEMLP(config)
        else:
            self.mlp = OptMoEMoE(config)
        self.input_layernorm = OptMoERMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = OptMoERMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Per-layer sliding window size (-1 = full attention, >0 = SWA).
        # Set by OptMoEModel after construction.
        self.sliding_window: int = -1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = (
            self.identity_scale * residual + self.block_scale * hidden_states
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = (
            self.identity_scale * residual + self.block_scale * hidden_states
        )

        return hidden_states


@auto_docstring
class OptMoEPreTrainedModel(PreTrainedModel):
    config_class = OptMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OptMoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True


@auto_docstring
class OptMoEModel(OptMoEPreTrainedModel):
    def __init__(self, config: OptMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        base_use_rope = bool(getattr(config, "use_rope", True))
        base_use_swa = int(getattr(config, "sliding_window_size", -1)) > 0
        self.use_rope_list = _parse_pattern(
            getattr(config, "rope_pattern", None),
            config.num_hidden_layers,
            true_char="R",
            false_char="N",
            default=base_use_rope,
        )
        self.use_swa_list = _parse_pattern(
            getattr(config, "swa_pattern", None),
            config.num_hidden_layers,
            true_char="S",
            false_char="F",
            default=base_use_swa,
        )
        sliding_window_size = getattr(config, "sliding_window_size", -1)
        config.layer_types = [
            "sliding_attention" if use_swa else "full_attention"
            for use_swa in self.use_swa_list
        ]
        config.sliding_window = (
            sliding_window_size
            if sliding_window_size > 0 and any(self.use_swa_list)
            else None
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                OptMoEDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = OptMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OptMoERotaryEmbedding(config=config)

        # Second rotary embedding for SWA layers — fully independent from the global one.
        # Mirrors native model's rope_of_swa: separate theta AND separate rope_parameters.
        rope_theta_swa = getattr(config, "rope_theta_swa", None)
        if rope_theta_swa is not None:
            from copy import deepcopy

            _swa_config = deepcopy(config)
            # Override both theta and scaling independently so the two RoPE
            # configurations are completely decoupled.
            _swa_config.rope_theta = rope_theta_swa
            _swa_config.rope_parameters = getattr(config, "rope_parameters_swa", None)
            self.rotary_emb_swa = OptMoERotaryEmbedding(config=_swa_config)
        else:
            self.rotary_emb_swa = None

        self.gradient_checkpointing = False

        for i, layer in enumerate(self.layers):
            layer.attention_type = config.layer_types[i]
            layer.self_attn.use_rope = self.use_rope_list[i]
            layer.sliding_window = sliding_window_size if self.use_swa_list[i] else -1
            layer.self_attn.sliding_window = (
                sliding_window_size if self.use_swa_list[i] else None
            )
            # SWA+RoPE layers use rotary_emb_swa when rope_theta_swa is configured
            layer.self_attn.use_swa_rope = (
                self.use_swa_list[i]
                and self.use_rope_list[i]
                and rope_theta_swa is not None
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = _build_dynamic_cache(self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.config.sliding_window is not None:
                causal_mask_mapping[
                    "sliding_attention"
                ] = create_sliding_window_causal_mask(**mask_kwargs)
            else:
                causal_mask_mapping["sliding_attention"] = causal_mask_mapping[
                    "full_attention"
                ]

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        # Compute SWA position embeddings (different theta) once if needed
        position_embeddings_swa = (
            self.rotary_emb_swa(hidden_states, position_ids=position_ids)
            if self.rotary_emb_swa is not None
            else None
        )

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            # SWA+RoPE layers use the SWA-specific rotary embedding when available
            pe = (
                position_embeddings_swa
                if position_embeddings_swa is not None
                and decoder_layer.self_attn.use_swa_rope
                else position_embeddings
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=pe,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class OptMoEForCausalLM(OptMoEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = OptMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, OptMoEForCausalLM

        >>> model = OptMoEForCausalLM.from_pretrained("path/to/model")
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/model")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if hasattr(self, "loss_function"):
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                    **kwargs,
                )
            else:
                loss = _causal_lm_loss(logits, labels, self.config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class OptMoEForSequenceClassification(
    GenericForSequenceClassification, OptMoEPreTrainedModel
):
    ...


class OptMoEForQuestionAnswering(GenericForQuestionAnswering, OptMoEPreTrainedModel):
    base_model_prefix = (
        "transformer"  # For BC, where `transformer` was used instead of `model`
    )


class OptMoEForTokenClassification(
    GenericForTokenClassification, OptMoEPreTrainedModel
):
    ...


__all__ = [
    "OptMoEForCausalLM",
    "OptMoEModel",
    "OptMoEPreTrainedModel",
    "OptMoEForSequenceClassification",
    "OptMoEForQuestionAnswering",
    "OptMoEForTokenClassification",
]
