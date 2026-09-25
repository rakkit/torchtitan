# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF configuration of torchtitan's opt_moe (`torchtitan/models/opt_moe`).

Written by `setup_hf.py` from the native config; every field is a value the
native model computes with, resolved at export (no native-only flags)."""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters

_LAYER_TYPES = ("full_attention", "sliding_attention")


@strict
class OptMoEConfig(PreTrainedConfig):
    r"""
    head_dim (`int`): Attention head width.
    qk_norm / v_norm (`bool`): Parameter-free RMSNorm on each q/k (v) head.
    attention_mid_norm (`bool`): Parameter-free RMSNorm on the attention output,
        per head if `head_wise_mid_norm`, else over all heads; applied before the
        gate if `mid_norm_position == "before"`.
    gated_attention (`str`, *optional*): `"head-wise"` or `"element-wise"`
        sigmoid gate on the attention output, from `gate_proj(hidden_states)`.
    rotary_dim (`int`): Leading head dims rotated by RoPE (partial RoPE).
    use_rope_layers (`list[bool]`): Per layer, whether it applies RoPE.
    layer_types (`list[str]`): Per layer, `"full_attention"` or `"sliding_attention"`.
    rope_parameters_sliding (`dict`, *optional*): RoPE of the sliding layers,
        when they use their own theta.
    mlp_mid_norm / moe_mid_norm (`bool`): RMSNorm on the SwiGLU output of the
        dense MLP / of every expert.
    first_k_dense_replace (`int`): The first k layers use a dense MLP.
    n_routed_experts, num_experts_per_tok, moe_intermediate_size, n_shared_experts:
        MoE shape; the shared expert is `n_shared_experts * moe_intermediate_size` wide.
    scoring_func (`str`): `"sigmoid"`, `"softmax"` or `"sqrtsoftplus"`.
    norm_topk_prob (`bool`): Renormalise the top-k scores to sum to 1.
    routed_scaling_factor (`float`): Multiplies the top-k scores.
    embeddings_norm (`bool`): RMSNorm right after the embedding.
    residual_identity_scale / residual_block_scale (`float`):
        `h = identity_scale * h + block_scale * block(h)` for every sub-block.
    """

    model_type = "OptMoE"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 201088
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-30
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    qk_norm: bool = True
    v_norm: bool = False
    attention_mid_norm: bool = False
    head_wise_mid_norm: bool = False
    mid_norm_position: str = "after"
    gated_attention: str | None = None
    rope_parameters: RopeParameters | dict | None = None
    rope_parameters_sliding: RopeParameters | dict | None = None
    rotary_dim: int | None = None
    use_rope_layers: list[bool] | None = None
    layer_types: list[str] | None = None
    sliding_window: int | None = None
    intermediate_size: int = 5120
    mlp_mid_norm: bool = False
    first_k_dense_replace: int = 0
    n_routed_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 640
    n_shared_experts: int = 1
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    moe_mid_norm: bool = False
    embeddings_norm: bool = False
    residual_identity_scale: float = 1.0
    residual_block_scale: float = 1.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        n = self.num_hidden_layers
        if self.rotary_dim is None:
            self.rotary_dim = self.head_dim
        if self.use_rope_layers is None:
            self.use_rope_layers = [True] * n
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * n
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        n = self.num_hidden_layers
        if len(self.use_rope_layers) != n or len(self.layer_types) != n:
            raise ValueError("use_rope_layers and layer_types need one entry per layer")
        if any(t not in _LAYER_TYPES for t in self.layer_types):
            raise ValueError(f"layer_types entries must be in {_LAYER_TYPES}")
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError("sliding layers need a sliding_window")
        if not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2:
            raise ValueError(f"rotary_dim {self.rotary_dim} must be even and in (0, head_dim]")
        if self.gated_attention not in (None, "head-wise", "element-wise"):
            raise ValueError(f"unknown gated_attention {self.gated_attention!r}")
        if self.mid_norm_position not in ("after", "before"):
            raise ValueError(f"unknown mid_norm_position {self.mid_norm_position!r}")
        if self.scoring_func not in ("sigmoid", "softmax", "sqrtsoftplus"):
            raise ValueError(f"unknown scoring_func {self.scoring_func!r}")
        if self.hidden_act != "silu":
            raise ValueError("opt_moe's MLPs are SwiGLU: hidden_act must be 'silu'")


__all__ = ["OptMoEConfig"]
