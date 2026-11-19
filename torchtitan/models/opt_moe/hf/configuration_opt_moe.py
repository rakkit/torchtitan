# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The HF config of opt_moe (shipped with an HF checkpoint, loaded with
``trust_remote_code``; no torchtitan import). Written from a torchtitan model
config by ``torchtitan.models.opt_moe.state_dict_adapter.hf_config``."""

from typing import Any

from transformers import PretrainedConfig


_LAYER_TYPES = {"full_attention", "sliding_attention"}
_CONV_LAYER_TYPES = {"hybrid", "hybrid_sliding"}


class OptMoeConfig(PretrainedConfig):
    """Every architecture setting of an opt_moe model; per layer where the
    model chooses per layer.

    Attention: GQA (``num_attention_heads`` / ``num_key_value_heads`` x
    ``head_dim``) with qk-norm (``qk_norm``); ``layer_types``, transformers'
    cache layer types: ``sliding_attention`` (with ``sliding_window``) /
    ``full_attention``, and with a conv ``hybrid_sliding`` / ``hybrid`` (the
    cache keeps the K / V and the conv inputs' last ``conv_kernel_size`` taps,
    ``number_of_conv_states`` of them: one per conv stream);
    ``rope_layers`` (``{"dim", "theta"}``, None: NoPE; ``dim`` < ``head_dim``:
    partial RoPE); ``attention_gate`` (None | ``element-wise``:
    ``sigmoid(gate_proj(x))`` on the output); ``attention_sinks`` (one logit
    per head in the softmax; not with the gate); ``conv_kernel_size`` (the
    Canon short conv ``x + conv(x)`` on k and v, 0: none).

    Feed-forward: the first ``first_k_dense_replace`` layers a SwiGLU FFN of
    ``intermediate_size``, the rest MoE: ``n_routed_experts`` of
    ``moe_intermediate_size``, a shared expert of
    ``shared_expert_intermediate_size``, top ``num_experts_per_tok`` by
    ``scoring_func`` (+ ``e_score_correction_bias`` for the choice when
    ``router_bias``), weights renormalized (``norm_topk_prob``,
    ``norm_topk_epsilon``) and scaled by ``routed_scaling_factor``.

    Blocks: ``x = identity_scale * x + block_scale * f(x)`` per layer
    (``residual_identity_scales`` / ``residual_block_scales``); RMSNorm
    (``rms_norm_eps``, gains when ``norm_has_weight``).
    ``num_nextn_predict_layers``: the MTP depths (DeepSeek-V3's MTP, HF layers
    after the decoder layers)."""

    model_type = "opt_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 4096,
        tie_word_embeddings: bool = False,
        rms_norm_eps: float = 1e-30,
        norm_has_weight: bool = True,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        qk_norm: bool = True,
        layer_types: list[str] | None = None,
        sliding_window: int | None = None,
        rope_layers: list[dict[str, Any] | None] | None = None,
        attention_gate: str | None = None,
        attention_sinks: bool = False,
        conv_kernel_size: int = 0,
        first_k_dense_replace: int = 0,
        intermediate_size: int | None = None,
        moe_intermediate_size: int | None = None,
        n_routed_experts: int | None = None,
        shared_expert_intermediate_size: int | None = None,
        num_experts_per_tok: int | None = None,
        scoring_func: str | None = None,
        norm_topk_prob: bool | None = None,
        norm_topk_epsilon: float | None = None,
        routed_scaling_factor: float | None = None,
        router_bias: bool | None = None,
        residual_block_scales: list[float] | None = None,
        residual_identity_scales: list[float] | None = None,
        num_nextn_predict_layers: int = 0,
        hidden_act: str = "silu",
        use_cache: bool = True,
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.norm_has_weight = norm_has_weight
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        self.sliding_window = sliding_window
        self.rope_layers = (
            rope_layers or [{"dim": head_dim, "theta": 10000.0}] * num_hidden_layers
        )
        self.attention_gate = attention_gate
        self.attention_sinks = attention_sinks
        self.conv_kernel_size = conv_kernel_size
        self.first_k_dense_replace = first_k_dense_replace
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.norm_topk_epsilon = norm_topk_epsilon
        self.routed_scaling_factor = routed_scaling_factor
        self.router_bias = router_bias
        self.residual_block_scales = residual_block_scales or [1.0] * num_hidden_layers
        self.residual_identity_scales = (
            residual_identity_scales or [1.0] * num_hidden_layers
        )
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        allowed = _CONV_LAYER_TYPES if conv_kernel_size else _LAYER_TYPES
        if not set(self.layer_types) <= allowed:
            raise ValueError(
                f"layer_types {sorted(set(self.layer_types))}: with conv_kernel_size "
                f"{conv_kernel_size} each layer is one of {sorted(allowed)}"
            )
        # what the HF and vLLM models implement; anything else would be ignored
        if attention_gate not in (None, "element-wise"):
            raise ValueError(
                f"attention_gate {attention_gate!r}: None or 'element-wise'"
            )
        if attention_gate is not None and attention_sinks:
            raise ValueError("attention_gate and attention_sinks: one or neither")
        if sliding_window is None and any(
            t in ("sliding_attention", "hybrid_sliding") for t in self.layer_types
        ):
            raise ValueError("sliding-window layers need sliding_window")
        if hidden_act != "silu":
            raise ValueError(f"hidden_act {hidden_act!r}: the FFNs are SwiGLU ('silu')")
        for name in (
            "layer_types",
            "rope_layers",
            "residual_block_scales",
            "residual_identity_scales",
        ):
            if len(getattr(self, name)) != num_hidden_layers:
                raise ValueError(
                    f"{name} needs one entry per layer ({num_hidden_layers})"
                )
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def number_of_conv_states(self) -> int:
        """The conv states a ``hybrid`` cache layer keeps (transformers'
        ``DynamicCache`` reads it): k and v."""
        return 2


__all__ = ["OptMoeConfig"]
