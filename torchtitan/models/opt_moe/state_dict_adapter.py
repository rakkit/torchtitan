# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native opt_moe <-> HF: the state dicts and the HF config.

`wqkv` and the dense / shared `w13` are split into HF's projections; routed
experts stay 3D (`w13` `[E, 2, F, D]` <-> `gate_up_proj` `[E, 2F, D]`). The MTP
depth `mtp_layers.i` is HF layer `L + i` (L decoder layers) with DeepSeek-V3's
names for its fusion (`enorm`, `hnorm`, `eh_proj`, `shared_head.norm`); the
embedding and the head are shared, not stored again. `hf_config` writes the
HF config of a model config (`hf/configuration_opt_moe.py`)."""

import re
from typing import Any

from torchtitan.models.base.state_dict_adapter import StateDictAdapter
from torchtitan.models.common.activation import Sigmoid, Softmax, SqrtSoftplus
from torchtitan.models.common.moe import MoE

# Native key (fused projections split) -> HF key; `{}` is the layer index.
_TO_HF = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    "layers.{}.attention.qkv_linear.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.qkv_linear.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.qkv_linear.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention.q_norm.weight": "model.layers.{}.self_attn.q_norm.weight",
    "layers.{}.attention.k_norm.weight": "model.layers.{}.self_attn.k_norm.weight",
    "layers.{}.attention.gate_proj.weight": "model.layers.{}.self_attn.gate_proj.weight",
    "layers.{}.attention.k_conv.weight": "model.layers.{}.self_attn.k_conv.weight",
    "layers.{}.attention.v_conv.weight": "model.layers.{}.self_attn.v_conv.weight",
    "layers.{}.attention.sinks": "model.layers.{}.self_attn.sinks",
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "layers.{}.feed_forward.router.gate.weight": "model.layers.{}.mlp.gate.weight",
    "layers.{}.feed_forward.expert_bias_E": "model.layers.{}.mlp.gate.e_score_correction_bias",
    "layers.{}.feed_forward.routed_experts.w13.weight": "model.layers.{}.mlp.experts.gate_up_proj",
    "layers.{}.feed_forward.routed_experts.w2.weight": "model.layers.{}.mlp.experts.down_proj",
    "layers.{}.feed_forward.shared_experts.w1.weight": "model.layers.{}.mlp.shared_experts.gate_proj.weight",
    "layers.{}.feed_forward.shared_experts.w3.weight": "model.layers.{}.mlp.shared_experts.up_proj.weight",
    "layers.{}.feed_forward.shared_experts.w2.weight": "model.layers.{}.mlp.shared_experts.down_proj.weight",
    # the MTP depth's fusion (DeepSeek-V3's names)
    "layers.{}.enorm.weight": "model.layers.{}.enorm.weight",
    "layers.{}.hnorm.weight": "model.layers.{}.hnorm.weight",
    "layers.{}.eh_proj.weight": "model.layers.{}.eh_proj.weight",
    "layers.{}.mtp_norm.weight": "model.layers.{}.shared_head.norm.weight",
}
_FROM_HF = {v: k for k, v in _TO_HF.items()}
_ROUTED_W13 = "feed_forward.routed_experts.w13.weight"
_SCORE_FUNCS = {
    Sigmoid.Config: "sigmoid",
    Softmax.Config: "softmax",
    SqrtSoftplus.Config: "sqrtsoftplus",
}


def _rename(state_dict: dict[str, Any], table: dict[str, str]) -> dict[str, Any]:
    out, unmapped = {}, []
    for key, value in state_dict.items():
        m = re.match(r"(model\.)?layers\.(\d+)\.", key)
        abstract = key.replace(f"layers.{m.group(2)}.", "layers.{}.", 1) if m else key
        target = table.get(abstract)
        if target is None:
            unmapped.append(key)
            continue
        out[target.format(m.group(2)) if m else target] = value
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} tensor(s) have no opt_moe <-> HF mapping: "
            f"{sorted(unmapped)[:8]}"
        )
    return out


def _renumber(state_dict: dict[str, Any], pattern: str, repl) -> dict[str, Any]:
    return {re.sub(pattern, repl, k, count=1): v for k, v in state_dict.items()}


def hf_config(model_config: Any) -> dict[str, Any]:
    """The HF config (``hf/configuration_opt_moe.py`` ``OptMoeConfig`` fields)
    of a complete opt_moe model config: every architecture setting, per layer
    where the model chooses per layer."""
    layers = model_config.layers
    attn = layers[0].attention
    qkv = attn.qkv_linear
    for layer in layers:
        a = layer.attention
        if (a.n_heads, a.n_kv_heads, a.head_dim) != (
            attn.n_heads,
            attn.n_kv_heads,
            attn.head_dim,
        ):
            raise ValueError(
                "hf_config: every layer must have the same attention shape"
            )
    windows = [
        getattr(l.attention.inner_attention, "sliding_window_size", None)
        for l in layers
    ]
    window = {w for w in windows if w is not None}
    if len(window) > 1:
        raise ValueError(
            f"hf_config: one sliding window for every SWA layer, got {window}"
        )
    moe = [l.feed_forward for l in layers if isinstance(l.feed_forward, MoE.Config)]
    dense = [
        l.feed_forward for l in layers if not isinstance(l.feed_forward, MoE.Config)
    ]
    n_dense = len(dense)
    if any(isinstance(l.feed_forward, MoE.Config) for l in layers[:n_dense]):
        raise ValueError("hf_config: the dense layers must come first")
    m = moe[0] if moe else None

    def rope(layer) -> dict[str, Any] | None:
        r = layer.attention.rope
        if r is None:
            return None
        if r.scaling != "none":
            raise NotImplementedError(f"hf_config: RoPE scaling {r.scaling!r}")
        return {"dim": r.dim, "theta": r.theta}

    conv = attn.k_conv
    return {
        "architectures": ["OptMoeForCausalLM"],
        "model_type": "opt_moe",
        "auto_map": {
            "AutoConfig": "configuration_opt_moe.OptMoeConfig",
            "AutoModelForCausalLM": "modeling_opt_moe.OptMoeForCausalLM",
        },
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.dim,
        "num_hidden_layers": len(layers),
        "max_position_embeddings": model_config.max_context_length,
        "tie_word_embeddings": model_config.enable_weight_tying,
        "rms_norm_eps": model_config.norm.eps,
        "norm_has_weight": model_config.norm.elementwise_affine,
        "num_attention_heads": attn.n_heads,
        "num_key_value_heads": attn.n_kv_heads,
        "head_dim": qkv.head_dim,
        "qk_norm": attn.qk_norm is not None,
        # transformers' cache layer types: a conv layer is "hybrid" (K / V and
        # conv states in one cache layer)
        "layer_types": [
            ("hybrid" if conv is not None else "full_attention")
            if w is None
            else ("hybrid_sliding" if conv is not None else "sliding_attention")
            for w in windows
        ],
        "sliding_window": next(iter(window), None),
        "rope_layers": [rope(l) for l in layers],
        "attention_gate": "element-wise" if attn.gate_proj is not None else None,
        "attention_sinks": attn.attention_sinks,
        "conv_kernel_size": 0 if conv is None else conv.kernel_size,
        "first_k_dense_replace": n_dense,
        "intermediate_size": dense[0].w2.in_features if dense else None,
        "moe_intermediate_size": None if m is None else m.routed_experts.w2.in_features,
        "n_routed_experts": None if m is None else m.num_experts,
        "shared_expert_intermediate_size": (
            None
            if m is None or m.shared_experts is None
            else m.shared_experts.w2.in_features
        ),
        "num_experts_per_tok": None if m is None else m.router.top_k,
        "scoring_func": None if m is None else _SCORE_FUNCS[type(m.router.score_func)],
        "norm_topk_prob": None if m is None else m.router.route_norm,
        "norm_topk_epsilon": None if m is None else m.router.route_norm_epsilon,
        "routed_scaling_factor": None if m is None else m.router.route_scale,
        "router_bias": None if m is None else m.load_balance_coeff is not None,
        "residual_block_scales": [l.block_scale for l in layers],
        "residual_identity_scales": [l.identity_scale for l in layers],
        "num_nextn_predict_layers": len(model_config.mtp_layers),
        "hidden_act": "silu",
    }


class OPTMoEStateDictAdapter(StateDictAdapter):
    # HF tensors the model computes with in fp32 whatever the export dtype.
    keep_in_fp32 = ("e_score_correction_bias",)

    def _qkv_args(self) -> dict[str, int]:
        attn = self.model_config.layers[0].attention
        n_kv = attn.n_kv_heads or attn.n_heads
        return {
            "head_dim": attn.head_dim or attn.dim // attn.n_heads,
            "heads_per_kv": attn.n_heads // n_kv,
        }

    @staticmethod
    def _prefixes(keys, suffix: str) -> list[str]:
        return sorted({k[: -len(suffix)] for k in keys if k.endswith(suffix)})

    def _unfuse(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """The fused projections as the tensors the HF model holds."""
        sd = dict(state_dict)
        for prefix in self._prefixes(sd, "wqkv.weight"):
            self._split_qkv_linear(sd, prefix=prefix, **self._qkv_args())
        for key in [k for k in sd if k.endswith(_ROUTED_W13)]:
            sd[key] = sd[key].flatten(1, 2)
        for prefix in self._prefixes(sd, "w13.weight"):
            if not prefix.endswith("routed_experts."):
                self._split_stacked_linear(
                    sd,
                    fused_key=f"{prefix}w13.weight",
                    logical_keys=(f"{prefix}w1.weight", f"{prefix}w3.weight"),
                    dim=0,
                )
        return sd

    def _fuse(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """The inverse of `_unfuse`."""
        sd = dict(state_dict)
        for prefix in self._prefixes(sd, "wq.weight"):
            self._merge_qkv_linear(sd, prefix=prefix, **self._qkv_args())
        for prefix in self._prefixes(sd, "w1.weight"):
            self._stack_logical_linears(
                sd,
                fused_key=f"{prefix}w13.weight",
                logical_keys=(f"{prefix}w1.weight", f"{prefix}w3.weight"),
                dim=0,
            )
        for key in [k for k in sd if k.endswith(_ROUTED_W13)]:
            sd[key] = sd[key].unflatten(1, (2, -1))
        return sd

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        n = len(self.model_config.layers)
        sd = _renumber(
            state_dict,
            r"^mtp_layers\.(\d+)\.",
            lambda m: f"layers.{n + int(m.group(1))}.",
        )
        return _rename(self._unfuse(sd), _TO_HF)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        n = len(self.model_config.layers)
        sd = self._fuse(_rename(hf_state_dict, _FROM_HF))
        return _renumber(
            sd,
            r"^layers\.(\d+)\.",
            lambda m: (
                m.group(0)
                if int(m.group(1)) < n
                else f"mtp_layers.{int(m.group(1)) - n}."
            ),
        )
