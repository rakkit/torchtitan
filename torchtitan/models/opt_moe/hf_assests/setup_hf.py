# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil

from ..utils.moe_utils import calc_gate_scaling_factor


def _resolve_model_config(model, model_config=None):
    materialized_config = getattr(model, "config", None)
    if materialized_config is not None:
        return materialized_config
    if model_config is not None:
        return model_config
    raise ValueError(
        "opt_moe HF asset export requires a materialized model.config or an "
        "explicit model_config."
    )


def _normalize_layer_pattern_for_export(pattern):
    """Normalize native per-layer patterns to HF-friendly JSON forms.

    Native accepts list-wrapped pattern strings such as ``['SSSF']``; HF/vLLM
    parsers expect either a plain pattern string or list[bool].
    """
    if pattern is None:
        return None
    if isinstance(pattern, tuple):
        pattern = list(pattern)
    if isinstance(pattern, list):
        if len(pattern) == 1 and isinstance(pattern[0], str):
            return pattern[0]
        if pattern and all(isinstance(x, str) and len(x) == 1 for x in pattern):
            return "".join(pattern)
    return pattern


def _native_to_hf_rope_parameters(rope_cfg, *, partial_rotary_factor: float):
    """Convert a native RoPE.Config to current HF rope_parameters format."""
    if rope_cfg is None:
        return None
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": rope_cfg.theta,
        "partial_rotary_factor": partial_rotary_factor,
    }
    if rope_cfg.scaling == "none":
        return rope_parameters
    if rope_cfg.scaling == "llama":
        rope_parameters.update(
            {
                "rope_type": "llama3",
                "factor": rope_cfg.scaling_factor,
                "low_freq_factor": rope_cfg.low_freq_factor,
                "high_freq_factor": rope_cfg.high_freq_factor,
                "original_max_position_embeddings": rope_cfg.original_max_position_embeddings,
            }
        )
        return rope_parameters
    if rope_cfg.scaling == "yarn":
        rope_parameters.update(
            {
                "rope_type": "yarn",
                "factor": rope_cfg.rope_factor,
                "original_max_position_embeddings": rope_cfg.original_seq_len,
            }
        )
        return rope_parameters
    return rope_parameters


def _head_dim_from_model_config(model_config) -> int:
    attn_cfg = model_config.layer.attention
    return getattr(attn_cfg, "head_dim", None) or (model_config.dim // attn_cfg.n_heads)


def _num_key_value_heads_from_model_config(model_config) -> int:
    attn_cfg = model_config.layer.attention
    return attn_cfg.n_kv_heads if attn_cfg.n_kv_heads is not None else attn_cfg.n_heads


def _qk_rope_dim_from_model_config(model_config, *, head_dim: int) -> int:
    attn_cfg = model_config.layer.attention
    return getattr(attn_cfg, "qk_rope_dim", None) or head_dim


def _hidden_act_from_layer_configs(
    feed_forward_cfg,
    moe_cfg,
    *,
    default_hidden_act: str | None = None,
) -> str | None:
    activation_types = {
        cfg.activation_type
        for cfg in (feed_forward_cfg, moe_cfg)
        if cfg is not None and getattr(cfg, "activation_type", None) is not None
    }
    if len(activation_types) > 1:
        raise ValueError(
            "opt_moe HF export only supports a single hidden_act across dense and "
            f"MoE blocks, got {sorted(activation_types)!r}."
        )
    if activation_types:
        return activation_types.pop()
    return default_hidden_act


def _moe_scaling_factor_from_config(moe_cfg) -> float | None:
    if moe_cfg is None:
        return None
    if moe_cfg.scaling_factor is not None:
        return moe_cfg.scaling_factor
    return calc_gate_scaling_factor(
        moe_cfg.num_experts,
        moe_cfg.top_k,
    )


def get_hf_config_overrides_from_model_config(model, model_config=None) -> dict:
    """Build the HF config values derived from the runtime opt_moe config."""
    model_config = _resolve_model_config(model, model_config)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config.json")) as f:
        default_config = json.load(f)

    attn_cfg = model_config.layer.attention
    feed_forward_cfg = getattr(model_config.layer, "feed_forward", None)
    moe_cfg = getattr(model_config.layer, "moe", None)
    head_dim = _head_dim_from_model_config(model_config)
    qk_rope_dim = _qk_rope_dim_from_model_config(model_config, head_dim=head_dim)
    ffn_norm_everywhere = (
        bool(feed_forward_cfg.norm_everywhere)
        if feed_forward_cfg is not None
        else False
    )

    overrides = {
        "num_hidden_layers": model_config.n_layers,
        "vocab_size": model_config.vocab_size,
        "rms_norm_eps": model_config.norm_eps,
        "qk_norm": attn_cfg.qk_norm,
        "mid_norm": bool(getattr(attn_cfg, "mid_norm", False)),
        "norm_everywhere": attn_cfg.norm_everywhere,
        "attention_norm_everywhere": attn_cfg.norm_everywhere,
        "ffn_norm_everywhere": ffn_norm_everywhere,
        "moe_norm_everywhere": (
            bool(moe_cfg.norm_everywhere)
            if moe_cfg is not None
            else ffn_norm_everywhere
        ),
        # HF router always runs its matmul in fp32 for deterministic routing behavior.
        "max_position_embeddings": model_config.rope.max_context_length,
        "num_attention_heads": attn_cfg.n_heads,
        "num_key_value_heads": _num_key_value_heads_from_model_config(model_config),
        "head_dim": head_dim,
        "rope_theta": model_config.rope.theta,
        "hidden_size": model_config.dim,
        "n_dense_layers": model_config.layer.n_dense_layers,
        "gated_attention_type": getattr(attn_cfg, "gated_attention_type", None),
        "gate_only": bool(getattr(attn_cfg, "gate_only", False)),
        "mid_norm_position": getattr(attn_cfg, "mid_norm_position", "after"),
        "head_wise_mid_norm": bool(getattr(attn_cfg, "head_wise_mid_norm", False)),
        "use_rope": bool(getattr(attn_cfg, "use_rope", True)),
        "sliding_window_size": getattr(attn_cfg, "sliding_window_size", -1),
        "qk_rope_dim": qk_rope_dim,
        "partial_rotary_factor": qk_rope_dim / head_dim,
        "residual_scale": getattr(model_config.layer, "residual_scale", "identity"),
        "rope_pattern": _normalize_layer_pattern_for_export(
            getattr(model_config, "rope_pattern", None)
        ),
        "swa_pattern": _normalize_layer_pattern_for_export(
            getattr(model_config, "swa_pattern", None)
        ),
    }

    hidden_act = _hidden_act_from_layer_configs(
        feed_forward_cfg,
        moe_cfg,
        default_hidden_act=default_config.get("hidden_act"),
    )
    if hidden_act is not None:
        overrides["hidden_act"] = hidden_act

    overrides["rope_parameters"] = _native_to_hf_rope_parameters(
        model_config.rope,
        partial_rotary_factor=overrides["partial_rotary_factor"],
    )

    rope_of_swa = getattr(model_config, "rope_of_swa", None)
    overrides["rope_theta_swa"] = (
        float(rope_of_swa.theta) if rope_of_swa is not None else None
    )
    overrides["rope_parameters_swa"] = _native_to_hf_rope_parameters(
        rope_of_swa,
        partial_rotary_factor=overrides["partial_rotary_factor"],
    )

    if feed_forward_cfg is not None:
        overrides["intermediate_size"] = feed_forward_cfg.hidden_dim

    if moe_cfg is not None:
        overrides["moe_intermediate_size"] = moe_cfg.hidden_dim
        overrides["n_active_experts"] = moe_cfg.top_k
        overrides["n_total_experts"] = moe_cfg.num_experts
        overrides["moe_scaling_factor"] = _moe_scaling_factor_from_config(moe_cfg)
        overrides["n_shared_experts"] = moe_cfg.num_shared_experts

    return overrides


def copy_and_overwrite_model_config(model, model_config, dst_path: str):
    """Copy HF config/modeling files to dst_path and overwrite config.json
    with parameters extracted from the trained model and model_config.

    Args:
        model: The trained OPTMoEModel instance (used to inspect actual sizes).
        model_config: OPTMoEModel.Config instance with training hyperparameters.
        dst_path: Destination directory for HF checkpoint.
    """
    model_config = _resolve_model_config(model, model_config)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_config_path = os.path.join(current_dir, "configuration_opt_moe.py")
    src_modeling_path = os.path.join(current_dir, "modeling_opt_moe.py")

    dst_config_py_path = os.path.join(dst_path, "configuration_opt_moe.py")
    dst_modeling_py_path = os.path.join(dst_path, "modeling_opt_moe.py")
    dst_config_json_path = os.path.join(dst_path, "config.json")

    dummy_chat_template_path = os.path.join(current_dir, "chat_template.jinja")

    new_config = overwrite_config(model, model_config)

    if os.path.exists(dst_path):
        shutil.copy(src_config_path, dst_config_py_path)
        shutil.copy(src_modeling_path, dst_modeling_py_path)
        with open(dst_config_json_path, "w") as f:
            json.dump(new_config, f, indent=4)
        if os.path.exists(dummy_chat_template_path):
            shutil.copy(
                dummy_chat_template_path, os.path.join(dst_path, "chat_template.jinja")
            )


def overwrite_config(model, model_config=None):
    """Build a config.json dict from the model and model_config.

    Args:
        model: The trained OPTMoEModel instance.
        model_config: OPTMoEModel.Config instance.

    Returns:
        dict suitable for writing as config.json.
    """
    model_config = _resolve_model_config(model, model_config)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config.json")) as f:
        default_config = json.load(f)
    default_config.update(
        get_hf_config_overrides_from_model_config(model, model_config)
    )
    return default_config
