# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The HF export of an opt_moe model: `hf_config` resolves the native config
into `OptMoEConfig` (what the model computes, not how it is authored), and
`copy_and_overwrite_model_config` writes it next to the modeling code."""

import os
import shutil

from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.rope import CosSinRoPE

from ..norm_moe import NormRouter, saint_check_scaling_factor
from ..utils.inits import setup_residual_scale
from .configuration_opt_moe import OptMoEConfig

_FILES = ("configuration_opt_moe.py", "modeling_opt_moe.py", "chat_template.jinja")


def _swiglu(activation_fn) -> None:
    if isinstance(activation_fn, SwiGLU.Config):
        return
    # The override's config is the same function.
    from torchtitan.overrides.fused_swiglu import FusedSwiGLU

    if not isinstance(activation_fn, FusedSwiGLU.Config):
        raise ValueError(
            f"HF export supports SwiGLU MLPs only, got {type(activation_fn).__qualname__}"
        )


def _rms_eps(model_config) -> float:
    """The one eps of every norm; each must be the parameter-free RMSNorm."""
    layer = model_config.layer
    owners = [model_config, layer, layer.attention, layer.feed_forward, layer.moe]
    kinds = {(o.norm_type, o.norm_eps) for o in owners if o is not None}
    if len(kinds) != 1 or next(iter(kinds))[0] != "np_rmsnorm":
        raise ValueError(
            f"HF export needs one parameter-free RMSNorm everywhere, got {sorted(kinds)}"
        )
    return next(iter(kinds))[1]


def _rope_parameters(rope) -> dict:
    if not isinstance(rope, CosSinRoPE.Config) or rope.scaling != "none":
        raise ValueError(f"HF export supports unscaled CosSinRoPE only, got {rope}")
    return {"rope_type": "default", "rope_theta": float(rope.theta)}


def _route_scale(model, moe) -> float:
    """What the routers multiply by. Taken from a built router when there is
    one: without a table entry the native value is a random estimate."""
    for module in model.modules() if hasattr(model, "modules") else ():
        if isinstance(module, NormRouter):
            return float(module.route_scale)
    return saint_check_scaling_factor(moe.num_experts, moe.top_k, moe.scaling_factor)


def hf_config(model, model_config=None) -> OptMoEConfig:
    """`OptMoEConfig` of an expanded native config (`model.config`, or
    `model_config` for a model that carries none)."""
    m = getattr(model, "config", None) or model_config
    if not getattr(m, "layers", None):
        raise ValueError("HF export needs the expanded config (update_from_config)")
    layer, attn, ffn, moe = m.layer, m.layer.attention, m.layer.feed_forward, m.layer.moe
    head_dim = attn.head_dim or m.dim // attn.n_heads
    per_layer = [lc.attention for lc in m.layers]
    sliding = [a.sliding_window_size > 0 for a in per_layer]
    rotary_dim = attn.qk_rope_dim or head_dim
    rope = _rope_parameters(m.rope)
    sliding_ropes = {
        a.rope.theta for a, s in zip(per_layer, sliding) if s and a.use_rope
    }
    if len(sliding_ropes) > 1:
        raise ValueError(f"sliding layers disagree on their RoPE: {sorted(sliding_ropes)}")
    rope_sliding = None
    if sliding_ropes and sliding_ropes != {rope["rope_theta"]}:
        rope_sliding = {"rope_type": "default", "rope_theta": float(sliding_ropes.pop())}
    block_scale, identity_scale = setup_residual_scale(layer.residual_scale, m.n_layers)
    fields = dict(
        vocab_size=m.vocab_size,
        hidden_size=m.dim,
        num_hidden_layers=m.n_layers,
        num_attention_heads=attn.n_heads,
        num_key_value_heads=attn.n_kv_heads or attn.n_heads,
        head_dim=head_dim,
        max_position_embeddings=m.rope.max_context_length,
        rms_norm_eps=_rms_eps(m),
        qk_norm=attn.qk_norm or attn.norm_everywhere,
        v_norm=attn.v_norm or attn.norm_everywhere,
        attention_mid_norm=(attn.mid_norm or attn.norm_everywhere) and not attn.gate_only,
        head_wise_mid_norm=attn.head_wise_mid_norm,
        mid_norm_position=attn.mid_norm_position,
        gated_attention=(
            None if attn.gated_attention_type in (None, "none") else attn.gated_attention_type
        ),
        rope_parameters=rope,
        rope_parameters_sliding=rope_sliding,
        rotary_dim=rotary_dim,
        use_rope_layers=[a.use_rope for a in per_layer],
        layer_types=["sliding_attention" if s else "full_attention" for s in sliding],
        sliding_window=attn.sliding_window_size if any(sliding) else None,
        embeddings_norm=m.use_embeddings_norm,
        residual_identity_scale=identity_scale,
        residual_block_scale=block_scale,
        dtype="bfloat16",
    )
    if ffn is not None:
        _swiglu(ffn.activation_fn)
        fields.update(intermediate_size=ffn.hidden_dim, mlp_mid_norm=ffn.norm_everywhere)
    if moe is None:
        fields.update(first_k_dense_replace=m.n_layers)
    else:
        _swiglu(moe.activation_fn)
        fields.update(
            first_k_dense_replace=layer.n_dense_layers,
            n_routed_experts=moe.num_experts,
            num_experts_per_tok=moe.top_k,
            moe_intermediate_size=moe.hidden_dim,
            n_shared_experts=moe.num_shared_experts,
            scoring_func=moe.score_func,
            norm_topk_prob=True,  # NormRouter's route_norm; the MoE config does not expose it
            routed_scaling_factor=_route_scale(model, moe),
            moe_mid_norm=moe.norm_everywhere,
        )
    config = OptMoEConfig(**fields)
    config.architectures = ["OptMoEForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_opt_moe.OptMoEConfig",
        "AutoModel": "modeling_opt_moe.OptMoEModel",
        "AutoModelForCausalLM": "modeling_opt_moe.OptMoEForCausalLM",
    }
    return config


def copy_and_overwrite_model_config(model, model_config, dst_path: str) -> None:
    """Write `config.json` and the modeling code of `model` into `dst_path`."""
    hf_config(model, model_config).save_pretrained(dst_path)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _FILES:
        shutil.copy(os.path.join(here, name), os.path.join(dst_path, name))
