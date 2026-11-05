# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from torchtitan.models.common.rope import CosSinRoPE

from .gated_norm_swattention import AttentionSpec

from .model import OPTMoEModel, OPTMoETransformerBlock
from .norm_ffn import FeedForward
from .norm_moe import MoE


__all__ = ["build_model_config", "model_registry", "OPTMoEModel", "moe_opt_moe_configs"]


moe_opt_moe_configs = {
    "mis-8b": OPTMoEModel.Config(
        n_layers=24,
        dim=2048,
        swa_pattern="SSSF" * 6,
        rope_pattern="RRRN" * 6,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=0,
            feed_forward=FeedForward.Config(
                hidden_dim=5120,
            ),
            moe=MoE.Config(
                hidden_dim=640,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,  # the MoE recipe (no output factor)
            ),
            attention=AttentionSpec(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
                sliding_window_size=128,
                attn_backend="flex",
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
    "mis-8b-proxy": OPTMoEModel.Config(
        n_layers=24,
        dim=512,
        swa_pattern="SSSF" * 6,
        rope_pattern="RRRN" * 6,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=0,
            feed_forward=FeedForward.Config(
                hidden_dim=1280,
            ),
            moe=MoE.Config(
                hidden_dim=160,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,  # the MoE recipe (no output factor)
            ),
            attention=AttentionSpec(
                n_heads=8,
                n_kv_heads=1,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
                sliding_window_size=128,
                attn_backend="flex",
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
    "mis-8b-proxy-qbalance": OPTMoEModel.Config(
        n_layers=24,
        dim=512,
        swa_pattern="SSSF" * 6,
        rope_pattern="RRRN" * 6,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=0,
            feed_forward=FeedForward.Config(
                hidden_dim=1280,
            ),
            moe=MoE.Config(
                hidden_dim=160,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,  # the MoE recipe (no output factor)
                # mis-8b-proxy, but with the expert bias solved from the
                # required-bias histogram instead of the DeepSeek sign nudge.
                quantile_balance_bins=1000,
            ),
            attention=AttentionSpec(
                n_heads=8,
                n_kv_heads=1,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
                sliding_window_size=128,
                attn_backend="flex",
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
    "mis-30b-a3b": OPTMoEModel.Config(
        n_layers=40,
        dim=2048,
        swa_pattern="SSSF" * 10,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=1,
            feed_forward=FeedForward.Config(
                hidden_dim=8192,
            ),
            moe=MoE.Config(
                hidden_dim=512,
                num_experts=256,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,  # the MoE recipe (no output factor)
            ),
            attention=AttentionSpec(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
                sliding_window_size=128,
                attn_backend="flex",
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
    "mis-100b-a5b": OPTMoEModel.Config(
        n_layers=40,
        dim=3072,
        swa_pattern="SSSF" * 10,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=1,
            feed_forward=FeedForward.Config(
                hidden_dim=8192,
            ),
            moe=MoE.Config(
                hidden_dim=1024,
                num_experts=256,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,  # the MoE recipe (no output factor)
            ),
            attention=AttentionSpec(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
                sliding_window_size=128,
                attn_backend="flex",
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
    "gauge-d8-w256": OPTMoEModel.Config(
        n_layers=8,
        dim=256,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=8,
            feed_forward=FeedForward.Config(
                hidden_dim=1024,
            ),
            attention=AttentionSpec(
                n_heads=4,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
    "gauge-d16-w256": OPTMoEModel.Config(
        n_layers=16,
        dim=256,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=16,
            feed_forward=FeedForward.Config(
                hidden_dim=1024,
            ),
            attention=AttentionSpec(
                n_heads=4,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
    "gauge-d16-w512": OPTMoEModel.Config(
        n_layers=16,
        dim=512,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=16,
            feed_forward=FeedForward.Config(
                hidden_dim=2048,
            ),
            attention=AttentionSpec(
                n_heads=8,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
    "gauge-d16-w1024": OPTMoEModel.Config(
        n_layers=16,
        dim=1024,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=16,
            feed_forward=FeedForward.Config(
                hidden_dim=4096,
            ),
            attention=AttentionSpec(
                n_heads=16,
                n_kv_heads=16,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-30,
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=128,
            max_context_length=4096,
            theta=10000.0,
        ),
    ),
}


def model_registry(flavor: str) -> OPTMoEModel.Config:
    model_config = copy.deepcopy(moe_opt_moe_configs[flavor])
    model_config.flavor = flavor
    model_config.max_context_length = model_config.rope.max_context_length
    return model_config


def build_model_config(
    spec: "OPTMoEModel.Config | str",
    *,
    parallelism,
    seq_len: int | None = None,
    rope_length: int | None = None,
    enable_token_mask_for_moe: bool | None = None,
) -> OPTMoEModel.Config:
    """The complete, sharded model config of a flat spec (a flavor name, or a
    `model_registry` result the run config edited): its `layer` template
    expanded per layer (patterns, depth init, inits), the sharding for
    `parallelism`, the CP inner attentions when CP is on. The spec is not
    changed. `seq_len`: the training context (default: the flavor's RoPE
    length); `rope_length`: a longer RoPE cache (validation sequences);
    `enable_token_mask_for_moe`: None keeps the spec's."""
    config = copy.deepcopy(model_registry(spec) if isinstance(spec, str) else spec)
    config.finalize(
        parallelism=parallelism,
        max_context_length=seq_len or config.max_context_length,
        rope_length=rope_length,
        enable_token_mask_for_moe=enable_token_mask_for_moe,
    )
    return config
