# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.tokenizer import build_hf_byte_tokenizer, build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.build_optimizer import build_moe_optimizers as build_optimizers
from .infra.parallelize import parallelize_llama
from .infra.pipeline import pipeline_llama
from .model.args import MoEModelArgs
from .model.model import Transformer


__all__ = [
    "MoEModelArgs",
    "Transformer",
    "moe_llama3_configs",
]


moe_llama3_configs = {
    "debugmodel": MoEModelArgs(
        dim=512,  # beaware this if 2x then the llama3-debugmodel
        n_layers=8,
        n_heads=16,
        rope_theta=10000,
        n_shared_experts=1,
        activate_experts=4,
        n_routed_experts=8,
        qk_norm=True,
        norm_everywhere=False,
        depth_init="total_depth",
        norm_eps=1e-30,
    ),
    "1B-7B-Proxy-8layers": MoEModelArgs(
        dim=512,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.01,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-Proxy": MoEModelArgs(
        dim=512,
        n_layers=24,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.01,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-Proxy-wo-bias": MoEModelArgs(
        dim=512,
        n_layers=24,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.01,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=False,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-Proxy-wo-aux-loss": MoEModelArgs(
        dim=512,
        n_layers=24,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.0,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-Proxy-wo-bias-wo-aux-loss": MoEModelArgs(
        dim=512,
        n_layers=24,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.0,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=False,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-Proxy-wo-aux-loss-2layers": MoEModelArgs(
        dim=512,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.0,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-Proxy-wo-aux-loss-8layers": MoEModelArgs(
        dim=512,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.0,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-Proxy-wo-aux-loss-64layers": MoEModelArgs(
        dim=512,
        n_layers=64,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.0,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-Proxy-wo-aux-loss-8layers-1dense": MoEModelArgs(
        dim=512,
        n_layers=8,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.0,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=256,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.01,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-wo-aux-loss": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=256,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.0,
        moe_router_scaling_factor=2.8232,  # 8 of 64 experts
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "test": MoEModelArgs(
        dim=256,
        n_layers=8,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim_multiplier=1,
        multiple_of=64,
        n_shared_experts=0,
        activate_experts=4,
        n_routed_experts=8,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init="total_depth",
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        # MoE specific args
        moe_router_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.0,
        moe_router_scaling_factor=1.0,
        moe_router_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
}

register_train_spec(
    TrainSpec(
        name="MoEllama3",
        model_cls=Transformer,
        model_args=moe_llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
)

register_train_spec(
    TrainSpec(
        name="byte_MoEllama3",
        model_cls=Transformer,
        model_args=moe_llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_byte_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
)
