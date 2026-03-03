# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec
from .hf_assests import setup_hf

from .infra.parallelize import parallelize_llama
from .model.args import MoEModelArgs, RoPEScalingArgs
from .model.model import Transformer
from .model.moe import MoEArgs
from .model.state_dict_adapter import MoEllamaStateDictAdapter

__all__ = [
    "MoEArgs",
    "MoEModelArgs",
    "Transformer",
    "moe_llama_configs",
    "MoEllamaStateDictAdapter",
]


moe_llama_configs = {
    "debugmodel": MoEModelArgs(
        dim=512,  # beaware this if 2x then the llama3-debugmodel
        n_layers=8,
        n_heads=16,
        rope_theta=10000,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=1,
            top_k=4,
        ),
        qk_norm=True,
        norm_everywhere=False,
        norm_eps=1e-30,
    ),
    "1B-7B-Proxy-8layers": MoEModelArgs(
        dim=512,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        n_dense_layers=1,
    ),
    "1B-7B-Proxy": MoEModelArgs(
        dim=512,
        n_layers=24,
        n_heads=4,
        n_kv_heads=2,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
    ),
    "debug-run-v1-1B-7B": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=256,
    ),
    "bsc-1B-7B-opt-c": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_dense_layers=1,
        n_heads=16,
        n_kv_heads=4,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        intermediate_size=5120,
        moe_intermediate_size=640,
    ),
    "bsc-1B-7B-opt-g": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_dense_layers=1,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        intermediate_size=5120,
        moe_intermediate_size=640,
    ),
    "qwen-30bA3b-norm-everywhere": MoEModelArgs(
        dim=2048,
        n_layers=48,
        n_dense_layers=0,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        intermediate_size=6144,
        moe_intermediate_size=768,
    ),
    "bsc-1B-7B-opt-g-32k": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_dense_layers=1,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        rope_scaling_args=RoPEScalingArgs(
            scaling_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=32.0,
            original_max_position_embeddings=4096,
            attention_factor=1.2079441541679836,
            # 0.1*ln(factor) + 1
        ),
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        intermediate_size=5120,
        moe_intermediate_size=640,
    ),
    "bsc-1B-7B-opt-g-64k": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_dense_layers=1,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        rope_scaling_args=RoPEScalingArgs(
            scaling_factor=16.0,
            low_freq_factor=1.0,
            high_freq_factor=32.0,
            original_max_position_embeddings=4096,
            attention_factor=1.2772588722239782,
            # 0.1*ln(factor) + 1
        ),
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        intermediate_size=5120,
        moe_intermediate_size=640,
    ),
    "bsc-1B-7B-opt-g-proxy": MoEModelArgs(
        dim=512,
        n_layers=24,
        n_dense_layers=1,
        n_heads=8,
        n_kv_heads=1,
        head_dim=128,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        intermediate_size=1280,
        moe_intermediate_size=160,
    ),
    "bsc-1B-7B-opt-g-proxy-8layers": MoEModelArgs(
        dim=512,
        n_layers=8,
        n_dense_layers=1,
        n_heads=8,
        n_kv_heads=1,
        head_dim=128,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        intermediate_size=1280,
        moe_intermediate_size=160,
    ),
    "test": MoEModelArgs(
        dim=256,
        n_layers=8,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim_multiplier=1,
        multiple_of=64,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=moe_llama_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=MoEllamaStateDictAdapter,
        hf_assets_setup_fn=setup_hf.copy_and_overwrite_model_config,
    )
