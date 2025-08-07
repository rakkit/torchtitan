# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_byte_tokenizer, build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_llama
from .infra.pipeline import pipeline_llama
from .model.args import TransformerModelArgs
from .model.bitnet_model import BitNetTransformer
from .model.model import Transformer
from .model.state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_configs",
]


llama3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2000, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2000,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "debugmodel_qk": TransformerModelArgs(
        dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    # 14M parameters
    "debugmodel-multiplier-1": TransformerModelArgs(
        dim=256,
        n_layers=16,
        n_heads=4,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    # 55M parameters
    "debugmodel-multiplier-2": TransformerModelArgs(
        dim=512,
        n_layers=16,
        n_heads=8,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    # 206M parameters
    "debugmodel-multiplier-4": TransformerModelArgs(
        dim=1024,
        n_layers=16,
        n_heads=16,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    # 824M parameters
    "debugmodel-multiplier-8": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    # 3.2B parameters
    "debugmodel-multiplier-16": TransformerModelArgs(
        dim=4096,
        n_layers=16,
        n_heads=64,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    # 1.9M parameters
    "debugmodel-2layers-multiplier-1": TransformerModelArgs(
        dim=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    # 609M parameters
    "debugmodel-2layers-multiplier-4": TransformerModelArgs(
        dim=1024,
        n_layers=2,
        n_heads=16,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    # 2.4B parameters
    "debugmodel-2layers-multiplier-8": TransformerModelArgs(
        dim=2048,
        n_layers=2,
        n_heads=32,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    "debugmodel-2layers-multiplier-16": TransformerModelArgs(
        dim=4096,
        n_layers=2,
        n_heads=64,
        n_kv_heads=None,
        init_gate_as_residual=False,
        multiple_of=256,
        rope_theta=500000,
        qk_norm=True,
        depth_init=False,
        norm_eps=1e-30,
    ),
    "1B-Proxy-2layers": TransformerModelArgs(
        dim=256,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,  # need KV_head > TP for TP debugging
        ffn_dim_multiplier=1,  # need to check
        multiple_of=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "1B-Proxy-8layers": TransformerModelArgs(
        dim=256,
        n_layers=8,
        n_heads=2,
        n_kv_heads=1,  # need KV_head > TP for TP debugging
        ffn_dim_multiplier=1,  # need to check
        multiple_of=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "1B-Proxy-16layers": TransformerModelArgs(
        dim=256,
        n_layers=16,
        n_heads=2,
        n_kv_heads=1,  # need KV_head > TP for TP debugging
        ffn_dim_multiplier=1,  # need to check
        multiple_of=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "1B-Proxy-64layers": TransformerModelArgs(
        dim=256,
        n_layers=64,
        n_heads=2,
        n_kv_heads=1,  # need KV_head > TP for TP debugging
        ffn_dim_multiplier=1,  # need to check
        multiple_of=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "1B-Proxy": TransformerModelArgs(
        dim=256,
        n_layers=24,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim_multiplier=1,  # need to check
        multiple_of=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "1B": TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1,  # need to check
        multiple_of=256,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "8B_qk": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        qk_norm=True,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "3B-norm-everywhere": TransformerModelArgs(
        dim=2048,
        n_layers=36,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=2,
        multiple_of=256,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "8B-norm-everywhere": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "70B-norm-everywhere": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
}


byte_llama3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=-1, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=-1,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "1B": TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        vocab_size=-1,
        ffn_dim_multiplier=1,  # need to check
        multiple_of=256,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=-1,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "8B_qk": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=-1,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        qk_norm=True,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        vocab_size=-1,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        vocab_size=-1,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}

llama2_configs = {
    "debugmodel": llama3_configs["debugmodel"],
    "7B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=32000,
    ),
}


byte_llama2_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=-1, rope_theta=500000
    ),
    "7B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=-1,
    ),
}


register_train_spec(
    TrainSpec(
        name="llama3",
        model_cls=Transformer,
        model_args=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
)

register_train_spec(
    TrainSpec(
        name="byte_llama3",
        model_cls=Transformer,
        model_args=byte_llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_byte_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
)

register_train_spec(
    TrainSpec(
        name="llama2",
        model_cls=Transformer,
        model_args=llama2_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        # TODO Not tested, but we expect that the
        #      `Llama3StateDictAdapter` works for Llama-2 as well.
        state_dict_adapter=Llama3StateDictAdapter,
    )
)

register_train_spec(
    TrainSpec(
        name="byte_llama2",
        model_cls=Transformer,
        model_args=byte_llama2_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_byte_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        # TODO Not tested, but we expect that the
        #      `Llama3StateDictAdapter` works for Llama-2 as well.
        state_dict_adapter=Llama3StateDictAdapter,
    )
)

register_train_spec(
    TrainSpec(
        name="bit_byte_llama3",
        model_cls=BitNetTransformer,
        model_args=byte_llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_byte_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
)
