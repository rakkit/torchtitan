# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

from torch import nn

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class MoEModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_dense_layers: int = 0
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1  # If -1, then take vocab size from tokenizer.
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and
    # if `False`, each uses the total number of transformer blocks. If
    # `None`, do not apply any depth scaling.
    depth_init: bool | None = True
    first_in_init_fn_type: str = "normal"
    first_in_init_std: float = 1.0
    # Exponent applied to the first input layer's input dimensionality
    # to obtain its init std factor.
    first_in_exp: float = 0.0
    router_init_fn_type: str = "trunc_normal"
    intermediate_init_fn_type: str = "trunc_normal"
    intermediate_init_std: float = 0.02
    # Exponent applied to the model's hidden dimensionality to obtain
    # intermediate layers' init std factors.
    intermediate_exp: float = 0.0
    # Whether to initialize the GLU gate as if it was a residual layer.
    init_gate_as_residual: bool = True
    final_out_init_fn_type: str = "trunc_normal"
    final_out_init_std: float = 1.0
    # Exponent applied to the final output layer's input dimensionality
    # to obtain its init std factor.
    final_out_exp: float = -0.5
    residual_scale: str = "identity"
    norm_type: str = "rmsnorm"
    qk_norm: bool = False
    # If this is True, it implies `qk_norm=True`.
    norm_everywhere: bool = False

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0
    pad_id: int = -1

    # Number of additional modules to insert for multi-token prediction.
    num_mtp_modules: int = 0

    # ==== MoE specific args ====
    n_shared_experts: int = 1
    n_routed_experts: int = 0
    activate_experts: int = 0
    moe_router_bias_update_speed: float = 0.001  # dpskv3, 0.001
    # TODO(JSC): Need ablation about the learning rate of the router bias
    moe_aux_loss_alpha: float = 0.01  # OLMoE, default 0.01
    moe_router_scaling_factor: float | None = None
    moe_router_bias_update_norm_factor: str = "sign"  # sign or spectral
    # dpskv3 2.5, moonlight 2.446, set None to auto-compute
    moe_router_use_bias_for_routing: bool = True
    moe_init_all_experts_same: bool = False

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        for name in [
            "first_in_init_fn_type",
            "first_in_init_std",
            "first_in_exp",
            "router_init_fn_type",
            "intermediate_init_fn_type",
            "intermediate_init_std",
            "intermediate_exp",
            "init_gate_as_residual",
            "final_out_init_fn_type",
            "final_out_init_std",
            "final_out_exp",
            "residual_scale",
            "norm_type",
        ]:
            value = getattr(job_config.model, name)
            setattr(self, name, value)

        self.num_mtp_modules = job_config.training.num_mtp_tokens
        assert self.num_mtp_modules >= 0

        # Normalize `depth_init`.
        depth_init = job_config.model.depth_init.lower()
        if depth_init == ["true", "depth"]:
            depth_init = True
        elif depth_init == ["false", "total_depth"]:
            depth_init = False
        elif depth_init in ["none", "null", "identity"]:
            depth_init = None
        self.depth_init = depth_init

        if self.vocab_size == -1:
            tokenizer = kwargs.get("tokenizer")
            assert isinstance(tokenizer, BaseTokenizer), (
                "Need a `BaseTokenizer` to be passed to `update_from_config` via the "
                "`tokenizer` keyword argument to automatically set `vocab_size` "
                "(since `vocab_size == -1`)."
            )
            self.vocab_size = tokenizer.get_vocab_size()
            # `eos_id` is not part of the `Tokenizer` interface, so keep it
            # optional.
            if hasattr(tokenizer, "eos_id"):
                self.eos_id = tokenizer.eos_id
            # `pad_id` is not part of the `Tokenizer` interface, so keep it
            # optional.
            if hasattr(tokenizer, "pad_id"):
                self.pad_id = tokenizer.pad_id
            # Add an additional vocab element if we are explicitly
            # supporting a pad token.
            if self.pad_id >= 0:
                self.vocab_size += 1

        if job_config.model.vocab_size_multiple_of:
            orig_vocab_size = self.vocab_size
            vocab_divisor = job_config.model.vocab_size_multiple_of
            self.vocab_size = int(
                math.ceil(self.vocab_size / vocab_divisor) * vocab_divisor
            )
            logger.info(
                f"Padded vocab size from {orig_vocab_size} to {self.vocab_size}."
            )

        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

        if (
            job_config.parallelism.pipeline_parallel_degree > 1
            and self.use_flex_attn
            and self.attn_mask_type == "block_causal"
        ):
            raise RuntimeError(
                "PP + block causal FlexAttention support will be fixed soon."
            )
        self.max_seq_len = seq_len
        self.qk_norm = self.qk_norm or self.norm_everywhere

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_expert = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "tok_embeddings" in name or "output" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "feed_forward.shared_expert" in name:
                nparams_shared_expert += p.numel()
            elif "feed_forward.router" in name:
                nparams_moe_router += p.numel()
            elif "feed_forward.experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_expert + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = nparams_moe_router + nparams_shared_expert
        if self.activate_experts > 0:
            nparams_sparse_active += (
                nparams_experts * self.activate_experts // self.n_routed_experts
            )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = (
            6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
            + 12 * l * h * q * t
        )

        active_params = nparams_sparse_active + nparams_embedding

        return active_params, nparams, num_flops_per_token
