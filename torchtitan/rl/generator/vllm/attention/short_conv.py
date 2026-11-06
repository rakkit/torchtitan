# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM generation for opt_moe's Canon short conv (``CanonShortConv``).

``x + conv(x)`` over the attention's fused q / k / v (or k / v) channels, with each
sequence's last ``W - 1`` inputs in vLLM's paged convolution state, on Attention
Gym's paging-aware kernels: the convolution half of ``VLLMInnerKDA`` /
``VLLMInnerGDN``, without activation and recurrence. The attention owns the conv
weights; this module owns only the cache plumbing and the kernel dispatch.
"""

from dataclasses import dataclass

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from torchtitan.models.common.attention import LinearAttentionMetadata
from torchtitan.models.common.attn_gym import causal_conv1d_decode, paged_causal_conv1d
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.opt_moe.gated_swa_attention import (
    CanonShortConv,
    GatedSWAttention,
)
from torchtitan.rl.generator.vllm.attention.linear_attention import (
    configure_hybrid_state,
    LinearAttentionExecutionPath,
    TorchTitanLinearAttentionBackend,
    TorchTitanLinearAttentionMetadata,
    unique_state_shape,
)


class VLLMCanonShortConv(CanonShortConv, MambaBase):
    """Paged-state Canon short conv. Head-sharded with the attention under
    tensor parallelism (the fused q / k / v channels are grouped per KV head).
    Speculative decoding is not supported (one state slot per sequence)."""

    @dataclass(kw_only=True, slots=True)
    class Config(CanonShortConv.Config):
        layer_idx: int
        num_channels: int
        """The fused q / k / v (or k / v) width over all tensor-parallel ranks."""
        conv_kernel_size: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        vllm_config = get_current_vllm_config()
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        speculative_config = vllm_config.speculative_config
        if speculative_config and speculative_config.num_speculative_tokens:
            raise ValueError(
                "The paged Canon short conv does not support speculative decoding."
            )
        if config.num_channels % self.tensor_parallel_size != 0:
            raise ValueError(
                f"num_channels ({config.num_channels}) must be divisible by "
                f"tensor_parallel_size ({self.tensor_parallel_size})."
            )
        if is_conv_state_dim_first():
            raise ValueError(
                "The paged Canon short conv requires VLLM_SSM_CONV_STATE_LAYOUT=SD "
                "so the paged convolution history has contiguous channels."
            )
        self.num_channels = config.num_channels
        self.conv_kernel_size = config.conv_kernel_size
        # vLLM populates this via the KV-cache allocator: (conv_state,).
        self.kv_cache = (torch.tensor([]),)
        self.prefix = f"model.layers.{config.layer_idx}.attention.short_conv"
        compilation_config = vllm_config.compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate short conv layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return TorchTitanLinearAttentionBackend

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        # The backend builds vLLM's GDN metadata (as for KDA).
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            self.tensor_parallel_size, self.num_channels, self.conv_kernel_size
        )

    def forward(
        self,
        x_TC: torch.Tensor,
        weight_C1W: torch.Tensor,
        conv_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        if conv_metadata is not None:
            raise ValueError("vLLM manages the sequences; no conv metadata expected")
        out_TC = x_TC.clone()  # rows no sequence owns pass through (graph replays)
        self._forward(x_TC, weight_C1W.squeeze(1).to(x_TC.dtype), out_TC)
        return out_TC

    @eager_break_during_capture
    def _forward(
        self, x_TC: torch.Tensor, weight_CW: torch.Tensor, out_TC: torch.Tensor
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        # vLLM's profiling / warmup runs have no attention metadata.
        if attn_metadata is None:
            return
        assert isinstance(attn_metadata, dict)
        metadata = attn_metadata[self.prefix]
        assert isinstance(metadata, TorchTitanLinearAttentionMetadata)
        num_actual_tokens = metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return
        state_indices = metadata.non_spec_state_indices_tensor
        cu_seqlens = metadata.non_spec_query_start_loc
        has_initial_state = metadata.has_initial_state
        assert (
            state_indices is not None
            and cu_seqlens is not None
            and has_initial_state is not None
        )
        (conv_state,) = self.kv_cache
        if metadata.execution_path is LinearAttentionExecutionPath.SINGLE_TOKEN:
            rows = state_indices.numel()
            out_TC[:rows] += causal_conv1d_decode(
                x_TC[:rows],
                weight_CW,
                conv_state,
                activation=None,
                state_indices=state_indices,
                has_initial_state=has_initial_state,
            )
            return
        out_TC[:num_actual_tokens] += paged_causal_conv1d(
            x_TC[:num_actual_tokens].unsqueeze(0),
            weight_CW,
            conv_state,
            state_indices,
            activation=None,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        ).squeeze(0)


def _short_conv_shape(attention_cfg: GatedSWAttention.Config) -> tuple[int, int]:
    """(channels of the one conv call: q / k / v or k / v, kernel width) of an
    attention's Canon conv."""
    convs = (attention_cfg.q_conv, attention_cfg.k_conv, attention_cfg.v_conv)
    channels = sum(conv.out_channels for conv in convs if conv is not None)
    return channels, attention_cfg.k_conv.kernel_size


def generation_short_conv_config(
    attention_cfg: GatedSWAttention.Config, layer_idx: int
) -> VLLMCanonShortConv.Config:
    channels, width = _short_conv_shape(attention_cfg)
    return VLLMCanonShortConv.Config(
        layer_idx=layer_idx, num_channels=channels, conv_kernel_size=width
    )


def maybe_configure_short_conv_hybrid_model(
    model_cls: type, model_config: Decoder.Config
) -> None:
    """Attach vLLM's hybrid-state interface when the model has Canon convs."""
    state_shape = unique_state_shape(
        "Canon short conv",
        (
            _short_conv_shape(layer.attention)
            for layer in model_config.layers
            if isinstance(layer.attention, GatedSWAttention.Config)
            and layer.attention.k_conv is not None
        ),
    )
    if state_shape is None:
        return
    num_channels, conv_kernel_size = state_shape

    def get_state_shape(vllm_config):
        return MambaStateShapeCalculator.short_conv_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            num_channels,
            conv_kernel_size,
        )

    def get_state_dtype(vllm_config):
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    configure_hybrid_state(
        model_cls,
        get_state_shape=get_state_shape,
        get_state_dtype=get_state_dtype,
        get_state_copy_func=MambaStateCopyFuncCalculator.short_conv_state_copy_func,
    )
