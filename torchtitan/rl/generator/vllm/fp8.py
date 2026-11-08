# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-wise FP8 rollouts on Hopper: the generator's linears and routed experts in
FP8, the trainer in bf16.

Each FP8 module keeps its bf16 parameters as the weight sync's load target and
an e4m3 copy with one fp32 scale per 128 x 128 block (per expert for the
experts). ``requantize`` refreshes the copy in place after a load and can free
the bf16 storage, which FSDP re-allocates at the next unshard; the forward reads
only the FP8 copy. Activations are quantized per 1 x 128 group; the GEMMs are
vLLM's Triton block-scaled kernels (the DeepSeek-V3 / FP8-RL format,
https://arxiv.org/abs/2601.18150). The LM head, routers, embeddings and norms
stay bf16.

Weight copy, at 128 x 128 blocks::

    w [..., out, in] bf16  -> w_fp8 [..., out, in] e4m3 + scale [..., out / 128, in / 128] fp32
"""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.config.transform.base import ModelConfigTransform
from torchtitan.models.base.module import Module
from torchtitan.models.common.linear.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.moe.moe import RoutedExperts
from torchtitan.rl.generator.vllm.moe import VLLMFusedRoutedExperts

_BLOCK = 128
_FP8 = torch.float8_e4m3fn
# fp32 elements quantized at once: bounds the transient memory per call (the
# generator's KV cache is already allocated when the weight sync runs)
_CHUNK_NUMEL = 1 << 25


def _alloc_block_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, o, i = w.shape
    return (
        torch.empty(w.shape, dtype=_FP8, device=w.device),
        torch.empty(
            (*lead, o // _BLOCK, i // _BLOCK), dtype=torch.float32, device=w.device
        ),
    )


def _block_quantize(
    w: torch.Tensor, out_q: torch.Tensor, out_scale: torch.Tensor
) -> None:
    """``w [..., O, I]`` -> e4m3 blocks in ``out_q`` and their dequantization scales
    ``out_scale [..., O / 128, I / 128]``, a few leading slices at a time."""
    o, i = w.shape[-2:]
    w_GOI = w.reshape(-1, o, i)
    q_GOI = out_q.view(-1, o, i)
    s_GBC = out_scale.view(-1, o // _BLOCK, i // _BLOCK)
    step = max(1, _CHUNK_NUMEL // (o * i))
    for start in range(0, w_GOI.shape[0], step):
        end = start + step
        blocks = (
            w_GOI[start:end]
            .float()
            .reshape(-1, o // _BLOCK, _BLOCK, i // _BLOCK, _BLOCK)
        )
        scale = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-12) / torch.finfo(_FP8).max
        q_GOI[start:end].copy_(
            (blocks / scale[:, :, None, :, None]).to(_FP8).reshape(-1, o, i)
        )
        s_GBC[start:end].copy_(scale)


def _free_storage(tensor: torch.Tensor) -> None:
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)


class BlockFP8InferenceLinear(Linear):
    """A generator ``Linear`` computed in block-wise FP8 (no bias, no gradient)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        def __post_init__(self) -> None:
            if self.bias:
                raise ValueError("BlockFP8InferenceLinear has no bias")
            if (
                self.in_features % _BLOCK
                or (self.num_linears * self.out_features) % _BLOCK
            ):
                raise ValueError(
                    f"BlockFP8InferenceLinear needs dimensions divisible by {_BLOCK}, got "
                    f"in {self.in_features}, out {self.num_linears} x {self.out_features}"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self._weight_fp8: torch.Tensor | None = None
        self._weight_scale: torch.Tensor | None = None

    def fp8_source_params(self) -> list[torch.Tensor]:
        return [self.weight]

    @torch.no_grad()
    def requantize(self, *, free_source: bool = False) -> None:
        """Refresh the FP8 copy from the current bf16 weight, in place after the
        first call (CUDA graphs read the same buffers); ``free_source`` then frees
        the bf16 storage (FSDP re-allocates it at the next unshard)."""
        w_OI = self.weight.detach().flatten(0, -2)
        if self._weight_fp8 is None:
            self._weight_fp8, self._weight_scale = _alloc_block_fp8(w_OI)
        _block_quantize(w_OI, self._weight_fp8, self._weight_scale)
        if free_source:
            _free_storage(self.weight)

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
            w8a8_triton_block_scaled_mm,
        )

        del weight, bias
        if self._weight_fp8 is None:
            raise RuntimeError("BlockFP8InferenceLinear used before requantize()")
        x_TI = input.reshape(-1, input.shape[-1])
        x_q, x_scale = per_token_group_quant_fp8(x_TI, _BLOCK)
        out_TO = w8a8_triton_block_scaled_mm(
            x_q,
            self._weight_fp8,
            x_scale,
            self._weight_scale,
            [_BLOCK, _BLOCK],
            output_dtype=input.dtype,
        )
        return out_TO.reshape(*input.shape[:-1], out_TO.shape[-1])


class BlockFP8FusedRoutedExperts(VLLMFusedRoutedExperts):
    """``RoutedExperts`` computed by vLLM's ``fused_experts`` in block-wise FP8
    (128 x 128 blocks per expert; activations quantized per 1 x 128 group inside)."""

    @dataclass(kw_only=True, slots=True)
    class Config(VLLMFusedRoutedExperts.Config):
        def __post_init__(self) -> None:
            VLLMFusedRoutedExperts.Config.__post_init__(self)
            if self.w13.in_features % _BLOCK or self.w13.out_features % _BLOCK:
                raise ValueError(
                    f"BlockFP8FusedRoutedExperts needs dimensions divisible by {_BLOCK}, "
                    f"got dim {self.w13.in_features}, hidden {self.w13.out_features}"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self._fp8: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        self._fp8 = None
        self._quant_config = None

    def fp8_source_params(self) -> list[torch.Tensor]:
        return [self.w13.weight, self.w2.weight]

    @torch.no_grad()
    def requantize(self, *, free_source: bool = False) -> None:
        """As ``BlockFP8InferenceLinear.requantize``, for w13 [E, 2F, D] and w2 [E, D, F]."""
        w13_ENK = self.w13.weight.detach().flatten(1, -2)
        w2_ENK = self.w2.weight.detach().flatten(1, -2)
        if self._fp8 is None:
            from vllm.model_executor.layers.fused_moe.config import (
                fp8_w8a8_moe_quant_config,
            )

            self._fp8 = (*_alloc_block_fp8(w13_ENK), *_alloc_block_fp8(w2_ENK))
            self._quant_config = fp8_w8a8_moe_quant_config(
                w1_scale=self._fp8[1],
                w2_scale=self._fp8[3],
                a1_scale=None,
                a2_scale=None,
                block_shape=[_BLOCK, _BLOCK],
            )
        _block_quantize(w13_ENK, self._fp8[0], self._fp8[1])
        _block_quantize(w2_ENK, self._fp8[2], self._fp8[3])
        if free_source:
            for param in self.fp8_source_params():
                _free_storage(param)

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        routing_map_TE: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        del routing_map_TE, padding_mask_T
        if self._fp8 is None:
            raise RuntimeError("BlockFP8FusedRoutedExperts used before requantize()")
        return fused_experts(
            x_TD.bfloat16(),
            self._fp8[0],
            self._fp8[2],
            topk_scores_TK.float(),
            topk_expert_ids_TK.to(torch.int32),
            quant_config=self._quant_config,
        ).type_as(x_TD)


FP8_MODULES = (BlockFP8InferenceLinear, BlockFP8FusedRoutedExperts)


def fp8_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, FP8_MODULES)]


_CONVERTED_LINEARS = (
    Linear.Config,
    ColumnParallelLinear.Config,
    RowParallelLinear.Config,
)
_CONVERTED_EXPERTS = (RoutedExperts.Config, VLLMFusedRoutedExperts.Config)


def _replace(parent, attr, new) -> None:
    if isinstance(attr, int):
        parent[attr] = new
    else:
        setattr(parent, attr, new)


class BlockFP8RolloutTransform(ModelConfigTransform):
    """For the generator's model config: every ``Linear`` / ``ColumnParallelLinear`` /
    ``RowParallelLinear`` config under ``layers.`` outside the router becomes a
    ``BlockFP8InferenceLinear.Config`` with the ``Linear`` fields, and every routed
    experts config a ``BlockFP8FusedRoutedExperts.Config``
    (``apply_transforms(config.generator, [BlockFP8RolloutTransform()])``; the
    parallel variants are plain linears at the TP 1 this needs: the model wrapper
    refuses it under tensor or expert parallelism). The LM head, embeddings and
    routers stay bf16."""

    def transform(self, model: Module.Config) -> Module.Config:
        names = [f.name for f in dataclasses.fields(Linear.Config)]
        for fqn, cfg, parent, attr in model.traverse(Linear.Config):
            if type(cfg) not in _CONVERTED_LINEARS or not fqn.startswith("layers."):
                continue
            if ".router." in f".{fqn}.":
                continue
            fp8 = BlockFP8InferenceLinear.Config(**{n: getattr(cfg, n) for n in names})
            _replace(parent, attr, fp8)
        for _fqn, cfg, parent, attr in model.traverse(RoutedExperts.Config):
            if type(cfg) not in _CONVERTED_EXPERTS:
                continue
            fp8 = BlockFP8FusedRoutedExperts.Config(
                **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
            )
            if parent is None:
                return fp8
            _replace(parent, attr, fp8)
        return model
