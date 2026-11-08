# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM's fused MoE kernel for the generator's routed experts.

One Triton call per layer (vLLM's ``fused_experts``: token sort, both expert
GEMMs, SwiGLU and the score-weighted sum) instead of the training path's
dispatch / grouped GEMM / combine chain, which at decode batch sizes is a
chain of small kernels. Same parameters as ``RoutedExperts`` (``w13`` stacks
the SwiGLU gate rows then the up rows, as vLLM's ``SiluAndMul`` expects), so
the weight sync is unchanged. Expert-parallel and tensor-parallel generators
keep the training path.
"""

import dataclasses
from dataclasses import dataclass

import torch

from torchtitan.config.transform.base import ModelConfigTransform
from torchtitan.models.base.module import Module
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.moe.moe import RoutedExperts


class VLLMFusedRoutedExperts(RoutedExperts):
    """``RoutedExperts`` computed by vLLM's ``fused_experts`` (bf16)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        def __post_init__(self) -> None:
            RoutedExperts.Config.__post_init__(self)
            if type(self.activation_fn) is not SwiGLU.Config:
                raise ValueError(
                    "VLLMFusedRoutedExperts needs SwiGLU experts (vLLM's SiluAndMul)"
                )
            if self.output_postprocess is not None:
                raise ValueError("VLLMFusedRoutedExperts has no output postprocess")

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
        return fused_experts(
            x_TD.bfloat16(),
            self.w13.weight.flatten(1, 2),
            self.w2.weight,
            topk_scores_TK.float(),
            topk_expert_ids_TK.to(torch.int32),
        ).type_as(x_TD)


class VLLMFusedExpertsTransform(ModelConfigTransform):
    """For the generator's model config: every ``RoutedExperts.Config`` becomes a
    ``VLLMFusedRoutedExperts.Config`` with the same fields (``apply_transforms(
    config.generator, [VLLMFusedExpertsTransform()])``). The model wrapper refuses
    it under tensor or expert parallelism and in batch-invariant mode."""

    def transform(self, model: Module.Config) -> Module.Config:
        for _fqn, cfg, parent, attr in model.traverse(RoutedExperts.Config):
            if type(cfg) is not RoutedExperts.Config:
                continue
            fused = VLLMFusedRoutedExperts.Config(
                **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
            )
            if parent is None:
                return fused
            if isinstance(attr, int):
                parent[attr] = fused
            else:
                setattr(parent, attr, fused)
        return model
