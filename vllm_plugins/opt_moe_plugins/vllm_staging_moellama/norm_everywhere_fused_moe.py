# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Norm-everywhere MoE wrappers that reuse vLLM fused kernels.

Key idea:
- call the original fused activation from vLLM kernels
- then apply parameter-free RMSNorm on the activation output

This gives `activation + no-parameter RMSNorm` behavior without writing any
new CUDA/Triton kernels and without patching vLLM core code.
"""

import inspect
import logging
import os
import shutil
import types
from pathlib import Path

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.fused_batched_moe import BatchedTritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

_H100_ALIAS_CONFIG_READY = False
logger = logging.getLogger(__name__)


def _kernel_fused_experts_replaceable() -> bool:
    """Return whether this vLLM build still allows replacing kernel experts."""
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import FusedMoEKernel
    except Exception:
        return True

    fused_experts_attr = inspect.getattr_static(FusedMoEKernel, "fused_experts", None)
    if isinstance(fused_experts_attr, property):
        return fused_experts_attr.fset is not None
    return True


def _maybe_enable_h100_tuned_moe_config_alias() -> None:
    """Map H100 generic device name to shipped H100_80GB_HBM3 tuned configs."""
    global _H100_ALIAS_CONFIG_READY
    if _H100_ALIAS_CONFIG_READY:
        return
    _H100_ALIAS_CONFIG_READY = True

    if os.environ.get("VLLM_TUNED_CONFIG_FOLDER"):
        return

    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod
        from vllm.platforms import current_platform

        device_name = str(current_platform.get_device_name()).strip()
        normalized_name = device_name.replace("-", "_").replace(" ", "_")
        # Only apply this alias workaround on H100-family devices.
        if "H100" not in normalized_name.split("_"):
            return

        src_dir = Path(fused_moe_mod.__file__).resolve().parent / "configs"
        if not src_dir.is_dir():
            return

        dst_dir = Path(__file__).resolve().parent / "moe_tuned_configs"
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src in src_dir.glob("E=*,N=*,device_name=NVIDIA_H100_80GB_HBM3*.json"):
            dst_name = src.name.replace("NVIDIA_H100_80GB_HBM3", "NVIDIA_H100")
            dst = dst_dir / dst_name
            if not dst.exists():
                shutil.copyfile(src, dst)

        if any(dst_dir.glob("E=*,N=*,device_name=NVIDIA_H100*.json")):
            os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(dst_dir)
    except Exception:
        # Non-critical performance optimization.
        return


def _weightless_rms_norm_inplace(x: torch.Tensor, eps: float) -> None:
    # Match HF reference behavior: RMS stats in fp32, output cast back.
    if x.dtype == torch.float32:
        variance = x.square().mean(dim=-1, keepdim=True)
    else:
        variance = x.to(torch.float32).square().mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    x.mul_(inv_rms.to(dtype=x.dtype))


def _patch_expert_activation_inplace(expert: object, rms_norm_eps: float) -> bool:
    activation = getattr(expert, "activation", None)
    if activation is None or getattr(
        expert, "_norm_everywhere_activation_patched", False
    ):
        return False

    def patched_activation(
        self, activation_name: str, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        activation(activation_name, output, input)
        _weightless_rms_norm_inplace(output, float(rms_norm_eps))

    expert.activation = types.MethodType(patched_activation, expert)
    expert._norm_everywhere_activation_patched = True
    expert.rms_norm_eps = float(rms_norm_eps)
    return True


class NormEverywhereTritonExperts(TritonExperts):
    def __init__(
        self,
        moe_config: mk.FusedMoEConfig | None,
        quant_config: mk.FusedMoEQuantConfig,
        rms_norm_eps: float,
    ) -> None:
        # vLLM has had both constructor shapes:
        # - TritonExperts(moe_config=..., quant_config=...)
        # - TritonExperts(quant_config=...)
        if moe_config is not None:
            try:
                super().__init__(moe_config=moe_config, quant_config=quant_config)
            except TypeError:
                super().__init__(quant_config=quant_config)
        else:
            super().__init__(quant_config=quant_config)
        self.rms_norm_eps = float(rms_norm_eps)

    def activation(
        self, activation: str, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        super().activation(activation, output, input)
        _weightless_rms_norm_inplace(output, self.rms_norm_eps)


class NormEverywhereBatchedTritonExperts(BatchedTritonExperts):
    def __init__(
        self,
        moe_config: mk.FusedMoEConfig | None,
        quant_config: mk.FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
        rms_norm_eps: float,
    ) -> None:
        # vLLM has had both constructor shapes:
        # - BatchedTritonExperts(moe_config=..., quant_config=..., ...)
        # - BatchedTritonExperts(max_num_tokens=..., num_dispatchers=..., quant_config=...)
        if moe_config is not None:
            try:
                super().__init__(
                    moe_config=moe_config,
                    quant_config=quant_config,
                    max_num_tokens=max_num_tokens,
                    num_dispatchers=num_dispatchers,
                )
            except TypeError:
                super().__init__(
                    quant_config=quant_config,
                    max_num_tokens=max_num_tokens,
                    num_dispatchers=num_dispatchers,
                )
        else:
            super().__init__(
                quant_config=quant_config,
                max_num_tokens=max_num_tokens,
                num_dispatchers=num_dispatchers,
            )
        self.rms_norm_eps = float(rms_norm_eps)

    def activation(
        self, activation: str, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        super().activation(activation, output, input)
        _weightless_rms_norm_inplace(output, self.rms_norm_eps)


class NormEverywhereUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Reuse vLLM unquantized MoE path, with RMSNorm injected after activation."""

    def __init__(self, moe: mk.FusedMoEConfig, rms_norm_eps: float) -> None:
        self.rms_norm_eps = float(rms_norm_eps)
        super().__init__(moe)
        # Force Triton backend so we can inject RMSNorm in activation hook.
        if not bool(getattr(self, "is_monolithic", False)):
            self.unquantized_backend = UnquantizedMoeBackend.TRITON
        self._kernel_fused_experts_replaceable = _kernel_fused_experts_replaceable()

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ):
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            return NormEverywhereBatchedTritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                rms_norm_eps=self.rms_norm_eps,
            )
        return NormEverywhereTritonExperts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            rms_norm_eps=self.rms_norm_eps,
        )

    def _replace_kernel_experts_with_norm(self) -> None:
        if self.kernel is None:
            return

        fused_experts = self.kernel.fused_experts
        patched = False
        if isinstance(fused_experts, (BatchedTritonExperts, TritonExperts)):
            patched = _patch_expert_activation_inplace(
                fused_experts,
                rms_norm_eps=self.rms_norm_eps,
            )
        if patched:
            logger.info(
                "Norm-everywhere fused MoE kernel active via in-place activation patch"
            )

    def _verify_norm_everywhere_kernel(self) -> None:
        if self.kernel is None:
            raise RuntimeError("norm-everywhere fused MoE kernel setup missing kernel")
        if getattr(self, "unquantized_backend", None) != UnquantizedMoeBackend.TRITON:
            raise RuntimeError(
                "norm-everywhere fused MoE requires TRITON backend, "
                f"got {getattr(self, 'unquantized_backend', None)!r}"
            )

        fused_experts = getattr(self.kernel, "fused_experts", None)
        if getattr(fused_experts, "_norm_everywhere_activation_patched", False):
            logger.info(
                "Norm-everywhere fused MoE kernel verified via activation patch"
            )
            return

        raise RuntimeError(
            "norm-everywhere fused MoE kernel verification failed: "
            f"expected patched TritonExperts, got {type(fused_experts).__name__}"
        )

    def _setup_kernel(
        self,
        layer: torch.nn.Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        super()._setup_kernel(layer=layer, w13=w13, w2=w2)
        self._replace_kernel_experts_with_norm()
        self._verify_norm_everywhere_kernel()


class NormEverywhereSharedFusedMoE(SharedFusedMoE):
    """SharedFusedMoE with norm-everywhere activation hook for unquantized path."""

    def __init__(self, *, rms_norm_eps: float, **kwargs) -> None:
        # _maybe_enable_h100_tuned_moe_config_alias()
        super().__init__(**kwargs)
        self.supports_norm_everywhere = False
        quant_method = self.quant_method
        if isinstance(quant_method, UnquantizedFusedMoEMethod):
            norm_quant_method = NormEverywhereUnquantizedFusedMoEMethod(
                self.moe_config,
                rms_norm_eps=rms_norm_eps,
            )
            backend = getattr(norm_quant_method, "unquantized_backend", None)
            is_monolithic = bool(getattr(norm_quant_method, "is_monolithic", False))
            if (backend == UnquantizedMoeBackend.TRITON) and not is_monolithic:
                self._replace_quant_method(norm_quant_method)
                self.base_quant_method = self.quant_method
                self.supports_norm_everywhere = True
