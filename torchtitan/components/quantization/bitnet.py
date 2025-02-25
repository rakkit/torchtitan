# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'torchao' package:
# This script requires the 'torchao' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch/ao by following the
# installation instructions.

import torch.nn as nn

from torchtitan.config.job_config import BitNet, JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger


class BitNetConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = False

        bitnet_config: BitNet = job_config.bitnet
        try:
            from torchao.prototype.quantized_training import (  # noqa: F401
                bitnet_training,
            )
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use BitLinear layers."
            ) from e

        self.enabled = True

        # for `precompute_bitnet_scale_for_fsdp`
        self.precompute_scale = bitnet_config.precompute_bitnet_scale_for_fsdp

        logger.info("BitNet training active")

    def _patch_pytorch(self):
        # We manually apply PyTorch commit
        # 75661f2036d36fd7f869cd749eb6ef5fb40e4772 here.

        import inspect
        from typing import cast

        import torch
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            _to_dtype_if_needed,
            compiled_autograd_enabled,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_param import (
            FSDPParam,
            ShardedState,
        )
        from torch.distributed.tensor.device_mesh import _mesh_resources

        if hasattr(FSDPParam, "shard_mesh_from_root"):
            logger.info("Not patching PyTorch; not necessary")
            return

        @property
        def all_gather_inputs(self) -> list[torch.Tensor]:  # 1D
            self._assert_in_states(
                ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD
            )
            if self.sharded_state == ShardedState.SHARDED:
                if not compiled_autograd_enabled() and hasattr(
                    self._sharded_local_tensor, "fsdp_pre_all_gather"
                ):
                    sharded_local_tensor = self._sharded_local_tensor
                    if self.offload_to_cpu:
                        sharded_local_tensor = sharded_local_tensor.to(
                            self.device, non_blocking=True
                        )
                    pre_all_gather_signature = inspect.signature(
                        sharded_local_tensor.fsdp_pre_all_gather
                    )
                    num_fn_params = len(pre_all_gather_signature.parameters)
                    # Old signature only passes mesh; keep for BC for now
                    assert num_fn_params in (1, 5,), (
                        f"Invalid fsdp_pre_all_gather: {pre_all_gather_signature}\n"
                        "Expects fsdp_pre_all_gather(self, mesh: DeviceMesh, "
                        "module: nn.Module, mp_policy: MixedPrecisionPolicy)"
                    )
                    if num_fn_params == 1:
                        (
                            all_gather_inputs,
                            self._extensions_data.all_gather_metadata,
                        ) = sharded_local_tensor.fsdp_pre_all_gather(
                            self.shard_mesh_from_root
                        )
                    else:
                        (
                            all_gather_inputs,
                            self._extensions_data.all_gather_metadata,
                        ) = sharded_local_tensor.fsdp_pre_all_gather(
                            self.shard_mesh_from_root,
                            self._orig_size,
                            self._contiguous_orig_stride,
                            self._module_info.module,
                            self.mp_policy,
                        )
                        if (
                            sharded_local_tensor.size()
                            != self.padded_sharded_param_size
                            and any(
                                all_gather_input.size()
                                != self.padded_sharded_param_size
                                for all_gather_input in all_gather_inputs
                            )
                        ):
                            # NOTE: Since this error can only be raised on the
                            # ranks that have padding, this can manifest as a NCCL
                            # watchdog timeout, as the other ranks will not error.
                            raise AssertionError(
                                "When a parameter is unevenly sharded by FSDP "
                                f"(orig size={self._orig_size}, FSDP world size={self.mesh_info.mesh.size()}), "
                                "fsdp_pre_all_gather must return all-gather inputs with the padded sharded size "
                                f"{self.padded_sharded_param_size} but got {[t.size() for t in all_gather_inputs]}"
                            )
                    self._extensions_data.all_gather_input_sizes = [
                        t.size() for t in all_gather_inputs
                    ]
                    return [t.view(-1) for t in all_gather_inputs]
                sharded_param_data = self._sharded_param_data
                if self.offload_to_cpu:
                    sharded_param_data = sharded_param_data.to(
                        self.device, non_blocking=True
                    )
                return [_to_dtype_if_needed(sharded_param_data, self.param_dtype)]
            elif self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
                if not compiled_autograd_enabled() and hasattr(
                    self._sharded_local_tensor, "fsdp_pre_all_gather"
                ):
                    raise NotImplementedError
                all_gather_input = _to_dtype_if_needed(
                    cast(torch.Tensor, self._sharded_post_forward_param_data),
                    self.param_dtype,
                )
                return [all_gather_input]
            return [torch.empty(0)]  # mypy

        @property
        def shard_mesh_from_root(self):
            mesh = self.mesh_info.mesh

            if mesh.ndim == 1:
                return mesh
            else:
                assert mesh.mesh_dim_names is not None
                shard_dim_name = mesh.mesh_dim_names[-1]

                root_mesh = _mesh_resources.get_root_mesh(mesh)
                return root_mesh[shard_dim_name]

        FSDPParam.all_gather_inputs = all_gather_inputs
        FSDPParam.shard_mesh_from_root = shard_mesh_from_root

    def convert(self, model: nn.Module):
        if not self.enabled:
            return

        return self.convert_to_bitnet_training(model)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        return self.precompute_bitnet_dynamic_scale_for_fsdp(model)

    def convert_to_bitnet_training(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `BitNetLinear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        self._patch_pytorch()

        from torchao import quantize_
        from torchao.prototype.quantized_training import bitnet_training

        # Mutates the model inplace replacing instances of nn.Linear with BitNetLinear
        quantize_(
            model.layers,
            bitnet_training(),
            set_inductor_config=False,
        )
        logger.info("Swapped to BitNetLinear layers")

    def precompute_bitnet_dynamic_scale_for_fsdp(
        self, model: nn.Module | list[nn.Module]
    ):
        from torchao.prototype.quantized_training import (
            precompute_bitnet_scale_for_fsdp,
        )

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_bitnet_scale_for_fsdp(m)


try:
    from torchao.prototype.quantized_training import BitNetTrainingLinearWeight
except ImportError:
    BitNetTrainingLinearWeight = None

if BitNetTrainingLinearWeight is not None:
    from torch.serialization import add_safe_globals

    # Allow serialization.
    add_safe_globals([BitNetTrainingLinearWeight])

register_model_converter(BitNetConverter, "bitnet")
