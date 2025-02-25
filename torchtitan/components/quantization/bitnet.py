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
