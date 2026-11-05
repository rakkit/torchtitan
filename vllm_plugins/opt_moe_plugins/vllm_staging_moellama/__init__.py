# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM plugin entrypoint for StagingMoEllama."""

from vllm import ModelRegistry


MODEL_ARCH = "StagingMoEllamaForCausalLM"
MODEL_FQN = f"{__name__}.model:StagingMoEllamaForCausalLM"


def register() -> None:
    # Always overwrite so this session uses the currently imported package path.
    # This avoids stale registrations from an older installed plugin module.
    ModelRegistry.register_model(MODEL_ARCH, MODEL_FQN)


__all__ = ["register"]
