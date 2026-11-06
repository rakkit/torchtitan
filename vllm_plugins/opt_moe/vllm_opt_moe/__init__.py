# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM plugin entry point for opt_moe (the ``vllm.general_plugins`` entry
point of ``setup.py``): adds ``OptMoeForCausalLM`` (``model.py``) to vLLM's
model registry, the architecture an opt_moe HF folder's config.json names."""

from vllm import ModelRegistry

MODEL_ARCH = "OptMoeForCausalLM"
MODEL_FQN = f"{__name__}.model:OptMoeForCausalLM"


def register() -> None:
    ModelRegistry.register_model(MODEL_ARCH, MODEL_FQN)


__all__ = ["MODEL_ARCH", "register"]
