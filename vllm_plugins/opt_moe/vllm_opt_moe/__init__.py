# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM plugin entry point for opt_moe (the ``vllm.general_plugins`` entry
point of ``pyproject.toml``): adds ``OptMoeForCausalLM`` (``model.py``) to vLLM's
model registry, the architecture an opt_moe HF folder's config.json names, and
its MTP depths as the speculative draft ``OptMoeMTPModel``
(``--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'``, as
for DeepSeek / Qwen)."""

import os
from pathlib import Path

from vllm import ModelRegistry
from vllm.config.speculative import SpeculativeConfig

MODEL_ARCH = "OptMoeForCausalLM"
MTP_ARCH = "OptMoeMTPModel"

_vllm_draft_hf_config = SpeculativeConfig.hf_config_override


def draft_hf_config(hf_config):
    """vLLM's MTP draft config (``SpeculativeConfig.hf_config_override``: the
    target's config with the draft's architecture), plus opt_moe's."""
    if hf_config.model_type != "opt_moe":
        return _vllm_draft_hf_config(hf_config)
    depths = hf_config.num_nextn_predict_layers
    if not depths:
        raise ValueError("this opt_moe checkpoint has no MTP depths to draft with")
    # vLLM takes "method": "mtp" only for a draft model_type it lists as MTP
    hf_config.model_type = "mtp"
    hf_config.update({"n_predict": depths, "architectures": [MTP_ARCH]})
    return hf_config


def register() -> None:
    ModelRegistry.register_model(MODEL_ARCH, f"{__name__}.model:OptMoeForCausalLM")
    ModelRegistry.register_model(MTP_ARCH, f"{__name__}.model:OptMoeMTP")
    # TODO: vLLM maps a target's model_type to its MTP draft architecture in
    # this static method only (no registry entry for out-of-tree models).
    SpeculativeConfig.hf_config_override = staticmethod(draft_hf_config)
    # fused-MoE kernel configs vLLM does not ship (GH200); vLLM reads this
    # folder before its own, and a user's folder takes precedence
    os.environ.setdefault(
        "VLLM_TUNED_CONFIG_FOLDER", str(Path(__file__).parent / "moe_configs")
    )


__all__ = ["MODEL_ARCH", "MTP_ARCH", "register"]
