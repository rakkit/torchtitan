# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

from torchtitan.config import TORCH_DTYPE_MAP


logger = logging.getLogger(__name__)


def remove_orig_mod_and_weight_for_p_name(
    name: str,
    remove_orig_mod: bool = True,
    remove_weight: bool = True,
    remove_checkpoint_wrapped_module: bool = True,
) -> str:
    """
    Remove "._orig_mod", ".weight", and "._checkpoint_wrapped_module" to
    get the clean layer name.
    """
    if remove_orig_mod:
        name = re.sub(r"\._orig_mod", "", name)  # comes from compiled model
    if remove_weight:
        name = re.sub(r"\.weight", "", name)  # param.weight
    if remove_checkpoint_wrapped_module:
        name = re.sub(
            r"\._checkpoint_wrapped_module", "", name
        )  # comes from activation checkpointing
    return name


def disco_factory_kwargs(optimizer_config, parallelism_context) -> dict[str, Any]:
    """Constructor kwargs for `DiSCO`: the `default` group (what every group
    inherits) plus the non-config objects."""
    default = optimizer_config.groups.get("default", {})
    if default.get("optimizer", "DiSCO") == "DiSCO":
        kw = {k: v for k, v in default.items() if k not in ("match", "optimizer")}
    else:  # DiSCO is not the default optimizer: its groups must be self-contained
        kw = {}
    if optimizer_config.communication_dtype not in ("float32", "bfloat16"):
        raise ValueError(
            f"communication_dtype {optimizer_config.communication_dtype!r}: use "
            "float32 or bfloat16 (float16 flushes momentum entries of ~1e-7)"
        )
    kw.update(
        parallelism_context=parallelism_context,
        communication_dtype=TORCH_DTYPE_MAP[optimizer_config.communication_dtype],
        enable_step_tracing=optimizer_config.enable_step_tracing,
    )
    return kw
