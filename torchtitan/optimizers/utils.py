# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

import torch



def remove_orig_mod_and_weight_for_p_name(
    name: str,
    remove_orig_mod: bool = True,
    remove_weight: bool = True,
    remove_checkpoint_wrapped_module: bool = True,

logger = logging.getLogger(__name__)
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


def create_disco_optimizer_kwargs_from_optimizer_config(
    optimizer_config,
    parallel_dims,
) -> dict[str, Any]:
    backend_steps = optimizer_config.backend_steps
    zeropower_backend_algorithm = optimizer_config.zeropower_backend
    momentum = optimizer_config.momentum
    nesterov = optimizer_config.nesterov
    is_light = optimizer_config.is_light
    weight_decay = optimizer_config.weight_decay
    lr = optimizer_config.lr
    eps = optimizer_config.eps
    norm_factor = optimizer_config.norm_factor
    pre_norm = getattr(optimizer_config, "pre_norm", "identity")

    optimizer_kwargs = {
        "parallel_dims": parallel_dims,
        "is_light": is_light,
        "weight_decay": weight_decay,
        "lr": lr,
        "momentum": momentum,
        "nesterov": nesterov,
        "eps": eps,
        "norm_factor": norm_factor,
        "pre_norm": pre_norm,
        "backend": zeropower_backend_algorithm,
        "backend_steps": backend_steps,
    }

    # Add extra_param_group_split_rules if present
    if (
        hasattr(optimizer_config, "extra_param_group_split_rules")
        and optimizer_config.extra_param_group_split_rules
    ):
        optimizer_kwargs[
            "extra_param_group_split_rules"
        ] = optimizer_config.extra_param_group_split_rules

    return optimizer_kwargs


def create_disco_param_groups(
    model: torch.nn.Module,
    optimizer_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Create and extract parameter groups for DiSCO optimizer.

    This function combines parameter group configuration creation and parameter extraction:
    1. Creates parameter group configurations from optimizer_kwargs
    2. Extracts actual parameters from the model based on the configurations
    3. Returns clean kwargs for the optimizer (without extra_param_group_split_rules)

    This function supports a new parameter group configuration system where:
    - Default values are taken from the top-level optimizer_kwargs
    - Special parameter groups are defined in the 'extra_param_group_split_rules' list
    - Each entry in extra_param_group_split_rules can override any default value

    Example configuration:
    ```python
    optimizer_kwargs = {
        "lr": 1e-3,
        "weight_decay": 0.1,
        "momentum": 0.9,
        "norm_factor": "spectral",
        "backend": "newtonschulz5",
        "backend_steps": 5,
        "extra_param_group_split_rules": [
            {
                "str_match": "embedding",
                "lr": 1e-4,  # Override default lr
                "norm_factor": "embed_sqrt"  # Override default norm_factor
            },
            {
                "str_match": "router",
                "lr": 5e-4,
                "backend": "identity"  # Override default backend
            }
        ]
    }
    ```
    Args:
        model: The model to extract parameters from
        optimizer_kwargs: Dictionary containing optimizer configuration

    Returns:
        Tuple of (parameter_groups, clean_kwargs) where:
        - parameter_groups: List of parameter groups with actual parameters
        - clean_kwargs: Clean kwargs for the optimizer (without extra_param_group_split_rules)
    """
    import functools
    import re
    from collections import OrderedDict

    # Step 1: Create parameter group configurations
    param_groups_config = []

    # Get default configuration
    default_config = {
        "lr": optimizer_kwargs.get("lr"),
        "weight_decay": optimizer_kwargs.get("weight_decay"),
        "momentum": optimizer_kwargs.get("momentum"),
        "nesterov": optimizer_kwargs.get("nesterov"),
        "eps": optimizer_kwargs.get("eps"),
        "norm_factor": optimizer_kwargs.get("norm_factor"),
        "pre_norm": optimizer_kwargs.get("pre_norm", "identity"),
        "backend": optimizer_kwargs.get("backend"),
        "backend_steps": optimizer_kwargs.get("backend_steps"),
        # should be explicitly set in the extra_param_group_split_rules
        "splits_into": None,
        "splits_dim": None,
    }

    # Process extra_param_group_split_rules if provided
    extra_param_group_split_rules = optimizer_kwargs.get(
        "extra_param_group_split_rules", []
    )
    for param_group in extra_param_group_split_rules:
        # Start with default config and override with param_group specific values
        group_config = default_config.copy()
        group_config.update(param_group)

        # Ensure str_match is present
        if "str_match" not in group_config:
            logger.warning(
                "extra_param_group_split_rules entry missing 'str_match', skipping"
            )
            continue

        # Rename str_match to param_str_match for compatibility
        group_config["param_str_match"] = group_config.pop("str_match")

        # if one config's norm_factor is with unembed*,
        # we could by default set backend to be identity
        if (
            group_config["norm_factor"].startswith("unembed")
            or group_config["norm_factor"].startswith("sign")
        ) and group_config["backend"] != "identity":
            group_config["backend"] = "identity"
            logger.info(
                f"[DISCO][init], For {group_config['param_str_match']},"
                f"Setting backend from {group_config['backend']} to identity"
            )

        # otherwise, if norm_factor is [spectral, image_spectral, conv_spectral],
        # the backend should not be identity
        if (
            group_config["norm_factor"]
            in ["spectral", "image_spectral", "conv_spectral"]
            and group_config["backend"] == "identity"
        ):
            raise ValueError(
                f"For {group_config['param_str_match']}, norm_factor is {group_config['norm_factor']}, "
                f"but backend is identity, which is not allowed"
            )

        param_groups_config.append(group_config)

    # Step 2: Extract actual parameters from the model
    param_dict = OrderedDict(
        (remove_orig_mod_and_weight_for_p_name(n, remove_weight=False), p)
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    params = []

    def _legacy_name(name: str) -> str:
        """Pre-0.5.0 spelling of a parameter name.

        Upstream renamed the decoder's unembedding module from ``output`` to
        ``lm_head``. Thousands of recorded configs in reproduce_cfgs/ and
        launch_scripts/ match on ``output.weight`` to give the unembedding its
        own norm factor, and a rule that silently matches nothing would quietly
        train that tensor under the default spectral norm factor instead.
        Matching against both spellings keeps those configs correct without
        rewriting them, and lets new configs use either name.
        """
        return name.replace("lm_head", "output")

    for param_group_config in param_groups_config:
        # Make a copy to avoid modifying the original
        group_config = param_group_config.copy()
        str_match = group_config.pop("param_str_match")
        filter_fn = functools.partial(re.search, str_match)
        param_names = [
            n for n in param_dict.keys() if filter_fn(n) or filter_fn(_legacy_name(n))
        ]

        group_params = {
            "params": [param_dict.pop(n) for n in param_names],
            "param_names": param_names,
        }
        assert len(group_params["params"]) == len(group_params["param_names"])

        if len(param_names) == 0:
            # WARN, not raise. Upstream raises here, but this fork has per-flavor
            # OPTIONAL modules, so a rule can legitimately match nothing: e.g.
            # `\.attention.gate_proj.weight` matches zero params on any flavor
            # with `gated_attention_type=None`, because gate_proj is an Identity
            # with no parameters -- and three live recipes carry exactly that
            # rule. A raise broke them.
            #
            # The risk this warning covers is real though: unmatched params stay
            # in the DEFAULT group, i.e. trained with a different norm factor
            # than the recipe asked for. That is what the `_legacy_name` shim
            # exists to prevent for the 0.5.0 `output.weight` -> `lm_head`
            # rename. If you see this for a rule you expect to match, check for
            # a renamed FQN before assuming the module is simply absent.
            logger.warning(
                "\033[31mNo parameters matched optimizer param-group rule "
                '`str_match` "%s" on global rank %s. Those parameters (if any) '
                "stay in the default group with its norm factor. This is "
                "expected when the module is absent for this flavor (e.g. "
                "gate_proj under gated_attention_type=None), and a BUG if the "
                "FQN was renamed.\033[0m",
                str_match,
                torch.distributed.get_rank(),
            )
            continue
        group_params.update(group_config)
        params.append(group_params)

    # Add remaining parameters as the default group
    param_names = list(param_dict.keys())
    if param_names:
        default_group = {
            "params": [param_dict.pop(n) for n in param_names],
            "param_names": param_names,
        }
        # Add default configuration to the default group
        default_group.update(default_config)
        params.insert(0, default_group)

    # Create clean kwargs for the optimizer (remove extra_param_group_split_rules)
    clean_kwargs = optimizer_kwargs.copy()
    clean_kwargs.pop("extra_param_group_split_rules", None)

    return params, clean_kwargs
