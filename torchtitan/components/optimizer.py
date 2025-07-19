# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import functools
import math
import re
from typing import Any, Generic, Iterator, TypeVar
import warnings
import os

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config_manager import JobConfig
from torchtitan.optimizers import DistributedScion, Scion, naive_param_norm
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color
from torchtitan.distributed import ParallelDims


__all__ = [
    "OptimizersContainer",
    "build_optimizers",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


def _extract_param_groups(
    model: torch.nn.Module,
    optimizer_config: dict[str, Any] | None = None,
):
    param_groups_config: list[dict[str, Any]] | None = (
        optimizer_config.pop("param_groups", None)
        if optimizer_config is not None
        else None
    )
    if param_groups_config is None:
        param_groups_config = []

    param_dict = OrderedDict(
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    )
    params = []

    for param_group_config in param_groups_config:
        str_match = param_group_config.pop("param_str_match")
        filter_fn = functools.partial(re.search, str_match)
        param_names = [n for n in param_dict.keys() if filter_fn(n)]
        group_params = {
            "params": [param_dict.pop(n) for n in param_names],
            "param_names": param_names,
        }
        assert len(group_params["params"]) == len(group_params["param_names"])

        if len(param_names) == 0:
            color = Color()
            logger.warning(
                f'{color.red}Notice: No parameters found for `str_match` "{str_match}" on '
                f"global rank {torch.distributed.get_rank()}{color.reset}"
            )
            continue
        group_params.update(param_group_config)
        params.append(group_params)

    param_names = list(param_dict.keys())
    params.insert(
        0,
        {
            "params": [param_dict.pop(n) for n in param_names],
            "param_names": param_names,
        },
    )
    assert not param_dict
    return params


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        param_groups_config = optimizer_kwargs.get("param_groups", None)
        # Whether to keep old LR values when loading.
        self.preserve_lrs_when_loading = False

        for model in self.model_parts:
            # copy parts we will pop from to preserve settings across model parts
            kwargs = optimizer_kwargs.copy()
            if "param_groups" in optimizer_kwargs:
                kwargs["param_groups"] = (
                    param_groups_config.copy()
                    if param_groups_config is not None
                    else None
                )

            extra_kwargs = kwargs.pop("extra_kwargs")
            params = _extract_param_groups(model, kwargs)
            is_scion = issubclass(optimizer_cls, (Scion, DistributedScion))
            if is_scion:
                kwargs.update(extra_kwargs)
            self.optimizers.append(optimizer_cls(params, **kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        # Do not separately save the external settings in
        # optimizer defaults.
        optimizer_kwargs.pop("param_groups", None)
        optimizer_kwargs.update(optimizer_kwargs.pop("extra_kwargs", {}))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            if not (
                isinstance(optimizer, (Scion, DistributedScion))
                and optimizer.is_light
                and optimizer.use_momentum
            ):
                optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.preserve_lrs_when_loading:
            # Store current learning rates
            prev_lrs = []
            for optimizer in self.optimizers:
                prev_lrs.append([group["lr"] for group in optimizer.param_groups])

        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

        if self.preserve_lrs_when_loading:
            # Restore the original learning rates
            for optimizer, optim_prev_lrs in zip(self.optimizers, prev_lrs):
                for param_group, prev_lr in zip(optimizer.param_groups, optim_prev_lrs):
                    if param_group["lr"] != prev_lr:
                        logger.warning(
                            f"Restoring lr from {param_group['lr']} to {prev_lr} | "
                            f"for {param_group['param_names']}"
                        )
                        param_group["lr"] = prev_lr

    def calculate_norm_at_next_step(self):
        # for Dist-scion, we tell the optimizer to calculate the norm at next step
        # in the step() function
        for i, _ in enumerate(self.model_parts):
            optimizer = self.optimizers[i]
            if isinstance(optimizer, DistributedScion):
                optimizer.calculate_norm_at_next_step()

    def get_parameter_norms(self):
        all_norms = {}
        for i, model_part in enumerate(self.model_parts):
            # NB: assumes correspondences between model parts and optimizers
            optimizer = self.optimizers[i]
            for group in optimizer.param_groups:
                if isinstance(optimizer, DistributedScion):
                    all_norms.update(optimizer.get_norms_at_current_step())
                else:
                    all_norms.update(
                        naive_param_norm.get_parameter_norms([model_part], [optimizer])
                    )
                # # To Debug, we can force using naive_param_norm
                # all_norms.update(
                #     naive_param_norm.get_parameter_norms([model_part], [optimizer])
                # )

        return all_norms

    def get_lrs(self):
        lrs = {}
        for i, optimizer in enumerate(self.optimizers):
            for k, group in enumerate(optimizer.param_groups):
                lrs[f"lr/opt_{i}/group_{k}"] = group["lr"]
        return lrs

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class FTOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        ft_manager: "ft.Manager",
    ) -> None:
        super().__init__(model_parts, optimizer_cls, optimizer_kwargs)

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        self._call_from_ft: bool = False

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Calling the correct step() depending on the caller.

        TorchFT's OptimizerWrapper.step() is designed to be callled only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._call_from_ft:
            super().step(*args, **kwargs)
        else:
            self._call_from_ft = True
            self._ft_optimizer.step(*args, **kwargs)
            self._call_from_ft = False

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._call_from_ft:
            super().zero_grad(*args, **kwargs)
        else:
            self._call_from_ft = True
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._call_from_ft = False


def build_optimizers(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager,
    extra_kwargs: dict[str, Any] = {},
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        job_config (JobConfig): Job config containing the optimizer name and parameters.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd:
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Expert Parallel."
            )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Pipeline Parallel."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with optimizers in backward."
            )
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    eps = job_config.optimizer.eps
    weight_decay = job_config.optimizer.weight_decay

    is_scion = name == "Scion" or name == "DistributedScion"

    width_multiplier = 1
    if name in ["Adam", "AdamW"]:
        optim_implementation = job_config.optimizer.implementation
        assert optim_implementation in ["fused", "foreach", "for-loop"]

        fused = optim_implementation == "fused"
        foreach = optim_implementation == "foreach"

        ep_enable = parallel_dims.ep_enabled
        if ep_enable:
            # Because for Expert Parallel, we have two different device meshes.
            fused, foreach = False, False

        width_multiplier = job_config.model.mup_width_multiplier
        # TODO Remove this deprecation handling at some point. Added on 2025-04-10.
        if "-multiplier-" in job_config.model.flavor:
            flavor_multiplier = int(job_config.model.flavor.split("-multiplier-")[-1])
            assert width_multiplier == flavor_multiplier, (
                "`--model.mup_width_multiplier` does not match multiplier specified in flavor. "
                "Please set `--model.mup_width_multiplier` to the μP multiplier; "
                "flavor parsing has been deprecated and this check will be removed in the future."
            )

        optimizer_kwargs = {
            "lr": lr / width_multiplier,
            "eps": eps / width_multiplier,
            "betas": (0.9, 0.95),
            "weight_decay": weight_decay
            * width_multiplier,  # WD is coupled with LR in torch AdamW
            "fused": fused,
            "foreach": foreach,
        }
    elif is_scion:
        backend_steps = job_config.optimizer.backend_steps
        zeropower_backend_algorithm = job_config.optimizer.zeropower_backend
        momentum = job_config.optimizer.momentum
        nesterov = job_config.optimizer.nesterov
        is_light = job_config.optimizer.is_light
        is_unconstrained = job_config.optimizer.is_unconstrained
        if os.environ.get("SCION_DEBUG_GRAD") == "1":
            # only if we wanna to debug the gradient
            # which we dont run SVD
            norm_factor = "none"
            zeropower_backend_algorithm = "identity"
            logger.warning(
                "SCION_DEBUG_GRAD is set to 1, we will not run SVD and use identity backend"
            )
        else:
            norm_factor = "spectral"

        optimizer_kwargs = {
            "is_light": is_light,
            "is_unconstrained": is_unconstrained,
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "eps": eps,
            "norm_factor": norm_factor,
            "backend": zeropower_backend_algorithm,
            "backend_steps": backend_steps,
        }
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")

    # Configure parameter group settings
    embed_lr = job_config.optimizer.embed_lr
    embed_str_match = job_config.optimizer.embed_str_match
    if embed_lr is not None and embed_str_match:
        param_groups_config = optimizer_kwargs.setdefault("param_groups", [])
        param_group_config = {
            "param_str_match": embed_str_match,
            "lr": embed_lr,
        }
        if is_scion:
            param_group_config["norm_factor"] = "embed_sqrt"
            param_group_config["backend"] = "identity"
        param_groups_config.append(param_group_config)
    unembed_lr = job_config.optimizer.unembed_lr
    unembed_str_match = job_config.optimizer.unembed_str_match
    if unembed_lr is not None and unembed_str_match:
        param_groups_config = optimizer_kwargs.setdefault("param_groups", [])
        param_group_config = {
            "param_str_match": unembed_str_match,
            "lr": unembed_lr / width_multiplier,
        }
        if is_scion:
            param_group_config["norm_factor"] = "unembed_sqrt"
            param_group_config["backend"] = "identity"
        param_groups_config.append(param_group_config)

    router_str_match = job_config.optimizer.router_str_match
    if router_str_match:
        param_groups_config = optimizer_kwargs.setdefault("param_groups", [])
        param_group_config = {
            "param_str_match": router_str_match,
            "lr": lr,
        }
        if is_scion:
            # param_group_config["norm_factor"] = "image_spectral"
            # param_group_config["backend"] = zeropower_backend_algorithm
            # param_group_config["norm_factor"] = "none"
            # param_group_config["backend"] = "identity"
            param_group_config["norm_factor"] = "spectral"
            param_group_config["backend"] = zeropower_backend_algorithm
        param_groups_config.append(param_group_config)

    ss_norm_str_match = "ssnorm_scale"
    if ss_norm_str_match:
        param_groups_config = optimizer_kwargs.setdefault("param_groups", [])
        param_group_config = {
            "param_str_match": ss_norm_str_match,
            "lr": lr,
        }
        if is_scion:
            param_group_config["norm_factor"] = "sign"
            param_group_config["backend"] = "identity"
        param_groups_config.append(param_group_config)

    optimizer_kwargs["extra_kwargs"] = {
        "parallel_dims": parallel_dims,
        **extra_kwargs,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "Scion": Scion,
        "DistributedScion": DistributedScion,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    elif optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )
    elif ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts, optimizer_cls, optimizer_kwargs, ft_manager.manager
        )
    else:
        return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
