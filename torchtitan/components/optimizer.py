# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
import re
import warnings
from collections import OrderedDict
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.optimizers import DistributedScion, Scion
from torchtitan.tools.logging import logger

__all__ = [
    "OptimizersContainer",
    "build_optimizers",
    "build_optimizers_with_moe_load_balancing",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


@torch.no_grad()
def rms_to_rms_norm(W):
    """
    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_uniform_(w.T, ...)``.
    """
    assert W.ndim == 2, "operator norm can only be applied to matrices"
    norm = torch.linalg.norm(W.to(torch.float32), ord=2, dtype=torch.float32)
    fan_out, fan_in = W.shape
    scale = math.sqrt(fan_in / fan_out)
    norm *= scale
    return norm


@torch.no_grad()
def l1_to_rms_norm(W):
    assert W.ndim == 2, "operator norm can only be applied to matrices"
    norm = torch.max(
        torch.linalg.norm(W.to(torch.float32), ord=2, dim=0, dtype=torch.float32)
    )
    scale = torch.sqrt(torch.tensor(W.shape[0], dtype=W.dtype, device=W.device))
    norm /= scale
    return norm


@torch.no_grad()
def rms_to_l1_norm(W):
    assert W.ndim == 2, "operator norm can only be applied to matrices"
    norm = torch.max(
        torch.linalg.norm(W.to(torch.float32), ord=2, dim=1, dtype=torch.float32)
    )
    scale = torch.sqrt(torch.tensor(W.shape[1], dtype=W.dtype, device=W.device))
    norm *= scale
    return norm


@torch.no_grad()
def supremum_norm(x):
    return x.abs().max()


@torch.no_grad()
def condition_number(W):
    assert W.ndim == 2, "condition number calculation can only be applied to matrices"
    S = torch.linalg.svdvals(W.to(torch.float32), driver="gesvd")
    return S[0] / S[-1]


NORM_FUNCTIONS = {
    "rms_to_rms": rms_to_rms_norm,
    "l1_to_rms": l1_to_rms_norm,
    "rms_to_l1": rms_to_l1_norm,
    "supremum": supremum_norm,
    "condition_number": condition_number,
}


def _remove_orig_mod_and_weight_for_p_name(name: str) -> str:
    # Remove ._orig_mod and .weight anywhere in the parameter name
    name = re.sub(r"\._orig_mod", "", name)
    name = re.sub(r"\.weight", "", name)
    return name


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
            logger.warning(
                f'Notice: No parameters found for `str_match` "{str_match}" on '
                f"global rank {torch.distributed.get_rank()}"
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

    @staticmethod
    def compute_grad(p, optimizer=None, **kwargs):
        if isinstance(optimizer, (Scion, DistributedScion)):
            momentum = kwargs.pop("momentum")
            nesterov = kwargs.pop("nesterov")
            g = optimizer.get_momentum_or_grad(
                p,
                momentum,
                nesterov,
                update_buffer=False,
                gather_to_local=optimizer.fsdp_enabled,
            )
            if g is None:
                return None
            else:
                return optimizer.lmo(g, **kwargs)
        elif isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            if p.ndim == 3:
                warnings.warn(
                    f"Optimizer {optimizer.__class__.__name__} does not support "
                    f"gradient computation for 3D tensors for logging."
                )
                return None

            eps = kwargs["eps"]
            weight_decay = kwargs["weight_decay"]
            beta1, beta2 = kwargs["betas"]
            assert (
                weight_decay == 0.0
            ), "Weight decay not supported for grad computation."

            param_optim_state = optimizer.state[p]
            if "step" not in param_optim_state:
                step = 0
            else:
                step = param_optim_state["step"].item()
            if "exp_avg_sq" in param_optim_state and "exp_avg" in param_optim_state:
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (
                    param_optim_state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
                ) + eps
                step_size = 1 / bias_correction1
                g = step_size * param_optim_state["exp_avg"].div(denom)
            else:
                # TODO(JSC): if we shard the MoE model, we need to remove the following code
                g = p.grad

            assert isinstance(g, DTensor), "Expected gradient to be a DTensor"
            return g.redistribute(placements=[Replicate()] * g.device_mesh.ndim)
        else:
            raise TypeError(
                f"Optimizer {optimizer.__class__.__name__} does not support "
                f"gradient computation."
            )

    def get_parameter_norms(self):
        norms = {}
        for i, _ in enumerate(self.model_parts):
            # NB: assumes correspondences between model parts and optimizers
            optimizer = self.optimizers[i]
            for group in optimizer.param_groups:
                if isinstance(optimizer, (Scion, DistributedScion)):
                    param_kwargs = {
                        "momentum": group["momentum"],
                        "nesterov": group["nesterov"],
                        "eps": group["eps"],
                        "norm_factor": group["norm_factor"],
                        "zeropower_backend": group["backend"],
                        "backend_steps": group["backend_steps"],
                    }
                elif isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                    param_kwargs = {
                        "eps": group["eps"],
                        "betas": group["betas"],
                        "weight_decay": group["weight_decay"],
                    }
                else:
                    warnings.warn(
                        f"Optimizer {optimizer.__class__.__name__} does not support "
                        f"norm computation."
                    )
                    continue

                for p_name, p in zip(group["param_names"], group["params"]):

                    """
                    the module name usally named
                    track_update_condition_number/model_part_0/layers.0._orig_mod.attention.wo.weight
                    we can remove '._orig_mod' and '.weight' to get the clean layer name
                    """
                    cleaned_p_name = _remove_orig_mod_and_weight_for_p_name(p_name)
                    g = self.compute_grad(p, optimizer, **param_kwargs)
                    if g is not None:
                        p = (
                            p.redistribute(
                                placements=[Replicate()] * p.device_mesh.ndim,
                            ).to_local()
                            if isinstance(p, DTensor)
                            else p
                        )
                        g = g.to_local() if isinstance(g, DTensor) else g
                        update = -group["lr"] * g
                        if "tok_embeddings" in p_name:
                            p, update = p.T, update.T
                        for norm_name, norm_func in NORM_FUNCTIONS.items():
                            if norm_name != "supremum" and (
                                p.ndim < 2 or update.ndim < 2
                            ):
                                # Operator norms require a matrix.
                                continue
                            elif p.ndim == 3 or update.ndim == 3:
                                # Special handling for grouped MoE.
                                for ep_idx in range(p.shape[0]):
                                    norms[
                                        f"track_update_{norm_name}/model_part_{i}/ep_{ep_idx}/"
                                        f"{cleaned_p_name}"
                                    ] = norm_func(update[ep_idx])

                                    norms[
                                        f"track_param_{norm_name}/model_part_{i}/ep_{ep_idx}/"
                                        f"{cleaned_p_name}"
                                    ] = norm_func(p[ep_idx])

                            else:
                                if p.ndim > 2 or update.ndim > 2:
                                    warnings.warn(
                                        f"Encountered parameter or update {cleaned_p_name} with "
                                        f"shape {p.shape} or {update.shape}, respectively; "
                                        f"this may not be an issue, but please ensure its "
                                        f"norms are calculated correctly."
                                    )
                                norms[
                                    f"track_param_{norm_name}/model_part_{i}/{cleaned_p_name}"
                                ] = norm_func(p)
                                norms[
                                    f"track_update_{norm_name}/model_part_{i}/{cleaned_p_name}"
                                ] = norm_func(update)

        return norms

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
        use_ft_optimizer: bool = True,
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
        # Whether to determine quorum using FT.optimizer,
        # in semi-sync training we use the synchronization step to start quorum
        self._use_ft_optimizer: bool = use_ft_optimizer

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

        TorchFT's OptimizerWrapper.step() is designed to be called only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.step(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().zero_grad(*args, **kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``optimizer_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Optimizer config containing the optimizer name and parameters.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
    """
    optim_in_bwd = optimizer_config.early_step_in_backward
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

    extra_kwargs = extra_kwargs if extra_kwargs is not None else {}

    name = optimizer_config.name
    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    is_scion = name == "Scion" or name == "DistributedScion"

    width_multiplier = 1
    if name in ["Adam", "AdamW"]:
        optim_implementation = optimizer_config.implementation
        assert optim_implementation in ["fused", "foreach", "for-loop"]

        fused = optim_implementation == "fused"
        foreach = optim_implementation == "foreach"

        if parallel_dims.ep_enabled:
            # Because for Expert Parallel, we have two different device meshes.
            fused, foreach = False, False

        width_multiplier = optimizer_config.mup_width_multiplier

        optimizer_kwargs = {
            "lr": lr / width_multiplier,
            "betas": (beta1, beta2),
            "eps": eps / width_multiplier,
            "weight_decay": weight_decay
            * width_multiplier,  # WD is coupled with LR in torch AdamW
            "fused": fused,
            "foreach": foreach,
        }
    elif is_scion:
        backend_steps = optimizer_config.backend_steps
        zeropower_backend_algorithm = optimizer_config.zeropower_backend
        momentum = optimizer_config.momentum
        nesterov = optimizer_config.nesterov
        is_light = optimizer_config.is_light
        is_unconstrained = optimizer_config.is_unconstrained

        optimizer_kwargs = {
            "is_light": is_light,
            "is_unconstrained": is_unconstrained,
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "eps": eps,
            "norm_factor": "spectral",
            "backend": zeropower_backend_algorithm,
            "backend_steps": backend_steps,
        }
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")

    # Configure parameter group settings
    embed_lr = optimizer_config.embed_lr
    embed_str_match = optimizer_config.embed_str_match
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
    unembed_lr = optimizer_config.unembed_lr
    unembed_str_match = optimizer_config.unembed_str_match
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

    router_str_match = optimizer_config.router_str_match
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

    if optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )

    if ft_manager and ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
        )

    return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)


def build_optimizers_with_moe_load_balancing(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    optimizers = build_optimizers(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )

    # for MoE auxiliary-loss-free load balancing
    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        dp_cp_mesh = (
            parallel_dims.world_mesh["dp_cp"] if parallel_dims.dp_cp_enabled else None
        )
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        tokens_per_expert_list = []
        for model_part in model_parts:
            for transformer_block in model_part.layers.values():
                if not transformer_block.moe_enabled:
                    continue
                if transformer_block.moe.load_balance_coeff is None:
                    return
                tokens_per_expert = transformer_block.moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    # TODO: This is a hack, we assume with full AC, the tokens_per_expert is counted twice.
                    # This does not affect to expert choice, but affects the experts usage metrics.
                    # We divide by 2 to correct for this double-counting due to recomputation
                    # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_list.append(tokens_per_expert)

        tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

        if dp_cp_mesh is not None:
            # Perform single all-reduce to get global statistics across all processes
            pg = dp_cp_mesh.get_group()
            torch.distributed.all_reduce(
                tokens_per_expert_by_layer, group=pg, op=torch.distributed.ReduceOp.SUM
            )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                for transformer_block in model_part.layers.values():
                    if not transformer_block.moe_enabled:
                        continue
                    moe = transformer_block.moe

                    tokens_per_expert = tokens_per_expert_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # update the expert bias
                    # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        tokens_per_expert.mean() - tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    moe.expert_bias.add_(expert_bias_delta)
                    moe.tokens_per_expert.zero_()

    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: _update_expert_bias(
            model_parts, parallel_dims=parallel_dims
        )
    )

    return optimizers
