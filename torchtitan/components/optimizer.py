# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import queue
import threading
from typing import Any, Callable, Generic, Iterator, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.optimizers import (
    create_disco_optimizer_kwargs_from_optimizer_config,
    create_disco_param_groups,
    DiSCO,
    naive_param_norm,
)
from torchtitan.tools.logging import logger


__all__ = [
    "OptimizersContainer",
    "build_optimizers",
    "build_optimizers_with_moe_load_balancing",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


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
        # Whether to keep old LR values when loading.
        self.preserve_lrs_when_loading = False
        self.norms_to_log: list[str] | None = None
        self.log_queue: queue.Queue | None = None
        self.log_thread: threading.Thread | None = None

        for model in self.model_parts:
            if issubclass(optimizer_cls, DiSCO):
                params, optimizer_kwargs = create_disco_param_groups(
                    model, optimizer_kwargs
                )
            else:
                params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    # pyrefly: ignore [bad-override]
    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
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
        # for Disco, we tell the optimizer to calculate the norm at next step
        # in the step() function
        for i, _ in enumerate(self.model_parts):
            optimizer = self.optimizers[i]
            if isinstance(optimizer, DiSCO):
                optimizer.calculate_norm_at_next_step(self.norms_to_log)

    def get_parameter_norms(self):
        all_norms = {}
        for i, model_part in enumerate(self.model_parts):
            # NB: assumes correspondences between model parts and optimizers
            optimizer = self.optimizers[i]
            for group in optimizer.param_groups:
                if isinstance(optimizer, DiSCO):
                    all_norms.update(optimizer.get_norms_at_current_step())
                else:
                    all_norms.update(
                        naive_param_norm.get_parameter_norms(
                            [model_part],
                            [optimizer],
                            self.norms_to_log,
                        )
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

    def set_up_async_logging(self, log_fn: Callable):
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=log_fn, args=(self.log_queue,))
        self.log_thread.start()
        return self.log_queue

    def close(self):
        if self.log_queue is not None:
            self.log_queue.put(None)
        if self.log_thread is not None:
            self.log_thread.join()

    def join_log_queue(self):
        if self.log_queue is not None:
            self.log_queue.join()

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class."""
        pass


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

    # pyrefly: ignore [bad-override]
    def step(self) -> None:
        pass

    # pyrefly: ignore [bad-override]
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

    name = optimizer_config.name
    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    width_multiplier = 1
    if name in ["Adam", "AdamW"]:
        optim_implementation = optimizer_config.implementation
        assert optim_implementation in ["fused", "foreach", "for-loop"]

        fused = optim_implementation == "fused"
        foreach = optim_implementation == "foreach"

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
    elif name in ["DiSCO"]:
        optimizer_kwargs = create_disco_optimizer_kwargs_from_optimizer_config(
            optimizer_config, parallel_dims
        )
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "DiSCO": DiSCO,
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


def moe_metrics_worker(log_queue: queue.Queue):
    """
    This function runs in the background. It waits for data,
    does the slow CPU work, and assigns the final dictionary.
    """
    while True:
        # 1. Wait for data from the main thread
        data = log_queue.get()
        if data is None:  # Sentinel to stop the thread
            break

        (
            moe_layers_info,
            usage_t,
            bias_t,
            ent_t,
            lb_t,
            maxvio_batch_t,
            maxvio_global_t,
            num_experts,
            cuda_event,
        ) = data
        cuda_event.synchronize()
        all_usages_cpu = usage_t.tolist()
        all_biases_cpu = bias_t.tolist()
        all_entropies_cpu = ent_t.tolist()
        all_load_balance_losses_cpu = lb_t.tolist()
        all_maxvio_batch_cpu = maxvio_batch_t.tolist()
        all_maxvio_global_cpu = maxvio_global_t.tolist()

        usage_offset = bias_offset = 0
        for i, info in enumerate(moe_layers_info):
            moe = info["module"]
            layer_id = info["layer_id"]

            metrics = {
                f"moe_entropy/L-{layer_id}": all_entropies_cpu[i],
                f"moe_maxvio_batch/L-{layer_id}": all_maxvio_batch_cpu[i],
                f"moe_maxvio_global/L-{layer_id}": all_maxvio_global_cpu[i],
            }
            layer_usages = all_usages_cpu[usage_offset : usage_offset + num_experts]
            layer_biases = all_biases_cpu[bias_offset : bias_offset + num_experts]

            pre_usage = f"moe_ep_usage/L-{layer_id}_EP-"
            pre_bias = f"moe_bias/L-{layer_id}_EP-"
            metrics.update({f"{pre_usage}{j}": v for j, v in enumerate(layer_usages)})
            metrics.update({f"{pre_bias}{j}": v for j, v in enumerate(layer_biases)})
            metrics.update(
                {
                    f"moe_load_balance_loss/L-{layer_id}": v
                    for v in all_load_balance_losses_cpu
                }
            )
            moe._log_expert_metrics = metrics
            usage_offset += num_experts
            bias_offset += num_experts

        # Aggregated scalars across all MoE layers — attached to first layer
        num_moe_layers = len(moe_layers_info)
        metrics.update(
            {
                "moe_maxvio_batch/aggregate": sum(all_maxvio_batch_cpu)
                / num_moe_layers,
                "moe_maxvio_global/aggregate": sum(all_maxvio_global_cpu)
                / num_moe_layers,
            }
        )

        log_queue.task_done()


def fused_hier_reduce_loss_stats(
    parallel_dims,
    all_tokens: torch.Tensor,
    all_entropies: torch.Tensor,
    all_load_balance_losses: torch.Tensor,
):
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    if loss_mesh is None:
        return

    # 1. Determine Topology
    fsdp_mesh = parallel_dims.get_optional_mesh("fsdp")
    dp_mesh = parallel_dims.get_optional_mesh("dp_replicate")

    # Check if we can do hierarchical reduction
    use_hierarchical = (fsdp_mesh is not None) and (dp_mesh is not None)

    # 2. Fuse & Pack (Float64)
    t0 = all_tokens.reshape(-1).to(torch.float64)
    t1 = all_entropies.reshape(-1).to(torch.float64)
    t2 = all_load_balance_losses.reshape(-1).to(torch.float64)

    buf = torch.cat([t0, t1, t2])

    # 3. Perform Reduction
    if use_hierarchical:
        # Hierarchical: FSDP (Intra-node) -> DP (Inter-node)
        # Using SUM for all, we will normalize averaging later
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=fsdp_mesh.get_group())
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
    else:
        # Fallback: Flat all-reduce on the global loss mesh
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=loss_mesh.get_group())

    # 4. Unpack & Normalize
    ws_loss = dist.get_world_size(group=loss_mesh.get_group())

    n0, n1, n2 = t0.numel(), t1.numel(), t2.numel()

    # Slicing views
    out_tokens = buf[0:n0].view_as(all_tokens)
    out_ent = buf[n0 : n0 + n1].view_as(all_entropies)
    out_lb = buf[n0 + n1 : n0 + n1 + n2].view_as(all_load_balance_losses)

    # 5. Copy back to inputs
    # Tokens: SUM (no division)
    all_tokens.copy_(out_tokens.to(all_tokens.dtype))

    # Stats: AVG (Divide SUM by world_size)
    # We do the division *after* unpacking to keep the buffer operations clean
    all_entropies.copy_((out_ent / ws_loss).to(all_entropies.dtype))
    all_load_balance_losses.copy_((out_lb / ws_loss).to(all_load_balance_losses.dtype))


def lmo_for_moe_bias(
    g,
    norm_factor="sign",
    epsilon=1e-32,
):
    if norm_factor in ["sign", "sign_zero_mean"]:
        return torch.sign(g)
    elif norm_factor in ["spectral", "spectral_zero_mean"]:
        is_flat = g.dim() == 1
        g = g.unsqueeze(0) if is_flat else g
        norms = torch.linalg.norm(g, ord=2, dim=1, keepdim=True)
        g = g / torch.clamp(norms, min=epsilon)
        g = g.squeeze(0) if is_flat else g
        return g
    elif norm_factor in ["rms", "rms_zero_mean"]:
        is_flat = g.dim() == 1
        g = g.unsqueeze(0) if is_flat else g
        rms = torch.sqrt(torch.mean(g.square(), dim=1, keepdim=True))
        g = g / torch.clamp(rms, min=epsilon)
        g = g.squeeze(0) if is_flat else g
        return g


def need_rescale_stats(module):
    return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT


# for MoE auxiliary-loss-free load balancing
def _is_recomputation_enabled(module):
    return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT


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

    log_queue = optimizers.set_up_async_logging(moe_metrics_worker)

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            for transformer_block in model_part.layers.values():
                if transformer_block.moe_enabled:
                    return True
        return False

    # if _should_register_moe_balancing_hook(model_parts):
    #     optimizers.register_step_pre_hook(
    #         lambda *args, **kwargs: _update_expert_bias(
    #             model_parts, parallel_dims=parallel_dims
    #         )
    #     )

    if _should_register_moe_balancing_hook(model_parts):
        # ------------------------------------------------------------------ #
        # P1 / P2: one-time pre-computation at registration time.             #
        # We discover MoE layers once and pre-allocate / pre-compute the      #
        # tensors that were previously re-created on every optimizer step.    #
        # ------------------------------------------------------------------ #

        # Collect stable buffer references (values change per step; tensors
        # are fixed after model build — safe as closure-captured variables).
        _tok_bufs: list[torch.Tensor] = []
        _ent_bufs: list[torch.Tensor] = []
        _lb_bufs: list[torch.Tensor] = []
        _acc_bufs: list[torch.Tensor] = []
        _cumul_bufs: list[torch.Tensor] = []
        _moe_layers_static: list[dict] = []
        for _part in model_parts:
            for _block in _part.layers.values():
                if not _block.moe_enabled:
                    continue
                _moe = _block.moe
                _moe_layers_static.append({"module": _moe, "layer_id": _block.layer_id})
                _tok_bufs.append(_moe.tokens_per_expert)
                _cumul_bufs.append(_moe.tokens_per_expert_cumul)
                _ent_bufs.append(_moe.router_entropy)
                _lb_bufs.append(_moe.load_balance_loss)
                _acc_bufs.append(_moe.acc_fwd_times)

        _num_experts: int = _tok_bufs[0].numel()
        _num_layers: int = len(_moe_layers_static)
        _device: torch.device = _tok_bufs[0].device

        # P1: Pre-allocate flat concatenation buffers.
        # Each step we fill them in-place via _foreach_copy_ instead of
        # allocating new tensors with torch.cat (eliminates 3 allocs/step).
        _buf_tokens = torch.empty(
            _num_layers * _num_experts, device=_device, dtype=_tok_bufs[0].dtype
        )
        _buf_ent = torch.empty(
            _num_layers * _num_experts, device=_device, dtype=_ent_bufs[0].dtype
        )
        _buf_lb = torch.empty(
            _num_layers * _num_experts, device=_device, dtype=_lb_bufs[0].dtype
        )
        # Per-layer views used by _foreach_copy_ to fill slots in one dispatch.
        _E = _num_experts
        _buf_views_tok = [
            _buf_tokens[i * _E : (i + 1) * _E] for i in range(_num_layers)
        ]
        _buf_views_ent = [_buf_ent[i * _E : (i + 1) * _E] for i in range(_num_layers)]
        _buf_views_lb = [_buf_lb[i * _E : (i + 1) * _E] for i in range(_num_layers)]

        # P2: Pre-compute static index / reduction tensors.
        # grp is the same every step: [0,0,...,0, 1,1,...,1, ..., L-1,...,L-1]
        # with _num_experts copies of each layer index.
        _grp = torch.repeat_interleave(
            torch.arange(_num_layers, device=_device, dtype=torch.long), _E
        )
        # Scratch buffer for per-layer token sums; reused in-place each step.
        _layer_sums_buf = torch.zeros(
            _num_layers, device=_device, dtype=_tok_bufs[0].dtype
        )

        # Evaluate rank once at registration (rank never changes).
        _loss_mesh = parallel_dims.get_optional_mesh("loss")
        _is_dp_rank_0: bool = (
            torch.distributed.get_rank(_loss_mesh.get_group()) == 0
            if _loss_mesh is not None
            else True
        )

        # P4: Dedicated side stream for bias compute + stats reset.
        # After fused_hier_reduce_loss_stats() completes on the default stream,
        # we switch to _bias_stream so that optimizer.step() can proceed on the
        # default stream concurrently.  The training loop waits for the
        # _optim_pre_hook_ev before the NEXT forward pass to ensure expert_bias
        # is updated and router stats are zeroed before they are read/written.
        # Safety: fused_hier_reduce_loss_stats uses dist.all_reduce with FSDP /
        # DP communicators that are also used by DiSCO's optimizer.  We keep
        # the NCCL call on the default stream to avoid multi-stream collectives
        # on shared communicators (which can deadlock).  Only the pure-GPU
        # compute after NCCL moves to _bias_stream.
        _bias_stream = torch.cuda.Stream()
        optimizers._optim_pre_hook_ev = None  # written by hook, read by train loop

        def _update_expert_bias_fast(*args, **kwargs) -> None:
            """
            Optimised replacement for _update_expert_bias.

            Changes vs. the original:
              P1 – fills pre-allocated flat buffers via _foreach_copy_ instead
                   of calling torch.cat(tok_buffers / ent_buffers / lb_buffers).
              P2 – reuses pre-computed _grp and _layer_sums_buf instead of
                   re-creating them with torch.full / arange / repeat_interleave
                   / zeros on every step.
              P3 – non-blocking D-H copies + CUDA event; the background worker
                   syncs the event before calling .tolist().
              P4 – bias compute + stats reset + logging enqueued on _bias_stream
                   so the hook returns before they complete, letting
                   optimizer.step() run concurrently on the default stream.
            """
            # P1: Fill pre-allocated buffers in-place (no torch.cat alloc).
            # Must run on the default stream so it sees the latest values from
            # the backward pass before we fork to _bias_stream.
            torch._foreach_copy_(_buf_views_tok, _tok_bufs)
            torch._foreach_copy_(_buf_views_ent, _ent_bufs)
            torch._foreach_copy_(_buf_views_lb, _lb_bufs)

            all_tokens = _buf_tokens
            all_entropies = _buf_ent
            all_load_balance_losses = _buf_lb

            scale_factor = _acc_bufs[-1]
            if scale_factor != 1:
                # Rare path (gradient accumulation > 1): must allocate new
                # tensors because dtype/value changes from integer division.
                all_tokens = all_tokens // scale_factor
                all_entropies = all_entropies / scale_factor

            # NCCL stays on the default stream to avoid sharing communicators
            # across streams (P4 safety constraint — see comment above).
            if _loss_mesh is not None:
                fused_hier_reduce_loss_stats(
                    parallel_dims,
                    all_tokens,
                    all_entropies,
                    all_load_balance_losses,
                )

            # P4: Fork to _bias_stream for all post-NCCL GPU compute.
            # _bias_stream.wait_stream ensures it sees the NCCL result and the
            # filled buffers before starting.  The hook then returns immediately
            # so optimizer.step() can proceed on the default stream.
            _bias_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_bias_stream):
                # P2: Reuse pre-allocated scratch buffer for per-layer sums.
                _layer_sums_buf.zero_()
                _layer_sums_buf.index_add_(0, _grp, all_tokens)
                layer_means = _layer_sums_buf / _num_experts

                # MaxVio: accumulate global counts, then compute batch and global
                _tok_split = list(all_tokens.view(_num_layers, _num_experts).unbind(0))
                torch._foreach_add_(_cumul_bufs, _tok_split)
                _layer_means_f = layer_means.float()
                _max_per_layer = (
                    all_tokens.view(_num_layers, _num_experts).max(dim=1).values.float()
                )
                maxvio_batch = (_max_per_layer - _layer_means_f) / _layer_means_f.clamp(
                    min=1.0
                )
                _all_cumul_f = (
                    torch.cat(_cumul_bufs).view(_num_layers, _num_experts).float()
                )
                _cumul_means = _all_cumul_f.sum(dim=1) / _num_experts
                maxvio_global = (
                    _all_cumul_f.max(dim=1).values - _cumul_means
                ) / _cumul_means.clamp(min=1.0)

                delta_flat = layer_means[_grp] - all_tokens
                recip = torch.clamp(_layer_sums_buf, min=1.0).reciprocal()
                usage_flat = all_tokens * recip[_grp]

                with torch.no_grad():
                    first_moe = _moe_layers_static[0]["module"]
                    norm_factor = first_moe.bias_update_norm_factor
                    load_balance_coeff = first_moe.load_balance_coeff

                    delta_2d = delta_flat.view(_num_layers, _num_experts)
                    updates_2d = lmo_for_moe_bias(delta_2d, norm_factor=norm_factor)

                    if norm_factor.endswith("zero_mean"):
                        updates_2d = updates_2d - updates_2d.mean(dim=1, keepdim=True)

                    bias_params = [
                        info["module"].expert_bias for info in _moe_layers_static
                    ]
                    updates_list = list(updates_2d.flatten().split(_num_experts))
                    torch._foreach_add_(
                        bias_params, updates_list, alpha=load_balance_coeff
                    )

                    # Reset router stats in bulk.
                    try:
                        torch._foreach_mul_(_tok_bufs, 0)
                        torch._foreach_mul_(_ent_bufs, 0.0)
                        torch._foreach_mul_(_acc_bufs, 0)
                        torch._foreach_mul_(_lb_bufs, 0.0)
                    except Exception:
                        for t in _tok_bufs:
                            t.zero_()
                        for t in _ent_bufs:
                            t.zero_()
                        for t in _acc_bufs:
                            t.zero_()
                        for t in _lb_bufs:
                            t.zero_()

                    if _is_dp_rank_0:
                        # P3: Non-blocking D-H copies — enqueue DMA transfers
                        # and record a CUDA event.  moe_metrics_worker syncs
                        # the event before calling .tolist().
                        usage_t = usage_flat.to("cpu", non_blocking=True)
                        bias_t = torch.cat(bias_params).to("cpu", non_blocking=True)
                        ent_t = all_entropies.to(
                            dtype=torch.float32, device="cpu", non_blocking=True
                        )
                        lb_t = all_load_balance_losses.to(
                            dtype=torch.float32, device="cpu", non_blocking=True
                        )
                        maxvio_batch_t = maxvio_batch.to("cpu", non_blocking=True)
                        maxvio_global_t = maxvio_global.to("cpu", non_blocking=True)
                        log_cuda_event = torch.cuda.Event()
                        log_cuda_event.record()
                        payload = (
                            _moe_layers_static,
                            usage_t,
                            bias_t,
                            ent_t,
                            lb_t,
                            maxvio_batch_t,
                            maxvio_global_t,
                            _num_experts,
                            log_cuda_event,
                        )
                        log_queue.put(payload)

                # Record the event that signals all bias/reset work is done.
                # The training loop waits for this before the next forward pass.
                _optim_pre_hook_ev = torch.cuda.Event()
                _optim_pre_hook_ev.record()

            # Expose the event so the training loop can sync on it.
            optimizers._optim_pre_hook_ev = _optim_pre_hook_ev
            # Return without waiting — optimizer.step() proceeds immediately.

        optimizers.register_step_pre_hook(_update_expert_bias_fast)
    return optimizers
