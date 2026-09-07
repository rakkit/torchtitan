# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import torch
import torch.distributed as dist
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.optimizers import (
    create_disco_optimizer_kwargs_from_optimizer_config,
    create_disco_param_groups,
    DiSCO,
    spectrum_logging,
)
from torchtitan.optimizers.gram_vector_logging import (
    GramVectorLoggingConfig,
    process_gram_vectors_for_logging,
)
from torchtitan.optimizers.spectrum_logging import process_norms_for_logging
from torchtitan.tools.logging import logger

__all__ = [
    "OptimizersContainer",
    "OptimizersInBackwardContainer",
    "register_moe_load_balancing_hook",
]

MAXVIO_EMA_BETA = 0.995
MAXVIO_EPS = 1e-12


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T]):
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

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        name: str = "AdamW"
        """Optimizer to use"""

        lr: float = 8e-4
        """Learning rate to use"""

        beta1: float = 0.9
        beta2: float = 0.95
        """Exponential moving average hyperparameters to use"""

        eps: float = 1e-8
        """Epsilon value to use"""

        weight_decay: float = 0.1
        """Weight decay to use"""

        mup_width_multiplier: float = 1.0
        """
        Width multiplier for the model to apply μP scaling (only used
        for Adam/Muon-based optimizers).
        """

        is_light: bool = False
        """Whether to use Scion's light (memory-saving) version"""

        norm_factor: str = "spectral"
        """Which norm factor to use"""

        pre_norm: str = "identity"
        """
        Pre-norm applied to the effective gradient before any communication
        for LMO. "identity" is a no-op. Prefix before the first "-" selects
        the variant: "row-*" (local, no comm), "col-*"/"mat-*" (one fused
        all-reduce across FSDP-sharded params using this pre_norm that
        step). See optimizers/pre_norm_helper.py.
        """

        zeropower_backend: str = "newtonschulz5"
        "Which `zeropower_backend` to use."

        backend_steps: int = 5
        """Number of steps for the DiSCO backend"""

        momentum: float = 0.95
        """DiSCO momentum to use"""

        nesterov: bool = False
        """Whether to use Nesterov momentum in DiSCO"""

        extra_param_group_split_rules: list[dict[str, Any]] = field(
            default_factory=list
        )
        """Extra parameter group splitting rules for DiSCO optimizers"""

        implementation: Literal["for-loop", "foreach", "fused"] = "fused"
        """
        Specify which optimizer implementation to use:
        - 'fused': Use fused implementation (CUDA only) for best performance.
        - 'foreach': Use some horizontal fusion of tensors for better performance.
        - 'for-loop': Use the default implementation for the optimizer (slowest).
        - more info: https://pytorch.org/docs/stable/optim.html
        """

        enable_spectrum_plot: bool = False
        """
        Whether to render each tracked parameter's singular-value spectrum as
        a plot image for W&B (see optimizers/spectrum_logging.py). Only
        takes effect when metrics.log_norm_freq > 0.
        """

        enable_spectrum_export: bool = False
        """
        Whether to export the full raw singular-value spectra (every tracked
        parameter, every norm-logging step) to a Parquet file uploaded as a
        versioned W&B Artifact, for offline/programmatic analysis beyond what
        the plot shows. Off by default: can be large for big MoE models (see
        optimizers/spectrum_logging.py). Only takes effect when
        metrics.log_norm_freq > 0.
        """

        enable_gram_plot: bool = False
        """
        Whether to render each tracked vector-valued gram metric as an
        atlas-grid plot image for W&B (see optimizers/gram_vector_logging.py).
        Only takes effect when metrics.gram_level > 0.
        """

        enable_gram_export: bool = False
        """
        Whether to export every tracked gram vector (every gram-tracking
        step) to a Parquet file uploaded as a versioned W&B Artifact, for
        offline/programmatic analysis beyond what the plot shows. Off by
        default: can be large, there are far more distinct gram vector
        metrics than spectrum has (see optimizers/gram_vector_logging.py).
        Only takes effect when metrics.gram_level > 0.
        """

    optimizers: list[T]
    model_parts: list[nn.Module]

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        optimizer_classes = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "DiSCO": DiSCO,
        }
        if name not in optimizer_classes:
            raise NotImplementedError(f"Optimizer {name} not added.")
        return optimizer_classes[name]

    @staticmethod
    def _build_optimizer_kwargs(
        config: Config, parallel_dims: ParallelDims
    ) -> dict[str, Any]:
        name = config.name
        if name in ["Adam", "AdamW"]:
            optim_implementation = config.implementation
            assert optim_implementation in ["fused", "foreach", "for-loop"]

            width_multiplier = config.mup_width_multiplier

            optimizer_kwargs = {
                "lr": config.lr / width_multiplier,
                "betas": (config.beta1, config.beta2),
                "eps": config.eps / width_multiplier,
                "weight_decay": config.weight_decay
                * width_multiplier,  # WD is coupled with LR in torch AdamW
                "fused": config.implementation == "fused",
                "foreach": config.implementation == "foreach",
            }
        elif name in ["DiSCO"]:
            optimizer_kwargs = create_disco_optimizer_kwargs_from_optimizer_config(
                config, parallel_dims
            )
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer_kwargs

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ) -> None:
        optimizer_cls = self._resolve_optimizer_cls(config.name)
        optimizer_kwargs = self._build_optimizer_kwargs(config, parallel_dims)
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        self.preserve_lrs_when_loading = False
        self.norms_to_log: list[str] | None = None
        self.gram_level: int = 0
        self.log_queue: queue.Queue | None = None
        self.log_thread: threading.Thread | None = None
        # Set by the trainer from MetricsProcessor.Config.save_all_shard_ranks;
        # False keeps the historical gather-to-one-rank behaviour.
        self.log_metrics_locally: bool = False
        self.spectrum_logging_config = spectrum_logging.SpectrumLoggingConfig(
            enable_plot=config.enable_spectrum_plot,
            enable_export=config.enable_spectrum_export,
        )
        self.gram_vector_logging_config = GramVectorLoggingConfig(
            enable_plot=config.enable_gram_plot,
            enable_export=config.enable_gram_export,
        )

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
                optimizer.calculate_norm_at_next_step(
                    self.norms_to_log,
                    self.gram_level,
                    # The spectra are only ever consumed by
                    # process_norms_for_logging, which drops them unless one of
                    # these is on. Telling the optimizer up front lets it skip
                    # computing, packing and all-gathering them entirely.
                    track_spectrum=(
                        self.spectrum_logging_config.enable_plot
                        or self.spectrum_logging_config.enable_export
                    ),
                    # Single source of truth with the logger: the optimizer must
                    # only skip the gather on ranks that actually have a logger,
                    # otherwise their metrics are computed and silently dropped.
                    log_metrics_locally=self.log_metrics_locally,
                )

    def get_parameter_norms(self, step: int):
        all_norms = {}
        for i, model_part in enumerate(self.model_parts):
            # NB: assumes correspondences between model parts and optimizers
            optimizer = self.optimizers[i]
            for group in optimizer.param_groups:
                if isinstance(optimizer, DiSCO):
                    all_norms.update(optimizer.get_norms_at_current_step())
                else:
                    logger.warning(
                        f"Optimizer {optimizer.__class__.__name__} does not support norm calculation."
                    )
                    # all_norms.update(
                    #     naive_param_norm.get_parameter_norms(
                    #         [model_part],
                    #         [optimizer],
                    #         self.norms_to_log,
                    #     )
                    # )
                # # To Debug, we can force using naive_param_norm
                # all_norms.update(
                #     naive_param_norm.get_parameter_norms([model_part], [optimizer])
                # )
        all_norms = process_gram_vectors_for_logging(
            all_norms, step=step, config=self.gram_vector_logging_config
        )
        return process_norms_for_logging(
            all_norms,
            step=step,
            config=self.spectrum_logging_config,
        )

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
        wrapper_optimizer_kwargs = optimizer_kwargs.copy()
        wrapper_optimizer_kwargs.pop("parallel_dims", None)
        Optimizer.__init__(self, all_params, wrapper_optimizer_kwargs)
        self._strip_wrapper_runtime_only_keys()

    def _strip_wrapper_runtime_only_keys(self) -> None:
        # `parallel_dims` is a runtime topology object containing DeviceMesh. The
        # wrapper optimizer only needs standard Optimizer hook machinery, so keeping
        # this key in wrapper defaults/param_groups only risks accidental serialization.
        self.defaults.pop("parallel_dims", None)
        for group in self.param_groups:
            group.pop("parallel_dims", None)

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

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        optimizer_cls = self._resolve_optimizer_cls(config.name)
        optimizer_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.model_parts = model_parts
        # Set by the trainer from MetricsProcessor.Config.save_all_shard_ranks;
        # False keeps the historical gather-to-one-rank behaviour.
        self.log_metrics_locally: bool = False
        self.spectrum_logging_config = spectrum_logging.SpectrumLoggingConfig(
            enable_plot=config.enable_spectrum_plot,
            enable_export=config.enable_spectrum_export,
        )
        self.gram_vector_logging_config = GramVectorLoggingConfig(
            enable_plot=config.enable_gram_plot,
            enable_export=config.enable_gram_export,
        )

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
            all_usages_cpu,
            all_biases_cpu,
            all_entropies_cpu,
            all_load_balance_losses_cpu,
            all_maxvio_batch_cpu,
            all_maxvio_ema_cpu,
            num_experts,
        ) = data

        usage_offset = bias_offset = 0
        for i, info in enumerate(moe_layers_info):
            moe = info["module"]
            layer_id = info["layer_id"]

            layer_usages = all_usages_cpu[usage_offset : usage_offset + num_experts]
            layer_biases = all_biases_cpu[bias_offset : bias_offset + num_experts]
            usage_tensor = torch.tensor(layer_usages, dtype=torch.float32)
            bias_tensor = torch.tensor(layer_biases, dtype=torch.float32)
            metrics = {
                f"moe_entropy/L-{layer_id}": all_entropies_cpu[i],
                f"moe_maxvio_batch/L-{layer_id}": all_maxvio_batch_cpu[i],
                f"moe_maxvio_ema/L-{layer_id}": all_maxvio_ema_cpu[i],
                f"moe_load_balance_loss/L-{layer_id}": all_load_balance_losses_cpu[i],
                f"moe_ep_usage_mean/L-{layer_id}": usage_tensor.mean().item(),
                f"moe_ep_usage_std/L-{layer_id}": usage_tensor.std(
                    unbiased=False
                ).item(),
                f"moe_bias_mean/L-{layer_id}": bias_tensor.mean().item(),
                f"moe_bias_std/L-{layer_id}": bias_tensor.std(unbiased=False).item(),
            }
            pre_usage = f"moe_ep_usage/L-{layer_id}_EP-"
            pre_bias = f"moe_bias/L-{layer_id}_EP-"
            metrics.update({f"{pre_usage}{j}": v for j, v in enumerate(layer_usages)})
            metrics.update({f"{pre_bias}{j}": v for j, v in enumerate(layer_biases)})
            moe._log_expert_metrics = metrics
            usage_offset += num_experts
            bias_offset += num_experts

        # Aggregated scalars across all MoE layers — attached to first layer
        num_moe_layers = len(moe_layers_info)
        moe_layers_info[0]["module"]._log_expert_metrics.update(
            {
                "moe_maxvio_batch/aggregate": sum(all_maxvio_batch_cpu)
                / num_moe_layers,
                "moe_maxvio_ema/aggregate": sum(all_maxvio_ema_cpu) / num_moe_layers,
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


def register_moe_load_balancing_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> OptimizersContainer:

    log_queue = optimizers.set_up_async_logging(moe_metrics_worker)

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

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        """
        Lets assume all MoE layers have same amount of experts.
        """

        loss_mesh = parallel_dims.get_optional_mesh("loss")

        # above is adapted from the upstream code
        is_dp_rank_0 = (
            torch.distributed.get_rank(loss_mesh.get_group()) == 0
            if loss_mesh is not None
            else True
        )
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.

        moe_layers_info = []
        tok_buffers, ent_buffers, load_balance_loss_buffers = [], [], []
        ema_buffers = []
        acc_fwd_times_buffers = []
        scale_factor = 1
        num_experts = 0

        for part in model_parts:
            for block in part.layers.values():
                if not block.moe_enabled:
                    continue
                moe = block.moe
                # Assuming num_experts is the same for all, so we can just grab it once
                num_experts = moe.tokens_per_expert.numel()
                moe_layers_info.append(
                    {
                        "module": moe,
                        "layer_id": block.layer_id,
                    }
                )
                tok_buffers.append(moe.tokens_per_expert)
                ema_buffers.append(moe.tokens_per_expert_cumul)
                ent_buffers.append(moe.router_entropy)
                # if need_rescale_stats(moe) or need_rescale_stats(block):
                #     scale_factor = 0.5
                acc_fwd_times_buffers.append(moe.acc_fwd_times)
                load_balance_loss_buffers.append(moe.load_balance_loss)
        # Early exit if no MoE layers were found
        if not moe_layers_info:
            return

        # assume all MoE layers are same
        scale_factor = acc_fwd_times_buffers[-1]

        all_tokens = torch.cat(tok_buffers)
        all_entropies = torch.cat(ent_buffers)
        all_load_balance_losses = torch.cat(load_balance_loss_buffers)
        if scale_factor != 1:
            all_tokens = all_tokens // scale_factor  # tokens count are integers
            all_entropies = all_entropies / scale_factor  # entropies are floats

        if loss_mesh is not None:
            # pg = loss_mesh.get_group()
            # torch.distributed.all_reduce(
            #     all_tokens, group=pg, op=torch.distributed.ReduceOp.SUM
            # )
            # torch.distributed.all_reduce(
            #     all_entropies, group=pg, op=torch.distributed.ReduceOp.AVG
            # )
            # torch.distributed.all_reduce(
            #     all_load_balance_losses, group=pg, op=torch.distributed.ReduceOp.AVG
            # )
            fused_hier_reduce_loss_stats(
                parallel_dims, all_tokens, all_entropies, all_load_balance_losses
            )
        num_layers = len(moe_layers_info)
        lens = torch.full(
            (num_layers,), num_experts, device=all_tokens.device, dtype=torch.long
        )
        grp = torch.repeat_interleave(
            torch.arange(num_layers, device=all_tokens.device, dtype=torch.long), lens
        )

        layer_sums = torch.zeros(
            num_layers, dtype=all_tokens.dtype, device=all_tokens.device
        )
        layer_sums.index_add_(0, grp, all_tokens)
        layer_means = layer_sums / num_experts

        # Globally-reduced per-step expert loads, used for both maxvio_batch and EMA maxvio.
        step_counts_2d = all_tokens.view(num_layers, num_experts).float()
        step_counts_split = list(step_counts_2d.unbind(0))

        # Update per-layer EMA state: ema = beta * ema + (1 - beta) * step_counts.
        try:
            torch._foreach_mul_(ema_buffers, MAXVIO_EMA_BETA)
            torch._foreach_add_(
                ema_buffers, step_counts_split, alpha=(1.0 - MAXVIO_EMA_BETA)
            )
        except Exception:
            for ema_buf, step_counts in zip(
                ema_buffers, step_counts_split, strict=True
            ):
                ema_buf.mul_(MAXVIO_EMA_BETA).add_(
                    step_counts, alpha=(1.0 - MAXVIO_EMA_BETA)
                )

        # MaxVio_batch: worst-case overload in current step window
        layer_means_f = step_counts_2d.mean(dim=1)
        max_per_layer = step_counts_2d.max(dim=1).values
        maxvio_batch = (max_per_layer - layer_means_f) / (layer_means_f + MAXVIO_EPS)

        # MaxVio_ema: worst-case overload over EMA-smoothed expert loads.
        all_ema = torch.cat(ema_buffers).view(num_layers, num_experts)
        ema_means = all_ema.mean(dim=1)
        maxvio_ema = (all_ema.max(dim=1).values - ema_means) / (ema_means + MAXVIO_EPS)

        # Vectorised deltas and usage
        delta_flat = layer_means[grp] - all_tokens
        recip = torch.clamp(layer_sums, min=1.0).reciprocal()
        usage_flat = all_tokens * recip[grp]

        # Vectorized bias update calculation (replaces the loop)
        with torch.no_grad():
            # Get norm factor and load_balance_coeff from the first MoE layer (assuming they are all the same)
            first_moe = moe_layers_info[0]["module"]
            norm_factor = first_moe.bias_update_norm_factor
            load_balance_coeff = first_moe.load_balance_coeff

            # Reshape for batched, per-layer operations
            delta_2d = delta_flat.view(num_layers, num_experts)

            # Calculate updates for all layers at once
            updates_2d = lmo_for_moe_bias(delta_2d, norm_factor=norm_factor)

            if norm_factor.endswith("zero_mean"):
                updates_2d = updates_2d - updates_2d.mean(dim=1, keepdim=True)

            # Collect all bias parameters and update them with a single multi-tensor op
            bias_params = [info["module"].expert_bias for info in moe_layers_info]
            updates_list = list(updates_2d.flatten().split(num_experts))

            torch._foreach_add_(bias_params, updates_list, alpha=load_balance_coeff)

            # Reset router stats in bulk
            try:
                torch._foreach_mul_(tok_buffers, 0)
                torch._foreach_mul_(ent_buffers, 0.0)
                torch._foreach_mul_(acc_fwd_times_buffers, 0)
                torch._foreach_mul_(load_balance_loss_buffers, 0.0)
            except Exception:
                for t in tok_buffers:
                    t.zero_()
                for t in ent_buffers:
                    t.zero_()
                for t in acc_fwd_times_buffers:
                    t.zero_()
                for t in load_balance_loss_buffers:
                    t.zero_()

            if is_dp_rank_0:
                all_usages_cpu = usage_flat.cpu().tolist()
                all_biases_cpu = torch.cat(bias_params).cpu().tolist()
                all_entropies_cpu = all_entropies.cpu().float().tolist()
                all_load_balance_losses_cpu = (
                    all_load_balance_losses.cpu().float().tolist()
                )
                all_maxvio_batch_cpu = maxvio_batch.cpu().tolist()
                all_maxvio_ema_cpu = maxvio_ema.cpu().tolist()
                payload = (
                    moe_layers_info,
                    all_usages_cpu,
                    all_biases_cpu,
                    all_entropies_cpu,
                    all_load_balance_losses_cpu,
                    all_maxvio_batch_cpu,
                    all_maxvio_ema_cpu,
                    num_experts,
                )
                log_queue.put(payload)

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if transformer_block.moe_enabled:
                    return True
        return False

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )
