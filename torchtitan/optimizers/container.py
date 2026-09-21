# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DiSCO-aware optimizer container, carried forward from the 0.4.0 base.

Upstream rebuilt ``OptimizersContainer`` around ``ParamGroupConfig``: a list of
regex/optimizer pairs, with no top-level ``name``/``lr``/``betas``. DiSCO does
not fit that shape -- its parameter groups are derived from module structure by
``create_disco_param_groups`` (norm factors per tensor role, embedding vs
unembedding vs router), not from FQN regexes, and its per-group kwargs are
computed together from one set of hyperparameters.

So construction is overridden wholesale while everything else -- the
``Optimizer``/``Stateful``/``Configurable`` machinery, hooks, and the flat
checkpoint state-dict format -- is inherited from upstream's container.

This module also carries the OPT MoE load-balancing hook, which differs from
upstream's: it drives the expert bias through an LMO step rather than a plain
sign update, and reports router entropy / max-violation metrics through an
async logging queue.
"""

import logging
import os
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import torch
import torch.distributed as dist
import torch.distributed.tensor
import torch.nn as nn
from torch.optim import Optimizer

from torchtitan.components.checkpointer.utils import canonical_fqn

from torchtitan.components.optimizer.optimizer import (
    OptimizersContainer as BaseOptimizersContainer,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import fsdp_shard_mesh
from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
from torchtitan.optimizers import (
    create_disco_optimizer_kwargs_from_optimizer_config,
    create_disco_param_groups,
    DiSCO,
    norm_helper,
    spectrum_logging,
)
from torchtitan.optimizers.gram_vector_logging import (
    GramVectorLoggingConfig,
    process_gram_vectors_for_logging,
)
from torchtitan.optimizers.spectrum_logging import process_norms_for_logging

__all__ = [
    "DiSCOOptimizersContainer",
    "register_moe_load_balancing_hook",
]
logger = logging.getLogger(__name__)


MAXVIO_EMA_BETA = 0.995
MAXVIO_EPS = 1e-12


T = TypeVar("T", bound=Optimizer)


class DiSCOOptimizersContainer(BaseOptimizersContainer, Generic[T]):
    """Upstream's optimizer container, extended for DiSCO.

    Named for the optimizer it exists to serve, not for the only one it can
    build -- ``name`` also accepts Adam/AdamW, so the same instrumentation can
    run the AdamW arm of an A/B. It used to be called ``OptimizersContainer``,
    exactly like its base, which made it read as a FORK of upstream's container
    rather than a subclass: 25 of the base's methods are inherited untouched.

    What it adds over the base:

    * DiSCO parameter groups derived from MESH TOPOLOGY rather than from FQN
      regexes, which is why it needs ``parallel_dims`` at construction.
    * Per-parameter norm / spectrum / gram metrics. These are produced INSIDE
      ``step()``, so a step that should report them has to be armed before it
      runs: ``wants_norms_at()`` decides, ``arm_norm_computation()`` arms, and
      ``collect_metrics()`` reads the result back afterwards.
    * The async MoE load-balancing metrics worker, also drained by
      ``collect_metrics()``.

    Upstream's own docstring follows.

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
    class Config(BaseOptimizersContainer.Config):
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

        log_metrics_locally: bool = False
        """Whether every shard rank logs its own slice instead of gathering.

        Derived, not user-set: ``Trainer.Config.__post_init__`` computes it
        from ``metrics.save_all_shard_ranks``/``save_for_all_ranks`` and
        whether any logger backend is on, because the decision spans two
        configs. It has to be identical on every rank -- the gather it skips is
        a collective, so ranks that disagree hang rather than misreport.
        """

        enable_step_tracing: bool = False
        """Emit DiSCO's per-step structured-logger spans. Off by default.

        These are 8 records per step per rank (one scalar plus seven spans
        around prepare_grads / workspaces / embedding / experts / fsdp / ddp /
        scalar). At 32 nodes x 4 GPUs that is 1024 records per step, which is a
        lot of storage for something only wanted while profiling the optimizer.

        Separate from `debug.enable_structured_logging`, which is global: this
        silences DiSCO's spans without also silencing the trainer's.
        """

        log_norm_freq: int = 0
        """How often, in steps, to log DiSCO's per-parameter norms. 0 disables.

        Lives here rather than on MetricsProcessor.Config because it is
        OPTIMIZER behaviour: the norms are produced inside step(). It must be a
        multiple of metrics.log_freq, which the trainer validates -- norms only
        land on steps that are also metric-logging steps, so otherwise they
        arrive on lcm(log_freq, log_norm_freq) instead of the cadence asked for.
        """

        norms_to_log: list[str] = field(default_factory=lambda: ["default"])
        """Which norm families to log. "default" is the curated set; see
        torchtitan.optimizers.norm_helper.get_norms_to_log."""

        gram_level: int = 0
        """Gram-matrix metric depth, on the same cadence as log_norm_freq.
        0 disables. Levels 2-3 need batched eigh and are expensive."""

        enable_spectrum_export: bool = False
        """
        Whether to export the full raw singular-value spectra (every tracked
        parameter, every norm-logging step) to a Parquet file uploaded as a
        versioned W&B Artifact, for offline/programmatic analysis beyond what
        the plot shows. Off by default: can be large for big MoE models (see
        optimizers/spectrum_logging.py). Only takes effect when
        log_norm_freq > 0.
        """

        enable_gram_export: bool = False
        """
        Whether to export every tracked gram vector (every gram-tracking
        step) to a Parquet file uploaded as a versioned W&B Artifact, for
        offline/programmatic analysis beyond what the plot shows. Off by
        default: can be large, there are far more distinct gram vector
        metrics than spectrum has (see optimizers/gram_vector_logging.py).
        Only takes effect when gram_level > 0.
        """

    optimizers: list[T]
    model_parts: list[nn.Module]

    @staticmethod
    def _resolve_optimizer_factory(name: str) -> Callable[..., Optimizer]:
        """Upstream's registry with DiSCO in place of DistMuon.

        Overrides the base hook rather than sitting beside it as a separate
        `_resolve_optimizer_cls`, so there is one name for this lookup and a
        future change to upstream's signature shows up as a conflict here
        instead of silently leaving a second, stale registry in the subclass.
        """
        optimizer_factories: dict[str, Callable[..., Optimizer]] = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "DiSCO": DiSCO,
        }
        if name not in optimizer_factories:
            raise NotImplementedError(f"Optimizer {name} not added.")
        return optimizer_factories[name]

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
        output_dir: str | None = None,
    ) -> None:
        optimizer_cls = self._resolve_optimizer_factory(config.name)
        optimizer_kwargs = self._build_optimizer_kwargs(config, parallel_dims)
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        self.preserve_lrs_when_loading = False
        # Set from config at build time. They used to be assigned onto the
        # container by the trainer after construction, which meant the object
        # was briefly in a state its own methods did not expect.
        self.norms_to_log: list[str] | None = norm_helper.get_norms_to_log(
            config.norms_to_log
        )
        self.gram_level: int = config.gram_level
        self.log_norm_freq: int = config.log_norm_freq
        self.log_queue: queue.Queue | None = None
        self.log_thread: threading.Thread | None = None
        self.log_metrics_locally: bool = config.log_metrics_locally
        # The export directories are the only part of the logging setup that
        # depends on a runtime value, so `output_dir` is taken as a build kwarg
        # rather than left for the trainer to patch in afterwards. It used to
        # assign both configs back onto this object post-construction, which
        # left the container briefly in a state its own methods did not expect.
        self.spectrum_logging_config = spectrum_logging.SpectrumLoggingConfig(
            enable_export=config.enable_spectrum_export,
            export_dir=(
                os.path.join(output_dir, "spectrum_export") if output_dir else None
            ),
        )
        self.gram_vector_logging_config = GramVectorLoggingConfig(
            enable_export=config.enable_gram_export,
            export_dir=os.path.join(output_dir, "gram_export") if output_dir else None,
        )

        # `param_groups` and `optimizer_factory_kwargs_by_name` are inherited
        # from BaseOptimizersContainer.Config but this container does NOT honour
        # them: DiSCO derives its groups from mesh topology and tensor role via
        # `create_disco_param_groups` (+ `extra_param_group_split_rules`), and
        # the non-DiSCO branch below takes a flat `model.parameters()`. Silently
        # dropping a user's param_groups would mean training with different
        # hyper-parameters than the config asks for, so refuse instead. Not
        # implemented rather than not wanted -- wiring upstream's grouping into
        # DiSCO's role-based split needs its own design.
        if getattr(config, "param_groups", None):
            raise ValueError(
                "optimizers.param_groups is not supported by this container: "
                "DiSCO builds its own groups from tensor role and mesh topology "
                "(create_disco_param_groups). Use "
                "optimizer.extra_param_group_split_rules instead."
            )
        if getattr(config, "optimizer_factory_kwargs_by_name", None):
            raise ValueError(
                "optimizers.optimizer_factory_kwargs_by_name is not supported "
                "by this container; it is never read. Set the optimizer kwargs "
                "directly on the optimizer config."
            )

        for model in self.model_parts:
            if issubclass(optimizer_cls, DiSCO):
                params, optimizer_kwargs = create_disco_param_groups(
                    model, optimizer_kwargs
                )
            else:
                # `param_names` is REQUIRED by the inherited upstream
                # `state_dict()` (components/optimizer/utils.py), which the
                # 0.5.0 port started using when it dropped 0.4.0's override.
                # Building from a bare parameter list left the group without it,
                # so a non-DiSCO run (name="AdamW", the Config default) trained
                # fine and then died at the first checkpoint with
                # "Optimizer must be built with (name, param) tuples".
                # Upstream's own `_build_param_groups` always sets it, and
                # upstream's EMA container does the same thing.
                named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                params = [
                    {
                        "params": [p for _, p in named],
                        "param_names": [canonical_fqn(n) for n, _ in named],
                    }
                ]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
            all_params.extend(params)

        assert len(self.optimizers) == len(self.model_parts), (
            f"{len(self.optimizers)} optimizers for {len(self.model_parts)} "
            "model parts; this container builds exactly one per part."
        )
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

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state, optionally keeping the current learning rates.

        Resuming a run whose LR schedule was changed (e.g. a new decay phase)
        would otherwise silently restore the checkpoint's learning rates and
        undo the new schedule.
        """
        if not self.preserve_lrs_when_loading:
            super().load_state_dict(state_dict)
            return

        prev_lrs = [
            [group["lr"] for group in optimizer.param_groups]
            for optimizer in self.optimizers
        ]
        super().load_state_dict(state_dict)
        for optimizer, optim_prev_lrs in zip(self.optimizers, prev_lrs):
            for param_group, prev_lr in zip(optimizer.param_groups, optim_prev_lrs):
                if param_group["lr"] != prev_lr:
                    logger.warning(
                        f"Restoring lr from {param_group['lr']} to {prev_lr} | "
                        f"for {param_group.get('param_names')}"
                    )
                    param_group["lr"] = prev_lr

    def wants_norms_at(
        self, step: int, *, should_log: bool, is_final_step: bool
    ) -> bool:
        """Whether `step` should compute per-parameter norms.

        The cadence lives here because `log_norm_freq` does. The two facts it
        cannot know -- whether this is a metrics-logging step at all, and
        whether it is the run's last -- are passed in by the trainer.

        Norm logging is gated on `should_log` downstream, so anything computed
        on a non-logging step would be thrown away; `log_norm_freq` is required
        to be a multiple of `metrics.log_freq` so that the two cadences line up
        (validated in `Trainer.Config.__post_init__`). The final step is armed
        unconditionally so a run always ends with a spectrum snapshot.
        """
        if self.log_norm_freq <= 0:
            return False
        if is_final_step:
            return True
        return should_log and (step == 1 or step % self.log_norm_freq == 0)

    def arm_norm_computation(self):
        """Tell DiSCO to produce norms during the NEXT `step()`.

        Has to happen before the step, not at the log site: the norms are a
        by-product of the update itself.
        """
        for i, _ in enumerate(self.model_parts):
            optimizer = self.optimizers[i]
            if isinstance(optimizer, DiSCO):
                optimizer.calculate_norm_at_next_step(
                    self.norms_to_log,
                    self.gram_level,
                    # The spectra are only ever consumed by
                    # process_norms_for_logging, which drops them unless the
                    # parquet export is on. Telling the optimizer up front lets
                    # it skip computing, packing and all-gathering them.
                    track_spectrum=self.spectrum_logging_config.enable_export,
                    # Single source of truth with the logger: the optimizer must
                    # only skip the gather on ranks that actually have a logger,
                    # otherwise their metrics are computed and silently dropped.
                    log_metrics_locally=self.log_metrics_locally,
                )

    def collect_metrics(self, step: int, *, include_norms: bool) -> dict[str, Any]:
        """Everything this container has to report for `step`.

        One call instead of three, so the trainer does not have to know that
        norms and MoE routing stats come from different machinery on different
        schedules. `include_norms` must match what `wants_norms_at` returned
        before the step -- asking for norms that were never armed yields an
        empty dict, not an error.
        """
        metrics: dict[str, Any] = {}
        if include_norms:
            metrics.update(self.get_parameter_norms(step=step))
        metrics.update(self.collect_moe_metrics())
        return metrics

    def collect_moe_metrics(self) -> dict[str, Any]:
        """Drain the async MoE metrics worker and gather what it produced.

        The worker writes each layer's dict onto the MoE MODULE
        (`_log_expert_metrics`) because that is the handle the load-balancing
        hook already holds. Reading them back belongs here, with the producer,
        rather than in the trainer -- which used to walk
        `model_parts -> layers -> moe` and read that private attribute itself.
        """
        # The worker reports router entropy / max-violation from a background
        # thread; drain it before reading the per-layer dicts.
        self.join_log_queue()
        metrics: dict[str, Any] = {}
        for model_part in self.model_parts:
            layers = getattr(model_part, "layers", None)
            if layers is None:
                continue
            for layer in layers.values():
                moe = getattr(layer, "moe", None)
                if moe is not None:
                    metrics.update(getattr(moe, "_log_expert_metrics", None) or {})
        return metrics

    def get_parameter_norms(self, step: int):
        """Post-process the norms DiSCO computed during the armed `step()`."""
        all_norms = {}
        for optimizer in self.optimizers:
            if isinstance(optimizer, DiSCO):
                # Returns the dict stashed by the armed step, so it is read
                # once per optimizer -- this used to sit inside a
                # `for group in optimizer.param_groups` loop, re-reading and
                # re-merging the same dict once per parameter group.
                all_norms.update(optimizer.get_norms_at_current_step())
            else:
                logger.warning(
                    f"Optimizer {optimizer.__class__.__name__} does not support "
                    "norm calculation."
                )
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

    def set_up_async_logging(self, log_fn: Callable):
        self.log_queue = queue.Queue()
        # daemon=True is the safety net for the teardown hang: the worker sits
        # in a blocking `log_queue.get()` until it receives the None sentinel,
        # and that sentinel is only sent by `close()`. Nothing in the tree
        # called `close()`, so a NON-daemon worker kept the process alive after
        # training finished -- jobs appeared to "hang in teardown" and had to be
        # killed by the sbatch timeout. `Trainer.close()` now calls `close()`
        # for a clean drain; daemon=True makes sure a missed call can never
        # block process exit again.
        self.log_thread = threading.Thread(
            target=log_fn, args=(self.log_queue,), daemon=True
        )
        self.log_thread.start()
        return self.log_queue

    def close(self):
        """Drain and stop the async metrics worker. Idempotent."""
        if self.log_queue is not None:
            self.log_queue.put(None)
        if self.log_thread is not None:
            # Bounded: a wedged worker must not turn shutdown into a hang.
            self.log_thread.join(timeout=30.0)
            if self.log_thread.is_alive():
                logger.warning(
                    "async metrics worker did not exit within 30s; abandoning it "
                    "(it is a daemon thread, so it cannot block process exit)."
                )
        self.log_queue = None
        self.log_thread = None

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


def moe_metrics_worker(log_queue: queue.Queue):
    """
    This function runs in the background. It waits for data,
    does the slow CPU work, and assigns the final dictionary.
    """
    while True:
        # 1. Wait for data from the main thread
        data = log_queue.get()
        if data is None:  # Sentinel to stop the thread
            # Balance the get() before leaving, or `join_log_queue()` (called
            # every logging step from the trainer) waits forever on an
            # unfinished_tasks count that can never reach zero.
            log_queue.task_done()
            break

        moe_layers_info: list[dict] = []
        # Any exception here used to kill the worker silently -- and then
        # `join_log_queue()`, which the trainer calls every logging step,
        # would block the MAIN thread forever on a queue nobody drains.
        # daemon=True does not help there. Log and keep serving instead.
        try:
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
                    f"moe_load_balance_loss/L-{layer_id}": all_load_balance_losses_cpu[
                        i
                    ],
                    f"moe_ep_usage_mean/L-{layer_id}": usage_tensor.mean().item(),
                    f"moe_ep_usage_std/L-{layer_id}": usage_tensor.std(
                        unbiased=False
                    ).item(),
                    f"moe_bias_mean/L-{layer_id}": bias_tensor.mean().item(),
                    f"moe_bias_std/L-{layer_id}": bias_tensor.std(
                        unbiased=False
                    ).item(),
                }
                pre_usage = f"moe_ep_usage/L-{layer_id}_EP-"
                pre_bias = f"moe_bias/L-{layer_id}_EP-"
                metrics.update(
                    {f"{pre_usage}{j}": v for j, v in enumerate(layer_usages)}
                )
                metrics.update(
                    {f"{pre_bias}{j}": v for j, v in enumerate(layer_biases)}
                )
                moe._log_expert_metrics = metrics
                usage_offset += num_experts
                bias_offset += num_experts

            # Aggregated scalars across all MoE layers — attached to first layer
            num_moe_layers = len(moe_layers_info)
            moe_layers_info[0]["module"]._log_expert_metrics.update(
                {
                    "moe_maxvio_batch/aggregate": sum(all_maxvio_batch_cpu)
                    / num_moe_layers,
                    "moe_maxvio_ema/aggregate": sum(all_maxvio_ema_cpu)
                    / num_moe_layers,
                }
            )

        except Exception:
            logger.exception("async MoE metrics worker failed on one payload")
            # Clear what this payload was meant to fill. `collect_moe_metrics`
            # reads whatever is still on the module, so leaving the previous
            # payload's dict in place republishes the PREVIOUS logging step's
            # entropy / maxvio / bias as if they were current -- the chart shows
            # a flat segment rather than a gap, and a flat maxvio is exactly
            # what a healthy run looks like. Missing is honest; stale is not.
            for info in moe_layers_info:
                info["module"]._log_expert_metrics = {}
        finally:
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
    fsdp_mesh = fsdp_shard_mesh(parallel_dims)
    dp_mesh = parallel_dims.get_optional_mesh("dp_replicate")

    # Hierarchical reduction applies only under HSDP (both meshes present); every
    # other topology takes the flat path. The two are mathematically equivalent --
    # both SUM over the same rank set, both normalised by |loss mesh| -- verified
    # across 18 mesh configurations under a fake process group, so the choice is a
    # communication-cost one and needs no knob. (An env override lived here to
    # isolate the reduce while chasing an apparent HSDP loss spread; that turned
    # out to be seed noise, so the knob is gone.)
    use_hierarchical = (fsdp_mesh is not None) and (dp_mesh is not None)

    # 2. Fuse & Pack (float32)
    #
    # fp32 is sufficient here, measured rather than assumed. Simulating the
    # reduce (sequential-ring and tree orders, E=128, T=40960) at 64/128/1024
    # ranks across expert spreads of 20%/1%/0.05% gives ZERO sign flips in all
    # 18 cells. Error: at gas=1 the per-rank values are integral and the fp32
    # reduce is EXACT; at gas=8 the `/sf` normalisation makes them non-integral
    # and max relative error is ~1.3e-6 at 1024 ranks.
    #
    # Sign flips are the dominant concern because the default
    # `bias_update_norm_factor` is `sign`, but NOT the only one: the `spectral`
    # and `rms` factors are magnitude-sensitive, and at least one recorded
    # config uses `bias_spectral`. 1.3e-6 relative is far below any meaningful
    # change to those updates, but they are not sign-quantised, so the margin
    # is the argument rather than exactness.
    #
    # Exact integer representation is also not at risk: fp32 is exact to 2^24 =
    # 16.7M and the summed counts land near 2.6M. That is independent of
    # gradient accumulation because `_update_expert_bias` divides by
    # `acc_fwd_times` BEFORE calling this, so each rank contributes a
    # per-forward mean. Do not move that division after the reduce -- a raw sum
    # at gas=8 x 1024 ranks reaches 21M and would start rounding.
    acc = torch.float32
    t0 = all_tokens.reshape(-1).to(acc)
    t1 = all_entropies.reshape(-1).to(acc)
    t2 = all_load_balance_losses.reshape(-1).to(acc)

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
    optimizers: DiSCOOptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> DiSCOOptimizersContainer:

    # NOTE: the worker thread is started lazily, at the bottom of this
    # function, only once `_should_register_moe_balancing_hook` says the hook
    # will actually be registered. It cannot be decided here: the predicate is
    # a nested `def` further down in this same function, so referencing it at
    # this point makes Python treat the name as an unassigned local
    # (UnboundLocalError) and kills every run.
    log_queue = None

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
        raise ValueError(
            f"unknown bias_update_norm_factor {norm_factor!r}. Returning None "
            "here would reach torch._foreach_add_ as a None element, and a "
            "near-miss like 'sign_zeromean' would silently drop zero-centring "
            "because the zero_mean check is endswith('zero_mean'). Valid: "
            "sign, spectral, rms, and their *_zero_mean variants."
        )

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        """
        Lets assume all MoE layers have same amount of experts.
        """

        loss_mesh = parallel_dims.get_optional_mesh("loss")

        # above is adapted from the upstream code
        #
        # The `loss` mesh is dp_replicate x dp_shard x cp -- it does NOT contain
        # tp (parallel_dims.py:311). So loss-rank 0 alone is true on EVERY tp
        # rank, and under tp > 1 every one of them would push a payload for the
        # same layers. TP ranks hold shards of the same router, not distinct
        # work, so require tp local rank 0 as well. This mirrors the predicate
        # in `components/metrics.py:478-481`. Inert at tp=1.
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        is_dp_rank_0 = (loss_mesh is None or loss_mesh.get_local_rank() == 0) and (
            tp_mesh is None or tp_mesh.get_local_rank() == 0
        )
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.

        moe_layers_info = []
        tok_buffers, ent_buffers, load_balance_loss_buffers = [], [], []
        ema_buffers = []
        acc_fwd_times_buffers = []
        num_experts = 0

        for part in model_parts:
            for block in part.layers.values():
                if not block.moe_enabled:
                    continue
                moe = block.moe
                # Assuming num_experts is the same for all, so we can just grab it once
                layer_num_experts = moe.tokens_per_expert_E.numel()
                if num_experts and layer_num_experts != num_experts:
                    raise ValueError(
                        "All MoE layers must have the same expert count: "
                        f"layer {block.layer_id} has {layer_num_experts}, an "
                        f"earlier layer has {num_experts}. The stats below are "
                        "packed with `.view(num_layers, num_experts)`, which "
                        "would silently mis-attribute them."
                    )
                num_experts = layer_num_experts
                moe_layers_info.append(
                    {
                        "module": moe,
                        "layer_id": block.layer_id,
                    }
                )
                tok_buffers.append(moe.tokens_per_expert_E)
                ema_buffers.append(moe.tokens_per_expert_cumul)
                ent_buffers.append(moe.router_entropy)
                acc_fwd_times_buffers.append(moe.acc_fwd_times)
                # Per-layer load-balance value. The old `moe.load_balance_loss`
                # buffer is gone: the loss now goes through upstream's AuxLoss,
                # which keeps a PER-INSTANCE (i.e. per-layer) accumulator, so the
                # `moe_load_balance_loss/L-*` breakdown survives the migration --
                # upstream's own `collect_aux_loss_metrics` only reports the mean
                # over layers. Falls back to a zero scalar when the layer has no
                # aux loss (moe.aux_loss is None).
                aux = getattr(moe, "aux_loss", None)
                load_balance_loss_buffers.append(
                    aux.instance_acc.reshape(1)
                    if aux is not None
                    else torch.zeros(
                        1, dtype=torch.float32, device=moe.acc_fwd_times.device
                    )
                )
        # Early exit if no MoE layers were found
        if not moe_layers_info:
            return

        # Everything below is [num_layers, num_experts]. The previous version
        # flattened to 1-D and then rebuilt the per-layer structure with a
        # `repeat_interleave` group index plus an `index_add_`; working 2-D
        # directly drops that index tensor and four gather/scatter kernels, and
        # removes a duplicated per-layer mean.
        num_layers = len(moe_layers_info)
        all_tokens = torch.stack(tok_buffers)  # [L, E]
        all_entropies = torch.cat(ent_buffers)  # [L]
        all_load_balance_losses = torch.cat(load_balance_loss_buffers)  # [L]

        # Every buffer above accumulates once per FORWARD, so each is a sum over
        # `gradient_accumulation_steps * (1 + AC recomputes)` passes. Divide by
        # each layer's OWN counter: taking the last layer's and applying it to
        # all was correct only while activation checkpointing was uniform across
        # layers, and silently wrong under per-layer/selective AC.
        #
        # `all_load_balance_losses` is normalised here too. It was previously
        # left as a raw sum while `all_entropies` right beside it was averaged,
        # so the logged `moe_load_balance_loss/L-*` scaled with `gas` and with
        # AC -- not comparable across configs.
        #
        # Division is unconditional: `x / 1.0` is exact, so guarding it on
        # `sf != 1` bought nothing and cost a `bool()` host sync every step.
        # True division, not `//`: `tokens_per_expert_E` is float32, so `//`
        # floored it and could collapse two distinct expert loads onto the same
        # value, zeroing a bias update that should have been +/-1.
        sf = torch.cat(acc_fwd_times_buffers).to(all_tokens.dtype).clamp_min(1.0)
        all_tokens = all_tokens / sf.unsqueeze(1)
        all_entropies = all_entropies / sf
        all_load_balance_losses = all_load_balance_losses / sf

        if loss_mesh is not None:
            fused_hier_reduce_loss_stats(
                parallel_dims, all_tokens, all_entropies, all_load_balance_losses
            )

        layer_sums = all_tokens.sum(dim=1, keepdim=True)  # [L, 1]
        layer_means = layer_sums / num_experts  # [L, 1]

        # Per-layer EMA: ema = beta * ema + (1 - beta) * step_counts.
        torch._foreach_mul_(ema_buffers, MAXVIO_EMA_BETA)
        torch._foreach_add_(
            ema_buffers, list(all_tokens.unbind(0)), alpha=(1.0 - MAXVIO_EMA_BETA)
        )

        # MaxVio_batch: worst-case overload in the current step window.
        means = layer_means.squeeze(1)  # [L]
        maxvio_batch = (all_tokens.max(dim=1).values - means) / (means + MAXVIO_EPS)

        # MaxVio_ema: worst-case overload over EMA-smoothed expert loads.
        all_ema = torch.stack(ema_buffers)  # [L, E]
        ema_means = all_ema.mean(dim=1)
        maxvio_ema = (all_ema.max(dim=1).values - ema_means) / (ema_means + MAXVIO_EPS)

        delta_2d = layer_means - all_tokens  # [L, E]
        usage_2d = all_tokens / layer_sums.clamp_min(1.0)  # [L, E]

        # Vectorized bias update calculation (replaces the loop)
        with torch.no_grad():
            # Get norm factor and load_balance_coeff from the first MoE layer (assuming they are all the same)
            first_moe = moe_layers_info[0]["module"]
            norm_factor = first_moe.bias_update_norm_factor
            load_balance_coeff = first_moe.load_balance_coeff
            # Both are read from layer 0 and applied to every layer.
            # `load_balance_coeff` consistency is already enforced by
            # `_should_register_moe_balancing_hook`; `bias_update_norm_factor`
            # was not checked at all, so a per-layer override was silently
            # ignored rather than rejected.
            for info in moe_layers_info[1:]:
                if info["module"].bias_update_norm_factor != norm_factor:
                    raise ValueError(
                        "All MoE layers must share bias_update_norm_factor; "
                        f"layer {info['layer_id']} has "
                        f"{info['module'].bias_update_norm_factor!r}, layer 0 "
                        f"has {norm_factor!r}."
                    )

            updates_2d = lmo_for_moe_bias(delta_2d, norm_factor=norm_factor)

            if norm_factor.endswith("zero_mean"):
                updates_2d = updates_2d - updates_2d.mean(dim=1, keepdim=True)

            # Collect all bias parameters and update them with a single multi-tensor op
            bias_params = [info["module"].expert_bias_E for info in moe_layers_info]
            torch._foreach_add_(
                bias_params, list(updates_2d.unbind(0)), alpha=load_balance_coeff
            )

            # Reset router stats in bulk. NOT load_balance_loss_buffers: those
            # are now AuxLoss.instance_acc, whose lifecycle upstream's
            # `register_aux_loss_zero_hook` owns -- it rolls each into the group
            # registers that `collect_aux_loss_metrics` reduces, then clears it.
            # Zeroing here too would race that hook: whichever ran second would
            # see zeros, silently reporting 0 for either our per-layer breakdown
            # or upstream's layer-mean.
            torch._foreach_zero_(tok_buffers)
            torch._foreach_zero_(ent_buffers)
            torch._foreach_zero_(acc_fwd_times_buffers)

            if is_dp_rank_0:
                # One packed D2H copy. Six separate `.cpu()` calls meant six
                # device syncs per logging step for a few KB of scalars.
                dt = usage_2d.dtype
                packed = torch.cat(
                    [
                        usage_2d.reshape(-1),
                        torch.stack(bias_params).reshape(-1).to(dt),
                        all_entropies.to(dt),
                        all_load_balance_losses.to(dt),
                        maxvio_batch.to(dt),
                        maxvio_ema.to(dt),
                    ]
                ).cpu()
                n_le = num_layers * num_experts
                all_usages_cpu = packed[:n_le].tolist()
                all_biases_cpu = packed[n_le : 2 * n_le].tolist()
                off = 2 * n_le
                all_entropies_cpu = packed[off : off + num_layers].tolist()
                off += num_layers
                all_load_balance_losses_cpu = packed[off : off + num_layers].tolist()
                off += num_layers
                all_maxvio_batch_cpu = packed[off : off + num_layers].tolist()
                off += num_layers
                all_maxvio_ema_cpu = packed[off : off + num_layers].tolist()
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
        # Presence of an MoE layer is NOT sufficient -- upstream also requires
        # `load_balance_coeff is not None`. `NormMoE.__init__` leaves
        # `expert_bias_E = None` when the coeff is None, so registering the hook
        # anyway makes `_update_expert_bias` call `torch._foreach_add_` with a
        # None element and die at the first optimizer step. Mirrors upstream's
        # `components/optimizer/optimizer.py::_should_register_moe_balancing_hook`,
        # including its consistency check across layers.
        moes = []
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if transformer_block.moe_enabled:
                    moes.append(transformer_block.moe)
        if not moes:
            return False
        enabled = moes[0].load_balance_coeff is not None
        for moe in moes[1:]:
            if (moe.load_balance_coeff is not None) != enabled:
                raise ValueError(
                    "MoE load_balance_coeff must be configured consistently "
                    "across all MoE layers. Either set it for every MoE layer "
                    "or leave it unset for all MoE layers."
                )
        return enabled

    if _should_register_moe_balancing_hook(model_parts):
        # Upstream's `_update_expert_bias` all-reduces `tokens_per_expert_E`
        # over `get_dense_tp_mesh()` when `ep_enabled and tp > 1`
        # (components/optimizer/optimizer.py). This container reduces only over
        # the `loss` mesh, which excludes tp, so under EP+TP each rank would
        # bias its experts on its own tp shard's token counts -- silently wrong,
        # never crashing. Refuse rather than train on it. No recipe in the tree
        # sets tensor_parallel_degree > 1, so this is inert today; implementing
        # the reduction needs a real EP+TP run to validate, not a fake pg.
        if parallel_dims.ep_enabled and parallel_dims.tp > 1:
            raise NotImplementedError(
                "MoE load-balancing bias updates are not implemented for "
                f"EP + TP (ep={parallel_dims.ep}, tp={parallel_dims.tp}): the "
                "expert token counts are reduced over the 'loss' mesh only, "
                "which excludes the tp axis, so each rank would see just its "
                "own tp shard's counts. Run with tensor_parallel_degree=1, or "
                "add the dense-tp all_reduce that upstream performs."
            )
        # Start the async metrics worker only now: a dense run, or one with
        # load_balance_coeff=None, would otherwise spawn an idle daemon
        # thread that nothing ever feeds.
        log_queue = optimizers.set_up_async_logging(moe_metrics_worker)
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )
        # Upstream pairs the load-balancing hook with an aux-loss zero hook (see
        # deepseek_v3's _post_optimizer_build_fn). It rolls each AuxLoss
        # instance's `instance_acc` into the group registers that
        # `collect_aux_loss_metrics` reduces at log time, then clears them, so
        # the accumulators cover exactly one optimizer step.
        register_aux_loss_zero_hook(optimizers, model_parts, parallel_dims)
