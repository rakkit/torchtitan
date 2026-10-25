# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DiSCO-aware optimizer container.

Groups are the named `optimizer.groups` (`default` required, the others
matched by `kind` or `match`, first match wins), or upstream's
`optimizer.param_groups` instead; a group's kwargs are DiSCO's
`lr / weight_decay / decoupled_wd / momentum / nesterov / rule / args`. On
top of upstream:
the MoE load-balance hook, and the arming and collection of the optimizer's
per-parameter metrics (`log_norm_freq`, `norms_to_log`, `gram_level`,
spectrum / gram-vector export).
"""

import dataclasses
import logging
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.optim import Optimizer

from torchtitan.components.optimizer.optimizer import (
    OptimizersContainer as BaseOptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.optimizers import DiSCO, disco_factory_kwargs, model_stats
from torchtitan.optimizers.export import spectrum_logging
from torchtitan.optimizers.export.gram_vector_logging import (
    GramVectorLoggingConfig,
    process_gram_vectors_for_logging,
)
from torchtitan.optimizers.export.spectrum_logging import process_norms_for_logging
from torchtitan.optimizers.gnoise import GradNoise, GradNoiseConfig
from torchtitan.optimizers.helpers import norm_helper

__all__ = ["DiSCOOptimizersContainer"]
logger = logging.getLogger(__name__)

_DISCO_ONLY_KEYS = frozenset(
    {"rule", "args", "momentum", "nesterov", "decoupled_wd", "momentum_k"}
)
_KIND = "\x00kind:"  # a group pattern resolved per model part from its declarations


T = TypeVar("T", bound=Optimizer)


class DiSCOOptimizersContainer(BaseOptimizersContainer, Generic[T]):
    """Upstream's optimizer container, extended for DiSCO.

    Adds mesh-derived parameter groups (hence ``parallel_dims``), per-parameter
    norm/spectrum/gram metrics, and the MoE router metrics. The norms are
    produced inside ``step()``, so a reporting step must be armed first:
    ``begin_step`` arms it, ``collect_metrics`` reads back. Also accepts
    Adam/AdamW.
    """

    effective_rank_freq: int = 0  # Config.effective_rank_freq; the class default for a container built without it
    spectral_norm_freq: int = 0  # Config.spectral_norm_freq; likewise
    gnoise: "GradNoise | None" = (
        None  # built in __init__ only when optimizer.gnoise.enable
    )

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizersContainer.Config):
        groups: dict[str, dict[str, Any]] = field(default_factory=dict)
        """Named parameter groups. `default` is required and is the group every
        other group inherits from; every other entry selects its parameters
        with exactly one of `kind` (what the model declares them to be:
        "embedding", "unembedding", "router", "expert_batch", "matrix", ...)
        or `match` (a regex on the FQN). Entries are tried in order, first
        match wins, the rest go to `default`. Other keys: `optimizer` (default
        "DiSCO"), and that optimizer's kwargs -- for DiSCO `lr`,
        `weight_decay`, `decoupled_wd` (weight decay `1 - weight_decay` per
        step instead of `1 - lr weight_decay`), `momentum`, `nesterov`,
        `rule`, `args` (`args` merge
        over the default's). Every key of a group is a CLI flag:
        `--optimizer.groups.default.lr 0.05`."""

        communication_dtype: Literal["float32", "bfloat16"] = "float32"
        """dtype of the FSDP all-to-all and DDP all-gather buffers. Parameters,
        gradients and momentum stay fp32; this is only what travels between
        ranks for the LMO and what the update is cast to on the way back.
        bfloat16 halves those bytes but moves the update direction by ~3%
        (up to 16%) on real momenta; float16 flushes them (CONFIG_MIGRATION
        section 70)."""

        implementation: Literal[
            "for-loop", "foreach", "fused", "fused_opt_states_bf16"
        ] = "fused"
        """
        Specify which optimizer implementation to use:
        - 'fused': Use fused implementation (CUDA only) for best performance.
        - 'foreach': Use some horizontal fusion of tensors for better performance.
        - 'for-loop': Use the default implementation for the optimizer (slowest).
        - more info: https://pytorch.org/docs/stable/optim.html
        """

        enable_step_tracing: bool = False
        """Emit DiSCO's per-step structured-logger spans. Off by default.

        These are 6 records per step per rank (one scalar plus five spans
        around prepare_grads / local / experts / sharded / replicated). At 32
        nodes x 4 GPUs that is 768 records per step, which is a lot of storage
        for something only wanted while profiling the optimizer.

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

        dense_norm_steps: int = 0
        """Also log the norms every step up to this one; 0 disables. Needs
        metrics.dense_log_steps at least as large, which the trainer checks."""

        norms_to_log: list[str] = field(default_factory=lambda: ["default"])
        """Which norm families to log. "default" is the curated set; see
        torchtitan.optimizers.helpers.norm_helper.get_norms_to_log."""

        effective_rank_freq: int = 0
        """0: every norm in `norms_to_log` on every norm-logging step. N > 0:
        `effective_rank`, the one norm that needs the whole spectrum, only on the
        norm-logging steps divisible by N, on step 1 and on the final step."""

        spectral_norm_freq: int = 0
        """0: the SVD-based metrics on every norm-logging step (as log_norm_freq).
        N > 0, a multiple of log_norm_freq: `rms_to_rms`, `stable_rank`,
        `condition_number`, `effective_rank*`, the spectra and the spectral radial
        metrics only on the norm-logging steps divisible by N, step 1 and the final
        step; the SVD-free norms keep log_norm_freq (dense phase included)."""

        gram_level: int = 0
        """Gram-matrix metric depth, on the same cadence as log_norm_freq.
        0 disables. Levels 2-3 need batched eigh and are expensive."""

        head_diag_probes: int = 512
        """Tokens of the last microbatch whose LM-head input the head geometry
        metrics (eta, rho_RI, gamma_H) use, on the log_norm_freq cadence; 0
        disables them. Taken from the training forward (the model's output
        under the chunked loss), so they cost no extra forward. Only mean/RMS
        over probes are reported, so a few hundred suffice."""

        log_residual_stats: bool = False
        """Log the RMS of every block's input and total residual write and of
        the final norm's input on the log_norm_freq cadence, taken in the
        model's eager layer loop (optimizers/model_stats.py; OPTMoEModel)."""

        log_gain_stats: bool = False
        """Log each RMSNorm gain's mean, RMS and ||dg|| / ||g|| on the
        log_norm_freq cadence (optimizers/model_stats.py). With angular_muown
        col_scale it also logs each unit's column gains c (col_gain_*: rms,
        mean log|c|, q10 / q50 / q90, relative update; DiSCO.col_gain_metrics)."""

        head_diag_diameter_iters: int = 10
        """Power-iteration count and farthest-point restarts for the row
        diameter behind gamma_H. Fixed, not convergence-tested, so the sweep
        stays fullgraph-safe. Recovers ~98%; past ~20 does not help."""

        log_gram_input: bool = False
        """On the spectral norm-logging steps, log sigma_1 / ||X||_F of the
        tensor each gram_polar_express call receives, per unit
        (gram_input_s1f/<unit>, its max over the unit's calls, and the count
        of units in the 1.01 failure band [0.94, 0.955])."""

        gnoise: GradNoiseConfig = field(default_factory=GradNoiseConfig)
        """Gradient-noise statistics (optimizers/gnoise.py); off by default."""

        enable_spectrum_export: bool = False
        """
        Whether to export the full raw singular-value spectra (every tracked
        parameter, every norm-logging step) to a Parquet file uploaded as a
        versioned W&B Artifact, for offline/programmatic analysis beyond what
        the plot shows. Off by default: can be large for big MoE models (see
        export/spectrum_logging.py). Only takes effect when
        log_norm_freq > 0.
        """

        enable_gram_export: bool = False
        """
        Whether to export every tracked gram vector (every gram-tracking
        step) to a Parquet file uploaded as a versioned W&B Artifact, for
        offline/programmatic analysis beyond what the plot shows. Off by
        default: can be large, there are far more distinct gram vector
        metrics than spectrum has (see export/gram_vector_logging.py).
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
    def _validate(config: Config) -> None:
        if not 0 <= config.gram_level <= 3:
            raise ValueError(f"optimizer.gram_level {config.gram_level}: use 0-3")
        n, every = config.spectral_norm_freq, config.log_norm_freq
        if n < 0 or (n and (every <= 0 or n % every)):
            raise ValueError(
                f"optimizer.spectral_norm_freq ({n}) must be 0 or a multiple of "
                f"optimizer.log_norm_freq ({every})"
            )

    @staticmethod
    def _param_group_configs(config: Config) -> list[ParamGroupConfig]:
        """`groups` -> upstream's ordered `ParamGroupConfig`s (`default` last as `.*`)."""
        if not config.groups:
            if not config.param_groups:
                raise ValueError("optimizer.groups needs at least a `default` entry")
            return list(config.param_groups)
        if config.param_groups:
            raise ValueError("set optimizer.groups or optimizer.param_groups, not both")
        if "default" not in config.groups:
            raise ValueError("optimizer.groups needs a `default` entry")
        for name, g in config.groups.items():
            stray = sorted(set(g) & _DISCO_ONLY_KEYS)
            if g.get("optimizer", "DiSCO") != "DiSCO" and stray:
                raise ValueError(
                    f"optimizer.groups[{name!r}] runs {g['optimizer']}, which does "
                    f"not use {stray}"
                )
        out = []
        for name, g in config.groups.items():
            if name == "default":
                continue
            if ("match" in g) == ("kind" in g):
                raise ValueError(
                    f"optimizer.groups[{name!r}] needs exactly one of `kind` (a "
                    "declared parameter kind) or `match` (an FQN regex)"
                )
            kw = {k: v for k, v in g.items() if k not in ("match", "kind", "optimizer")}
            out.append(
                ParamGroupConfig(
                    pattern=g["match"] if "match" in g else _KIND + g["kind"],
                    optimizer_name=g.get("optimizer", "DiSCO"),
                    optimizer_kwargs=kw,
                )
            )
        out.append(
            ParamGroupConfig(
                pattern=".*",
                optimizer_name=config.groups["default"].get("optimizer", "DiSCO"),
            )
        )
        return out

    @staticmethod
    def _by_kind(
        configs, model: nn.Module, layouts: dict, *, pp: bool = False
    ) -> list[ParamGroupConfig]:
        """Groups that select by declared kind, as exact-name patterns for this
        model part (the names and the declarations come from the same module
        tree, so no FQN spelling can drift between them). Under PP a stage
        holds only its own parameters, so a group that matches none of them
        (first match wins, as upstream claims them) is dropped for this part."""
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        claimed: set[str] = set()
        out = []
        for c in configs:
            if c.pattern.startswith(_KIND):
                kind = c.pattern[len(_KIND) :]
                names = [
                    n for n in trainable if n in layouts and layouts[n].kind == kind
                ]
                if not names and not pp:
                    raise ValueError(
                        f"an optimizer group selects kind {kind!r}, which no parameter "
                        "of this model part is declared as"
                    )
                c = dataclasses.replace(
                    c, pattern="^(?:" + "|".join(map(re.escape, names)) + ")$"
                )
            hits = [
                n for n in trainable if n not in claimed and re.search(c.pattern, n)
            ]
            if pp and not hits:
                continue
            claimed.update(hits)
            out.append(c)
        return out

    @staticmethod
    def _build_optimizer_kwargs(
        config: Config, parallel_dims: ParallelDims, name: str
    ) -> dict[str, Any]:
        if name == "DiSCO":
            return disco_factory_kwargs(config, parallel_dims)
        # Any other optimizer: its `default` group (when it is the default) or
        # upstream's factory kwargs, plus the implementation switches.
        default = config.groups.get("default", {})
        base = (
            {k: v for k, v in default.items() if k not in ("match", "optimizer")}
            if default.get("optimizer", "DiSCO") == name
            else config.optimizer_factory_kwargs_by_name.get(name, {})
        )
        return {**BaseOptimizersContainer._build_impl_kwargs(config), **base}

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
        output_dir: str | None = None,
    ) -> None:
        self._validate(config)
        group_configs = self._param_group_configs(config)
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
        self.effective_rank_freq: int = config.effective_rank_freq
        self.spectral_norm_freq: int = config.spectral_norm_freq
        self.gram_level: int = config.gram_level
        self.log_norm_freq: int = config.log_norm_freq
        self.dense_norm_steps: int = config.dense_norm_steps
        self.head_diag_probes: int = config.head_diag_probes
        self.head_diag_diameter_iters: int = config.head_diag_diameter_iters
        self.parallel_dims = parallel_dims
        self.log_residual_stats: bool = config.log_residual_stats
        self.log_gain_stats: bool = config.log_gain_stats
        self._model_stats_armed: bool = False
        self._gains_before: list[torch.Tensor] | None = None
        self._col_before: list | None = None  # per optimizer: c of the col_scale units before the step (log_gain_stats)
        self._moe_publish_armed: bool = False
        self._norms_armed: bool = False
        self._head_input: Callable[[], torch.Tensor | None] | None = None
        # Every shard rank logs its own slice; set by the trainer from metrics.
        self.log_metrics_locally = False
        # `output_dir` is a build kwarg so these are set at construction
        # rather than patched on by the trainer afterwards.
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

        for part_idx, model in enumerate(self.model_parts):
            declare = getattr(model, "param_layouts", None)
            layouts = declare() if declare is not None else {}
            configs = self._by_kind(
                group_configs,
                model,
                layouts,
                pp=parallel_dims is not None and parallel_dims.pp_enabled,
            )
            by_name, patterns = self._build_param_groups(model, configs, {})
            for opt_name, groups in by_name.items():
                cls = self._resolve_optimizer_factory(opt_name)
                kwargs = self._build_optimizer_kwargs(config, parallel_dims, opt_name)
                if issubclass(cls, DiSCO):
                    if "lr" not in kwargs:
                        raise ValueError(
                            "DiSCO has no default lr: set it in optimizer.groups."
                            "default, or, when the default group is another "
                            "optimizer (or upstream's param_groups is used), in "
                            "optimizer_factory_kwargs_by_name['DiSCO']"
                        )
                    kwargs["param_layouts"] = layouts
                optimizer = cls(groups, **kwargs)
                if isinstance(optimizer, DiSCO):
                    optimizer.head_diag_diameter_iters = config.head_diag_diameter_iters
                    optimizer.log_gram_input = config.log_gram_input
                self.optimizers.append(optimizer)
                self._log_optimizer(optimizer, part_idx, patterns[opt_name])
                for g in groups:
                    all_params.extend(g["params"])
        if config.log_norm_freq > 0 and not any(
            isinstance(o, DiSCO) for o in self.optimizers
        ):
            logger.warning(
                f"optimizer.log_norm_freq={config.log_norm_freq} has no effect: "
                "no DiSCO optimizer computes parameter norms."
            )
        self._validate_params(all_params)
        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        self._post_init(all_params)
        config.gnoise.validate(log_norm_freq=config.log_norm_freq)
        self.gnoise: GradNoise | None = (
            GradNoise(
                config.gnoise,
                model_parts=self.model_parts,
                parallel_dims=parallel_dims,
                container=self,
            )
            if config.gnoise.enable
            else None
        )
        self._gnoise_research_freq = config.gnoise.research_freq

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    # pyrefly: ignore [bad-override]
    def step(self, *args, **kwargs) -> None:
        if self._head_input is not None:
            h, self._head_input = self._head_input(), None
            for optimizer in self.optimizers:
                if isinstance(optimizer, DiSCO):
                    optimizer.head_probe_h = h
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    # The lr scheduler reads a group's lr_table from the config; checkpoints
    # leave it out, so a branch may run another table than its source run.
    _CONFIG_ONLY_SCHEDULER_KEYS = ("lr_table", "momentum_k")

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        drop = tuple(f".{k}" for k in self._CONFIG_ONLY_SCHEDULER_KEYS)
        return {
            k: v
            for k, v in sd.items()
            if not (k.startswith("param_groups.") and k.endswith(drop))
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state, optionally keeping the current learning rates.

        Resuming a run whose LR schedule was changed (e.g. a new decay phase)
        would otherwise silently restore the checkpoint's learning rates and
        undo the new schedule.
        """
        # DiSCO leaves its rule config out of checkpoints (the run's config
        # wins); the flat loader wants every live group key, so supply them.
        state_dict = dict(state_dict)
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for key in getattr(optimizer, "_CONFIG_ONLY_GROUP_KEYS", ()):
                    for fqn in group.get("param_names", ()):
                        state_dict.setdefault(f"param_groups.{fqn}.{key}", group[key])
                for key in self._CONFIG_ONLY_SCHEDULER_KEYS:
                    if key in group:
                        for fqn in group.get("param_names", ()):
                            state_dict[f"param_groups.{fqn}.{key}"] = group[key]
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

        The cadence lives with `log_norm_freq`; the trainer supplies the two
        facts it cannot know. The final step is always armed.
        """
        if self.log_norm_freq <= 0:
            return False
        if is_final_step:
            return True
        return should_log and (
            step == 1 or step <= self.dense_norm_steps or step % self.log_norm_freq == 0
        )

    def begin_step(
        self,
        step: int,
        *,
        should_log: bool,
        is_final_step: bool,
        head_input: Callable[[], "torch.Tensor | None"] | None = None,
    ) -> int:
        """Arm the next `step()` for what `collect_metrics` reads back; call it
        before the forward. Returns how many rows of the LM head's input the
        head geometry wants from that forward (0: none), which `step()` then
        takes from `head_input` and hands to DiSCO for that step only.
        """
        self._head_input = None
        if self.gnoise is not None:
            self.gnoise.note_step(step)
        if should_log:
            self.arm_moe_publish()
        self._arm_model_stats(step, should_log=should_log, is_final_step=is_final_step)
        if not self.wants_norms_at(
            step, should_log=should_log, is_final_step=is_final_step
        ):
            return 0
        self.arm_norm_computation(
            self._norms_at(step, is_final_step),
            spectral=self._spectral_at(step, is_final_step),
        )
        if self.gnoise is not None and self.gnoise.due(
            step, dense=step == 1 or step <= self.dense_norm_steps or is_final_step
        ):
            freq = self._gnoise_research_freq
            self.gnoise.arm(step, research=freq > 0 and step % freq == 0)
        discos = [o for o in self.optimizers if isinstance(o, DiSCO)]
        if head_input is None or not any(o.has_head_geometry for o in discos):
            return 0
        self._head_input = head_input
        return self.head_diag_probes

    def _norms_at(self, step: int, is_final_step: bool) -> list[str]:
        """The norms for a norm-logging step: all of `norms_to_log`, less
        `effective_rank` off its `effective_rank_freq` cadence and every
        SVD-based norm off the `spectral_norm_freq` cadence."""
        if not self._spectral_at(step, is_final_step):
            return [x for x in self.norms_to_log if x in norm_helper._SVD_FREE_NORMS]
        n = self.effective_rank_freq
        if n > 0 and "effective_rank" in self.norms_to_log:
            if not (is_final_step or step == 1 or step % n == 0):
                return [x for x in self.norms_to_log if x != "effective_rank"]
        return self.norms_to_log

    def _spectral_at(self, step: int, is_final_step: bool) -> bool:
        """Whether a norm-logging step computes the SVD-based metrics."""
        n = self.spectral_norm_freq
        return n <= 0 or is_final_step or step == 1 or step % n == 0

    def arm_norm_computation(
        self, norms: list[str] | None = None, spectral: bool = True
    ):
        """Tell DiSCO to produce norms (`norms`, default `norms_to_log`)
        during the next `step()`; `spectral` False skips the SVD-based ones."""
        self._norms_armed = True
        norms = self.norms_to_log if norms is None else norms
        # Every DiSCO instance, not `optimizers[part_index]`: a config that
        # sends some parameters to AdamW builds one optimizer per (part, name),
        # so the indices stopped lining up with the parts.
        for optimizer in self.optimizers:
            if isinstance(optimizer, DiSCO):
                optimizer.calculate_norm_at_next_step(
                    norms,
                    self.gram_level,
                    # Dropped by process_norms_for_logging unless the parquet
                    # export is on, so let the optimizer skip computing,
                    # packing and all-gathering them.
                    track_spectrum=self.spectrum_logging_config.enable_export
                    and spectral,
                    # Single source of truth with the logger: the optimizer must
                    # only skip the gather on ranks that actually have a logger,
                    # otherwise their metrics are computed and silently dropped.
                    log_metrics_locally=self.log_metrics_locally,
                    spectral=spectral,
                )

    def _arm_model_stats(
        self, step: int, *, should_log: bool, is_final_step: bool
    ) -> None:
        """On the per-unit cadence, before the forward: arm the residual
        capture and snapshot the gains for `_collect_model_stats`."""
        self._gains_before = None
        self._col_before = None
        self._model_stats_armed = self.wants_norms_at(
            step, should_log=should_log, is_final_step=is_final_step
        )
        if not self._model_stats_armed:
            return
        if self.log_residual_stats:
            model_stats.arm_residual_capture(self.model_parts)
        if self.log_gain_stats:
            self._gains_before = model_stats.snapshot_gains(self.model_parts)
            self._col_before = [
                o.col_gain_snapshot() if isinstance(o, DiSCO) else None
                for o in self.optimizers
            ]

    def _collect_model_stats(self) -> dict[str, Any]:
        """The armed step's residual and gain statistics, read once."""
        if not self._model_stats_armed:
            return {}
        self._model_stats_armed = False
        metrics = model_stats.residual_metrics(self.model_parts, self.parallel_dims)
        if self._gains_before is not None:
            metrics.update(
                model_stats.gain_metrics(
                    self.model_parts, self._gains_before, self.parallel_dims
                )
            )
            self._gains_before = None
        if self._col_before is not None:
            for o, before in zip(self.optimizers, self._col_before, strict=True):
                if before is not None:
                    metrics.update(o.col_gain_metrics(before))
            self._col_before = None
        return metrics

    def arm_moe_publish(self) -> None:
        """Tell the MoE hook to publish router metrics during the next `step()`.

        The metrics cost a device-to-host copy, so the hook only pays it on
        steps whose `collect_metrics` will read them back.
        """
        self._moe_publish_armed = True

    def consume_moe_publish_arm(self) -> bool:
        """One-shot read by the MoE hook: publish, or do the bias update alone."""
        armed = self._moe_publish_armed
        self._moe_publish_armed = False
        return armed

    def collect_metrics(self, step: int) -> dict[str, Any]:
        """Everything this container reports for `step`; the norms only when
        the step was armed for them."""
        metrics: dict[str, Any] = {}
        armed, self._norms_armed = self._norms_armed, False
        if armed:
            metrics.update(self.get_parameter_norms(step=step))
        for optimizer in self.optimizers:
            if isinstance(optimizer, DiSCO):
                for gi, (s, eta) in optimizer.eta_used.items():
                    metrics[f"eta_schedule/DiSCO/{gi}"] = s
                    metrics[f"eta_used/DiSCO/{gi}"] = eta
                for gi, m in optimizer.momentum_used.items():
                    metrics[f"momentum_used/DiSCO/{gi}"] = m
        if self.gnoise is not None:
            metrics.update(self.gnoise.collect())
        metrics.update(self.collect_moe_metrics())
        metrics.update(self._collect_model_stats())
        return metrics

    def collect_moe_metrics(self) -> dict[str, Any]:
        """The router metrics the MoE hook published per layer."""
        metrics: dict[str, Any] = {}
        for moe in self._moe_modules():
            metrics.update(getattr(moe, "_log_expert_metrics", None) or {})
        return metrics

    def _moe_modules(self) -> Iterator[nn.Module]:
        for model_part in self.model_parts:
            layers = getattr(model_part, "layers", None)
            if layers is None:
                continue
            for layer in layers.values():
                moe = getattr(layer, "moe", None)
                if moe is not None:
                    yield moe

    def _drop_moe_metrics(self) -> None:
        """Forget the cached per-layer dicts; nothing stale can be logged."""
        for moe in self._moe_modules():
            moe._log_expert_metrics = {}

    def get_parameter_norms(self, step: int):
        """Post-process the norms DiSCO computed during the armed `step()`."""
        all_norms = {}
        for optimizer in self.optimizers:
            if isinstance(optimizer, DiSCO):
                all_norms.update(optimizer.get_norms_at_current_step())
        all_norms = process_gram_vectors_for_logging(
            all_norms, step=step, config=self.gram_vector_logging_config
        )
        return process_norms_for_logging(
            all_norms,
            step=step,
            config=self.spectrum_logging_config,
        )

    def close(self):
        """Forget any armed publish and the cached MoE metrics. Idempotent."""
        self._moe_publish_armed = False
        self._model_stats_armed = False
        self._gains_before = None
        self._col_before = None
        for part in self.model_parts:
            if getattr(part, "residual_capture", None) is not None:
                part.residual_capture = None
        self._drop_moe_metrics()
