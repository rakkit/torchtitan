# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpointer import DATALOADER
from torchtitan.components.data.balance import NodeAttentionBalancer
from torchtitan.components.data.loader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validator import BaseValidator, Validator
from torchtitan.config import (
    apply_overrides,
    Configurable,
    OverrideConfig,
    PrecisionConfig,
)
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.base.model import BaseModel
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.observability import structured_logger as sl

from torchtitan.observability.data_consumption import (
    build_data_mix_metrics,
    DataMixMetricsLike,
)
from torchtitan.observability.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.training_engine import TrainingEngine


logger = logging.getLogger(__name__)


def _disco_container(engine: TrainingEngine) -> DiSCOOptimizersContainer | None:
    """The engine's optimizer container if it is DiSCO's (an AdamW run has
    upstream's)."""
    optimizers = engine.optim.optimizers
    return optimizers if isinstance(optimizers, DiSCOOptimizersContainer) else None


def logs_metrics_locally(metrics) -> bool:
    """Whether every shard rank logs its own slice of DiSCO's metrics, so its
    logging gather can be skipped. From config alone, so every rank agrees (the
    gather is a collective); `save_all_shard_ranks` alone narrows to one
    logger, which would then log only its own slice."""
    return bool(
        metrics.save_all_shard_ranks
        and metrics.save_for_all_ranks
        and bool(metrics.loggers)
    )


class Trainer(Configurable):
    """Dataset-driven training application that owns a :class:`TrainingEngine`.

    The trainer owns input, validation, and reporting policy. Its training engine
    owns model execution, optimization, checkpointing, profiling, and replay.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TrainingEngine.Config):
        """
        Default container for training configuration.
        """

        model: BaseModel.Config

        precision: PrecisionConfig = field(default_factory=PrecisionConfig)
        """Parameter, compute, and gradient-reduction dtypes of the model."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        attention_balancer: NodeAttentionBalancer.Config | None = None
        """Regroup each training microbatch's documents over the ranks of a node
        so they have about equal attention work (components/data/balance.py).
        None: the microbatches as packed."""
        validator: Validator.Config | None = None
        dump_folder: str = "./outputs"

        create_seed_checkpoint: bool = False
        """Initialize and save an unsharded model-only checkpoint, then exit."""

        override: OverrideConfig = field(default_factory=OverrideConfig)
        """Registered overrides applied to this config tree when the trainer is
        constructed."""

        def __post_init__(self):
            TrainingEngine.Config.__post_init__(self)
            if (
                self.dist_moe_runtime is not None
                and self.precision.param_dtype != "bfloat16"
            ):
                raise ValueError("Dist-MoE requires precision.param_dtype='bfloat16'")
            if self.debug.batch_invariant:
                raise ValueError(
                    "Batch-invariant mode is not needed in supervised learning."
                )

            if self.model is not None:
                validate_model_training_config(
                    self.model,
                    parallelism=self.parallelism,
                    training=self.training,
                    debug=self.debug,
                    activation_remat=self.activation_remat,
                    max_num_documents=self.dataloader.max_num_documents,
                )
            if self.attention_balancer is not None and (
                self.parallelism.tensor_parallel_degree > 1
                or self.parallelism.context_parallel_degree > 1
                or self.parallelism.pipeline_parallel_degree > 1
            ):
                raise ValueError(
                    "attention_balancer regroups documents over a node's "
                    "data-parallel ranks; it needs TP, CP and PP degree 1"
                )
            if self.create_seed_checkpoint and (
                self.checkpointer is None or self.checkpointer.load_only
            ):
                raise ValueError(
                    "create_seed_checkpoint requires a checkpointer with "
                    "load_only=False."
                )

            self._resolve_norm_logging()

        def _resolve_norm_logging(self) -> None:
            """Reconcile DiSCO's norm logging with the metrics config.

            Both settings live on the optimizer, but their validity depends on
            `metrics`, so this is the one place that sees both. Doing it at
            config time means a bad combination fails before the job allocates
            anything, and means the container is built already correct rather
            than patched afterwards.

            No-op for upstream's container (an AdamW run).
            """
            if not isinstance(self.optim.optimizer, DiSCOOptimizersContainer.Config):
                return
            log_norm_freq = self.optim.optimizer.log_norm_freq
            # Norm logging is gated on `should_log`, so norms would otherwise
            # land on lcm(log_freq, log_norm_freq) rather than the requested
            # cadence: log_freq=100 with log_norm_freq=250 silently gives norms
            # every 500 steps. Require the multiple so the configured cadence
            # is the real one.
            if (
                log_norm_freq > 0
                and self.metrics.log_freq > 0
                and log_norm_freq % self.metrics.log_freq != 0
            ):
                raise ValueError(
                    f"optim.optimizer.log_norm_freq ({log_norm_freq}) must be a "
                    f"multiple of metrics.log_freq ({self.metrics.log_freq}); "
                    "norm logging only happens on steps that are also "
                    "metric-logging steps, so otherwise norms land on "
                    "lcm(log_freq, log_norm_freq) instead of the cadence you "
                    "asked for."
                )
            if (
                self.metrics.save_all_shard_ranks
                and not self.metrics.save_for_all_ranks
            ):
                raise ValueError(
                    "metrics.save_all_shard_ranks requires "
                    "metrics.save_for_all_ranks. Without it only one rank "
                    "builds a logger, so skipping the logging all_gather would "
                    "silently drop every parameter owned by the other shard "
                    "ranks."
                )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                val = getattr(self, f.name)
                if hasattr(val, "to_dict"):
                    d[f.name] = val.to_dict()
                elif dataclasses.is_dataclass(val):
                    d[f.name] = asdict(val)
                else:
                    d[f.name] = val
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    config: Config
    engine: TrainingEngine
    engine_cls: type[TrainingEngine] = TrainingEngine

    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    attention_balancer: NodeAttentionBalancer | None
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    gradient_accumulation_steps: int
    num_pp_microbatches: int

    data_mix_metrics: DataMixMetricsLike

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        self.config = config
        config.model = copy.deepcopy(config.model)
        if config.override.imports:
            apply_overrides(config.override, config)
        model_config = config.model

        self.engine = self.engine_cls(
            config,
            model_config=model_config,
            precision=config.precision,
            max_num_documents=config.dataloader.max_num_documents,
            output_dir=config.dump_folder,
        )
        engine = self.engine
        parallelism_context = engine.parallelism_context

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallelism_context.dp_enabled:
            dp_mesh = parallelism_context.get_mesh("dp")
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallelism_context=parallelism_context,
            device_memory_monitor=engine.device_memory_monitor,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            has_quantization=engine.has_quantization,
        )
        self._local_num_flops_since_last_log = 0
        color = self.metrics_processor.color

        self.num_pp_microbatches = (
            config.parallelism.num_pp_microbatches
            if parallelism_context.pp_enabled
            else 1
        )
        num_tokens_per_dp_rank = (
            config.training.num_tokens_per_microbatch_per_dp_rank
            * self.num_pp_microbatches
        )
        num_tokens_per_train_step = config.training.num_tokens_per_train_step
        if num_tokens_per_train_step < 0:
            num_tokens_per_train_step = num_tokens_per_dp_rank * dp_degree
        if num_tokens_per_train_step % (num_tokens_per_dp_rank * dp_degree) != 0:
            raise ValueError(
                "training.num_tokens_per_train_step "
                f"({num_tokens_per_train_step}) must be divisible by the number "
                "of tokens processed globally in one gradient accumulation "
                f"iteration ({num_tokens_per_dp_rank * dp_degree})."
            )
        self.gradient_accumulation_steps = num_tokens_per_train_step // (
            num_tokens_per_dp_rank * dp_degree
        )

        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
        num_tokens_per_microbatch = (
            config.training.num_tokens_per_microbatch_per_dp_rank
        )
        self.dataloader = config.dataloader.build(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=self.tokenizer,
            max_context_length=config.model.max_context_length,
            num_tokens_per_microbatch=num_tokens_per_microbatch,
        )
        self.attention_balancer = (
            config.attention_balancer.build(
                model_config=config.model,
                num_mtp_layers=config.dataloader.num_mtp_layers,
            )
            if config.attention_balancer is not None
            else None
        )

        try:
            engine.initialize(
                extra_states={DATALOADER: self.dataloader},
                hf_assets_path=config.hf_assets_path,
                create_seed_checkpoint=config.create_seed_checkpoint,
            )
        except Exception:
            # config.build() cannot return this partially initialized Trainer,
            # so the outer entrypoint has no object through which to close it.
            engine.close()
            raise

        if parallelism_context.pp_enabled:
            ensure_pp_loss_visible(
                parallelism_context=parallelism_context,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        self.metrics_processor.optimizers = engine.optim.optimizers
        self.metrics_processor.model_parts = engine.model_parts

        disco = _disco_container(engine)
        if disco is not None:
            disco.log_metrics_locally = logs_metrics_locally(config.metrics)

        self.data_mix_metrics = build_data_mix_metrics(self.dataloader)
        logger.info(
            f"mixing weights: {self.data_mix_metrics.get_log_dict_at_step(0)[0]}"
        )

        logger.info(
            "Peak FLOPS used for computing MFU: "
            f"{self.metrics_processor.gpu_peak_flops:.3e}"
        )
        logger.info(
            f"{engine.device.type.upper()} memory usage for model: "
            f"{engine.model_device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({engine.model_device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # Build validator if validation is configured
        if config.validator is not None:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    engine.pp_schedule,
                    engine.pp_has_first_stage,
                    engine.pp_has_last_stage,
                )
                if parallelism_context.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                accelerator_graph_enabled=engine._forward_backward_accelerator_graph_enabled,
                cuda_graph_annotation_config=config.profiler.maybe_cuda_graph_annotation_config(),
                parallelism=config.parallelism,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,
                parallelism_context=parallelism_context,
                loss_fn=engine.loss_fn,
                metrics_processor=self.metrics_processor,
                seq_len=config.model.max_context_length,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"{num_tokens_per_dp_rank} tokens per DP rank, "
            f"{num_tokens_per_train_step} tokens per train step, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"maximum context length {config.model.max_context_length}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.optim.lr_scheduler.warmup_steps})"
        )

    def _is_final_step(self, current_step: int) -> bool:
        """Whether `current_step` is the run's last.

        The final step gets a metrics point (and, with norm logging on, a final
        spectrum snapshot) even when it does not land on a log_freq boundary.
        """
        return current_step == self.config.training.steps

    def microbatch_generator(
        self, data_iterable: Iterable[TrainingMicrobatch]
    ) -> Iterator[TrainingMicrobatch]:
        """Return microbatches while recording data-loading metrics.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                microbatch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            if self.attention_balancer is not None:
                microbatch = self.attention_balancer(microbatch)
            ntokens_microbatch = (
                self.config.training.num_tokens_per_microbatch_per_dp_rank
            )
            self.metrics_processor.ntokens_since_last_log += ntokens_microbatch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield microbatch

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]):
        engine = self.engine
        current_step = engine.num_completed_steps + 1
        # Always log the final step even when it does not land on a log_freq
        # boundary, so a run ends with a metrics point. This has to be decided
        # here, not at the log site: the loss is only accumulated when
        # should_log is true.
        is_final_step = self._is_final_step(current_step)
        should_log = self.metrics_processor.should_log(current_step) or is_final_step
        # Start a new metrics window on the first step after loading and right
        # after the last log or validation, so it leaves out a checkpoint saved
        # or a validation run at that step. should_log() above initializes
        # step_last_log on the first step, so keep this check after it.
        if self.metrics_processor.step_last_log == engine.num_completed_steps:
            self.metrics_processor.reset()
            self._local_num_flops_since_last_log = 0

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallelism_context = engine.parallelism_context
        # DiSCO computes its metrics inside step(), so the step is armed before
        # the forward: the LM-head geometry takes the head's input from it.
        disco = _disco_container(engine)
        if disco is not None:
            rows = disco.begin_step(
                current_step,
                should_log=should_log,
                is_final_step=is_final_step,
                head_input=engine.take_head_input_rows,
            )
            engine.keep_head_input_rows(rows, seed=current_step)

        # All groups form one optimizer step. Each microbatch group forms one
        # complete PP step, or one local forward/backward when PP is disabled.
        microbatch_groups: list[list[TrainingMicrobatch]] = []
        local_loss_token_counts: torch.Tensor | None = None
        local_routing_token_counts: torch.Tensor | None = None
        for _ in range(self.gradient_accumulation_steps):
            microbatch_group = []
            for _ in range(self.num_pp_microbatches):
                with sl.log_trace_span("fetching_batch"):
                    microbatch = next(data_iterator)
                if local_loss_token_counts is None:
                    local_loss_token_counts = torch.zeros_like(
                        microbatch.loss_token_counts
                    )
                    local_routing_token_counts = torch.zeros_like(
                        microbatch.routing_token_counts
                    )
                local_loss_token_counts.add_(microbatch.loss_token_counts)
                assert local_routing_token_counts is not None
                local_routing_token_counts.add_(microbatch.routing_token_counts)
                microbatch_group.append(microbatch)
            microbatch_groups.append(microbatch_group)
        assert local_loss_token_counts is not None
        assert local_routing_token_counts is not None
        local_main_loss_token_count = (
            local_loss_token_counts
            if local_loss_token_counts.ndim == 0
            else local_loss_token_counts[0]
        )
        sl.log_trace_scalar({"local_valid_tokens": int(local_main_loss_token_count)})
        num_loss_objectives = local_loss_token_counts.numel()
        global_token_counts = torch.cat(
            (
                local_loss_token_counts.reshape(-1),
                local_routing_token_counts.reshape(-1),
            )
        ).to(engine.device)
        if parallelism_context.dp_enabled:
            global_token_counts = dist_utils.dist_sum_tensor(
                global_token_counts, parallelism_context.get_mesh("dp")
            )
        global_loss_token_counts = global_token_counts[:num_loss_objectives].reshape(
            local_loss_token_counts.shape
        )
        global_routing_token_counts = global_token_counts[num_loss_objectives:].reshape(
            local_routing_token_counts.shape
        )
        forward_backward_result = engine.forward_backward(
            microbatch_groups=microbatch_groups,
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
        )
        global_main_loss_token_count = (
            global_loss_token_counts
            if global_loss_token_counts.ndim == 0
            else global_loss_token_counts[0]
        )

        # Capture the learning rates used by this optimizer update before the
        # scheduler advances in engine.optim_step().
        lr_metrics = engine.optim.lr_schedulers.get_metrics() if should_log else {}
        grad_norm = engine.optim_step()
        self._local_num_flops_since_last_log += forward_backward_result.num_flops

        # log metrics
        if not should_log:
            return

        accumulated_loss = forward_backward_result.loss
        mean_num_flops = self._local_num_flops_since_last_log

        with sl.log_trace_span("collect_dist_metrics"):
            sl.log_trace_scalar(
                {"global_valid_tokens": int(global_main_loss_token_count)}
            )

            if parallelism_context.dp_cp_enabled:
                loss_mesh = parallelism_context.get_mesh("loss")

                # For global_avg_loss, we want the average loss across all ranks:
                # accumulated_loss = local_loss_sum / global_valid_tokens
                # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
                #                 = sum(accumulated_loss)
                #
                # For global_max_loss, we want the max of local average losses across ranks:
                # local_avg_loss = local_loss_sum / local_valid_tokens
                #                = (accumulated_loss * global_valid_tokens) / local_valid_tokens
                # global_max_loss = max(local_avg_loss)
                local_avg_loss = (
                    accumulated_loss
                    * global_main_loss_token_count
                    / local_main_loss_token_count
                )
                global_avg_loss, global_max_loss, global_ntokens_seen = (
                    dist_utils.dist_sum(accumulated_loss, loss_mesh),
                    dist_utils.dist_max(local_avg_loss, loss_mesh),
                    dist_utils.dist_sum(
                        torch.tensor(
                            engine.ntokens_seen,
                            dtype=torch.int64,
                            device=engine.device,
                        ),
                        loss_mesh,
                    ),
                )
                if parallelism_context.dp_enabled:
                    dp_mesh = parallelism_context.get_mesh("dp")
                    mean_num_flops = (
                        dist_utils.dist_sum(
                            torch.tensor(
                                mean_num_flops,
                                dtype=torch.float64,
                                device=engine.device,
                            ),
                            dp_mesh,
                        )
                        / dp_mesh.size()
                    )
            else:
                global_avg_loss = global_max_loss = float(accumulated_loss.item())
                global_ntokens_seen = engine.ntokens_seen

            # The loss function's own metrics (e.g. the MTP loss's terms), with
            # ChunkedLossWrapper's suffix rule: mean / fraction keys are
            # normalized as the loss, so they sum over the microbatches and the
            # loss mesh; "/max" and "/min" keys take their max / min.
            loss_fn_metrics: dict[str, torch.Tensor] = {}
            for group_metrics in forward_backward_result.loss_metrics:
                for microbatch_metrics in group_metrics:
                    for key, value in microbatch_metrics.items():
                        value = value.detach().float()
                        if key not in loss_fn_metrics:
                            loss_fn_metrics[key] = value
                        elif key.endswith("/max"):
                            loss_fn_metrics[key] = torch.maximum(
                                loss_fn_metrics[key], value
                            )
                        elif key.endswith("/min"):
                            loss_fn_metrics[key] = torch.minimum(
                                loss_fn_metrics[key], value
                            )
                        else:
                            loss_fn_metrics[key] = loss_fn_metrics[key] + value
            loss_fn_logged = {}
            for key, value in sorted(loss_fn_metrics.items()):
                if not parallelism_context.dp_cp_enabled:
                    loss_fn_logged[key] = float(value.item())
                elif key.endswith("/min"):
                    loss_fn_logged[key] = -dist_utils.dist_max(
                        -value, parallelism_context.get_mesh("loss")
                    )
                else:
                    reduce = (
                        dist_utils.dist_max
                        if key.endswith("/max")
                        else dist_utils.dist_sum
                    )
                    loss_fn_logged[key] = reduce(
                        value, parallelism_context.get_mesh("loss")
                    )

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
            **collect_aux_loss_metrics(parallelism_context),
            **loss_fn_logged,
        }
        # No None guard: "nothing to report" is the null object's empty
        # dicts, not a missing attribute.
        data_mix, data_docs = self.data_mix_metrics.get_log_dict_at_step(
            engine.num_completed_steps
        )
        extra_metrics.update(data_mix)
        extra_metrics.update({k: int(v) for k, v in data_docs.items()})
        if disco is not None:
            extra_metrics.update(disco.collect_metrics(engine.num_completed_steps))
        self.metrics_processor.log(
            engine.num_completed_steps,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
            num_flops=mean_num_flops,
        )

    @record
    def train(self):
        config = self.config
        engine = self.engine

        sl.log_trace_instant("training_start")

        engine.load_checkpoint()

        # Capture loaded step for relative_step calculation.
        # After checkpoint load, this is the restored number of completed updates.
        loaded_step = engine.num_completed_steps

        averages = engine.averages
        if (
            averages is not None
            and averages.config.start_from_gamma is not None
            and engine.checkpointer.loaded_from_initial_path
        ):
            # angular WSD: the branch starts from the renormalised average of the
            # loaded checkpoint, with its raw optimizer state
            averages.start_from_average()
            logger.info(
                f"averages: the weights are the gamma {averages.config.start_from_gamma:g} "
                f"average at step {engine.num_completed_steps}"
            )
            if self.config.validator is not None:
                self.validator.validate(
                    engine.model_parts, engine.num_completed_steps, tag="avgstart"
                )
        if averages is not None and averages.config.score_files:
            if self.config.validator is None:
                raise ValueError("averages.score_files needs the validator")
            averages.score(self.validator, engine.num_completed_steps)
            logger.info("averages: scored the score_files; no training")
            return

        logger.info(f"Training starts at step {engine.num_completed_steps + 1}")

        engine.start_profiler()
        try:
            data_iterator = self.microbatch_generator(self.dataloader)
            while self.should_continue_training():
                current_step = engine.num_completed_steps + 1
                sl.set_step(current_step, relative_step=current_step - loaded_step)

                with sl.log_trace_span("step"):
                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break

                    engine.save_checkpoint(
                        last_step=(engine.num_completed_steps == config.training.steps)
                    )

                    if config.validator is not None and self.validator.should_validate(
                        engine.num_completed_steps,
                        total_steps=self.config.training.steps,
                    ):
                        self.validator.validate(
                            engine.model_parts, engine.num_completed_steps
                        )
                        if engine.averages is not None:
                            engine.averages.validate(
                                self.validator, engine.num_completed_steps
                            )

                    engine.step_profiler()

                    # Reduce timeout after the first train step of THIS process
                    # (assuming lazy init and compilation are finished). Use the
                    # relative step so this fires on resumed runs too.
                    if engine.num_completed_steps - loaded_step == 1:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(
                                seconds=config.comm.train_timeout_seconds
                            ),
                            parallelism_context=engine.parallelism_context,
                        )
        finally:
            # The entry point also calls close() for checkpoint and graph
            # cleanup; close the profiler here so direct train() callers get
            # balanced lifecycle handling.
            engine.close_profiler()

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.engine.num_completed_steps < self.config.training.steps

    def close(self) -> None:
        # `and self.dataloader` would call __len__ on an IterableDataset-backed
        # loader, which raises TypeError. Test for presence, not truthiness.
        if getattr(self, "dataloader", None) is not None:
            self.dataloader.close()
        if hasattr(self, "engine"):
            self.engine.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
