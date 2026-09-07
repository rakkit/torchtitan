# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import datetime

import functools
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field

from typing import Annotated, Any, cast

import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.data_mix_scheduler import (
    build_data_mix_scheduler,
    DataMixScheduler,
)
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.ema import EMAOptimizersContainer
from torchtitan.components.loss import IGNORE_INDEX, LossFunction, moe_loss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.hf_datasets.text_datasets import infer_dataloader_snapshot_every_n_steps
from torchtitan.models.common.decoder import Decoder
from torchtitan.optimizers import norm_helper
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
    ProfilingConfig,
)


class Trainer(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """
        Default container for training configuration.
        """

        # NOTE: model_spec is suppressed from tyro CLI parsing and is always
        # set programmatically by the model registry before Trainer construction.
        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        dump_folder: str = "./outputs"
        """Folder to dump job outputs"""

        profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        # TODO: remove the optional flag once Flux tokenizer is modeled properly
        tokenizer: BaseTokenizer.Config | None = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        model_converters: ModelConvertersContainer.Config = field(
            default_factory=ModelConvertersContainer.Config
        )
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        ema_weights: EMAOptimizersContainer.Config = field(
            default_factory=EMAOptimizersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointConfig = field(
            default_factory=ActivationCheckpointConfig
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)
        debug: DebugConfig = field(default_factory=DebugConfig)

        def __post_init__(self):
            if isinstance(self.optimizer, OptimizersInBackwardContainer.Config):
                if self.parallelism.expert_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Expert Parallel."
                    )
                if self.parallelism.pipeline_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Pipeline Parallel."
                    )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    assert self.model_spec is not None
                    # ModelSpec contains callables that can't be serialized
                    model_spec_dict = {
                        "name": self.model_spec.name,
                        "flavor": self.model_spec.flavor,
                    }
                    # Keep a JSON-serializable snapshot of model arguments so
                    # overrides under model_spec.model are tracked in saved configs.
                    if dataclasses.is_dataclass(self.model_spec.model):
                        model_spec_dict["model"] = asdict(self.model_spec.model)
                    d["model_spec"] = model_spec_dict
                else:
                    d[f.name] = (
                        asdict(getattr(self, f.name))
                        if dataclasses.is_dataclass(getattr(self, f.name))
                        else getattr(self, f.name)
                    )
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            # if self.debug.save_config_file:
            config_file = os.path.join(
                self.dump_folder,
                "job_config_"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M")
                + ".json",
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

    # core configs
    config: Config
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: BaseTokenizer | None
    dataloader: BaseDataLoader
    model_config: BaseModel.Config
    # TODO: we should make this list[BaseModel / Decoder] but this will affect many components.
    # will do this in a separate PR
    data_mix_scheduler: DataMixScheduler
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    ema_optimizer: EMAOptimizersContainer
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.TrainContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int

    # used for loggering, not need to put it in the stateful class
    prev_data_sampled_tensor: torch.Tensor

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )
        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # build tokenizer
        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )

        snapshot_every_n_steps = infer_dataloader_snapshot_every_n_steps(
            checkpoint_enabled=config.checkpoint.enable,
            checkpoint_interval=config.checkpoint.interval,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        # build dataloader
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
            snapshot_every_n_steps=snapshot_every_n_steps,
            seed=config.debug.seed,
        )

        if (
            getattr(config.dataloader, "pack_strategy", "greedy") == "best_fit"
            and config.training.all_tokens_valid
        ):
            raise ValueError(
                "training.all_tokens_valid=True is incompatible with "
                "dataloader.pack_strategy='best_fit': best-fit packing emits "
                "IGNORE_INDEX labels on pad positions, so not every label is "
                "valid. Set training.all_tokens_valid=False."
            )

        mixing_scheduler_configs = getattr(
            config.dataloader, "data_mixing_scheduler_configs", None
        )

        self.data_mix_scheduler = build_data_mix_scheduler(
            self.dataloader, mixing_scheduler_configs, config.training.steps
        )
        self.data_mix_scheduler.dump_mixing_configs(config.dump_folder)
        self.data_mix_scheduler.step(0)
        logger.info(
            f"mixing weights at step 0: {self.data_mix_scheduler.get_log_dict_at_step(0)}"
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            trainer_config=config,
        )
        self.model_config = model_config

        logger.info(
            f"Building {model_spec.name} {model_spec.flavor} "
            f"with {json.dumps(dataclasses.asdict(model_config), indent=2, ensure_ascii=False)}"
        )
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        logger.info(f"model: {model}")

        # Build the collection of model converters. No-op if converters empty
        model_compile_enabled = (
            config.compile.enable and "model" in config.compile.components
        )
        model_converters = config.model_converters.build(
            parallel_dims=parallel_dims,
            model_compile_enabled=model_compile_enabled,
        )
        model_converters.convert(model)

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            active_param,
            embedding_param,
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} - "
            f"Flops: {self.metrics_processor.num_flops_per_token:,}"
        )
        logger.info(
            f"{color.red}total parameters: {model_param_count:,} | "
            f"active: {active_param:,} | embedding: {embedding_param:,} {color.reset}"
        )
        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = model_spec.build_loss_fn(
            config.compile, parallel_dims=parallel_dims
        )

        pre_moe_loss_fn = self.loss_fn
        self.loss_fn = functools.partial(
            moe_loss,
            loss_fn=pre_moe_loss_fn,
            grad_accumulation_steps=self.gradient_accumulation_steps,
        )

        skip_weight_init = CheckpointManager.can_skip_weight_init(config)

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not model_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {model_spec.name} "
                    f"does not support pipelining"
                )

            # apply both Pipeline Parallel and SPMD-style scaling techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model_spec.pipelining_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                model_converters=config.model_converters,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
                device=self.device,
                model_config=model_config,
                parallelize_fn=model_spec.parallelize_fn,
                loss_fn=self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    cast(Decoder, m).init_weights(
                        buffer_device=buffer_device, skip_init=skip_weight_init
                    )
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(
                parallel_dims=parallel_dims,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        else:
            # apply Tensor/Context/Expert Parallel, activation checkpointing, torch.compile, Data Parallel
            model = model_spec.parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                model_converters=config.model_converters,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                cast(BaseModel, model).init_weights(
                    buffer_device=buffer_device, skip_init=skip_weight_init
                )
            model.train()

            self.model_parts = [model]

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(
            model_parts=self.model_parts, parallel_dims=parallel_dims
        )
        # Per-rank metric logging: DiSCO may only skip the logging all_gather
        # when the metrics side genuinely gives EVERY shard rank a logger.
        #
        # `save_all_shard_ranks` alone is not enough. It is applied inside
        # `_build_metric_logger` only after `should_log` is already true, and
        # with `save_for_all_ranks=False` that has been narrowed to a single
        # global rank. Enabling local mode on the raw flag would have every rank
        # skip the gather while only one rank owns a logger -- so that rank logs
        # its own 1/N of the parameters and the rest are silently dropped.
        #
        # It also has to be decided from config alone, identically on every
        # rank: the gather is a collective, so if some ranks skipped it and
        # others did not the job would hang rather than misreport.
        _m = config.metrics
        _logging_on = bool(
            getattr(_m, "enable_wandb", False)
            or getattr(_m, "enable_tensorboard", False)
        )
        if getattr(_m, "save_all_shard_ranks", False) and not getattr(
            _m, "save_for_all_ranks", False
        ):
            raise ValueError(
                "metrics.save_all_shard_ranks requires metrics.save_for_all_ranks. "
                "Without it only one rank builds a logger, so skipping the "
                "logging all_gather would silently drop every parameter owned by "
                "the other shard ranks."
            )
        self.optimizers.log_metrics_locally = bool(
            getattr(_m, "save_all_shard_ranks", False)
            and getattr(_m, "save_for_all_ranks", False)
            and _logging_on
        )
        if self.optimizers.log_metrics_locally:
            # The invariant that actually matters: every rank owning a distinct
            # slice of the metrics must hold a logger to write it to. Both
            # sides now derive that from `rank_owns_metrics_shard`, so this is
            # inert -- it exists to make a future divergence fail loudly at
            # init instead of silently producing a partial dashboard, which is
            # how the pure-DDP case went unnoticed (ownership spread over
            # dp_replicate while only replica 0 got a logger).
            _owns_shard = dist_utils.rank_owns_metrics_shard(parallel_dims)
            # `_build_metric_logger` returns a bare `BaseLogger` (a no-op with
            # no `number_of_loggers`) on ranks that do not log, and a
            # `LoggerContainer` on ranks that do -- so this must not assume the
            # container type. Getting that wrong crashed every non-logging rank
            # at init on the first HSDP run.
            _logger_obj = self.metrics_processor.logger
            _has_logger = getattr(_logger_obj, "number_of_loggers", 0) > 0
            # Only the direction that loses data is fatal: this rank owns a
            # slice nobody else will log, and has nowhere to put it. The
            # reverse (a logger with nothing to log) is harmless, and firing on
            # it would turn a soft wandb failure -- an empty LoggerContainer
            # because the backend could not be constructed -- into an
            # init-time abort of the whole job.
            if _owns_shard and not _has_logger:
                raise RuntimeError(
                    "per-rank metric logging would silently drop data on this "
                    "rank: it owns a disjoint slice of the per-parameter "
                    "metrics but has no logger to write them to. The "
                    "optimizer's ownership split and "
                    "metrics._build_metric_logger's rank predicate have "
                    "diverged; both must come from "
                    "distributed.utils.rank_owns_metrics_shard."
                )
            elif _has_logger and not _owns_shard:
                logger.warning(
                    "this rank has a metrics logger but owns no metrics shard; "
                    "it will log global scalars only. Harmless, but it means "
                    "the two rank predicates disagree."
                )
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts
        self.optimizers.norms_to_log = norm_helper.get_norms_to_log(
            config.metrics.norms_to_log
        )
        self.optimizers.gram_level = config.metrics.gram_level
        # enable_plot/enable_export live on OptimizersContainer.Config itself
        # (config.optimizer.*) and are already set by its __init__; only
        # export_dir depends on the top-level dump_folder, so patch that in.
        self.optimizers.spectrum_logging_config = (
            self.optimizers.spectrum_logging_config._replace(
                export_dir=os.path.join(config.dump_folder, "spectrum_export")
            )
        )
        self.optimizers.gram_vector_logging_config = (
            self.optimizers.gram_vector_logging_config._replace(
                export_dir=os.path.join(config.dump_folder, "gram_export")
            )
        )

        # Online EMA of model weights (e.g. for cheap mid-WSD-training eval
        # without a full LR decay). Always built and the hook always
        # registered; EMAOptimizersContainer internally no-ops everything
        # when config.ema_weights.enable is False.
        self.ema_optimizer = config.ema_weights.build(model_parts=self.model_parts)
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: self.ema_optimizer.step(self.step)
        )

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0
        self.prev_data_sampled_tensor = None

        self.checkpointer = config.checkpoint.build(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            ema_optimizer=self.ema_optimizer,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not config.parallelism.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled)
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            config.training.mixed_precision_param,
            device_type,
        )

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
                enable_token_mask_for_moe=config.training.enable_token_mask_for_moe,
                seed=config.debug.seed,
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        parallelism_config = config.parallelism
        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors (all keys except
                    "input" from input_dict). These are passed to the model forward
                    but are NOT forwarded across pipeline parallel stages.
                - extra_kwargs: Dict of additional keyword arguments for model forward.
                    These ARE forwarded across pipeline parallel stages. Contains
                    attention_masks if flex attention is enabled.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage.
        """
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {}

        # Always delegate mask creation to the model's get_attention_masks().
        # Checking base attn_backend is insufficient: per-layer patterns (swa_pattern,
        # rope_pattern) may promote individual layers to flex even when the base
        # config says "sdpa". The model returns None for all-SDPA (no overhead),
        # a dict for flex/mixed, or delegates to super() for varlen.
        model = cast(Decoder, self.model_parts[0])
        if hasattr(model, "get_attention_masks"):
            attention_masks = model.get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
                extra_inputs=extra_inputs,
            )
            if attention_masks is not None:
                extra_kwargs["attention_masks"] = attention_masks

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                self.config.parallelism.context_parallel_load_balancer,
            )
            # TODO(SFT x CP): SFT loader emits per-doc "positions" in
            # extra_inputs while prepare_context_parallel_input also writes
            # extra_kwargs["positions"] (sequential arange). The forward at
            # L752 unpacks both and raises TypeError on the duplicate kwarg.
            # Temporary workaround: drop loader-provided positions so RoPE
            # uses CP's sequential positions. Per-doc RoPE on multi-doc
            # packed bins is therefore not semantically correct, but the
            # attention mask was already built from doc-local positions in
            # get_attention_masks above. Proper fix: make
            # prepare_context_parallel_input use loader-provided positions
            # (shard them through cp_shard like inputs/labels) so RoPE
            # matches doc boundaries.
            extra_inputs.pop("positions", None)

        # Build the MoE token mask AFTER CP sharding so its token layout matches
        # the CP-sharded labels / hidden states (same shard + same load-balancer
        # reorder). Building it before CP leaves it full-length and breaks
        # idx[m] in the MoE router under CP.
        if self.config.training.enable_token_mask_for_moe:
            extra_kwargs["loss_mask"] = labels != IGNORE_INDEX

        return inputs, labels, extra_inputs, extra_kwargs

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                # Rescale PP loss to be "local loss sum / global valid tokens)
                # because each microbathes could have different number of valid tokens
                (torch.sum(torch.stack(losses)) / global_valid_tokens).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            assert len(model_parts) == 1
            with self.train_context():
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                    # Compute loss sum (reduction='sum')
                    loss_sum = self.loss_fn(pred, labels)

                    # Scale the loss by the inverse of the total weight denominator before backward
                    # This ensures gradients are properly normalized across all microbatches
                    loss = loss_sum / global_valid_tokens

                # need to free pred before bwd to avoid peaking memory
                del pred
                loss.backward()

        # The returned loss here is local SUM loss / global_valid_tokens
        return loss

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and compute global valid token count.
        microbatches = []
        data_load_start = time.perf_counter()
        if self.config.training.all_tokens_valid:
            # Fast path: every label is valid and every rank has the same fixed
            # token count. Count from the actual CPU microbatches to avoid any
            # communication while staying robust to shape drift.
            local_valid_tokens = 0

            for _microbatch in range(self.gradient_accumulation_steps):
                input_dict, labels = next(data_iterator)
                local_valid_tokens += labels.numel()
                microbatches.append((input_dict, labels))

            batch_degree = (
                parallel_dims.get_mesh("batch").size()
                if parallel_dims.dp_enabled
                else 1
            )
            local_valid_tokens = torch.tensor(
                float(local_valid_tokens), device=self.device
            )
            global_valid_tokens = local_valid_tokens * batch_degree
            global_total_tokens = global_valid_tokens
        else:
            local_valid_tokens = torch.tensor(0, dtype=torch.int64)
            local_total_tokens = torch.tensor(0, dtype=torch.int64)
            for _microbatch in range(self.gradient_accumulation_steps):
                input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                local_total_tokens += labels.numel()
                microbatches.append((input_dict, labels))

            # All-reduce to get global token count across DP ranks
            # Move to GPU for distributed communication
            local_valid_tokens = local_valid_tokens.to(self.device)
            local_total_tokens = local_total_tokens.to(self.device)
            if parallel_dims.dp_enabled:
                batch_mesh = parallel_dims.get_mesh("batch")
                global_valid_tokens = dist_utils.dist_sum(
                    local_valid_tokens, batch_mesh
                )
                global_total_tokens = dist_utils.dist_sum(
                    local_total_tokens, batch_mesh
                )
            else:
                global_valid_tokens = local_valid_tokens.float()
                global_total_tokens = local_total_tokens.float()

        self.metrics_processor.data_loading_times.append(
            time.perf_counter() - data_load_start
        )
        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        fwd_bwd_start = time.perf_counter()
        for input_dict, labels in microbatches:
            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                # pyrefly: ignore [bad-argument-type]
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

        self.metrics_processor.fwd_bwd_times.append(time.perf_counter() - fwd_bwd_start)

        grad_norm = None

        if self.config.training.max_norm > 0:
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
        self.checkpointer.maybe_wait_for_staging()

        # Here we let the optimizer know that we need to calculate the
        # norm at the next step. Also always do it on the very last training
        # step (even if it doesn't land on a log_norm_freq/log_freq boundary)
        # so we get a final snapshot of the spectrum/norms at the end of
        # training rather than possibly missing it entirely.
        is_last_training_step = self.step == self.config.training.steps
        need_to_calculate_norm = self.config.metrics.log_norm_freq > 0 and (
            is_last_training_step
            or (
                self.metrics_processor.should_log(self.step)
                and (
                    self.step == 1 or self.step % self.config.metrics.log_norm_freq == 0
                )
            )
        )
        if need_to_calculate_norm:
            self.optimizers.calculate_norm_at_next_step()

        optim_step_start = time.perf_counter()

        self.optimizers.step()
        self.lr_schedulers.step()
        self.data_mix_scheduler.step(self.step + 1)
        self.metrics_processor.optim_step_times.append(
            time.perf_counter() - optim_step_start
        )
        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not (self.metrics_processor.should_log(self.step) or is_last_training_step):
            return

        data_mix, data_docs, data_tokens = self.data_mix_scheduler.get_log_dict_at_step(
            self.step
        )

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # For global_avg_loss, we want the average loss across all ranks:
            # loss = local_loss_sum / global_valid_tokens
            # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
            #                 = sum(loss)
            #
            # For global_max_loss, we want the max of local average losses across ranks:
            # local_avg_loss = local_loss_sum / local_valid_tokens
            #                = (loss * global_valid_tokens) / local_valid_tokens
            # global_max_loss = max(local_avg_loss)
            local_avg_loss = loss * global_valid_tokens / local_valid_tokens
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(local_avg_loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
            ##############################################################
            # Fuse doc counts + token counts into one tensor → single dist_sum.
            doc_keys = sorted(data_docs.keys())
            token_keys = sorted(data_tokens.keys())
            N = len(doc_keys)
            fused = torch.stack(
                [data_docs[k] for k in doc_keys] + [data_tokens[k] for k in token_keys]
            ).to(self.device)

            fused = dist_utils.dist_sum(
                fused,
                parallel_dims.get_optional_mesh("loss"),
                keep_tensor=True,
            )
            # On the first log of a Trainer instance (fresh start OR resume)
            # we don't have a previous reference, so initialize prev to the
            # current cumulative. The first computed delta is then 0 (one log
            # point with all-zero actual_*_ratio) instead of dumping the whole
            # carried-over cumulative as a spurious "this interval" value —
            # which on resume would show phase-1's historical mix and
            # subsequently amplify into several noisy logs as per-interval
            # deltas catch up. Shape guard handles a hypothetical resume with
            # a different number of datasets without raising.
            if (
                self.prev_data_sampled_tensor is None
                or self.prev_data_sampled_tensor.shape != fused.shape
            ):
                self.prev_data_sampled_tensor = fused.clone()
            delta = fused - self.prev_data_sampled_tensor
            self.prev_data_sampled_tensor = fused.clone()

            delta_docs, delta_tokens = delta[:N], delta[N:]

            # Cumulative counts (summed across ranks)
            data_docs = {k: int(fused[i].item()) for i, k in enumerate(doc_keys)}
            data_tokens = {
                k: int(fused[N + i].item()) for i, k in enumerate(token_keys)
            }

            # Actual ratios (% of this logging interval)
            eps = 1e-20
            actual_doc_ratio = delta_docs / (delta_docs.sum() / 100 + eps)
            actual_token_ratio = delta_tokens / (delta_tokens.sum() / 100 + eps)
            actual_doc_ratio_dict = {
                k.replace("data_docs/", "actual_doc_ratio/"): actual_doc_ratio[i].item()
                for i, k in enumerate(doc_keys)
            }
            actual_token_ratio_dict = {
                k.replace("data_tokens/", "actual_token_ratio/"): actual_token_ratio[
                    i
                ].item()
                for i, k in enumerate(token_keys)
            }
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen
            actual_doc_ratio_dict = {}
            actual_token_ratio_dict = {}

        global_valid_tokens_f = float(global_valid_tokens)
        global_total_tokens_f = float(global_total_tokens)
        valid_token_fraction = (
            global_valid_tokens_f / global_total_tokens_f
            if global_total_tokens_f > 0
            else 1.0
        )
        extra_metrics = {
            "training_metrics/n_tokens_seen": global_ntokens_seen,
            "training_metrics/lr": lr,
            "training_metrics/valid_token_fraction": valid_token_fraction,
            "training_metrics/padding_fraction": 1.0 - valid_token_fraction,
        }
        extra_metrics.update(self.optimizers.get_lrs())
        extra_metrics.update(data_mix)
        extra_metrics.update(data_docs)
        extra_metrics.update(data_tokens)
        extra_metrics.update(actual_doc_ratio_dict)
        extra_metrics.update(actual_token_ratio_dict)

        if need_to_calculate_norm:
            param_norms = self.optimizers.get_parameter_norms(step=self.step)
            extra_metrics.update(param_norms)

        self.optimizers.join_log_queue()
        for model_part in self.model_parts:
            for layer in model_part.layers.values():
                if not hasattr(layer, "moe") or not hasattr(
                    layer.moe, "_log_expert_metrics"
                ):
                    continue
                extra_metrics.update(layer.moe._log_expert_metrics)

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item() if grad_norm is not None else grad_norm,
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        config = self.config

        self.checkpointer.load(step=config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")
        self.data_mix_scheduler.step(self.step)
        # lets over write the weights for the first step in case the weights are not set in configs
        # cause for now the data mixing scheduler is stateless
        with (
            maybe_enable_profiling(
                config.profiling,
                global_step=self.step,
                base_folder=config.dump_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                config.profiling,
                global_step=self.step,
                base_folder=config.dump_folder,
            ) as memory_profiler,
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                self.checkpointer.save(
                    self.step, last_step=(self.step == config.training.steps)
                )

                # Run validation if validator is available
                if self.config.validator.enable and self.validator.should_validate(
                    self.step, total_steps=config.training.steps
                ):
                    self.validator.validate(self.model_parts, self.step)

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=datetime.timedelta(
                            seconds=config.comm.train_timeout_seconds
                        ),
                        parallel_dims=self.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    def close(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
