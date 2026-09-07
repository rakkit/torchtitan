# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import rank_owns_metrics_shard
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, device_module, device_type, NoColor


# named tuple for passing device memory stats for logging
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        # pyrefly: ignore [read-only]
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_index = device_module.current_device()
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(self, log_dir: str, tag: str | None = None):
        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            if not isinstance(v, (int, float, torch.Tensor)):
                # optimizers/spectrum_logging.py's process_norms_for_logging
                # (called from OptimizersContainer.get_parameter_norms())
                # builds wandb-specific objects (wandb.Image/wandb.Histogram)
                # for spectrum entries, since this codebase only logs to
                # W&B. TensorBoard has no use for those — skip rather than
                # crash on a type it can't pass to add_scalar.
                continue
            self.writer.add_scalar(tag, v, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(
        self,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
        project: str | None = None,
        group: str | None = None,
        name: str | None = None,
    ):
        # Import wandb here to avoid startup import
        import wandb

        self.wandb = wandb
        self.tag = tag

        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)

        tags = tag or os.getenv("WANDB_RUN_TAGS", None)
        group = group or os.getenv("WANDB_RUN_GROUP", None)
        name = name or os.getenv("WANDB_RUN_NAME", None)
        project = project or os.getenv("WANDB_PROJECT", "torchtitan")

        self.wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=project,
            name=name,
            id=os.getenv("WANDB_RUN_ID", None),
            notes=os.getenv("WANDB_RUN_NOTES", None),
            tags=tags,
            group=group,
            job_type=os.getenv("WANDB_RUN_JOB_TYPE", None),
            resume_from=os.getenv("WANDB_RESUME_FROM", None),
            fork_from=os.getenv("WANDB_FORK_FROM", None),
            dir=log_dir,
            config=config_dict,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        # Spectrum entries arrive already as wandb.Image/wandb.Histogram
        # objects — see optimizers/spectrum_logging.py's
        # process_norms_for_logging(), called from
        # OptimizersContainer.get_parameter_norms() — so every value here
        # is already exactly what wandb.log() expects, no dispatch needed.
        wandb_metrics = (
            metrics
            if self.tag is None
            else {f"{self.tag}/{k}": v for k, v in metrics.items()}
        )
        self.wandb.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


class LoggerContainer(BaseLogger):
    """Container to call all loggers enabled in the job config."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    def close(self) -> None:
        for logger_instance in self._loggers:
            logger_instance.close()


def ensure_pp_loss_visible(
    *, parallel_dims: ParallelDims, pp_schedule: str, color: Color | NoColor
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.

    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    # Check if the loss-visible rank is included in LOG_RANK environment variable
    env_logged_ranks = os.environ.get("LOG_RANK", "").split(",")
    if env_logged_ranks == [""]:
        env_logged_ranks = []

    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            f"{color.red}Pipeline Parallel loss is not visible. "
            f"Please add {color.yellow}rank {loss_visible_rank}{color.red} "
            f"to LOG_RANK environment variable in run_train.sh.{color.reset}"
        )


def _get_metrics_rank(
    *,
    parallel_dims: ParallelDims,
    pp_schedule: str,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


class MetricsProcessor(Configurable):
    """Metrics processor to processes the metrics and log metrics.

    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard or WandB.

    Args:
        config (Config): Metrics configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        dump_folder (str): Base folder for log output.
        pp_schedule (str): Pipeline parallel schedule name.
        ft_enable (bool): Whether fault tolerance is enabled.
        ft_replica_id (int): Fault tolerance replica ID.
        config_dict (dict | None): Full job config dict for WandB logging.
        tag (str | None): Tag to use for TensorBoard or WandB. Defaults to None.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log metrics to TensorBoard, in iterations"""

        log_norm_freq: int = 0
        """How often to log parameter norm metrics to TensorBoard, in iterations"""

        norms_to_log: list[str] = field(default_factory=lambda: ["default"])
        """
        Which parameter norms to log. If "all" or "everything" is specified,
        log all available norms. If "default" is specified, use the following:
        - "rms_to_rms"
        - "l1_to_rms"
        - "rms_to_inf"
        - "supremum"
        - "condition_number"
        """

        gram_level: int = 0
        """
        Level of Gram-based weight/momentum-geometry metrics to compute and
        log (see optimizers/gram_helper.py), gated by the same log_norm_freq
        cadence as norms_to_log:
        - 0: off (default; no-op, zero overhead).
        - 1: cheap O(m^2) entrywise geometry metrics.
        - 2: adds O(m^3) spectral metrics (cumulative with level 1).
        - 3: adds whitened/generalised metrics, most numerically sensitive
             (cumulative with levels 1-2).
        """

        enable_tensorboard: bool = False
        """Whether to log metrics to TensorBoard"""

        disable_color_printing: bool = False
        """Whether to disable color printing in logs"""

        save_tb_folder: str = "tb"
        """Folder to dump TensorBoard states"""

        save_for_all_ranks: bool = False
        """
        Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
        When this option is False and pipeline_parallel_degree is > 1, the metrics
        component uses the 0th rank of the last stage pipeline group, which is the
        only stage that computes loss metrics.
        """

        save_all_shard_ranks: bool = False
        """
        Whether every FSDP/EP shard rank saves its own metrics, instead of one
        rank saving metrics gathered from all of them.

        Logs on exactly the ranks that own a distinct slice of the metrics --
        i.e. the mesh DiSCO partitions parameter ownership over is opened up,
        and local rank 0 is required in every other mesh (see
        `distributed/utils.rank_owns_metrics_shard`):

          * with FSDP/EP: any fsdp/ep rank, at dp_replicate rank 0 and tp
            rank 0. Replicas hold bit-identical copies, so only one logs.
          * pure DDP (no dp_shard/cp): ownership is spread over dp_replicate
            itself, so *every* dp_replicate rank logs, at tp rank 0.

        Contrast `save_first_dp_and_tp`, which narrows to `loss`-mesh local
        rank 0, and the `loss` mesh is dp_replicate x dp_shard x cp, so it
        excludes every other shard rank.

        Each shard rank owns a disjoint subset of the parameters and already
        computes exactly that subset's metrics, so the union across ranks is
        identical to what the single-rank path logs. Setting this lets DiSCO
        skip the logging all_gather and, more importantly, stops one rank
        building the metrics dict for the *whole* model: for qwen30b-a3b that is
        845,664 entries per logging step on one rank versus 13,213 per rank
        across 64 shards.

        The cost is one W&B run per shard rank (`base_log_dir` already gets a
        `rank_{n}` suffix), so a full picture means reading N runs.
        """

        save_first_dp_and_tp: bool = False
        """
        Whether to save metrics only for DP+CP+TP rank 0, meaning that the
        0th rank of all PP stages will end up saving metrics.
        """

        enable_wandb: bool = False
        """Whether to log metrics to Weights & Biases"""

        wandb_project: str | None = None
        """
        Weights & Biases project name. Use "torchtitan" if neither this is given
        nor the `WANDB_PROJECT` environment variable set.
        """

        wandb_group: str | None = None
        """Weights & Biases group name"""

        wandb_name: str | None = None
        """
        Weights & Biases run name. Use `None` if neither this is given
        nor the `WANDB_RUN_NAME` environment variable set.
        """

    config: Config
    logger: BaseLogger
    parallel_dims: ParallelDims
    device_memory_monitor: DeviceMemoryMonitor
    color: utils.NoColor | utils.Color

    gpu_peak_flops: float
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float

    num_flops_per_token: int
    optimizers: OptimizersContainer | None
    lr_schedulers: LRSchedulersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        dump_folder: str = "./outputs",
        pp_schedule: str = "1F1B",
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        self.logger = self._build_metric_logger(
            config=config,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
            pp_schedule=pp_schedule,
            ft_enable=ft_enable,
            ft_replica_id=ft_replica_id,
            config_dict=config_dict,
            tag=tag,
        )
        self.parallel_dims = parallel_dims
        self.config = config
        self.device_memory_monitor = build_device_memory_monitor()
        # used for colorful printing
        self.color = utils.NoColor() if config.disable_color_printing else utils.Color()

        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.optim_step_times = []
        self.fwd_bwd_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        # These variables have to be set later as they depend on other components or model.
        self.num_flops_per_token = -1
        self.optimizers = None
        self.lr_schedulers = None
        self.model_parts = None

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.config.log_freq == 0

    def _build_metric_logger(
        self,
        *,
        config: Config,
        parallel_dims: ParallelDims,
        dump_folder: str,
        pp_schedule: str,
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        """
        Build an appropriate metric logger based on configuration.
        """
        # Log initial config state
        logger.debug(
            f"Building logger with config: wandb={config.enable_wandb}, "
            f"tensorboard={config.enable_tensorboard}"
        )

        # Check if any logging backend is enabled
        has_logging_enabled = config.enable_tensorboard or config.enable_wandb

        # Determine if this rank should log
        should_log = has_logging_enabled
        if (not config.save_for_all_ranks) and should_log:
            metrics_rank = _get_metrics_rank(
                parallel_dims=parallel_dims, pp_schedule=pp_schedule
            )
            should_log = torch.distributed.get_rank() == metrics_rank

        if config.save_all_shard_ranks and should_log:
            # Exactly the ranks that own a distinct slice of the per-parameter
            # metrics -- see `rank_owns_metrics_shard`. Under FSDP/EP that is
            # any fsdp rank at dp_replicate 0; under pure DDP the shard
            # dimension IS dp_replicate, so every replica logs. Hard-coding
            # "dp_replicate rank 0" here (as this did) silently dropped
            # (R-1)/R of every metric in a pure-DDP run.
            should_log = rank_owns_metrics_shard(parallel_dims)

        if (
            config.save_first_dp_and_tp
            and not config.save_all_shard_ranks
            and should_log
        ):
            # The first data-parallel group

            is_dp_rank_0 = (
                parallel_dims.get_optional_mesh("loss").get_local_rank() == 0
                if parallel_dims.dp_cp_enabled
                else True
            )
            is_tp_rank_0 = (
                (parallel_dims.get_optional_mesh("tp").get_local_rank() == 0)
                if parallel_dims.tp_enabled
                else True
            )
            should_log = is_dp_rank_0 and is_tp_rank_0

        logger.debug(
            f"Logging decision: has_logging_enabled={has_logging_enabled}, should_log={should_log}"
        )

        if not should_log:
            logger.debug("Returning BaseLogger due to should_log=False")
            return BaseLogger()

        # Setup logging directory
        base_log_dir = os.path.join(
            dump_folder,
            config.save_tb_folder,
            datetime.now().strftime("%Y%m%d-%H%M"),
        )

        if ft_enable:
            base_log_dir = os.path.join(
                base_log_dir,
                f"replica_{ft_replica_id}",
            )

        if config.save_for_all_ranks:
            base_log_dir = os.path.join(
                base_log_dir, f"rank_{torch.distributed.get_rank()}"
            )

        # Create logger container
        logger_container = LoggerContainer()

        # Create loggers in priority order
        if config.enable_wandb:
            logger.debug("Attempting to create WandB logger")
            project = config.wandb_project
            group = config.wandb_group
            name = config.wandb_name
            try:
                wandb_logger = WandBLogger(
                    base_log_dir,
                    config_dict,
                    tag=tag,
                    project=project,
                    group=group,
                    name=name,
                )
                logger_container.add_logger(wandb_logger)
            except Exception as e:
                if "No module named 'wandb'" in str(e):
                    logger.error(
                        "Failed to create WandB logger: No module named 'wandb'. Please install it using 'pip install wandb'."
                    )
                else:
                    logger.error(f"Failed to create WandB logger: {e}")

        if config.enable_tensorboard:
            logger.debug("Creating TensorBoard logger")
            tensorboard_logger = TensorBoardLogger(base_log_dir, tag)
            logger_container.add_logger(tensorboard_logger)

        if logger_container.number_of_loggers == 0:
            logger.debug("No loggers enabled, returning an empty LoggerContainer")
        return logger_container

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ):
        """
        Log training metrics including loss, throughput, and memory statistics.

        Args:
            step: Current training step
            global_avg_loss: Global average loss across all valid tokens on all ranks
                Defined as global_loss_sum / global_valid_tokens
            global_max_loss: Maximum local loss across all ranks
                Defined as max(local_loss_sum / local_valid_tokens)
            grad_norm: Gradient norm after clipping
            extra_metrics: Optional additional metrics to log

        """
        assert self.num_flops_per_token > 0, "num_flops_per_token must be set"

        time_delta = time.perf_counter() - self.time_last_log

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
        tflops = self.num_flops_per_token * tps / 1e12

        time_end_to_end = time_delta / self.config.log_freq
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        # Lets also compute the "tps and mfu" without the data loading time and optim step time
        time_optim_step = sum(self.optim_step_times) / len(self.optim_step_times)
        time_optim_step_pct = 100 * sum(self.optim_step_times) / time_delta

        time_fwd_bwd = sum(self.fwd_bwd_times) / len(self.fwd_bwd_times)
        time_fwd_bwd_pct = 100 * sum(self.fwd_bwd_times) / time_delta

        # dont use this for now, because due to some overlap, the total percentage is already over 100%
        # so `others` here is not really meaningful
        # others_time_pct = (
        #     100 - time_data_loading_pct - time_optim_step_pct - time_fwd_bwd_pct
        # )
        # others_time = time_end_to_end * others_time_pct / 100

        iso_tps = self.ntokens_since_last_log / (
            sum(self.fwd_bwd_times) * self.parallel_dims.non_data_parallel_size
        )
        iso_tflops = self.num_flops_per_token * iso_tps / 1e12
        iso_mfu = 100 * self.num_flops_per_token * iso_tps / self.gpu_peak_flops

        metrics = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "training_metrics/grad_norm": grad_norm,
            "training_metrics/throughput(tps)": tps,
            "training_metrics/tflops": tflops,
            "training_metrics/mfu(%)": mfu,
            "training_metrics/iso_throughput(tps)": iso_tps,
            "training_metrics/iso_tflops": iso_tflops,
            "training_metrics/iso_mfu(%)": iso_mfu,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "time_metrics/optim_step(s)": time_optim_step,
            "time_metrics/optim_step(%)": time_optim_step_pct,
            "time_metrics/fwd_bwd(s)": time_fwd_bwd,
            "time_metrics/fwd_bwd(%)": time_fwd_bwd_pct,
            # "time_metrics/others(s)": others_time,
            # "time_metrics/others(%)": others_time_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }

        if grad_norm is None:
            del metrics["training_metrics/grad_norm"]
            grad_norm_str = ""
        else:
            grad_norm_str = f"{self.color.orange}grad_norm: {grad_norm:7.4f}  "

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.red}step: {step:2}  "
            f"{color.green}loss: {global_avg_loss:8.5f}  "
            f"{grad_norm_str}"
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}  "
            f"{color.cyan}tflops: {tflops:,.2f}  "
            f"{color.magenta}mfu: {mfu:.2f}% | "
            f"{color.yellow}iso_tps: {round(iso_tps):,}  "
            f"{color.cyan}iso_tflops: {iso_tflops:,.2f}  "
            f"{color.magenta}iso_mfu: {iso_mfu:.2f}%{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.optim_step_times.clear()
        self.fwd_bwd_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def log_validation(
        self,
        loss: float,
        step: int,
        ntokens: int | None = None,
        elapsed_time: float | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ):
        # ntokens/elapsed_time are optional so callers that don't track their
        # own validation-only counters (e.g. flux's validator) keep reading
        # the shared training counters, same as before this became overridable.
        used_shared_counter = ntokens is None
        if ntokens is None:
            ntokens = self.ntokens_since_last_log
        if elapsed_time is None:
            elapsed_time = time.perf_counter() - self.time_last_log

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        # tokens per second per device, abbreviated as tps
        tps = (
            ntokens / (elapsed_time * self.parallel_dims.non_data_parallel_size)
            if elapsed_time > 0
            else 0.0
        )

        metrics = {
            # Under loss_metrics/ (not validation_metrics/) so it lands on
            # the same wandb/tensorboard chart as loss_metrics/global_avg_loss
            # (train), letting train and val loss curves overlay directly.
            "loss_metrics/val_loss": loss,
            "validation_metrics/throughput(tps)": tps,
            "validation_metrics/memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "validation_metrics/memory/max_active(%)": device_mem_stats.max_active_pct,
            "validation_metrics/memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "validation_metrics/memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.yellow}validate step: {step:2}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}{color.reset}"
        )

        # Only clear the shared counter if we actually consumed it above;
        # callers that pass their own ntokens never touched it, so training's
        # own accounting must keep accruing across the validation pause.
        if used_shared_counter:
            self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def close(self):
        self.logger.close()
