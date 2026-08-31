# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model loading, dense loss evaluation, and durable CSV output."""

from __future__ import annotations

import csv
import fcntl
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as functional

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.evaluation.config import (
    EvaluationConfigError,
    EvaluationDatasetConfig,
    force_ddp_evaluation_parallelism,
    validate_single_node_gpu_world_size,
)
from torchtitan.evaluation.data import build_evaluation_dataloader
from torchtitan.protocols import BaseModel
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


CSV_FIELDS = (
    "checkpoint_path",
    "step",
    "job_config_path",
    "seq_len",
    "total_nll",
    "total_tokens",
    "total_bytes",
    "loss_nats_per_token",
    "bpb",
    "ppl",
    "elapsed_seconds",
)


@dataclass(frozen=True, slots=True)
class EvaluationTotals:
    total_nll: float
    total_tokens: int
    total_bytes: int

    @property
    def loss_nats_per_token(self) -> float:
        if self.total_tokens == 0:
            raise EvaluationConfigError("validation set contains no scored tokens")
        return self.total_nll / self.total_tokens

    @property
    def bpb(self) -> float:
        if self.total_bytes == 0:
            raise EvaluationConfigError("validation set contains no UTF-8 bytes")
        return self.total_nll / (math.log(2) * self.total_bytes)

    @property
    def ppl(self) -> float:
        try:
            return math.exp(self.loss_nats_per_token)
        except OverflowError:
            return math.inf


@dataclass(slots=True)
class EvaluationRuntime:
    """A DDP-only inference runtime built without a :class:`Trainer`."""

    config: Trainer.Config
    device: torch.device
    parallel_dims: ParallelDims
    tokenizer: BaseTokenizer
    model: torch.nn.Module
    amp_context: Any

    @classmethod
    def build(cls, config: Trainer.Config, *, base_folder: str) -> "EvaluationRuntime":
        if config.model_spec is None:
            raise EvaluationConfigError("saved job config did not produce a model spec")

        device_module, device_type = utils.device_module, utils.device_type
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible_gpu_count = device_module.device_count()
        if not 0 <= local_rank < visible_gpu_count:
            raise EvaluationConfigError(
                f"LOCAL_RANK={local_rank} is outside the {visible_gpu_count} visible GPU(s)"
            )
        device = torch.device(f"{device_type}:{local_rank}")
        device_module.set_device(device)

        world_size = dist_utils.init_distributed(config.comm, base_folder=base_folder)
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        try:
            parsed_local_world_size = (
                int(local_world_size) if local_world_size is not None else None
            )
        except ValueError as error:
            raise EvaluationConfigError(
                f"LOCAL_WORLD_SIZE must be an integer, got {local_world_size!r}"
            ) from error
        validate_single_node_gpu_world_size(
            world_size=world_size,
            visible_gpu_count=visible_gpu_count,
            local_world_size=parsed_local_world_size,
        )
        force_ddp_evaluation_parallelism(config, world_size)
        parallel_dims = ParallelDims(
            dp_replicate=world_size,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size,
        )
        parallel_dims.build_mesh()

        if config.tokenizer is None:
            raise EvaluationConfigError("offline evaluation requires a tokenizer")
        tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        model_spec = config.model_spec
        model_config = model_spec.model
        model_config.update_from_config(trainer_config=config)
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Snapshot serialization does not retain the concrete converter config
        # types, and config loading already rejected non-empty converter lists.
        model_converters = config.model_converters.build(
            parallel_dims=parallel_dims,
            model_compile_enabled=(
                config.compile.enable and "model" in config.compile.components
            ),
        )
        model_converters.convert(model)
        model = model_spec.parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            training=config.training,
            model_converters=config.model_converters,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=base_folder,
        )
        model.to_empty(device=device)
        with torch.no_grad():
            cast(BaseModel, model).init_weights(buffer_device=None, skip_init=True)

        amp_context = dist_utils.maybe_enable_amp(
            parallel_dims, config.training.mixed_precision_param, device_type
        )
        return cls(
            config=config,
            device=device,
            parallel_dims=parallel_dims,
            tokenizer=tokenizer,
            model=model,
            amp_context=amp_context,
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load only model parameters from a DCP checkpoint."""

        checkpoint = str(Path(checkpoint_path).resolve())
        wrapper = ModelWrapper(self.model)
        state_dict = wrapper.state_dict()
        begin = time.monotonic()
        dcp.load(state_dict, checkpoint_id=checkpoint)
        wrapper.load_state_dict(state_dict)
        logger.info("Loaded model checkpoint in %.2fs", time.monotonic() - begin)

    @torch.inference_mode()
    def evaluate(
        self, dataset: EvaluationDatasetConfig
    ) -> tuple[EvaluationTotals, float]:
        """Score one set, reduce its totals, and return rank-identical metrics."""

        seq_len = dataset.seq_len or self.config.training.seq_len
        local_batch_size = (
            dataset.local_batch_size or self.config.training.local_batch_size
        )
        # force_ddp_evaluation_parallelism guarantees dp_shard=cp=tp=pp=ep=1,
        # so "batch" is the only active parallel dimension and the global
        # rank *is* the batch/dp rank. Read parallel_dims.world_size and
        # dist.get_rank() directly rather than going through
        # parallel_dims.get_mesh("batch"): that mesh is only materialized
        # when its degree is > 1 (see ParallelDims._mesh_exist), so a
        # single-GPU (world_size=1) evaluation has no "batch" mesh at all and
        # get_mesh("batch") would raise.
        dp_rank = dist.get_rank()
        dp_world_size = self.parallel_dims.world_size
        dataloader = build_evaluation_dataloader(
            dataset.dataloader,
            tokenizer=self.tokenizer,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )

        self.model.eval()
        total_nll = torch.zeros((), dtype=torch.float64, device=self.device)
        total_tokens = torch.zeros((), dtype=torch.float64, device=self.device)
        total_bytes = torch.zeros((), dtype=torch.float64, device=self.device)
        begin = time.monotonic()

        for batch_index, (input_dict, labels, batch_bytes) in enumerate(dataloader):
            if dataset.max_batches is not None and batch_index >= dataset.max_batches:
                break
            inputs = input_dict["input"].to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_bytes = batch_bytes.to(self.device, non_blocking=True)
            extra_kwargs = self._attention_kwargs(inputs)

            try:
                with dist_utils.get_train_context(False)():
                    with self.amp_context:
                        raw_output = self.model(inputs, **extra_kwargs)
                        logits = _unwrap_model_output(raw_output)
                        nll = dense_token_nll(logits, labels)
            except torch.OutOfMemoryError:
                if self.device.type == "cuda":
                    logger.error(
                        "OOM at batch_index=%d, local_batch_size=%d, seq_len=%d, "
                        "rank=%d:\n%s",
                        batch_index,
                        inputs.shape[0],
                        seq_len,
                        dist.get_rank(),
                        torch.cuda.memory_summary(
                            device=self.device, abbreviated=False
                        ),
                    )
                raise
            total_nll += nll.detach().to(torch.float64)
            total_tokens += (labels != IGNORE_INDEX).sum().to(torch.float64)
            total_bytes += batch_bytes.sum().to(torch.float64)

        totals_vector = torch.stack((total_nll, total_tokens, total_bytes))
        dist.all_reduce(totals_vector, op=dist.ReduceOp.SUM)
        totals = EvaluationTotals(
            total_nll=totals_vector[0].item(),
            total_tokens=int(totals_vector[1].item()),
            total_bytes=int(totals_vector[2].item()),
        )
        return totals, time.monotonic() - begin

    def _attention_kwargs(self, inputs: torch.Tensor) -> dict[str, Any]:
        try:
            masks = cast(BaseModel, self.model).get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
                extra_inputs={},
            )
        except TypeError:
            return {}
        return {} if masks is None else {"attention_masks": masks}


def _unwrap_model_output(raw_output: Any) -> Any:
    """Strip the auxiliary load-balance loss some model families always return.

    ``OPTMoEModel.forward`` (and other MoE-capable model families in this repo)
    unconditionally returns ``(logits, aux_loss)`` or ``{"tokens_list": [...],
    "load_balance_loss": ...}``, even for purely dense flavors. Normal training
    strips this via ``components/loss.py::moe_loss`` before the loss function
    ever sees it; offline evaluation has no equivalent step in the training
    loop to do this for us, so we mirror that unwrapping here. The aux
    load-balance loss itself isn't meaningful for offline scoring and is
    discarded.
    """

    if isinstance(raw_output, tuple):
        return raw_output[0]
    if isinstance(raw_output, dict) and "load_balance_loss" in raw_output:
        return raw_output["tokens_list"][0]
    return raw_output


def dense_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return summed next-token NLL with TorchTitan's normal ignore-index mask."""

    if not isinstance(logits, torch.Tensor):
        raise EvaluationConfigError(
            "offline evaluation v1 supports dense tensor logits only; "
            f"got {type(logits).__name__}"
        )
    if logits.ndim != 3:
        raise EvaluationConfigError(
            f"expected dense logits with shape [batch, seq, vocab], got {logits.shape}"
        )
    return functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def result_row(
    *,
    checkpoint_path: Path,
    step: int,
    job_config_path: Path,
    seq_len: int,
    totals: EvaluationTotals,
    elapsed_seconds: float,
) -> dict[str, str]:
    """Build the stable, CSV-serializable row emitted for one validation set."""

    return {
        "checkpoint_path": str(checkpoint_path),
        "step": str(step),
        "job_config_path": str(job_config_path),
        "seq_len": str(seq_len),
        "total_nll": repr(totals.total_nll),
        "total_tokens": str(totals.total_tokens),
        "total_bytes": str(totals.total_bytes),
        "loss_nats_per_token": repr(totals.loss_nats_per_token),
        "bpb": repr(totals.bpb),
        "ppl": repr(totals.ppl),
        "elapsed_seconds": repr(elapsed_seconds),
    }


def upsert_result(csv_path: str | Path, row: dict[str, str]) -> None:
    """Atomically upsert one checkpoint row while serializing concurrent jobs."""

    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            existing_rows = _read_rows(destination)
            key = (row["checkpoint_path"], row["step"])
            updated_rows = [
                existing
                for existing in existing_rows
                if (existing.get("checkpoint_path"), existing.get("step")) != key
            ]
            updated_rows.append(row)
            _write_rows_atomically(destination, updated_rows)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise EvaluationConfigError(
                f"existing result file has incompatible columns: {path}"
            )
        return list(reader)


def _write_rows_atomically(path: Path, rows: list[dict[str, str]]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
