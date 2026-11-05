# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model loading, dense loss evaluation, and durable CSV output."""

from __future__ import annotations

import contextlib
import csv
import fcntl
import logging
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

from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelismContext, utils as dist_utils
from torchtitan.evaluation.config import (
    EvaluationConfigError,
    EvaluationDatasetConfig,
    force_ddp_evaluation_parallelism,
    validate_single_node_gpu_world_size,
)
from torchtitan.evaluation.data import build_evaluation_dataloader
from torchtitan.models.base.model import BaseModel
from torchtitan.tools import utils
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

logger = logging.getLogger(__name__)


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
    parallelism_context: ParallelismContext
    tokenizer: BaseTokenizer
    model: torch.nn.Module
    amp_context: Any

    @classmethod
    def build(cls, config: Trainer.Config, *, base_folder: str) -> "EvaluationRuntime":
        if config.model is None:
            raise EvaluationConfigError(
                "saved job config did not produce a model config"
            )

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
        parallelism_context = ParallelismContext(
            dp_replicate=world_size,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=world_size,
            enable_sequence_parallel=False,
        )
        parallelism_context.build_mesh()

        if config.tokenizer is None:
            raise EvaluationConfigError("offline evaluation requires a tokenizer")
        tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        model_config = config.model
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.precision.dtype]),
        ):
            model = model_config.build()

        model = model.parallelize(
            parallelism_context=parallelism_context,
            precision=config.precision,
            parallelism=config.parallelism,
            local_compile=model_config.local_compile,
            activation_remat=None,  # no backward in evaluation
            dump_folder=base_folder,
        )
        model.to_empty(device=device)
        with torch.no_grad():
            cast(BaseModel, model).init_states(buffer_device=None)

        amp_context = contextlib.nullcontext()
        return cls(
            config=config,
            device=device,
            parallelism_context=parallelism_context,
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

    @torch.no_grad()
    def evaluate(
        self, dataset: EvaluationDatasetConfig
    ) -> tuple[EvaluationTotals, float]:
        """Score one set, reduce its totals, and return rank-identical metrics."""

        # The offline-eval dataloader packs a [B, S] rectangle: S the model's
        # context and B from the microbatch tokens, unless the set overrides them.
        seq_len = dataset.seq_len or self.config.model.max_context_length
        local_batch_size = dataset.local_batch_size or max(
            1,
            self.config.training.num_tokens_per_microbatch_per_dp_rank // seq_len,
        )
        # force_ddp_evaluation_parallelism guarantees dp_shard=cp=tp=pp=ep=1,
        # so "batch" is the only active parallel dimension and the global
        # rank *is* the batch/dp rank. Read parallelism_context.world_size and
        # dist.get_rank() directly rather than going through
        # parallelism_context.get_mesh("batch"): that mesh is only materialized
        # when its degree is > 1 (see ParallelismContext._mesh_exist), so a
        # single-GPU (world_size=1) evaluation has no "batch" mesh at all and
        # get_mesh("batch") would raise.
        dp_rank = dist.get_rank()
        dp_world_size = self.parallelism_context.world_size
        dataloader = build_evaluation_dataloader(
            dataset.dataset,
            tokenizer=self.tokenizer,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            drop_long_samples=dataset.drop_long_samples,
            num_workers=dataset.num_workers,
            pin_memory=dataset.pin_memory,
            prefetch_factor=dataset.prefetch_factor,
            persistent_workers=dataset.persistent_workers,
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
            # The models are token-flat: [T] in, [T, V] out, documents marked by
            # `positions` resetting to 0. Each packed row is one sequence, so flatten
            # [B, S] -> [B * S] with positions restarting per row.
            rows, row_len = inputs.shape
            positions = torch.arange(
                row_len, device=inputs.device, dtype=torch.int64
            ).repeat(rows)
            input_dict = {
                "input": inputs.reshape(-1),
                "labels": labels.reshape(-1),
                "positions": positions,
            }
            try:
                with self.parallelism_context.activate_spmd():
                    inputs, labels, model_kwargs, _ = self.model.preprocess_inputs(
                        input_dict,
                        parallelism_context=self.parallelism_context,
                        parallelism=self.config.parallelism,
                        max_num_documents=rows,
                    )
                    with self.amp_context:
                        logits = self.model(inputs, **model_kwargs)
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


def dense_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return summed next-token NLL with TorchTitan's normal ignore-index mask."""

    if not isinstance(logits, torch.Tensor):
        raise EvaluationConfigError(
            "offline evaluation v1 supports dense tensor logits only; "
            f"got {type(logits).__name__}"
        )
    # The models are token-flat: logits [T, V], labels [T].
    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise EvaluationConfigError(
            f"expected logits [tokens, vocab] for {labels.numel()} labels, "
            f"got {tuple(logits.shape)}"
        )
    return functional.cross_entropy(
        logits.float(),
        labels.reshape(-1),
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
