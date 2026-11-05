#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Score one checkpoint with the training Validator on a fixed slice of a validation corpus.

evaluate_checkpoint.py packs documents with its own greedy packer and one position range per
row, so attention crosses documents, which training never does. This script runs the training
path instead: the pretrain dataloader (concat_then_split, per-document positions), the model's
preprocess_inputs and the training loss (the chunked LM-head loss), at the data-parallel degree
of this launch. Every checkpoint scored at the same degree, sequences per rank and steps sees the
same rows; a run trained at that degree reproduces its logged validation loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch.distributed as dist

from torchtitan.components.data.mix import DatasetSpec, make_pretrain_dataloader_config
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.validator import Validator
from torchtitan.evaluation.config import (
    _overlay_dataclass,
    load_training_config,
    resolve_checkpoint,
)
from torchtitan.evaluation.runtime import EvaluationRuntime
from torchtitan.observability.logging import init_logger

logger = logging.getLogger(__name__)


class _LossCapture:
    """The two members of MetricsProcessor that Validator.validate uses."""

    def __init__(self) -> None:
        self.ntokens_since_last_log = 0
        self.loss: float | None = None

    def log_validation(self, *, loss: float, step: int) -> None:
        self.loss = float(loss)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-folder", required=True)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--job-config", required=True)
    p.add_argument(
        "--data-path",
        required=True,
        help="Validation corpus: parquet shards + manifest.json.",
    )
    p.add_argument("--data-alias", required=True)
    p.add_argument("--seqs-per-rank", required=True, type=int)
    p.add_argument("--val-steps", required=True, type=int)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def main() -> None:
    init_logger()
    args = parse_args()
    dump_folder = Path(args.dump_folder).expanduser().resolve()
    checkpoint_path = resolve_checkpoint(dump_folder, args.step)
    training = load_training_config(args.job_config)
    config = training.config
    with open(args.job_config) as handle:
        loss_snapshot = json.load(handle)["loss"]
    if "num_chunks" not in loss_snapshot:
        raise ValueError(f"expected the chunked training loss, got {loss_snapshot}")
    loss_config = ChunkedLossWrapper.Config()
    _overlay_dataclass(loss_config, loss_snapshot, "loss")

    out_csv = Path(args.out_csv).expanduser().resolve()
    runtime = EvaluationRuntime.build(config, base_folder=str(out_csv.parent))
    runtime.load_checkpoint(checkpoint_path)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # As TrainingEngine wires it: the chunked loss applies the LM head, the model stops before it.
    loss_fn = loss_config.build(compile_config=None)
    model = runtime.model
    loss_fn.set_lm_head(model.lm_head)
    model._skip_lm_head = True

    capture = _LossCapture()
    world, rank = dist.get_world_size(), dist.get_rank()
    seq_len = config.training.max_context_length
    validator = Validator.Config(
        freq=1,
        steps=args.val_steps,
        dataloader=make_pretrain_dataloader_config(
            [
                DatasetSpec(
                    alias=args.data_alias,
                    path=args.data_path,
                    weight=1.0,
                    text_key="text",
                )
            ],
            packing="concat_then_split",
            shuffle=False,
            read_in_order=True,
            seed=420000,
        ),
    ).build(
        parallelism=config.parallelism,
        dp_world_size=world,
        dp_rank=rank,
        tokenizer=runtime.tokenizer,
        parallelism_context=runtime.parallelism_context,
        loss_fn=loss_fn,
        metrics_processor=capture,
        seq_len=seq_len,
        num_tokens_per_microbatch=args.seqs_per_rank * seq_len,
    )
    validator.validate([model], args.step)
    if rank == 0:
        logger.info(
            f"revalidate: loss={capture.loss:.6f} ({world} ranks x {args.seqs_per_rank} seqs x "
            f"{args.val_steps} steps of {seq_len}) {checkpoint_path}"
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(
                [
                    "checkpoint",
                    "step",
                    "loss",
                    "ranks",
                    "seqs_per_rank",
                    "val_steps",
                    "seq_len",
                ]
            )
            w.writerow(
                [
                    checkpoint_path,
                    args.step,
                    capture.loss,
                    world,
                    args.seqs_per_rank,
                    args.val_steps,
                    seq_len,
                ]
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
