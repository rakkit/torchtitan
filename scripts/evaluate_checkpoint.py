#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate one TorchTitan DCP checkpoint against multiple validation sets.

This program is deliberately separate from ``torchtitan/train.py``. It builds
only an inference model, loads model weights only, and never instantiates a
Trainer, optimizer, scheduler, or training dataloader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist

from torchtitan.evaluation.config import (
    find_latest_job_config,
    load_evaluation_suite,
    load_training_config,
    resolve_checkpoint,
)
from torchtitan.evaluation.runtime import EvaluationRuntime, result_row, upsert_result
from torchtitan.tools.logging import init_logger, logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-folder",
        required=True,
        help="Training run directory containing checkpoint/ and job_config_*.json.",
    )
    parser.add_argument(
        "--step", required=True, type=int, help="Checkpoint step to evaluate."
    )
    parser.add_argument(
        "--eval-config",
        required=True,
        help="Python config path, optionally suffixed with :factory.",
    )
    parser.add_argument(
        "--job-config",
        default=None,
        help="Optional explicit job_config_*.json. Defaults to the newest snapshot.",
    )
    parser.add_argument(
        "--checkpoint-folder",
        default="checkpoint",
        help="Checkpoint directory name relative to --dump-folder (default: checkpoint).",
    )
    parser.add_argument(
        "--max-batches",
        default=None,
        type=int,
        help=(
            "Cap on batches evaluated per rank, per validation set. Overrides "
            "each set's own max_batches (if any). Useful for a quick smoke "
            "test against a large corpus. Omit to evaluate every batch."
        ),
    )
    parser.add_argument(
        "--local-batch-size",
        default=None,
        type=int,
        help=(
            "Per-rank batch size. Overrides each set's own local_batch_size "
            "(if any), which otherwise falls back to the checkpoint's "
            "training local_batch_size."
        ),
    )
    args = parser.parse_args()
    for name in ("max_batches", "local_batch_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive, got {value}")
    return args


def main() -> None:
    print("[debug]   ===== start evaluating  ===== ")

    init_logger()

    args = parse_args()
    dump_folder = Path(args.dump_folder).expanduser().resolve()
    checkpoint_path = resolve_checkpoint(
        dump_folder, args.step, checkpoint_folder=args.checkpoint_folder
    )
    suite = load_evaluation_suite(args.eval_config)
    if args.max_batches is not None or args.local_batch_size is not None:
        for dataset in suite.validation_sets:
            if args.max_batches is not None:
                dataset.max_batches = args.max_batches
            if args.local_batch_size is not None:
                dataset.local_batch_size = args.local_batch_size
    job_config_path = (
        Path(args.job_config).expanduser().resolve()
        if args.job_config is not None
        else find_latest_job_config(dump_folder)
    )
    training = load_training_config(job_config_path)

    runtime: EvaluationRuntime | None = None
    try:
        runtime = EvaluationRuntime.build(
            training.config,
            base_folder=str(Path(suite.output_dir).expanduser().resolve()),
        )
        runtime.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        has_cuda = torch.cuda.is_available()
        for dataset in suite.validation_sets:
            if has_cuda:
                # Reset so the peak reported below is this set's own peak, not
                # a running max carried over from an earlier set/checkpoint
                # load.
                torch.cuda.reset_peak_memory_stats()
            totals, elapsed_seconds = runtime.evaluate(dataset)
            # totals is already identical on every rank (evaluate() all-reduces
            # before returning), so force these properties to resolve here, on
            # every rank, rather than only inside the rank==0 block below. If a
            # set has zero scored tokens, loss_nats_per_token/bpb raise -- doing
            # that on every rank means every rank raises together instead of
            # only rank 0 dying while the rest hang at the barrier below
            # waiting for a peer that already crashed.
            loss_nats_per_token = totals.loss_nats_per_token
            bpb = totals.bpb
            ppl = totals.ppl
            if has_cuda:
                peak_allocated_gb = torch.cuda.max_memory_allocated() / 2**30
                peak_reserved_gb = torch.cuda.max_memory_reserved() / 2**30
                device_total_gb = (
                    torch.cuda.get_device_properties(runtime.device).total_memory
                    / 2**30
                )
            if dist.get_rank() == 0:
                seq_len = dataset.seq_len or training.config.training.seq_len
                row = result_row(
                    checkpoint_path=checkpoint_path,
                    step=args.step,
                    job_config_path=training.snapshot_path,
                    seq_len=seq_len,
                    totals=totals,
                    elapsed_seconds=elapsed_seconds,
                )
                output_path = (
                    Path(suite.output_dir).expanduser() / f"{dataset.name}.csv"
                )
                upsert_result(output_path, row)
                logger.info(
                    "%s: loss=%.6f, bpb=%.6f, ppl=%.6f (%d tokens, %d bytes)",
                    dataset.name,
                    loss_nats_per_token,
                    bpb,
                    ppl,
                    totals.total_tokens,
                    totals.total_bytes,
                )
                if has_cuda:
                    # Peak memory *this rank* used scoring this set -- if
                    # this is close to device_total_gb, --local-batch-size
                    # is close to the ceiling; raise it further and watch
                    # this number to find the max that still fits.
                    logger.info(
                        "%s: peak_mem_allocated=%.2fGB, peak_mem_reserved=%.2fGB, device_total=%.2fGB",
                        dataset.name,
                        peak_allocated_gb,
                        peak_reserved_gb,
                        device_total_gb,
                    )
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
