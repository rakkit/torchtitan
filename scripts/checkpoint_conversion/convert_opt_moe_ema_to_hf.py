# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An opt_moe run's EMA weights -> a complete HF folder (``trust_remote_code``).

``convert_opt_moe_to_hf.py`` (the run's model from its ``job_config.json``, no
init, the threaded DCP read, the complete HF folder) for the EMA weights:
CheckpointManager keeps the EMA container under the ``ema`` key (DCP:
``ema.state.<fqn>.ema_params.<ema_key>``, one copy per decay / half-life
fraction). The copy exported (``--ema_key``, or the checkpoint's only one) is read
straight into the model's tensors under those keys, the tensors it does not
cover (frozen parameters, buffers it does not average) from the trained weights:
one fp32 copy of the model in memory (upstream's ``convert_ema_to_hf.py``
rebuilds the container next to the model, about 2.5x that).

    python scripts/checkpoint_conversion/convert_opt_moe_ema_to_hf.py \\
        <dump>/checkpoint/step-N <dump>/job_config.json <out> --export_dtype bfloat16
"""

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from torchtitan.components.checkpointer import EMA, ModelWrapper

if __package__:
    from scripts.checkpoint_conversion.convert_opt_moe_to_hf import (
        load_run_model,
        ParallelFileSystemReader,
        save_hf_folder,
    )
else:
    from convert_opt_moe_to_hf import (  # pyrefly: ignore [missing-import]
        load_run_model,
        ParallelFileSystemReader,
        save_hf_folder,
    )

_EMA_PREFIX, _EMA_INFIX = f"{EMA}.state.", ".ema_params."


def _ema_copy(checkpoint_dir: Path, ema_key: str | None) -> tuple[str, set[str]]:
    """The EMA copy to export (``ema_key``, else the checkpoint's only one) and
    the FQNs the checkpoint holds it for, from the DCP metadata."""
    fqns_by_key: dict[str, set[str]] = {}
    for key in FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata:
        if key.startswith(_EMA_PREFIX):
            fqn, sep, copy = key[len(_EMA_PREFIX) :].rpartition(_EMA_INFIX)
            if sep:
                fqns_by_key.setdefault(copy, set()).add(fqn)
    if not fqns_by_key:
        raise ValueError(
            f"{checkpoint_dir} holds no EMA weights (EMA off in the run, or a "
            "model-only save): convert_opt_moe_to_hf.py exports the trained ones"
        )
    if ema_key is None:
        if len(fqns_by_key) > 1:
            raise ValueError(f"EMA copies {sorted(fqns_by_key)}: pass --ema_key")
        (ema_key,) = fqns_by_key
    if ema_key not in fqns_by_key:
        raise ValueError(
            f"no EMA copy {ema_key!r}; the checkpoint has {sorted(fqns_by_key)}"
        )
    return ema_key, fqns_by_key[ema_key]


@torch.inference_mode()
def convert_opt_moe_ema_to_hf(
    checkpoint_dir: Path,
    job_config: Path,
    output_dir: Path,
    export_dtype: str,
    *,
    read_threads: int = 16,
    ema_key: str | None = None,
) -> None:
    run, model = load_run_model(job_config)
    ema_key, ema_fqns = _ema_copy(checkpoint_dir, ema_key)
    state_dict = ModelWrapper(model)._get_state_dict()
    # An EMA tensor without its state-dict key would export the trained
    # weights in its place.
    missing = sorted(ema_fqns - set(state_dict))
    if missing:
        raise ValueError(
            f"the EMA copy does not map onto the model's state dict: {missing[:5]}"
        )
    # One fp32 copy of the model: the tensors the EMA copy covers are read from
    # their EMA entries into the model's own tensors, the rest (frozen
    # parameters, buffers the EMA does not average) from the trained weights.
    dcp.load(
        {
            f"{_EMA_PREFIX}{fqn}{_EMA_INFIX}{ema_key}" if fqn in ema_fqns else fqn: t
            for fqn, t in state_dict.items()
        },
        storage_reader=ParallelFileSystemReader(
            checkpoint_dir, thread_count=read_threads
        ),
    )
    save_hf_folder(run, state_dict, output_dir, export_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint_dir", type=Path, help="the DCP checkpoint (step-N)")
    parser.add_argument("job_config", type=Path, help="the run's job_config.json")
    parser.add_argument("output_dir", type=Path, help="the HF folder to write")
    parser.add_argument(
        "--export_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--read_threads",
        type=int,
        default=16,
        help="threads reading the checkpoint's shard files",
    )
    parser.add_argument(
        "--ema_key",
        default=None,
        help="the EMA copy, e.g. decay_0p999 (default: the checkpoint's only one)",
    )
    args = parser.parse_args()
    convert_opt_moe_ema_to_hf(
        args.checkpoint_dir,
        args.job_config,
        args.output_dir,
        args.export_dtype,
        read_threads=args.read_threads,
        ema_key=args.ema_key,
    )
