# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An opt_moe run's EMA weights -> a complete HF folder (``trust_remote_code``).

``convert_opt_moe_to_hf.py`` (the run's model from its ``job_config.json``, no
init, the threaded DCP read, the complete HF folder) with upstream's
``convert_ema_to_hf.py`` way of reading the EMA: CheckpointManager keeps the EMA
container under the ``ema`` key (DCP: ``ema.state.<fqn>.ema_params.<ema_key>``,
one copy per decay / half-life fraction); the container is rebuilt for the one
copy exported (``--ema_key``, or the checkpoint's only one) and loaded with the
trained weights, which the parameters EMA does not track (frozen ones) and the
buffers it does not average keep.

    python scripts/checkpoint_conversion/convert_opt_moe_ema_to_hf.py \\
        <dump>/checkpoint/step-N <dump>/job_config.json <out> --export_dtype bfloat16
"""

import argparse
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from torchtitan.components.checkpointer import EMA, ModelWrapper
from torchtitan.components.optim import EMA as EMAContainer  # noqa: N811
from torchtitan.components.optim.ema import parse_ema_key

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
    # The container tracks the parameters the copy covers (the others frozen:
    # they export their trained weights) and the buffers it averaged.
    params = {name for name, _ in model.named_parameters()}
    for name, param in model.named_parameters():
        param.requires_grad_(name in ema_fqns)
    is_decay, value = parse_ema_key(ema_key)
    ema = EMAContainer.Config(
        decays=[value] if is_decay else [],
        half_life_fractions=[] if is_decay else [value],
        buffer_patterns=[f"^{re.escape(f)}$" for f in sorted(ema_fqns - params)],
    ).build(model_parts=[model])

    state_dict = ModelWrapper(model)._get_state_dict()
    dcp.load(
        {EMA: ema, **state_dict},
        storage_reader=ParallelFileSystemReader(
            checkpoint_dir, thread_count=read_threads
        ),
    )
    averaged = {
        name: opt.state[tensor]["ema_params"][ema_key]
        for opt in ema.optimizers
        for group in opt.param_groups
        for name, tensor in zip(group["param_names"], group["params"], strict=True)
        if tensor in opt.state
    }
    # An EMA tensor without its state-dict key (or the reverse) would export
    # the trained weights in its place.
    if set(averaged) != ema_fqns or not set(averaged) <= set(state_dict):
        raise ValueError(
            "the EMA copy does not map onto the model's state dict: "
            f"{sorted(ema_fqns ^ set(averaged) | set(averaged) - set(state_dict))[:5]}"
        )
    state_dict.update(averaged)
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
