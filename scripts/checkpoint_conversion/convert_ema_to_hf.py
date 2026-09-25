# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a DCP checkpoint's EMA weights to HuggingFace safetensors.

A sibling of ``convert_to_hf.py`` rather than an ``--ema_output`` flag on it, so
the two can be reviewed and changed independently. The argument surface mirrors
``convert_to_hf.py``, so the same invocation exports either set of weights by
swapping the script name.

Everything except the EMA overlay is shared with ``convert_to_hf`` and imported
from it: the job-config model-spec resolution, ``set_init_fn_type``, the
threaded DCP reader, the HF write and the exported-config validation.

On-disk layout: ``CheckpointManager`` keeps the EMA container under the ``ema``
state key, and DCP flattens its per-tensor state to
``ema.state.<fqn>.ema_params``.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpointer import EMA, ModelWrapper
from torchtitan.components.optimizer import EMA as EMAContainer  # noqa: N811
from torchtitan.config import TORCH_DTYPE_MAP

# Same-directory import: this is a script folder, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_hf import (  # noqa: E402
    _copy_hf_assets,
    _export_hf_weights,
    _resolve_model_config_for_conversion,
    ParallelFileSystemReader,
    set_init_fn_type,
)

_EMA_STATE_KEY = "ema_params"
_EMA_KEY_PREFIX = f"{EMA}.state."
_EMA_KEY_SUFFIX = f".{_EMA_STATE_KEY}"


def _checkpoint_ema_fqns(input_dir: Path) -> set[str]:
    """The FQNs this checkpoint holds EMA weights for.

    Read from the DCP metadata rather than by loading, so a checkpoint without
    EMA is reported cleanly instead of failing part-way through a load.
    """
    metadata = dcp.FileSystemReader(input_dir).read_metadata()
    return {
        key[len(_EMA_KEY_PREFIX) : -len(_EMA_KEY_SUFFIX)]
        for key in metadata.state_dict_metadata
        if key.startswith(_EMA_KEY_PREFIX) and key.endswith(_EMA_KEY_SUFFIX)
    }


def _ema_state_dict(ema: EMAContainer) -> dict[str, torch.Tensor]:
    """Native FQN -> EMA tensor, flattening the container's per-tensor state."""
    return {
        name: ema_opt.state[tensor][_EMA_STATE_KEY]
        for ema_opt in ema.optimizers
        for group in ema_opt.param_groups
        for name, tensor in zip(group["param_names"], group["params"])
        if tensor in ema_opt.state and _EMA_STATE_KEY in ema_opt.state[tensor]
    }


@torch.inference_mode()
def convert_ema_to_hf(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    model_flavor: "str | None",
    hf_assets_path: "Path | None",
    export_dtype: str,
    job_config: "Path | None" = None,
    read_threads: int = 16,
):
    """Export the EMA weights of a DCP checkpoint in HF safetensors format.

    Steps mirror ``convert_to_hf``:
      1. Resolve the model config (for opt_moe, from the saved job_config).
      2. Build an empty CPU model and wrap it.
      3. Create a state dict adapter.
      4. Rebuild the EMA container the run trained with.
      5. Load the trained weights and the EMA state together.
      6. Overlay the EMA values onto the model's parameters.
      7. Convert native -> HF, cast dtype, write safetensors.
      8. Copy HF config/modeling files and generate config.json.
    """
    ema_fqns = _checkpoint_ema_fqns(input_dir)
    if not ema_fqns:
        raise ValueError(
            f"The checkpoint at {input_dir} holds no EMA weights: nothing under "
            f"'{_EMA_KEY_PREFIX}*{_EMA_KEY_SUFFIX}'. Either EMA was disabled for "
            "that training run (config.ema is None), or this is a model-only "
            "export, which never carries EMA state. Use convert_to_hf.py for the "
            "trained weights."
        )

    # 1. Model config
    model_config = _resolve_model_config_for_conversion(
        model_name=model_name,
        model_flavor=model_flavor,
        job_config_path=job_config,
    )
    model_flavor = getattr(model_config, "flavor", None) or model_flavor

    # 2. Build empty model on CPU
    set_init_fn_type(model_config)
    with torch.device("cpu"):
        actual_model = model_config.build()
    model_config = getattr(actual_model, "config", model_config)

    # 3. State dict adapter
    adapter_cls = type(actual_model).state_dict_adapter_cls
    assert adapter_cls is not None, (
        "state_dict_adapter_cls is required for HF checkpoint conversion. "
        f"Model '{model_name}/{model_flavor}' has none registered."
    )
    sd_adapter = adapter_cls(model_config, hf_assets_path)

    # 4. Rebuild the container the run trained with. Parameters are always
    #    tracked, so only buffer EMA has to be recovered from the checkpoint:
    #    naming those buffers exactly is what keeps one that WAS averaged during
    #    training from silently exporting its live value instead.
    param_fqns = {name for name, _ in actual_model.named_parameters()}
    buffer_fqns = ema_fqns - param_fqns
    # EMA only tracks parameters that required grad during training, so a run
    # with frozen parameters has no entry for them. Track exactly what the
    # checkpoint holds, or the rebuilt container asks DCP for keys that were
    # never saved and the load dies on a bare missing-key error. Those
    # parameters still reach the export, from the trained weights.
    untracked = sorted(param_fqns - ema_fqns)
    if untracked:
        print(
            f" {len(untracked)} parameter(s) have no EMA state and will be "
            f"exported from the trained weights, e.g. {untracked[:3]}"
        )
        for name, param in actual_model.named_parameters():
            if name not in ema_fqns:
                param.requires_grad_(False)

    ema = EMAContainer.Config(
        buffer_patterns=[f"^{re.escape(fqn)}$" for fqn in sorted(buffer_fqns)]
    ).build(model_parts=[actual_model])

    uncovered = ema_fqns - set(_ema_state_dict(ema))
    if uncovered:
        raise ValueError(
            f"Model '{model_name}/{model_flavor}' has no parameter or buffer for "
            f"{len(uncovered)} EMA-tracked FQN(s) in the checkpoint, e.g. "
            f"{sorted(uncovered)[:5]}. Exporting would silently drop them; check "
            "that model_name/model_flavor/job_config match the training run."
        )

    # 5. Load the trained weights alongside the EMA ones, through the threaded
    #    reader. EMA covers parameters plus the buffers named above, so the
    #    remaining tensors (unaveraged buffers -- opt_moe's MoE routing
    #    `expert_bias` among them) still have to come from the checkpoint for
    #    the export to be complete and loadable.
    wrapper = ModelWrapper(actual_model)
    state_dict = wrapper._get_state_dict()
    dcp.load(
        {EMA: ema, **state_dict},
        storage_reader=ParallelFileSystemReader(input_dir, thread_count=read_threads),
    )

    # 6. EMA is keyed by the FUSED named_parameters() FQNs while a model state
    #    dict holds the keys the save hooks SPLIT them into, and those fused
    #    entries do not alias the live parameters. Overlay through the MODULE,
    #    not the dict: re-apply the trained weights so every parameter is
    #    populated (including any EMA does not track), copy the EMA values over
    #    the parameters, then re-read the state dict so the split hooks
    #    re-derive from them. Assigning EMA into state_dict directly would add
    #    fused keys that to_hf silently drops -- exporting the TRAINED weights
    #    under an EMA filename.
    wrapper.load_state_dict(state_dict)
    with torch.no_grad():
        for ema_opt in ema.optimizers:
            for tensor, param_state in ema_opt.state.items():
                tensor.copy_(param_state[_EMA_STATE_KEY])
    state_dict = wrapper._get_state_dict()

    # 7. Convert native -> HF, cast, write safetensors
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    _export_hf_weights(state_dict, sd_adapter, target_dtype, output_dir)

    # 8. HF config/modeling files
    hf_assets_setup_fn = getattr(type(actual_model), "hf_assets_setup_fn", None)
    if hf_assets_setup_fn is not None:
        hf_assets_setup_fn(actual_model, model_config, str(output_dir))
    else:
        print(
            f"[WARNING] No hf_assets_setup_fn registered for "
            f"'{model_name}/{model_flavor}'. Skipping config.json generation."
        )

    print(f"EMA weights saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a DCP checkpoint's EMA weights to HuggingFace "
        "safetensors format. Mirrors convert_to_hf.py's arguments."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing the DCP checkpoint.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the HF checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="opt_moe",
        help="Model module name under torchtitan.models (default: opt_moe).",
    )
    parser.add_argument(
        "--model_flavor",
        type=str,
        default=None,
        help="Model flavor / config key. For opt_moe this must match job_config.",
    )
    parser.add_argument(
        "--job_config",
        type=Path,
        default=None,
        help="Path to the saved job_config_*.json. Required for opt_moe.",
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        default=None,
        help="Path to a pre-existing HF assets directory containing "
        "model.safetensors.index.json for fqn_to_index_mapping.",
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Export dtype for HF checkpoint (default: bfloat16).",
    )
    parser.add_argument(
        "--read_threads",
        type=int,
        default=min(32, os.cpu_count() or 16),
        help="Number of threads used to read DCP checkpoint shard files in "
        "parallel (default: min(32, cpu_count)). This step is CPU/disk I/O "
        "bound, not GPU bound; the default torch DCP reader reads shard "
        "files one at a time on a single thread.",
    )
    parser.add_argument(
        "--copy_assets_from",
        type=Path,
        default=None,
        help="Reuse the config/tokenizer files from an already-exported HF "
        "directory instead of regenerating them. EMA weights share the main "
        "export's architecture, so this is faster and guaranteed consistent "
        "with it.",
    )
    args = parser.parse_args()

    convert_ema_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
        args.job_config,
        args.read_threads,
    )
    if args.copy_assets_from is not None:
        _copy_hf_assets(args.copy_assets_from, args.output_dir)
        print(f"HF assets copied from {args.copy_assets_from}")
