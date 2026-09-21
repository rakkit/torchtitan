# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import logging
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpointer import EMA, ModelWrapper
from torchtitan.components.optimizer import EMA as EMAContainer  # noqa: N811
from torchtitan.config import TORCH_DTYPE_MAP

# CheckpointManager keeps the EMA container under the EMA state key, so DCP
# flattens its per-tensor state to "ema.state.<fqn>.ema_params".
logger = logging.getLogger(__name__)

_EMA_KEY_PREFIX = f"{EMA}.state."
_EMA_STATE_KEY = "ema_params"
_EMA_KEY_SUFFIX = f".{_EMA_STATE_KEY}"


def _checkpoint_ema_fqns(input_dir: Path) -> set[str]:
    """FQNs the on-disk checkpoint holds EMA weights for."""
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
    }


@torch.inference_mode()
def convert_ema_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
):
    # load model and model args so that we can get the state dict shape
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    ema_fqns = _checkpoint_ema_fqns(input_dir)
    if not ema_fqns:
        raise ValueError(
            f"The checkpoint at {input_dir} holds no EMA weights. Either EMA was "
            "disabled for that training run, or this is a model-only export, which "
            "never carries EMA state. Use convert_to_hf.py for the trained weights."
        )

    # Rebuild the container the run trained with. Parameters are always tracked,
    # so only buffer EMA has to be recovered from the checkpoint: naming those
    # buffers exactly is what keeps one that was EMA-ed during training from
    # silently exporting its live value instead.
    param_fqns = {name for name, _ in model.named_parameters()}
    buffer_fqns = ema_fqns - param_fqns
    # EMA tracks the parameters that required grad during training, so a run
    # with frozen parameters has no entry for them. Track exactly what the
    # checkpoint holds: without this the rebuilt container asks DCP for keys
    # that were never saved and the load fails with a bare missing-key error.
    # Those parameters still reach the export, from the trained weights.
    untracked_params = sorted(param_fqns - ema_fqns)
    if untracked_params:
        logger.info(
            "%d parameter(s) have no EMA state and will be exported from the "
            "trained weights, e.g. %s",
            len(untracked_params),
            untracked_params[:3],
        )
        for name, param in model.named_parameters():
            if name not in ema_fqns:
                param.requires_grad_(False)
    ema = EMAContainer.Config(
        buffer_patterns=[f"^{re.escape(fqn)}$" for fqn in sorted(buffer_fqns)]
    ).build(model_parts=[model])

    uncovered = ema_fqns - set(_ema_state_dict(ema))
    if uncovered:
        raise ValueError(
            f"Model '{model_name}/{model_flavor}' has no parameter or buffer for "
            f"{len(uncovered)} EMA-tracked FQN(s) in the checkpoint, e.g. "
            f"{sorted(uncovered)[:5]}. Exporting would silently drop them; check "
            "that model_name/model_flavor match the training run."
        )

    # Load the trained weights alongside the EMA ones. EMA covers parameters
    # plus the buffers named above, so the remaining tensors (unaveraged
    # buffers) still have to come from the checkpoint for the HF export to be
    # complete and loadable.
    wrapper = ModelWrapper(model)
    state_dict = wrapper._get_state_dict()
    dcp.load(
        {EMA: ema, **state_dict},
        checkpoint_id=input_dir,
    )
    # EMA is keyed by the fused named_parameters() FQNs (e.g.
    # attention.qkv_linear.wqkv.weight) while a model state dict holds the keys
    # the save hooks split them into (wq/wk/wv, w1/w3), and those fused entries
    # do not alias the live parameters. Overlay through the module rather than
    # the dict: re-apply the trained weights so every parameter is populated
    # (including any that EMA does not track), copy the EMA values over the
    # parameters, then re-read the state dict so the split hooks re-derive from
    # them. Assigning EMA into state_dict directly would instead add fused keys
    # that to_hf silently drops, exporting the trained weights.
    wrapper.load_state_dict(state_dict)
    with torch.no_grad():
        for ema_opt in ema.optimizers:
            for tensor, param_state in ema_opt.state.items():
                tensor.copy_(param_state["ema_params"])
    state_dict = wrapper._get_state_dict()

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the EMA weights of a DCP checkpoint to HF format."
    )
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    args = parser.parse_args()

    convert_ema_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )
