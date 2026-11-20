# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An opt_moe VL run's DCP checkpoint (the SigLIP 2 encoder) -> a complete HF
folder for ``OptMoeForConditionalGeneration``
(``opt_moe/hf/modeling_opt_moe_vl.py``, ``trust_remote_code``).

``convert_opt_moe_to_hf.py`` does the text model (load, adapter, dtype, files);
this adds the vision tower in transformers' SigLIP names (the patch embedding
back to its 3 x 16 x 16 convolution), the connector, ``vision_config`` and the
VL architecture in ``config.json``.

    python scripts/checkpoint_conversion/convert_opt_moe_vl_to_hf.py \\
        <dump>/checkpoint/step-N <dump>/job_config.json <out> --export_dtype bfloat16
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.models.opt_moe import hf as hf_files

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

_BLOCK = {
    "norm1": "layer_norm1",
    "norm2": "layer_norm2",
    "attn.wq": "self_attn.q_proj",
    "attn.wk": "self_attn.k_proj",
    "attn.wv": "self_attn.v_proj",
    "attn.proj": "self_attn.out_proj",
    "mlp.linear_fc1": "mlp.fc1",
    "mlp.linear_fc2": "mlp.fc2",
}


def vision_to_hf(
    state: dict[str, torch.Tensor], encoder_config
) -> dict[str, torch.Tensor]:
    """Our ``vision_encoder.*`` tensors -> ``visual.*`` (SigLIP) + ``merger.*``."""
    p = encoder_config.patch_size
    out = {
        "visual.embeddings.patch_embedding.weight": state[
            "vision_encoder.patch_embed.weight"
        ].reshape(-1, 3, p, p),
        "visual.embeddings.patch_embedding.bias": state[
            "vision_encoder.patch_embed.bias"
        ],
        "visual.embeddings.position_embedding.weight": state[
            "vision_encoder.pos_embed"
        ],
        "visual.post_layernorm.weight": state["vision_encoder.post_norm.weight"],
        "visual.post_layernorm.bias": state["vision_encoder.post_norm.bias"],
    }
    for i in range(encoder_config.num_layers):
        for ours, theirs in _BLOCK.items():
            for kind in ("weight", "bias"):
                out[f"visual.encoder.layers.{i}.{theirs}.{kind}"] = state[
                    f"vision_encoder.layers.{i}.{ours}.{kind}"
                ]
    for name in (
        "norm.weight",
        "norm.bias",
        "linear_fc1.weight",
        "linear_fc1.bias",
        "linear_fc2.weight",
        "linear_fc2.bias",
    ):
        out[f"merger.{name}"] = state[f"vision_encoder.merger.{name}"]
    used = {k for k in state if k.startswith("vision_encoder.")}
    if len(out) != len(used):
        raise ValueError(f"{len(used)} vision tensors, {len(out)} mapped")
    return out


@torch.inference_mode()
def convert(
    checkpoint_dir: Path,
    job_config: Path,
    output_dir: Path,
    export_dtype: str,
    *,
    read_threads: int,
) -> None:
    run, model = load_run_model(job_config, export_dtype)
    encoder = run.model.vision_encoder
    if encoder.rotary_pos_emb is not None or encoder.temporal_patch_size != 1:
        raise ValueError(
            "this converter writes the SigLIP 2 encoder (no RoPE, single-frame patches) only"
        )
    state = ModelWrapper(model)._get_state_dict()
    dcp.load(
        state,
        storage_reader=ParallelFileSystemReader(
            checkpoint_dir, thread_count=read_threads
        ),
    )
    text = {k: v for k, v in state.items() if not k.startswith("vision_encoder.")}
    save_hf_folder(
        run, text, output_dir, export_dtype, extra_hf_state=vision_to_hf(state, encoder)
    )

    # the VL parts of config.json and the VL modeling file; load_training_config
    # rebuilds the model only, so the tokenizer and the tiling come from the run's file
    job = json.loads(job_config.read_text())
    vp = job["vision_processor"]
    image_token_id = (
        MultiModalTokenizer.Config(**job["tokenizer"])
        .build(tokenizer_path=run.hf_assets_path)
        .image_id
    )
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["OptMoeForConditionalGeneration"]
    config["auto_map"][
        "AutoModelForCausalLM"
    ] = "modeling_opt_moe_vl.OptMoeForConditionalGeneration"
    config["auto_map"][
        "AutoModelForImageTextToText"
    ] = "modeling_opt_moe_vl.OptMoeForConditionalGeneration"
    config["vision_config"] = {
        "hidden_size": encoder.dim,
        "intermediate_size": encoder.block.mlp.fc1.out_features,
        "num_hidden_layers": encoder.num_layers,
        "num_attention_heads": encoder.num_heads,
        "patch_size": encoder.patch_size,
        "image_size": vp["tile_size"],
        "num_channels": 3,
        "layer_norm_eps": encoder.post_norm.eps,
        "hidden_act": "gelu_pytorch_tanh",
        "spatial_merge_size": encoder.spatial_merge_size,
        "tile_size": vp["tile_size"],
        "max_tiles": vp["max_tiles"],
        "image_token_id": image_token_id,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy(
        Path(hf_files.__file__).parent / "modeling_opt_moe_vl.py",
        output_dir / "modeling_opt_moe_vl.py",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("job_config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--export_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--read_threads", type=int, default=16)
    args = parser.parse_args()
    convert(
        args.checkpoint_dir,
        args.job_config,
        args.output_dir,
        args.export_dtype,
        read_threads=args.read_threads,
    )
