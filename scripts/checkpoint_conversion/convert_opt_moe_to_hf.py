# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An opt_moe run's DCP checkpoint -> an HF folder (``trust_remote_code``).

The model is the run's own, rebuilt from its ``job_config.json`` (flavor and
the recipe's overrides, as offline evaluation); the folder holds
``config.json`` (``state_dict_adapter.hf_config``), the HF model files
(``torchtitan/models/opt_moe/hf/``), the run's tokenizer files
(``hf_assets_path``) and the weights as safetensors::

    python scripts/checkpoint_conversion/convert_opt_moe_to_hf.py \\
        <dump>/checkpoint/step-N <dump>/job_config.json <out> --export_dtype bfloat16
"""

import argparse
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter

from torchtitan.components.checkpointer import export_state_dict, ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.evaluation.config import load_training_config
from torchtitan.models.opt_moe import hf as hf_files
from torchtitan.models.opt_moe.hf.configuration_opt_moe import OptMoeConfig
from torchtitan.models.opt_moe.state_dict_adapter import (
    hf_config,
    OPTMoEStateDictAdapter,
)

# The tokenizer files an HF folder carries (whichever the run's assets have).
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


@torch.inference_mode()
def convert_opt_moe_to_hf(
    checkpoint_dir: Path, job_config: Path, output_dir: Path, export_dtype: str
) -> None:
    run = load_training_config(job_config).config
    model_config = run.model
    if model_config.name != "opt_moe":
        raise ValueError(f"{job_config} is not an opt_moe run: {model_config.name}")
    with torch.device("cpu"):
        model = model_config.build()
    state_dict = ModelWrapper(model)._get_state_dict()
    dcp.load(state_dict, checkpoint_id=str(checkpoint_dir))
    state_dict = export_state_dict(
        state_dict, dtype=TORCH_DTYPE_MAP[export_dtype], model_parts=[model]
    )
    adapter = OPTMoEStateDictAdapter(model_config, None)
    hf_state_dict = adapter.to_hf(state_dict)
    for key in hf_state_dict:
        if key.endswith(adapter.keep_in_fp32):
            hf_state_dict[key] = hf_state_dict[key].float()

    output_dir.mkdir(parents=True, exist_ok=True)
    dcp.save(
        hf_state_dict,
        storage_writer=HuggingFaceStorageWriter(
            path=str(output_dir),
            save_distributed=True,
            fqn_to_index_mapping=adapter.fqn_to_index_mapping_for_save(hf_state_dict),
            enable_consolidation=True,
            thread_count_consolidation=5,
        ),
    )
    OptMoeConfig(**hf_config(model_config), torch_dtype=export_dtype).save_pretrained(
        output_dir
    )
    hf_dir = Path(hf_files.__file__).parent
    for name in ("configuration_opt_moe.py", "modeling_opt_moe.py"):
        shutil.copy(hf_dir / name, output_dir / name)
    assets = Path(run.hf_assets_path)
    for name in _TOKENIZER_FILES:
        if (assets / name).is_file():
            shutil.copy(assets / name, output_dir / name)


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
    args = parser.parse_args()
    convert_opt_moe_to_hf(
        args.checkpoint_dir, args.job_config, args.output_dir, args.export_dtype
    )
