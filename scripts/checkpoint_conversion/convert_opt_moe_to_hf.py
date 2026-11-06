# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An opt_moe run's DCP checkpoint -> a complete HF folder (``trust_remote_code``).

Upstream's ``convert_to_hf.py`` (DCP load, ``export_state_dict``, the adapter's
``to_hf``, ``HuggingFaceStorageWriter``), plus what an opt_moe run needs:

1. The run's own model, rebuilt from its ``job_config.json`` (flavor and the
   recipe's overrides, as offline evaluation), not a flavor name.
2. No init: the model is built on the meta device and materialized empty.
   Building it on CPU would run ``nn.Linear``'s default init over every matrix
   (and the recipe's own init is an SVD per matrix); every tensor comes from the
   checkpoint anyway.
3. The checkpoint's shard files read by a thread pool
   (``ParallelFileSystemReader``).
4. A complete folder: ``config.json`` (``hf_config`` + the tokenizer's bos /
   eos ids), the modeling files (``opt_moe/hf/``), the run's tokenizer files
   (``hf_assets_path``), a generation config (the run's, else one with the bos /
   eos ids) and a chat template (the run's, else
   ``opt_moe/hf/chat_template.jinja``, the turns' text concatenated). A pretrain
   run's assets have neither; vLLM needs both.

The run's EMA weights: ``convert_opt_moe_ema_to_hf.py`` (upstream's pair
``convert_to_hf.py`` / ``convert_ema_to_hf.py``).

    python scripts/checkpoint_conversion/convert_opt_moe_to_hf.py \\
        <dump>/checkpoint/step-N <dump>/job_config.json <out> --export_dtype bfloat16
"""

import argparse
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import FileSystemReader, HuggingFaceStorageWriter
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
)
from torch.futures import Future

from torchtitan.components.checkpointer import export_state_dict, ModelWrapper
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.evaluation.config import load_training_config
from torchtitan.models.opt_moe import hf as hf_files
from torchtitan.models.opt_moe.hf.configuration_opt_moe import OptMoeConfig
from torchtitan.models.opt_moe.state_dict_adapter import (
    hf_config,
    OPTMoEStateDictAdapter,
)

from transformers import GenerationConfig

# The tokenizer files an HF folder carries (whichever the run's assets have).
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


class ParallelFileSystemReader(FileSystemReader):
    """``FileSystemReader`` reading the checkpoint's shard files on a thread pool.

    TODO: torch's ``FileSystemReader.read_data`` reads the files one after
    another in the calling thread, which serializes an I/O-bound load of
    hundreds of shard files (one per training rank). This runs its per-file
    body (torch 2.16) on ``thread_count`` threads; each file's items are read
    and committed independently.
    """

    def __init__(self, path: str | Path, *, thread_count: int) -> None:
        super().__init__(path)
        self.thread_count = thread_count

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file: dict[str, list[ReadItem]] = {}
        for item in plan.items:
            path = self.storage_data[item.storage_index].relative_path
            per_file.setdefault(path, []).append(item)
        # No more threads than files (more has been seen to hang).
        workers = max(1, min(self.thread_count, len(per_file)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda kv: self._read_file(*kv, planner), per_file.items()))
        fut: Future = Future()
        fut.set_result(None)
        return fut

    def _read_file(
        self, relative_path: str, reqs: list[ReadItem], planner: LoadPlanner
    ) -> None:
        with self.fs.create_stream(
            self.fs.concat_path(self.path, relative_path), "rb"
        ) as stream:
            for req in reqs:
                item_md = self.storage_data[req.storage_index]
                transform_from = self.transforms.transform_load_stream(
                    req,
                    item_md.transform_descriptors or (),
                    self._slice_file(stream, item_md),
                )
                if req.type == LoadItemType.BYTE_IO:
                    read_bytes = io.BytesIO(transform_from.read(-1))
                    read_bytes.seek(0)
                    planner.load_bytes(req, read_bytes)
                    continue
                if transform_from.seekable():
                    seekable = transform_from
                else:
                    seekable = io.BytesIO(transform_from.read(-1))
                    seekable.seek(0)
                tensor = cast(
                    torch.Tensor,
                    torch.load(seekable, map_location="cpu", weights_only=True),
                )
                tensor = narrow_tensor_by_index(
                    tensor, req.storage_offsets, req.lengths
                )
                target = planner.resolve_tensor(req).detach()
                if target.size() != tensor.size():
                    raise AssertionError(
                        f"req {req.storage_index} mismatch sizes {target.size()} vs {tensor.size()}"
                    )
                target.copy_(tensor)
                planner.commit_tensor(req, target)


def _write_hf_files(
    model_config, hf_assets_path: Path, output_dir: Path, export_dtype: str
) -> None:
    """config.json (with the tokenizer's bos / eos ids), the modeling files, the
    tokenizer files, a generation config and a chat template."""
    token_ids = {}
    if (hf_assets_path / "tokenizer.json").is_file():
        tokenizer = HuggingFaceTokenizer.Config().build(
            tokenizer_path=str(hf_assets_path)
        )
        token_ids = {"bos_token_id": tokenizer.bos_id, "eos_token_id": tokenizer.eos_id}
    config = OptMoeConfig(
        **hf_config(model_config), torch_dtype=export_dtype, **token_ids
    )
    config.save_pretrained(output_dir)
    if OptMoeConfig.from_pretrained(output_dir).to_dict() != config.to_dict():
        raise RuntimeError(f"{output_dir}/config.json does not read back as written")
    hf_dir = Path(hf_files.__file__).parent
    for name in ("configuration_opt_moe.py", "modeling_opt_moe.py"):
        shutil.copy(hf_dir / name, output_dir / name)
    for name in _TOKENIZER_FILES:
        if (hf_assets_path / name).is_file():
            shutil.copy(hf_assets_path / name, output_dir / name)
    tokenizer_config = output_dir / "tokenizer_config.json"
    has_template = (output_dir / "chat_template.jinja").is_file() or (
        tokenizer_config.is_file()
        and "chat_template" in json.loads(tokenizer_config.read_text())
    )
    if not has_template:
        shutil.copy(hf_dir / "chat_template.jinja", output_dir / "chat_template.jinja")
    # A pretrain run's assets carry no generation config: without its eos id,
    # HF and vLLM generation would never stop.
    if token_ids and not (output_dir / "generation_config.json").is_file():
        GenerationConfig(**token_ids).save_pretrained(output_dir)


def load_run_model(job_config: Path):
    """The run (its training config) and its model, rebuilt from the run's
    ``job_config.json`` without init: built on meta, materialized empty on CPU."""
    run = load_training_config(job_config).config
    if run.model.name != "opt_moe":
        raise ValueError(f"{job_config} is not an opt_moe run: {run.model.name}")
    with torch.device("meta"):
        model = run.model.build()
    model.to_empty(device="cpu")
    return run, model


def save_hf_folder(run, model, state_dict, output_dir: Path, export_dtype: str) -> None:
    """``state_dict`` (native names) as a complete HF folder: the weights in
    ``export_dtype`` (the adapter's fp32 tensors kept fp32), the config, the
    modeling and tokenizer files, a generation config and a chat template."""
    state_dict = export_state_dict(
        state_dict, dtype=TORCH_DTYPE_MAP[export_dtype], model_parts=[model]
    )
    adapter = OPTMoEStateDictAdapter(run.model, None)
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
    _write_hf_files(run.model, Path(run.hf_assets_path), output_dir, export_dtype)


@torch.inference_mode()
def convert_opt_moe_to_hf(
    checkpoint_dir: Path,
    job_config: Path,
    output_dir: Path,
    export_dtype: str,
    *,
    read_threads: int = 16,
) -> None:
    run, model = load_run_model(job_config)
    state_dict = ModelWrapper(model)._get_state_dict()
    dcp.load(
        state_dict,
        storage_reader=ParallelFileSystemReader(
            checkpoint_dir, thread_count=read_threads
        ),
    )
    save_hf_folder(run, model, state_dict, output_dir, export_dtype)


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
    args = parser.parse_args()
    convert_opt_moe_to_hf(
        args.checkpoint_dir,
        args.job_config,
        args.output_dir,
        args.export_dtype,
        read_threads=args.read_threads,
    )
