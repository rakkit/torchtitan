# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import io
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass, replace as dc_replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import save_torch_state_dict
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.planner import LoadItemType
from torch.futures import Future
from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config.configs import ParallelismConfig, TrainingConfig


class ParallelFileSystemReader(FileSystemReader):
    """FileSystemReader that reads shard files concurrently.

    torch's stock FileSystemReader.read_data (as of the torch version
    installed here) reads DCP shard files strictly one at a time, one tensor
    read_item at a time, in the calling thread -- no thread pool. For a
    checkpoint sharded into hundreds of files (one per training rank), that
    serializes what should be an I/O-bound operation onto a single core,
    even though loading a checkpoint for HF conversion needs no GPU/compute
    and the host typically has dozens of idle cores. This subclass keeps
    torch's exact per-item read/deserialize logic but fans the per-file work
    out across a thread pool, since each file (and each tensor within it) is
    read and committed independently.
    """

    def __init__(self, path, thread_count: int = 16):
        super().__init__(path)
        self.thread_count = max(1, thread_count)

    def read_data(self, plan, planner):
        per_file: dict[str, list] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            per_file.setdefault(item_md.relative_path, []).append(read_item)

        def _read_one_file(relative_path, reqs) -> None:
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        if transform_from.seekable():
                            seekable = transform_from
                        else:
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)

                        tensor = torch.load(
                            seekable, map_location="cpu", weights_only=True
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()
                        assert target_tensor.size() == tensor.size(), (
                            f"req {req.storage_index} mismatch sizes "
                            f"{target_tensor.size()} vs {tensor.size()}"
                        )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        # Clamp to the number of files this plan actually touches -- far more
        # pool threads than files (e.g. 64 threads for a 4-shard checkpoint)
        # has been observed to hang.
        num_workers = min(self.thread_count, len(per_file))
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(pool.map(lambda kv: _read_one_file(*kv), per_file.items()))

        fut: Future = Future()
        fut.set_result(None)
        return fut


def _normalize_layer_pattern_for_validation(pattern):
    """Match HF export normalization for per-layer pattern fields."""
    if pattern is None:
        return None
    if isinstance(pattern, tuple):
        pattern = list(pattern)
    if isinstance(pattern, list):
        if len(pattern) == 1 and isinstance(pattern[0], str):
            return pattern[0]
        if pattern and all(isinstance(x, str) and len(x) == 1 for x in pattern):
            return "".join(pattern)
    return pattern


def _resolve_dataclass_type(annotation):
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    for candidate in get_args(annotation):
        if isinstance(candidate, type) and is_dataclass(candidate):
            return candidate
    return None


def _build_dataclass_from_dict(dataclass_type, values: dict, *, field_path: str):
    if not isinstance(values, dict):
        raise ValueError(f"Expected {field_path} to be a JSON object.")

    kwargs = {}
    field_map = {f.name: f for f in fields(dataclass_type)}
    for key, value in values.items():
        if key not in field_map:
            raise ValueError(f"Unknown key '{field_path}.{key}' in job_config.")
        field_info = field_map[key]
        nested_type = _resolve_dataclass_type(field_info.type)
        if isinstance(value, dict):
            if nested_type is None:
                raise ValueError(f"Expected '{field_path}.{key}' to be a scalar value.")
            kwargs[key] = _build_dataclass_from_dict(
                nested_type,
                value,
                field_path=f"{field_path}.{key}",
            )
        else:
            kwargs[key] = value
    return dataclass_type(**kwargs)


def _apply_dataclass_overrides(target_obj, overrides: dict, *, field_path: str):
    if not is_dataclass(target_obj):
        raise ValueError(f"Expected {field_path} to be a dataclass instance.")
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected {field_path} to be a JSON object.")

    field_map = {f.name: f for f in fields(type(target_obj))}
    for key, value in overrides.items():
        if key not in field_map:
            raise ValueError(f"Unknown key '{field_path}.{key}' in job_config.")
        current_value = getattr(target_obj, key)
        field_info = field_map[key]
        nested_type = _resolve_dataclass_type(field_info.type)
        if isinstance(value, dict):
            if isinstance(current_value, dict) and not any(
                is_dataclass(v) for v in current_value.values()
            ):
                # A LEAF mapping -- flex `kernel_options` and the like. Narrow
                # deliberately: `model.layers` is also a dict, but of layer
                # CONFIGS, and assigning raw JSON over it replaces dataclasses
                # with dicts that blow up later inside Decoder.__init__.
                setattr(target_obj, key, value)
            elif isinstance(current_value, dict):
                # dict of dataclasses (per-layer configs): recurse per entry.
                for sub_key, sub_value in value.items():
                    if sub_key in current_value and is_dataclass(
                        current_value[sub_key]
                    ):
                        _apply_dataclass_overrides(
                            current_value[sub_key],
                            sub_value,
                            field_path=f"{field_path}.{key}.{sub_key}",
                        )
            elif is_dataclass(current_value):
                _apply_dataclass_overrides(
                    current_value,
                    value,
                    field_path=f"{field_path}.{key}",
                )
            elif nested_type is not None:
                setattr(
                    target_obj,
                    key,
                    _build_dataclass_from_dict(
                        nested_type,
                        value,
                        field_path=f"{field_path}.{key}",
                    ),
                )
            elif current_value is None:
                # An `Any`-annotated field the model's own config expansion
                # fills in -- lm_head, tok_embeddings, the per-layer configs.
                # Since llm-0.5.2 the dumped job_config contains them EXPANDED,
                # and they are not reconstructible: their param_init holds
                # reprs like "functools.partial(<function scion_normal_ at
                # 0x4003a13c40e0>)", a function address from a dead process.
                # Skipping is correct as well as necessary -- build() re-derives
                # them from dim/vocab_size/init settings, which ARE restored
                # from this same job_config.
                print(
                    f"[job_config] '{field_path}.{key}' is expansion-derived; "
                    "skipping the serialized copy and letting build() re-derive it."
                )
            else:
                raise ValueError(f"Expected '{field_path}.{key}' to be a scalar value.")
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(v, dict) for v in value)
        ):
            # A list of CONFIGS -- `layers`, which expansion derives from
            # `layer` plus n_layers/swa_pattern. Same story as the Any-annotated
            # fields above: assigning the raw JSON list would replace dataclasses
            # with dicts, and Decoder.__init__ then dies on `layer.attention`.
            print(
                f"[job_config] '{field_path}.{key}' is expansion-derived; "
                "skipping the serialized copy and letting build() re-derive it."
            )
        else:
            setattr(target_obj, key, value)


def _load_job_config(job_config_path: Path) -> dict:
    if not job_config_path.exists():
        raise FileNotFoundError(f"job_config file does not exist: {job_config_path}")
    try:
        return json.loads(job_config_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed job_config JSON at {job_config_path}: {e}") from e


def _resolve_model_config_for_conversion(
    *,
    model_name: str,
    model_flavor: str | None,
    job_config_path: "Path | None",
):
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    default_flavor = "bsc-1B-7B-opt-g"

    if model_name != "opt_moe":
        resolved_flavor = model_flavor or default_flavor
        return model_module.model_registry(resolved_flavor)

    if job_config_path is None:
        raise ValueError(
            "--job_config is required when converting opt_moe checkpoints to HF."
        )

    job_config = _load_job_config(job_config_path)
    model_overrides = job_config.get("model")
    if not isinstance(model_overrides, dict):
        raise ValueError(
            f"Malformed job_config at {job_config_path}: missing object 'model'."
        )

    job_config_flavor = model_overrides.get("flavor")
    if not isinstance(job_config_flavor, str) or not job_config_flavor:
        raise ValueError(
            f"Malformed job_config at {job_config_path}: missing string "
            "'model.flavor'."
        )

    if model_flavor is not None and model_flavor != job_config_flavor:
        raise ValueError(
            "opt_moe conversion flavor mismatch: "
            f"--model_flavor={model_flavor!r} but "
            f"job_config.model.flavor={job_config_flavor!r}."
        )

    model_config = model_module.model_registry(job_config_flavor)
    _apply_dataclass_overrides(
        model_config,
        model_overrides,
        field_path="model",
    )

    # Run the model's OWN config expansion rather than trusting the serialized
    # copy. Since llm-0.5.2 the dumped job_config contains the model config
    # fully EXPANDED -- tok_embeddings, lm_head, the per-layer list -- and those
    # are not reconstructible: their param_init holds reprs like
    # "functools.partial(<function scion_normal_ at 0x4003a13c40e0>)", a
    # function address from a dead process. `update_from_config` derives them
    # from the scalar fields (dim, vocab_size, n_layers, the init settings)
    # that ARE restored above, which is exactly what the trainer does before
    # building the model.
    def _rebuild(cls, data: dict):
        """Construct a real config dataclass from the job_config's dict.

        Upstream asserts the concrete type (`config.parallelism must be a
        ParallelismConfig`), so a SimpleNamespace will not do. Only scalar
        fields the class actually declares are carried across.
        """
        known = {f.name: f for f in fields(cls)}
        kwargs = {
            k: v
            for k, v in (data or {}).items()
            if k in known and not isinstance(v, (dict, list))
        }
        return cls(**kwargs)

    expansion_config = SimpleNamespace(
        parallelism=_rebuild(ParallelismConfig, job_config.get("parallelism")),
        training=_rebuild(TrainingConfig, job_config.get("training")),
    )
    if hasattr(model_config, "update_from_config"):
        model_config.update_from_config(config=expansion_config)

    # Sync training.seq_len → rope.max_seq_len so max_position_embeddings in the
    # exported HF config reflects the actual training context length, not the default.
    # Mirrors OPTMoEModel.Config.update_from_config (model.py:283-285).
    training_seq_len = job_config.get("training", {}).get("max_context_length")
    if isinstance(training_seq_len, int) and training_seq_len > 0:
        if getattr(model_config, "rope", None) is not None:
            model_config.rope = dc_replace(
                model_config.rope, max_context_length=training_seq_len
            )
        if getattr(model_config, "rope_of_swa", None) is not None:
            model_config.rope_of_swa = dc_replace(
                model_config.rope_of_swa, max_context_length=training_seq_len
            )
    return model_config


def _validate_exported_hf_config(
    *,
    model_name: str,
    model_config,
    output_dir: Path,
):
    if model_name != "opt_moe":
        return

    from torchtitan.models.opt_moe.hf_assests.setup_hf import (
        get_hf_config_overrides_from_model_config,
    )

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected HF config at {config_path}, but it was not created."
        )

    exported_config = json.loads(config_path.read_text())
    expected_fields = get_hf_config_overrides_from_model_config(None, model_config)
    expected_fields["rope_pattern"] = _normalize_layer_pattern_for_validation(
        expected_fields.get("rope_pattern")
    )
    expected_fields["swa_pattern"] = _normalize_layer_pattern_for_validation(
        expected_fields.get("swa_pattern")
    )

    mismatches = {
        field: (expected_value, exported_config.get(field))
        for field, expected_value in expected_fields.items()
        if exported_config.get(field) != expected_value
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{field}: expected {expected!r}, got {actual!r}"
            for field, (expected, actual) in mismatches.items()
        )
        raise ValueError(
            f"HF config export lost opt_moe runtime config settings: {mismatch_text}."
        )


def _checkpoint_has_prefix(input_dir: Path, prefix: str) -> bool:
    """Whether the on-disk DCP checkpoint's metadata has any key starting with
    ``prefix``. Local to this script -- the CheckpointManager copy was dead and removed
    here since this script calls dcp.load directly rather than through
    CheckpointManager, so it needs the same guard against a missing key
    raising inside DCP's load planner.
    """
    try:
        metadata = dcp.FileSystemReader(str(input_dir)).read_metadata()
        return any(k.startswith(prefix) for k in metadata.state_dict_metadata)
    except Exception:
        return False


def set_init_fn_type(config):
    for name in dir(config):
        if name.startswith("_"):
            continue

        value = getattr(config, name)

        if "init_fn_type" in name:
            setattr(config, name, "normal")
            # print(f"Set {name} = normal")

        elif type(value).__name__ == "Config":
            set_init_fn_type(value)


def _export_hf_weights(
    state_dict: dict, sd_adapter, target_dtype, output_dir: Path
) -> None:
    """Convert a native state dict to HF format, cast, and write safetensors.
    Shared by the main-weights and EMA-weights export paths."""
    hf_state_dict = sd_adapter.to_hf(state_dict)
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}
    # Give each exported tensor its OWN storage before writing. Upstream's
    # stacked projections (#4808: w13 is one [2, F, D] parameter) reach `to_hf`
    # split into gate_proj / up_proj, which are VIEWS into the stacked storage.
    # safetensors refuses those -- "found no suitable name to keep for saving
    # ... None is covering the entire storage" -- because it cannot tell an
    # alias from a duplicate. `.contiguous()` is NOT enough: a view that is
    # already contiguous is returned as-is, still sharing the storage. Clone.
    # One extra buffer per tensor, at conversion time on a CPU host.
    hf_state_dict = {
        k: (v.detach().clone() if torch.is_tensor(v) else v)
        for k, v in hf_state_dict.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_torch_state_dict(
        hf_state_dict,
        output_dir,
        max_shard_size="5GB",
        safe_serialization=True,
        metadata={"format": "pt"},
    )


def _copy_hf_assets(src_dir: Path, dst_dir: Path) -> None:
    """Copy the non-weight HF assets (config.json, modeling files, tokenizer,
    etc.) from an already-exported HF directory into a second one -- so the
    EMA export can reuse them instead of regenerating, since EMA weights
    share the main export's architecture/config."""
    weight_names = {"model.safetensors", "model.safetensors.index.json"}
    for item in src_dir.iterdir():
        if item.name in weight_names or item.name.endswith(".safetensors"):
            continue
        dest = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy(item, dest)


def try_to_copy_tokenizer(output_dir, hf_assets_path):
    """
    if these files exist in the hf_assets_path, then copy them to the output_dir
    """
    if hf_assets_path is None:
        return

    tokenizer_assests_lists = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ]
    for asset in tokenizer_assests_lists:
        if os.path.exists(os.path.join(hf_assets_path, asset)):
            shutil.copy(
                os.path.join(hf_assets_path, asset), os.path.join(output_dir, asset)
            )


@torch.inference_mode()
def convert_to_hf(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    model_flavor: "str | None",
    hf_assets_path: "Path | None",
    export_dtype: str,
    job_config: "Path | None" = None,
    read_threads: int = 16,
):
    """Convert a DCP checkpoint to HuggingFace safetensors format.

    Steps:
      1. Resolve the model config from the registry (+ the dumped job_config).
      2. Build an empty CPU model and wrap it.
      3. Create a state dict adapter.
      4. Load the DCP checkpoint.
      5. Convert native -> HF state dict.
      6. Optionally cast dtype.
      7. Write HF safetensors.
      8. Copy HF config/modeling files and generate config.json.
    """
    model_config = _resolve_model_config_for_conversion(
        model_name=model_name,
        model_flavor=model_flavor,
        job_config_path=job_config,
    )
    model_flavor = getattr(model_config, "flavor", None) or model_flavor

    set_init_fn_type(model_config)
    # for field in fields(model_config):
    #     value = getattr(model_config, field.name)
    #     print(f" {field.name} = {value}")
    # return

    with torch.device("cpu"):
        actual_model = model_config.build()
    model_config = getattr(actual_model, "config", model_config)
    adapter_cls = type(actual_model).state_dict_adapter_cls
    model = ModelWrapper(actual_model)

    print(" Model is build, now loading the state")
    assert adapter_cls is not None, (
        "state_dict_adapter_cls is required for HF checkpoint conversion. "
        f"Model '{model_name}/{model_flavor}' has none registered."
    )
    sd_adapter = adapter_cls(model_config, hf_assets_path)

    # 4. Load DCP checkpoint into empty state dict
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        storage_reader=ParallelFileSystemReader(input_dir, thread_count=read_threads),
    )

    print(" DCP is load, now write to local")
    # 5-7. Convert native → HF state dict, apply export dtype, write safetensors
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    _export_hf_weights(state_dict, sd_adapter, target_dtype, output_dir)

    # 8. Copy HF config/modeling files and generate config.json
    hf_assets_setup_fn = getattr(type(actual_model), "hf_assets_setup_fn", None)
    if hf_assets_setup_fn is not None:
        hf_assets_setup_fn(actual_model, model_config, str(output_dir))
        _validate_exported_hf_config(
            model_name=model_name,
            model_config=model_config,
            output_dir=output_dir,
        )
    else:
        print(
            f"[WARNING] No hf_assets_setup_fn registered for '{model_name}/{model_flavor}'. "
            "Skipping config.json generation."
        )

    # hf_assets_setup_fn will create a dummy chat-template.jinja,
    # we need to copy the tokenizer files to the output_dir
    # to maybe override the dummy chat-template.jinja
    try_to_copy_tokenizer(output_dir, hf_assets_path)

    print(f"model is saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a DCP checkpoint to HuggingFace safetensors format."
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
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Export dtype for HF checkpoint (default: float32).",
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
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
        args.job_config,
        args.read_threads,
    )
