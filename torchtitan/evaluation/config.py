# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration and checkpoint-discovery helpers for offline evaluation."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

from torchtitan.config.configs import ParallelismConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


class EvaluationConfigError(ValueError):
    """Raised when an offline-evaluation input cannot be used safely."""


@dataclass(kw_only=True, slots=True)
class EvaluationDatasetConfig:
    """One named validation set in an :class:`EvaluationSuiteConfig`.

    ``dataloader`` deliberately reuses TorchTitan's regular text-dataset
    configuration. The offline evaluator supports one source dataset per named
    validation set, finite greedy or best-fit packing, tokenizer offsets for
    exact raw-byte accounting, and dense LM loss only.
    """

    name: str
    dataloader: HuggingFaceTextDataLoader.Config
    seq_len: int | None = None
    local_batch_size: int | None = None
    max_batches: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise EvaluationConfigError("validation set names must be non-empty")
        if self.seq_len is not None and self.seq_len <= 0:
            raise EvaluationConfigError("seq_len must be positive when provided")
        if self.local_batch_size is not None and self.local_batch_size <= 0:
            raise EvaluationConfigError(
                "local_batch_size must be positive when provided"
            )
        if self.max_batches is not None and self.max_batches <= 0:
            raise EvaluationConfigError("max_batches must be positive when provided")


@dataclass(kw_only=True, slots=True)
class EvaluationSuiteConfig:
    """Reusable list of validation sets and their CSV output directory."""

    output_dir: str
    validation_sets: list[EvaluationDatasetConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise EvaluationConfigError("output_dir must be non-empty")
        names = [dataset.name for dataset in self.validation_sets]
        if not names:
            raise EvaluationConfigError("validation_sets must contain at least one set")
        if len(names) != len(set(names)):
            raise EvaluationConfigError("validation set names must be unique")


@dataclass(kw_only=True, slots=True)
class LoadedTrainingConfig:
    """The evaluation-relevant portion of a saved TorchTitan job snapshot."""

    config: Trainer.Config
    snapshot_path: Path


def resolve_checkpoint(
    dump_folder: str | Path, step: int, checkpoint_folder: str = "checkpoint"
) -> Path:
    """Resolve and validate the DCP directory for one completed training step."""

    if step < 0:
        raise EvaluationConfigError("step must be non-negative")
    checkpoint_path = (
        Path(dump_folder).expanduser().resolve() / checkpoint_folder / f"step-{step}"
    )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {checkpoint_path}"
        )
    if not (checkpoint_path / ".metadata").is_file():
        raise EvaluationConfigError(
            f"checkpoint is incomplete (missing .metadata): {checkpoint_path}"
        )
    return checkpoint_path


def find_latest_job_config(dump_folder: str | Path) -> Path:
    """Return the latest saved TorchTitan config by its timestamped filename."""

    root = Path(dump_folder).expanduser().resolve()
    configs = sorted(path for path in root.glob("job_config_*.json") if path.is_file())
    if not configs:
        raise FileNotFoundError(
            f"no job_config_*.json snapshot found in dump folder: {root}"
        )
    return configs[-1]


def load_evaluation_suite(config_spec: str) -> EvaluationSuiteConfig:
    """Load ``/path/to/file.py[:factory]`` and return its suite config."""

    config_path_raw, factory_name = _split_python_spec(config_spec)
    config_path = Path(config_path_raw).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"evaluation config file does not exist: {config_path}")

    module_name = f"torchtitan_offline_evaluation_{abs(hash(config_path))}"
    module_spec = importlib.util.spec_from_file_location(module_name, config_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import evaluation config: {config_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise EvaluationConfigError(
            f"evaluation config factory {factory_name!r} was not found in {config_path}"
        )
    suite = factory()
    if not isinstance(suite, EvaluationSuiteConfig):
        raise TypeError(
            f"{config_path}:{factory_name} must return EvaluationSuiteConfig, "
            f"got {type(suite).__name__}"
        )
    return suite


def load_training_config(snapshot_path: str | Path) -> LoadedTrainingConfig:
    """Rehydrate a model-construction config from a saved ``job_config`` JSON.

    The JSON intentionally omits ``ModelSpec`` callables. We recreate those
    from the installed model registry and only overlay the serialized model
    dataclass values. Optimizers, schedulers, checkpointing, and the training
    dataloader are intentionally not reconstructed or used by the evaluator.
    """

    path = Path(snapshot_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"job config snapshot does not exist: {path}")
    with path.open() as handle:
        snapshot = json.load(handle)
    if not isinstance(snapshot, dict):
        raise EvaluationConfigError(f"job config must contain a JSON object: {path}")

    model_snapshot = _required_mapping(snapshot, "model_spec")
    name = _required_string(model_snapshot, "name")
    flavor = _required_string(model_snapshot, "flavor")
    model_spec = _resolve_model_spec(name, flavor)
    _overlay_dataclass(
        model_spec.model,
        _required_mapping(model_snapshot, "model"),
        "model_spec.model",
    )

    converters = _optional_mapping(snapshot, "model_converters")
    if converters and converters.get("converters"):
        raise EvaluationConfigError(
            "offline evaluation v1 does not support checkpoints requiring "
            "model_converters; use a converter-free dense checkpoint"
        )
    if snapshot.get("tokenizer", {}) is None:
        raise EvaluationConfigError(
            "offline evaluation v1 requires a HuggingFace tokenizer configuration"
        )

    config = Trainer.Config(
        model_spec=model_spec,
        hf_assets_path=_required_string(snapshot, "hf_assets_path"),
    )
    _overlay_dataclass(
        config.training,
        _required_mapping(snapshot, "training"),
        "training",
    )
    _overlay_dataclass(
        config.parallelism,
        _required_mapping(snapshot, "parallelism"),
        "parallelism",
    )
    _overlay_dataclass(
        config.compile,
        _required_mapping(snapshot, "compile"),
        "compile",
    )
    _overlay_dataclass(
        config.activation_checkpoint,
        _required_mapping(snapshot, "activation_checkpoint"),
        "activation_checkpoint",
    )
    debug_snapshot = _optional_mapping(snapshot, "debug")
    if debug_snapshot:
        _overlay_dataclass(config.debug, debug_snapshot, "debug")

    return LoadedTrainingConfig(config=config, snapshot_path=path)


def force_ddp_evaluation_parallelism(config: Trainer.Config, world_size: int) -> None:
    """Replace training parallelism with the supported replicated-DP layout."""

    if not 1 <= world_size <= 8:
        raise EvaluationConfigError(
            "offline evaluation supports 1 to 8 ranks on one node, "
            f"got world size {world_size}"
        )
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=world_size,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=1,
    )


def validate_single_node_gpu_world_size(
    *, world_size: int, visible_gpu_count: int, local_world_size: int | None
) -> None:
    """Ensure ``torchrun`` launched one evaluation rank per visible local GPU."""

    if local_world_size is not None and local_world_size != world_size:
        raise EvaluationConfigError(
            "offline evaluation is single-node only: "
            f"LOCAL_WORLD_SIZE={local_world_size} does not match WORLD_SIZE={world_size}"
        )
    if visible_gpu_count != world_size:
        raise EvaluationConfigError(
            "offline evaluation requires one rank per visible GPU: "
            f"WORLD_SIZE={world_size}, but {visible_gpu_count} GPU(s) are visible"
        )


def _split_python_spec(config_spec: str) -> tuple[str, str]:
    if ":" not in config_spec:
        return config_spec, "make_evaluation_suite"
    path, factory = config_spec.rsplit(":", 1)
    if not path or not factory:
        raise EvaluationConfigError(
            "evaluation config must be /path/to/file.py or /path/to/file.py:factory"
        )
    return path, factory


def _resolve_model_spec(name: str, flavor: str) -> ModelSpec:
    normalized_name = name.replace("/", ".")
    module_candidates = (
        f"torchtitan.models.{normalized_name}",
        f"torchtitan.experiments.{normalized_name}",
    )
    errors: list[str] = []
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            errors.append(f"{module_name}: {error}")
            continue
        registry: Callable[[str], ModelSpec] | None = getattr(
            module, "model_registry", None
        )
        if callable(registry):
            return registry(flavor)
    raise EvaluationConfigError(
        f"could not resolve model registry for model_spec.name={name!r}; "
        f"tried {module_candidates}. Import errors: {'; '.join(errors)}"
    )


def _overlay_dataclass(target: Any, values: dict[str, Any], path: str) -> None:
    if not is_dataclass(target):
        raise TypeError(f"{path} is not a dataclass")
    for key, value in values.items():
        if not hasattr(target, key):
            raise EvaluationConfigError(f"unknown saved config field {path}.{key}")
        current = getattr(target, key)
        child_path = f"{path}.{key}"
        if is_dataclass(current) and isinstance(value, dict):
            _overlay_dataclass(current, value, child_path)
        else:
            setattr(target, key, value)


def _required_mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise EvaluationConfigError(f"job config field {key!r} must be an object")
    return value


def _optional_mapping(values: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EvaluationConfigError(f"job config field {key!r} must be an object")
    return value


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationConfigError(
            f"job config field {key!r} must be a non-empty string"
        )
    return value
