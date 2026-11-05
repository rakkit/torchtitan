# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load Python configuration recipes from the command line."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import pprint
from typing import Any, get_args

from torchtitan.config.configs import CommBackend
from torchtitan.config.configurable import Configurable
from torchtitan.config.override import OverrideConfig, parse_cli_imports


logger = logging.getLogger(__name__)


def _parse_resume_step(value: str) -> int:
    try:
        step = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("resume step must be an integer") from error
    if step < -1:
        raise argparse.ArgumentTypeError("resume step must be -1 or non-negative")
    return step


class ConfigLoader:
    """Load a Python config recipe with a deliberately small CLI surface."""

    def load(self, args: list[str] | None = None) -> Configurable.Config:
        parser = self._parser()
        namespace, extra = parser.parse_known_args(args)
        if _is_config_file(namespace.config):
            config = _load_config_file(namespace.config)
        elif namespace.module is None:
            parser.error("--module is required unless --config names a .py recipe file")
        else:
            config = self._load_config(namespace.module, namespace.config)
        _apply_field_overrides(config, extra)

        if namespace.override:
            override_config = getattr(config, "override", None)
            if not isinstance(override_config, OverrideConfig):
                raise ValueError(
                    "--override requires a recipe whose top-level config has an "
                    f"`override` field; {type(config).__qualname__} has none."
                )
            override_config.imports.extend(parse_cli_imports(namespace.override))

        self._apply_launch_options(config, namespace)
        self._validate_assets_path(config)
        # Recipes assign fields after the constructing __post_init__, and the
        # launch options assign more. Overrides run later, in apply_overrides.
        config.__post_init__()

        if namespace.print_config:
            pprint.pprint(config)
            raise SystemExit(0)
        return config

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Load and run a TorchTitan Python configuration recipe."
        )
        parser.add_argument(
            "--module",
            required=False,
            help="Fully qualified Python module containing the config factory.",
        )
        parser.add_argument(
            "--config",
            required=True,
            help="Config factory function in the selected module.",
        )
        parser.add_argument(
            "--override",
            action="append",
            default=[],
            metavar="TARGET[=JSON]",
            help=(
                "Apply a registered config override. Repeat this flag to apply "
                "multiple overrides."
            ),
        )
        parser.add_argument(
            "--comm-backend",
            choices=get_args(CommBackend),
            help="Override the distributed communication backend.",
        )
        parser.add_argument(
            "--output-dir",
            help="Override the root output directory.",
        )
        parser.add_argument(
            "--resume-step",
            type=_parse_resume_step,
            help="Checkpoint step to load; -1 selects the latest checkpoint.",
        )
        parser.add_argument(
            "--print-config",
            action="store_true",
            help="Print the resolved config and exit.",
        )
        return parser

    @staticmethod
    def _load_config(module_name: str, config_name: str) -> Configurable.Config:
        """Call ``config_name`` from an importable Python module."""
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            missing_module = error.name is not None and (
                error.name == module_name or module_name.startswith(f"{error.name}.")
            )
            if not missing_module:
                raise
            raise ImportError(
                f"Cannot import config module {module_name!r}."
            ) from error

        config_fn = getattr(module, config_name, None)
        if config_fn is None or not callable(config_fn):
            available = [
                name
                for name in dir(module)
                if not name.startswith("_")
                and callable(getattr(module, name))
                and name[0].islower()
            ]
            raise ValueError(
                f"Config function {config_name!r} not found in {module_name}. "
                f"Available config functions: {available}"
            )
        config = config_fn()
        if not isinstance(config, Configurable.Config):
            raise TypeError(
                f"{module_name}.{config_name} must return Configurable.Config, "
                f"got {type(config).__qualname__}."
            )
        return config

    @staticmethod
    def _apply_launch_options(
        config: Configurable.Config, args: argparse.Namespace
    ) -> None:
        if args.output_dir is not None:
            ConfigLoader._set_attr(config, "dump_folder", args.output_dir)

        if args.comm_backend is not None:
            trainer_config = ConfigLoader._trainer_config(config)
            trainer_config.comm.backend = args.comm_backend

        if args.resume_step is not None:
            trainer_config = ConfigLoader._trainer_config(config)
            checkpointer = trainer_config.checkpointer
            if checkpointer is None:
                raise ValueError(
                    "--resume-step requires checkpointing to be configured."
                )
            checkpointer.load_step = args.resume_step

    @staticmethod
    def _trainer_config(config: Configurable.Config):
        """Return the training-engine config of a pretraining or RL recipe.

        Pretraining recipes are a ``TrainingEngine.Config`` subclass; RL recipes
        nest one under ``trainer``.
        """
        from torchtitan.training_engine import TrainingEngine

        if isinstance(config, TrainingEngine.Config):
            return config
        trainer_config = getattr(config, "trainer", None)
        if isinstance(trainer_config, TrainingEngine.Config):
            return trainer_config
        raise ValueError(
            "--comm-backend and --resume-step require a recipe that "
            "is or contains a training-engine config (TrainingEngine.Config)."
        )

    @staticmethod
    def _set_attr(config: object, name: str, value: Any) -> None:
        if not hasattr(config, name):
            raise ValueError(
                f"--{name.replace('_', '-')} is not supported by "
                f"{type(config).__qualname__}."
            )
        setattr(config, name, value)

    @staticmethod
    def _validate_assets_path(config: object) -> None:
        assets_path = getattr(config, "hf_assets_path", None)
        if assets_path is None:
            return
        if not os.path.exists(assets_path):
            logger.warning("HF assets path %s does not exist!", assets_path)
        elif assets_path.endswith("tokenizer.model"):
            raise ValueError(
                "hf_assets_path must name an assets directory, not the legacy "
                "tokenizer.model file. Download the Hugging Face assets and "
                "update the recipe."
            )


# ---------------------------------------------------------------------------
# This fork's launchers (reproduce_cfgs/sbatch): `--config` may be a recipe FILE
# (`/path/recipe.py` or `/path/recipe.py:factory`, default factory `make_config`),
# and `--a.b.c value` arguments set config fields. Upstream's own surface
# (`--module` + factory name) is unchanged.


def _is_config_file(spec: str) -> bool:
    return spec.split(":", 1)[0].endswith(".py")


def _load_config_file(spec: str) -> Configurable.Config:
    import importlib.util

    path, _, fn = spec.partition(":")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--config file {path!r} does not exist")
    module_spec = importlib.util.spec_from_file_location("_titan_recipe", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    name = fn or "make_config"
    factory = getattr(module, name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"{path} has no config factory {name!r}")
    config = factory()
    if not isinstance(config, Configurable.Config):
        raise TypeError(f"{path}:{name} must return Configurable.Config")
    return config


def _apply_field_overrides(config: Any, extra: list[str]) -> None:
    """`--a.b.c value` (or `--a.b.c=value`, or `--a.b.c` alone for a bool field:
    true): set that field, the value read as
    the field's current type (bool: true / false; a list or dict: JSON; None
    for a field that is None: JSON, else the string)."""
    import json

    i = 0
    while i < len(extra):
        arg = extra[i]
        if not arg.startswith("--") or len(arg) == 2:
            raise ValueError(f"unexpected argument {arg!r}")
        key, eq, value = arg[2:].partition("=")
        bare = not eq and (i + 1 >= len(extra) or extra[i + 1].startswith("--"))
        if not eq and not bare:
            value, i = extra[i + 1], i + 1
        i += 1
        # Field names take `_` for `-`; dict keys (e.g. optimizer group names)
        # are used as written.
        *path, leaf = key.split(".")
        target = config
        for part in path:
            target = _child(target, part, key)
        current = _child(target, leaf, key)
        if bare:  # `--a.b` alone: a bool field set to true
            if not isinstance(current, bool):
                raise ValueError(f"--{key} needs a value")
            value = "true"
        if isinstance(current, bool):
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"--{key} is a bool: true or false, got {value!r}")
            new = value.lower() in ("true", "1")
        elif isinstance(current, int):
            new = int(value)
        elif isinstance(current, float):
            new = float(value)
        elif isinstance(current, str):
            new = value
        elif isinstance(current, (list, tuple, dict)) or current is None:
            try:
                new = json.loads(value)
            except json.JSONDecodeError:
                if current is not None:
                    raise ValueError(f"--{key} needs JSON, got {value!r}") from None
                new = value
            if isinstance(current, tuple):
                new = tuple(new)
        else:
            raise ValueError(
                f"--{key}: cannot set a {type(current).__qualname__} from the command line"
            )
        if isinstance(target, dict):
            target[leaf] = new
        else:
            setattr(target, leaf.replace("-", "_"), new)


def _child(target: Any, part: str, key: str) -> Any:
    if isinstance(target, dict):
        if part not in target:
            raise ValueError(f"--{key}: no key {part!r} (has {sorted(target)})")
        return target[part]
    name = part.replace("-", "_")
    if not hasattr(target, name):
        raise ValueError(f"--{key}: {type(target).__qualname__} has no {name!r}")
    return getattr(target, name)
