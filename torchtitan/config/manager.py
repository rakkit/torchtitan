# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import importlib.util
import logging
import os
import sys
import warnings
from dataclasses import field, fields, is_dataclass, make_dataclass
from typing import Any

import tyro

from torchtitan.observability.logging import init_logger


logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Parses, merges, and validates a config from --module/--config and CLI sources.

    Configuration precedence:
        CLI args > config_registry function defaults

    --module selects the module (e.g., llama3, deepseek_v3).
    --config selects a config_registry function (e.g., llama3_debugmodel).
    CLI arguments use the format <section>.<key> to override config values.
    """

    def __init__(self):
        self.register_tyro_rules(custom_registry)

    def parse_args(self, args: list[str] | None = None):
        if args is None:
            args = sys.argv[1:]
        loaded_config, args = self._load_config(args)
        config_cls = type(loaded_config)

        self.config = tyro.cli(
            config_cls, args=args, default=loaded_config, registry=custom_registry
        )

        self._validate_config()

        return self.config

    @staticmethod
    def _extract_model_flavor_override(args: list[str]) -> tuple[list[str], str | None]:
        """
        Parse and strip model flavor overrides from args.

        Supports:
        - --model-flavor=<flavor>
        - --model-flavor <flavor>
        - --model.flavor=<flavor>
        - --model.flavor <flavor>
        """
        remaining: list[str] = []
        model_flavor: str | None = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--model-flavor="):
                model_flavor = arg.split("=", 1)[1]
            elif arg == "--model-flavor":
                if i + 1 >= len(args):
                    raise ValueError("--model-flavor requires a value")
                model_flavor = args[i + 1]
                i += 1
            elif arg.startswith("--model.flavor="):
                model_flavor = arg.split("=", 1)[1]
            elif arg == "--model.flavor":
                if i + 1 >= len(args):
                    raise ValueError("--model.flavor requires a value")
                model_flavor = args[i + 1]
                i += 1
            else:
                remaining.append(arg)
            i += 1
        return remaining, model_flavor

    @staticmethod
    def _model_registry_candidates(
        *,
        module_name: str | None,
        model_spec_name: str | None,
    ) -> list[str]:
        candidates: list[str] = []
        if module_name:
            candidates.extend(
                [
                    f"torchtitan.models.{module_name}",
                    f"torchtitan.experiments.{module_name}",
                ]
            )
        if model_spec_name:
            normalized = model_spec_name.replace("/", ".")
            candidates.extend(
                [
                    f"torchtitan.models.{normalized}",
                    f"torchtitan.experiments.{normalized}",
                ]
            )

        # Preserve order but deduplicate.
        seen = set()
        deduped: list[str] = []
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            deduped.append(c)
        return deduped

    def _apply_model_flavor_override(
        self,
        *,
        loaded_config,
        model_flavor: str,
        module_name: str | None,
    ) -> None:
        """
        Replace loaded_config.model_spec via module's model_registry(model_flavor),
        while preserving user overrides under model_spec.model when possible.
        """
        model_spec = getattr(loaded_config, "model_spec", None)
        model_spec_name = getattr(model_spec, "name", None)
        candidates = self._model_registry_candidates(
            module_name=module_name,
            model_spec_name=model_spec_name,
        )

        registry_fn = None
        chosen_module = None
        for module_path in candidates:
            try:
                m = importlib.import_module(module_path)
            except ImportError:
                continue
            candidate = getattr(m, "model_registry", None)
            if callable(candidate):
                registry_fn = candidate
                chosen_module = module_path
                break

        if registry_fn is None:
            raise ValueError(
                "Could not resolve model_registry for --model-flavor override. "
                f"Tried modules: {candidates}. "
                "Provide --module with an importable model/experiment module."
            )

        previous_model_spec = getattr(loaded_config, "model_spec", None)
        try:
            new_model_spec = registry_fn(model_flavor)
        except Exception as e:
            raise ValueError(
                f"Failed to apply --model-flavor='{model_flavor}' "
                f"via {chosen_module}.model_registry(...): {e}"
            ) from e

        # Preserve explicit user overrides under model_spec.model from the previous
        # config object. This keeps external-config changes even when flavor is switched.
        old_flavor = getattr(previous_model_spec, "flavor", None)
        if (
            old_flavor is not None
            and old_flavor != model_flavor
            and is_dataclass(getattr(previous_model_spec, "model", None))
            and is_dataclass(getattr(new_model_spec, "model", None))
        ):
            old_base_spec = registry_fn(old_flavor)
            overrides = self._dataclass_overrides(
                old_base_spec.model,
                previous_model_spec.model,
            )
            self._apply_dataclass_overrides(new_model_spec.model, overrides)

        loaded_config.model_spec = new_model_spec

    @staticmethod
    def _dataclass_overrides(base_obj, current_obj) -> dict[str, Any]:
        """Collect values in current_obj that differ from base_obj (recursively)."""
        if not (is_dataclass(base_obj) and is_dataclass(current_obj)):
            return {}

        out: dict[str, Any] = {}
        for f in fields(base_obj):
            name = f.name
            if not hasattr(current_obj, name):
                continue
            base_v = getattr(base_obj, name)
            cur_v = getattr(current_obj, name)
            if is_dataclass(base_v) and is_dataclass(cur_v):
                sub = ConfigManager._dataclass_overrides(base_v, cur_v)
                if sub:
                    out[name] = sub
            elif cur_v != base_v:
                out[name] = copy.deepcopy(cur_v)
        return out

    @staticmethod
    def _apply_dataclass_overrides(target_obj, overrides: dict[str, Any]) -> None:
        """Apply recursive overrides (from _dataclass_overrides) to target_obj."""
        for name, value in overrides.items():
            if not hasattr(target_obj, name):
                continue
            target_v = getattr(target_obj, name)
            if isinstance(value, dict) and is_dataclass(target_v):
                ConfigManager._apply_dataclass_overrides(target_v, value)
            else:
                setattr(target_obj, name, copy.deepcopy(value))

    @staticmethod
    def _is_external_config_spec(config_name: str) -> bool:
        """Whether --config points at an external Python file rather than a
        config_registry function name."""
        if ".py" in config_name:
            return True
        if "/" in config_name or "\\" in config_name:
            return True
        if config_name.startswith(".") or config_name.startswith("~"):
            return True
        return False

    @staticmethod
    def _split_external_config_spec(config_spec: str) -> tuple[str, str | None]:
        """Parse '/path/to/config.py[:function_name]' into (path, function_name)."""
        if ":" not in config_spec:
            return config_spec, None

        maybe_path, maybe_func = config_spec.rsplit(":", 1)
        if maybe_path.endswith(".py") and maybe_func:
            return maybe_path, maybe_func
        return config_spec, None

    @staticmethod
    def _public_callable_names(module_obj) -> list[str]:
        return sorted(
            name
            for name in dir(module_obj)
            if not name.startswith("_") and callable(getattr(module_obj, name))
        )

    def _load_external_config(self, config_spec: str):
        """Build a config object from a standalone Python file.

        Upstream only resolves --config against a module's config_registry.
        Experiment configs here live as free-standing files under
        launch_scripts/ and reproduce_cfgs/ so a run is reproducible from one
        file, so --config also accepts a path, optionally with ':function'.
        """
        config_path_raw, config_fn_name = self._split_external_config_spec(config_spec)
        config_path = os.path.abspath(os.path.expanduser(config_path_raw))
        if not os.path.isfile(config_path):
            raise ValueError(
                f"External config file '{config_path_raw}' was not found "
                f"(resolved path: '{config_path}')."
            )

        module_name = f"torchtitan_external_config_{abs(hash(config_path))}"
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import external config file '{config_path}'.")

        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        available = self._public_callable_names(config_module)
        if config_fn_name is None:
            for candidate in ("make_config", "config", "get_config"):
                config_fn = getattr(config_module, candidate, None)
                if callable(config_fn):
                    return config_fn()

            if len(available) == 1:
                return getattr(config_module, available[0])()

            raise ValueError(
                f"External config file '{config_path}' requires selecting a function. "
                "Use --config /path/to/config.py:function_name. "
                f"Available callables: {available}"
            )

        config_fn = getattr(config_module, config_fn_name, None)
        if config_fn is None or not callable(config_fn):
            raise ValueError(
                f"Config function '{config_fn_name}' not found in external config file "
                f"'{config_path}'. Available callables: {available}"
            )
        return config_fn()

    def _load_config(self, args: list[str]) -> tuple[object, list[str]]:
        # --model.flavor / --model-flavor swaps the arch config after the
        # config file or registry function has produced it, keeping any
        # overrides the config made under model_spec.model. Upstream has no
        # equivalent: flavor is baked into the registry function name.
        args, model_flavor = self._extract_model_flavor_override(args)
        """Parse --module and --config from args, load config from config_registry.

        Both --module and --config are required.
        Returns (loaded_config, filtered_args) with --module/--config stripped.
        """
        module_name = None
        config_name = None
        filtered_args = []

        i = 0
        while i < len(args):
            arg = args[i]

            # Handle --module=X and --module X forms
            if arg.startswith("--module="):
                module_name = arg.split("=", 1)[1]
            elif arg == "--module":
                if i + 1 < len(args):
                    module_name = args[i + 1]
                    i += 1
                else:
                    raise ValueError("--module requires a value")
            # Handle --config=X and --config X forms
            elif arg.startswith("--config="):
                config_name = arg.split("=", 1)[1]
            elif arg == "--config":
                if i + 1 < len(args):
                    config_name = args[i + 1]
                    i += 1
                else:
                    raise ValueError("--config requires a value")
            else:
                filtered_args.append(arg)

            i += 1

        if module_name is None:
            raise ValueError(
                "--module is required. Example: --module llama3 --config llama3_debugmodel"
            )
        if config_name is None:
            raise ValueError(
                "--config is required. Example: --module llama3 --config llama3_debugmodel"
            )

        if self._is_external_config_spec(config_name):
            loaded_config = self._load_external_config(config_name)
            if model_flavor is not None:
                self._apply_model_flavor_override(
                    loaded_config=loaded_config,
                    model_flavor=model_flavor,
                    module_name=module_name,
                )
            return loaded_config, filtered_args

        from torchtitan.experiments import _supported_experiments

        # Validate module name
        from torchtitan.models import _supported_models

        all_supported = _supported_models | _supported_experiments

        module = None
        module_path = None

        # Import config_registry from module based on module specification
        if module_name in all_supported:
            # short module from supported module list (search models, then
            # experiments, then RL examples for their per-example config_registry)
            for prefix in (
                "torchtitan.models",
                "torchtitan.experiments",
                "torchtitan.rl.examples",
            ):
                module_path = f"{prefix}.{module_name}.config_registry"
                try:
                    module = importlib.import_module(module_path)
                    break
                except ImportError:
                    continue
            if module is None:
                raise ImportError(
                    f"Cannot import config_registry for module '{module_name}' "
                    f"from torchtitan.models, torchtitan.experiments, or "
                    f"torchtitan.rl.examples"
                )
        else:
            # Fully qualified module path: try appending .config_registry first,
            # then fall back to importing directly (e.g., torchtitan.models.llama3
            # -> torchtitan.models.llama3.config_registry)
            for candidate in (f"{module_name}.config_registry", module_name):
                try:
                    module = importlib.import_module(candidate)
                    module_path = candidate
                    break
                except ImportError:
                    continue
            if module is None:
                raise ImportError(
                    f"Cannot import module '{module_name}' or "
                    f"'{module_name}.config_registry'. "
                    f"For shorthands, supported modules are: {sorted(all_supported)}"
                )

        # Get the config function
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
                f"Config function '{config_name}' not found in {module_path}. "
                f"Available config functions: {available}"
            )

        loaded_config = config_fn()
        if model_flavor is not None:
            self._apply_model_flavor_override(
                loaded_config=loaded_config,
                model_flavor=model_flavor,
                module_name=module_name,
            )
        return loaded_config, filtered_args

    @staticmethod
    def _merge_configs(base, custom) -> type:
        """
        Merges a base config class with user-defined extensions.
        """
        warnings.warn(
            "ConfigManager._merge_configs is deprecated. "
            "Use Config subclasses with config_registry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result: list[str | tuple[str, Any] | tuple[str, Any, Any]] = []
        b_map = {f.name: f for f in fields(base)}
        c_map = {f.name: f for f in fields(custom)}

        for name, f in b_map.items():
            if (
                name in c_map
                and is_dataclass(f.type)
                and is_dataclass(c_map[name].type)
            ):
                m_type = ConfigManager._merge_configs(f.type, c_map[name].type)
                result.append((name, m_type, field(default_factory=m_type)))

            # Custom field overrides base type
            elif name in c_map:
                result.append((name, c_map[name].type, c_map[name]))

            # Only in Base
            else:
                result.append((name, f.type, f))

        # Only in Custom
        for name, f in c_map.items():
            if name not in b_map:
                result.append((name, f.type, f))

        return make_dataclass(f"Merged{base.__name__}", result, bases=(base,))

    def _validate_config(self) -> None:
        # TODO: temporary mitigation of BC breaking change in hf_assets_path
        #       tokenizer default path, need to remove later
        if not os.path.exists(
            self.config.hf_assets_path  # pyrefly: ignore[missing-attribute]
        ):
            logger.warning(
                f"HF assets path {self.config.hf_assets_path} does not exist!"
            )
            old_tokenizer_path = (
                "torchtitan/datasets/tokenizer/original/tokenizer.model"
            )
            if os.path.exists(old_tokenizer_path):
                self.config.hf_assets_path = (  # pyrefly: ignore[missing-attribute]
                    old_tokenizer_path
                )
                logger.warning(
                    f"Temporarily switching to previous default tokenizer path {old_tokenizer_path}. "
                    "Please download the new tokenizer files (python scripts/download_hf_assets.py) and update your config."
                )
        else:
            # Check if we are using tokenizer.model, if so then we need to alert users to redownload the tokenizer
            if self.config.hf_assets_path.endswith(  # pyrefly: ignore[missing-attribute]
                "tokenizer.model"
            ):
                raise Exception(
                    "You are using the old tokenizer.model, please redownload the tokenizer ",
                    "(python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer) ",
                    " and update your config to the directory of the downloaded tokenizer.",
                )

    @staticmethod
    def register_tyro_rules(registry: tyro.constructors.ConstructorRegistry) -> None:
        @registry.primitive_rule
        def list_str_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Support for comma separated string parsing"""
            if type_info.type != list[str]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="A,B,C,...",
                instance_from_str=lambda args: args[0].split(","),
                is_instance=lambda instance: all(isinstance(i, str) for i in instance),
                str_from_instance=lambda instance: [",".join(instance)],
            )

        @registry.primitive_rule
        def override_imports_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Parse ``--override.imports`` targets, each with optional kwargs.

            Needed because ``OverrideConfig.imports`` is a ``str | tuple`` union,
            which tyro cannot render as a CLI flag on its own. Tokens are space-
            or comma-separated ``module.function`` targets; attach kwargs to a
            target with ``target=<json>`` (see ``parse_cli_imports``),
            e.g. ``--override.imports
            'my_pkg.triton_rope.triton_rope={"block_size": 256}'``.
            """
            from torchtitan.config.override import (
                format_cli_imports,
                OverrideImport,
                parse_cli_imports,
            )

            if type_info.type != list[OverrideImport]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs="*",
                metavar="TARGET[='{\"k\": v}'] ...",
                instance_from_str=lambda args: parse_cli_imports(args),
                is_instance=lambda instance: isinstance(instance, list)
                and all(
                    isinstance(i, str)
                    or (
                        isinstance(i, tuple)
                        and len(i) == 2
                        and isinstance(i[0], str)
                        and isinstance(i[1], dict)
                    )
                    for i in instance
                ),
                str_from_instance=lambda instance: format_cli_imports(instance),
            )


# Initialize the custom registry for tyro
custom_registry = tyro.constructors.ConstructorRegistry()


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Run this module directly to debug or inspect configuration parsing.
    #
    # Examples:
    #   Parse and print a config with CLI arguments:
    #     > python -m torchtitan.config.manager --module llama3 --config llama3_debugmodel
    #
    #   Show help message:
    #     > python -m torchtitan.config.manager --module llama3 --config llama3_debugmodel --help
    #
    # -----------------------------------------------------------------------------

    init_logger()

    try:

        from rich import print as rprint
        from rich.pretty import Pretty

        config_manager = ConfigManager()
        config = config_manager.parse_args()

        rprint(Pretty(config))
    except ImportError:
        config_manager = ConfigManager()
        config = config_manager.parse_args()
        logger.info(config)
        logger.warning("rich is not installed, show the raw config")
