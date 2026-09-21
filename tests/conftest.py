# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Skip test modules whose imports this environment cannot satisfy.

``torchvision`` is deliberately absent: installing it resolves torch back to a
release build and downgraded this environment from 2.15.0.dev20260906+cu132 to
2.14.0+cu130 once already. Nothing this fork trains needs it -- it is reached
only by the multimodal/vision paths of models we do not use -- so the modules
that import it are ignored at collection rather than left as hard errors that
mask real failures elsewhere in the suite.
"""

import importlib.util


def _has_attr_module(module: str, submodule: str) -> bool:
    try:
        return importlib.util.find_spec(f"{module}.{submodule}") is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        return True


# Modules that import torchvision at module scope, directly or transitively.
_TORCHVISION_MODULES = frozenset(
    {
        "test_qwen_multimodal_data.py",
        "test_mm_dataset_checkpointing.py",
        "test_mm_dataset_preprocessing.py",
        "test_packed_vision.py",
        "test_qwen3_6.py",
        "test_qwen3_8.py",
    }
)

_TORCHVISION_ABSENT = _missing("torchvision")

# Upstream's alternative `torch_checkpointing` backend imports
# `torch_checkpointing.default_resharder`, which does not exist in any published
# version of that package (neither PyPI 0.1.0 nor the GitHub main branch) --
# upstream is building against an unreleased one. This fork checkpoints through
# DCP (`components/checkpointer/dcp.py`), and nothing under `torchtitan/`
# imports the alternative backend, so the two modules that do are ignored.
# `test_debug_config_defaults` reaches it transitively and only covers other
# models' config registries (deepseek_v3, llama3, muse_glimmer, qwen3, torchft).
_TORCH_CHECKPOINTING_MODULES = frozenset(
    {
        "test_torch_checkpointing.py",
        "test_debug_config_defaults.py",
    }
)

_DEFAULT_RESHARDER_ABSENT = not _has_attr_module(
    "torch_checkpointing", "default_resharder"
)


def pytest_ignore_collect(collection_path, config):
    name = collection_path.name
    if _TORCHVISION_ABSENT and name in _TORCHVISION_MODULES:
        return True
    if _DEFAULT_RESHARDER_ABSENT and name in _TORCH_CHECKPOINTING_MODULES:
        return True
    return None
