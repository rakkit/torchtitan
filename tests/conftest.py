# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Skip test modules whose imports need packages this fork does not install.

``torchvision`` is deliberately absent: installing it resolves torch back to a
release build and downgraded this environment from 2.15.0.dev20260906+cu132 to
2.14.0+cu130 once already. Nothing this fork trains needs it -- it is reached
only by the multimodal/vision paths of models we do not use -- so the modules
that import it are ignored at collection rather than left as hard errors that
mask real failures elsewhere in the suite.
"""

import importlib.util


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


def pytest_ignore_collect(collection_path, config):
    if _TORCHVISION_ABSENT and collection_path.name in _TORCHVISION_MODULES:
        return True
    return None
