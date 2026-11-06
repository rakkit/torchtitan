# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Skip test modules whose imports this environment cannot satisfy.

``torchvision`` is not installed in llm_env (installing it resolves torch to a
release build instead of the nightly the env pins). Only the multimodal / vision
paths of models this fork does not train need it, so the modules that import it
are ignored at collection rather than left as hard errors that mask real
failures elsewhere in the suite.
"""

import importlib.util

import pytest


def _has_attr_module(module: str, submodule: str) -> bool:
    try:
        return importlib.util.find_spec(f"{module}.{submodule}") is not None
    except (ImportError, ValueError):
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

# `helion` is an optional kernel-authoring library backing
# `overrides/helion_rope.py`. It is not installed here and nothing this fork
# trains selects that override, so its whole test module is environmental
# noise -- 15 failures that say only "no module named helion".
_HELION_MODULES = frozenset({"test_helion_rope.py"})
_HELION_ABSENT = _missing("helion")


def pytest_ignore_collect(collection_path, config):
    name = collection_path.name
    if _TORCHVISION_ABSENT and name in _TORCHVISION_MODULES:
        return True
    if _DEFAULT_RESHARDER_ABSENT and name in _TORCH_CHECKPOINTING_MODULES:
        return True
    if _HELION_ABSENT and name in _HELION_MODULES:
        return True
    return None


@pytest.fixture(autouse=True)
def _reset_aux_loss_registry():
    """Clear the global aux-loss registry between tests.

    `AuxLoss._group_counts` is a ClassVar that every AuxLoss construction
    increments and nothing ever clears, so a test that builds an MoE leaks its
    registration into every test that runs after it. `collect_aux_loss_metrics`
    then returns a non-empty dict where the next test expects `{}` --
    test_trainer_accumulates_reused_cuda_graph_losses fails with

        Expected: extra_metrics={'n_tokens_seen': 3}
          Actual: extra_metrics={'n_tokens_seen': 3, 'load_balance_loss/mean': 0.0}

    but only when an opt_moe test ran first, which is why it passed in
    isolation and failed in the full suite. Snapshot and restore rather than
    blanket-clear, so a test that deliberately sets up state still sees it.
    """
    try:
        from torchtitan.models.common.aux_loss import AuxLoss
    except Exception:
        yield
        return
    before = dict(AuxLoss._group_counts)
    try:
        yield
    finally:
        AuxLoss._group_counts.clear()
        AuxLoss._group_counts.update(before)
