# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every opt_moe tensor must have a HuggingFace mapping.

A tensor missing from `from_hf_map` used to be dropped while the export still
reported success -- that is how the 0.5.0 rename shipped 10 of 21 tensors.
`to_hf` now raises, but that is export time; this moves the failure to the PR
that renames the parameter.
"""

from __future__ import annotations

import re

import pytest
import torch

from torchtitan.models.opt_moe import model_registry
from torchtitan.models.opt_moe.state_dict_adapter import OPTMoEStateDictAdapter

# One per architecture shape, not per flavor: MoE+SWA, dense+SWA, MoE full
# attention, dense proxy. 118 flavors would be slow and would not cover more.
FLAVORS = [
    "mis-8b-proxy",
    "dense-1B-Proxy-8layers-test",
    "bsc-1B-7B-opt-g",
    "M1-dense-600M-full_rope-swa_128",
    "M3-moe-600M-full_rope-proxy",
]

# Skipped by name inside `to_hf`, deliberately and before the lookup.
DELIBERATE_SKIPS = ("tokens_per_expert",)


def _build(flavor: str):
    from types import SimpleNamespace

    from torchtitan.config import ParallelismConfig, TrainingConfig

    model_config = model_registry(flavor)
    model_config.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=4096),
        )
    )
    # Meta device: this test is about names, not values, and a real allocation
    # of bsc-1B-7B-opt-g would not fit in a CPU test.
    with torch.device("meta"):
        model = model_config.build()
    return model, getattr(model, "config", model_config)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_every_tensor_has_an_hf_mapping(flavor: str):
    model, model_config = _build(flavor)
    adapter = OPTMoEStateDictAdapter(model_config, None)
    to_hf_map = {v: k for k, v in adapter.from_hf_map.items()}

    unmapped = []
    for key in model.state_dict():
        if any(s in key for s in DELIBERATE_SKIPS):
            continue
        abstract = re.sub(r"(\d+)", "{}", key, count=1) if "layers" in key else key
        if abstract not in to_hf_map:
            unmapped.append(key)

    assert not unmapped, (
        f"{flavor}: {len(unmapped)} tensor(s) have no HF mapping, so "
        f"convert_to_hf would refuse to export this model: {sorted(unmapped)[:8]}. "
        "A parameter was renamed without updating "
        "OPTMoEStateDictAdapter.from_hf_map."
    )


def test_the_map_has_no_entry_that_no_flavor_owns():
    """The other direction: a mapping for a tensor that exists nowhere.

    Checked against the UNION of all the shapes, not one flavor. Most entries
    are legitimately conditional -- `feed_forward.w1` is absent from an all-MoE
    model, `attention_norm.weight` is absent under the parameter-free norm every
    flavor uses, `gate_proj` is absent without gated attention. An entry is only
    stale if NO shape owns it, which is the fingerprint of a half-applied
    rename.
    """
    native: set[str] = set()
    adapter = None
    for flavor in FLAVORS:
        model, model_config = _build(flavor)
        adapter = OPTMoEStateDictAdapter(model_config, None)
        for key in model.state_dict():
            native.add(re.sub(r"(\d+)", "{}", key, count=1) if "layers" in key else key)

    # Norm weights exist only under a PARAMETRIC `norm_type`. Every flavor
    # currently runs `np_rmsnorm`, which owns no parameters, so these are
    # absent from every shape without being dead: they are what the map would
    # need the moment anyone selects a parametric norm.
    conditional_on_parametric_norm = {
        "layers.{}.attention_norm.weight",
        "layers.{}.ffn_norm.weight",
        "norm.weight",
    }

    stale = [
        name
        for name in adapter.from_hf_map.values()
        if name not in conditional_on_parametric_norm
    ]
    stale = [
        name
        for name in stale
        # Some entries map to None: an HF key we deliberately do not import.
        if name is not None
        # Expert tensors carry two indices, so the single-substitution
        # normalisation above does not produce comparable names for them.
        and "experts" not in name and name not in native
    ]
    assert not stale, (
        f"from_hf_map references {len(stale)} tensor(s) that no opt_moe shape "
        f"owns: {sorted(stale)}. Either a parameter was renamed and the map "
        "half-updated, or the entry is dead."
    )
