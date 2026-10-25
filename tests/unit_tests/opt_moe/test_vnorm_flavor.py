# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gauge-d16-w1024-vnorm is gauge-d16-w1024 plus a v RMSNorm over head_dim: the same config delta as
gauge-d16-w512 -> gauge-d16-w512-vnorm, and with the study's gains (rmsnorm everywhere, cell.py GAINS on)
exactly one extra [head_dim] weight per layer."""

import copy
import dataclasses

import pytest
import torch

from torchtitan.models.opt_moe import model_registry, moe_opt_moe_configs
from torchtitan.models.opt_moe.config_registry import moe_template_config


def _delta(a, b, path=""):
    """{field path: (a, b)} over nested dataclass configs."""
    if dataclasses.is_dataclass(a) and type(a) is type(b):
        out = {}
        for f in dataclasses.fields(a):
            out.update(
                _delta(
                    getattr(a, f.name),
                    getattr(b, f.name),
                    f"{path}.{f.name}" if path else f.name,
                )
            )
        return out
    return {} if a == b else {path: (a, b)}


def _params(flavor, gains):
    spec = copy.deepcopy(model_registry(flavor))
    if gains:  # cell.py GAINS on: the norms carry weights
        spec.norm_type = (
            spec.layer.norm_type
        ) = spec.layer.attention.norm_type = "rmsnorm"
    job = moe_template_config()
    job.model = spec
    spec.update_from_config(config=job)
    with torch.device("meta"):
        model = spec.build()
    return {n: tuple(p.shape) for n, p in model.named_parameters()}


def test_the_flavor_exists():
    assert "gauge-d16-w1024-vnorm" in moe_opt_moe_configs


@pytest.mark.parametrize("width", [512, 1024])
def test_the_vnorm_delta_is_one_field(width):
    base, vn = model_registry(f"gauge-d16-w{width}"), model_registry(
        f"gauge-d16-w{width}-vnorm"
    )
    d = _delta(base, vn)
    assert d.pop("flavor") == (f"gauge-d16-w{width}", f"gauge-d16-w{width}-vnorm")
    assert d == {"layer.attention.v_norm": (False, True)}


@pytest.mark.parametrize("gains", [True, False])
def test_w1024_vnorm_adds_one_head_dim_gain_per_layer(gains):
    base, vn = _params("gauge-d16-w1024", gains), _params(
        "gauge-d16-w1024-vnorm", gains
    )
    extra = {n: s for n, s in vn.items() if n not in base}
    assert all(vn[n] == s for n, s in base.items())
    want = (
        {f"layers.{i}.attention.v_norm.weight": (128,) for i in range(16)}
        if gains
        else {}
    )
    assert extra == want
    n_base = sum(torch.Size(s).numel() for s in base.values())
    n_vn = sum(torch.Size(s).numel() for s in vn.values())
    assert n_vn - n_base == (16 * 128 if gains else 0)


def test_w512_pair_adds_the_same_gains():
    """The reference pair the 84M runs used: the same per-layer [head_dim] gain."""
    base, vn = _params("gauge-d16-w512", True), _params("gauge-d16-w512-vnorm", True)
    assert {n: s for n, s in vn.items() if n not in base} == {
        f"layers.{i}.attention.v_norm.weight": (128,) for i in range(16)
    }
