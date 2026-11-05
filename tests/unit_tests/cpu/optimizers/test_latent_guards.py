# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarations and rules that nothing shipped gets wrong today, but that would
compute silently wrong updates: each is refused, or zeroed, explicitly."""

import os
import socket
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from torchtitan.models.common.param_layout import ParamLayout, RunList, UnitSpec
from torchtitan.optimizers.disco.layout import resolve_units
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.rules import Compute, register, Result, Rule
from torchtitan.optimizers.kernels.zeropower import zeropower_via_svd


@pytest.fixture(autouse=True, scope="module")
def _process_group():
    """One gloo rank for this module; destroyed after it if this module made it
    (a group left behind breaks other modules that create their own)."""
    created = not dist.is_initialized()
    if created:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    if created:
        dist.destroy_process_group()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


def _units(*specs):
    return tuple(
        UnitSpec(n, RunList.contiguous(a, k), shape) for n, a, k, shape in specs
    )


def test_a_unit_declared_with_the_wrong_shape_is_refused():
    layout = ParamLayout("matrix", _units(("a", 0, 6, (5, 4)), ("b", 6, 2, (2, 4))))
    with pytest.raises(ValueError, match="shape"):
        resolve_units("w", torch.Size([8, 4]), layout)


def test_overlapping_units_are_refused():
    layout = ParamLayout("matrix", _units(("a", 0, 6, (6, 4)), ("b", 4, 4, (4, 4))))
    with pytest.raises(ValueError, match="overlap"):
        resolve_units("w", torch.Size([8, 4]), layout)


@dataclass(frozen=True)
class _PerRowState(Compute):
    needs_weight = True
    has_state = True
    output = "weight"

    def state_shape(self, m, n):
        return (m,)

    def run(self, x, w, state, ctx):
        return Result(weight=w, state=state)


@register("per_row_state_probe")
def _per_row_state_probe():
    return Rule((), _PerRowState())


def test_units_of_one_parameter_keep_states_of_their_own_shapes():
    """The rule state is one tensor per unit (GQA's fused wqkv: q larger than
    k / v), keyed by the unit's position."""
    w = torch.nn.Parameter(torch.randn(8, 4))
    layout = ParamLayout("matrix", _units(("a", 0, 6, (6, 4)), ("b", 6, 2, (2, 4))))
    opt = DiSCO(
        [{"params": [w], "param_names": ["w"]}],
        parallelism_context=_NoParallel(),
        lr=0.1,
        rule="per_row_state_probe",
        param_layouts={"w": layout},
    )
    rs = opt.state[w]["rule_state"]
    cm = opt.entries[0].rule.compute
    assert {k: tuple(v.shape) for k, v in rs.items()} == {
        "0": tuple(cm.state_shape(6, 4)),
        "1": tuple(cm.state_shape(2, 4)),
    }


def test_svd_gives_no_direction_below_the_norm_floor():
    """Like every other backend: a moment decayed to ~1e-25 moves nothing."""
    tiny = 1e-25 * torch.randn(8, 4)
    assert zeropower_via_svd(tiny).abs().max().item() == 0.0
    assert zeropower_via_svd(torch.randn(8, 4)).abs().max().item() > 0.1
