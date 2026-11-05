# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Under PP a stage holds only its own parameters: the embedding lives on the
first stage, the head on the last. A group that matches nothing on a stage is
dropped for it there; without PP it stays an error (it catches a typo)."""

import socket

import pytest
import torch.distributed as dist
import torch.nn as nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.container import _Group, _KIND, DiSCOOptimizersContainer


class _MiddleStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])


def _configs():
    return [
        _Group(pattern=_KIND + "embedding", optimizer_name="DiSCO"),
        _Group(pattern=r"tok_embeddings\.weight", optimizer_name="DiSCO"),
        _Group(pattern=".*", optimizer_name="DiSCO"),
    ]


def _layouts():
    return {"layers.0.weight": ParamLayout("matrix")}


def test_a_stage_drops_the_groups_it_has_no_parameters_for():
    out = DiSCOOptimizersContainer._by_kind(
        _configs(), _MiddleStage(), _layouts(), pp=True
    )
    assert [c.pattern for c in out] == [".*"]


def test_without_pp_an_empty_kind_group_is_an_error():
    with pytest.raises(ValueError, match="kind 'embedding'"):
        DiSCOOptimizersContainer._by_kind(_configs(), _MiddleStage(), _layouts())


@pytest.fixture(scope="module", autouse=True)
def one_rank():
    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


class _PP:
    """One PP rank seen alone (no PP mesh): the union is this rank's."""

    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = ep_enabled = etp_enabled = False
    pp_enabled = True
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


class _Stage(nn.Module):
    """A stage of dense layers only (`with_router` False) or with an MoE router."""

    def __init__(self, with_router: bool):
        super().__init__()
        self.wq = nn.Linear(8, 8, bias=False)
        if with_router:
            self.moe = nn.Module()
            self.moe.router = nn.Module()
            self.moe.router.gate = nn.Linear(8, 8, bias=False)

    def param_layouts(self):
        out = {"wq.weight": ParamLayout("matrix")}
        if hasattr(self, "moe"):
            out["moe.router.gate.weight"] = ParamLayout("router")
        return out


def _container(*parts):
    cfg = DiSCOOptimizersContainer.Config()
    cfg.groups = {
        "default": {
            "lr": 0.01,
            "momentum": 0.95,
            "nesterov": True,
            "weight_decay": 0.0,
            "rule": "amphione",
            "args": {"compute": "svd", "frozen_scale_units": ["moe.router.gate"]},
        }
    }
    return DiSCOOptimizersContainer(cfg, model_parts=list(parts), parallelism_context=_PP())


def test_a_stage_without_the_frozen_type_builds_and_the_union_is_checked():
    """The frozen router gate lives on another stage: this stage builds; the
    union over the stages (here the rank's two parts) finds it."""
    _container(_Stage(with_router=False), _Stage(with_router=True))


def test_a_frozen_type_on_no_stage_is_refused():
    with pytest.raises(ValueError, match="no unit of that type on any pipeline stage"):
        _container(_Stage(with_router=False))
