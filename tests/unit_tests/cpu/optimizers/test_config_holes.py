# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configs that used to build and then do the wrong thing, or fail with an
unrelated error: each must either work or say what is wrong."""

import os
import socket

import pytest
import torch.distributed as dist
import torch.nn as nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan_recipes.models.opt_moe import moe_template_config


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


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(64, 16)
        self.wq = nn.Linear(16, 16, bias=False)
        self.feed_forward = (
            nn.Module()
        )  # the router at the model's path: the template freezes feed_forward.router.gate
        self.feed_forward.router = nn.Module()
        self.feed_forward.router.gate = nn.Linear(
            16, 8, bias=False
        )  # 8 experts: Gram Polar Express needs dims % 8
        self.norm = nn.RMSNorm(16)  # learnable gains: the template's gains group
        self.lm_head = nn.Linear(16, 64, bias=False)

    def param_layouts(self):
        return {
            "tok_embeddings.weight": ParamLayout("embedding"),
            "wq.weight": ParamLayout("matrix"),
            "feed_forward.router.gate.weight": ParamLayout("router"),
            "norm.weight": ParamLayout("vector"),
            "lm_head.weight": ParamLayout("unembedding"),
        }


def test_the_template_groups_build():
    """The template's groups must be a valid run on their own."""
    cfg = moe_template_config().optim.optimizer
    DiSCOOptimizersContainer(
        cfg, model_parts=[_Decoder()], parallelism_context=_NoParallel()
    )


@pytest.mark.parametrize("level", [-1, 4])
def test_gram_level_outside_0_to_3_is_refused(level):
    """Level 4 computed all of level 3 and logged no gram key at all."""
    cfg = DiSCOOptimizersContainer.Config(gram_level=level)
    with pytest.raises(ValueError, match="gram_level"):
        DiSCOOptimizersContainer._validate(cfg)


@pytest.mark.parametrize("key", ["rule", "args", "momentum", "nesterov"])
def test_an_adamw_group_refuses_disco_keys(key):
    """AdamW stored them and ignored them."""
    cfg = DiSCOOptimizersContainer.Config()
    cfg.groups = {
        "default": {"lr": 0.1},
        "head": {"match": "lm_head", "optimizer": "AdamW", "lr": 1e-3, key: None},
    }
    with pytest.raises(ValueError, match="AdamW"):
        DiSCOOptimizersContainer._param_group_configs(cfg)


def test_upstreams_optimizers_list_is_refused():
    """Upstream's container field `optimizers` (one optimizer per pattern) sits on
    this Config by inheritance; DiSCO's container builds from `groups` only, so a
    list set there would be ignored without this check."""
    from torchtitan.components.optim import AdamW

    cfg = DiSCOOptimizersContainer.Config()
    cfg.groups = {"default": {"lr": 0.1}}
    cfg.optimizers = [AdamW.Config(pattern=".*", lr=1e-3)]
    with pytest.raises(ValueError, match="optimizer.groups"):
        DiSCOOptimizersContainer._validate(cfg)


def test_disco_without_a_default_lr_says_so():
    """An AdamW default group used to end in `TypeError: missing 'lr'`."""
    cfg = DiSCOOptimizersContainer.Config()
    cfg.groups = {
        "default": {"optimizer": "AdamW", "lr": 1e-3},
        "matrices": {"match": "wq", "lr": 0.1},
    }
    with pytest.raises(ValueError, match="no default lr"):
        DiSCOOptimizersContainer(
            cfg, model_parts=[_Decoder()], parallelism_context=_NoParallel()
        )


def test_a_named_group_with_only_an_lr_does_not_take_the_defaults_own_args():
    """A named group that sets neither `rule` nor `args` runs the default's rule
    but is not the default group: it must not take the rule's `own` args (here
    the router's frozen row lengths, which name no unit of the group)."""
    cfg = (
        moe_template_config().optim.optimizer
    )  # its default group freezes moe.router.gate
    cfg.groups = {"wq": {"match": r"^wq\.weight$", "lr": 0.003}, **cfg.groups}
    c = DiSCOOptimizersContainer(
        cfg, model_parts=[_Decoder()], parallelism_context=_NoParallel()
    )
    groups = {tuple(g["param_names"]): g for o in c.optimizers for g in o.param_groups}
    wq = groups[("wq.weight",)]
    assert wq["rule"] == "amphione" and wq["lr"] == 0.003
    assert "frozen_scale_units" not in wq["args"]
    assert wq["args"]["polar_express_safety"] == 1.05  # inherited
    gate = next(
        g for names, g in groups.items() if "feed_forward.router.gate.weight" in names
    )
    assert gate["args"]["frozen_scale_units"] == ["feed_forward.router.gate"]


def test_a_named_group_under_the_defaults_rule_keeps_its_other_default_args():
    """Only the rule's `own` args stay with the default group: a named amphione
    group with args of its own still takes the default's polar_express_safety
    (an arg of no preset, so no rule `inherits` it)."""
    cfg = (
        moe_template_config().optim.optimizer
    )  # its default group freezes moe.router.gate
    cfg.groups = {
        "wq": {
            "match": r"^wq\.weight$",
            "rule": "amphione",
            "args": {"scale_nesterov": True},
        },
        **cfg.groups,
    }
    c = DiSCOOptimizersContainer(
        cfg, model_parts=[_Decoder()], parallelism_context=_NoParallel()
    )
    wq = next(
        g
        for o in c.optimizers
        for g in o.param_groups
        if g["param_names"] == ["wq.weight"]
    )
    assert wq["args"]["polar_express_safety"] == 1.05
    assert wq["args"]["scale_nesterov"] is True
    assert "frozen_scale_units" not in wq["args"]
