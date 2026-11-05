# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resuming the trainer's lr schedule through DCP, as the checkpointer does:
`dcp.load` loads the Statefuls in key order (lr_scheduler before optimizer).
A group's lr is the run's config (no checkpoint holds it): the scheduler sets
it from this run's base lr and schedule at the checkpoint's step, so a run
resumed under a changed schedule or base lr runs it from the next step on, in
the DiSCO group and in the AdamW group alike, and a plain resume is bitwise."""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn

from torchtitan.components.optim.lr_scheduler import LRSchedulersContainer
from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.container import DiSCOOptimizersContainer

T, K = 40, 12  # steps, the checkpoint's step


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


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8, bias=False)
        self.b = nn.Linear(8, 8, bias=False)

    def param_layouts(self):
        return {"a.weight": ParamLayout("matrix"), "b.weight": ParamLayout("matrix")}


def _trainer(*, base_lr=0.1, decay_ratio=0.2, decay_type="linear"):
    cfg = DiSCOOptimizersContainer.Config()
    cfg.implementation = "for-loop"
    cfg.groups = {
        "default": {  # DiSCO
            "lr": base_lr,
            "weight_decay": 0.0,
            "momentum": 0.9,
            "nesterov": False,
            "rule": "sgd",
            "args": {"pre": "identity", "compute": "identity", "scale": "one"},
        },
        "b": {"match": r"^b\.weight$", "optimizer": "AdamW", "lr": 0.4 * base_lr},
    }
    model = _Model()
    opt = DiSCOOptimizersContainer(
        cfg, model_parts=[model], parallelism_context=_NoParallel()
    )
    sched = LRSchedulersContainer.Config(
        warmup_steps=0,
        decay_ratio=decay_ratio,
        decay_type=decay_type,
        min_lr_factor=0.0,
    ).build(optimizers=opt, training_steps=T)
    return model, opt, sched


def _lrs(model, opt, sched, steps):
    """The groups' lrs at each of the next `steps` optimizer steps."""
    out = []
    for _ in range(steps):
        out.append([g["lr"] for o in opt.optimizers for g in o.param_groups])
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        sched.step()
    return out


def _resumed(saved: dict, live: dict, path):
    """A run with `saved` checkpointed after K steps, loaded through DCP by a
    run with `live`."""
    model, opt, sched = _trainer(**saved)
    _lrs(model, opt, sched, K)
    dcp.save({"optimizer": opt, "lr_scheduler": sched}, checkpoint_id=str(path))
    model2, opt2, sched2 = _trainer(**live)
    dcp.load({"optimizer": opt2, "lr_scheduler": sched2}, checkpoint_id=str(path))
    return model2, opt2, sched2


def test_a_plain_resume_is_bitwise(tmp_path):
    full = _lrs(*_trainer(), T)
    assert _lrs(*_resumed({}, {}, tmp_path), T - K) == full[K:]


@pytest.mark.parametrize(
    "new",
    [
        {"decay_ratio": 1.0, "decay_type": "sqrt"},  # another schedule
        {"base_lr": 0.05},  # another base lr
    ],
)
def test_a_resume_with_another_lr_runs_it_from_the_next_step(new, tmp_path):
    """Saved in the stable phase, resumed under `new`: step K + 1 already runs
    it, in both groups."""
    want = _lrs(*_trainer(**new), T)[K:]
    assert want[0] != _lrs(*_trainer(), T)[K]  # the two runs differ at K + 1
    assert _lrs(*_resumed({}, new, tmp_path), T - K) == want


def test_no_checkpoint_holds_a_group_lr(tmp_path):
    _, opt, _ = _trainer()
    keys = [k for k in opt.state_dict() if k.startswith("param_groups.")]
    assert keys and not [k for k in keys if k.endswith((".lr", ".initial_lr"))]
