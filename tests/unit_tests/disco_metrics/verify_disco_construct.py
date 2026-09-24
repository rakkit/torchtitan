# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Construct a real DiSCO optimizer and touch the attributes the class is supposed
to have.

This exists because of a real failure: a `@property` was inserted into the
middle of `DiSCO.__init__`, which silently terminated the method -- everything
after it became unreachable class-body code, `super().__init__` never ran, and
`param_groups` did not exist. Every other suite here passed, because none of
them instantiate the class (the ordering checks parse the AST, the helper
checks call free functions). It only surfaced as an AttributeError inside the
LR scheduler on a 4-GPU job, minutes of queue time later.

Cheap structural smoke test: no distributed init, no CUDA required.
"""
import os as _os
import sys

_REPO = _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")
)
sys.path.insert(0, _REPO)
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torchtitan.models.common.param_layout import by_rank  # noqa: E402

from torchtitan.optimizers.disco import DiSCO

# DiSCO queries the default process group during construction. A single-rank
# gloo group is enough and needs neither CUDA nor a second process.
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group("gloo", rank=0, world_size=1)

fails = []


def check(name, cond, detail=""):
    print(
        ("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else "")
    )
    if not cond:
        fails.append(name)


model = nn.Sequential(nn.Linear(32, 64, bias=False), nn.Linear(64, 32, bias=False))
# Every parameter must be declared; this model's are plain matrices.
model.param_layouts = lambda: {n: by_rank(p.shape) for n, p in model.named_parameters()}
named = list(model.named_parameters())
groups = [{"params": [p for _, p in named], "param_names": [n for n, _ in named]}]


class _NoParallel:
    """Minimal stand-in for ParallelDims: everything disabled, single rank."""

    dp_replicate_enabled = False
    dp_shard_enabled = False
    cp_enabled = False
    dp_cp_enabled = False
    fsdp_enabled = False
    tp_enabled = False
    pp_enabled = False
    ep_enabled = False
    etp_enabled = False

    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


try:
    opt = DiSCO(
        groups,
        parallel_dims=_NoParallel(),
        lr=0.1,
        momentum=0.0,
        args={"compute": "sign", "scale": "inv_n"},
        param_layouts=model.param_layouts(),
    )
    built = True
except Exception as e:  # pragma: no cover
    built = False
    check("DiSCO constructs", False, f"{type(e).__name__}: {str(e)[:120]}")

if built:
    check("DiSCO constructs", True)
    # torch.optim.Optimizer contract -- absent if __init__ was truncated
    for attr in ["param_groups", "state", "defaults"]:
        check(f"has .{attr}", hasattr(opt, attr))
    check(
        "param_groups non-empty",
        hasattr(opt, "param_groups") and len(opt.param_groups) > 0,
    )
    # attributes __init__ sets AFTER the point where the property was wrongly
    # inserted -- these are exactly what a truncated __init__ loses
    for attr in ["is_dp_rank_0", "entries", "fam", "_sink"]:
        check(f"__init__ ran past the property: .{attr}", hasattr(opt, attr))
    # the flags this work added
    for attr in ["track_spectrum", "log_metrics_locally", "need_to_calculate_norm"]:
        check(f"has .{attr}", hasattr(opt, attr))
    check(
        "sink.stores_norms is a bool",
        isinstance(opt._sink.stores_norms, bool),
    )
    opt.log_metrics_locally = True
    check("stores_norms follows log_metrics_locally", opt._sink.stores_norms is True)
    opt.log_metrics_locally = False
    check(
        "stores_norms falls back to is_dp_rank_0",
        opt._sink.stores_norms == opt.is_dp_rank_0,
    )
    check(
        "calculate_norm_at_next_step accepts the new kwargs",
        (
            opt.calculate_norm_at_next_step(
                ["supremum"], 0, track_spectrum=False, log_metrics_locally=True
            )
            or True
        )
        and opt.track_spectrum is False
        and opt.log_metrics_locally is True,
    )

print("\n== the container arms every DiSCO, not optimizers[part_index] ==")
try:
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

    cfg = DiSCOOptimizersContainer.Config()
    # AdamW first on purpose: one optimizer is built per (model part, group
    # name), so `optimizers[0]` is AdamW's and indexing by part skips DiSCO.
    cfg.groups = {
        "head": {"match": r"1\.weight", "optimizer": "AdamW", "lr": 1e-3},
        "default": {
            "lr": 0.1,
            "momentum": 0.0,
            "rule": "muon",
            "args": {"compute": "sign", "scale": "inv_n"},
        },
    }
    cfg.log_norm_freq = 1
    container = DiSCOOptimizersContainer(
        cfg, model_parts=[model], parallel_dims=_NoParallel()
    )
    discos = [o for o in container.optimizers if isinstance(o, DiSCO)]
    check("container built one optimizer per group", len(container.optimizers) == 2)
    check("a DiSCO exists", len(discos) == 1)
    check(
        "DiSCO is not at index 0",
        discos and container.optimizers[0] is not discos[0],
    )
    container.arm_norm_computation()
    check(
        "arm_norm_computation reached it",
        bool(discos) and discos[0].need_to_calculate_norm,
    )
    rows = container.begin_step(
        1, should_log=True, is_final_step=False, head_input=lambda: torch.zeros(1)
    )
    check("no head, so no head input is asked for", rows == 0)
except Exception as e:  # pragma: no cover
    check("container arming", False, f"{type(e).__name__}: {str(e)[:140]}")

if dist.is_initialized():
    dist.destroy_process_group()

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL DISCO CONSTRUCTION CHECKS PASSED")
