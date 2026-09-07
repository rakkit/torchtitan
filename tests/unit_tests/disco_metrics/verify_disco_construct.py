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
import sys

sys.path.insert(0, "resources/torchtitan")
import os

import torch
import torch.distributed as dist
import torch.nn as nn

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
        is_light=False,
        weight_decay=0.0,
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        eps=1e-20,
        norm_factor="spectral",
        backend="identity",
        backend_steps=5,
        parallel_dims=_NoParallel(),
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
    for attr in [
        "experts_need_transpose",
        "extra_reduce_for_HSDP",
        "is_dp_rank_0",
        "scale_params",
        "embed_params",
        "ddp_params",
        "fsdp_params",
        "expert_params",
    ]:
        check(f"__init__ ran past the property: .{attr}", hasattr(opt, attr))
    # the flags this work added
    for attr in ["track_spectrum", "log_metrics_locally", "need_to_calculate_norm"]:
        check(f"has .{attr}", hasattr(opt, attr))
    check(
        "_stores_norms is a property returning bool",
        isinstance(type(opt)._stores_norms, property)
        and isinstance(opt._stores_norms, bool),
    )
    check(
        "_stores_norms follows log_metrics_locally",
        (setattr(opt, "log_metrics_locally", True) or opt._stores_norms is True),
    )
    opt.log_metrics_locally = False
    check(
        "_stores_norms falls back to is_dp_rank_0",
        opt._stores_norms == opt.is_dp_rank_0,
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

if dist.is_initialized():
    dist.destroy_process_group()

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL DISCO CONSTRUCTION CHECKS PASSED")
