# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Residual-stream and RMSNorm-gain statistics, read on the per-unit logging cadence.

Residual: on an armed step the model's eager layer loop (``OPTMoEModel._trunk``,
between the compiled blocks) appends, per forward, the mean squares of every
block's input and total residual write (output - input) and of the final norm's
input. The blocks, their compiled graphs and the forward / backward are the same
whether or not a step is armed.

Gains: every RMSNorm weight, from the rank's local shard, before and after the
optimizer step.

Sinks: every attention sink value (one logit per head and layer) after the optimizer
step.

Everything here runs eagerly on logging steps only, outside the compiled
region, with one packed all-reduce and one device-to-host copy per family.
"""

from __future__ import annotations

import re

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.utils import fsdp_shard_mesh

__all__ = [
    "arm_residual_capture",
    "residual_metrics",
    "snapshot_gains",
    "gain_metrics",
    "sink_metrics",
]

_WRAPPERS = re.compile(r"(_checkpoint_wrapped_module|_orig_mod|_fsdp_wrapped_module)\.")


def _capturing(model_parts: list[nn.Module]) -> list[nn.Module]:
    return [p for p in model_parts if hasattr(p, "residual_capture")]


def _unsupported(parallel_dims) -> None:
    if parallel_dims is not None and (
        parallel_dims.tp_enabled or parallel_dims.pp_enabled
    ):
        raise NotImplementedError(
            "residual / gain statistics support neither TP nor PP"
        )


def arm_residual_capture(model_parts: list[nn.Module]) -> None:
    """Before an armed step's forward: start a fresh capture on every model part."""
    parts = _capturing(model_parts)
    if not parts:
        raise ValueError(
            "log_residual_stats needs a model that captures residual statistics "
            "(OPTMoEModel.residual_capture)"
        )
    for part in parts:
        part.residual_capture = []


def residual_metrics(model_parts: list[nn.Module], parallel_dims) -> dict[str, float]:
    """RMS of every block's input and total write, and of the final norm's input,
    over all of the step's tokens and ranks; disarms the capture."""
    rows = []
    for part in _capturing(model_parts):
        if part.residual_capture:
            rows.extend(part.residual_capture)
        part.residual_capture = None
    if not rows:
        return {}
    _unsupported(parallel_dims)
    stats = torch.stack(rows).sum(0)  # [2L + 1], summed over forwards
    packed = torch.cat([stats, stats.new_tensor([float(len(rows))])])
    mesh = (
        parallel_dims.get_optional_mesh("loss") if parallel_dims is not None else None
    )
    if mesh is not None:
        # Every rank's microbatches have the same token count, so summed mean
        # squares over summed forward counts is the global mean square.
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=mesh.get_group())
    *sums, count = packed.tolist()
    rms = [(v / count) ** 0.5 for v in sums]
    n_layers = (len(rms) - 1) // 2
    out: dict[str, float] = {}
    for i in range(n_layers):
        out[f"residual_rms/in/L-{i}"] = rms[2 * i]
        out[f"residual_rms/write/L-{i}"] = rms[2 * i + 1]
    out["residual_rms/final_norm_in"] = rms[-1]
    return out


def _gains(model_parts: list[nn.Module]) -> list[tuple[str, nn.Parameter]]:
    found = []
    for part in model_parts:
        for name, module in part.named_modules():
            weight = getattr(module, "weight", None)
            if isinstance(module, nn.RMSNorm) and isinstance(weight, nn.Parameter):
                found.append((_WRAPPERS.sub("", name), weight))
    return found


def _local(p: torch.Tensor) -> torch.Tensor:
    # Detached first: a metrics read, no gradient flows, so no grad_placements.
    p = p.detach()
    return p.to_local() if isinstance(p, DTensor) else p


def snapshot_gains(model_parts: list[nn.Module]) -> list[torch.Tensor]:
    """The local shard of every RMSNorm gain before the optimizer step; no gather."""
    return [_local(p).clone() for _, p in _gains(model_parts)]


def gain_metrics(
    model_parts: list[nn.Module], before: list[torch.Tensor], parallel_dims
) -> dict[str, float]:
    """Mean and RMS of each gain after the step, and ||g - g_before|| / ||g_before||."""
    gains = _gains(model_parts)
    if not gains:
        return {}
    _unsupported(parallel_dims)
    rows = []
    for (_, p), b in zip(gains, before, strict=True):
        a, b = _local(p).float(), b.float()
        rows.append(
            torch.stack(
                [a.sum(), a.square().sum(), (a - b).square().sum(), b.square().sum()]
            )
        )
    n = torch.tensor(
        [float(_local(p).numel()) for _, p in gains], device=rows[0].device
    )
    packed = torch.cat(
        [torch.stack(rows), n.unsqueeze(1)], dim=1
    )  # [P, 5], local sums and counts
    sharded = any(
        isinstance(p, DTensor) and any(pl.is_shard() for pl in p.placements)
        for _, p in gains
    )
    mesh = (
        fsdp_shard_mesh(parallel_dims)
        if (sharded and parallel_dims is not None)
        else None
    )
    if mesh is not None:
        # Shards add up; the replicate axis already holds full copies and is not reduced.
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=mesh.get_group())
    s = packed.tolist()
    out: dict[str, float] = {}
    for (name, _), (s1, s2, sd, sb, n) in zip(gains, s, strict=True):
        out[f"gain_mean/{name}"] = s1 / n
        out[f"gain_rms/{name}"] = (s2 / n) ** 0.5
        out[f"gain_rel_update/{name}"] = (sd / sb) ** 0.5 if sb > 0 else 0.0
    return out


def sink_metrics(model_parts: list[nn.Module], parallel_dims) -> dict[str, float]:
    """Every attention sink logit after the optimizer step, one value per head and layer.

    The sinks are tiny vectors, so each is gathered in full (a collective on every rank
    when sharded) and all of them are copied to the host at once."""
    sinks = [
        (_WRAPPERS.sub("", name)[: -len(".sinks")], p)
        for part in model_parts
        for name, p in part.named_parameters()
        if _WRAPPERS.sub("", name).endswith(".attention.sinks")
    ]
    if not sinks:
        return {}
    _unsupported(parallel_dims)
    # Detached: a metrics read, no gradient flows, so no grad_placements.
    full = [
        p.detach().full_tensor() if isinstance(p, DTensor) else p.detach()
        for _, p in sinks
    ]
    values = torch.stack([f.float() for f in full]).tolist()
    return {
        f"sink_value/{name}/head-{h}": v
        for (name, _), row in zip(sinks, values, strict=True)
        for h, v in enumerate(row)
    }
