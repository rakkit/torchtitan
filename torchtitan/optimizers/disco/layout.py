# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a parameter's rows live, and which transport its units take.

What a parameter IS -- its units, whether dim 0 is a batch of experts, its role
in the network -- is declared by the model (`models/common/param_layout.py`).
This module maps those declarations onto storage: the rows of the `[R, C]` view
each rank holds, for any mesh coordinate, as unions of contiguous runs, so
pack / unpack never needs an index tensor.
"""

from enum import Enum
from math import prod

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import (
    _compute_local_shape_and_global_offset,
    compute_local_shape_and_global_offset,
)
from torch.distributed.tensor.placement_types import _is_shard_like

from torchtitan.models.common.param_layout import (
    ParamLayout,
    qkv_units,
    RunList,
    stacked_units,
    UnitSpec,
)


class Family(Enum):
    """Which transport a parameter takes."""

    LOCAL = "local"  # row-separable rule: embed / lm head / vectors / scalars
    EXPERT = "expert"  # batched on the rank that holds the experts
    SHARDED = "sharded"  # rows over the FSDP axis: one all-to-all pair
    REPLICATED = "replicated"  # whole on every rank: owners compute, all-gather


def matrix_view(shape: torch.Size) -> tuple[int, int]:
    """The `[R, C]` view the units are rows of. Vectors and scalars are one row."""
    if len(shape) <= 1:
        return (1, int(shape[0]) if len(shape) == 1 else 1)
    return (int(prod(shape[:-1])), int(shape[-1]))


def _checked_units(fqn: str, units, r: int, c: int) -> tuple[UnitSpec, ...]:
    """Declared units must tile the `[r, c]` matrix exactly."""
    covered = RunList.merged([run for u in units for run in u.rows.runs])
    if covered.runs != ((0, r),):
        raise ValueError(
            f"{fqn}: declared units must tile all {r} rows, cover {covered.runs}"
        )
    if sum(u.rows.n_rows for u in units) != r:
        raise ValueError(f"{fqn}: declared units overlap")
    for u in units:
        if tuple(u.shape) != (u.rows.n_rows, c):
            raise ValueError(
                f"{fqn}: unit {u.name!r} has {u.rows.n_rows} rows of {c} "
                f"but declares shape {tuple(u.shape)}"
            )
    return tuple(units)


def resolve_units(
    fqn: str, shape: torch.Size, layout: ParamLayout
) -> tuple[UnitSpec, ...]:
    """Fill in the default single unit when the layout declares none."""
    if layout.kind == "expert_batch":
        # Dim 0 is the batch: one [m, n] unit per expert, named by index, or
        # the declared per-expert units (a fused [E, 2, F, D]) named `i.name`.
        e, m, n = shape[0], prod(shape[1:-1]), shape[-1]
        per = (
            _checked_units(fqn, layout.units, m, n)
            if layout.units
            else (UnitSpec("", RunList.contiguous(0, m), (m, n)),)
        )
        return tuple(
            UnitSpec(
                f"{i}.{u.name}" if u.name else str(i),
                RunList(tuple((i * m + s, k) for s, k in u.rows.runs)),
                u.shape,
            )
            for i in range(e)
            for u in per
        )
    if layout.units:
        return _checked_units(fqn, layout.units, *matrix_view(shape))
    if layout.kind == "scalar":
        return (UnitSpec("", RunList.contiguous(0, 1), (1, 1)),)
    if layout.kind == "vector":
        return (UnitSpec("", RunList.contiguous(0, 1), (1, int(shape[0]))),)
    r, c = matrix_view(shape)
    return (UnitSpec("", RunList.contiguous(0, r), (r, c)),)


def _sharded_axes(p: DTensor) -> list[tuple[int, int, int]]:
    """`(mesh_axis, tensor_dim, axis_size)` for every Shard (or strided shard)
    placement on a non-unit axis."""
    out = []
    for axis, placement in enumerate(p.placements):
        if _is_shard_like(placement) and p.device_mesh.size(axis) > 1:
            out.append((axis, placement.dim, p.device_mesh.size(axis)))
    return out


def rows_of_coordinate(
    shape: torch.Size, view: tuple[int, int], p: DTensor, coordinate: list[int]
) -> RunList:
    """Global rows of the `[R, C]` view held by the rank at `coordinate`.

    DTensor's own chunking, for an arbitrary mesh coordinate, so a rank can
    build the tables for every peer without a collective. Handles the nested
    `_StridedShard` that FSDP-over-EP puts on expert parameters. The last dim
    must be unsharded, and a rank's block must be contiguous per leading dim,
    which every placement DiSCO accepts satisfies (checked against the data by
    the layout tests).
    """
    ndim = len(shape)
    for axis, placement in enumerate(p.placements):
        if _is_shard_like(placement) and p.device_mesh.size(axis) > 1:
            if placement.dim in (ndim - 1, -1):
                raise ValueError(
                    "a Shard on the last dim is tensor parallelism, which DiSCO refuses"
                )
    local, offset = _compute_local_shape_and_global_offset(
        tuple(shape), tuple(p.device_mesh.shape), list(coordinate), p.placements
    )
    strides = [prod(shape[d + 1 : -1]) for d in range(ndim - 1)]
    lead = ndim - 2
    if lead < 0:
        return RunList.contiguous(0, 1)
    if any(n == 0 for n in local[: lead + 1]):
        return RunList(())  # e.g. an expert Shard(0) over more ranks than experts
    runs = []
    idx = [0] * lead

    def emit(base):
        runs.append((base + offset[lead] * strides[lead], local[lead] * strides[lead]))

    if lead == 0:
        emit(0)
    else:
        while True:
            base = sum((offset[d] + idx[d]) * strides[d] for d in range(lead))
            emit(base)
            d = lead - 1
            while d >= 0:
                idx[d] += 1
                if idx[d] < local[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0:
                break
    return RunList.merged(runs)


def local_rows(p: torch.Tensor, view: tuple[int, int]) -> RunList:
    """This rank's rows of the `[R, C]` view; checked against DTensor's own answer."""
    r, _ = view
    if not isinstance(p, DTensor):
        return RunList.contiguous(0, r)
    coordinate = p.device_mesh.get_coordinate()
    rows = rows_of_coordinate(p.shape, view, p, coordinate)
    local_shape, _ = compute_local_shape_and_global_offset(
        p.shape, p.device_mesh, p.placements
    )
    if rows.n_rows != prod(local_shape[:-1]):
        raise AssertionError(
            f"row table disagrees with DTensor: {rows.n_rows} rows vs local shape {local_shape}"
        )
    return rows


def classify(
    fqn: str, p: torch.Tensor, layout: ParamLayout, *, row_rule: bool
) -> Family:
    """Which transport the parameter takes: layout kind first, then placement.
    `row_rule`: the rule runs on the rows a rank holds (a row or Gram rule).
    Experts are whole on their rank, so any rule runs on them there."""
    sharded = _sharded_axes(p) if isinstance(p, DTensor) else []
    if layout.kind == "expert_batch":
        for _, dim, _ in sharded:
            if dim not in (0, -p.ndim):
                raise ValueError(
                    f"{fqn}: expert_batch must be sharded on dim 0 (whole experts per "
                    f"rank), got Shard({dim})"
                )
        return Family.EXPERT
    if layout.kind in ("scalar", "vector") or row_rule:
        return Family.LOCAL  # the rows a rank holds are enough (plus a reduce at most)
    for _, dim, _ in sharded:
        if dim in (p.ndim - 1, -1):
            raise ValueError(f"{fqn}: Shard on the last dim is tensor parallelism")
    return Family.SHARDED if sharded else Family.REPLICATED


__all__ = [
    "classify",
    "Family",
    "local_rows",
    "matrix_view",
    "ParamLayout",
    "qkv_units",
    "resolve_units",
    "rows_of_coordinate",
    "RunList",
    "stacked_units",
    "UnitSpec",
]
