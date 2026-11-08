# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a parameter IS, declared by the model, for optimizers that need more
than its shape.

A parameter is viewed as an `[R, C]` matrix (`R = prod(shape[:-1])`) and split
into UNITS -- the matrices an update rule works on -- each a set of global rows
kept as contiguous runs. `kind` says what the parameter is to the network:

    matrix        a plain weight matrix (one unit, or declared units for fused
                  projections: q / k / v, w1 / w3)
    expert_batch  dim 0 is a batch of independent matrices (one unit each)
    embedding     the input embedding table: rows are the vocabulary
    unembedding   the output projection: rows are the vocabulary
    router        an MoE router's gate: rows are the experts
    conv          a depthwise Conv1d `[C, 1, K]`: one `[C, K]` unit
    vector        a 1-D parameter: one row
    scalar        one number

A depthwise Conv1d `[C, 1, K]` and an expert batch `[E, F, D]` look alike by
shape and placement, which is why a rank-3+ parameter has to be declared.
These are facts about the model; which update rule a parameter gets stays in
the optimizer's config, which can select parameters by `kind`.
"""

from dataclasses import dataclass
from math import prod

import torch
import torch.nn as nn

KINDS = (
    "matrix",
    "expert_batch",
    "embedding",
    "unembedding",
    "router",
    "conv",
    "vector",
    "scalar",
)


@dataclass(frozen=True)
class RunList:
    """Sorted, disjoint `(start, length)` runs of row indices."""

    runs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        prev_end = -1
        for start, n in self.runs:
            if n <= 0 or start < prev_end:
                raise ValueError(f"runs must be positive and disjoint: {self.runs}")
            prev_end = start + n

    @classmethod
    def contiguous(cls, start: int, n: int) -> "RunList":
        return cls(((start, n),)) if n > 0 else cls(())

    @classmethod
    def merged(cls, runs) -> "RunList":
        """Sort and merge adjacent/overlapping runs."""
        out: list[tuple[int, int]] = []
        for start, n in sorted(r for r in runs if r[1] > 0):
            if out and start <= out[-1][0] + out[-1][1]:
                s, m = out[-1]
                out[-1] = (s, max(s + m, start + n) - s)
            else:
                out.append((start, n))
        return cls(tuple(out))

    @property
    def n_rows(self) -> int:
        return sum(n for _, n in self.runs)

    def intersect(self, other: "RunList") -> "RunList":
        """Rows in both lists, split at every run boundary of either side."""
        out = []
        for a_start, a_n in self.runs:
            a_end = a_start + a_n
            for b_start, b_n in other.runs:
                lo, hi = max(a_start, b_start), min(a_end, b_start + b_n)
                if hi > lo:
                    out.append((lo, hi - lo))
        return RunList(tuple(sorted(out)))

    def positions_within(self, within: "RunList") -> "RunList":
        """Where this list's rows sit when `within`'s rows are enumerated 0..n-1.

        Every run of `self` must lie inside one run of `within` (true for
        anything produced by `intersect`).
        """
        out = []
        base = 0
        for w_start, w_n in within.runs:
            w_end = w_start + w_n
            for s, n in self.runs:
                if w_start <= s and s + n <= w_end:
                    out.append((base + s - w_start, n))
                elif s < w_end and s + n > w_start:
                    raise ValueError(f"run {(s, n)} straddles a run of {within.runs}")
            base += w_n
        return RunList(tuple(sorted(out)))


@dataclass(frozen=True)
class UnitSpec:
    name: str
    rows: RunList
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ParamLayout:
    kind: str
    units: tuple[UnitSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown layout kind {self.kind!r}; known: {KINDS}")


def by_rank(shape: torch.Size) -> ParamLayout:
    """The layout a model states for a plain parameter of this shape."""
    if len(shape) == 0 or prod(shape) == 1:
        return ParamLayout("scalar")
    if len(shape) == 1:
        return ParamLayout("vector")
    if len(shape) == 2:
        return ParamLayout("matrix")
    raise ValueError(f"a {len(shape)}-D parameter needs an explicit layout")


def named(module: nn.Module, param: torch.Tensor) -> str:
    """`param`'s name relative to `module`, found by identity rather than
    spelled out, so a declaration cannot drift from the module's attributes."""
    for name, p in module.named_parameters():
        if p is param:
            return name
    raise ValueError("the parameter does not belong to this module")


def stacked_units(names: tuple[str, ...], rows: int, cols: int) -> tuple[UnitSpec, ...]:
    """`[k, rows, cols]` stacked projections (fused w1/w3): unit i is rows [i*rows, (i+1)*rows)."""
    return tuple(
        UnitSpec(n, RunList.contiguous(i * rows, rows), (rows, cols))
        for i, n in enumerate(names)
    )


def qkv_units(
    n_heads: int, n_kv_heads: int, head_dim: int, cols: int
) -> tuple[UnitSpec, ...]:
    """Upstream `QKVLinear` rows: per KV group, `heads_per_kv` q heads, one k head,
    one v head. Three units named wq / wk / wv, each a union of one run per group."""
    hpk = n_heads // n_kv_heads
    block = (hpk + 2) * head_dim
    q = [(g * block, hpk * head_dim) for g in range(n_kv_heads)]
    k = [(g * block + hpk * head_dim, head_dim) for g in range(n_kv_heads)]
    v = [(g * block + (hpk + 1) * head_dim, head_dim) for g in range(n_kv_heads)]
    return (
        UnitSpec("wq", RunList(tuple(q)), (n_heads * head_dim, cols)),
        UnitSpec("wk", RunList(tuple(k)), (n_kv_heads * head_dim, cols)),
        UnitSpec("wv", RunList(tuple(v)), (n_kv_heads * head_dim, cols)),
    )


def decoder_param_layouts(model: nn.Module) -> dict[str, ParamLayout]:
    """What every parameter of a ``Decoder`` is, keyed by FQN: the common modules'
    facts (fused SwiGLU ``w13``, fused ``wqkv``, the MoE router gate, routed
    expert batches), each module's own ``param_layouts()`` where it declares one,
    the two vocabulary matrices (one parameter when tied: the embedding's), and
    the rest by rank."""
    from torchtitan.models.common.attention.attention import QKVLinear
    from torchtitan.models.common.feed_forward import DenseFeedForward
    from torchtitan.models.common.moe import RoutedExperts, TokenChoiceTopKRouter

    layouts: dict[str, ParamLayout] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, DenseFeedForward):
            w13 = module.w13.weight  # [2, F, D]: w1 and w3 stacked
            units = stacked_units(("w1", "w3"), w13.shape[-2], w13.shape[-1])
            layouts[f"{prefix}w13.weight"] = ParamLayout("matrix", units=units)
            continue
        if isinstance(module, TokenChoiceTopKRouter):  # the gate's rows: experts
            layouts[f"{prefix}gate.weight"] = ParamLayout("router")
            continue
        if isinstance(module, QKVLinear):  # rows per KV group: q heads, k, v
            w = module.wqkv.weight
            n_kv_heads = w.shape[0] // (module.r_dim * module.head_dim)
            units = qkv_units(
                module.heads_per_kv * n_kv_heads,
                n_kv_heads,
                module.head_dim,
                w.shape[-1],
            )
            layouts[f"{prefix}wqkv.weight"] = ParamLayout("matrix", units=units)
            continue
        declare = getattr(module, "param_layouts", None)
        if module is model or declare is None:
            continue
        for param_name, layout in declare().items():
            layouts[f"{prefix}{param_name}"] = layout
    for module in model.modules():
        if isinstance(module, RoutedExperts):
            w13 = module.w13.weight
            slots = stacked_units(("w1", "w3"), w13.shape[-2], w13.shape[-1])
            layouts[named(model, w13)] = ParamLayout("expert_batch", units=slots)
            layouts[named(model, module.w2.weight)] = ParamLayout("expert_batch")
    embedding = getattr(model, "tok_embeddings", None)
    lm_head = getattr(model, "lm_head", None)
    if embedding is not None:  # PP stages hold one or neither
        layouts[named(model, embedding.weight)] = ParamLayout("embedding")
    if lm_head is not None and (
        embedding is None or lm_head.weight is not embedding.weight
    ):
        layouts[named(model, lm_head.weight)] = ParamLayout("unembedding")
    for name, p in model.named_parameters():
        if name not in layouts:
            layouts[name] = by_rank(p.shape)
    return layouts


__all__ = [
    "by_rank",
    "decoder_param_layouts",
    "KINDS",
    "named",
    "ParamLayout",
    "qkv_units",
    "RunList",
    "stacked_units",
    "UnitSpec",
]
