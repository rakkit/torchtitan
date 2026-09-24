# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The built-in rules: their phases, computes and presets.

    presets   sgd, muon, right_polar, lm_head, left_polar, aus
    computes  RowL2, Sign (rows), Orthogonalise (unit), GramPolar (gram), AUSMap
    phases    RowNorm, ColNorm, UnitNorm, Centre: the `pre` table

A custom optimizer is written the same way, here or in its own module: a
`Compute` (see `rules.py` for what it may declare) and a preset,

    @register("my_rule")
    def my_rule(compute="newtonschulz5", steps=5, scale="spectral") -> Rule:
        return Rule((), MyCompute(...), scale=scale)

selected by `optimizer.groups.<g>.rule: my_rule`. Every normalisation is
guarded by `EPS`, with a floor below which the update is 0
(`kernels/floors.py`); the row maps and Gram computes normalise in fp32, the
Newton-Schulz and polar-express backends the norm of their bf16 operand.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from ..helpers.power_iteration import sigma_max_warm
from ..kernels.floors import SUMSQ_FLOOR
from ..kernels.gram_newton_schulz import gram_polar_factor
from ..kernels.zeropower import zeropower_backends
from .rules import (
    Chain,
    Compute,
    Phase,
    pick,
    register,
    Result,
    RowsIdentity,
    Rule,
    StepCtx,
)

EPS = 1e-30  # the division guard of every fp32 normalisation (not AdamW's eps)


def _norm(x, dim, eps, scale=1.0, sumsq=None):
    """`x / (||x||_dim + eps) * scale`, in fp32; `sumsq` may come from an all-reduce."""
    x32 = x.float()
    if sumsq is None:
        sumsq = x32.pow(2).sum(dim=dim, keepdim=True)
    denom = torch.where(sumsq >= SUMSQ_FLOOR, sumsq.sqrt() + eps, torch.inf)
    return (x32 / denom * scale).to(x.dtype)


# ------------------------------------------------------------------ pre phases
@dataclass(frozen=True)
class RowNorm(Phase):
    """Each row to unit L2 norm."""

    def __call__(self, rows, stat, m):
        return _norm(rows, -1, EPS)


@dataclass(frozen=True)
class ColNorm(Phase):
    """Each column to unit L2 norm over all rows of the unit."""

    reduce = True

    def stat(self, rows):
        return rows.float().pow(2).sum(dim=-2, keepdim=True)

    def __call__(self, rows, stat, m):
        return _norm(rows, -2, EPS, sumsq=stat)


@dataclass(frozen=True)
class UnitNorm(Phase):
    """The whole unit to unit Frobenius norm."""

    reduce = True

    def stat(self, rows):
        return rows.float().pow(2).sum(dim=(-2, -1), keepdim=True)

    def __call__(self, rows, stat, m):
        return _norm(rows, (-2, -1), EPS, sumsq=stat)


@dataclass(frozen=True)
class Centre(Phase):
    """Subtract the mean over the unit's rows. For rows that form an
    interchangeable set (a vocabulary, a router's experts), adding one vector
    to all of them is invisible downstream."""

    reduce = True

    def stat(self, rows):
        return rows.float().sum(dim=-2, keepdim=True)

    def __call__(self, rows, stat, m):
        return (rows.float() - stat / m).to(rows.dtype)


PRE: dict[str, tuple[Phase, ...]] = {  # args.pre -> phases on the moment
    "identity": (),
    "row-l2": (RowNorm(),),
    "col-l2": (ColNorm(),),
    "mat-l2": (UnitNorm(),),
    "col-center": (Centre(),),
}


# ------------------------------------------------------------------ computes
@dataclass(frozen=True)
class RowL2(Compute):
    """Each row of the update to unit L2 norm (embeddings, LM head, routers)."""

    kind = "rows"

    def core(self, x):
        return _norm(x, -1, EPS)


@dataclass(frozen=True)
class Sign(Compute):
    """The elementwise sign (scalars; signSGD-like rows)."""

    kind = "rows"

    def core(self, x):
        return torch.sign(x)


@dataclass(frozen=True)
class Orthogonalise(Compute):
    """A zeropower backend on the whole unit (Newton-Schulz, polar express, svd)."""

    backend: str = "newtonschulz5"
    steps: int = 5

    def __post_init__(self):
        if self.backend not in zeropower_backends or self.backend == "identity":
            raise ValueError(f"unknown backend {self.backend!r}")

    def core(self, x):
        return zeropower_backends[self.backend](x, steps=self.steps, eps=EPS)


@dataclass(frozen=True)
class GramPolar(Compute):
    """Right polar factor `X (X^T X)^{-1/2}` from the Gram alone: the
    polar-express recurrence on `R = X^T X`, then `rows @ Q` on the rows a rank
    holds. A tall row-sharded unit costs one `[n, n]` all-reduce, no all-to-all."""

    steps: int = 5
    work_dtype: str = "float16"  # the Gram iteration squares its operand: fp16 > bf16
    kind = "gram"

    def __post_init__(self):
        if not isinstance(getattr(torch, self.work_dtype, None), torch.dtype):
            raise ValueError(f"unknown work_dtype {self.work_dtype!r}")

    def gram(self, rows):
        rows = rows.float()
        return rows.mT @ rows

    def finish(self, rows, R, m):
        q = gram_polar_factor(R, self.steps, EPS, getattr(torch, self.work_dtype))
        return (rows.float() @ q).to(rows.dtype)


@dataclass(frozen=True)
class EtaSchedule:
    """`max(kappa (t + tau)^-alpha, min)` times a terminal anneal to
    `anneal_floor` over the last `anneal_ratio` of `anneal_steps`."""

    kappa: float = 0.1
    tau: float = 0.0
    alpha: float = 0.5
    min: float = 0.0
    anneal_ratio: float = 0.0
    anneal_steps: int = 0
    anneal_floor: float = 0.0
    anneal_type: str = "linear"
    anneal_power: float = 1.0

    def __post_init__(self):
        if self.anneal_type not in ("linear", "sqrt", "cosine", "poly"):
            raise ValueError(f"unknown anneal_type {self.anneal_type!r}")

    def __call__(self, t: int) -> float:
        t = max(int(t), 1)
        eta = max(self.kappa * (float(t) + self.tau) ** (-self.alpha), self.min)
        if not self.anneal_steps or self.anneal_ratio <= 0.0:
            return eta
        n = max(1, int(self.anneal_steps * self.anneal_ratio))
        start = self.anneal_steps - n
        if t <= start:
            return eta
        x = min(1.0, (t - start) / n)
        adj = {
            "linear": 1.0 - x,
            "sqrt": 1.0 - math.sqrt(x),
            "cosine": 0.5 * (1.0 + math.cos(math.pi * x)),
            "poly": (1.0 - x) ** self.anneal_power,
        }[self.anneal_type]
        return eta * (self.anneal_floor + (1.0 - self.anneal_floor) * adj)


@dataclass(frozen=True)
class AUSMap(Compute):
    """Angular Update Size: the step rotates `W / ||W||_2` by `eta` along the
    direction `inner` produces (an orthogonalise + scale rule); the radius
    follows the plain step. `W_new = alpha W - U'`, `U' = beta U`:

        R = sigma(W), sU = sigma(U), B = W - eta R U / sU, sB = sigma(B),
        R' = sigma(W - eta U):  alpha = R' / sB,  beta = alpha * eta * R / sU

    with W decayed by `1 - wd lr` first and that factor folded into alpha.
    The sigmas are warm-started power iterations (`sigma_iters`); the iterate
    is the per-unit state. `eta` is a schedule of the step, or `"lr"`."""

    inner: "Rule"
    sigma_iters: int = 8
    eta: EtaSchedule | str = field(default_factory=EtaSchedule)
    needs_weight = True
    has_state = True

    def state_shape(self, m, n):
        return (n,)  # the power iteration's right vector

    output = "update_coeffs"

    def eta_at(self, ctx: StepCtx) -> float:
        return ctx.lr if self.eta == "lr" else self.eta(ctx.t)

    def run(self, x, w, state, ctx):
        eta, decay = self.eta_at(ctx), 1.0 - ctx.wd * ctx.lr
        u = self.inner.run_whole(x).update.float()
        W = w.float() * decay
        tiny = torch.finfo(torch.float32).tiny
        R, v = sigma_max_warm(W, state, self.sigma_iters)
        sU, _ = sigma_max_warm(u, v, 1)
        D = u / sU.clamp_min(tiny)[:, None, None]
        B = W - (eta * R)[:, None, None] * D
        sB, _ = sigma_max_warm(B, v, self.sigma_iters)
        R_new, _ = sigma_max_warm(W - eta * u, v, self.sigma_iters)
        alpha = R_new / sB.clamp_min(tiny)
        beta = alpha * eta * R / sU.clamp_min(tiny)
        return Result(
            update=(beta[:, None, None] * u).to(x.dtype),
            alpha=alpha * decay,  # the holders scale the undecayed W
            state=v,
        )


# ------------------------------------------------------------------ presets
ROW_COMPUTES: dict[str, Callable[[], Compute]] = {  # need no whole unit
    "identity": RowsIdentity,
    "row-l2": RowL2,
    "sign": Sign,
}


def _one_compute(name: str, steps: int, work_dtype: str) -> Compute:
    if name in ROW_COMPUTES:
        return ROW_COMPUTES[name]()
    if name != "gram_polar" and (name not in zeropower_backends or name == "identity"):
        known = sorted({*ROW_COMPUTES, "gram_polar", *zeropower_backends})
        raise ValueError(f"unknown compute {name!r}; known: {known}")
    if name != "svd" and int(steps) < 1:
        raise ValueError(f"compute {name!r} iterates: it needs steps >= 1")
    if name == "gram_polar":
        return GramPolar(int(steps), work_dtype)
    return Orthogonalise(name, int(steps))


def compute_of(spec: str, steps: int = 0, work_dtype: str = "float16") -> Compute:
    """`args.compute`: one name, or a chain `main+row_map+...`."""
    main, *after = (_one_compute(n.strip(), steps, work_dtype) for n in spec.split("+"))
    return Chain(main, tuple(after)) if after else main


def plain(pre: str, compute: str, steps: int, scale: str, work_dtype="float16"):
    """`pre` -> `compute` -> `scale`."""
    c = compute_of(compute, steps, work_dtype)
    if isinstance(c, RowsIdentity) and scale in ("spectral", "image_spectral"):
        raise ValueError(f"scale {scale!r} needs a compute that normalises")
    return Rule(pick(PRE, pre, "pre"), c, scale=scale)


@register("sgd")
def sgd(pre="identity", compute="identity", steps=0, scale="one") -> Rule:
    """Momentum SGD, or a row compute (`row-l2`, `sign`): no communication."""
    return plain(pre, compute, steps, scale)


@register("muon")
def muon(pre="identity", compute="newtonschulz5", steps=5, scale="spectral") -> Rule:
    """Muon / Scion: an orthogonalising compute and the spectral scale."""
    return plain(pre, compute, steps, scale)


@register("right_polar", own={"compute"})
def right_polar(
    pre="identity",
    compute="gram_polar",
    steps=5,
    scale="spectral",
    work_dtype="float16",
) -> Rule:
    """Tall units (embeddings, LM head) orthogonalised through the Gram: one
    [n, n] all-reduce, no all-to-all. Never the default group's compute."""
    return plain(pre, compute, steps, scale, work_dtype)


@register("lm_head", own={"scale"})
def lm_head(steps=5, scale="inv_sqrt_n", spectral=True, work_dtype="float16") -> Rule:
    """Lau & Su's softmax-invariant head rule (arXiv 2605.18106): centre over
    the vocabulary rows, the right polar factor through the Gram with rows to
    unit norm (`spectral: false`: the rows alone), centre again -- the second
    projection is not optional -- then scale. The centring is the rule, so
    `pre` is not an argument."""
    rows = RowL2()
    compute = Chain(GramPolar(int(steps), work_dtype), (rows,)) if spectral else rows
    return Rule((Centre(),), compute, (Centre(),), scale)


@register("left_polar")
def left_polar(compute="newtonschulz5", steps=5, scale="spectral") -> Rule:
    """The same centring around an ordinary orthogonalisation, for units whose
    rows are an interchangeable set but which are too wide for the Gram (an MoE
    router `[experts, dim]`)."""
    c = compute_of(compute, steps)
    if c.kind != "unit":
        raise ValueError("left_polar needs an orthogonalising compute")
    return Rule((Centre(),), c, (Centre(),), scale)


@register("aus")
def aus(
    pre="identity",
    compute="newtonschulz5",
    steps=5,
    scale="spectral",
    sigma_iters=8,
    eta=None,
) -> Rule:
    """AUS around a Muon direction. `eta`: an `EtaSchedule` as a dict (None:
    the default schedule), or "lr". A row compute has no direction to
    constrain, so a group that inherits `aus` with one (an embedding group)
    stays that row rule."""
    inner = plain(pre, compute, steps, scale)
    if inner.kind == "rows":
        return inner
    if inner.kind == "gram":
        raise ValueError(
            f"aus needs the unit whole on its owner; {compute!r} is a Gram compute"
        )
    if eta is None or isinstance(eta, dict):
        eta = EtaSchedule(**(eta or {}))
    elif eta != "lr":
        raise ValueError(f"aus eta must be a schedule dict or 'lr', got {eta!r}")
    direction = Rule((), inner.compute, scale=inner.scale)
    return Rule(inner.pre, AUSMap(direction, int(sigma_iters), eta))


__all__ = [
    "aus",
    "AUSMap",
    "Centre",
    "ColNorm",
    "compute_of",
    "EPS",
    "EtaSchedule",
    "GramPolar",
    "left_polar",
    "lm_head",
    "muon",
    "Orthogonalise",
    "plain",
    "PRE",
    "right_polar",
    "ROW_COMPUTES",
    "RowL2",
    "RowNorm",
    "sgd",
    "Sign",
    "UnitNorm",
]
