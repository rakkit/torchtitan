# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The built-in rules: their phases, computes and presets.

    presets   sgd, muon, right_polar, lm_head, left_polar, aus, hyperball,
              elr, sso
    computes  RowL2, Sign (rows), Orthogonalise (unit), GramPolar (gram),
              AUSMap, FrobeniusAUS, SpectralSphere
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

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from ..helpers.power_iteration import top_pair_warm
from ..kernels.floors import SUMSQ_FLOOR
from ..kernels.gram_newton_schulz import gram_polar_factor
from ..kernels.zeropower import zeropower_backends
from .rules import (
    call_compiled,
    Chain,
    Compute,
    Phase,
    pick,
    register,
    Result,
    RowsIdentity,
    Rule,
    SCALES,
    StepCtx,
)

logger = logging.getLogger(__name__)

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
    `anneal_floor` over the last `anneal_ratio` of `anneal_steps`, times a
    linear warmup over the first `warmup` steps."""

    kappa: float = 0.1
    tau: float = 0.0
    alpha: float = 0.5
    min: float = 0.0
    anneal_ratio: float = 0.0
    anneal_steps: int = 0
    anneal_floor: float = 0.0
    anneal_type: str = "linear"
    anneal_power: float = 1.0
    warmup: int = 0

    def __post_init__(self):
        if self.anneal_type not in ("linear", "sqrt", "cosine", "poly"):
            raise ValueError(f"unknown anneal_type {self.anneal_type!r}")

    def __call__(self, t: int) -> float:
        t = max(int(t), 1)
        eta = max(self.kappa * (float(t) + self.tau) ** (-self.alpha), self.min)
        if self.warmup:
            eta *= min(1.0, t / self.warmup)
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


def eta_of(name: str, eta) -> "EtaSchedule | str":
    """A preset's `eta` arg: "lr" (the group's lr), or an `EtaSchedule` dict."""
    if eta == "lr":
        return eta
    if eta is None or isinstance(eta, dict):
        return EtaSchedule(**(eta or {}))
    raise ValueError(f"{name} eta must be a schedule dict or 'lr', got {eta!r}")


def eta_at(eta, ctx: StepCtx) -> float:
    return ctx.lr if eta == "lr" else eta(ctx.t)


@dataclass(frozen=True)
class AUSMap(Compute):
    """Angular Update Size: the step rotates `W / ||W||_2` by `eta` along the
    direction `inner` produces (an orthogonalise + scale rule); the radius
    follows the plain step. `W_new = alpha W - U'`, `U' = beta U`:

        R = sigma(W), sU = sigma(U), B = W - eta R U / sU, sB = sigma(B),
        R' = sigma(W - eta U):  alpha = R' / sB,  beta = alpha * eta * R / sU

    with W decayed first and that factor folded into alpha. sigma(W) is a
    warm-started power iteration (`sigma_iters`; the iterate is the per-unit
    state); sigma(U), sigma(B) and R' one iteration from its vector, B and
    W - eta U never formed. `eta` is a schedule of the step, or `"lr"`."""

    inner: "Rule"
    sigma_iters: int = 8
    eta: EtaSchedule | str = field(default_factory=EtaSchedule)
    needs_weight = True
    has_state = True

    def state_shape(self, m, n):
        return (n,)  # the power iteration's right vector

    output = "update_coeffs"

    def eta_at(self, ctx: StepCtx) -> float:
        return eta_at(self.eta, ctx)

    def run(self, x, w, state, ctx):
        eta, decay = self.eta_at(ctx), ctx.decay
        u = self.inner.run_whole(x).update.float()
        W = w.float()
        tiny = torch.finfo(torch.float32).tiny
        R0, u1, v = top_pair_warm(W, state, self.sigma_iters)
        Wv, uv = R0[:, None] * u1, _bmv(u, v)
        z = torch.einsum("bmn,bm->bn", u, uv)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(tiny)
        sU = _bmv(u, z).norm(dim=-1)
        R = decay * R0
        sB = _sigma_combo(W, u, decay, eta * R / sU.clamp_min(tiny), Wv, uv)
        R_new = _sigma_combo(W, u, decay, torch.full_like(R, eta), Wv, uv)
        alpha = R_new / sB.clamp_min(tiny)
        beta = alpha * eta * R / sU.clamp_min(tiny)
        return Result(
            update=(beta[:, None, None] * u).to(x.dtype),
            alpha=alpha * decay,  # the holders scale the undecayed W
            state=v,
        )


def _fro(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-unit Frobenius inner product of two `[B, m, n]` stacks, without
    forming their product."""
    return torch.einsum("bmn,bmn->b", a, b)


@dataclass(frozen=True)
class FrobeniusAUS(Compute):
    """AUS in the Frobenius geometry: the step turns W by about `eta` (the ELR:
    the group's lr, or a schedule) along the direction `inner` produces. With
    r = ||W||_F, U^ = U / ||U||_F, c = <W, U^> / r and d the weight decay:

        keep  W+ = r N(W - s r U^)     Hyperball / MuonH (arXiv 2606.16899)
        free  W+ = d W - s r U^        ELR (hy.tencent.ai/research/elr, Eq. 2)

    N normalises to unit Frobenius norm. s = eta, or with `exact_angle` the
    step whose angle between W and W+ is exactly eta (arXiv 2607.22444):
    s = d tan(eta) / (sqrt(1 - c^2) + c tan(eta)), with d = 1 on the sphere;
    where no step reaches that angle (U^ nearly opposite W), s = eta.
    `tangent` (MuonH-T) first removes the momentum's component along W.
    ||W+|| is closed form, so no pass over W+ is needed."""

    inner: "Rule"
    radius: str = "keep"
    exact_angle: bool = False
    tangent: bool = False
    eta: EtaSchedule | str = "lr"
    needs_weight = True
    output = "update_coeffs"

    def __post_init__(self):
        if self.radius not in ("keep", "free"):
            raise ValueError(f"radius must be 'keep' or 'free', got {self.radius!r}")

    @property
    def keeps_radius(self) -> bool:
        return self.radius == "keep"

    def run(self, x, w, state, ctx):
        tiny = torch.finfo(torch.float32).tiny
        W = w.float()
        r2 = _fro(W, W)
        if ctx.t == 1 and bool((r2 < SUMSQ_FLOOR).any()):
            logger.warning(
                f"{int((r2 < SUMSQ_FLOOR).sum())} unit(s) of shape "
                f"{tuple(W.shape[1:])} have zero norm: a step relative to the "
                "radius never moves them"
            )
        M = x
        if self.tangent:
            M = x.float()
            M = M - (_fro(W, M) / r2.clamp_min(tiny))[:, None, None] * W
        U = self.inner.run_whole(M).update.float()
        uu = _fro(U, U)
        r, nu = r2.sqrt(), uu.sqrt()
        c = (_fro(W, U) / (r * nu).clamp_min(tiny)).clamp(-1.0, 1.0)
        d = ctx.decay if self.radius == "free" else 1.0
        eta = eta_at(self.eta, ctx)
        s = torch.full_like(c, eta)
        if self.exact_angle:
            t, p = math.tan(eta), (1.0 - c * c).sqrt()
            den = p + c * t
            s = torch.where((den > 0) & (p > 0), d * t / den.clamp_min(tiny), s)
        s = torch.where(uu >= SUMSQ_FLOOR, s, 0.0)  # no direction: no step
        if self.radius == "keep":
            alpha = torch.rsqrt((1.0 - 2.0 * s * c + s * s).clamp_min(tiny))
            step = alpha * s * r
        else:
            alpha, step = torch.full_like(s, d), s * r
        scale = step / nu.clamp_min(tiny)  # the step along U / ||U||
        return Result(update=(scale[:, None, None] * U).to(x.dtype), alpha=alpha)


_SHIFTED: dict[Compute, Callable] = {}


def _shifted_msign(orth: Compute) -> Callable:
    """`(M, lam, u, v) -> (Phi, h)`: Phi = orth(M + lam u v^T) and the tangent
    residual h = u^T Phi v, per unit, in one compiled graph."""
    fn = _SHIFTED.get(orth)
    if fn is None:
        core = orth.core

        @torch.compile(dynamic=False, fullgraph=True)
        def fn(M, lam, u, v):
            phi = core(M + lam[:, None, None] * u[:, :, None] * v[:, None, :])
            return phi, torch.einsum("bm,bmn,bn->b", u, phi.float(), v)

        _SHIFTED[orth] = fn
    return fn


@dataclass(frozen=True)
class SpectralSphere(Compute):
    """SSO (arXiv 2601.08393): steepest descent on the spectral sphere of
    radius R = SCALES[scale](m, n). With M^ = M / ||M||_F and (sigma, u, v)
    the top singular triple of W (a warm power iteration):

        Phi = orth(M^ + lambda* u v^T)   with   <u v^T, Phi> = 0
        W+  = (R / sigma) W - eta R Phi

    The retraction puts W back on the sphere before the step, and the tangent
    constraint keeps it there to first order, turning W by about `eta` (the
    group's lr, or a schedule) in the spectral norm. h(lambda) = <u v^T, Phi>
    rises from -1 to 1; lambda* is bracketed by steps growing 4x from
    `lambda0` away from 0, then found by Illinois false position (a secant
    that keeps the bracket: bisection's guarantee, superlinear convergence).
    That costs `solve_steps` orth calls after the one at 0, the same count for
    every unit (no host sync); the evaluated lambda with the smallest |h| is
    used. The defaults resolve |lambda*| up to 0.57 to 2e-5 (real hidden
    matrices: <= 0.03). State: v."""

    orth: Compute
    scale: str = "spectral"
    sigma_iters: int = 8
    solve_steps: int = 12
    lambda0: float = 1e-3
    eta: EtaSchedule | str = "lr"
    needs_weight = True
    has_state = True
    keeps_radius = True
    output = "update_coeffs"

    def __post_init__(self):
        pick(SCALES, self.scale, "scale")
        if self.solve_steps < 1 or self.lambda0 <= 0:
            raise ValueError("sso needs solve_steps >= 1 and lambda0 > 0")

    def state_shape(self, m, n):
        return (n,)

    def solve(self, Mh, u, v):
        """`(Phi, lambda, h)` at the evaluated lambda with the smallest |h|."""
        f = _shifted_msign(self.orth)
        lam = Mh.new_zeros(Mh.shape[0])
        phi, h = call_compiled(f, Mh, lam, u, v)
        side = -torch.sign(h)  # h rises with lambda: the root is on this side
        near, far, step = lam, lam, torch.full_like(lam, self.lambda0)
        h_near, h_far = h, h
        found = torch.zeros_like(lam, dtype=torch.bool)
        kept = torch.zeros_like(lam)  # +1: far was replaced last, -1: near
        for _ in range(self.solve_steps):
            dh = h_far - h_near
            sec = (near * h_far - far * h_near) / torch.where(dh != 0, dh, 1.0)
            inside = (sec - near) * (sec - far) < 0
            sec = torch.where(inside, sec, 0.5 * (near + far))
            probe = torch.where(found, sec, near + side * step)
            phi_p, h_p = call_compiled(f, Mh, probe, u, v)
            crossed = side * h_p >= 0
            # Illinois: an end kept twice in a row has its residual halved.
            h_near = torch.where(found & crossed & (kept > 0), 0.5 * h_near, h_near)
            h_far = torch.where(found & ~crossed & (kept < 0), 0.5 * h_far, h_far)
            kept = torch.where(crossed, 1.0, -1.0)
            far, h_far = torch.where(crossed, probe, far), torch.where(
                crossed, h_p, h_far
            )
            near = torch.where(crossed, near, probe)
            h_near = torch.where(crossed, h_near, h_p)
            step = torch.where(found, step, 4.0 * step)
            found = found | crossed
            better = h_p.abs() < h.abs()
            lam, h = torch.where(better, probe, lam), torch.where(better, h_p, h)
            phi = torch.where(better[:, None, None], phi_p, phi)
        return phi, lam, h

    def run(self, x, w, state, ctx):
        tiny = torch.finfo(torch.float32).tiny
        m, n = x.shape[-2:]
        R = SCALES[self.scale](m, n)
        sigma, u, v = top_pair_warm(w.float(), state, self.sigma_iters)
        M = x.float()
        sumsq = _fro(M, M)
        Mh = (
            M
            / torch.where(sumsq >= SUMSQ_FLOOR, sumsq.sqrt(), torch.inf)[:, None, None]
        )
        phi, _, _ = self.solve(Mh, u, v)
        alpha = torch.where(sigma > 0, R / sigma.clamp_min(tiny), 1.0)
        return Result(
            update=(eta_at(self.eta, ctx) * R * phi.float()).to(x.dtype),
            alpha=alpha,
            state=v,
        )


def _bmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bmn,bn->bm", A, x)


def _sigma_combo(W, U, a, b, Wv, Uv) -> torch.Tensor:
    """sigma_max(a W - b U) after one power iteration from the vector v with
    `W v = Wv`, `U v = Uv`; the matrix is never formed. `b` is per unit."""
    b = torch.as_tensor(b, device=W.device).expand(W.shape[0])[:, None]
    y = a * Wv - b * Uv
    z = a * torch.einsum("bmn,bm->bn", W, y) - b * torch.einsum("bmn,bm->bn", U, y)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return (a * _bmv(W, z) - b * _bmv(U, z)).norm(dim=-1)


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
    direction = Rule((), inner.compute, scale=inner.scale)
    return Rule(inner.pre, AUSMap(direction, int(sigma_iters), eta_of("aus", eta)))


def _whole_unit(name: str, pre: str, compute: str, steps: int, scale: str) -> Rule:
    """A sphere preset's direction; a row compute (an embedding group that
    inherits the preset) is returned as that row rule, with its scale."""
    inner = plain(pre, compute, steps, scale)
    if inner.kind == "gram":
        raise ValueError(
            f"{name} needs the unit whole on its owner; {compute!r} is a Gram compute"
        )
    return inner


def _frobenius(name, radius, pre, compute, steps, scale, tangent, exact_angle, eta):
    inner = _whole_unit(name, pre, compute, steps, scale)
    if inner.kind == "rows":
        return inner
    if scale != "one":
        raise ValueError(
            f"{name} normalises the update (the step is lr ||W||), so a scale "
            f"({scale!r}) has no effect on its matrices: drop it"
        )
    direction = Rule((), inner.compute)
    return Rule(
        inner.pre,
        FrobeniusAUS(
            direction, radius, bool(exact_angle), bool(tangent), eta_of(name, eta)
        ),
    )


@register("hyperball", own={"scale"})
def hyperball(
    pre="identity",
    compute="newtonschulz5",
    steps=5,
    scale="one",
    tangent=False,
    exact_angle=False,
    eta="lr",
) -> Rule:
    """Hyperball (MuonH): each unit stays on its Frobenius sphere and turns by
    `eta`, the ELR: "lr" (the paper), or an `EtaSchedule` dict as for `aus`.
    `tangent`: MuonH-T. `exact_angle`: the angle is `eta` exactly. Weight
    decay must be 0. `scale` only applies to a row compute."""
    return _frobenius(
        "hyperball", "keep", pre, compute, steps, scale, tangent, exact_angle, eta
    )


@register("elr", own={"scale"})
def elr(
    pre="identity",
    compute="newtonschulz5",
    steps=5,
    scale="one",
    tangent=False,
    exact_angle=False,
    eta="lr",
) -> Rule:
    """The ELR rule: a step of `eta ||W||_F` along the normalised direction,
    weight decay as usual, the radius free. `tangent`, `exact_angle`, `eta`
    and `scale` as for `hyperball`."""
    return _frobenius(
        "elr", "free", pre, compute, steps, scale, tangent, exact_angle, eta
    )


@register("sso")
def sso(
    pre="identity",
    compute="newtonschulz5",
    steps=5,
    scale="spectral",
    sigma_iters=8,
    solve_steps=12,
    lambda0=1e-3,
    eta="lr",
) -> Rule:
    """SSO: steepest descent on the spectral sphere of radius `scale(m, n)`,
    the spectral muP scaler sqrt(d_out / d_in) by default, turning each unit
    by `eta` ("lr", or an `EtaSchedule` dict). Weight decay must be 0."""
    inner = _whole_unit("sso", pre, compute, steps, scale)
    if inner.kind == "rows":
        return inner
    return Rule(
        inner.pre,
        SpectralSphere(
            inner.compute,
            scale,
            int(sigma_iters),
            int(solve_steps),
            float(lambda0),
            eta_of("sso", eta),
        ),
    )


__all__ = [
    "aus",
    "AUSMap",
    "Centre",
    "ColNorm",
    "compute_of",
    "EPS",
    "elr",
    "EtaSchedule",
    "FrobeniusAUS",
    "GramPolar",
    "hyperball",
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
    "SpectralSphere",
    "sso",
    "UnitNorm",
]
