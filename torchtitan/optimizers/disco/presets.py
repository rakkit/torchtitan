# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The built-in rules: their phases, computes and presets.

    presets   sgd, muon, right_polar, lm_head, left_polar, aus, hyperball,
              elr, sso
    computes  RowL2, Sign (rows), Orthogonalise, Blocks (unit), GramPolar (gram),
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

import bisect
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..helpers.power_iteration import top_pair_warm
from ..kernels.floors import SUMSQ_FLOOR
from ..kernels.gram_newton_schulz import gram_polar_factor
from ..kernels.zeropower import POLAR_EXPRESS_SAFETY_DEFAULT, zeropower_backends
from . import gram_input_stats
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
    Scale,
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
    """Each row of the update to unit L2 norm (embeddings, LM head, routers).
    `eps` is the division guard of x / (||x|| + eps); a row far below it keeps
    its size / eps."""

    eps: float = EPS
    kind = "rows"

    def core(self, x):
        return _norm(x, -1, self.eps)


@dataclass(frozen=True)
class RowL2Deflate(Compute):
    """RowL2 after removing M's top right singular direction v from every row:
    v by `iters` power iterations from normalise(1^T M). Needs the whole unit."""

    iters: int = 2

    def core(self, x):
        x32 = x.float()
        v = _norm(x32.sum(dim=-2, keepdim=True), -1, EPS)  # [.., 1, n]
        for _ in range(self.iters):
            v = _norm((x32 @ v.mT).mT @ x32, -1, EPS)  # (M v)^T M
        return _norm(x32 - (x32 @ v.mT) @ v, -1, EPS).to(x.dtype)


@dataclass(frozen=True)
class Sign(Compute):
    """The elementwise sign (scalars; signSGD-like rows)."""

    kind = "rows"

    def core(self, x):
        return torch.sign(x)


_TILED_BACKENDS = {
    "gram_polar_express": 8,
    "newtonschulz5_triton": 16,
    "polar_express_triton": 16,
}


# the backends whose polynomial takes the polar express safety factor
POLAR_EXPRESS_BACKENDS = ("gram_polar_express", "polar_express", "polar_express_triton")


@dataclass(frozen=True)
class Orthogonalise(Compute):
    """A zeropower backend on the whole unit (Newton-Schulz, polar express, svd)."""

    backend: str = "newtonschulz5"
    steps: int = 5
    polar_express_safety: float = POLAR_EXPRESS_SAFETY_DEFAULT  # the rule arg

    def __post_init__(self):
        if self.backend not in zeropower_backends or self.backend == "identity":
            raise ValueError(f"unknown backend {self.backend!r}")
        if (
            self.polar_express_safety != POLAR_EXPRESS_SAFETY_DEFAULT
            and self.backend not in POLAR_EXPRESS_BACKENDS
        ):
            raise ValueError(
                f"polar_express_safety is for {POLAR_EXPRESS_BACKENDS}, "
                f"not {self.backend!r}"
            )

    def check_unit(self, m, n):
        # Measured: quack's GEMM needs 8-aligned dims; the Triton kernels hang
        # below 16 (a [1024, 4] conv unit breaks both).
        need = _TILED_BACKENDS.get(self.backend)
        if need and (m % 8 or n % 8 or min(m, n) < need):
            raise ValueError(
                f"{self.backend} runs tiled kernels needing both dims divisible by 8 "
                f"and >= {need}, not [{m}, {n}]: use newtonschulz5, polar_express or "
                "svd, or a row rule (row-l2, sign)"
            )

    def core(self, x):
        if self.polar_express_safety != POLAR_EXPRESS_SAFETY_DEFAULT:
            return zeropower_backends[self.backend](
                x, steps=self.steps, eps=EPS, safety=self.polar_express_safety
            )
        return zeropower_backends[self.backend](x, steps=self.steps, eps=EPS)


@dataclass(frozen=True)
class Blocks(Compute):
    """`inner` on equal blocks of the unit, each times `scale` of its own
    shape: blocks of `rows` rows (the heads of a q, k or v unit) or of `cols`
    columns (the heads feeding `wo`). Head-wise Muon."""

    inner: Compute
    rows: int = 0
    cols: int = 0
    scale: str = "one"

    def __post_init__(self):
        if (self.rows > 0) == (self.cols > 0):
            raise ValueError("set exactly one of block_rows, block_cols")
        if self.inner.kind != "unit" or self.inner.output != "update":
            raise ValueError(
                f"blocks need a whole-unit orthogonaliser, not {self.inner}"
            )
        Scale(self.scale)

    def check_unit(self, m, n):
        size, dim = (self.rows, m) if self.rows else (self.cols, n)
        if dim % size:
            raise ValueError(f"blocks of {size} do not divide the [{m}, {n}] unit")
        self.inner.check_unit(self.rows or m, self.cols or n)

    def core(self, x):
        *b, m, n = x.shape
        if self.rows:
            y = self.inner.core(x.reshape(-1, self.rows, n))
            return (y * SCALES[self.scale](self.rows, n)).reshape(x.shape)
        k = n // self.cols
        blocks = x.reshape(*b, m, k, self.cols).movedim(-2, -3)
        y = self.inner.core(blocks.reshape(-1, m, self.cols))
        y = y.reshape(blocks.shape) * SCALES[self.scale](m, self.cols)
        return y.movedim(-3, -2).reshape(x.shape)


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
class GramPolarNuclear(GramPolar):
    """The Gram polar factor times `nu ** alpha`, `nu = <X, XQ> = tr(R Q)` its
    nuclear norm: PolarGrad's step size (Lau & Su). `nu` comes from the Gram,
    so it costs no reduction beyond the Gram's."""

    alpha: float = 1.0

    def finish(self, rows, R, m):
        q = gram_polar_factor(R, self.steps, EPS, getattr(torch, self.work_dtype))
        nu = (R.float() * q.mT).sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
        return (rows.float() @ q * nu**self.alpha).to(rows.dtype)


_ETA_KEYS = {
    "inverse_poly": {"shape", "start", "c", "p", "hold"},
    "sqrt_decay": {"shape", "start", "total", "hold"},
    "linear": {"shape", "start", "total", "hold"},
    "table": {"shape", "points"},
}


@dataclass(frozen=True)
class EtaSchedule:
    """eta(t), a multiplier on the group's lr for a rule's direction step: the
    step is `lr_t eta(t)`, lr_t the trainer's schedule. 1 up to `start`, then,
    with x = t - start,

        inverse_poly  (1 + c x)^-p                  angular-muown's kappa
        sqrt_decay    1 - sqrt(x / (total - start))  Tencent ELR's msqrt
        linear        1 - x / (total - start)

    With `hold` > 0, eta is held at its value at step `hold` from then on.
    `table`: `points` = ((step, value), ...) with increasing steps, eta linear
    between them and the first / last value outside them (`start` is not used).
    Weight decay, and a rule's other uses of lr (angular-muown's magnitudes),
    keep lr_t alone."""

    shape: str
    start: int = 100
    c: float = 1e-3
    p: float = 1.0
    total: int = 0
    hold: int = 0
    points: tuple = ()

    def __post_init__(self):
        if self.shape not in _ETA_KEYS:
            raise ValueError(f"eta shape {self.shape!r}; known: {sorted(_ETA_KEYS)}")
        if self.shape == "table":
            pts = self.points
            if len(pts) < 2 or any(len(q) != 2 for q in pts):
                raise ValueError("eta table needs at least two (step, value) points")
            if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
                raise ValueError("eta table steps must increase")
            if not all(math.isfinite(q[1]) for q in pts):
                raise ValueError("eta table values must be finite")
            return
        if self.start < 0 or self.c <= 0 or self.p <= 0:
            raise ValueError("eta needs start >= 0, c > 0, p > 0")
        if self.shape in ("sqrt_decay", "linear") and self.total <= self.start:
            raise ValueError(f"eta {self.shape} needs total > start")
        if self.hold < 0 or 0 < self.hold <= self.start:
            raise ValueError("eta hold must be 0 (none) or > start")

    def __call__(self, t: int) -> float:
        t = int(t)
        if self.shape == "table":
            i = bisect.bisect_right(self.points, (t, math.inf))
            if i == 0:
                return self.points[0][1]
            if i == len(self.points):
                return self.points[-1][1]
            (s0, v0), (s1, v1) = self.points[i - 1], self.points[i]
            return v0 if s0 == t else v0 + (v1 - v0) * (t - s0) / (s1 - s0)
        if self.hold and t > self.hold:
            t = self.hold
        x = t - self.start
        if x <= 0:
            return 1.0
        if self.shape == "inverse_poly":
            return (1.0 + self.c * x) ** -self.p
        if self.shape == "linear":
            return 1.0 - min(1.0, x / (self.total - self.start))
        return 1.0 - math.sqrt(min(1.0, x / (self.total - self.start)))


def eta_of(name: str, eta) -> "EtaSchedule | None":
    """A preset's `eta` arg: None (the step is lr), or an `EtaSchedule` dict
    with `shape` and only the keys that shape uses."""
    if eta is None:
        return None
    if not isinstance(eta, dict) or "shape" not in eta:
        raise ValueError(
            f"{name} eta must be None or a schedule dict with a shape, got {eta!r}"
        )
    extra = set(eta) - _ETA_KEYS.get(eta["shape"], set(eta))
    if extra:
        raise ValueError(f"{name} eta {eta['shape']!r} does not use {sorted(extra)}")
    if eta["shape"] == "table":
        eta = {
            **eta,
            "points": tuple((int(s), float(v)) for s, v in eta.get("points", ())),
        }
    return EtaSchedule(**eta)


def eta_at(eta: "EtaSchedule | None", ctx: StepCtx) -> float:
    """The direction step's size: lr, times eta(t) with a schedule."""
    return ctx.lr if eta is None else ctx.lr * eta(ctx.t)


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
    W - eta U never formed. eta is lr, times the `EtaSchedule` `eta` if set."""

    inner: "Rule"
    sigma_iters: int = 8
    eta: EtaSchedule | None = None
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
    """AUS in the Frobenius geometry: the step turns W by about eta (the ELR:
    the group's lr, times the schedule `eta` if set) along the direction
    `inner` produces. With
    r = ||W||_F, U^ = U / ||U||_F, c = <W, U^> / r and d the weight decay:

        keep  W+ = r N(W - s r U^)     Hyperball / MuonH (arXiv 2606.16899)
        free  W+ = d W - s r U^        ELR (hy.tencent.ai/research/elr, Eq. 2)

    N normalises to unit Frobenius norm. s = eta, or with `exact_angle` the
    step whose angle between W and W+ is exactly eta (arXiv 2607.22444):
    s = d tan(eta) / (sqrt(1 - c^2) + c tan(eta)), with d = 1 on the sphere;
    where no step reaches that angle (U^ nearly opposite W), s = eta.
    `tangent` (MuonH-T) first removes the momentum's component along W.
    ||W+|| is closed form, so no pass over W+ is needed. `radius_ratio`
    (keep only): points (step, c) of a radius path, linear between them and
    flat outside; step t scales W+ by c(t) / c(t - 1), so the radius after
    step t is c(t) / c(0) times the starting one and the turn is unchanged."""

    inner: "Rule"
    radius: str = "keep"
    exact_angle: bool = False
    tangent: bool = False
    eta: EtaSchedule | None = None
    radius_ratio: tuple[tuple[int, float], ...] = ()
    needs_weight = True
    output = "update_coeffs"

    def __post_init__(self):
        if self.radius not in ("keep", "free"):
            raise ValueError(f"radius must be 'keep' or 'free', got {self.radius!r}")
        if self.radius_ratio:
            steps = [p[0] for p in self.radius_ratio]
            if self.radius != "keep":
                raise ValueError("radius_ratio sets the radius path of radius='keep'")
            if steps != sorted(set(steps)) or min(p[1] for p in self.radius_ratio) <= 0:
                raise ValueError(
                    "radius_ratio needs increasing steps and positive ratios"
                )

    def ratio(self, t: int) -> float:
        """The radius path c(t), linear between its points, flat outside."""
        pts = self.radius_ratio
        if t <= pts[0][0]:
            return pts[0][1]
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if t <= s1:
                return v0 + (v1 - v0) * (t - s0) / (s1 - s0)
        return pts[-1][1]

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
        if self.radius_ratio:  # rescale W+ = alpha W - step U^ onto the path
            k = self.ratio(ctx.t) / self.ratio(ctx.t - 1)
            alpha, step = alpha * k, step * k
        scale = step / nu.clamp_min(tiny)  # the step along U / ||U||
        return Result(update=(scale[:, None, None] * U).to(x.dtype), alpha=alpha)


@dataclass(frozen=True)
class Scheduled(Compute):
    """`inner`'s plain step times the schedule `eta`:
    W+ = (1 - lr wd) W - lr eta(t) U (1 - wd when decoupled); the decay keeps
    lr alone. Scion's hidden matrices on an angular schedule."""

    inner: "Rule"
    eta: EtaSchedule
    output = "update_coeffs"

    def run(self, x, w, state, ctx):
        U = self.inner.run_whole(x).update
        alpha = torch.full((x.shape[0],), ctx.decay, device=x.device)
        step = eta_at(self.eta, ctx)
        return Result(update=(step * U.float()).to(x.dtype), alpha=alpha)


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
    radius R = radius * SCALES[scale](m, n) (the paper's c; its main runs use 1). With
    M^ = M / ||M||_F and (sigma, u, v)
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
    eta: EtaSchedule | None = None
    radius: float = 1.0
    unit_eta: tuple[tuple[str, float], ...] = ()
    """Per unit type (its log name without `layers.<i>.`), a factor on the
    step: units of one fused parameter can turn by different angles. The
    optimizer applies it (DiSCO._run); every unit of the group needs one."""
    needs_weight = True
    has_state = True
    keeps_radius = True
    output = "update_coeffs"
    eta_linear = True  # the update is linear in eta and alpha is eta-free

    def __post_init__(self):
        pick(SCALES, self.scale, "scale")
        if self.solve_steps < 1 or self.lambda0 <= 0 or self.radius <= 0:
            raise ValueError("sso needs solve_steps >= 1, lambda0 > 0, radius > 0")
        names = [k for k, _ in self.unit_eta]
        if len(set(names)) != len(names) or any(
            not k or not math.isfinite(v) or v <= 0 for k, v in self.unit_eta
        ):
            raise ValueError(
                f"sso unit_eta: distinct unit types with factors > 0, got {self.unit_eta}"
            )

    def state_shape(self, m, n):
        return (n,)

    def solve(self, Mh, u, v):
        """`(Phi, lambda, h)` at the evaluated lambda with the smallest |h|."""
        f = _shifted_msign(self.orth)
        lam = Mh.new_zeros(Mh.shape[0])
        if gram_input_stats.is_armed():
            gram_input_stats.record(self.orth, Mh)
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
            if gram_input_stats.is_armed():
                gram_input_stats.record(
                    self.orth, Mh + probe[:, None, None] * u[:, :, None] * v[:, None, :]
                )
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
        R = self.radius * SCALES[self.scale](m, n)
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


_AM_EPS = 1e-7  # the floor on |g| and on a row norm, as the reference code


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """softplus^-1(y) = y + log(1 - exp(-y)), y > 0."""
    return y + torch.log(-torch.expm1(-y))


_AM_ZERO_ROW = 0.33**0.5  # a zero row's starting magnitude, as the reference
_TINY = torch.finfo(torch.float32).tiny
_INDUCED_POWER_ITERS = 8  # warm-started, as SSO's sigma_iters
_F_LENGTH_RULES = ("polar", "polar_t", "v3b", "cosine", "cosine_raw")  # F's recipe only
_SPECTRAL_POWER_ITERS = 20  # turn_norm spectral: from U's mean row, no state (real units: sigma2/sigma1 ~0.94)


@dataclass(frozen=True)
class AngularMuownMap(Compute):
    """AngularMuown (arXiv 2606.23637, Alg. 1; code fhueb/angular-muown): each
    unit is W = Diag(g) U, unit-norm rows U and signed row magnitudes g kept
    in the state with the moments. From the raw gradient G (the group passes
    it through):

        U = W / g,  gamma = rowdot(G, U),  G_U = Diag(g) (G - Diag(gamma) U)
        M <- beta M + G_U,  O = direction(G_U + beta M)  (M without nesterov)
        U+ = RowNorm(U - eta kappa O),  g+ = Adam(g; gamma, eta),  W+ = Diag(g+) U+

    `direction` is the orthogonaliser with its scale, sqrt(max(1, m / n)).
    eta = lr min(1, t / warmup), the paper's WSD warmup; kappa = (1 + c
    (t - 1 - warmup)_+)^-p turns U only, while g's Adam runs at eta times
    `magnitude_lr` (0: the row norms stay). g starts at W's row norms (a zero
    row at sqrt(0.33)) and keeps its sign with |g| >= 1e-7. State per unit:
    [M | g | Adam's m_g | v_g], shape (m, n + 3). `eta`, an `EtaSchedule`,
    multiplies the turn of U: eta kappa eta(t); the magnitudes keep eta.
    `magnitude_shape` "unit" learns one magnitude per unit instead of one per
    row: g = s g0, g0 the starting row norms, and Adam moves the unit's RMS
    row norm a = s rms(g0) at the same rate, on dL/da = sum(g0 gamma) / rms(g0).
    State then [M | g0 | m_a | v_a | s], shape (m, n + 4). `magnitude_rule`
    "sign" replaces Adam by the l-inf steepest step, the norm of Diag(g) as an
    RMS -> RMS map: g+ = g - eta sign(m_g), m_g the betas[0] average (v_g
    unused). "log_rms" (per-row g only) steps s = log|g| by the RMS-normalised
    average of dL/ds = g gamma, at rho = `magnitude_lr` times the turn:
    mu <- betas[0] mu + (1 - betas[0]) g gamma (g where G was taken),
    g+ = g exp(-rho turn mu / RMS_rows(mu)), turn = eta kappa eta(t); mu in
    m_g's column, v_g unused. It moves log|g| by exactly rho turn in RMS and
    leaves any scale symmetry of the loss within the unit's rows fixed.
    "rel_rms" is the same with the plain relative step g+ = g (1 + s),
    s = -rho turn mu / RMS_rows(mu) (such symmetries move at O(s^2)); it needs
    rho <= 0.25 and rho turn <= 0.5, and 1 + s > 0 is asserted (guaranteed if
    rho turn sqrt(m) < 1). "induced" (row sphere) takes the U-frame RMS -> RMS
    operator-norm step in closed form: x = mu / q, q_i = sum_j (u_i . u_j)^2 |mu_j|,
    s = -turn sqrt(max(1, m/n)) x / ||Diag(x) U||_op, g+ = g exp(s); the norm by a
    power iteration warm-started from v_g's column (no free constant). "polar",
    "polar_t" and "v3b" (theory 6.6; F's row sphere, tangent off) take the radius
    and the turn from one Orth of F's momentum of g G: Z = Orth(D h), h = 1 or W's
    column norms over their RMS (v3b), Dh = Z h ||U||_F / ||Z||_F, a = rowdot(Dh, U),
    g+ = g exp(-turn a), U+ = RowNorm(U - turn (Dh - Diag(a) U)); polar_t and v3b
    then parallel-transport the momentum rows U -> U+ (minimal rotation), polar
    does not (the ambient lag kept). "cosine"
    keeps F's step and moves log g by -turn c, c = DR / |(DR, T)|: DR the Nesterov
    sum of g gamma (m_g's column), T the part of F's Nesterov momentum tangent to
    U; "cosine_raw" takes c = <Mn, U> / |Mn| from an ambient raw-G momentum kept
    after v_g (state (m, 2n + 3)).

    Stage-2 switches (the defaults are the rule above). `sphere` "matrix"
    keeps U on its Frobenius sphere of radius sqrt(m), g starting at the RMS
    row norm, and turns it by FrobeniusAUS's step. `tangent` False skips the
    projection: G_U = Diag(g) G. `magnitude_shape` "none" keeps g at its start
    (state [M | g], shape (m, n + 1)). `native_norm` False rescales O to
    ||O||_F = ||U||_F instead of the direction's scale. `radius` M: U is
    formed from the step-1 weights and g starts at M times their norms.
    `turn_norm` "spectral" (row sphere, native_norm false) rescales O to
    ||O||_2 = sigma_max(U) instead: sigma by 20 power iterations on U and on O,
    both started from U's mean row. `frozen_units`: unit types (DiSCO's log
    names without `layers.<i>.`, e.g. `attention.qkv_linear.wq`) whose per-row
    lengths stay at their start while the other units learn theirs; DiSCO
    passes the stack's mask in the step context. `col_scale`: W = Diag(g) U Diag(c)
    (MuonMD's column gains, arXiv 2606.25971 Eq. 1): c starts at 1 and moves by
    g's rule (adam / sign, betas, eps, eta x `magnitude_lr`) on dL/dc =
    coldot(G, Diag(g) U); U = Diag(1/g) W Diag(1/c), its gradient Diag(g) G
    Diag(c), gamma = rowdot(G Diag(c), U). Three extra state rows [c; m_c; v_c],
    shape (m + 3, n + 3); magnitude_shape row, magnitude_rule adam or sign.
    `gain_param` "softplus": the gains (rows, and columns with col_scale) are
    softplus(r), r = softplus^-1 of the start value; the rule steps r on
    dL/dg sigmoid(r) (the state keeps g; r is recovered from it). `gain_lr`
    "tied": the gains' rate is the turn's, lr_t kappa(t) eta(t) x
    `gain_lr_mult`, instead of lr_t x `magnitude_lr` (left at 1). Both need
    magnitude_shape row and magnitude_rule adam or sign; `betas` are the
    gains' Adam betas. With col_scale, `col_gain_param` sets the column gains'
    form alone (None: gain_param's) and `col_gain_lr_mult` k their rate alone:
    k lr_t, lr_t = the rows' fixed base (lr with the warmup ramp; no kappa, no
    eta(t)) when gain_lr is fixed, k x the turn (lr_t kappa(t) eta(t)) when it is
    tied; None: the row gains' rate. `col_gauge` (with col_scale) keeps the column
    gains on the unit-RMS sphere: after each column step c <- c / RMS_j(c_j) per
    unit; g, its state, U and every momentum are unchanged and W is formed from the
    normalised c, so the matrix's scale is g's (the length rule) alone. `col_units`
    (with col_scale): the unit types (DiSCO's log names without `layers.<i>.`, as
    `frozen_units`) whose columns learn gains c; the other units keep c = 1 (their
    column state does not move). Empty: every unit."""

    direction: "Rule"
    beta: float = 0.95
    nesterov: bool = True
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup: int = 100
    c: float = 1e-3
    p: float = 1.0
    magnitude_lr: float = 1.0
    eta: "EtaSchedule | None" = None
    magnitude_shape: str = "row"
    magnitude_rule: str = "adam"
    sphere: str = "row"
    tangent: bool = True
    native_norm: bool = True
    radius: float = 1.0
    turn_norm: str = "frobenius"
    frozen_units: tuple[str, ...] = ()
    col_scale: bool = False
    gain_param: str = (
        "raw"  # "softplus": g = softplus(r), r stepped (row and column gains)
    )
    gain_lr: str = "fixed"  # "tied": the gains' rate is the turn's x gain_lr_mult
    gain_lr_mult: float = 1.0
    col_gain_param: str | None = None  # None: gain_param
    col_gain_lr_mult: float | None = (
        None  # None: the rows' rate; k: k x lr_t (gain_lr fixed) or k x the turn (tied)
    )
    col_gauge: bool = False  # c <- c / RMS(c) per unit after each column step; g, U and the momenta unchanged
    col_units: tuple[
        str, ...
    ] = ()  # with col_scale: only these unit types learn column gains (empty: all)
    needs_weight = True
    has_state = True
    owns_momentum = True
    output = "weight"

    def __post_init__(self):
        b1, b2 = self.betas
        if not (0.0 <= self.beta < 1.0 and 0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("angular_muown needs beta and betas in [0, 1)")
        if self.gain_param not in ("raw", "softplus") or self.gain_lr not in (
            "fixed",
            "tied",
        ):
            raise ValueError(
                f"angular_muown gain_param raw | softplus, gain_lr fixed | tied; got "
                f"{self.gain_param!r}, {self.gain_lr!r}"
            )
        if self.gain_lr == "fixed" and self.gain_lr_mult != 1.0:
            raise ValueError(
                "angular_muown gain_lr_mult scales the tied gain lr: needs gain_lr tied"
            )
        if self.gain_lr == "tied" and (
            self.magnitude_lr != 1.0 or not self.gain_lr_mult >= 0
        ):
            raise ValueError(
                "angular_muown gain_lr tied: the rate is the turn's x gain_lr_mult (>= 0); "
                "magnitude_lr is unused and stays 1"
            )
        if (self.gain_param, self.gain_lr) != ("raw", "fixed") and (
            self.magnitude_shape != "row" or self.magnitude_rule not in ("adam", "sign")
        ):
            raise ValueError(
                "angular_muown gain_param / gain_lr need magnitude_shape row and magnitude_rule adam or sign"
            )
        cp, ck = self.col_gain_param, self.col_gain_lr_mult
        if (cp is not None or ck is not None) and not self.col_scale:
            raise ValueError(
                "angular_muown col_gain_param / col_gain_lr_mult set the column gains: need col_scale"
            )
        if self.col_gauge and not self.col_scale:
            raise ValueError(
                "angular_muown col_gauge normalises the column gains: need col_scale"
            )
        cu = self.col_units
        if cu:
            if not self.col_scale:
                raise ValueError(
                    "angular_muown col_units picks the units with column gains: need col_scale"
                )
            if len(set(cu)) != len(cu) or not all(isinstance(u, str) and u for u in cu):
                raise ValueError(
                    f"angular_muown col_units: distinct unit types, got {cu!r}"
                )
        if cp not in (None, "raw", "softplus"):
            raise ValueError(f"angular_muown col_gain_param raw | softplus, got {cp!r}")
        if ck is not None and not (math.isfinite(ck) and ck >= 0):
            raise ValueError(
                f"angular_muown col_gain_lr_mult must be finite and >= 0, got {ck!r}"
            )
        if self.warmup < 0 or self.c < 0 or self.p < 0 or self.magnitude_lr < 0:
            raise ValueError("angular_muown needs warmup, c, p, magnitude_lr >= 0")
        if self.magnitude_shape not in ("row", "unit", "none"):
            raise ValueError(
                "angular_muown magnitude_shape is row, unit or none, got "
                f"{self.magnitude_shape!r}"
            )
        rules = ("adam", "sign", "log_rms", "rel_rms", "induced") + _F_LENGTH_RULES
        if self.magnitude_rule not in rules:
            raise ValueError(
                f"angular_muown magnitude_rule is one of {rules}, got {self.magnitude_rule!r}"
            )
        if self.magnitude_rule in _F_LENGTH_RULES:
            r = self.magnitude_rule
            if self.native_norm:
                raise ValueError(f"angular_muown {r} is F's: native_norm false")
            if self.tangent:
                raise ValueError(
                    f"angular_muown {r} is specified on F's recipe: tangent false"
                )
            if self.sphere != "row":
                raise ValueError(
                    f"angular_muown {r} is the step of unit rows: sphere row"
                )
            if self.magnitude_shape != "row":
                raise ValueError(
                    f"angular_muown magnitude_rule {r} steps per-row lengths: use magnitude_shape row"
                )
            if self.magnitude_lr != 1.0:
                raise ValueError(
                    f"angular_muown {r} has no free constant (the turn's step): drop magnitude_lr"
                )
            if (self.eps, tuple(self.betas)) != (1e-8, (0.9, 0.95)):
                raise ValueError(f"angular_muown {r} uses no betas or eps: drop them")
        if self.magnitude_rule in ("log_rms", "rel_rms", "induced"):
            if self.magnitude_shape == "unit":
                raise ValueError(
                    f"angular_muown magnitude_rule {self.magnitude_rule} steps per-row "
                    "lengths: use magnitude_shape row"
                )
            if (self.eps, self.betas[1]) != (1e-8, 0.95):
                raise ValueError(
                    f"angular_muown magnitude_rule {self.magnitude_rule} uses betas[0] only: "
                    "drop eps and betas[1]"
                )
            if self.magnitude_rule == "rel_rms" and self.magnitude_lr > 0.25:
                raise ValueError(
                    "angular_muown rel_rms runs at magnitude_lr (rho) <= 0.25: its (1 + s) "
                    f"drift grows as rho^2, got {self.magnitude_lr}"
                )
            if self.magnitude_rule == "induced" and self.magnitude_lr != 1.0:
                raise ValueError(
                    "angular_muown induced has no free constant (its budget is the turn's): "
                    "drop magnitude_lr"
                )
            if self.magnitude_rule == "induced" and self.sphere != "row":
                raise ValueError(
                    "angular_muown induced is the step of unit rows: sphere row"
                )
        if self.sphere not in ("row", "matrix"):
            raise ValueError(
                f"angular_muown sphere is row or matrix, got {self.sphere!r}"
            )
        if not (math.isfinite(self.radius) and self.radius > 0):
            raise ValueError(f"angular_muown radius must be > 0, got {self.radius!r}")
        if self.magnitude_shape == "none" and (
            self.magnitude_lr,
            tuple(self.betas),
            self.eps,
            self.magnitude_rule,
        ) != (1.0, (0.9, 0.95), 1e-8, "adam"):
            raise ValueError(
                "angular_muown magnitude_shape none learns no magnitudes: drop "
                "magnitude_lr, betas, eps and magnitude_rule"
            )
        if self.turn_norm not in ("frobenius", "spectral"):
            raise ValueError(
                f"angular_muown turn_norm is frobenius or spectral, got {self.turn_norm!r}"
            )
        if self.turn_norm == "spectral" and (
            self.native_norm
            or self.sphere != "row"
            or self.magnitude_rule in ("polar", "polar_t", "v3b")
        ):
            raise ValueError(
                "angular_muown turn_norm spectral rescales F's row-sphere O: needs "
                "native_norm false, sphere row and a rule that turns by O "
                "(not polar / polar_t / v3b)"
            )
        fu = self.frozen_units
        if fu:
            if len(set(fu)) != len(fu) or not all(isinstance(u, str) and u for u in fu):
                raise ValueError(
                    f"angular_muown frozen_units: distinct unit types, got {fu!r}"
                )
            if self.magnitude_shape != "row":
                raise ValueError(
                    "angular_muown frozen_units freezes per-row lengths: magnitude_shape row"
                )
            if self.magnitude_rule not in ("adam", "sign"):
                raise ValueError(
                    "angular_muown frozen_units runs with magnitude_rule adam or sign"
                )
        if self.col_scale and (
            self.magnitude_shape != "row"
            or self.magnitude_rule not in ("adam", "sign")
            or self.frozen_units
        ):
            raise ValueError(
                "angular_muown col_scale learns column gains as the row gains: needs "
                "magnitude_shape row, magnitude_rule adam or sign, no frozen_units"
            )
        if self.native_norm:
            if self.sphere == "matrix":
                raise ValueError(
                    "angular_muown sphere matrix rescales O to ||U||_F: set native_norm false"
                )
            return
        if self.warmup or self.c:
            raise ValueError(
                "angular_muown native_norm false turns U by lr eta(t) only: set warmup 0, c 0"
            )
        if self.direction.scale != "one":
            raise ValueError(
                "angular_muown native_norm false rescales O to ||U||_F, so the "
                f"direction's scale ({self.direction.scale!r}) has no effect: set scale one"
            )
        if (
            self.sphere == "matrix"
            and self.magnitude_shape == "none"
            and not self.tangent
        ):
            raise ValueError(
                "sphere matrix, magnitude_shape none, tangent false is hyperball on the "
                "group's momentum: the angular_muown preset builds it"
            )

    def check_unit(self, m, n):
        # The step is the direction's compute on the unit: its limits are ours.
        self.direction.compute.check_unit(m, n)

    def state_shape(self, m, n):
        if self.magnitude_rule == "cosine_raw":  # + the ambient raw-G momentum
            return (m, 2 * n + 3)
        if self.col_scale:  # + rows [c; m_c; v_c]
            return (m + 3, n + 3)
        return (m, n + {"row": 3, "unit": 4, "none": 1}[self.magnitude_shape])

    def run(self, x, w, state, ctx):
        n, t = x.shape[-1], ctx.t
        dt = torch.promote_types(w.dtype, torch.float32)  # at least fp32
        W, G, S = w.to(dt), x.to(dt), state.to(dt)
        if self.col_scale:  # c = 1 on an empty state
            m = x.shape[-2]
            Sc, S = S[..., m:, :n], S[..., :m, :]
            c = torch.where(Sc[..., 0, :] == 0, 1.0, Sc[..., 0, :])
        M, g = S[..., :n], S[..., n]
        # A row whose g is exactly 0 has no state yet (|g| >= 1e-7 after any
        # step): it starts at its norm, as the reference on an empty state.
        r = (W / c[..., None, :] if self.col_scale else W).norm(dim=-1)
        if self.sphere == "matrix":  # every row at the RMS row norm
            r = r.square().mean(dim=-1, keepdim=True).sqrt().expand_as(r)
        g = torch.where(
            g == 0, torch.where(r > _AM_EPS, r, torch.full_like(r, _AM_ZERO_ROW)), g
        )
        per_unit = self.magnitude_shape == "unit"
        if per_unit:  # the state's column is g0; s is 0 on an empty state
            s = S[..., n + 3]
            s = torch.where(s == 0, torch.ones_like(s), s)
            g0, g = g, s * g
        U = W / torch.copysign(g.abs().clamp_min(_AM_EPS), g)[..., None]
        if self.col_scale:
            U = U / c[..., None, :]
        g_at_G = g  # the lengths G was taken at (log_rms's dL/ds)
        if self.radius != 1.0 and t == 1:
            g = self.radius * g
            if per_unit:
                s = self.radius * s
        if (
            self.col_scale
        ):  # dL/dc at Diag(g) U Diag(c); then G Diag(c) is G for U and g
            gamma_c = (G * (g[..., None] * U)).sum(dim=-2)
            G = G * c[..., None, :]
        gamma = (G * U).sum(dim=-1)
        if not self.tangent:
            G_U = g[..., None] * G
        elif self.sphere == "row":
            G_U = g[..., None] * (G - gamma[..., None] * U)
        else:
            G_U = g[..., None] * G
            uu = U.square().sum(dim=(-2, -1), keepdim=True).clamp_min(_TINY)
            G_U = G_U - (G_U * U).sum(dim=(-2, -1), keepdim=True) / uu * U
        M = self.beta * M + G_U
        D = G_U + self.beta * M if self.nesterov else M
        eta = ctx.lr * min(1.0, t / self.warmup) if self.warmup else ctx.lr
        kappa = (1.0 + self.c * max(0, t - 1 - self.warmup)) ** -self.p
        turn = eta * kappa * (1.0 if self.eta is None else self.eta(t))
        if self.magnitude_rule in (
            "polar",
            "polar_t",
            "v3b",
        ):  # theory 6.6: P0, P1, V3b
            h = 1.0
            if self.magnitude_rule == "v3b":  # W's column norms over their RMS
                cn = W.norm(dim=-2, keepdim=True)
                h = cn / cn.square().mean(dim=-1, keepdim=True).sqrt()
            Z = self.direction.run_whole(D * h).update.to(dt)
            uu = U.square().sum(dim=(-2, -1), keepdim=True)
            zz = Z.square().sum(dim=(-2, -1), keepdim=True)
            Dh = Z * torch.where(zz >= SUMSQ_FLOOR, (uu / zz).sqrt(), 0.0) * h
            a = (Dh * U).sum(dim=-1)
            g_new = g * torch.exp(-turn * a)
            g_new = torch.copysign(g_new.abs().clamp_min(_AM_EPS), g_new)
            msg = f"angular_muown {self.magnitude_rule}: a length is not finite"
            torch._assert_async(torch.isfinite(g_new).all(), msg)
            V = U - turn * (Dh - a[..., None] * U)
            U_new = V / V.norm(dim=-1, keepdim=True).clamp_min(_AM_EPS)
            if (
                self.magnitude_rule != "polar"
            ):  # parallel transport of M's rows, U -> U+
                pv, qv = (M * U).sum(-1, keepdim=True), (M * U_new).sum(
                    -1, keepdim=True
                )
                pq = (U * U_new).sum(-1, keepdim=True)
                M = M - (pv + qv) / (1.0 + pq) * (U + U_new) + 2.0 * pv * U_new
            new = torch.cat([M, g_new[..., None], S[..., n + 1 :]], dim=-1)
            return Result(
                weight=(g_new[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
            )
        if self.sphere == "matrix":  # warmup = c = 0 here: the turn is lr eta(t)
            res = FrobeniusAUS(self.direction, eta=self.eta).run(D, U, None, ctx)
            U_new = res.alpha[:, None, None] * U - res.update
        else:
            O = self.direction.run_whole(D).update.to(dt)
            if self.turn_norm == "spectral":  # ||O||_2 = sigma_max(U)
                v0 = U.mean(dim=-2)
                su = top_pair_warm(U, v0, _SPECTRAL_POWER_ITERS)[0]
                so = top_pair_warm(O, v0, _SPECTRAL_POWER_ITERS)[0]
                k = torch.where(
                    so.square() >= SUMSQ_FLOOR, su / so.clamp_min(_TINY), 0.0
                )
                O = O * k[..., None, None]
            elif not self.native_norm:
                uu = U.square().sum(dim=(-2, -1), keepdim=True)
                oo = O.square().sum(dim=(-2, -1), keepdim=True)
                O = O * torch.where(oo >= SUMSQ_FLOOR, (uu / oo).sqrt(), 0.0)
            V = U - turn * O
            U_new = V / V.norm(dim=-1, keepdim=True).clamp_min(_AM_EPS)
        if self.magnitude_shape == "none":
            new = torch.cat([M, g[..., None]], dim=-1)
            return Result(
                weight=(g[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
            )
        mg, vg = S[..., n + 1], S[..., n + 2]
        b1, b2 = self.betas
        if self.magnitude_rule in ("cosine", "cosine_raw"):  # theory 6.5
            extra = []
            if self.magnitude_rule == "cosine":
                mg = self.beta * mg + g * gamma
                dr = g * gamma + self.beta * mg if self.nesterov else mg
                T = D - (D * U).sum(dim=-1, keepdim=True) * U
                cos = dr / (dr.square() + T.square().sum(dim=-1) + _TINY).sqrt()
            else:  # the ambient lag kept: a raw-G momentum, not transported
                raw = self.beta * S[..., n + 3 :] + G
                Mn = G + self.beta * raw if self.nesterov else raw
                cos = (Mn * U).sum(dim=-1) / (Mn.square().sum(dim=-1) + _TINY).sqrt()
                extra = [raw]
            g_new = g * torch.exp(-turn * cos)
            g_new = torch.copysign(g_new.abs().clamp_min(_AM_EPS), g_new)
            new = torch.cat(
                [M, g_new[..., None], mg[..., None], vg[..., None], *extra], dim=-1
            )
            return Result(
                weight=(g_new[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
            )
        if self.magnitude_rule == "induced":
            mg = b1 * mg + (1.0 - b1) * g_at_G * gamma
            a = mg.abs()
            Ut = U.transpose(-2, -1)
            if U.shape[-2] > n:  # q = rowsum((U M) * U), M = U^T Diag(|mu|) U
                q = ((U @ (Ut @ (a[..., None] * U))) * U).sum(dim=-1)
            else:  # q = (K o K) |mu|, K = U U^T
                K = U @ Ut
                q = ((K * K) @ a[..., None])[..., 0]
            x = mg / torch.where(q > 0, q, torch.inf)
            ax = x.abs()  # ||Diag(x) U||_op^2 = lambda_max(|x| K |x|)
            y = torch.where((vg != 0).any(dim=-1, keepdim=True), vg, ax)
            y = y / y.norm(dim=-1, keepdim=True).clamp_min(_TINY)
            for _ in range(_INDUCED_POWER_ITERS):
                By = ax * (U @ (Ut @ (ax * y)[..., None]))[..., 0]
                lam = By.norm(dim=-1, keepdim=True)
                y = By / lam.clamp_min(_TINY)
            budget = turn * math.sqrt(max(1.0, U.shape[-2] / n))
            g_new = g * torch.exp(
                -budget * x / torch.where(lam > 0, lam.sqrt(), torch.inf)
            )
            new = torch.cat([M, g_new[..., None], mg[..., None], y[..., None]], dim=-1)
            return Result(
                weight=(g_new[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
            )
        if self.magnitude_rule in ("log_rms", "rel_rms"):
            mg = b1 * mg + (1.0 - b1) * g_at_G * gamma
            rms = mg.square().mean(dim=-1, keepdim=True).sqrt()
            ds = mg / torch.where(rms > 0, rms, torch.inf)
            if self.magnitude_rule == "log_rms":
                g_new = g * torch.exp(-(turn * self.magnitude_lr) * ds)
            else:
                step = turn * self.magnitude_lr
                if step > 0.5:
                    raise ValueError(
                        f"angular_muown rel_rms needs magnitude_lr x turn <= 0.5, got {step}"
                    )
                factor = 1.0 - step * ds
                torch._assert_async(
                    (factor > 0).all(),
                    "angular_muown rel_rms: a length factor 1 + s <= 0",
                )
                g_new = g * factor
            new = torch.cat([M, g_new[..., None], mg[..., None], vg[..., None]], dim=-1)
            return Result(
                weight=(g_new[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
            )
        if per_unit:  # one gradient per unit, the same in each of its rows
            a0 = g0.square().mean(dim=-1, keepdim=True).sqrt()
            gamma = ((g0 * gamma).sum(dim=-1, keepdim=True) / a0).expand_as(g0)
        rate = (
            eta * self.magnitude_lr
            if self.gain_lr == "fixed"
            else turn * self.gain_lr_mult
        )
        soft = self.gain_param == "softplus"
        if soft:  # the rule steps r = softplus^-1(g) on dL/dr
            r_g = _softplus_inv(g)
            gamma = gamma * torch.sigmoid(r_g)
        mg = b1 * mg + (1.0 - b1) * gamma
        if self.magnitude_rule == "sign":
            step = torch.sign(mg)
        else:
            vg = b2 * vg + (1.0 - b2) * gamma * gamma
            step = (mg / (1.0 - b1**t)) / ((vg / (1.0 - b2**t)).sqrt() + self.eps)
        if per_unit:
            s_new = s - (eta * self.magnitude_lr) * step / a0
            s_new = torch.copysign(s_new.abs().clamp_min(_AM_EPS), s_new)
            g_new, cols = s_new * g0, [g0, mg, vg, s_new]
        else:
            if soft:
                g_new = F.softplus(r_g - rate * step)
            else:
                g_new = g - rate * step
                g_new = torch.copysign(g_new.abs().clamp_min(_AM_EPS), g_new)
            if self.frozen_units:  # these units keep their start lengths
                f = ctx.frozen
                if f is None or f.shape != g.shape[:1]:
                    raise ValueError(
                        "angular_muown frozen_units needs the stack's mask in the step context"
                    )
                g_new = torch.where(f.to(g.device)[:, None], g, g_new)
            cols = [g_new, mg, vg]
        new = torch.cat([M, *(v[..., None] for v in cols)], dim=-1)
        if not self.col_scale:
            return Result(
                weight=(g_new[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
            )
        mc, vc = Sc[..., 1, :], Sc[..., 2, :]
        if self.col_gain_param is not None:
            soft = self.col_gain_param == "softplus"
        if self.col_gain_lr_mult is not None:
            rate = (turn if self.gain_lr == "tied" else eta) * self.col_gain_lr_mult
        if soft:
            r_c = _softplus_inv(c)
            gamma_c = gamma_c * torch.sigmoid(r_c)
        mc = b1 * mc + (1.0 - b1) * gamma_c
        if self.magnitude_rule == "sign":
            step = torch.sign(mc)
        else:
            vc = b2 * vc + (1.0 - b2) * gamma_c * gamma_c
            step = (mc / (1.0 - b1**t)) / ((vc / (1.0 - b2**t)).sqrt() + self.eps)
        if soft:
            c_new = F.softplus(r_c - rate * step)
        else:
            c_new = c - rate * step
            c_new = torch.copysign(c_new.abs().clamp_min(_AM_EPS), c_new)
        if self.col_gauge:  # c on the unit-RMS sphere; the matrix scale stays with g
            c_new = c_new / c_new.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
                _AM_EPS
            )
        if self.col_units:  # the other units keep c = 1 and their column moments
            cm = ctx.colmask
            if cm is None or cm.shape != c.shape[:1]:
                raise ValueError(
                    "angular_muown col_units needs the stack's mask in the step context"
                )
            cm = cm.to(c.device)[:, None]
            c_new = torch.where(cm, c_new, c)
            mc, vc = torch.where(cm, mc, Sc[..., 1, :]), torch.where(
                cm, vc, Sc[..., 2, :]
            )
        rows = torch.stack([c_new, mc, vc], dim=-2)
        rows = torch.cat([rows, torch.zeros_like(rows[..., :3])], dim=-1)
        return Result(
            weight=(g_new[..., None] * U_new * c_new[..., None, :]).to(w.dtype),
            state=torch.cat([new, rows], dim=-2).to(state.dtype),
        )


# the input-side unit types whose column gains the optimizer study learned
AMPHIONE_COLUMN_UNITS = (
    "attention.qkv_linear.wq",
    "attention.qkv_linear.wk",
    "attention.qkv_linear.wv",
    "feed_forward.w1",
    "feed_forward.w3",
)


@dataclass(frozen=True)
class AmphioneMap(Compute):
    """Amphione: one learning rate for weight amplitude and phase. Each unit
    is W = Diag(g) U, U with unit rows and g its signed row lengths. From the
    raw gradient G, with eta_t = lr eta(t) (`eta` an `EtaSchedule`, None: 1)
    and b = `gain_beta` (None: `beta`):

        gamma = rowdot(G, U),  D = Diag(g) G
        M <- beta M + D,  O = direction(D + beta M)  (M without nesterov)
        O <- O ||U||_F / ||O||_F
        U <- RowNorm(U - eta_t O)
        m_g <- b m_g + (1 - b) gamma,  g <- g - eta_t sign(m_g)

    g starts at the step-1 row norms (a zero row at sqrt(0.33)), U at the
    step-1 rows; |g| >= 1e-7 keeps its sign.

    `columns`: W = Diag(g) U Diag(c) on the `column_units` types (DiSCO
    passes their mask in the step context; other units keep c = 1). c starts
    at 1; before the step gamma_c = coldot(G, Diag(g) U) and G becomes
    G Diag(c), then m_c <- b m_c + (1 - b) gamma_c and
    c <- c - `column_lr_mult` eta_t sign(m_c).

    State per unit [M | g | m_g | r], shape (m, n + 3); with columns three
    more rows [c; m_c; r], shape (m + 3, n + 3). The r column and row are
    unused and carried as they are: the layout is the one the optimizer
    study's runs checkpointed, and DiSCO's column-gain logging reads c from
    the third row from the bottom."""

    direction: Rule
    beta: float = 0.95
    nesterov: bool = True
    gain_beta: float | None = None
    eta: EtaSchedule | None = None
    columns: bool = False
    column_units: tuple[str, ...] = AMPHIONE_COLUMN_UNITS
    column_lr_mult: float = 0.0625
    needs_weight = True
    has_state = True
    owns_momentum = True
    output = "weight"

    def __post_init__(self):
        b = self.beta if self.gain_beta is None else self.gain_beta
        if not (0.0 <= self.beta < 1.0 and 0.0 <= b < 1.0):
            raise ValueError("amphione needs beta and gain_beta in [0, 1)")
        if self.columns and not self.column_units:
            raise ValueError("amphione columns needs column_units")
        if not (math.isfinite(self.column_lr_mult) and self.column_lr_mult >= 0):
            raise ValueError(
                f"amphione column_lr_mult must be finite and >= 0, got {self.column_lr_mult!r}"
            )

    @property
    def col_scale(self) -> bool:  # DiSCO's column-gain logging reads this
        return self.columns

    @property
    def col_units(self) -> tuple[str, ...]:  # DiSCO maps these to ctx.colmask
        return self.column_units if self.columns else ()

    def check_unit(self, m, n):
        self.direction.compute.check_unit(m, n)

    def state_shape(self, m, n):
        return (m + 3, n + 3) if self.columns else (m, n + 3)

    def run(self, x, w, state, ctx):
        n, t = x.shape[-1], ctx.t
        b = self.beta if self.gain_beta is None else self.gain_beta
        dt = torch.promote_types(w.dtype, torch.float32)  # at least fp32
        W, G, S = w.to(dt), x.to(dt), state.to(dt)
        if self.columns:  # c = 1 on an empty state
            m = x.shape[-2]
            Sc, S = S[..., m:, :n], S[..., :m, :]
            c = torch.where(Sc[..., 0, :] == 0, 1.0, Sc[..., 0, :])
        M, g, mg, r = S[..., :n], S[..., n], S[..., n + 1], S[..., n + 2]
        # g == 0 exactly: no state yet (|g| >= 1e-7 after any step)
        norm = (W / c[..., None, :] if self.columns else W).norm(dim=-1)
        start = torch.where(norm > _AM_EPS, norm, torch.full_like(norm, _AM_ZERO_ROW))
        g = torch.where(g == 0, start, g)
        U = W / torch.copysign(g.abs().clamp_min(_AM_EPS), g)[..., None]
        if self.columns:  # dL/dc at Diag(g) U Diag(c); then G Diag(c) is G for U and g
            U = U / c[..., None, :]
            gamma_c = (G * (g[..., None] * U)).sum(dim=-2)
            G = G * c[..., None, :]
        gamma = (G * U).sum(dim=-1)
        D = g[..., None] * G
        M = self.beta * M + D
        O = self.direction.run_whole(D + self.beta * M if self.nesterov else M)
        O = O.update.to(dt)
        eta_t = ctx.lr * (1.0 if self.eta is None else self.eta(t))
        uu = U.square().sum(dim=(-2, -1), keepdim=True)
        oo = O.square().sum(dim=(-2, -1), keepdim=True)
        O = O * torch.where(oo >= SUMSQ_FLOOR, (uu / oo).sqrt(), 0.0)
        V = U - eta_t * O
        U_new = V / V.norm(dim=-1, keepdim=True).clamp_min(_AM_EPS)
        mg = b * mg + (1.0 - b) * gamma
        g_new = g - eta_t * torch.sign(mg)
        g_new = torch.copysign(g_new.abs().clamp_min(_AM_EPS), g_new)
        new = torch.cat([M, g_new[..., None], mg[..., None], r[..., None]], dim=-1)
        if not self.columns:
            return Result(
                weight=(g_new[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
            )
        mask = ctx.colmask
        if mask is None or mask.shape != c.shape[:1]:
            raise ValueError(
                "amphione columns needs the stack's mask in the step context"
            )
        mask = mask.to(c.device)[:, None]
        mc = b * Sc[..., 1, :] + (1.0 - b) * gamma_c
        c_new = c - (eta_t * self.column_lr_mult) * torch.sign(mc)
        c_new = torch.copysign(c_new.abs().clamp_min(_AM_EPS), c_new)
        c_new = torch.where(mask, c_new, c)
        mc = torch.where(mask, mc, Sc[..., 1, :])
        rows = torch.stack([c_new, mc, Sc[..., 2, :]], dim=-2)
        rows = torch.cat([rows, torch.zeros_like(rows[..., :3])], dim=-1)
        return Result(
            weight=(g_new[..., None] * U_new * c_new[..., None, :]).to(w.dtype),
            state=torch.cat([new, rows], dim=-2).to(state.dtype),
        )


@dataclass(frozen=True)
class LieRelativeMap(Compute):
    """V3a (theory-radius-norm.md 6.6, reduced form): relative trust in the RMS -> RMS
    norm, ||dW P^-1||_S = eta, one lr and no shape constant. Per unit (a wide unit,
    m < n, runs on W^T, G^T and keeps M^T):

        M <- beta M + G,  Mn = G + beta M (Nesterov, `nesterov`)
        Q = Orth(W),  P = sym(Q^T W),  C = Orth(Mn P),  S = Q^T C
        W+ = W - eta C P + 0.5 eta^2 C S P            (exp(-eta C Q^T) W to 2nd order)
        M+ = M + eta Q T + 0.5 eta^2 Q S^T T,  T = C^T M   (cotangent transport)

    eta = lr, times the `EtaSchedule` `eta` if set. `direction` is the
    orthogonaliser for both Orth's (scale one). State: M, (m, n) or (n, m)."""

    direction: "Rule"
    beta: float = 0.95
    nesterov: bool = True
    eta: EtaSchedule | None = None
    needs_weight = True
    has_state = True
    owns_momentum = True
    output = "weight"

    def __post_init__(self):
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("lie_relative needs beta in [0, 1)")

    def state_shape(self, m, n):
        return (n, m) if m < n else (m, n)

    def run(self, x, w, state, ctx):
        dt = torch.promote_types(w.dtype, torch.float32)
        W, G, M = w.to(dt), x.to(dt), state.to(dt)
        wide = W.shape[-2] < W.shape[-1]
        if wide:
            W, G = W.mT, G.mT
        M = self.beta * M + G
        Mn = G + self.beta * M if self.nesterov else M
        Q = self.direction.run_whole(W).update.to(dt)
        P = Q.mT @ W
        P = 0.5 * (P + P.mT)
        C = self.direction.run_whole(Mn @ P).update.to(dt)
        S = Q.mT @ C
        e = eta_at(self.eta, ctx)
        W = W - e * (C @ P) + (0.5 * e * e) * (C @ (S @ P))
        T = C.mT @ M
        M = M + e * (Q @ T) + (0.5 * e * e) * (Q @ (S.mT @ T))
        torch._assert_async(
            torch.isfinite(W).all(), "lie_relative: a weight is not finite"
        )
        return Result(weight=(W.mT if wide else W).to(w.dtype), state=M.to(state.dtype))


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
    if name == "row-l2-deflate":
        return RowL2Deflate()
    if name != "gram_polar" and (name not in zeropower_backends or name == "identity"):
        known = sorted(
            {*ROW_COMPUTES, "gram_polar", "row-l2-deflate", *zeropower_backends}
        )
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


@register("muon", own={"eta", "eps"})
def muon(
    pre="identity",
    compute="newtonschulz5",
    steps=5,
    scale="spectral",
    block_rows=0,
    block_cols=0,
    eta=None,
    eps=None,
) -> Rule:
    """Muon / Scion: an orthogonalising compute and the spectral scale.
    `block_rows` / `block_cols` (e.g. head_dim): orthogonalise and scale each
    block of that many rows / columns on its own -- head-wise Muon. `eta`: an
    `EtaSchedule` dict multiplying the step's lr; not inherited, and a row
    compute (an embedding group under such a default) keeps its lr. `eps`: the
    row-l2 compute's division guard (default EPS); not inherited."""
    if eps is not None:
        if compute != "row-l2" or block_rows or block_cols or eta is not None:
            raise ValueError(
                "muon eps is the guard of the row-l2 compute: compute row-l2, no blocks, no eta"
            )
        return Rule(pick(PRE, pre, "pre"), RowL2(float(eps)), scale=scale)
    if eta is not None:
        inner = plain(pre, compute, steps, scale)
        if inner.kind == "rows":
            return inner
        if inner.kind != "unit" or block_rows or block_cols:
            raise ValueError("muon eta needs a whole-unit compute and no blocks")
        direction = Rule((), inner.compute, scale=inner.scale)
        return Rule(inner.pre, Scheduled(direction, eta_of("muon", eta)))
    if not (block_rows or block_cols):
        return plain(pre, compute, steps, scale)
    blocks = Blocks(compute_of(compute, steps), int(block_rows), int(block_cols), scale)
    return Rule(pick(PRE, pre, "pre"), blocks)


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
def lm_head(
    steps=5,
    scale="inv_sqrt_n",
    spectral=True,
    order="row_then_polar",
    alpha=1.0,
    work_dtype="float16",
) -> Rule:
    """Lau & Su's softmax-invariant head rule (arXiv 2605.18106) as their code
    runs it (timlautk/equivariant_optimizers, optim/hybrid.py), every step
    between centrings over the vocabulary rows:
    - `row_then_polar` (their default): rows to unit norm, centre, the right
      polar factor through the Gram times `nu ** alpha`;
    - `polar_then_row`: the polar factor, rows to unit norm (`alpha` cancels);
    - `spectral: false`: rows to unit norm only (their RowNormM).
    The final centring is not optional. The centring is the rule, so `pre` is
    not an argument."""
    if order not in ("row_then_polar", "polar_then_row"):
        raise ValueError(
            f"lm_head order must be row_then_polar or polar_then_row, got {order!r}"
        )
    if not spectral:
        return Rule((Centre(),), RowL2(), (Centre(),), scale)
    if order == "row_then_polar":
        polar = GramPolarNuclear(int(steps), work_dtype, float(alpha))
        return Rule((Centre(), RowNorm(), Centre()), polar, (Centre(),), scale)
    polar = Chain(GramPolar(int(steps), work_dtype), (RowL2(),))
    return Rule((Centre(),), polar, (Centre(),), scale)


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
    """AUS around a Muon direction, turning by lr, times the `EtaSchedule`
    dict `eta` if set. A row compute has no direction to
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


def _row_rule(name: str, inner: Rule, compute: str, row_direction: bool) -> bool:
    """A row compute is the group's whole rule (an embedding group that inherits
    the preset), unless `row_direction` makes it the sphere rule's direction."""
    if row_direction and inner.kind != "rows":
        raise ValueError(
            f"{name} row_direction needs a row compute (row-l2), got {compute!r}"
        )
    return inner.kind == "rows" and not row_direction


def _whole_unit(name: str, pre: str, compute: str, steps: int, scale: str) -> Rule:
    """A sphere preset's direction; a row compute (an embedding group that
    inherits the preset) is returned as that row rule, with its scale."""
    inner = plain(pre, compute, steps, scale)
    if inner.kind == "gram":
        raise ValueError(
            f"{name} needs the unit whole on its owner; {compute!r} is a Gram compute"
        )
    return inner


def _frobenius(
    name,
    radius,
    pre,
    compute,
    steps,
    scale,
    tangent,
    exact_angle,
    eta,
    radius_ratio=(),
    row_direction=False,
):
    inner = _whole_unit(name, pre, compute, steps, scale)
    if _row_rule(name, inner, compute, row_direction):
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
            direction,
            radius,
            bool(exact_angle),
            bool(tangent),
            eta_of(name, eta),
            tuple((int(t), float(c)) for t, c in radius_ratio),
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
    eta=None,
    radius_ratio=(),
    row_direction=False,
) -> Rule:
    """Hyperball (MuonH): each unit stays on its Frobenius sphere and turns by
    the ELR: lr (the paper), times the `EtaSchedule` dict `eta` if set.
    `tangent`: MuonH-T. `exact_angle`: the angle is the ELR exactly. Weight
    decay must be 0. `scale` only applies to a row compute. `radius_ratio`:
    a prescribed radius path instead of a fixed radius (see `FrobeniusAUS`).
    `row_direction`: a row compute (row-l2) is the direction, not the rule."""
    return _frobenius(
        "hyperball",
        "keep",
        pre,
        compute,
        steps,
        scale,
        tangent,
        exact_angle,
        eta,
        radius_ratio,
        row_direction,
    )


@register("elr", own={"scale"})
def elr(
    pre="identity",
    compute="newtonschulz5",
    steps=5,
    scale="one",
    tangent=False,
    exact_angle=False,
    eta=None,
    row_direction=False,
) -> Rule:
    """The ELR rule: a step of `ELR ||W||_F` along the normalised direction,
    weight decay as usual, the radius free. `tangent`, `exact_angle`, `eta`,
    `scale` and `row_direction` as for `hyperball`."""
    return _frobenius(
        "elr",
        "free",
        pre,
        compute,
        steps,
        scale,
        tangent,
        exact_angle,
        eta,
        (),
        row_direction,
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
    eta=None,
    radius=1.0,
    unit_eta=None,
) -> Rule:
    """SSO: steepest descent on the spectral sphere of radius `radius * scale(m, n)`,
    the spectral muP scaler sqrt(d_out / d_in) by default, turning each unit
    by lr, times the `EtaSchedule` dict `eta` if set. Weight decay must be 0.
    `unit_eta`: {unit type: factor} on each unit's step (see SpectralSphere)."""
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
            float(radius),
            tuple(sorted((str(k), float(v)) for k, v in (unit_eta or {}).items())),
        ),
    )


@register("lie_relative", own={"scale"})
def lie_relative(
    compute="gram_polar_express",
    steps=6,
    scale="one",
    beta=0.95,
    nesterov=True,
    eta=None,
) -> Rule:
    """V3a (see `LieRelativeMap`): relative operator-norm trust on the whole weight,
    no row lengths. The rule keeps its own momentum, so the group sets momentum:
    1.0, nesterov: false, weight_decay: 0. A row compute (an embedding group that
    inherits the preset) stays that row rule, with its `scale`."""
    inner = _whole_unit("lie_relative", "identity", compute, steps, scale)
    if inner.kind == "rows":
        return inner
    if scale != "one":
        raise ValueError(
            f"lie_relative's step is relative to W, so a scale ({scale!r}) has no effect: drop it"
        )
    return Rule(
        (),
        LieRelativeMap(
            Rule((), inner.compute),
            float(beta),
            bool(nesterov),
            eta_of("lie_relative", eta),
        ),
    )


@register("angular_muown", own={"scale"})
def angular_muown(
    compute="polar_express",
    steps=5,
    scale="image_spectral",
    beta=0.95,
    nesterov=True,
    betas=(0.9, 0.95),
    eps=1e-8,
    warmup=100,
    c=1e-3,
    p=1.0,
    magnitude_lr=1.0,
    eta=None,
    magnitude_shape="row",
    magnitude_rule="adam",
    sphere="row",
    tangent=True,
    native_norm=True,
    radius=1.0,
    turn_norm="frobenius",
    frozen_units=(),
    col_scale=False,
    gain_param="raw",
    gain_lr="fixed",
    gain_lr_mult=1.0,
    col_gain_param=None,
    col_gain_lr_mult=None,
    col_gauge=False,
    col_units=(),
    row_direction=False,
) -> Rule:
    """AngularMuown (arXiv 2606.23637): each unit W = Diag(g) U turns its
    unit-norm rows by the orthogonalised momentum under the angular schedule
    kappa, and moves its row magnitudes g by Adam, both at the group's lr.
    The rule keeps its own momentum (`beta`, Nesterov with `nesterov`), so the
    group sets momentum: 1.0, nesterov: false, weight_decay: 0. `warmup` is the
    paper's WSD warmup for a trainer schedule without one: it ramps eta and
    holds kappa. `eta`, an `EtaSchedule` dict, multiplies the turn only.
    `magnitude_shape` "unit" learns one magnitude per unit (the
    rows keep their starting ratios) instead of one per row. `magnitude_rule`
    "sign" moves each magnitude by the l-inf step eta sign(m_g); "log_rms" moves
    log|g| by RMS rho = `magnitude_lr` times the turn, "rel_rms" g by the same
    relative step without the exp, "induced" log|g| by the operator-norm step of
    the turn's size, "polar" / "polar_t" / "v3b" both from one Orth and "cosine" /
    "cosine_raw" by a cosine, on F's row sphere (see AngularMuownMap). A
    row compute (an embedding group that inherits the preset) stays that row rule,
    unless `row_direction` makes it the direction (RMNP's row-l2).
    `sphere`, `tangent`, `magnitude_shape` "none", `native_norm`, `radius`,
    `turn_norm`, `frozen_units`, `col_scale`, `gain_param`, `gain_lr`, `gain_lr_mult`,
    `col_gain_param`, `col_gain_lr_mult`, `col_gauge` and `col_units`: see
    `AngularMuownMap`. With native_norm
    false, sphere matrix, magnitude_shape none and tangent false it is
    hyperball with radius_ratio [[0, 1], [1, radius]],
    on the group's momentum (momentum 0.05, nesterov: true)."""
    schedule = eta_of("angular_muown", eta)
    inner = _whole_unit("angular_muown", "identity", compute, steps, scale)
    if _row_rule("angular_muown", inner, compute, row_direction):
        return inner
    if (
        not native_norm
        and sphere == "matrix"
        and magnitude_shape == "none"
        and not tangent
    ):
        unused = dict(
            beta=(beta, 0.95),
            nesterov=(nesterov, True),
            betas=(tuple(betas), (0.9, 0.95)),
            eps=(eps, 1e-8),
            p=(p, 1.0),
            magnitude_lr=(magnitude_lr, 1.0),
            magnitude_rule=(magnitude_rule, "adam"),
            warmup=(warmup, 0),
            c=(c, 0.0),
            scale=(scale, "one"),
            turn_norm=(turn_norm, "frobenius"),
            frozen_units=(tuple(frozen_units), ()),
            col_scale=(bool(col_scale), False),
            gain_param=(gain_param, "raw"),
            gain_lr=(gain_lr, "fixed"),
            gain_lr_mult=(float(gain_lr_mult), 1.0),
            col_gain_param=(col_gain_param, None),
            col_gain_lr_mult=(col_gain_lr_mult, None),
            col_gauge=(bool(col_gauge), False),
            col_units=(tuple(col_units), ()),
        )
        bad = sorted(k for k, (got, want) in unused.items() if got != want)
        if bad:
            raise ValueError(
                f"angular_muown (matrix, none, tangent false) is hyperball: {bad} "
                "must be at their hyperball values (warmup 0, c 0, scale one, the rest default)"
            )
        ratio = [[0, 1.0], [1, float(radius)]] if float(radius) != 1.0 else ()
        return hyperball(
            compute=compute,
            steps=steps,
            eta=eta,
            radius_ratio=ratio,
            row_direction=row_direction,
        )
    b1, b2 = betas
    return Rule(
        (),
        AngularMuownMap(
            Rule((), inner.compute, scale=inner.scale),
            float(beta),
            bool(nesterov),
            (float(b1), float(b2)),
            float(eps),
            int(warmup),
            float(c),
            float(p),
            float(magnitude_lr),
            schedule,
            str(magnitude_shape),
            str(magnitude_rule),
            sphere=str(sphere),
            tangent=bool(tangent),
            native_norm=bool(native_norm),
            radius=float(radius),
            turn_norm=str(turn_norm),
            frozen_units=tuple(frozen_units),
            col_scale=bool(col_scale),
            gain_param=str(gain_param),
            gain_lr=str(gain_lr),
            gain_lr_mult=float(gain_lr_mult),
            col_gain_param=None if col_gain_param is None else str(col_gain_param),
            col_gain_lr_mult=None
            if col_gain_lr_mult is None
            else float(col_gain_lr_mult),
            col_gauge=bool(col_gauge),
            col_units=tuple(col_units),
        ),
    )


@register("onerate", own={"scale"})
def onerate(
    compute="gram_polar_express",
    steps=6,
    scale="one",
    beta=0.95,
    nesterov=True,
    betas=(0.9, 0.95),
    eps=1e-8,
    warmup=0,
    c=0.0,
    p=1.0,
    magnitude_lr=1.0,
    eta=None,
    magnitude_shape="row",
    magnitude_rule="sign",
    sphere="row",
    tangent=False,
    native_norm=False,
    radius=4.0,
    turn_norm="frobenius",
    frozen_units=(),
    col_scale=False,
    gain_param="raw",
    gain_lr="tied",
    gain_lr_mult=1.0,
    col_gain_param=None,
    col_gain_lr_mult=None,
    col_gauge=False,
    col_units=(),
    row_direction=False,
) -> Rule:
    """OneRate, the optimizer study's name for Amphione, kept for ablations:
    `angular_muown` with the study recipe's settings as its defaults (radius
    4, the lengths' sign average betas[0] 0.9) and every switch of it still
    open. `amphione` with radius 4 and gain_beta 0.9 steps as it does; its
    `columns` is col_scale with col_gain_lr_mult 1/16 on col_units
    AMPHIONE_COLUMN_UNITS. The final recipe is `amphione`."""
    return angular_muown(**locals())


@register("amphione")
def amphione(
    eta=None,
    beta=0.95,
    nesterov=True,
    gain_beta=None,
    columns=False,
    column_units=AMPHIONE_COLUMN_UNITS,
    column_lr_mult=0.0625,
    compute="gram_polar_express",
    steps=6,
) -> Rule:
    """Amphione (see `AmphioneMap`): the direction is the orthogonaliser
    `compute` (a whole-unit backend: gram_polar_express, polar_express,
    newtonschulz5, svd, ...) with `steps` iterations; a polar express
    backend's safety factor is the rule arg `polar_express_safety` (default
    1.01). `eta`: an `EtaSchedule` dict on the
    group's lr (None: the trainer's schedule alone). The rule keeps its own
    momentum: the group sets momentum 1.0, nesterov false, weight_decay 0.

    What the optimizer study ran, where it differs from the defaults: Gram
    Polar Express (6 steps) at polar_express_safety 1.05 (its configs spelled
    it `safety`); a constant trainer lr with eta = 1 - sqrt(t / T) (the
    applied step lr eta(t) is the logged eta_used/DiSCO/<group>, not lr/);
    the row lengths' (and columns') sign average gain_beta 0.9, never run at
    beta; g scaled x4 at step 1 by the optimizer (`onerate` radius 4), which
    this rule leaves to the init. Its column runs used column_units
    AMPHIONE_COLUMN_UNITS at column_lr_mult 0.0625."""
    direction = _whole_unit("amphione", "identity", compute, steps, "one")
    if direction.kind == "rows":
        raise ValueError(
            f"amphione orthogonalises the whole unit; {compute!r} is a row compute"
        )
    return Rule(
        (),
        AmphioneMap(
            Rule((), direction.compute, scale=direction.scale),
            beta=float(beta),
            nesterov=bool(nesterov),
            gain_beta=None if gain_beta is None else float(gain_beta),
            eta=eta_of("amphione", eta),
            columns=bool(columns),
            column_units=tuple(column_units),
            column_lr_mult=float(column_lr_mult),
        ),
    )


__all__ = [
    "AMPHIONE_COLUMN_UNITS",
    "amphione",
    "AmphioneMap",
    "angular_muown",
    "AngularMuownMap",
    "lie_relative",
    "LieRelativeMap",
    "aus",
    "AUSMap",
    "Centre",
    "ColNorm",
    "compute_of",
    "EPS",
    "elr",
    "eta_of",
    "EtaSchedule",
    "FrobeniusAUS",
    "GramPolar",
    "hyperball",
    "left_polar",
    "lm_head",
    "muon",
    "onerate",
    "Orthogonalise",
    "plain",
    "PRE",
    "right_polar",
    "ROW_COMPUTES",
    "RowL2",
    "RowL2Deflate",
    "RowNorm",
    "sgd",
    "Sign",
    "SpectralSphere",
    "sso",
    "UnitNorm",
]
