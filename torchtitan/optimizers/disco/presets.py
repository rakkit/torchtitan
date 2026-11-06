# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The built-in rules: their phases, computes and presets.

    presets   sgd, muon, hyperball, amphione
    computes  RowL2, Sign (rows), Orthogonalise, Blocks (unit), GramPolar (gram),
              FrobeniusAUS (hyperball), AmphioneMap (amphione)
    phases    RowNorm, ColNorm, UnitNorm, Centre: the `pre` table

A custom optimizer is written the same way, here or in its own module: a
`Compute` (see `rules.py` for what it may declare) and a preset,

    @register("my_rule")
    def my_rule(compute="newtonschulz5", steps=5, scale="spectral") -> Rule:
        return Rule((), MyCompute(...), scale=scale)

selected by `optimizer.groups.<g>.rule: my_rule`. Every normalisation is
guarded by `EPS`, with a floor below which the update is 0
(`kernels/floors.py`); the row maps, the Gram computes and gram_polar_express
normalise in fp32, the Newton-Schulz and polar-express backends the norm of their
bf16 operand.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..kernels.floors import SUMSQ_FLOOR
from ..kernels.gram_newton_schulz import (
    gram_polar_factor,
    resolve_gram_polar_express_kernels,
)
from ..kernels.zeropower import POLAR_EXPRESS_SAFETY_DEFAULT, zeropower_backends
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
    """Each row of the update to unit L2 norm (embeddings, LM head):
    x / (||x|| + EPS); a row far below EPS keeps its size / EPS."""

    kind = "rows"

    def core(self, x):
        return _norm(x, -1, EPS)


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
        if self.backend == "gram_polar_express" and torch.cuda.is_available():
            # Import quack's GEMM now, at optimizer build: importing it registers
            # a pytree node type, and doing that at the first optimizer step
            # (after the model's first forward compiled) fails every compiled
            # graph's pytree-registry guard, so FlexAttention recompiled and
            # re-autotuned at step 2.
            resolve_gram_polar_express_kernels()
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


def _fro(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-unit Frobenius inner product of two `[B, m, n]` stacks, without
    forming their product."""
    return torch.einsum("bmn,bmn->b", a, b)


@dataclass(frozen=True)
class FrobeniusAUS(Compute):
    """Hyperball / MuonH (arXiv 2606.16899): each unit stays on its Frobenius
    sphere and turns by about lr (the ELR: the group's lr) along the direction
    `inner` produces. With r = ||W||_F, U^ = U / ||U||_F, c = <W, U^> / r and
    N the normalisation to unit Frobenius norm:

        W+ = r N(W - s r U^)

    s = lr, or with `exact_angle` the step whose angle between W and W+ is
    exactly lr (arXiv 2607.22444): s = tan(lr) / (sqrt(1 - c^2) + c tan(lr));
    where no step reaches that angle (U^ nearly opposite W), s = lr.
    `tangent` (MuonH-T) first removes the momentum's component along W.
    ||W+|| is closed form, so no pass over W+ is needed."""

    inner: "Rule"
    exact_angle: bool = False
    tangent: bool = False
    needs_weight = True
    keeps_radius = True
    output = "update_coeffs"

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
        s = torch.full_like(c, ctx.lr)
        if self.exact_angle:
            t, p = math.tan(ctx.lr), (1.0 - c * c).sqrt()
            den = p + c * t
            s = torch.where((den > 0) & (p > 0), t / den.clamp_min(tiny), s)
        s = torch.where(uu >= SUMSQ_FLOOR, s, 0.0)  # no direction: no step
        alpha = torch.rsqrt((1.0 - 2.0 * s * c + s * s).clamp_min(tiny))
        step = alpha * s * r
        scale = step / nu.clamp_min(tiny)  # the step along U / ||U||
        return Result(update=(scale[:, None, None] * U).to(x.dtype), alpha=alpha)


_AM_EPS = 1e-7  # the floor on |g| and on a row norm, as the reference code
_AM_ZERO_ROW = 0.33**0.5  # a zero row's starting magnitude, as the reference


def _amphione_prepare(W, G, S, beta, nesterov: bool, first: bool):
    """Amphione before the direction (no column scales): g, U, gamma, the
    averaged M and the direction's input. One compiled graph on CUDA per
    `nesterov` / `first` (step 1: the row scales start at the row norms; after
    any step |g| >= _AM_EPS, so later steps skip the norms)."""
    n = W.shape[-1]
    M, g = S[..., :n], S[..., n]
    if first:
        norm = W.norm(dim=-1)
        start = torch.where(norm > _AM_EPS, norm, torch.full_like(norm, _AM_ZERO_ROW))
        g = torch.where(g == 0, start, g)
    U = W / torch.copysign(g.abs().clamp_min(_AM_EPS), g)[..., None]
    gamma = (G * U).sum(dim=-1)
    D = g[..., None] * G
    M = beta * M + (1.0 - beta) * D
    X = beta * M + (1.0 - beta) * D if nesterov else M
    return g, U, gamma, M, X


def _amphione_finish(U, O, g, M, S, gamma, eta, beta, scale_nesterov: bool, frozen):
    """Amphione after the direction (no column scales): the rescale, the turn,
    the row scales' sign step; the new weight and state. One compiled graph."""
    n = U.shape[-1]
    mg = S[..., n + 1]
    uu = U.square().sum(dim=(-2, -1), keepdim=True)
    oo = O.square().sum(dim=(-2, -1), keepdim=True)
    O = O * torch.where(oo >= SUMSQ_FLOOR, (uu / oo).sqrt(), 0.0)
    V = U - eta * O
    U_new = V / V.norm(dim=-1, keepdim=True).clamp_min(_AM_EPS)
    mg = beta * mg + (1.0 - beta) * gamma
    sg = beta * mg + (1.0 - beta) * gamma if scale_nesterov else mg
    g_new = g - eta * torch.sign(sg)
    g_new = torch.copysign(g_new.abs().clamp_min(_AM_EPS), g_new)
    g_new = torch.where(frozen[:, None], g, g_new)
    weight = g_new[..., None] * U_new
    return weight, torch.cat([M, g_new[..., None], mg[..., None]], dim=-1)


_AMPHIONE_COMPILED: dict = {}


def _amphione_compiled(name):
    fn = _AMPHIONE_COMPILED.get(name)
    if fn is None:
        fn = _AMPHIONE_COMPILED[name] = torch.compile(
            {"prepare": _amphione_prepare, "finish": _amphione_finish}[name],
            fullgraph=True,
            dynamic=False,
        )
    return fn


# the input-side unit types whose column scales the optimizer study learned
AMPHIONE_COLUMN_SCALE_UNITS = (
    "attention.qkv_linear.wq",
    "attention.qkv_linear.wk",
    "attention.qkv_linear.wv",
    "feed_forward.w1",
    "feed_forward.w3",
)


@dataclass(frozen=True)
class AmphioneMap(Compute):
    """Amphione: one learning rate for weight amplitude and phase. Each unit
    is W = Diag(g) U, U with unit rows and g its signed row scales. From the
    raw gradient G (DiSCO averages nothing for this rule), with eta_t = lr (the
    trainer's schedule) and beta = the group's `momentum` (its `nesterov` on
    the direction):

        gamma = rowdot(G, U),  D = Diag(g) G
        M <- beta M + (1 - beta) D
        O = direction(beta M + (1 - beta) D)  (direction(M) without nesterov)
        O <- O ||U||_F / ||O||_F
        U <- RowNorm(U - eta_t O)
        m_g <- beta m_g + (1 - beta) gamma,  g <- g - eta_t sign(m_g)

    `scale_nesterov`: the scales' sign steps take
    sign(beta m + (1 - beta) gamma) instead (off in the production recipe).
    g starts at the step-1 row norms (a zero row at sqrt(0.33)), U at the
    step-1 rows; |g| >= 1e-7 keeps its sign.

    `column_scales`: W = Diag(g) U Diag(c) on the `column_scale_units` types (DiSCO
    passes their mask in the step context; other units keep c = 1). c, the
    column scales, starts at 1; before the step gamma_c = coldot(G, Diag(g) U)
    and G becomes G Diag(c), then m_c <- beta m_c + (1 - beta) gamma_c and
    c <- c - `column_scale_lr_mult` eta_t sign(m_c).

    `frozen_scale_units`: unit types whose row lengths g stay at their start
    (DiSCO passes their mask in the step context); their rows U still turn and
    m_g still averages. Not with `column_scales` (never run).

    State per unit [M | g | m_g], shape (m, n + 2); with column scales two
    more rows [c; m_c], shape (m + 2, n + 2) (`column_scales_of` reads c)."""

    direction: Rule
    scale_nesterov: bool = False
    column_scales: bool = False
    column_scale_units: tuple[str, ...] = AMPHIONE_COLUMN_SCALE_UNITS
    column_scale_lr_mult: float = 0.0625
    frozen_scale_units: tuple[str, ...] = ()
    needs_weight = True
    has_state = True
    owns_momentum = True
    output = "weight"

    def __post_init__(self):
        if self.column_scales and not self.column_scale_units:
            raise ValueError("amphione column_scales needs column_scale_units")
        fu = self.frozen_scale_units
        if len(set(fu)) != len(fu) or not all(isinstance(u, str) and u for u in fu):
            raise ValueError(
                f"amphione frozen_scale_units: distinct unit types, got {fu!r}"
            )
        if fu and self.column_scales:
            raise ValueError(
                "amphione frozen_scale_units with column_scales: not supported (never run)"
            )
        if not (
            math.isfinite(self.column_scale_lr_mult) and self.column_scale_lr_mult >= 0
        ):
            raise ValueError(
                f"amphione column_scale_lr_mult must be finite and >= 0, got {self.column_scale_lr_mult!r}"
            )

    @property
    def has_column_scales(self) -> bool:  # DiSCO's column-scale logging reads this
        return self.column_scales

    @property
    def col_units(self) -> tuple[str, ...]:  # DiSCO maps these to ctx.colmask
        return self.column_scale_units if self.column_scales else ()

    @property
    def frozen_units(self) -> tuple[str, ...]:  # DiSCO maps these to ctx.frozen
        return self.frozen_scale_units

    def check_unit(self, m, n):
        self.direction.compute.check_unit(m, n)

    def state_shape(self, m, n):
        return (m + 2, n + 2) if self.column_scales else (m, n + 2)

    def _run_compiled(self, x, w, state, ctx):
        """`run` without column scales, its element-wise maths and reductions
        compiled around the direction (the same equations; reductions in
        another order, so equal to `run` to rounding)."""
        if self.frozen_scale_units:
            f = ctx.frozen
            if f is None or f.shape != x.shape[:1]:
                raise ValueError(
                    "amphione frozen_scale_units needs the stack's mask in the step context"
                )
            frozen = f.to(x.device)
        else:
            frozen = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        g, U, gamma, M, X = call_compiled(
            _amphione_compiled("prepare"),
            w.float(),  # fp32 compute, as the eager path (bf16 transport: G is bf16)
            x.float(),
            state.float(),
            ctx.momentum,
            ctx.nesterov,
            ctx.t == 1,
        )
        O = self.direction.run_whole(X).update.float()
        eta = torch.full((), ctx.lr, device=x.device)  # a tensor: no recompile per lr
        weight, new = call_compiled(
            _amphione_compiled("finish"),
            U,
            O,
            g,
            M,
            state,
            gamma,
            eta,
            ctx.momentum,
            self.scale_nesterov,
            frozen,
        )
        return Result(weight=weight.to(w.dtype), state=new.to(state.dtype))

    @staticmethod
    def column_scales_of(state: torch.Tensor) -> torch.Tensor:
        """c of a column-scale unit's state, [..., n] (zeros before its first step)."""
        return state[..., -2, :-2]

    def run(self, x, w, state, ctx):
        if not self.column_scales and x.is_cuda:
            return self._run_compiled(x, w, state, ctx)
        return self._run_eager(x, w, state, ctx)

    def _run_eager(self, x, w, state, ctx):
        n = x.shape[-1]
        beta = ctx.momentum
        dt = torch.promote_types(w.dtype, torch.float32)  # at least fp32
        W, G, S = w.to(dt), x.to(dt), state.to(dt)
        if self.column_scales:  # c = 1 on an empty state
            m = x.shape[-2]
            Sc, S = S[..., m:, :n], S[..., :m, :]
            c = torch.where(Sc[..., 0, :] == 0, 1.0, Sc[..., 0, :])
        M, g, mg = S[..., :n], S[..., n], S[..., n + 1]
        if ctx.t == 1:  # g == 0: no state yet (|g| >= _AM_EPS after any step)
            norm = (W / c[..., None, :] if self.column_scales else W).norm(dim=-1)
            start = torch.where(
                norm > _AM_EPS, norm, torch.full_like(norm, _AM_ZERO_ROW)
            )
            g = torch.where(g == 0, start, g)
        U = W / torch.copysign(g.abs().clamp_min(_AM_EPS), g)[..., None]
        if (
            self.column_scales
        ):  # dL/dc at Diag(g) U Diag(c); then G Diag(c) is G for U and g
            U = U / c[..., None, :]
            gamma_c = (G * (g[..., None] * U)).sum(dim=-2)
            G = G * c[..., None, :]
        gamma = (G * U).sum(dim=-1)
        D = g[..., None] * G
        M = beta * M + (1.0 - beta) * D
        O = self.direction.run_whole(beta * M + (1.0 - beta) * D if ctx.nesterov else M)
        O = O.update.to(dt)
        eta_t = ctx.lr
        uu = U.square().sum(dim=(-2, -1), keepdim=True)
        oo = O.square().sum(dim=(-2, -1), keepdim=True)
        O = O * torch.where(oo >= SUMSQ_FLOOR, (uu / oo).sqrt(), 0.0)
        V = U - eta_t * O
        U_new = V / V.norm(dim=-1, keepdim=True).clamp_min(_AM_EPS)
        mg = beta * mg + (1.0 - beta) * gamma
        sg = beta * mg + (1.0 - beta) * gamma if self.scale_nesterov else mg
        g_new = g - eta_t * torch.sign(sg)
        g_new = torch.copysign(g_new.abs().clamp_min(_AM_EPS), g_new)
        if self.frozen_scale_units:  # these units keep their start lengths
            f = ctx.frozen
            if f is None or f.shape != g.shape[:1]:
                raise ValueError(
                    "amphione frozen_scale_units needs the stack's mask in the step context"
                )
            g_new = torch.where(f.to(g.device)[:, None], g, g_new)
        new = torch.cat([M, g_new[..., None], mg[..., None]], dim=-1)
        if not self.column_scales:
            return Result(
                weight=(g_new[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
            )
        mask = ctx.colmask
        if mask is None or mask.shape != c.shape[:1]:
            raise ValueError(
                "amphione column_scales needs the stack's mask in the step context"
            )
        mask = mask.to(c.device)[:, None]
        mc = beta * Sc[..., 1, :] + (1.0 - beta) * gamma_c
        sc = beta * mc + (1.0 - beta) * gamma_c if self.scale_nesterov else mc
        c_new = c - (eta_t * self.column_scale_lr_mult) * torch.sign(sc)
        c_new = torch.copysign(c_new.abs().clamp_min(_AM_EPS), c_new)
        c_new = torch.where(mask, c_new, c)
        mc = torch.where(mask, mc, Sc[..., 1, :])
        rows = torch.stack([c_new, mc], dim=-2)
        rows = torch.cat([rows, torch.zeros_like(rows[..., :2])], dim=-1)
        return Result(
            weight=(g_new[..., None] * U_new * c_new[..., None, :]).to(w.dtype),
            state=torch.cat([new, rows], dim=-2).to(state.dtype),
        )


# ------------------------------------------------------------------ presets
ROW_COMPUTES: dict[str, Callable[[], Compute]] = {  # need no whole unit
    "identity": RowsIdentity,
    "row-l2": RowL2,
    "sign": Sign,
}


def _one_compute(name: str, steps: int) -> Compute:
    if name in ROW_COMPUTES:
        return ROW_COMPUTES[name]()
    if name != "gram_polar" and (name not in zeropower_backends or name == "identity"):
        known = sorted({*ROW_COMPUTES, "gram_polar", *zeropower_backends})
        raise ValueError(f"unknown compute {name!r}; known: {known}")
    if name != "svd" and int(steps) < 1:
        raise ValueError(f"compute {name!r} iterates: it needs steps >= 1")
    if name == "gram_polar":
        return GramPolar(int(steps))
    return Orthogonalise(name, int(steps))


def compute_of(spec: str, steps: int = 0) -> Compute:
    """`args.compute`: one name, or a chain `main+row_map+...`."""
    main, *after = (_one_compute(n.strip(), steps) for n in spec.split("+"))
    return Chain(main, tuple(after)) if after else main


def plain(pre: str, compute: str, steps: int, scale: str):
    """`pre` -> `compute` -> `scale`."""
    c = compute_of(compute, steps)
    if isinstance(c, RowsIdentity) and scale in ("spectral", "image_spectral"):
        raise ValueError(f"scale {scale!r} needs a compute that normalises")
    return Rule(pick(PRE, pre, "pre"), c, scale=scale)


@register("sgd")
def sgd(pre="identity", compute="identity", steps=0, scale="one") -> Rule:
    """Momentum SGD, or a row compute (`row-l2`, `sign`): no communication."""
    return plain(pre, compute, steps, scale)


@register("muon")
def muon(
    pre="identity",
    compute="newtonschulz5",
    steps=5,
    scale="spectral",
    block_rows=0,
    block_cols=0,
) -> Rule:
    """Muon / Scion: an orthogonalising compute and the spectral scale.
    `block_rows` / `block_cols` (e.g. head_dim): orthogonalise and scale each
    block of that many rows / columns on its own -- head-wise Muon."""
    if not (block_rows or block_cols):
        return plain(pre, compute, steps, scale)
    blocks = Blocks(compute_of(compute, steps), int(block_rows), int(block_cols), scale)
    return Rule(pick(PRE, pre, "pre"), blocks)


def _whole_unit(name: str, pre: str, compute: str, steps: int, scale: str) -> Rule:
    """The direction of a rule that needs the unit whole on its owner
    (hyperball, amphione): an orthogonaliser or a row compute; a Gram compute
    cannot be one."""
    inner = plain(pre, compute, steps, scale)
    if inner.kind == "gram":
        raise ValueError(
            f"{name} needs the unit whole on its owner; {compute!r} is a Gram compute"
        )
    return inner


@register("hyperball")
def hyperball(
    pre="identity",
    compute="newtonschulz5",
    steps=5,
    tangent=False,
    exact_angle=False,
) -> Rule:
    """Hyperball (MuonH, see `FrobeniusAUS`): each unit stays on its Frobenius
    sphere and turns by lr. `compute` picks the direction: an orthogonaliser,
    or a row compute (`row-l2`: each row of the unit normalised on its own).
    `tangent`: MuonH-T. `exact_angle`: the angle is lr exactly. Weight decay
    must be 0 (the rule holds each unit's norm)."""
    inner = _whole_unit("hyperball", pre, compute, steps, "one")
    return Rule(
        inner.pre,
        FrobeniusAUS(
            Rule((), inner.compute),
            exact_angle=bool(exact_angle),
            tangent=bool(tangent),
        ),
    )


@register("amphione", own={"frozen_scale_units"})
def amphione(
    scale_nesterov=False,
    column_scales=False,
    column_scale_units=AMPHIONE_COLUMN_SCALE_UNITS,
    column_scale_lr_mult=0.0625,
    frozen_scale_units=(),
    compute="gram_polar_express",
    steps=6,
) -> Rule:
    """Amphione (see `AmphioneMap`). `compute` picks the direction O:

    - an orthogonaliser (gram_polar_express, polar_express, newtonschulz5,
      svd, ...) with `steps` iterations: O = Orth(beta M + (1 - beta) D), the
      hidden matrices' rule. A polar express backend's safety factor is the rule arg
      `polar_express_safety` (default 1.01).
    - a row compute (`row-l2`): O = rowl2(beta M + (1 - beta) D), each row of the unit
      normalised on its own (a [C, K] conv: every channel's filter as its own
      1 x K matrix, whose spectral normalisation is its l2 norm).

    `frozen_scale_units`: unit types of the group whose row scales stay at
    their start (their rows still turn), e.g. `["feed_forward.router.gate"]`. Each type
    must name a unit of the group; a group never inherits it.
    `scale_nesterov`: Nesterov on the row (and column) scales' sign steps.

    The step size is the group's lr from the trainer's scheduler; the momentum
    is the group's `momentum` (beta) and `nesterov` (on the direction), kept in
    the rule's state. The group sets weight_decay 0.

    What the optimizer study ran, where it differs from the defaults: Gram
    Polar Express (6 steps) at polar_express_safety 1.05; a constant trainer
    lr with an eta table 1 - sqrt(t / T), where this rule takes the trainer's
    sqrt schedule; the row (and column) scales averaged at 0.9, the direction
    at 0.95;
    g scaled x4 at step 1 by the optimizer, which this rule leaves to the
    init. Its column runs used column_scale_units AMPHIONE_COLUMN_SCALE_UNITS at
    column_scale_lr_mult 0.0625. Its MoE runs froze the router gate's lengths
    (`feed_forward.router.gate`)."""
    inner = _whole_unit("amphione", "identity", compute, steps, "one")
    return Rule(
        (),
        AmphioneMap(
            Rule((), inner.compute),
            scale_nesterov=bool(scale_nesterov),
            column_scales=bool(column_scales),
            column_scale_units=tuple(column_scale_units),
            column_scale_lr_mult=float(column_scale_lr_mult),
            frozen_scale_units=tuple(frozen_scale_units),
        ),
    )


__all__ = [
    "AMPHIONE_COLUMN_SCALE_UNITS",
    "amphione",
    "AmphioneMap",
    "Blocks",
    "Centre",
    "ColNorm",
    "compute_of",
    "EPS",
    "FrobeniusAUS",
    "GramPolar",
    "hyperball",
    "muon",
    "Orthogonalise",
    "plain",
    "PRE",
    "ROW_COMPUTES",
    "RowL2",
    "RowNorm",
    "sgd",
    "Sign",
    "UnitNorm",
]
