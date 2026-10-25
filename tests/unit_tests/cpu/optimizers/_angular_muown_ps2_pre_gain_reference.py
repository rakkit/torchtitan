# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A verbatim copy of `AngularMuownMap` (presets.py lines 661-1086) in wt-ps2
before the gain switches (gain_param / gain_lr), i.e. wt-p4's class with col_scale.
Frozen: the switches-off bitwise test runs the live class against it."""

import math
from dataclasses import dataclass

import torch

from torchtitan.optimizers.disco.presets import (
    _AM_EPS,
    _AM_ZERO_ROW,
    _F_LENGTH_RULES,
    _INDUCED_POWER_ITERS,
    _SPECTRAL_POWER_ITERS,
    _TINY,
    FrobeniusAUS,
    top_pair_warm,
)
from torchtitan.optimizers.disco.rules import Compute, Result, Rule  # noqa: F401
from torchtitan.optimizers.kernels.floors import SUMSQ_FLOOR


@dataclass(frozen=True)
class PreGainAngularMuownMap(Compute):
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
    shape (m + 3, n + 3); magnitude_shape row, magnitude_rule adam or sign."""

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
    needs_weight = True
    has_state = True
    owns_momentum = True
    output = "weight"

    def __post_init__(self):
        b1, b2 = self.betas
        if not (0.0 <= self.beta < 1.0 and 0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("angular_muown needs beta and betas in [0, 1)")
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
            g_new = g - (eta * self.magnitude_lr) * step
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
        mc = b1 * mc + (1.0 - b1) * gamma_c
        if self.magnitude_rule == "sign":
            step = torch.sign(mc)
        else:
            vc = b2 * vc + (1.0 - b2) * gamma_c * gamma_c
            step = (mc / (1.0 - b1**t)) / ((vc / (1.0 - b2**t)).sqrt() + self.eps)
        c_new = c - (eta * self.magnitude_lr) * step
        c_new = torch.copysign(c_new.abs().clamp_min(_AM_EPS), c_new)
        rows = torch.stack([c_new, mc, vc], dim=-2)
        rows = torch.cat([rows, torch.zeros_like(rows[..., :3])], dim=-1)
        return Result(
            weight=(g_new[..., None] * U_new * c_new[..., None, :]).to(w.dtype),
            state=torch.cat([new, rows], dim=-2).to(state.dtype),
        )
