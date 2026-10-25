# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A verbatim copy of `AngularMuownMap` (presets.py:616-729) at tree c301efe9d07513fb, before the
Stage-2 switches. Frozen: the native-AM bitwise test runs the live class against it."""

from dataclasses import dataclass

import torch

from torchtitan.optimizers.disco.presets import EtaSchedule
from torchtitan.optimizers.disco.rules import Compute, Result, Rule

_AM_EPS = 1e-7  # the floor on |g| and on a row norm, as the reference code
_AM_ZERO_ROW = 0.33**0.5  # a zero row's starting magnitude, as the reference


@dataclass(frozen=True)
class PreChangeAngularMuownMap(Compute):
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
    unused)."""

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
        if self.magnitude_shape not in ("row", "unit"):
            raise ValueError(
                f"angular_muown magnitude_shape is row or unit, got {self.magnitude_shape!r}"
            )
        if self.magnitude_rule not in ("adam", "sign"):
            raise ValueError(
                f"angular_muown magnitude_rule is adam or sign, got {self.magnitude_rule!r}"
            )

    def state_shape(self, m, n):
        return (m, n + 4) if self.magnitude_shape == "unit" else (m, n + 3)

    def run(self, x, w, state, ctx):
        n, t = x.shape[-1], ctx.t
        dt = torch.promote_types(w.dtype, torch.float32)  # at least fp32
        W, G, S = w.to(dt), x.to(dt), state.to(dt)
        M, g, mg, vg = S[..., :n], S[..., n], S[..., n + 1], S[..., n + 2]
        # A row whose g is exactly 0 has no state yet (|g| >= 1e-7 after any
        # step): it starts at its norm, as the reference on an empty state.
        r = W.norm(dim=-1)
        g = torch.where(
            g == 0, torch.where(r > _AM_EPS, r, torch.full_like(r, _AM_ZERO_ROW)), g
        )
        per_unit = self.magnitude_shape == "unit"
        if per_unit:  # the state's column is g0; s is 0 on an empty state
            s = S[..., n + 3]
            s = torch.where(s == 0, torch.ones_like(s), s)
            g0, g = g, s * g
        U = W / torch.copysign(g.abs().clamp_min(_AM_EPS), g)[..., None]
        gamma = (G * U).sum(dim=-1)
        G_U = g[..., None] * (G - gamma[..., None] * U)
        M = self.beta * M + G_U
        D = G_U + self.beta * M if self.nesterov else M
        O = self.direction.run_whole(D).update.to(dt)
        eta = ctx.lr * min(1.0, t / self.warmup) if self.warmup else ctx.lr
        kappa = (1.0 + self.c * max(0, t - 1 - self.warmup)) ** -self.p
        turn = eta * kappa * (1.0 if self.eta is None else self.eta(t))
        V = U - turn * O
        U_new = V / V.norm(dim=-1, keepdim=True).clamp_min(_AM_EPS)
        b1, b2 = self.betas
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
            cols = [g_new, mg, vg]
        new = torch.cat([M, *(c[..., None] for c in cols)], dim=-1)
        return Result(
            weight=(g_new[..., None] * U_new).to(w.dtype), state=new.to(state.dtype)
        )
