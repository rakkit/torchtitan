# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The joint radius / turn rules of theory-radius-norm.md 6.6 on F's row sphere (tangent off), in
float64: F's momentum of g G (Nesterov), Z = Orth(D h) (h = 1; v3b: W's column norms over their
RMS), Dh = (||U||_F / ||Z||_F) Z h, a = rowdot(Dh, U), g <- g exp(-turn a),
U <- RowNorm(U - turn (Dh - Diag(a) U)). "polar_t" (P1) and "v3b" then parallel-transport the
momentum rows U -> U+ by the minimal rotation p -> q,
R v = v - ((<p,v> + <q,v>) / (1 + <p,q>)) (p + q) + 2 <p,v> q; "polar" (P0) is P1 without it. "cosine"
(6.5) keeps F's direction step and moves log g by -turn c, c = DR / sqrt(DR^2 + ||T||^2), DR the
Nesterov sum of g gamma, T the part of F's Nesterov momentum tangent to U. The other magnitude
rules are bitwise the tree before these (6a2de99b)."""

import copy
import itertools
import math
import os
import socket
import sys
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import AngularMuownMap, EtaSchedule
from torchtitan.optimizers.disco.rules import Compute, Rule, rule_of, StepCtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _angular_muown_induced_reference import InducedAngularMuownMap  # noqa: E402
from _angular_muown_polar_reference import PolarAngularMuownMap  # noqa: E402

LR, M_START = 0.05, 4.0
F_ETA = EtaSchedule("inverse_poly", start=1, c=0.5)


@dataclass(frozen=True)
class _Polar(Compute):
    """k U V^T in float64."""

    k: float = 1.0

    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return self.k * (u @ vh)


def _exact_polar(X, k=1.0):
    u, _, vh = torch.linalg.svd(X, full_matrices=False)
    return k * (u @ vh)


def _data(m, n, seed, steps=6):
    gen = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(
        torch.randn(max(m, n), max(m, n), generator=gen, dtype=torch.float64)
    )[0][:m, :n]
    W = W * torch.logspace(-0.5, 0.5, m, dtype=torch.float64)[:, None]
    return W, [
        0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]


def _max_diff(xs, ys):
    return max(float((x - y).abs().max()) for x, y in zip(xs, ys))


def _transport(v, p, q):
    """The minimal rotation p -> q of each row of v (p, q unit rows)."""
    pv, qv, pq = ((a * b).sum(dim=1, keepdim=True) for a, b in ((p, v), (q, v), (p, q)))
    return v - (pv + qv) / (1 + pq) * (p + q) + 2 * pv * q


def _reference_p1(
    W,
    grads,
    *,
    M=M_START,
    lr=LR,
    beta=0.95,
    nesterov=True,
    k=0.5,
    eta=F_ETA,
    cols=False,
    transport=True,
    mutate=None,
):
    """Section 6.6: P1 (cols False, h = 1), V3b (h_j = ||W_:,j|| / RMS_j ||W_:,j||, W the weights the
    step starts from) and P0 (P1 without the transport)."""
    m, n = W.shape
    start = W.norm(dim=1, keepdim=True)
    U, g, mom, w, out = W / start, M * start, torch.zeros_like(W), W, []
    for t, G in enumerate(grads, start=1):
        GU = g * G
        mom = beta * mom + GU
        D = GU + beta * mom if nesterov else mom
        c = w.norm(dim=0)
        h = c / c.square().mean().sqrt() if cols else torch.ones(n, dtype=W.dtype)
        Z = _exact_polar(D * (1 if mutate == "h_outside_only" else h), k)
        s = U.norm() / (Z * h if mutate == "rescale_with_Zh" else Z).norm()
        Dh = s * Z * (1 if mutate == "h_inside_only" else h)
        a = (Dh * U).sum(dim=1, keepdim=True)
        turn = lr * eta(t)
        g = g * torch.exp(-turn * a)
        V = U - turn * (Dh - a * U)
        U_next = V / V.norm(dim=1, keepdim=True)
        if transport and mutate != "no_transport":
            mom = _transport(mom, U, U_next)
        U = U_next
        w = g * U
        out.append(w)
    return out


def _polar(k=0.5, rule="polar", **kw):
    return AngularMuownMap(
        Rule((), _Polar(k), scale="one"),
        warmup=0,
        c=0.0,
        eta=F_ETA,
        magnitude_rule=rule,
        sphere="row",
        tangent=False,
        native_norm=False,
        radius=M_START,
        **kw,
    )


def _run(cm, W, grads, lr=LR):
    m, n = W.shape
    w, state, ws, states = W[None], W.new_zeros(1, *cm.state_shape(m, n)), [], []
    for t, G in enumerate(grads, start=1):
        res = cm.run(G[None], w, state, StepCtx(lr, 0.0, t))
        w, state = res.weight, res.state
        ws.append(w[0])
        states.append(state[0])
    return ws, states


# ---------------------------------------------------------------- the definition
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize("nesterov", [True, False])
def test_polar_is_its_definition_step_after_step(m, n, nesterov):
    W, grads = _data(m, n, seed=m + 3 * n)
    ref = _reference_p1(W, grads, nesterov=nesterov, transport=False)
    got, _ = _run(_polar(nesterov=nesterov), W, grads)
    assert _max_diff(ref, got) < 1e-10


def test_p0_is_p1_without_the_transport():
    """From the same weights and state, P0 and P1 return bitwise the same weights; P1's stored
    momentum is P0's rotated U -> U+. Along a run they part once the rows have turned."""
    W, grads = _data(12, 8, seed=22)
    p0, p1 = _polar(rule="polar"), _polar(rule="polar_t")
    w, state = W[None], W.new_zeros(1, *p1.state_shape(12, 8))
    for t, G in enumerate(grads, start=1):
        a = p0.run(G[None], w, state, StepCtx(LR, 0.0, t))
        b = p1.run(G[None], w, state, StepCtx(LR, 0.0, t))
        assert torch.equal(a.weight, b.weight) and torch.equal(
            a.state[..., 8:], b.state[..., 8:]
        ), t
        U, Un = w[0] / w[0].norm(dim=1, keepdim=True), b.weight[0] / b.weight[0].norm(
            dim=1, keepdim=True
        )
        assert (
            float(
                (_transport(a.state[0, :, :8], U, Un) - b.state[0, :, :8]).abs().max()
            )
            < 1e-12
        ), t
        assert float((a.state[0, :, :8] - b.state[0, :, :8]).abs().max()) > 1e-8, t
        w, state = b.weight, b.state
    assert _max_diff(_run(p0, W, grads)[0][1:], _run(p1, W, grads)[0][1:]) > 1e-6


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("rule", ["polar_t", "v3b"])
def test_p1_and_v3b_are_their_definition_step_after_step(m, n, nesterov, rule):
    W, grads = _data(m, n, seed=m + 7 * n)
    W = W * torch.linspace(0.5, 2.0, n, dtype=torch.float64)  # unequal column norms
    ref = _reference_p1(W, grads, nesterov=nesterov, cols=rule == "v3b")
    got, _ = _run(_polar(nesterov=nesterov, rule=rule), W, grads)
    assert _max_diff(ref, got) < 1e-10


@pytest.mark.parametrize(
    "mutate", ["no_transport", "h_outside_only", "h_inside_only", "rescale_with_Zh"]
)
def test_the_p1_v3b_comparison_sees_each_plausible_mistake(mutate):
    """Wide: for tall units with an exact polar, ||Z h||_F = ||Z||_F (unit columns, RMS(h) = 1)."""
    W, grads = _data(8, 12, seed=3)
    W = W * torch.linspace(0.5, 2.0, 12, dtype=torch.float64)
    cols = mutate != "no_transport"
    assert (
        _max_diff(
            _reference_p1(W, grads, cols=cols),
            _reference_p1(W, grads, cols=cols, mutate=mutate),
        )
        > 1e-4
    )


def test_v3b_with_equal_column_norms_is_p1_bitwise():
    """+-1 entries: every column norm is sqrt(m) exactly, so h = 1 exactly and v3b is P1."""
    gen = torch.Generator().manual_seed(19)
    W = torch.randint(0, 2, (12, 8), generator=gen).to(torch.float64) * 2 - 1
    grads = [
        0.1 * torch.randn(12, 8, generator=gen, dtype=torch.float64) for _ in range(1)
    ]
    a, sa = _run(_polar(rule="polar_t"), W, grads)
    b, sb = _run(_polar(rule="v3b"), W, grads)
    assert torch.equal(a[0], b[0]) and torch.equal(sa[0], sb[0])


def test_a_uniform_rescale_of_W_leaves_the_v3b_step_unchanged():  # noqa: N802
    W, grads = _data(12, 8, seed=20)
    W = W * torch.linspace(0.5, 2.0, 8, dtype=torch.float64)
    a, _ = _run(_polar(rule="v3b"), W, grads)
    b, _ = _run(_polar(rule="v3b"), 3.0 * W, grads)
    assert _max_diff([3.0 * x for x in a], b) < 1e-12


def _steps_with_frames(cm, W, grads_of, steps=6, lr=LR):
    """Runs the rule; grads_of(t, U) gives G from the pre-step unit rows. Per step: (U before, G,
    U after, the stored momentum)."""
    m, n = W.shape
    w, state, out = W[None], W.new_zeros(1, *cm.state_shape(m, n)), []
    for t in range(1, steps + 1):
        U = w[0] / w[0].norm(dim=1, keepdim=True)
        G = grads_of(t, U)
        res = cm.run(G[None], w, state, StepCtx(lr, 0.0, t))
        w, state = res.weight, res.state
        out.append((U, G, w[0] / w[0].norm(dim=1, keepdim=True), state[0, :, :n]))
    return out


@pytest.mark.parametrize("rule", ["polar_t", "v3b"])
def test_transport_rotates_the_old_row_onto_the_new_one(rule):
    """From an empty state with G = U, the momentum is g U (F's momentum of g G); transported it
    is g U+ (R p = q), with its norm kept."""
    W, _ = _data(12, 8, seed=4, steps=0)
    ((U, G, U_next, M),) = _steps_with_frames(
        _polar(rule=rule), W, lambda t, U: U.clone(), steps=1
    )
    g = M_START * W.norm(
        dim=1, keepdim=True
    )  # the step-1 lengths after the start-radius jump
    assert float((M - g * U_next).abs().max()) < 1e-12
    assert float((M.norm(dim=1, keepdim=True) - g).abs().max()) < 1e-12


@pytest.mark.parametrize("rule", ["polar_t", "v3b"])
def test_the_transported_radial_part_is_the_ema_of_the_radial_gradients(rule):
    """<M, U> after transport equals r_t = beta r_(t-1) + g_t <G_t, U_t> exactly (the rotation keeps
    <v, p> as <Rv, q>); the row norms of M are those of the untransported sum."""
    W, _ = _data(12, 8, seed=5, steps=0)
    gen = torch.Generator().manual_seed(6)
    rand = lambda t, U: 0.1 * torch.randn(
        12, 8, generator=gen, dtype=torch.float64
    )  # noqa: E731
    r = torch.zeros(12, dtype=torch.float64)
    cm = _polar(rule=rule)
    w, state = W[None], W.new_zeros(1, *cm.state_shape(12, 8))
    for t in range(1, 7):
        U = w[0] / w[0].norm(dim=1, keepdim=True)
        g = M_START * W.norm(dim=1) if t == 1 else state[0, :, 8]
        G = rand(t, U)
        untransported = 0.95 * state[0, :, :8] + g[:, None] * G
        res = cm.run(G[None], w, state, StepCtx(LR, 0.0, t))
        w, state = res.weight, res.state
        M, U_next = state[0, :, :8], w[0] / w[0].norm(dim=1, keepdim=True)
        r = 0.95 * r + g * (G * U).sum(dim=1)
        assert float(((M * U_next).sum(dim=1) - r).abs().max()) < 1e-12, t
        assert float((M.norm(dim=1) - untransported.norm(dim=1)).abs().max()) < 1e-12, t


@pytest.mark.parametrize("rule", ["polar_t", "v3b"])
def test_a_tangent_gradient_leaves_no_radial_momentum(rule):
    W, _ = _data(12, 8, seed=7, steps=0)
    gen = torch.Generator().manual_seed(8)

    def tangent(t, U):
        G = 0.1 * torch.randn(12, 8, generator=gen, dtype=torch.float64)
        return G - (G * U).sum(dim=1, keepdim=True) * U

    for _, _, U_next, M in _steps_with_frames(_polar(rule=rule), W, tangent):
        assert float((M * U_next).sum(dim=1).abs().max()) < 1e-12
    # the ambient momentum (polar) keeps a lagging radial part under the same gradients
    gen = torch.Generator().manual_seed(8)
    lag = [
        float((M * Un).sum(dim=1).abs().max())
        for _, _, Un, M in _steps_with_frames(_polar(), W, tangent)
    ]
    assert lag[-1] > 1e-4


def test_each_wide_row_moves_exactly_the_step():
    """Wide unit, exact polar: every row of Dh has norm 1, so a_i^2 + |Dt_i|^2 = 1 and the row's
    joint move is exactly the step (doc 6.1 #6): s_i^2 + tan(theta_i)^2 = step^2, with
    s_i = log(g+ / g)_i and theta_i the row's turn."""
    m, n = 8, 12
    W, grads = _data(m, n, seed=2)
    cm = _polar(k=1.0)
    w, state = W[None], W.new_zeros(1, *cm.state_shape(m, n))
    for t, G in enumerate(grads, start=1):
        g = state[0, :, n] if t > 1 else M_START * W.norm(dim=1)
        U = w[0] / w[0].norm(dim=1, keepdim=True)
        res = cm.run(G[None], w, state, StepCtx(LR, 0.0, t))
        w, state = res.weight, res.state
        s = (state[0, :, n] / g).log()
        Un = w[0] / w[0].norm(dim=1, keepdim=True)
        c = (Un * U).sum(dim=1)
        tan = (Un - c[:, None] * U).norm(dim=1) / c
        step = LR * F_ETA(t)
        assert float((s.square() + tan.square() - step**2).abs().max()) < 1e-12 * max(
            1.0, step**2
        ), t


def test_each_unit_of_a_stack_is_its_own():
    (Wa, ga), (Wb, gb) = _data(12, 8, seed=5), _data(12, 8, seed=6)
    cm = _polar()
    w, state = torch.stack([Wa, 3.0 * Wb]), Wa.new_zeros(2, *cm.state_shape(12, 8))
    for t, (A, B) in enumerate(zip(ga, gb), start=1):
        res = cm.run(torch.stack([A, B]), w, state, StepCtx(LR, 0.0, t))
        w, state = res.weight, res.state
    for k, (W, grads) in enumerate(((Wa, ga), (3.0 * Wb, gb))):
        assert float((w[k] - _run(cm, W, grads)[0][-1]).abs().max()) < 1e-12


# ---------------------------------------------------------------- cosine (6.5) and cosine_raw
def _reference_cosine(
    W,
    grads,
    *,
    M=M_START,
    lr=LR,
    beta=0.95,
    nesterov=True,
    k=0.5,
    eta=F_ETA,
    raw=False,
    mutate=None,
):
    """F's direction step (momentum of g G, Nesterov, O = k polar rescaled to ||U||_F, RowNorm) with
    log g moved by -turn c. cosine: DR = Nesterov sum of g gamma, T = D - <D, U> U,
    c = DR / sqrt(DR^2 + ||T||^2). cosine_raw: Mn = Nesterov sum of the raw G, c = <Mn, U> / ||Mn||."""
    m, n = W.shape
    start = W.norm(dim=1, keepdim=True)
    U, g = W / start, M * start
    mom, R, raw_mom, out = (
        torch.zeros_like(W),
        torch.zeros_like(start),
        torch.zeros_like(W),
        [],
    )
    for t, G in enumerate(grads, start=1):
        gamma = (G * U).sum(dim=1, keepdim=True)
        GU = g * G
        mom = beta * mom + GU
        D = GU + beta * mom if nesterov else mom
        turn = lr * eta(t)
        if raw:
            raw_mom = beta * raw_mom + G
            Mn = G + beta * raw_mom if nesterov else raw_mom
            c = (Mn * U).sum(dim=1, keepdim=True) / Mn.norm(dim=1, keepdim=True)
        else:
            src = gamma if mutate == "raw_gamma" else g * gamma
            R = beta * R + src
            DR = src + beta * R if nesterov and mutate != "no_nesterov_R" else R
            if mutate == "ambient":
                c = (D * U).sum(dim=1, keepdim=True) / D.norm(dim=1, keepdim=True)
            else:
                T = D - (D * U).sum(dim=1, keepdim=True) * U
                c = DR / (DR.square() + T.square().sum(dim=1, keepdim=True)).sqrt()
        O = _exact_polar(D, k)
        V = U - turn * O * U.norm() / O.norm()
        U = V / V.norm(dim=1, keepdim=True)
        s_mn = math.sqrt(max(1.0, m / n)) if mutate == "shape_factor" else 1.0
        g = g * torch.exp(-turn * s_mn * c)
        out.append(g * U)
    return out


def _cosine(k=0.5, rule="cosine", **kw):
    return AngularMuownMap(
        Rule((), _Polar(k), scale="one"),
        warmup=0,
        c=0.0,
        eta=F_ETA,
        magnitude_rule=rule,
        sphere="row",
        tangent=False,
        native_norm=False,
        radius=M_START,
        **kw,
    )


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("rule", ["cosine", "cosine_raw"])
def test_cosine_is_its_definition_step_after_step(m, n, nesterov, rule):
    W, grads = _data(m, n, seed=m + 5 * n)
    ref = _reference_cosine(W, grads, nesterov=nesterov, raw=rule == "cosine_raw")
    got, _ = _run(_cosine(rule=rule, nesterov=nesterov), W, grads)
    assert _max_diff(ref, got) < 1e-10


@pytest.mark.parametrize(
    "mutate", ["raw_gamma", "no_nesterov_R", "ambient", "shape_factor"]
)
def test_the_cosine_comparison_sees_each_plausible_mistake(mutate):
    W, grads = _data(12, 8, seed=13)
    assert (
        _max_diff(
            _reference_cosine(W, grads), _reference_cosine(W, grads, mutate=mutate)
        )
        > 1e-4
    )


@pytest.mark.parametrize("rule", ["cosine", "cosine_raw"])
def test_a_tangent_gradient_moves_cosine_raw_only(rule):
    """G perpendicular to U at every step while U turns: cosine's R stays 0 and g does not move
    (doc 6.5 test 1); cosine_raw's ambient momentum lags and moves g."""
    W, _ = _data(12, 8, seed=14, steps=0)
    gen = torch.Generator().manual_seed(15)
    cm = _cosine(rule=rule)
    w, state, logs = W[None], W.new_zeros(1, *cm.state_shape(12, 8)), []
    for t in range(1, 7):
        U = w[0] / w[0].norm(dim=1, keepdim=True)
        G = 0.1 * torch.randn(12, 8, generator=gen, dtype=torch.float64)
        G = G - (G * U).sum(dim=1, keepdim=True) * U
        g = state[0, :, 8] if t > 1 else M_START * W.norm(dim=1)
        res = cm.run(G[None], w, state, StepCtx(LR, 0.0, t))
        w, state = res.weight, res.state
        logs.append(float((state[0, :, 8] / g).log().abs().max()))
        if rule == "cosine":
            assert float(state[0, :, 9].abs().max()) < 1e-12, t  # R
    assert max(logs) < 1e-12 if rule == "cosine" else logs[-1] > 1e-4


def test_a_radial_gradient_gives_cosine_one_and_a_full_turn():
    """G = lambda_i U_i: T = 0, so c = sign(g gamma) and |log g+ - log g| = turn (doc 6.5 test 2)."""
    W, _ = _data(12, 8, seed=16, steps=0)
    lam = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64) + 0.05
    G = lam[:, None] * W / W.norm(dim=1, keepdim=True)
    res = _cosine().run(G[None], W[None], W.new_zeros(1, 12, 11), StepCtx(LR, 0.0, 1))
    s = (res.state[0, :, 8] / (M_START * W.norm(dim=1))).log()
    assert float((s + LR * F_ETA(1) * torch.sign(lam)).abs().max()) < 1e-12


@pytest.mark.parametrize("rule", ["cosine", "cosine_raw"])
def test_the_cosine_rules_turn_U_exactly_as_F(rule):  # noqa: N802
    """Step 1 from the same state: the unit rows equal F's (adam) to the last bits of the final
    multiply; only the lengths differ (doc 6.5 test 3)."""
    W, grads = _data(12, 8, seed=17, steps=1)
    a = _cosine(rule=rule).run(
        grads[0][None],
        W[None],
        W.new_zeros(1, *_cosine(rule=rule).state_shape(12, 8)),
        StepCtx(LR, 0.0, 1),
    )
    f = AngularMuownMap(
        Rule((), _Polar(0.5), scale="one"),
        warmup=0,
        c=0.0,
        eta=F_ETA,
        magnitude_lr=0.3,
        sphere="row",
        tangent=False,
        native_norm=False,
        radius=M_START,
    ).run(grads[0][None], W[None], W.new_zeros(1, 12, 11), StepCtx(LR, 0.0, 1))
    ua = a.weight[0] / a.state[0, :, 8:9]
    uf = f.weight[0] / f.state[0, :, 8:9]
    assert float((ua - uf).abs().max()) < 4e-16
    assert float((a.state[0, :, 8] - f.state[0, :, 8]).abs().max()) > 1e-6


def test_cosine_raw_keeps_the_raw_momentum_in_extra_state():
    cm = _cosine(rule="cosine_raw")
    assert cm.state_shape(12, 8) == (12, 2 * 8 + 3)
    W, grads = _data(12, 8, seed=18, steps=2)
    _, states = _run(cm, W, grads)
    raw = 0.95 * grads[0] + grads[1]
    assert float((states[-1][:, 11:] - raw).abs().max()) < 1e-14


# ---------------------------------------------------------------- the other rules unchanged
_F = {
    "scale": "one",
    "warmup": 0,
    "c": 0.0,
    "native_norm": False,
    "radius": 4.0,
    "eta": {"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0},
}
_CASES = [
    *(
        (dict(warmup=2, c=0.5), r)
        for r in ("adam", "sign", "log_rms", "rel_rms", "induced")
    ),
    (dict(warmup=2, c=0.5, nesterov=False, magnitude_shape="unit"), "adam"),
    (
        dict(warmup=2, c=0.5, eta={"shape": "sqrt_decay", "start": 2, "total": 9}),
        "sign",
    ),
    *(
        ({**_F, "sphere": s, "magnitude_shape": g, "tangent": t}, r)
        for s, g, t in itertools.product(
            ("row", "matrix"), ("unit", "row"), (False, True)
        )
        for r in ("adam", "sign")
    ),
    *(
        ({**_F, "sphere": s, "tangent": t}, r)
        for s, t in itertools.product(("row", "matrix"), (False, True))
        for r in ("log_rms", "rel_rms")
    ),
    *(({**_F, "sphere": "row", "tangent": t}, "induced") for t in (False, True)),
    ({**_F, "sphere": "row", "magnitude_shape": "none", "tangent": False}, "adam"),
]


@pytest.mark.parametrize("args,rule", _CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute", ["svd", "newtonschulz5"])
def test_the_other_rules_are_the_6a2de99b_rule_bitwise(args, rule, dtype, compute):
    """Stacks of 3 units with a zero row, a lost state at step 4, 8 steps."""
    extra = {}
    if args.get("magnitude_shape") != "none":
        extra["magnitude_rule"] = rule
        lr_mag = {"adam": 0.3, "sign": 0.3, "log_rms": 0.3, "rel_rms": 0.2}.get(rule)
        if lr_mag is not None:
            extra["magnitude_lr"] = lr_mag
    live = rule_of(
        {"rule": "angular_muown", "args": {"compute": compute, **args, **extra}}
    ).compute
    c = live
    old = InducedAngularMuownMap(
        c.direction,
        c.beta,
        c.nesterov,
        c.betas,
        c.eps,
        c.warmup,
        c.c,
        c.p,
        c.magnitude_lr,
        c.eta,
        c.magnitude_shape,
        c.magnitude_rule,
        c.sphere,
        c.tangent,
        c.native_norm,
        c.radius,
    )
    gen = torch.Generator().manual_seed(6)
    w = torch.randn(3, 12, 8, generator=gen).to(dtype)
    w[1, 4] = 0
    s_live = s_old = w.new_zeros(3, *live.state_shape(12, 8))
    w_live = w_old = w
    for t in range(1, 9):
        G = (0.1 * torch.randn(3, 12, 8, generator=gen)).to(dtype)
        if t == 4:
            s_live, s_old = torch.zeros_like(s_live), torch.zeros_like(s_old)
        a = live.run(G, w_live, s_live, StepCtx(0.05, 0.0, t))
        b = old.run(G, w_old, s_old, StepCtx(0.05, 0.0, t))
        assert torch.equal(a.weight, b.weight) and torch.equal(a.state, b.state), t
        w_live, s_live, w_old, s_old = a.weight, a.state, b.weight, b.state


_POLAR_CASES = [
    *(({**_F, "sphere": "row", "tangent": False}, r) for r in ("cosine", "cosine_raw")),
    (dict(warmup=2, c=0.5), "adam"),
    (dict(warmup=2, c=0.5), "log_rms"),
    (dict(warmup=2, c=0.5), "induced"),
    ({**_F, "sphere": "matrix", "magnitude_shape": "row", "tangent": True}, "sign"),
    ({**_F, "sphere": "row", "tangent": False}, "rel_rms"),
]


@pytest.mark.parametrize("args,rule", _POLAR_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute", ["svd", "newtonschulz5"])
def test_the_rules_but_p1_are_the_57080012_rule_bitwise(args, rule, dtype, compute):
    """cosine, cosine_raw (and a sample of the older rules) against the frozen copy taken before
    the 6.6 rework of polar / polar_t and v3b."""
    extra = {"magnitude_rule": rule}
    if rule in ("adam", "sign", "log_rms"):
        extra["magnitude_lr"] = 0.3
    if rule == "rel_rms":
        extra["magnitude_lr"] = 0.2
    live = rule_of(
        {"rule": "angular_muown", "args": {"compute": compute, **args, **extra}}
    ).compute
    c = live
    old = PolarAngularMuownMap(
        c.direction,
        c.beta,
        c.nesterov,
        c.betas,
        c.eps,
        c.warmup,
        c.c,
        c.p,
        c.magnitude_lr,
        c.eta,
        c.magnitude_shape,
        c.magnitude_rule,
        c.sphere,
        c.tangent,
        c.native_norm,
        c.radius,
    )
    gen = torch.Generator().manual_seed(7)
    w = torch.randn(3, 12, 8, generator=gen).to(dtype)
    w[1, 4] = 0
    s_live = s_old = w.new_zeros(3, *live.state_shape(12, 8))
    w_live = w_old = w
    for t in range(1, 9):
        G = (0.1 * torch.randn(3, 12, 8, generator=gen)).to(dtype)
        if t == 4:
            s_live, s_old = torch.zeros_like(s_live), torch.zeros_like(s_old)
        a = live.run(G, w_live, s_live, StepCtx(0.05, 0.0, t))
        b = old.run(G, w_old, s_old, StepCtx(0.05, 0.0, t))
        assert torch.equal(a.weight, b.weight) and torch.equal(a.state, b.state), t
        w_live, s_live, w_old, s_old = a.weight, a.state, b.weight, b.state


# ---------------------------------------------------------------- settings
@pytest.mark.parametrize(
    "rule,shape",
    [
        ("polar", (12, 11)),
        ("polar_t", (12, 11)),
        ("v3b", (12, 11)),
        ("cosine", (12, 11)),
        ("cosine_raw", (12, 19)),
    ],
)
def test_the_preset_builds_it(rule, shape):
    cm = rule_of(
        {
            "rule": "angular_muown",
            "args": {"compute": "svd", **_F, "tangent": False, "magnitude_rule": rule},
        }
    ).compute
    assert (cm.magnitude_rule, cm.magnitude_lr, cm.state_shape(12, 8)) == (
        rule,
        1.0,
        shape,
    )


@pytest.mark.parametrize("rule", ["polar", "polar_t", "v3b", "cosine", "cosine_raw"])
@pytest.mark.parametrize(
    "bad,match",
    [
        (dict(tangent=True), "{} is specified on F's recipe: tangent false"),
        (dict(sphere="matrix"), "{} is the step of unit rows"),
        (dict(magnitude_lr=0.5), "{} has no free constant"),
        (dict(magnitude_shape="unit"), "{} steps per-row lengths"),
        (dict(eps=1e-6), "{} uses no betas or eps: drop them"),
        (dict(betas=(0.8, 0.95)), "{} uses no betas or eps: drop them"),
    ],
)
def test_settings_polar_does_not_use_are_refused(bad, match, rule):
    kw = dict(
        warmup=0, c=0.0, sphere="row", tangent=False, native_norm=False, radius=M_START
    )
    with pytest.raises(ValueError, match=match.format(rule)):
        AngularMuownMap(
            Rule((), _Polar(), scale="one"), magnitude_rule=rule, **{**kw, **bad}
        )


@pytest.mark.parametrize("rule", ["polar", "polar_t", "v3b", "cosine", "cosine_raw"])
def test_native_polar_is_refused(rule):
    with pytest.raises(ValueError, match=f"{rule} is F's: native_norm false"):
        AngularMuownMap(
            Rule((), _Polar(), scale="image_spectral"),
            magnitude_rule=rule,
            tangent=False,
        )


# ---------------------------------------------------------------- in DiSCO
@pytest.fixture(scope="module")
def one_rank():
    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


def _disco(w, rule="polar"):
    return DiSCO(
        [
            {
                "params": [w],
                "param_names": ["w"],
                "momentum": 1.0,
                "nesterov": False,
                "weight_decay": 0.0,
            }
        ],
        parallel_dims=_NoParallel(),
        lr=LR,
        rule="angular_muown",
        args={
            "compute": "svd",
            **_F,
            "sphere": "row",
            "tangent": False,
            "magnitude_rule": rule,
        },
        param_layouts={"w": ParamLayout("matrix")},
    )


def _steps(opt, w, grads):
    for G in grads:
        w.grad = G.float().clone()
        opt.step()


@pytest.mark.parametrize("rule", ["polar", "polar_t", "v3b", "cosine", "cosine_raw"])
def test_a_disco_step_with_polar_is_the_rule(one_rank, rule):
    W, grads = _data(12, 8, seed=9, steps=4)
    w = nn.Parameter(W.float().clone())
    _steps(_disco(w, rule), w, grads)
    if rule in ("polar", "polar_t", "v3b"):
        ref = _reference_p1(
            W, grads, k=1.0, cols=rule == "v3b", transport=rule != "polar"
        )[-1]
    else:
        ref = _reference_cosine(W, grads, k=1.0, raw=rule == "cosine_raw")[-1]
    assert float((w.detach().double() - ref).abs().max()) < 1e-5


@pytest.mark.parametrize("rule", ["polar", "polar_t", "v3b", "cosine", "cosine_raw"])
def test_a_resumed_polar_run_is_the_continuous_one(one_rank, rule):
    """The momenta (ambient, transported, R, cosine_raw's extra buffer) live in rule_state: saved and
    loaded, the next steps are bitwise the same; a fresh state is not."""
    W, grads = _data(12, 8, seed=10, steps=5)
    wa = nn.Parameter(W.float().clone())
    a = _disco(wa, rule)
    _steps(a, wa, grads[:3])
    saved = wa.detach().clone()
    sd = copy.deepcopy(a.state_dict())
    _steps(a, wa, grads[3:])
    if rule == "cosine_raw":
        assert sd["state"][0]["rule_state"].shape == (1, 12, 19)
        assert (
            sd["state"][0]["rule_state"][0, :, 11:].abs().sum() > 0
        )  # the raw momentum
    wb, wc = nn.Parameter(saved.clone()), nn.Parameter(saved.clone())
    b, c = _disco(wb, rule), _disco(wc, rule)
    b.load_state_dict(sd)
    _steps(b, wb, grads[3:])
    _steps(c, wc, grads[3:])
    assert torch.equal(wa.detach(), wb.detach())
    assert not torch.equal(wa.detach(), wc.detach())


@dataclass(frozen=True)
class _PassThrough(Compute):
    """Hands the momentum on unchanged, as an iterative orthogonaliser does a NaN."""

    def core(self, x):
        return x


@pytest.mark.parametrize("rule", ["polar", "polar_t", "v3b"])
def test_a_non_finite_length_is_an_error(rule):
    W, grads = _data(12, 8, seed=11, steps=1)
    G = grads[0].clone()
    G[3, 2] = float("nan")
    cm = AngularMuownMap(
        Rule((), _PassThrough(), scale="one"),
        warmup=0,
        c=0.0,
        magnitude_rule=rule,
        sphere="row",
        tangent=False,
        native_norm=False,
    )
    with pytest.raises(RuntimeError, match=f"{rule}: a length is not finite"):
        cm.run(G[None], W[None], W.new_zeros(1, 12, 11), StepCtx(LR, 0.0, 1))
