# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""magnitude_rule "log_rms" (optimizer-study/tools/review/theory-radius-norm.md 3.7) against
its definition, in float64:

    mu  <- beta1 mu + (1 - beta1) g gamma          (dL/ds, s = log|g|, gamma = <G_i, U_i>)
    g   <- g exp(-rho turn_t mu / RMS_rows(mu))     (rho = magnitude_lr)
    turn_t = lr_t kappa_t eta(t)                    (the step U turns by)

plus its symmetry claim (Prop. 1-2: an exact per-head scale gauge stays frozen) and
adam / sign bitwise against the wt-s2 class. "rel_rms" is the same with the plain relative
step g <- g (1 + s), s = -rho turn_t mu / RMS_rows(mu) (no exp). "induced" (doc 5.4) is the
U-frame operator-norm length step: x = mu / q, q_i = sum_j (u_i . u_j)^2 |mu_j|,
s = -turn_t sqrt(max(1, m/n)) x / ||Diag(x) U||_op, g <- g exp(s)."""

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
from torchtitan.optimizers.helpers.norm_helper import get_norms_to_log

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _angular_muown_logrms_reference import LogRmsAngularMuownMap  # noqa: E402
from _angular_muown_relrms_reference import RelRmsAngularMuownMap  # noqa: E402
from _angular_muown_s2_reference import S2AngularMuownMap  # noqa: E402

LR, RHO, M_START = 0.05, 0.25, 4.0
NATIVE = dict(warmup=2, c=0.5)  # lr ramps and kappa moves within six steps
SCHED = EtaSchedule("sqrt_decay", start=2, total=6)
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
    """Rows of unequal norm and gradients."""
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


RULES = ("log_rms", "rel_rms")


def _log_rms(g, mu, dLds, rate, b1, mutate, rule="log_rms"):
    """One length step from the equations; returns (g+, mu+)."""
    mu = b1 * mu + (1 - b1) * dLds
    d = torch.sign(mu) if mutate == "log_sign" else mu / mu.square().mean().sqrt()
    if mutate == "additive":
        return g - rate * d, mu
    if rule == "rel_rms":
        return g * (1 - rate * d), mu
    return g * torch.exp(-rate * d), mu


def _reference(
    W,
    grads,
    *,
    lr=LR,
    rho=RHO,
    beta=0.95,
    nesterov=True,
    b1=0.9,
    warmup=2,
    c=0.5,
    p=1.0,
    eta=None,
    mutate=None,
    rule="log_rms",
):
    """Native AM (Alg. 1: unit rows U, momentum on Diag(g)(G - gamma U), U turned by
    turn_t sqrt(max(1, m/n)) polar, RowNorm) with log-RMS lengths."""
    m, n = W.shape
    g = W.norm(dim=1, keepdim=True)
    u = W / g
    mom, mu, out = torch.zeros_like(W), torch.zeros_like(g), []
    for t, G in enumerate(grads, start=1):
        lr_t = lr * min(1.0, t / warmup) if warmup else lr
        kappa = (1.0 + c * max(0, t - 1 - warmup)) ** -p
        turn = lr_t * kappa * (1.0 if eta is None else eta(t))
        gamma = (G * u).sum(dim=1, keepdim=True)
        GU = g * (G - gamma * u)
        mom = beta * mom + GU
        D = GU + beta * mom if nesterov else mom
        V = u - turn * math.sqrt(max(1.0, m / n)) * _exact_polar(D)
        dLds = gamma if mutate == "raw_gamma" else g * gamma
        rate = rho * (lr_t if mutate == "length_on_lr" else turn)
        g, mu = _log_rms(g, mu, dLds, rate, b1, mutate, rule)
        u = V / V.norm(dim=1, keepdim=True)
        out.append(g * u)
    return out


def _native(rho=RHO, rule="log_rms", **kw):
    return AngularMuownMap(
        Rule((), _Polar(), scale="image_spectral"),
        magnitude_rule=rule,
        magnitude_lr=rho,
        **{**NATIVE, **kw},
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


# ---------------------------------------------------------------- (1) the definition
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("sched", [None, SCHED])
@pytest.mark.parametrize("rule", RULES)
def test_native_log_rms_is_its_definition_step_after_step(m, n, nesterov, sched, rule):
    W, grads = _data(m, n, seed=m + 2 * n)
    ref = _reference(W, grads, nesterov=nesterov, eta=sched, rule=rule)
    got, _ = _run(_native(nesterov=nesterov, eta=sched, rule=rule), W, grads)
    assert _max_diff(ref, got) < 1e-10


@pytest.mark.parametrize(
    "mutate", ["length_on_lr", "log_sign", "additive", "raw_gamma"]
)
def test_the_comparison_sees_each_plausible_mistake(mutate):
    W, grads = _data(12, 8, seed=1)
    assert (
        _max_diff(
            _reference(W, grads, eta=SCHED),
            _reference(W, grads, eta=SCHED, mutate=mutate),
        )
        > 1e-4
    )


def test_rel_and_log_references_differ():
    """1 + s vs exp(s): far above the 1e-10 the definition tests hold."""
    W, grads = _data(12, 8, seed=1)
    native = _max_diff(
        _reference(W, grads, eta=SCHED), _reference(W, grads, eta=SCHED, rule="rel_rms")
    )
    kw = dict(sphere="row", tangent=False)
    f = _max_diff(
        _reference_f(W, grads, **kw), _reference_f(W, grads, rule="rel_rms", **kw)
    )
    assert native > 1e-6 and f > 1e-6, (native, f)


@pytest.mark.parametrize("rule", RULES)
def test_the_relative_length_step_has_rms_rho_turn(rule):
    """RMS_rows(log g+ - log g) (log_rms) or RMS_rows(g+ / g - 1) (rel_rms) = rho turn_t
    exactly, whatever g0 and gamma."""
    W, grads = _data(12, 8, seed=2)
    _, states = _run(_native(eta=SCHED, rule=rule), W, grads)
    g = [W.norm(dim=1)] + [st[:, 8] for st in states]
    for t in range(1, len(g)):
        turn = LR * min(1.0, t / 2) * (1.0 + 0.5 * max(0, t - 3)) ** -1 * SCHED(t)
        if rule == "log_rms":
            step = g[t].abs().log() - g[t - 1].abs().log()
        else:
            step = g[t] / g[t - 1] - 1
        rms = step.square().mean().sqrt()
        assert abs(float(rms) - RHO * turn) < 1e-13, t


def _reference_f(
    W,
    grads,
    *,
    sphere,
    tangent,
    M=M_START,
    lr=LR,
    rho=RHO,
    beta=0.95,
    b1=0.9,
    k=0.5,
    eta=F_ETA,
    mutate=None,
    rule="log_rms",
):
    """F's unified rule (PREREG 7b.3, D1): row sphere unit rows / matrix sphere ||U||_F = sqrt(m)
    with g at the RMS row norm; g = M x that from step 1 (the step-1 gradient is W0's, so its
    dL/ds is taken at the start lengths); O rescaled to ||U||_F; turn_t = lr eta(t)."""
    m, n = W.shape
    norms = W.norm(dim=1, keepdim=True)
    start = (
        norms if sphere == "row" else norms.square().mean().sqrt().expand(m, 1).clone()
    )
    U, g = W / start, M * start
    mom, mu, out = torch.zeros_like(W), torch.zeros_like(g), []
    for t, G in enumerate(grads, start=1):
        GU = g * G
        if tangent:
            GU = GU - (
                (GU * U).sum(dim=1, keepdim=True) * U
                if sphere == "row"
                else (GU * U).sum() / (U * U).sum() * U
            )
        mom = beta * mom + GU
        D = GU + beta * mom
        O = _exact_polar(D, k)
        V = U - lr * eta(t) * O * U.norm() / O.norm()
        U_next = (
            V / V.norm(dim=1, keepdim=True)
            if sphere == "row"
            else V * (U.norm() / V.norm())
        )
        gamma = (G * U).sum(dim=1, keepdim=True)
        at = start if t == 1 and mutate != "jump_in_log_gradient" else g
        g, mu = _log_rms(g, mu, at * gamma, rho * lr * eta(t), b1, None, rule)
        U = U_next
        out.append(g * U)
    return out


def _f_map(sphere, tangent, rho=RHO, M=M_START, k=0.5, rule="log_rms"):
    return AngularMuownMap(
        Rule((), _Polar(k), scale="one"),
        warmup=0,
        c=0.0,
        eta=F_ETA,
        magnitude_lr=rho,
        magnitude_rule=rule,
        sphere=sphere,
        tangent=tangent,
        native_norm=False,
        radius=M,
    )


F_TOL = {"row": 1e-10, "matrix": 1e-5}  # the matrix step is FrobeniusAUS's, in fp32


@pytest.mark.parametrize(
    "sphere,tangent", itertools.product(("row", "matrix"), (False, True))
)
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
@pytest.mark.parametrize("rule", RULES)
def test_f_cells_with_log_rms_are_their_definition(sphere, tangent, m, n, rule):
    W, grads = _data(m, n, seed=m + 3 * n)
    ref = _reference_f(W, grads, sphere=sphere, tangent=tangent, rule=rule)
    got, _ = _run(_f_map(sphere, tangent, rule=rule), W, grads)
    assert _max_diff(ref, got) < F_TOL[sphere]


def test_the_f_comparison_sees_the_jump_in_the_log_gradient():
    W, grads = _data(12, 8, seed=4)
    kw = dict(sphere="row", tangent=True)
    assert (
        _max_diff(
            _reference_f(W, grads, **kw),
            _reference_f(W, grads, mutate="jump_in_log_gradient", **kw),
        )
        > 1e-4
    )


@pytest.mark.parametrize("rule", RULES + ("induced",))
def test_each_unit_of_a_stack_has_its_own_rms(rule):
    (Wa, ga), (Wb, gb) = _data(12, 8, seed=5), _data(12, 8, seed=6)
    Wb = 3.0 * Wb
    cm = _induced() if rule == "induced" else _native(rule=rule)
    w, state = torch.stack([Wa, Wb]), Wa.new_zeros(2, *cm.state_shape(12, 8))
    for t, (A, B) in enumerate(zip(ga, gb), start=1):
        res = cm.run(torch.stack([A, B]), w, state, StepCtx(LR, 0.0, t))
        w, state = res.weight, res.state
    for k, (W, grads) in enumerate(((Wa, ga), (Wb, gb))):
        assert float((w[k] - _run(cm, W, grads)[0][-1]).abs().max()) < 1e-12


# ---------------------------------------------------------------- (2) the symmetry
def _head_loss(n_heads, hd, n, seed):
    """sum tanh(A * q~): q = x W^T per token, q~ RMS-normalised within each head (no eps),
    so L is invariant under scaling the rows of any one head (doc 3.2, k = 1_head)."""
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(16, n, generator=gen, dtype=torch.float64)
    A = torch.randn(16, n_heads, hd, generator=gen, dtype=torch.float64)

    def grad(W):
        W = W.detach().requires_grad_(True)
        q = (X @ W.T).view(16, n_heads, hd)
        q = q / q.square().mean(dim=-1, keepdim=True).sqrt()
        (G,) = torch.autograd.grad(torch.tanh(A * q).sum(), W)
        return G

    return grad


def _head_drift(cm, W, grad, n_heads, steps=8):
    """Per step, the per-head mean of log|g+| - log|g| (g from the rule's state)."""
    m, n = W.shape
    w, state, g_prev, drift = (
        W[None],
        W.new_zeros(1, *cm.state_shape(m, n)),
        W.norm(dim=1),
        [],
    )
    for t in range(1, steps + 1):
        res = cm.run(grad(w[0])[None], w, state, StepCtx(LR, 0.0, t))
        w, state = res.weight, res.state
        g = state[0, :, n]
        drift.append((g.abs().log() - g_prev.abs().log()).view(n_heads, -1).mean(dim=1))
        g_prev = g
    return torch.stack(drift)


@pytest.mark.parametrize("rule", ["log_rms", "adam", "sign", "induced"])
def test_a_head_scale_gauge_is_frozen_only_by_log_rms(rule):
    n_heads, hd, n = 3, 4, 8
    W, _ = _data(n_heads * hd, n, seed=7, steps=0)
    grad = _head_loss(n_heads, hd, n, seed=8)
    G0 = grad(W)
    euler = (
        (W * G0).sum(dim=1).view(n_heads, hd).sum(dim=1)
    )  # sum_i g_i gamma_i per head
    assert float(euler.abs().max()) < 1e-12  # the loss has the gauge
    cm = AngularMuownMap(
        Rule((), _Polar(), scale="image_spectral"),
        magnitude_rule=rule,
        magnitude_lr=1.0,
        **NATIVE,
    )
    drift = _head_drift(
        cm, W, grad, n_heads
    )  # induced: not symmetry-aware (doc 5.3, 5.8)
    if rule == "log_rms":
        assert float(drift.abs().max()) < 1e-12
    else:
        assert float(drift.abs().max()) > 1e-4


def test_rel_rms_moves_a_head_gauge_only_at_second_order():
    """The head-mean of s is 0 exactly (sum_head g gamma = 0), so the head's mean log length
    moves by mean log(1 + s) = -mean(s^2) / 2 + O(s^3): below 2 max(s)^2, and not 0."""
    n_heads, hd, n = 3, 4, 8
    W, _ = _data(n_heads * hd, n, seed=7, steps=0)
    grad = _head_loss(n_heads, hd, n, seed=8)
    cm = _native(rho=0.25, rule="rel_rms")
    w, state, g_prev = (
        W[None],
        W.new_zeros(1, *cm.state_shape(n_heads * hd, n)),
        W.norm(dim=1),
    )
    for t in range(1, 9):
        res = cm.run(grad(w[0])[None], w, state, StepCtx(LR, 0.0, t))
        w, state = res.weight, res.state
        g = state[0, :, n]
        s = (g / g_prev - 1).view(n_heads, hd)
        log_mean = (g / g_prev).log().view(n_heads, hd).mean(dim=1)
        assert float(s.mean(dim=1).abs().max()) < 1e-15, t  # the gauge component of s
        assert float(log_mean.abs().max()) < 2 * float(s.abs().max()) ** 2, t
        assert float(log_mean.abs().min()) > 1e-9, t
        g_prev = g


# ---------------------------------------------------------------- rel_rms positivity
def test_rel_rms_refuses_rho_above_one_quarter_and_a_step_above_one_half():
    _native(rho=0.25, rule="rel_rms")
    with pytest.raises(
        ValueError, match="rel_rms runs at magnitude_lr \\(rho\\) <= 0.25"
    ):
        _native(rho=0.26, rule="rel_rms")
    W, grads = _data(12, 8, seed=12, steps=1)
    cm = _native(rho=0.25, rule="rel_rms", warmup=0, c=0.0)
    cm.run(
        grads[0][None], W[None], W.new_zeros(1, 12, 11), StepCtx(2.0, 0.0, 1)
    )  # rho turn = 0.5: allowed
    with pytest.raises(ValueError, match="rel_rms needs magnitude_lr x turn <= 0.5"):
        cm.run(grads[0][None], W[None], W.new_zeros(1, 12, 11), StepCtx(2.04, 0.0, 1))


@pytest.mark.parametrize("rule", RULES)
def test_a_factor_at_or_below_zero_is_an_error_not_a_flip(rule):
    """One row holding all of the log-gradient has |s| = rho turn sqrt(m): at rho turn = 0.4,
    m = 16 that is 1.6, so 1 + s < 0. rel_rms stops there; log_rms (exp) stays positive."""
    m, n = 16, 8
    W, _ = _data(m, n, seed=13, steps=0)
    G = torch.zeros(m, n, dtype=torch.float64)
    G[5] = W[5]  # gamma_5 > 0, every other gamma 0
    cm = _native(rho=0.25, rule=rule, warmup=0, c=0.0)
    run = lambda: cm.run(
        G[None], W[None], W.new_zeros(1, m, n + 3), StepCtx(1.6, 0.0, 1)
    )  # noqa: E731
    if rule == "rel_rms":
        with pytest.raises(RuntimeError, match="rel_rms"):
            run()
    else:
        assert bool((run().state[0, :, n] > 0).all())


# ---------------------------------------------------------------- (3) adam / sign unchanged
def _s2(cm):
    return S2AngularMuownMap(
        cm.direction,
        cm.beta,
        cm.nesterov,
        cm.betas,
        cm.eps,
        cm.warmup,
        cm.c,
        cm.p,
        cm.magnitude_lr,
        cm.eta,
        cm.magnitude_shape,
        cm.magnitude_rule,
        cm.sphere,
        cm.tangent,
        cm.native_norm,
        cm.radius,
    )


_GPE = {"compute": "gram_polar_express", "steps": 6}
_F = {
    "scale": "one",
    "warmup": 0,
    "c": 0.0,
    "native_norm": False,
    "radius": 4.0,
    "eta": {"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0},
}
_CASES = [
    dict(warmup=2, c=0.5),
    dict(warmup=2, c=0.5, nesterov=False, magnitude_shape="unit"),
    dict(warmup=2, c=0.5, eta={"shape": "sqrt_decay", "start": 2, "total": 9}),
    *(
        {**_F, "sphere": s, "magnitude_shape": g, "tangent": t, "magnitude_lr": 0.5}
        for s, g, t in itertools.product(
            ("row", "matrix"), ("unit", "row"), (False, True)
        )
    ),
    {**_F, "sphere": "row", "magnitude_shape": "none", "tangent": False},
    {**_F, "sphere": "matrix", "magnitude_shape": "none", "tangent": True},
]


@pytest.mark.parametrize("args", _CASES)
@pytest.mark.parametrize("rule", ["adam", "sign"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute", ["svd", "newtonschulz5"])
def test_adam_and_sign_are_the_s2_rule_bitwise(args, rule, dtype, compute):
    """Stacks of 3 units with a zero row, a lost state at step 4, 8 steps."""
    if args.get("magnitude_shape") == "none" and rule == "sign":
        pytest.skip("none learns no magnitudes")
    extra = {} if args.get("magnitude_shape") == "none" else {"magnitude_rule": rule}
    live = rule_of(
        {"rule": "angular_muown", "args": {"compute": compute, **args, **extra}}
    ).compute
    old = _s2(live)
    gen = torch.Generator().manual_seed(3)
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


_LOG_CASES = [
    dict(warmup=2, c=0.5),
    dict(
        warmup=2,
        c=0.5,
        nesterov=False,
        eta={"shape": "sqrt_decay", "start": 2, "total": 9},
    ),
    *(
        {**_F, "sphere": s, "tangent": t}
        for s, t in itertools.product(("row", "matrix"), (False, True))
    ),
]


@pytest.mark.parametrize("args", _LOG_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute", ["svd", "newtonschulz5"])
def test_log_rms_is_the_cc77dab_rule_bitwise(args, dtype, compute):
    """log_rms against its frozen copy (before rel_rms): the same stacks as above."""
    a_ = {
        "compute": compute,
        **args,
        "magnitude_rule": "log_rms",
        "magnitude_lr": 0.3,
    }  # not 2^k
    live = rule_of({"rule": "angular_muown", "args": a_}).compute
    c = live
    old = LogRmsAngularMuownMap(
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
    gen = torch.Generator().manual_seed(4)
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


@pytest.mark.parametrize("args", _LOG_CASES)
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute", ["svd", "newtonschulz5"])
def test_log_and_rel_rms_are_the_9e09896_rule_bitwise(args, rule, dtype, compute):
    """log_rms and rel_rms against the frozen copy taken before "induced"."""
    a_ = {
        "compute": compute,
        **args,
        "magnitude_rule": rule,
        "magnitude_lr": 0.2,
    }  # not 2^k
    live = rule_of({"rule": "angular_muown", "args": a_}).compute
    c = live
    old = RelRmsAngularMuownMap(
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
    gen = torch.Generator().manual_seed(5)
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


# ---------------------------------------------------------------- induced (doc 5.4)
def _q(U, c):
    """q_i = sum_j (u_i . u_j)^2 |c_j| (doc 5.4)."""
    K = U @ U.T
    return (K * K) @ c.abs()


def _op(x, U):
    """N_U(x) = ||Diag(x) U||_op, exactly (SVD)."""
    return float(torch.linalg.matrix_norm(x[:, None] * U, ord=2))


def _induced(**kw):
    return AngularMuownMap(
        Rule((), _Polar(), scale="image_spectral"),
        magnitude_rule="induced",
        **{**NATIVE, **kw},
    )


def _turn(t, lr=LR, warmup=2, c=0.5):
    return lr * min(1.0, t / warmup) * (1.0 + c * max(0, t - 1 - warmup)) ** -1


def _one_step(cm, W, G, lr=LR):
    """g before, the log-length step s, and mu after one step from an empty state."""
    m, n = W.shape
    res = cm.run(G[None], W[None], W.new_zeros(1, m, n + 3), StepCtx(lr, 0.0, 1))
    g0, g1 = W.norm(dim=1), res.state[0, :, n]
    return g0, (g1 / g0).log(), res.state[0, :, n + 1]


@pytest.mark.parametrize("m,n", [(8, 12), (8, 8)])
def test_induced_with_orthonormal_rows_is_log_sign(m, n):
    """K = I: q = |c|, x = sign(c), N_U(x) = 1, so s = -turn sqrt(max(1, m/n)) sign(c)."""
    gen = torch.Generator().manual_seed(m + n)
    Q = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))[0][:m]
    W = torch.logspace(-0.5, 0.5, m, dtype=torch.float64)[:, None] * Q
    G = 0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
    _, s, mu = _one_step(_induced(), W, G)
    want = -_turn(1) * math.sqrt(max(1.0, m / n)) * torch.sign(mu)
    assert float((s - want).abs().max()) < 1e-10


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
def test_induced_with_identical_rows_is_log_rms(m, n):
    """K = 1 1^T: q = ||c||_1, N_U(x) = ||x||_2, so s = -turn sqrt(max(1, m/n)) c / ||c||_2:
    the log-RMS step at rho = sqrt(max(1, m/n) / m)."""
    gen = torch.Generator().manual_seed(m * n)
    u = torch.randn(n, generator=gen, dtype=torch.float64)
    W = torch.logspace(-0.5, 0.5, m, dtype=torch.float64)[:, None] * (u / u.norm())
    G = 0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
    _, s, mu = _one_step(_induced(), W, G)
    budget = _turn(1) * math.sqrt(max(1.0, m / n))
    assert float((s - (-budget * mu / mu.norm())).abs().max()) < 1e-10
    rho = math.sqrt(max(1.0, m / n) / m)
    _, s_log, _ = _one_step(_native(rho=rho, rule="log_rms"), W, G)
    assert float((s - s_log).abs().max()) < 1e-12


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
def test_induced_is_the_closed_form_and_never_under_budget(m, n):
    """Each step: s points along -x, x = mu / q (q from the doc's sum over the pre-step rows),
    and ||Diag(s) U||_op >= turn sqrt(max(1, m/n)) (a power iteration underestimates N_U)."""
    W, grads = _data(m, n, seed=m + 5 * n, steps=6)
    cm = _induced()
    w, state, mu = (
        W[None],
        W.new_zeros(1, m, n + 3),
        torch.zeros(m, dtype=torch.float64),
    )
    for t, G in enumerate(grads, start=1):
        g = state[0, :, n] if t > 1 else W.norm(dim=1)
        U = w[0] / g[:, None]
        mu = 0.9 * mu + 0.1 * g * (G * U).sum(dim=1)  # the reference's own EMA
        x = mu / _q(U, mu)
        res = cm.run(G[None], w, state, StepCtx(LR, 0.0, t))
        s = (res.state[0, :, n] / g).log()
        assert float((s / s.norm() + x / x.norm()).abs().max()) < 1e-10, t
        ratio = _op(s, U) / (_turn(t) * math.sqrt(max(1.0, m / n)))
        assert 1 - 1e-12 <= ratio < 1.5, (t, ratio)
        w, state = res.weight, res.state


@pytest.mark.parametrize("tangent", [False, True])
def test_induced_on_f_cells_is_the_closed_form(tangent):
    """F's row sphere at start radius 4: the step-1 log-gradient is taken at the start lengths
    (pre-jump), and the step is applied to 4x them; turn_t = lr eta(t)."""
    m, n = 12, 8
    W, grads = _data(m, n, seed=21, steps=4)
    cm = AngularMuownMap(
        Rule((), _Polar(0.5), scale="one"),
        warmup=0,
        c=0.0,
        eta=F_ETA,
        magnitude_rule="induced",
        sphere="row",
        tangent=tangent,
        native_norm=False,
        radius=M_START,
    )
    w, state, mu = (
        W[None],
        W.new_zeros(1, m, n + 3),
        torch.zeros(m, dtype=torch.float64),
    )
    for t, G in enumerate(grads, start=1):
        g_at = state[0, :, n] if t > 1 else W.norm(dim=1)
        U = w[0] / g_at[:, None]
        mu = 0.9 * mu + 0.1 * g_at * (G * U).sum(dim=1)  # the reference's own EMA
        x = mu / _q(U, mu)
        res = cm.run(G[None], w, state, StepCtx(LR, 0.0, t))
        s = (res.state[0, :, n] / (M_START * g_at if t == 1 else g_at)).log()
        assert float((s / s.norm() + x / x.norm()).abs().max()) < 1e-10, t
        assert _op(s, U) / (LR * F_ETA(t) * math.sqrt(m / n)) >= 1 - 1e-12, t
        w, state = res.weight, res.state


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
def test_induced_meets_its_budget_once_the_power_vector_has_converged(m, n):
    """The power vector is warm-started from the state: under a slowly moving gradient the
    step's operator norm reaches the budget turn sqrt(max(1, m/n)) to 1e-4."""
    W, grads = _data(m, n, seed=m + 7 * n, steps=1)
    gen = torch.Generator().manual_seed(1)
    cm = _induced(warmup=0, c=0.0)
    w, state, lr, ratio = W[None], W.new_zeros(1, m, n + 3), 1e-3, None
    for t in range(1, 41):
        G = grads[0] + 1e-3 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        g = state[0, :, n] if t > 1 else W.norm(dim=1)
        U = w[0] / g[:, None]
        res = cm.run(G[None], w, state, StepCtx(lr, 0.0, t))
        s = (res.state[0, :, n] / g).log()
        ratio = _op(s, U) / (lr * math.sqrt(max(1.0, m / n)))
        w, state = res.weight, res.state
    assert abs(ratio - 1) < 1e-4, ratio


# ---------------------------------------------------------------- settings
@pytest.mark.parametrize("rule", RULES + ("induced",))
def test_the_preset_builds_it(rule):
    rho = 1.0 if rule == "induced" else 0.25
    args = {
        "compute": "svd",
        "magnitude_rule": rule,
        **({} if rule == "induced" else {"magnitude_lr": rho}),
    }
    cm = rule_of({"rule": "angular_muown", "args": args}).compute
    assert (cm.magnitude_rule, cm.magnitude_lr, cm.state_shape(12, 8)) == (
        rule,
        rho,
        (12, 11),
    )


@pytest.mark.parametrize("rule", RULES + ("induced",))
@pytest.mark.parametrize(
    "bad,match",
    [
        (dict(magnitude_shape="unit"), "{} steps per-row lengths"),
        (dict(magnitude_shape="none"), "learns no magnitudes"),
        (dict(eps=1e-6), "{} uses betas\\[0\\] only"),
        (dict(betas=(0.9, 0.99)), "{} uses betas\\[0\\] only"),
    ],
)
def test_settings_log_rms_does_not_use_are_refused(bad, match, rule):
    with pytest.raises(ValueError, match=match.format(rule)):
        rho = {"magnitude_lr": 0.25} if rule == "rel_rms" else {}
        if rule == "induced" and bad.get("magnitude_shape") == "none":
            match = "learns no magnitudes|induced steps per-row lengths"
        AngularMuownMap(
            Rule((), _Polar(), scale="image_spectral"),
            magnitude_rule=rule,
            **rho,
            **bad,
        )


@pytest.mark.parametrize(
    "bad,match",
    [
        (dict(magnitude_lr=0.5), "induced has no free constant"),
        (
            dict(
                sphere="matrix",
                native_norm=False,
                warmup=0,
                c=0.0,
                direction=Rule((), _Polar(), scale="one"),
            ),
            "induced is the step of unit rows",
        ),
    ],
)
def test_induced_settings_it_does_not_use_are_refused(bad, match):
    kw = {"direction": Rule((), _Polar(), scale="image_spectral"), **bad}
    with pytest.raises(ValueError, match=match):
        AngularMuownMap(magnitude_rule="induced", **kw)


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


def _disco(w, rule="log_rms"):
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
            **NATIVE,
            "magnitude_rule": rule,
            **({} if rule == "induced" else {"magnitude_lr": RHO}),
        },
        param_layouts={"w": ParamLayout("matrix")},
    )


def _steps(opt, w, grads):
    for G in grads:
        w.grad = G.float().clone()
        opt.step()


@pytest.mark.parametrize("rule", RULES)
def test_a_disco_step_with_log_rms_is_the_rule(one_rank, rule):
    W, grads = _data(12, 8, seed=9, steps=4)
    w = nn.Parameter(W.float().clone())
    _steps(_disco(w, rule), w, grads)
    ref = _reference(W, grads, rule=rule)[-1]
    assert float((w.detach().double() - ref).abs().max()) < 1e-5


@pytest.mark.parametrize("rule", RULES + ("induced",))
def test_a_resumed_log_rms_run_is_the_continuous_one(one_rank, rule):
    """mu (and induced's power vector) live in rule_state: saved and loaded, the next steps are
    bitwise the same; without the load (a fresh state) they are not."""
    W, grads = _data(12, 8, seed=10, steps=5)
    wa = nn.Parameter(W.float().clone())
    a = _disco(wa, rule)
    _steps(a, wa, grads[:3])
    saved = wa.detach().clone()
    sd = copy.deepcopy(a.state_dict())  # a checkpoint is written at save time
    assert sd["state"][0]["rule_state"][0, :, 9].abs().sum() > 0  # mu, the m_g column
    _steps(a, wa, grads[3:])
    if rule == "induced":
        assert (
            sd["state"][0]["rule_state"][0, :, 10].abs().sum() > 0
        )  # the power vector
    wb, wc = nn.Parameter(saved.clone()), nn.Parameter(saved.clone())
    b, c = _disco(wb, rule), _disco(wc, rule)
    b.load_state_dict(sd)
    _steps(b, wb, grads[3:])
    _steps(c, wc, grads[3:])
    assert torch.equal(wa.detach(), wb.detach())
    assert not torch.equal(wa.detach(), wc.detach())


def test_a_log_rms_logging_step_logs_the_row_norms(one_rank):
    W, grads = _data(12, 8, seed=11, steps=2)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w)
    for G in grads:
        opt.calculate_norm_at_next_step(
            norms_to_log=get_norms_to_log("all"), gram_level=0, track_spectrum=False
        )
        w.grad = G.float().clone()
        opt.step()
        logged = opt.get_norms_at_current_step()
    rows = {k: float(v) for k, v in logged.items() if "rows_norm_q" in k}
    assert len(rows) == 3, sorted(logged)
    g = opt.state[w]["rule_state"][0, :, 8].abs()
    assert rows[next(k for k in rows if "q50" in k)] == pytest.approx(
        float(g.quantile(0.5)), rel=1e-5
    )
    assert all(math.isfinite(float(v)) for v in logged.values())
