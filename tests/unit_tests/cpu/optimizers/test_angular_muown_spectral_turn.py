# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""S1 "row-sphere aus" (theory-radius-norm.md section 7, PREREG "Spectral candidate S1"): F's recipe
with the turn's norm spectral. Per unit, W = Diag(g) U with unit rows; F's momentum of g G (Nesterov),
O = Orth(D), and the step rescaled to

    F:   dU = -turn (||U||_F / ||O||_F) O     ->  ||dU||_F = turn ||U||_F
    S1:  dU = -turn (sigma(U) / sigma(O)) O   ->  ||dU||_2 = turn sigma_max(U)

sigma by k = 20 power iterations (coordinator 10-05; section 7 said 5), batched over the stack,
started from U's normalised mean row (for U and for O), no state; U+ = RowNorm(U + dU);
lengths by F's Adam. References in float64, written from
these equations; turn_norm "frobenius" is bitwise the tree before (517ab8b2)."""

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
from torchtitan.optimizers.helpers.power_iteration import top_pair_warm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _angular_muown_s4_reference import S4AngularMuownMap  # noqa: E402

LR, MAG_LR, M_START, K_POWER = 0.05, 0.3, 4.0, 20
ETA = EtaSchedule("inverse_poly", start=1, c=0.5)


@dataclass(frozen=True)
class _Polar(Compute):
    """k U V^T in float64: sigma(O) = k, so a missing 1 / sigma(O) shows."""

    k: float = 0.5

    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return self.k * (u @ vh)


@dataclass(frozen=True)
class _Identity(Compute):
    """O = D: a spread spectrum, so where O's power iteration starts shows."""

    def core(self, x):
        return x


def _orth(X, kind):
    if kind == "identity":
        return X.clone()
    u, _, vh = torch.linalg.svd(X, full_matrices=False)
    return kind * (u @ vh)


def _power_sigma(A, v0, k=K_POWER):
    """k power iterations on A^T A from v0; sigma = ||A v||."""
    v = v0 / v0.norm()
    for _ in range(k):
        v = A.T @ (A @ v)
        v = v / v.norm()
    return (A @ v).norm()


def _data(m, n, seed, steps=6):
    gen = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(
        torch.randn(max(m, n), max(m, n), generator=gen, dtype=torch.float64)
    )[0][:m, :n]
    W = W + 0.3 * torch.randn(n, generator=gen, dtype=torch.float64) / math.sqrt(
        n
    )  # a shared row part
    W = W * torch.logspace(-0.3, 0.3, m, dtype=torch.float64)[:, None]
    return W, [
        0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]


def _spiky(m, n, a, seed, mixed=False):
    """Unit rows a u_i c + noise: a planted rank-1 spike (u = 1, or random signs with `mixed`)."""
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(n, generator=gen, dtype=torch.float64)
    c = c / c.norm()
    u = torch.ones(m, dtype=torch.float64)
    if mixed:
        u = torch.randint(0, 2, (m,), generator=gen).to(torch.float64) * 2 - 1
    X = a * u[:, None] * c + torch.randn(
        m, n, generator=gen, dtype=torch.float64
    ) / math.sqrt(n)
    return X / X.norm(dim=1, keepdim=True)


def _adam(state, x, t, b1=0.9, b2=0.95, eps=1e-8):
    state["m"] = b1 * state["m"] + (1 - b1) * x
    state["v"] = b2 * state["v"] + (1 - b2) * x * x
    return (state["m"] / (1 - b1**t)) / ((state["v"] / (1 - b2**t)).sqrt() + eps)


def _reference(
    W,
    grads,
    *,
    turn_norm="spectral",
    scale="row",
    tangent=False,
    nesterov=True,
    orth=0.5,
    M=M_START,
    lr=LR,
    mag_lr=MAG_LR,
    beta=0.95,
    eta=ETA,
    sigma="power",
    mutate=None,
):
    """F's row sphere: U = W0 / its row norms, g = M x those norms from step 1; G_U = g G (tangent:
    minus its radial part); M <- beta M + G_U; D = G_U + beta M (Nesterov); O = Orth(D); O~ = O times
    sigma(U) / sigma(O) (spectral) or ||U||_F / ||O||_F; U+ = RowNorm(U - lr eta(t) O~). Lengths:
    Adam on <G_i, U_i> per row, or on sum_i q_i <G_i, U_i> per matrix (W = a Diag(q) U), at lr mag_lr;
    "none" keeps g."""
    m, n = W.shape
    start = W.norm(dim=1, keepdim=True)
    U, g = W / start, M * start
    q, a = start / start.square().mean().sqrt(), M * start.square().mean().sqrt()
    mom, out = torch.zeros_like(W), []
    adam = (
        {"m": torch.zeros_like(g), "v": torch.zeros_like(g)}
        if scale == "row"
        else {"m": torch.zeros(()), "v": torch.zeros(())}
    )
    for t, G in enumerate(grads, start=1):
        GU = g * G
        if tangent:
            GU = GU - (GU * U).sum(dim=1, keepdim=True) * U
        mom = beta * mom + GU
        D = GU + beta * mom if nesterov else mom
        O = _orth(D, orth)
        if turn_norm == "spectral":
            v0 = (
                torch.ones(n, dtype=W.dtype)
                if mutate == "start_ones"
                else U.mean(dim=0)
            )
            vO = O.mean(dim=0) if mutate == "O_from_its_mean_row" else v0
            k = K_POWER - 1 if mutate == "four_iterations" else K_POWER
            if sigma == "exact":
                su, so = (torch.linalg.matrix_norm(X, ord=2) for X in (U, O))
            else:
                su, so = _power_sigma(U, v0, k), _power_sigma(O, vO, k)
            O_turn = O * (su if mutate == "no_sigma_O" else su / so)
        else:
            O_turn = O * U.norm() / O.norm()
        V = U - lr * eta(t) * O_turn
        U_next = V / V.norm(dim=1, keepdim=True)
        gam = (G * U).sum(dim=1, keepdim=True)
        if scale == "row":
            g = g - lr * mag_lr * _adam(adam, gam, t)
        elif scale == "unit":
            a = a - lr * mag_lr * _adam(adam, (q * gam).sum(), t)
            g = a * q
        U = U_next
        out.append(g * U)
    return out


def _map(
    turn_norm="spectral",
    scale="row",
    tangent=False,
    nesterov=True,
    orth=0.5,
    rule="adam",
    **kw,
):
    comp = _Identity() if orth == "identity" else _Polar(orth)
    extra = {} if scale == "none" else {"magnitude_lr": MAG_LR, "magnitude_rule": rule}
    return AngularMuownMap(
        Rule((), comp, scale="one"),
        nesterov=nesterov,
        warmup=0,
        c=0.0,
        eta=ETA,
        magnitude_shape=scale,
        sphere="row",
        tangent=tangent,
        native_norm=False,
        radius=M_START,
        turn_norm=turn_norm,
        **extra,
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


def _max_diff(xs, ys):
    return max(float((x - y).abs().max()) for x, y in zip(xs, ys))


# ---------------------------------------------------------------- the definition
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize(
    "scale,tangent,nesterov",
    [
        ("row", False, True),
        ("row", False, False),
        ("row", True, True),
        ("none", False, True),
        ("unit", False, True),
    ],
)
@pytest.mark.parametrize("orth", [0.5, "identity"])
def test_spectral_is_its_definition_step_after_step(
    m, n, scale, tangent, nesterov, orth
):
    W, grads = _data(m, n, seed=m + 3 * n)
    ref = _reference(
        W, grads, scale=scale, tangent=tangent, nesterov=nesterov, orth=orth
    )
    got, _ = _run(
        _map(scale=scale, tangent=tangent, nesterov=nesterov, orth=orth), W, grads
    )
    assert _max_diff(ref, got) < 1e-10


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
def test_the_frobenius_reference_is_the_live_frobenius_path(m, n):
    """The same reference with the Frobenius rescale is F (so the spectral one differs in that line only)."""
    W, grads = _data(m, n, seed=m + 5 * n)
    got, _ = _run(_map(turn_norm="frobenius"), W, grads)
    assert _max_diff(_reference(W, grads, turn_norm="frobenius"), got) < 1e-10


@pytest.mark.parametrize(
    "mutate,orth",
    [
        ("start_ones", 0.5),
        ("four_iterations", 0.5),
        ("no_sigma_O", 0.5),
        ("O_from_its_mean_row", "identity"),
    ],
)
def test_the_comparison_sees_each_plausible_mistake(mutate, orth):
    W, grads = _data(12, 8, seed=3)
    good = _reference(W, grads, orth=orth)
    assert _max_diff(good, _reference(W, grads, orth=orth, mutate=mutate)) > 1e-6
    assert (
        _max_diff(good, _reference(W, grads, orth=orth, turn_norm="frobenius")) > 1e-4
    )


# ---------------------------------------------------------------- corners and sizes
def _orthonormal_rows(m, n, seed):
    """Wide / square: orthonormal rows. Tall: [Q; Q; ...], unit rows with U^T U = (m / n) I."""
    gen = torch.Generator().manual_seed(seed)
    if m <= n:
        return torch.linalg.qr(torch.randn(n, m, generator=gen, dtype=torch.float64))[
            0
        ].T.contiguous()
    Q = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))[0]
    return Q.repeat(m // n, 1)


@pytest.mark.parametrize("m,n", [(8, 12), (8, 8), (16, 8), (24, 8)])
@pytest.mark.parametrize("orth", [1.0, 0.5])
def test_orthonormal_rows_spectral_equals_frobenius(m, n, orth):
    """sigma(U) = s_mn = ||U||_F / sqrt(min(m, n)) and an exact polar factor has sigma(O) = ||O||_F /
    sqrt(min(m, n)): the two rescales are one number, so the first step is the same."""
    U = _orthonormal_rows(m, n, seed=m + n)
    W = U * torch.logspace(-0.2, 0.2, m, dtype=torch.float64)[:, None]
    G = 0.1 * torch.randn(
        m, n, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    state = W.new_zeros(1, *_map().state_shape(m, n))
    a = _map(orth=orth).run(G[None], W[None], state, StepCtx(LR, 0.0, 1))
    b = _map(turn_norm="frobenius", orth=orth).run(
        G[None], W[None], state, StepCtx(LR, 0.0, 1)
    )
    assert float((a.weight - b.weight).abs().max()) < 1e-12
    assert float((a.state - b.state).abs().max()) < 1e-12


def _step_size(cm, W, G):
    """One step from an empty state; the code's dU = -s O recovered row by row from U+ (U+ is
    parallel to U - s O): s = (a - r) / (b - r a), a = <U_i, O_i>, b = |O_i|^2, r = <U+_i, O_i> /
    <U+_i, U_i>. Returns (s, the per-row spread of s, U, O)."""
    m, n = W.shape
    res = cm.run(
        G[None], W[None], W.new_zeros(1, *cm.state_shape(m, n)), StepCtx(LR, 0.0, 1)
    )
    U = W / W.norm(dim=1, keepdim=True)
    g = M_START * W.norm(dim=1, keepdim=True)
    O = cm.direction.run_whole(((1 + cm.beta) * g * G)[None]).update[0]
    Un = res.weight[0] / res.state[0, :, n : n + 1]
    a_, b_ = (U * O).sum(dim=1), (O * O).sum(dim=1)
    r = (Un * O).sum(dim=1) / (Un * U).sum(dim=1)
    s = (a_ - r) / (b_ - r * a_)
    return s.median(), float((s / s.median() - 1).abs().max()), U, O


@pytest.mark.parametrize("m,n", [(1024, 512), (512, 1024), (2048, 512), (512, 2048)])
@pytest.mark.parametrize("amp", [0.08, 0.12])
def test_a_planted_spike_turns_by_turn_sigma_max(m, n, amp):
    """||dU||_2 = turn sigma_max(U) to 1e-3 (theory 7.4 test 3), sigma_max(U) / s_mn in 2.1-3.0 (the
    measured range of most units); F's step on the same unit is that ratio smaller."""
    U = _spiky(m, n, amp, seed=m + n)
    W = U * (
        1
        + 0.02
        * torch.randn(
            m, 1, generator=torch.Generator().manual_seed(2), dtype=torch.float64
        )
    )
    G = 0.1 * torch.randn(
        m, n, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    turn = LR * ETA(1)
    s, spread, U, O = _step_size(_map(orth=1.0), W, G)
    sU, sO = (torch.linalg.matrix_norm(X, ord=2) for X in (U, O))
    smn = math.sqrt(max(1.0, m / n))
    assert 2.0 < float(sU) / smn < 3.2
    assert spread < 1e-9
    assert abs(float(s * sO / (turn * sU)) - 1) < 1e-3
    s_f, _, _, _ = _step_size(_map(turn_norm="frobenius", orth=1.0), W, G)
    assert abs(float(s / s_f) / (float(sU) / smn) - 1) < 1e-3


@pytest.mark.parametrize("m,n", [(1024, 512), (512, 1024), (2048, 512), (512, 2048)])
def test_the_power_iteration_on_a_spike_is_sigma_max(m, n):
    """k iterations from the mean row against torch.linalg.matrix_norm(ord=2): 1e-3 on a same-sign
    spike at sigma / s_mn 2.1-3 (theory 7.4 test 4), and on a random-sign spike at 2.9."""
    from torchtitan.optimizers.disco.presets import _SPECTRAL_POWER_ITERS

    assert _SPECTRAL_POWER_ITERS == K_POWER
    for amp, mixed in ((0.08, False), (0.12, False), (0.12, True)):
        U = _spiky(m, n, amp, seed=7 + m, mixed=mixed)
        est = top_pair_warm(U, U.mean(dim=-2), _SPECTRAL_POWER_ITERS)[0]
        assert abs(float(est / torch.linalg.matrix_norm(U, ord=2)) - 1) < 1e-3, (
            amp,
            mixed,
        )


@pytest.mark.parametrize("m,n", [(1024, 512), (512, 2048)])
def test_without_a_spike_the_estimate_is_a_lower_bound(m, n):
    """Random rows (sigma2 / sigma1 > 0.99): k iterations reach sigma_max only to a few % (measured), from
    below, so the step is at most turn sigma_max(U) / sigma(O)'s estimate."""
    U = _spiky(m, n, 0.0, seed=11)
    sU = torch.linalg.matrix_norm(U, ord=2)
    est = top_pair_warm(U, U.mean(dim=-2), K_POWER)[0]
    assert float(est) <= float(sU) * (1 + 1e-12)
    assert float(est / sU) > 0.9


def test_each_unit_of_a_stack_is_its_own():
    """sigma per unit: a stack of a spiky, an orthonormal-row and a random unit."""
    Ws = [
        _spiky(16, 8, 0.3, seed=1),
        _orthonormal_rows(16, 8, seed=2),
        _data(16, 8, seed=3, steps=0)[0],
    ]
    gen = torch.Generator().manual_seed(4)
    grads = [
        [0.1 * torch.randn(16, 8, generator=gen, dtype=torch.float64) for _ in range(4)]
        for _ in Ws
    ]
    cm = _map()
    w, state = torch.stack(Ws), Ws[0].new_zeros(3, *cm.state_shape(16, 8))
    for t in range(4):
        res = cm.run(
            torch.stack([gs[t] for gs in grads]), w, state, StepCtx(LR, 0.0, t + 1)
        )
        w, state = res.weight, res.state
    for k, W in enumerate(Ws):
        assert float((w[k] - _run(cm, W, grads[k])[0][-1]).abs().max()) < 1e-12, k


def test_a_unit_without_a_direction_does_not_turn():
    """G = 0 and an orthogonaliser that maps 0 to 0: sigma(O) = 0 gives no step (as F's ||O||_F
    floor), finite, the rows unchanged."""
    cm = rule_of(
        {
            "rule": "angular_muown",
            "args": {
                "compute": "newtonschulz5",
                "scale": "one",
                "warmup": 0,
                "c": 0.0,
                "native_norm": False,
                "tangent": False,
                "radius": M_START,
                "magnitude_lr": MAG_LR,
                "turn_norm": "spectral",
            },
        }
    ).compute
    W, grads = _data(12, 8, seed=5, steps=1)
    G = torch.stack([grads[0], torch.zeros_like(grads[0])])
    w = torch.stack([W, W])
    res = cm.run(G, w, w.new_zeros(2, *cm.state_shape(12, 8)), StepCtx(LR, 0.0, 1))
    assert torch.isfinite(res.weight).all() and torch.isfinite(res.state).all()
    U = W / W.norm(dim=1, keepdim=True)
    assert float((res.weight[1] / res.state[1, :, 8:9] - U).abs().max()) < 1e-15
    assert float((res.weight[0] / res.state[0, :, 8:9] - U).abs().max()) > 1e-4


@pytest.mark.parametrize(
    "rule", ["sign", "log_rms", "rel_rms", "induced", "cosine", "cosine_raw"]
)
def test_the_spectral_switch_moves_the_turn_only(rule):
    """One step from the same state: each length rule's lengths are bitwise those of its Frobenius run,
    and its unit rows are those of adam-spectral (to the last bits of the final multiply)."""
    W, grads = _data(12, 8, seed=17, steps=1)
    kw = {
        "magnitude_lr": 0.2
        if rule in ("log_rms", "rel_rms")
        else 1.0
        if rule in ("induced", "cosine", "cosine_raw")
        else MAG_LR,
        "magnitude_rule": rule,
    }

    def build(tn, **over):
        return AngularMuownMap(
            Rule((), _Polar(0.5), scale="one"),
            warmup=0,
            c=0.0,
            eta=ETA,
            sphere="row",
            tangent=False,
            native_norm=False,
            radius=M_START,
            turn_norm=tn,
            **{**kw, **over},
        )

    s, f = build("spectral"), build("frobenius")
    adam = build("spectral", magnitude_rule="adam", magnitude_lr=MAG_LR)
    state = W.new_zeros(1, *s.state_shape(12, 8))
    a, b = (
        cm.run(grads[0][None], W[None], state, StepCtx(LR, 0.0, 1)) for cm in (s, f)
    )
    c = adam.run(
        grads[0][None],
        W[None],
        W.new_zeros(1, *adam.state_shape(12, 8)),
        StepCtx(LR, 0.0, 1),
    )
    assert torch.equal(a.state[..., 8:], b.state[..., 8:])
    ua, uc = a.weight[0] / a.state[0, :, 8:9], c.weight[0] / c.state[0, :, 8:9]
    assert float((ua - uc).abs().max()) < 4e-16
    assert float((a.weight - b.weight).abs().max()) > 1e-6


# ---------------------------------------------------------------- frobenius unchanged
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
    *(
        ({**_F, "sphere": "row", "tangent": False}, r)
        for r in ("polar", "polar_t", "v3b", "cosine", "cosine_raw")
    ),
    *(
        ({**_F, "sphere": s, "magnitude_shape": "none", "tangent": t}, "adam")
        for s, t in (("row", False), ("row", True), ("matrix", True))
    ),
]


@pytest.mark.parametrize("args,rule", _CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute", ["svd", "newtonschulz5"])
def test_frobenius_is_the_517ab8b2_rule_bitwise(args, rule, dtype, compute):
    """Every rule at the default turn_norm against the frozen wt-s4 class: stacks of 3 units with a zero
    row, a lost state at step 4, 8 steps; weights and state torch.equal."""
    extra = {}
    if args.get("magnitude_shape") != "none":
        extra["magnitude_rule"] = rule
        lr_mag = {"adam": 0.3, "sign": 0.3, "log_rms": 0.3, "rel_rms": 0.2}.get(rule)
        if lr_mag is not None:
            extra["magnitude_lr"] = lr_mag
    live = rule_of(
        {"rule": "angular_muown", "args": {"compute": compute, **args, **extra}}
    ).compute
    assert live.turn_norm == "frobenius"
    c = live
    old = S4AngularMuownMap(
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


# ---------------------------------------------------------------- settings
def test_the_preset_builds_it():
    cm = rule_of(
        {
            "rule": "angular_muown",
            "args": {
                "compute": "svd",
                **_F,
                "tangent": False,
                "magnitude_lr": MAG_LR,
                "turn_norm": "spectral",
            },
        }
    ).compute
    assert (cm.turn_norm, cm.sphere, cm.native_norm, cm.state_shape(12, 8)) == (
        "spectral",
        "row",
        False,
        (12, 11),
    )


@pytest.mark.parametrize(
    "bad,match",
    [
        (dict(turn_norm="operator"), "turn_norm is frobenius or spectral"),
        (
            dict(native_norm=True, scale="image_spectral", warmup=100, c=1e-3),
            "turn_norm spectral rescales",
        ),
        (dict(sphere="matrix"), "turn_norm spectral rescales"),
        *(
            (dict(magnitude_rule=r, magnitude_lr=1.0), "turn_norm spectral rescales")
            for r in ("polar", "polar_t", "v3b")
        ),
    ],
)
def test_settings_the_spectral_turn_does_not_use_are_refused(bad, match):
    args = {"compute": "svd", **_F, "tangent": False, "turn_norm": "spectral", **bad}
    with pytest.raises(ValueError, match=match):
        rule_of({"rule": "angular_muown", "args": args})


def test_the_corner_refuses_a_spectral_turn():
    with pytest.raises(ValueError, match="turn_norm"):
        rule_of(
            {
                "rule": "angular_muown",
                "args": {
                    "compute": "svd",
                    **_F,
                    "sphere": "matrix",
                    "magnitude_shape": "none",
                    "tangent": False,
                    "turn_norm": "spectral",
                },
            }
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


def _disco(w, turn_norm="spectral"):
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
            "magnitude_lr": MAG_LR,
            "turn_norm": turn_norm,
        },
        param_layouts={"w": ParamLayout("matrix")},
    )


def _steps(opt, w, grads):
    for G in grads:
        w.grad = G.float().clone()
        opt.step()


def test_a_disco_step_is_the_rule(one_rank):
    W, grads = _data(12, 8, seed=9, steps=4)
    w = nn.Parameter(W.float().clone())
    _steps(_disco(w), w, grads)
    ref = _reference(W, grads, orth=1.0)[-1]
    assert float((w.detach().double() - ref).abs().max()) < 1e-5
    assert (
        float(
            (
                w.detach().double()
                - _reference(W, grads, orth=1.0, turn_norm="frobenius")[-1]
            )
            .abs()
            .max()
        )
        > 1e-3
    )


def test_a_resumed_run_is_the_continuous_one(one_rank):
    """No new state: rule_state is F's [M | g | m_g | v_g]; saved and loaded, the next steps are bitwise
    the same; a fresh state is not."""
    W, grads = _data(12, 8, seed=10, steps=5)
    wa = nn.Parameter(W.float().clone())
    a = _disco(wa)
    _steps(a, wa, grads[:3])
    saved = wa.detach().clone()
    sd = copy.deepcopy(a.state_dict())
    assert sd["state"][0]["rule_state"].shape == (1, 12, 11)
    _steps(a, wa, grads[3:])
    wb, wc = nn.Parameter(saved.clone()), nn.Parameter(saved.clone())
    b, c = _disco(wb), _disco(wc)
    b.load_state_dict(sd)
    _steps(b, wb, grads[3:])
    _steps(c, wc, grads[3:])
    assert torch.equal(wa.detach(), wb.detach())
    assert not torch.equal(wa.detach(), wc.detach())
