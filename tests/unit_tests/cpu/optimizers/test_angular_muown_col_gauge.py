# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""angular_muown `col_gauge` (ps2n C1), from the definition. W = Diag(g) U Diag(c). `col_gauge`
projects the column gains onto the unit-RMS sphere after every column step: c <- c / RMS_j(c_j)
per unit (RMS = sqrt(mean_j c_j^2)); g, its state, U and every momentum are not touched, and W is
formed from the normalised c. The matrix's overall scale is then g's alone (the length rule),
exactly as without columns.

(i) The reference runs the class WITHOUT col_gauge one step at a time and applies exactly that
rescale (c <- c / s, W <- W / s, s = RMS(c), g unchanged) to the carried state, so a bug in the
class cannot hide in a shared helper. (ii) The mechanism: under a gradient that is a pure scale
direction (G = -lam W, also with a random scalar lam_t), the gauged class's W is the no-column
class's W at every step; under elementwise noise its Frobenius norm stays at the no-column level
while the free columns inflate it. (iii) gauge off = the class before, bitwise.
Also the tied column rate: with gain_lr tied, col_gain_lr_mult k is k x the TURN's rate
(lr_t kappa(t) eta(t)), not k x lr_t (the fixed-rate meaning of md2); all in float64."""

import os
import sys

import pytest
import torch

from torchtitan.optimizers.disco.presets import AngularMuownMap
from torchtitan.optimizers.disco.rules import Rule, rule_of, StepCtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_angular_muown_col_scale as base  # noqa: E402
import test_angular_muown_gain_switches as gs  # noqa: E402
from test_angular_muown_col_scale import one_rank  # noqa: E402,F401

LR, LRS = gs.LR, gs.LRS
M, N = 12, 8


def _gmap(
    sphere, tangent, param, lr_mode, rule, gauge, col_param=None, col_k=None, mult=0.7
):
    """gs._map with the one extra field (gs._map is the md2 test's builder and has no col_gauge argument)."""
    cm = gs._map(
        sphere,
        tangent,
        True,
        param,
        lr_mode,
        mult,
        rule,
        (0.9, 0.95),
        col_param=col_param,
        col_k=col_k,
    )
    if not gauge:
        return cm
    fields = {
        f: getattr(cm, f)
        for f in (
            "beta",
            "nesterov",
            "betas",
            "eps",
            "warmup",
            "c",
            "p",
            "magnitude_lr",
            "eta",
            "magnitude_shape",
            "magnitude_rule",
            "sphere",
            "tangent",
            "native_norm",
            "radius",
            "turn_norm",
            "frozen_units",
            "col_scale",
            "gain_param",
            "gain_lr",
            "gain_lr_mult",
            "col_gain_param",
            "col_gain_lr_mult",
        )
    }
    return AngularMuownMap(
        Rule((), base._Polar(0.5), scale="one"), **fields, col_gauge=True
    )


def _rms_c(st):
    return st[M, :N].square().mean().sqrt()


def _stepwise_rescaled(cm, W, grads, lrs):
    """The ungauged class, one step at a time, c and W rescaled after each step by the definition (g unchanged)."""
    w, state = W[None].double(), torch.zeros(
        1, *cm.state_shape(M, N), dtype=torch.float64
    )
    out = []
    for t, (G, lr) in enumerate(zip(grads, lrs), start=1):
        res = cm.run(G[None].double(), w, state, StepCtx(lr, 0.0, t))
        w, state = res.weight, res.state.clone()
        s = state[0, M, :N].square().mean().sqrt()
        state[0, M, :N] = state[0, M, :N] / s
        w = w / s
        out.append((w[0], state[0], s))
    return out


GRID = [
    (s, t, p, lm, r)
    for (s, t) in base.CELLS
    for p in ("raw", "softplus")
    for lm in ("fixed", "tied")
    for r in ("adam", "sign")
]


@pytest.mark.parametrize("sphere,tangent,param,lr_mode,rule", GRID)
def test_the_gauge_is_the_rms_projection_of_the_ungauged_step(
    sphere, tangent, param, lr_mode, rule
):
    W, grads = base._data(M, N, seed=61)
    ref = _stepwise_rescaled(
        _gmap(sphere, tangent, param, lr_mode, rule, False), W, grads, LRS
    )
    got = gs._run_lrs(_gmap(sphere, tangent, param, lr_mode, rule, True), W, grads, LRS)
    tol = base.TOL[sphere]
    for t, ((w, st), (rw, rst, s)) in enumerate(zip(got, ref), start=1):
        assert float((w - rw).abs().max()) < tol * float(
            rw.abs().max()
        ), f"W at step {t}"
        assert float((st[M, :N] - rst[M, :N]).abs().max()) < tol, f"c at step {t}"
        assert float((st[:M, -3] - rst[:M, -3]).abs().max()) < tol * float(
            rst[:M, -3].abs().max()
        ), f"g (unchanged) at step {t}"
        assert (
            float((st[:M, :-3] - rst[:M, :-3]).abs().max()) < tol
        ), f"M (U's momentum) at step {t}"
        assert (
            float((st[M + 1 :, :N] - rst[M + 1 :, :N]).abs().max()) < tol
        ), f"m_c, v_c at step {t}"
        assert abs(float(_rms_c(st)) - 1.0) < 1e-12, f"RMS(c) at step {t}"
    # power: the projection changed c past the tolerance on at least one step
    free = gs._run_lrs(
        _gmap(sphere, tangent, param, lr_mode, rule, False), W, grads, LRS
    )
    assert (
        max(float((a[1][M, :N] - b[1][M, :N]).abs().max()) for a, b in zip(got, free))
        > 10 * tol
    )


@pytest.mark.parametrize("rule", ["adam", "sign"])
def test_without_the_gauge_rms_c_drifts_and_with_it_stays_one_with_g_untouched(rule):
    """Every column's gradient has the same sign (G = W elementwise): the ungauged c all move one
    way and RMS(c) drifts; the gauge pins it at 1, leaves g alone, and W is the free W / RMS(c)."""
    W, _ = base._data(M, N, seed=63)
    grads = [0.1 * W.clone() for _ in range(8)]
    lrs = [LR] * 8
    free = gs._run_lrs(_gmap("row", False, "raw", "fixed", rule, False), W, grads, lrs)
    gauged = gs._run_lrs(_gmap("row", False, "raw", "fixed", rule, True), W, grads, lrs)
    assert abs(float(_rms_c(free[-1][1])) - 1.0) > 0.02
    assert all(abs(float(_rms_c(st)) - 1.0) < 1e-12 for _, st in gauged)
    s1 = float(_rms_c(free[0][1]))
    assert abs(s1 - 1.0) > 0.01
    assert float((free[0][0] / s1 - gauged[0][0]).abs().max()) < 1e-12 * float(
        free[0][0].abs().max()
    )  # W = W_free / RMS(c)
    assert (
        float((gauged[0][1][:M, -3] - free[0][1][:M, -3]).abs().max()) == 0.0
    )  # g untouched


def _onerate_form(col, gauge=False):
    """The OneRate form (sign rule, row sphere, tangent off, radius 4, no angle schedule), with or without columns."""
    kw = dict(col_scale=True) if col else {}
    if gauge:
        kw["col_gauge"] = True
    return AngularMuownMap(
        Rule((), base._Polar(0.5), scale="one"),
        warmup=0,
        c=0.0,
        eta=None,
        magnitude_shape="row",
        magnitude_rule="sign",
        sphere="row",
        tangent=False,
        native_norm=False,
        radius=4.0,
        magnitude_lr=1.0,
        **kw,
    )


def _closed_loop(cm, W0, T, lr, grad):
    """T steps with the gradient grad(t, w) of the CURRENT weight (the same noise stream for every variant)."""
    m, n = W0.shape
    w, st = W0[None].double().clone(), torch.zeros(
        1, *cm.state_shape(m, n), dtype=torch.float64
    )
    ws = []
    for t in range(1, T + 1):
        res = cm.run(grad(t, w), w, st, StepCtx(lr, 0.0, t))
        w, st = res.weight, res.state.clone()
        ws.append(w[0].clone())
    return ws, st[0]


def _w0(m=24, n=16, seed=5):
    torch.manual_seed(seed)
    return torch.randn(m, n, dtype=torch.float64) / n**0.5


@pytest.mark.parametrize("noisy_scale", [False, True])
def test_a_pure_scale_gradient_gives_the_no_column_weights_at_every_step(noisy_scale):
    """G = -lam_t W with lam_t = 0.1 (or 0.1 + 0.05 xi_t, xi_t random, so the sign varies): every column gets the same
    step, RMS-normalising c removes it, and W is the no-column class's W (the scale is g's alone). The geomean
    gauge (c / gm, g gm, W unchanged) fails this by orders of magnitude."""
    W0, T = _w0(), 300
    xi = torch.randn(
        T + 1, generator=torch.Generator().manual_seed(2), dtype=torch.float64
    )
    lam = (lambda t: 0.1 + 0.05 * float(xi[t])) if noisy_scale else (lambda t: 0.1)
    grad = lambda t, w: -lam(t) * w  # noqa: E731
    ref, _ = _closed_loop(_onerate_form(False), W0, T, 0.044, grad)
    got, st = _closed_loop(_onerate_form(True, True), W0, T, 0.044, grad)
    for t, (a, b) in enumerate(zip(got, ref), start=1):
        assert float((a - b).norm()) <= 1e-10 * float(
            b.norm()
        ), f"W vs no-column W at step {t}"
    assert (
        abs(float(st[W0.shape[0], : W0.shape[1]].square().mean().sqrt()) - 1.0) < 1e-12
    )
    # power: free columns (b1) leave the no-column W at the first step
    free, _ = _closed_loop(_onerate_form(True), W0, T, 0.044, grad)
    assert float((free[-1] - ref[-1]).norm()) > 1e-3 * float(ref[-1].norm())


@pytest.mark.parametrize("lam,sig,T", [(0.0, 0.01, 600), (0.002, 0.01, 600)])
def test_under_elementwise_noise_the_frobenius_scale_stays_at_the_no_column_level(
    lam, sig, T
):
    """G = -lam W + sig N_t (N_t fresh Gaussian, the same stream in every variant). The free columns random-walk (about
    half of c turns negative) and inflate ||W||_F; the RMS projection keeps it at the no-column level (g's rule alone)."""
    W0 = _w0()

    def grad(variant):
        g = torch.Generator().manual_seed(11)
        return lambda t, w: -lam * w + sig * torch.randn(
            w.shape, generator=g, dtype=torch.float64
        )

    ref, _ = _closed_loop(_onerate_form(False), W0, T, 0.044, grad(0))
    got, st = _closed_loop(_onerate_form(True, True), W0, T, 0.044, grad(1))
    free, _ = _closed_loop(_onerate_form(True), W0, T, 0.044, grad(2))
    r_got, r_free = float(got[-1].norm() / ref[-1].norm()), float(
        free[-1].norm() / ref[-1].norm()
    )
    assert (
        0.5 < r_got < 1.4
    ), f"gauged / no-column ||W||_F = {r_got}"  # about 0.77 and 1.00 here; the free columns: 3.4 and 27
    assert r_free > 1.5, f"power: free columns / no-column = {r_free}"
    assert (
        abs(float(st[W0.shape[0], : W0.shape[1]].square().mean().sqrt()) - 1.0) < 1e-12
    )


def test_gauge_off_is_the_class_before_bitwise():
    W, grads = base._data(M, N, seed=67)
    for sphere, tangent in base.CELLS:
        a = gs._run_lrs(
            gs._map(sphere, tangent, True, "raw", "fixed", 0.7, "adam", (0.9, 0.95)),
            W,
            grads,
            LRS,
            dtype=torch.float32,
        )
        b = gs._run_lrs(
            gs._map(
                sphere,
                tangent,
                True,
                "raw",
                "fixed",
                0.7,
                "adam",
                (0.9, 0.95),
                cls=gs.PreColAngularMuownMap,
            ),
            W,
            grads,
            LRS,
            dtype=torch.float32,
        )
        for (wa, sa), (wb, sb) in zip(a, b):
            assert torch.equal(wa, wb) and torch.equal(sa, sb)


@pytest.mark.parametrize("rule", ["adam", "sign"])
@pytest.mark.parametrize("param", ["raw", "softplus"])
def test_tied_columns_run_at_k_times_the_turn(rule, param):
    """gain_lr tied (rows: turn x gain_lr_mult 0.7), col_gain_lr_mult k = 1.0 (columns: the turn x 1, NOT lr_t, NOT 0.7)."""
    W, grads = base._data(M, N, seed=71)
    cm = gs._map("row", False, True, param, "tied", 0.7, rule, (0.9, 0.95), col_k=1.0)
    got = gs._run_lrs(cm, W, grads, LRS)
    want = gs._reference(
        W,
        grads,
        sphere="row",
        tangent=False,
        col=True,
        param=param,
        lr_mode="tied",
        mult=0.7,
        rule=rule,
        betas=(0.9, 0.95),
        col_k=1.0,
        lrs=LRS,
    )
    for t, ((w, st), (rw, rg, rc)) in enumerate(zip(got, want), start=1):
        assert float((st[M, :N] - rc).abs().max()) < 1e-10, f"c at step {t}"
        assert float((st[:M, -3] - rg).abs().max()) < 1e-10 * float(
            rg.abs().max()
        ), f"g at step {t}"
    # power: the md2 meaning (k x lr_t, no eta) would give another c
    md2 = gs._reference(
        W,
        grads,
        sphere="row",
        tangent=False,
        col=True,
        param=param,
        lr_mode="fixed",
        mult=0.7,
        rule=rule,
        betas=(0.9, 0.95),
        col_k=1.0,
        lrs=LRS,
    )
    assert float((want[-1][2] - md2[-1][2]).abs().max()) > 1e-4


def test_tied_columns_with_k_1_and_mult_1_are_the_rows_rate_bitwise():
    W, grads = base._data(M, N, seed=73)
    a = gs._run_lrs(
        gs._map("row", False, True, "raw", "tied", 1.0, "sign", (0.9, 0.95), col_k=1.0),
        W,
        grads,
        LRS,
        dtype=torch.float32,
    )
    b = gs._run_lrs(
        gs._map("row", False, True, "raw", "tied", 1.0, "sign", (0.9, 0.95)),
        W,
        grads,
        LRS,
        dtype=torch.float32,
    )
    for (wa, sa), (wb, sb) in zip(a, b):
        assert torch.equal(wa, wb) and torch.equal(sa, sb)


def test_col_gauge_needs_col_scale_and_the_hyperball_corner_refuses_it():
    kw = dict(
        warmup=0,
        c=0.0,
        magnitude_shape="row",
        sphere="matrix",
        tangent=False,
        native_norm=False,
        radius=4.0,
    )
    with pytest.raises(ValueError, match="col_gauge"):
        AngularMuownMap(Rule((), base._Polar(0.5), scale="one"), col_gauge=True, **kw)
    AngularMuownMap(
        Rule((), base._Polar(0.5), scale="one"), col_gauge=True, col_scale=True, **kw
    )
    args = dict(
        compute="svd",
        scale="one",
        warmup=0,
        c=0.0,
        native_norm=False,
        sphere="matrix",
        tangent=False,
        radius=4.0,
        col_scale=True,
        magnitude_lr=10.24,
    )
    assert rule_of({"rule": "angular_muown", "args": args}).compute.col_gauge is False
    assert (
        rule_of(
            {"rule": "angular_muown", "args": {**args, "col_gauge": True}}
        ).compute.col_gauge
        is True
    )
    corner = dict(
        compute="svd",
        scale="one",
        warmup=0,
        c=0.0,
        native_norm=False,
        sphere="matrix",
        tangent=False,
        magnitude_shape="none",
    )
    with pytest.raises(ValueError, match="col_gauge"):
        rule_of({"rule": "angular_muown", "args": {**corner, "col_gauge": True}})
