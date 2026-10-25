# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""angular_muown's learned-gain switches, each default-off:

  gain_param  "raw" (today: g itself, sign kept, |g| >= 1e-7) | "softplus":
              g = softplus(r), r starts at softplus^-1 of today's start value,
              the gain rule (adam / sign) steps r on dL/dr = dL/dg sigmoid(r);
  gain_lr     "fixed" (today: rate lr_t x magnitude_lr) | "tied": the turn's
              rate lr_t kappa(t) eta(t) x gain_lr_mult;
  betas       the gains' Adam betas (an existing field).

Row and column gains (col_scale) alike, both from the pre-update U and G.
Per side (md2): col_gain_param sets the column gains' form alone (None: the
rows'), col_gain_lr_mult k their rate alone: k x lr_t with no eta(t) when the
gain_lr is fixed, k x the turn (lr_t kappa eta(t); ps2n) when it is tied (None:
the rows' rate). The reference below is written from those equations in
float64; off is bitwise the class before each change."""

import itertools
import math
import os
import sys

import pytest
import torch
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import AngularMuownMap, EtaSchedule
from torchtitan.optimizers.disco.rules import Rule, rule_of, StepCtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_angular_muown_col_scale as base  # noqa: E402
from _angular_muown_md2_pre_col_reference import PreColAngularMuownMap  # noqa: E402
from _angular_muown_ps2_pre_gain_reference import PreGainAngularMuownMap  # noqa: E402
from test_angular_muown_col_scale import one_rank  # noqa: E402,F401

LR, MAG_LR, M_START = base.LR, base.MAG_LR, base.M_START
ETA = EtaSchedule("inverse_poly", start=1, c=0.5)


def _softplus(x):
    return torch.log1p(torch.exp(-x.abs())) + x.clamp_min(0)


def _softplus_inv(y):
    return y + torch.log(-torch.expm1(-y))


def _step(rule, st, x, t, betas):
    b1, b2 = betas
    st["m"] = b1 * st["m"] + (1 - b1) * x
    if rule == "sign":
        return torch.sign(st["m"])
    st["v"] = b2 * st["v"] + (1 - b2) * x * x
    return (st["m"] / (1 - b1**t)) / ((st["v"] / (1 - b2**t)).sqrt() + 1e-8)


def _reference(
    W,
    grads,
    *,
    sphere,
    tangent,
    col,
    param,
    lr_mode,
    mult,
    rule,
    betas,
    col_param=None,
    col_k=None,
    lrs=None,
):
    """W_t = Diag(g) U Diag(c) (c = 1 without col). Start: g = W0's row norms (row
    sphere) or their RMS (matrix sphere), times M from step 1 on; c = 1. Raw: the
    gains are stepped directly, |x| >= 1e-7 with the sign kept. Softplus: g = sp(r),
    c = sp(s), r = sp^-1(g_start), s = sp^-1(1); r, s are stepped on dL/dr =
    dL/dg sp'(r). Rate: lr x MAG_LR (fixed) or lr x eta(t) x mult (tied; kappa = 1
    at c = 0). dL/dg_i = <G_i, U_i c>, dL/dc_j = <G_:j, g U_:j> at the pre-update
    (g, U, c); U turns as in base._reference. Per side: the columns take the form
    col_param (None: param) and the rate lr_t x col_k (None: the rows' rate).
    lrs: lr_t per step (None: LR throughout)."""
    m, n = W.shape
    norms = W.norm(dim=1, keepdim=True)
    start = (
        norms if sphere == "row" else norms.square().mean().sqrt().expand(m, 1).clone()
    )
    U, g, c = W / start, M_START * start, torch.ones(1, n, dtype=W.dtype)
    mom = torch.zeros_like(W)
    sg = {"m": torch.zeros_like(g), "v": torch.zeros_like(g)}
    sc = {"m": torch.zeros_like(c), "v": torch.zeros_like(c)}

    def proj(X):
        if sphere == "row":
            return X - (X * U).sum(dim=1, keepdim=True) * U
        return X - (X * U).sum() / (U * U).sum() * U

    def move(x, grad, st, rate, t, param=param):
        if param == "raw":
            x = x - rate * _step(rule, st, grad, t, betas)
            return torch.copysign(x.abs().clamp_min(1e-7), x)
        r = _softplus_inv(x)
        r = r - rate * _step(rule, st, grad * torch.sigmoid(r), t, betas)
        return _softplus(r)

    out = []
    for t, G in enumerate(grads, start=1):
        X = g * G * c
        if tangent:
            X = proj(X)
        mom = 0.95 * mom + X
        D = X + 0.95 * mom
        O = base._polar(D, 0.5)
        lr = LR if lrs is None else lrs[t - 1]
        V = U - lr * ETA(t) * O * (U.norm() / O.norm())
        U_next = (
            V / V.norm(dim=1, keepdim=True)
            if sphere == "row"
            else V * (U.norm() / V.norm())
        )
        dg = (G * U * c).sum(dim=1, keepdim=True)
        dc = (G * U * g).sum(dim=0, keepdim=True)
        rate = lr * MAG_LR if lr_mode == "fixed" else lr * ETA(t) * mult
        rate_c = (
            rate
            if col_k is None
            else lr * (ETA(t) if lr_mode == "tied" else 1.0) * col_k
        )  # ps2n: tied = k x the turn (kappa = 1), fixed = k x lr_t (md2)
        g_next = move(g, dg, sg, rate, t)
        if col:
            c = move(c, dc, sc, rate_c, t, param if col_param is None else col_param)
        g, U = g_next, U_next
        out.append((g * U * c, g[:, 0].clone(), c[0].clone()))
    return out


def _map(
    sphere,
    tangent,
    col,
    param,
    lr_mode,
    mult,
    rule,
    betas,
    col_param=None,
    col_k=None,
    cls=AngularMuownMap,
):
    kw = dict(col_scale=True) if col else {}
    if col_param is not None:
        kw["col_gain_param"] = col_param
    if col_k is not None:
        kw["col_gain_lr_mult"] = col_k
    if param != "raw":
        kw["gain_param"] = param
    if lr_mode != "fixed":
        kw["gain_lr"] = lr_mode
        kw["gain_lr_mult"] = mult
    mag = {} if lr_mode == "tied" else {"magnitude_lr": MAG_LR}
    return cls(
        Rule((), base._Polar(0.5), scale="one"),
        warmup=0,
        c=0.0,
        eta=ETA,
        betas=betas,
        magnitude_shape="row",
        magnitude_rule=rule,
        sphere=sphere,
        tangent=tangent,
        native_norm=False,
        radius=M_START,
        **mag,
        **kw,
    )


CASES = [
    (sphere, tangent, col, param, lr_mode, rule)
    for (sphere, tangent), col, param, lr_mode, rule in itertools.product(
        base.CELLS,
        (False, True),
        ("raw", "softplus"),
        ("fixed", "tied"),
        ("adam", "sign"),
    )
    if (param, lr_mode) != ("raw", "fixed")
]


@pytest.mark.parametrize("sphere,tangent,col,param,lr_mode,rule", CASES)
def test_the_switches_are_their_equations(sphere, tangent, col, param, lr_mode, rule):
    m, n = 12, 8
    W, grads = base._data(m, n, seed=29)
    betas, mult = (0.9, 0.99), 0.7
    cm = _map(sphere, tangent, col, param, lr_mode, mult, rule, betas)
    got = base._run(cm, W, grads)
    want = _reference(
        W,
        grads,
        sphere=sphere,
        tangent=tangent,
        col=col,
        param=param,
        lr_mode=lr_mode,
        mult=mult,
        rule=rule,
        betas=betas,
    )
    tol = base.TOL[sphere]
    for t, ((w, st), (rw, rg, rc)) in enumerate(zip(got, want), start=1):
        assert float((w - rw).abs().max()) < tol * float(
            rw.abs().max()
        ), f"W at step {t}"
        g = st[:m, n] if not col else st[:m, -3]
        assert float((g - rg).abs().max()) < tol * float(
            rg.abs().max()
        ), f"g at step {t}"
        if col:
            assert float((st[m, :-3] - rc).abs().max()) < tol, f"c at step {t}"
    # power: the switch moves the result past the tolerance
    today = _reference(
        W,
        grads,
        sphere=sphere,
        tangent=tangent,
        col=col,
        param="raw",
        lr_mode="fixed",
        mult=1.0,
        rule=rule,
        betas=betas,
    )
    assert float((want[-1][1] - today[-1][1]).abs().max()) > 10 * tol * float(
        today[-1][1].abs().max()
    )


def test_softplus_starts_at_todays_start_value():
    W, grads = base._data(12, 8, seed=31)
    a = _reference(
        W,
        grads[:1],
        sphere="matrix",
        tangent=False,
        col=True,
        param="softplus",
        lr_mode="fixed",
        mult=1.0,
        rule="adam",
        betas=(0.9, 0.95),
    )
    assert _softplus(
        _softplus_inv(torch.tensor(1.0, dtype=torch.float64))
    ) == pytest.approx(1.0, rel=1e-15)
    # with a zero rate the softplus gains stay at the start values
    cm = AngularMuownMap(
        Rule((), base._Polar(0.5), scale="one"),
        warmup=0,
        c=0.0,
        eta=ETA,
        magnitude_lr=0.0,
        magnitude_shape="row",
        sphere="matrix",
        tangent=False,
        native_norm=False,
        radius=M_START,
        col_scale=True,
        gain_param="softplus",
    )
    ((w, st),) = base._run(cm, W, grads[:1])
    start = M_START * W.norm(dim=1).square().mean().sqrt()
    assert torch.allclose(st[:12, -3], start.expand(12), rtol=1e-12)
    assert torch.allclose(st[12, :-3], torch.ones(8, dtype=torch.float64), rtol=1e-12)
    assert len(a) == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("col", [False, True])
@pytest.mark.parametrize("rule", ["adam", "sign"])
@pytest.mark.parametrize("sphere,tangent", base.CELLS)
def test_off_is_the_class_before_the_change_bitwise(sphere, tangent, rule, col, dtype):
    W, grads = base._data(12, 8, seed=37)
    kw = dict(
        warmup=0,
        c=0.0,
        eta=ETA,
        magnitude_lr=MAG_LR,
        magnitude_shape="row",
        magnitude_rule=rule,
        sphere=sphere,
        tangent=tangent,
        native_norm=False,
        radius=M_START,
        col_scale=col,
    )
    new = AngularMuownMap(Rule((), base._Polar(0.5), scale="one"), **kw)
    old = PreGainAngularMuownMap(Rule((), base._Polar(0.5), scale="one"), **kw)
    explicit = AngularMuownMap(
        Rule((), base._Polar(0.5), scale="one"), gain_param="raw", gain_lr="fixed", **kw
    )
    for (a, sa), (b, sb), (c, sc) in zip(
        *(base._run(x, W, grads, dtype=dtype) for x in (new, old, explicit))
    ):
        assert (
            torch.equal(a, b)
            and torch.equal(sa, sb)
            and torch.equal(a, c)
            and torch.equal(sa, sc)
        )


@pytest.mark.parametrize(
    "bad",
    [
        dict(gain_param="exp"),
        dict(gain_lr="adaptive"),
        dict(gain_lr_mult=2.0),  # only with gain_lr tied
        dict(gain_lr="tied", magnitude_lr=3.0),  # magnitude_lr unused when tied
        dict(gain_lr="tied", gain_lr_mult=-1.0),
        dict(gain_param="softplus", magnitude_rule="log_rms"),
        dict(gain_param="softplus", magnitude_shape="unit"),
        dict(gain_lr="tied", magnitude_shape="none"),
        dict(col_gain_param="softplus"),  # the per-side fields need col_scale
        dict(col_gain_lr_mult=1.0),
        dict(col_scale=True, col_gain_param="exp"),
        dict(col_scale=True, col_gain_lr_mult=-1.0),
        dict(col_scale=True, col_gain_lr_mult=float("nan")),
    ],
)
def test_bad_settings_are_refused(bad):
    kw = dict(
        warmup=0,
        c=0.0,
        magnitude_shape="row",
        sphere="matrix",
        tangent=False,
        native_norm=False,
        radius=M_START,
    )
    kw.update(bad)
    with pytest.raises(ValueError):
        AngularMuownMap(Rule((), base._Polar(0.5), scale="one"), **kw)


def test_the_preset_passes_the_switches():
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
        gain_param="softplus",
        gain_lr="tied",
        gain_lr_mult=0.5,
        betas=(0.9, 0.99),
    )
    cm = rule_of({"rule": "angular_muown", "args": args}).compute
    assert (cm.gain_param, cm.gain_lr, cm.gain_lr_mult, cm.betas) == (
        "softplus",
        "tied",
        0.5,
        (0.9, 0.99),
    )
    assert math.isclose(cm.magnitude_lr, 1.0)


# ---------------------------------------------------------------- md2: the column gains alone
LRS = [LR * f for f in (1.0, 1.0, 0.9, 0.7, 0.4, 0.15)]  # lr_t of a decaying schedule


def _run_lrs(cm, W, grads, lrs, dtype=torch.float64):
    m, n = W.shape
    w, state, out = (
        W[None].to(dtype),
        torch.zeros(1, *cm.state_shape(m, n), dtype=dtype),
        [],
    )
    for t, (G, lr) in enumerate(zip(grads, lrs), start=1):
        res = cm.run(G[None].to(dtype), w, state, StepCtx(lr, 0.0, t))
        w, state = res.weight, res.state
        out.append((w[0], state[0]))
    return out


# (row param, row lr mode, column param, column k): the design's cells first
COL_CASES = [
    ("raw", "fixed", "softplus", 0.25),
    ("raw", "fixed", "softplus", 4.0),
    ("raw", "fixed", "raw", 4.0),
    ("raw", "fixed", "softplus", None),
    ("raw", "fixed", None, 1.0),
    ("softplus", "tied", "raw", None),
    ("softplus", "tied", None, 1.0),
]


@pytest.mark.parametrize("row_param,lr_mode,col_param,col_k", COL_CASES)
@pytest.mark.parametrize("rule", ["adam", "sign"])
@pytest.mark.parametrize("sphere,tangent", base.CELLS)
def test_the_column_fields_are_their_equations(
    sphere, tangent, rule, row_param, lr_mode, col_param, col_k
):
    m, n = 12, 8
    W, grads = base._data(m, n, seed=41)
    betas, mult = (0.9, 0.99), 0.7
    cm = _map(
        sphere,
        tangent,
        True,
        row_param,
        lr_mode,
        mult,
        rule,
        betas,
        col_param=col_param,
        col_k=col_k,
    )
    got = _run_lrs(cm, W, grads, LRS)
    ref = dict(
        sphere=sphere,
        tangent=tangent,
        col=True,
        param=row_param,
        lr_mode=lr_mode,
        mult=mult,
        rule=rule,
        betas=betas,
        lrs=LRS,
    )
    want = _reference(W, grads, col_param=col_param, col_k=col_k, **ref)
    tol = base.TOL[sphere]
    for t, ((w, st), (rw, rg, rc)) in enumerate(zip(got, want), start=1):
        assert float((w - rw).abs().max()) < tol * float(
            rw.abs().max()
        ), f"W at step {t}"
        assert float((st[:m, -3] - rg).abs().max()) < tol * float(
            rg.abs().max()
        ), f"g at step {t}"
        assert float((st[m, :-3] - rc).abs().max()) < tol * float(
            rc.abs().max()
        ), f"c at step {t}"
    # power: the per-side setting moves c past the tolerance against the columns following the rows
    follow = _reference(W, grads, **ref)
    assert float((want[-1][2] - follow[-1][2]).abs().max()) > 10 * tol * float(
        follow[-1][2].abs().max()
    )


@pytest.mark.parametrize("row_param,lr_mode,col_param,col_k", COL_CASES)
@pytest.mark.parametrize("rule", ["adam", "sign"])
@pytest.mark.parametrize("sphere,tangent", base.CELLS)
def test_the_column_fields_leave_the_step_one_rows_bitwise(
    sphere, tangent, rule, row_param, lr_mode, col_param, col_k
):
    """c_0 = 1 in both, so the rows' step 1 (M, U, g and its moments) cannot see the
    column setting; from step 2 they read c_1 (here G Diag(c_1) feeds M)."""
    m, n = 12, 8
    W, grads = base._data(m, n, seed=43)
    args = (sphere, tangent, True, row_param, lr_mode, 0.7, rule, (0.9, 0.95))
    a = _run_lrs(
        _map(*args, col_param=col_param, col_k=col_k),
        W,
        grads,
        LRS,
        dtype=torch.float32,
    )
    b = _run_lrs(_map(*args), W, grads, LRS, dtype=torch.float32)
    assert torch.equal(a[0][1][:m], b[0][1][:m])
    assert not torch.equal(a[0][1][m, :-3], b[0][1][m, :-3])  # c_1
    assert not torch.equal(a[1][1][:m, :n], b[1][1][:m, :n])  # M_2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("col", [False, True])
@pytest.mark.parametrize("rule", ["adam", "sign"])
@pytest.mark.parametrize(
    "param,lr_mode",
    [("raw", "fixed"), ("softplus", "fixed"), ("raw", "tied"), ("softplus", "tied")],
)
@pytest.mark.parametrize("sphere,tangent", base.CELLS)
def test_unset_is_the_class_before_the_column_fields_bitwise(
    sphere, tangent, param, lr_mode, rule, col, dtype
):
    """Unset: wt-ps2's class (e9f431991ad4efa8). With col, setting the column fields to
    the rows' own form and rate (k = MAG_LR on a fixed row rate) is the same, bitwise."""
    W, grads = base._data(12, 8, seed=47)
    args = (sphere, tangent, col, param, lr_mode, 0.7, rule, (0.9, 0.95))
    runs = [_map(*args), _map(*args, cls=PreColAngularMuownMap)]
    if col:
        runs.append(_map(*args, col_param=param))
        if lr_mode == "fixed":
            runs.append(_map(*args, col_param=param, col_k=MAG_LR))
    outs = [_run_lrs(cm, W, grads, LRS, dtype=dtype) for cm in runs]
    for steps in zip(*outs):
        for w, st in steps[1:]:
            assert torch.equal(w, steps[0][0]) and torch.equal(st, steps[0][1])


def test_raw_columns_keep_the_sign_floor_and_softplus_columns_stay_positive():
    """A large sign-rule column rate drives raw columns through 0 (|c| >= 1e-7, the sign
    kept), while softplus columns at the same rate stay > 0; the rows (raw) are the same
    at step 1 in both."""
    m, n = 12, 8
    W, grads = base._data(m, n, seed=53, steps=12)
    lrs = [LR] * 12
    args = ("matrix", False, True, "raw", "fixed", 1.0, "sign", (0.9, 0.95))
    raw = _run_lrs(_map(*args, col_param="raw", col_k=4.0), W, grads, lrs)
    soft = _run_lrs(_map(*args, col_param="softplus", col_k=4.0), W, grads, lrs)
    c_raw = torch.stack([st[m, :-3] for _, st in raw])
    c_soft = torch.stack([st[m, :-3] for _, st in soft])
    assert bool((c_raw < 0).any()) and float(c_raw.abs().min()) >= 1e-7
    assert bool((c_soft > 0).all())
    want = _reference(
        W,
        grads,
        sphere="matrix",
        tangent=False,
        col=True,
        param="raw",
        lr_mode="fixed",
        mult=1.0,
        rule="sign",
        betas=(0.9, 0.95),
        col_param="raw",
        col_k=4.0,
        lrs=lrs,
    )
    assert float((c_raw[-1] - want[-1][2]).abs().max()) < base.TOL["matrix"] * float(
        want[-1][2].abs().max()
    )


def test_the_preset_passes_the_column_fields_and_the_hyperball_corner_refuses_them():
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
    cm = rule_of({"rule": "angular_muown", "args": args}).compute
    assert (cm.col_gain_param, cm.col_gain_lr_mult) == (None, None)
    cm = rule_of(
        {
            "rule": "angular_muown",
            "args": {**args, "col_gain_param": "softplus", "col_gain_lr_mult": 4},
        }
    ).compute
    assert (cm.gain_param, cm.gain_lr, cm.col_gain_param, cm.col_gain_lr_mult) == (
        "raw",
        "fixed",
        "softplus",
        4.0,
    )
    assert isinstance(cm.col_gain_lr_mult, float) and cm.magnitude_lr == 10.24
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
    for bad in (dict(col_gain_param="softplus"), dict(col_gain_lr_mult=1.0)):
        with pytest.raises(ValueError, match="col_gain"):
            rule_of({"rule": "angular_muown", "args": {**corner, **bad}})


def test_a_disco_resume_with_softplus_columns_continues_bitwise(one_rank):  # noqa: F811
    """The state keeps c itself (softplus(c-hat)); save after 3 steps, load into a fresh
    optimizer, continue: bitwise the run without the break, every c > 0, and the
    definition at the end."""
    W, grads = base._data(12, 8, seed=59, steps=6)
    a = {**base._ARGS, "col_gain_param": "softplus", "col_gain_lr_mult": 4.0}

    def make(w):
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
            parallel_dims=base._NoParallel(),
            lr=LR,
            rule="angular_muown",
            args=dict(a),
            param_layouts={"w": ParamLayout("matrix")},
        )

    def run(break_at):
        w = nn.Parameter(W.float().clone())
        opt = make(w)
        for i, G in enumerate(grads):
            if i == break_at:
                sd = opt.state_dict()
                w2 = nn.Parameter(w.detach().clone())
                opt = make(w2)
                opt.load_state_dict(sd)
                w = w2
            w.grad = G.float().clone()
            opt.step()
        return w.detach(), opt.state_dict()["state"][0]["rule_state"][0]

    (straight, st), (resumed, _) = run(-1), run(3)
    assert torch.equal(straight, resumed)
    c = st[12, :8]
    assert bool((c > 0).all()) and float((c - 1.0).abs().min()) > 0
    ref = _reference(
        W,
        grads,
        sphere="matrix",
        tangent=False,
        col=True,
        param="raw",
        lr_mode="fixed",
        mult=1.0,
        rule="adam",
        betas=(0.9, 0.95),
        col_param="softplus",
        col_k=4.0,
    )
    assert float((straight.double() - ref[-1][0]).abs().max()) < 1e-4
    assert float((c.double() - ref[-1][2]).abs().max()) < 1e-4
