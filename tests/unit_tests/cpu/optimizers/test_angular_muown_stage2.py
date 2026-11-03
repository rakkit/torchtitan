# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Stage-2 unified rule (angular_muown with sphere / tangent / magnitude_shape
none / native_norm / radius) against its definition (optimizer-study PREREG 7b.3,
design D1: "none" freezes g at M x the start row norms), in float64."""

import itertools
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
from torchtitan.optimizers.disco.presets import (
    AngularMuownMap,
    EtaSchedule,
    FrobeniusAUS,
)
from torchtitan.optimizers.disco.rules import Compute, Rule, rule_of, StepCtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _angular_muown_pre_change_reference import PreChangeAngularMuownMap  # noqa: E402

LR, MAG_LR, M_START, STEPS = 0.05, 0.3, 4.0, 6
TOL = {"row": 1e-10, "matrix": 1e-5}  # the matrix step is FrobeniusAUS's, in fp32
ETA = EtaSchedule("inverse_poly", start=1, c=0.5)  # moves within six steps
SPHERES, SCALES = ("row", "matrix"), ("none", "unit", "row")
CELLS = [
    c
    for c in itertools.product(SPHERES, SCALES, (False, True))
    if c != ("matrix", "none", False)
]


@dataclass(frozen=True)
class _Polar(Compute):
    """k U V^T in float64: a non-unit orthogonaliser, so a missing rescale shows."""

    k: float = 0.5

    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return self.k * (u @ vh)


def _exact_polar(X, k):
    u, _, vh = torch.linalg.svd(X, full_matrices=False)
    return k * (u @ vh)


def _data(m, n, seed, steps=STEPS):
    """Rows of unequal norm (a spread far wider than the init's 2-3%) and gradients."""
    gen = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(
        torch.randn(max(m, n), max(m, n), generator=gen, dtype=torch.float64)
    )[0][:m, :n]
    W = W * torch.logspace(-0.3, 0.3, m, dtype=torch.float64)[:, None]
    return W, [
        0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]


def _adam(state, x, t, b1=0.9, b2=0.95, eps=1e-8):
    state["m"] = b1 * state["m"] + (1 - b1) * x
    state["v"] = b2 * state["v"] + (1 - b2) * x * x
    return (state["m"] / (1 - b1**t)) / ((state["v"] / (1 - b2**t)).sqrt() + eps)


def _reference(
    W,
    grads,
    *,
    sphere,
    scale,
    tangent,
    k=0.5,
    M=M_START,
    lr=LR,
    mag_lr=MAG_LR,
    beta=0.95,
    eta=ETA,
    mutate=None,
):
    """W = Diag(g) U. Row sphere: unit rows, g from W0's row norms; matrix
    sphere: ||U||_F = sqrt(m), g from the RMS row norm. The scale is M times that
    from step 1 on (the step-1 gradient is W0's). Momentum (beta, Nesterov) on the
    U-gradient Diag(g) G, projected on U's tangent space first with `tangent`.
    O = polar(D) rescaled to ||O||_F = ||U||_F; U moves by lr eta(t) O~ and is
    retracted to its sphere. Magnitudes: Adam on dL/dg_i = <G_i, U_i> (per row), or
    on dL/da = sum_i q_i <G_i, U_i> with W = a Diag(q) U (per matrix, q the start
    ratios, rms(q) = 1), at lr mag_lr, no eta. "none": g stays at M x its start.
    `mutate` makes one plausible mistake, to show the comparison sees it."""
    m, n = W.shape
    norms = W.norm(dim=1, keepdim=True)
    if sphere == "row" and mutate != "uniform_none":
        start = norms
    else:
        start = norms.square().mean().sqrt().expand(m, 1).clone()
    U = W / (M * start if mutate == "jump_before_U" else start)
    g = M * start
    q = start / start.square().mean().sqrt()
    a = M * start.square().mean().sqrt()
    mom = torch.zeros_like(W)
    adam = (
        {"m": torch.zeros_like(g), "v": torch.zeros_like(g)}
        if scale == "row"
        else {"m": torch.zeros(()), "v": torch.zeros(())}
    )

    def proj(X):
        if (
            sphere == "row"
            and mutate != "matrix_tangent_on_rows"
            or mutate == "row_tangent_on_matrix"
        ):
            return X - (X * U).sum(dim=1, keepdim=True) * U
        return X - (X * U).sum() / (U * U).sum() * U

    out = []
    for t, G in enumerate(grads, start=1):
        GU = G.clone() if mutate == "momentum_on_G" else g * G
        if tangent and mutate != "tangent_after_momentum":
            GU = proj(GU)
        mom = beta * mom + GU
        D = GU + beta * mom
        if tangent and mutate == "tangent_after_momentum":
            D = proj(D)
        O = _exact_polar(D, k)
        O_turn = O if mutate == "no_rescale" else O * U.norm() / O.norm()
        V = U - lr * eta(t) * O_turn
        U_next = (
            V / V.norm(dim=1, keepdim=True)
            if sphere == "row"
            else V * (U.norm() / V.norm())
        )
        gam = (G * (U_next if mutate == "gamma_after_step" else U)).sum(
            dim=1, keepdim=True
        )
        rate = lr * mag_lr * (eta(t) if mutate == "magnitudes_on_eta" else 1.0)
        if scale == "row":
            g = g - rate * _adam(adam, gam, t)
        elif scale == "unit":
            a = a - rate * _adam(adam, (q * gam).sum(), t)
            g = a * q
        U = U_next
        out.append(g * U)
    return out


def _map(
    sphere,
    scale,
    tangent,
    *,
    k=0.5,
    M=M_START,
    mag_lr=MAG_LR,
    eta=ETA,
    native_norm=False,
):
    return AngularMuownMap(
        Rule((), _Polar(k), scale="image_spectral" if native_norm else "one"),
        warmup=0,
        c=0.0,
        magnitude_lr=mag_lr if scale != "none" else 1.0,
        eta=eta,
        magnitude_shape=scale,
        sphere=sphere,
        tangent=tangent,
        native_norm=native_norm,
        radius=M,
    )


def _run(cm, W, grads, lr=LR):
    m, n = W.shape
    w, state, ws = W[None], W.new_zeros(1, *cm.state_shape(m, n)), []
    for t, G in enumerate(grads, start=1):
        res = cm.run(G[None], w, state, StepCtx(lr, 0.0, t))
        w, state = res.weight, res.state
        ws.append(w[0])
    return ws


def _corner(W, grads, *, k=0.5, M=M_START, lr=LR, eta=ETA, mom=0.05):
    """(matrix, none, off): hyperball on DiSCO's group momentum, buf <- (1 - m)
    buf + m g, Nesterov lerp(buf, g, m) (optimizer.py, DiSCO.__init__ docstring)."""
    cm = FrobeniusAUS(Rule((), _Polar(k)), eta=eta, radius_ratio=((0, 1.0), (1, M)))
    w, buf, ws = W[None].clone(), torch.zeros_like(W)[None], []
    for t, G in enumerate(grads, start=1):
        buf = (1 - mom) * buf + mom * G[None]
        res = cm.run(torch.lerp(buf, G[None], mom), w, None, StepCtx(lr, 0.0, t))
        w = res.alpha[:, None, None] * w - res.update
        ws.append(w[0])
    return ws


def _max_diff(xs, ys):
    return max(float((x - y).abs().max()) for x, y in zip(xs, ys))


# ---------------------------------------------------------------- (a) the definition
@pytest.mark.parametrize("sphere,scale,tangent", CELLS)
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
def test_each_cell_is_its_definition_step_after_step(sphere, scale, tangent, m, n):
    W, grads = _data(m, n, seed=m + 3 * n)
    ref = _reference(W, grads, sphere=sphere, scale=scale, tangent=tangent)
    got = _run(_map(sphere, scale, tangent), W, grads)
    assert _max_diff(ref, got) < TOL[sphere]


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
def test_the_corner_is_its_definition(m, n):
    W, grads = _data(m, n, seed=5 + m)
    ref = _reference(W, grads, sphere="matrix", scale="none", tangent=False)
    assert _max_diff(ref, _corner(W, grads)) < TOL["matrix"]


@pytest.mark.parametrize(
    "mutate,cell",
    [
        ("tangent_after_momentum", ("row", "row", True)),
        ("tangent_after_momentum", ("matrix", "unit", True)),
        ("momentum_on_G", ("row", "row", False)),
        ("momentum_on_G", ("matrix", "row", False)),
        ("no_rescale", ("row", "none", True)),
        ("no_rescale", ("matrix", "row", True)),
        ("magnitudes_on_eta", ("row", "row", True)),
        ("magnitudes_on_eta", ("matrix", "unit", False)),
        ("jump_before_U", ("row", "unit", True)),
        ("row_tangent_on_matrix", ("matrix", "none", True)),
        ("matrix_tangent_on_rows", ("row", "none", True)),
        ("gamma_after_step", ("row", "row", True)),
        ("uniform_none", ("row", "none", False)),
    ],
)
def test_the_comparison_sees_each_plausible_mistake(mutate, cell):
    W, grads = _data(12, 8, seed=11)
    sphere, scale, tangent = cell
    ref = _reference(
        W, grads, sphere=sphere, scale=scale, tangent=tangent, mutate=mutate
    )
    assert _max_diff(ref, _run(_map(*cell), W, grads)) > 1e-4


# ---------------------------------------------------------------- (b) native = today's rule
_NATIVE = [
    dict(),
    dict(nesterov=False),
    dict(magnitude_shape="unit"),
    dict(magnitude_rule="sign"),
    dict(magnitude_shape="unit", magnitude_rule="sign"),
    dict(eta={"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0}),
    dict(warmup=0, c=0.0, eta={"shape": "sqrt_decay", "start": 2, "total": 9}),
    dict(magnitude_lr=0.0),
]


def _pre_change(rule):
    c = rule.compute
    return PreChangeAngularMuownMap(
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
    )


@pytest.mark.parametrize("args", _NATIVE)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute", ["svd", "newtonschulz5"])
def test_native_is_the_pre_change_rule_bitwise(args, dtype, compute):
    """Stacks of 3 units with a zero row, a lost state at step 4, 8 steps."""
    rule = rule_of(
        {
            "rule": "angular_muown",
            "args": {"compute": compute, "warmup": 2, "c": 0.5, **args},
        }
    )
    live, old = rule.compute, _pre_change(rule)
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


# ---------------------------------------------------------------- (c) the corner is hyperball
_GPE = {"compute": "gram_polar_express", "steps": 6}
_SSTAR = {"shape": "sqrt_decay", "start": 100, "total": 25600}
_IPOLY = {"shape": "inverse_poly", "start": 100, "c": 1e-3, "p": 1.0}


def _f_corner(**kw):
    args = {
        **_GPE,
        "scale": "one",
        "warmup": 0,
        "c": 0.0,
        "native_norm": False,
        "sphere": "matrix",
        "magnitude_shape": "none",
        "tangent": False,
        "radius": 4.0,
        "eta": _SSTAR,
        **kw,
    }
    return rule_of({"rule": "angular_muown", "args": args})


def _hyperball(radius=4.0, eta=_SSTAR):
    """The study's hyperball cell as cell.py builds it."""
    args = {**_GPE, "pre": "identity", "scale": "one", "eta": eta, "exact_angle": False}
    if radius != 1.0:
        args["radius_ratio"] = [[0, 1.0], [1, radius]]
    return rule_of({"rule": "hyperball", "args": args})


@pytest.mark.parametrize("eta", [_SSTAR, _IPOLY])
@pytest.mark.parametrize("radius", [4.0, 1.0])
def test_the_corner_is_the_hyperball_rule(eta, radius):
    assert _f_corner(eta=eta, radius=radius) == _hyperball(radius, eta)
    assert not _f_corner(eta=eta, radius=radius).compute.owns_momentum


def test_the_corner_equality_sees_a_different_rule():
    assert _f_corner(radius=2.0) != _hyperball(4.0)
    assert _f_corner(eta=_IPOLY) != _hyperball(4.0, _SSTAR)
    assert _f_corner(steps=5) != _hyperball(4.0)


@pytest.mark.parametrize(
    "bad",
    [
        dict(beta=0.9),
        dict(magnitude_lr=0.16),
        dict(warmup=100),
        dict(scale="image_spectral"),
    ],
)
def test_the_corner_refuses_settings_hyperball_would_drop(bad):
    with pytest.raises(ValueError, match="hyperball"):
        _f_corner(**bad)


# ---------------------------------------------------------------- (d) across the paths
def test_a_frozen_per_matrix_scale_is_the_corner():
    """The weight-returning path on owned momentum = hyperball on group momentum."""
    W, grads = _data(12, 8, seed=21, steps=8)
    got = _run(_map("matrix", "unit", False, mag_lr=0.0), W, grads)
    assert _max_diff(got, _corner(W, grads)) < TOL["matrix"]


@pytest.mark.parametrize(
    "sphere,tangent", [("row", False), ("row", True), ("matrix", True)]
)
@pytest.mark.parametrize("scale", ["unit", "row"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_frozen_learned_scale_is_none_bitwise(sphere, tangent, scale, dtype):
    W, grads = _data(12, 8, seed=22)
    W, grads = W.to(dtype), [G.to(dtype) for G in grads]
    frozen = _run(_map(sphere, scale, tangent, mag_lr=0.0), W, grads)
    none = _run(_map(sphere, "none", tangent), W, grads)
    assert all(torch.equal(a, b) for a, b in zip(frozen, none))


@pytest.mark.parametrize("scale", ["row", "unit"])
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
def test_native_norm_switches_only_the_normalisation(scale, m, n):
    """At full rank an exact polar factor has ||O||_F = sqrt(min(m, n)), so
    image_spectral and the rescale to ||U||_F = sqrt(m) agree."""
    W, grads = _data(m, n, seed=23)
    native = _run(_map("row", scale, True, k=1.0, M=1.0, native_norm=True), W, grads)
    normed = _run(_map("row", scale, True, k=1.0, M=1.0), W, grads)
    assert _max_diff(native, normed) < 1e-10
    half = _run(_map("row", scale, True, k=0.5, M=1.0, native_norm=True), W, grads)
    assert _max_diff(half, normed) > 1e-4  # native keeps the backend's scale


# ---------------------------------------------------------------- (e) declarations, refusals, DiSCO
def test_what_it_declares():
    cm = _map("matrix", "row", True)
    assert (cm.owns_momentum, cm.needs_weight, cm.has_state, cm.output) == (
        True,
        True,
        True,
        "weight",
    )
    assert [_map("row", s, True).state_shape(12, 8) for s in SCALES] == [
        (12, 9),
        (12, 12),
        (12, 11),
    ]


@pytest.mark.parametrize(
    "bad,match",
    [
        (dict(native_norm=True, sphere="matrix"), "native_norm false"),
        (dict(native_norm=False, warmup=100), "warmup 0"),
        (
            dict(native_norm=False, scale="image_spectral", warmup=0, c=0.0),
            "set scale one",
        ),
        (dict(magnitude_shape="none", magnitude_lr=0.5), "learns no magnitudes"),
        (dict(sphere="ball"), "sphere is row or matrix"),
        (dict(radius=0.0), "radius must be > 0"),
    ],
)
def test_bad_settings_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        rule_of({"rule": "angular_muown", "args": {"compute": "svd", **bad}})


def test_the_class_refuses_the_corner():
    with pytest.raises(ValueError, match="preset builds it"):
        _map("matrix", "none", False)


@pytest.fixture(scope="module")
def one_rank():
    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
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


def _disco(w, args, momentum, nesterov):
    return DiSCO(
        [
            {
                "params": [w],
                "param_names": ["w"],
                "momentum": momentum,
                "nesterov": nesterov,
                "weight_decay": 0.0,
            }
        ],
        parallel_dims=_NoParallel(),
        lr=LR,
        rule="angular_muown",
        args=args,
        param_layouts={"w": ParamLayout("matrix")},
    )


def _f_args(sphere, scale, tangent):
    args = {
        "compute": "svd",
        "scale": "one",
        "warmup": 0,
        "c": 0.0,
        "native_norm": False,
        "sphere": sphere,
        "magnitude_shape": scale,
        "tangent": tangent,
        "radius": M_START,
        "eta": {"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0},
    }
    if scale != "none":
        args["magnitude_lr"] = MAG_LR
    return args


@pytest.mark.parametrize("sphere,scale,tangent", CELLS + [("matrix", "none", False)])
def test_a_disco_step_is_the_rule(one_rank, sphere, scale, tangent):
    """The group momentum the rule asks for (owned: 1.0; the corner: 0.05 +
    Nesterov), state carried, the weight applied; save / load mid-run continues."""
    args = _f_args(sphere, scale, tangent)
    owned = rule_of({"rule": "angular_muown", "args": args}).compute.owns_momentum
    mom, nest = (1.0, False) if owned else (0.05, True)
    W, grads = _data(12, 8, seed=31, steps=4)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w, args, mom, nest)
    for i, G in enumerate(grads):
        if i == 2:
            sd = opt.state_dict()
            w2 = nn.Parameter(w.detach().clone())
            opt = _disco(w2, args, mom, nest)
            opt.load_state_dict(sd)
            w = w2
        w.grad = G.float().clone()
        opt.step()
    eta = EtaSchedule("inverse_poly", start=1, c=0.5)
    cell = dict(sphere=sphere, scale=scale, tangent=tangent, k=1.0, eta=eta)
    if (sphere, scale, tangent) == ("matrix", "none", False):
        ref = _corner(W, grads, k=1.0, eta=eta)[-1]
    else:
        ref = _reference(W, grads, **cell)[-1]
    assert float((w.detach().double() - ref).abs().max()) < 1e-4


@pytest.mark.parametrize("cell", [("row", "row", True), ("matrix", "none", False)])
def test_the_wrong_group_momentum_is_refused_or_wrong(one_rank, cell):
    """An owning cell with momentum 0.05 raises; the corner with momentum 1.0
    runs without momentum, which the resolve.py assert exists to catch."""
    args = _f_args(*cell)
    w = nn.Parameter(torch.randn(12, 8))
    if cell == ("row", "row", True):
        with pytest.raises(ValueError, match="raw gradient"):
            _disco(w, args, 0.05, True)
    else:
        _disco(w, args, 1.0, False)  # accepted by DiSCO: the config check guards it
        assert not rule_of(
            {"rule": "angular_muown", "args": args}
        ).compute.owns_momentum
