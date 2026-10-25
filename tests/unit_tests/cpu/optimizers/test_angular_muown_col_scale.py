# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""angular_muown `col_scale`: W = Diag(g) U Diag(c) (MuonMD, arXiv 2606.25971 Eq. 1),
against its definition in float64, written from the equations:

    dL/dU = Diag(g) G Diag(c),  dL/dg_i = sum_j G_ij U_ij c_j,  dL/dc_j = sum_i G_ij g_i U_ij,

U on its sphere turned by the Stage-2 step, g and c by the same Adam (or sign) at
lr x magnitude_lr, c from 1. Off is bitwise the class before the change."""

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
from torchtitan.optimizers.disco.presets import AngularMuownMap, EtaSchedule
from torchtitan.optimizers.disco.rules import Compute, Rule, rule_of, StepCtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _angular_muown_p4_base_reference import BaseAngularMuownMap  # noqa: E402

LR, MAG_LR, M_START, STEPS = 0.05, 0.3, 4.0, 6
ETA = EtaSchedule("inverse_poly", start=1, c=0.5)
TOL = {"row": 1e-10, "matrix": 1e-5}  # the matrix step is FrobeniusAUS's, in fp32
CELLS = [  # (sphere, tangent): f-mat-row-off first
    ("matrix", False),
    ("matrix", True),
    ("row", False),
    ("row", True),
]


@pytest.fixture(autouse=True, scope="module")
def _fresh_dynamo():
    yield
    torch._dynamo.reset()  # this module's shapes and dtypes must not count toward later recompile limits


@dataclass(frozen=True)
class _Polar(Compute):
    k: float = 0.5

    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return self.k * (u @ vh)


def _data(m, n, seed, steps=STEPS):
    gen = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(
        torch.randn(max(m, n), max(m, n), generator=gen, dtype=torch.float64)
    )[0][:m, :n]
    W = W * torch.logspace(-0.3, 0.3, m, dtype=torch.float64)[:, None]
    W = (
        W * torch.logspace(0.2, -0.2, n, dtype=torch.float64)[None, :]
    )  # unequal columns too
    return W, [
        0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]


def _adam(st, x, t, b1=0.9, b2=0.95, eps=1e-8):
    st["m"] = b1 * st["m"] + (1 - b1) * x
    st["v"] = b2 * st["v"] + (1 - b2) * x * x
    return (st["m"] / (1 - b1**t)) / ((st["v"] / (1 - b2**t)).sqrt() + eps)


def _sign(st, x, t, b1=0.9, b2=0.95, eps=1e-8):
    st["m"] = b1 * st["m"] + (1 - b1) * x
    return torch.sign(st["m"])


def _polar(X, k):
    u, _, vh = torch.linalg.svd(X, full_matrices=False)
    return k * (u @ vh)


def _reference(
    W,
    grads,
    *,
    sphere,
    tangent,
    rule="adam",
    k=0.5,
    M=M_START,
    lr=LR,
    mag_lr=MAG_LR,
    beta=0.95,
    eta=ETA,
):
    """W_t = Diag(g) U Diag(c). Start: c = 1; g = W0's row norms (row sphere) or their
    RMS (matrix sphere), times M from step 1 on; U = W0 / start. Each step, at the
    current (g, U, c): X = dL/dU = Diag(g) G Diag(c), projected on U's tangent space
    with `tangent`; momentum (beta, Nesterov) on X; O = polar(D) rescaled to ||U||_F;
    U <- retract(U - lr eta(t) O). dL/dg_i = <G_i, U_i c>, dL/dc_j = <G_:j, g U_:j>,
    both by Adam (or sign) at lr mag_lr. Returns [(W, g, c)] per step."""
    m, n = W.shape
    norms = W.norm(dim=1, keepdim=True)
    start = (
        norms if sphere == "row" else norms.square().mean().sqrt().expand(m, 1).clone()
    )
    U, g, c = W / start, M * start, torch.ones(1, n, dtype=W.dtype)
    mom = torch.zeros_like(W)
    upd = _adam if rule == "adam" else _sign
    sg = {"m": torch.zeros_like(g), "v": torch.zeros_like(g)}
    sc = {"m": torch.zeros_like(c), "v": torch.zeros_like(c)}

    def proj(X):
        if sphere == "row":
            return X - (X * U).sum(dim=1, keepdim=True) * U
        return X - (X * U).sum() / (U * U).sum() * U

    out = []
    for t, G in enumerate(grads, start=1):
        X = g * G * c
        if tangent:
            X = proj(X)
        mom = beta * mom + X
        D = X + beta * mom
        O = _polar(D, k)
        V = U - lr * eta(t) * O * (U.norm() / O.norm())
        U_next = (
            V / V.norm(dim=1, keepdim=True)
            if sphere == "row"
            else V * (U.norm() / V.norm())
        )
        dg = (G * U * c).sum(dim=1, keepdim=True)
        dc = (G * U * g).sum(dim=0, keepdim=True)
        g = g - lr * mag_lr * upd(sg, dg, t)
        c = c - lr * mag_lr * upd(sc, dc, t)
        U = U_next
        out.append((g * U * c, g[:, 0].clone(), c[0].clone()))
    return out


def _map(sphere, tangent, *, rule="adam", col=True, cls=AngularMuownMap, k=0.5):
    kw = dict(col_scale=True) if col else {}
    return cls(
        Rule((), _Polar(k), scale="one"),
        warmup=0,
        c=0.0,
        magnitude_lr=MAG_LR,
        eta=ETA,
        magnitude_shape="row",
        magnitude_rule=rule,
        sphere=sphere,
        tangent=tangent,
        native_norm=False,
        radius=M_START,
        **kw,
    )


def _run(cm, W, grads, lr=LR, dtype=torch.float64):
    m, n = W.shape
    w, state, out = (
        W[None].to(dtype),
        torch.zeros(1, *cm.state_shape(m, n), dtype=dtype),
        [],
    )
    for t, G in enumerate(grads, start=1):
        res = cm.run(G[None].to(dtype), w, state, StepCtx(lr, 0.0, t))
        w, state = res.weight, res.state
        out.append((w[0], state[0]))
    return out


def _gc(cm, state, m):
    """(g, c) from a col_scale state."""
    return state[:m, -3], state[m, :-3]


# ---------------------------------------------------------------- (a) the definition
@pytest.mark.parametrize("rule", ["adam", "sign"])
@pytest.mark.parametrize("sphere,tangent", CELLS)
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
def test_it_is_its_definition_step_after_step(sphere, tangent, rule, m, n):
    W, grads = _data(m, n, seed=7 * m + n)
    cm = _map(sphere, tangent, rule=rule)
    got, want = _run(cm, W, grads), _reference(
        W, grads, sphere=sphere, tangent=tangent, rule=rule
    )
    tol = TOL[sphere]
    for t, ((w, st), (rw, rg, rc)) in enumerate(zip(got, want), start=1):
        g, c = _gc(cm, st, m)
        assert float((w - rw).abs().max()) < tol * float(
            rw.abs().max()
        ), f"W at step {t}"
        assert float((g - rg).abs().max()) < tol * float(
            rg.abs().max()
        ), f"g at step {t}"
        assert float((c - rc).abs().max()) < tol, f"c at step {t}"
    assert (
        float((got[-1][1][m, :-3] - 1.0).abs().max()) > 1e-3
    )  # the column scales moved


def test_the_column_scales_change_the_result():
    """Power: with c the step differs from the row-only rule well beyond the tolerance."""
    W, grads = _data(12, 8, seed=3)
    on = _run(_map("matrix", False), W, grads)[-1][0]
    off = _run(_map("matrix", False, col=False), W, grads)[-1][0]
    assert float((on - off).abs().max()) > 1e3 * TOL["matrix"] * float(off.abs().max())


# ---------------------------------------------------------------- (b) off is the old class
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "args",
    [
        dict(sphere=s, tangent=t, magnitude_shape=sh, magnitude_rule=r)
        for s, t, sh, r in itertools.product(
            ("row", "matrix"), (False, True), ("row", "unit", "none"), ("adam", "sign")
        )
        if not (sh == "none" and r != "adam")
        and (s, sh, t) != ("matrix", "none", False)
    ],
)
def test_off_is_the_class_before_the_change_bitwise(args, dtype):
    W, grads = _data(12, 8, seed=11)
    kw = dict(warmup=0, c=0.0, eta=ETA, native_norm=False, radius=M_START, **args)
    if args["magnitude_shape"] != "none":
        kw["magnitude_lr"] = MAG_LR
    new = AngularMuownMap(Rule((), _Polar(0.5), scale="one"), **kw)
    old = BaseAngularMuownMap(Rule((), _Polar(0.5), scale="one"), **kw)
    assert new.state_shape(12, 8) == old.state_shape(12, 8)
    for (a, sa), (b, sb) in zip(
        _run(new, W, grads, dtype=dtype), _run(old, W, grads, dtype=dtype)
    ):
        assert torch.equal(a, b) and torch.equal(sa, sb)


def test_off_native_is_the_class_before_the_change_bitwise():
    W, grads = _data(8, 12, seed=5)
    kw = dict(warmup=2, c=1e-2)
    new = AngularMuownMap(Rule((), _Polar(0.5), scale="image_spectral"), **kw)
    old = BaseAngularMuownMap(Rule((), _Polar(0.5), scale="image_spectral"), **kw)
    for (a, sa), (b, sb) in zip(_run(new, W, grads), _run(old, W, grads)):
        assert torch.equal(a, b) and torch.equal(sa, sb)


# ---------------------------------------------------------------- (c) step 1 at c = 1
@pytest.mark.parametrize("sphere,tangent", CELLS)
def test_step_one_is_the_row_rule_times_the_first_column_scales(sphere, tangent):
    """At c = 1 the forward is unchanged (W itself), U's step and g's step equal the
    row-only rule's: W_on(1) = W_off(1) Diag(c_1), c_1 = 1 - lr mag_lr adam_1(dL/dc)."""
    W, grads = _data(12, 8, seed=9, steps=1)
    ((on, st),) = _run(_map(sphere, tangent), W, grads)
    ((off, st_off),) = _run(_map(sphere, tangent, col=False), W, grads)
    g, c = _gc(None, st, 12)
    assert torch.equal(st[:12], st_off)  # momentum and g's moments: the row rule's
    assert float((on - off * c).abs().max()) < 1e-14 * float(off.abs().max())
    assert float((c - 1.0).abs().max()) > 0.5 * LR * MAG_LR


# ---------------------------------------------------------------- (d) settings
@pytest.mark.parametrize(
    "bad",
    [
        dict(magnitude_shape="unit"),
        dict(magnitude_shape="none", magnitude_lr=1.0),
        dict(magnitude_rule="log_rms", magnitude_lr=0.1, betas=(0.9, 0.95)),
        dict(frozen_units=("attention.qkv_linear.wq",)),
    ],
)
def test_unsupported_settings_are_refused_only_with_col_scale(bad):
    base = dict(
        warmup=0,
        c=0.0,
        native_norm=False,
        sphere="row",
        tangent=False,
        magnitude_lr=MAG_LR,
    )
    base.update(bad)
    AngularMuownMap(
        Rule((), _Polar(0.5), scale="one"), **base
    )  # accepted without col_scale
    with pytest.raises(ValueError, match="col_scale"):
        AngularMuownMap(Rule((), _Polar(0.5), scale="one"), col_scale=True, **base)


def test_the_preset_passes_it_and_the_hyperball_corner_refuses_it():
    args = dict(
        compute="svd",
        scale="one",
        warmup=0,
        c=0.0,
        native_norm=False,
        sphere="matrix",
        tangent=False,
        radius=M_START,
        magnitude_lr=MAG_LR,
    )
    rule = rule_of({"rule": "angular_muown", "args": {**args, "col_scale": True}})
    assert rule.compute.col_scale and rule.compute.state_shape(12, 8) == (15, 11)
    assert not rule_of({"rule": "angular_muown", "args": args}).compute.col_scale
    corner = {k: v for k, v in args.items() if k != "magnitude_lr"}
    with pytest.raises(ValueError, match="col_scale"):
        rule_of(
            {
                "rule": "angular_muown",
                "args": {**corner, "magnitude_shape": "none", "col_scale": True},
            }
        )


# ---------------------------------------------------------------- (e) through DiSCO
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


_ARGS = dict(
    compute="svd",
    scale="one",
    warmup=0,
    c=0.0,
    native_norm=False,
    sphere="matrix",
    tangent=False,
    radius=M_START,
    magnitude_lr=MAG_LR,
    col_scale=True,
    eta={"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0},
)


def _disco(w):
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
        args=dict(_ARGS),
        param_layouts={"w": ParamLayout("matrix")},
    )


def test_a_disco_step_is_the_definition_and_a_resume_continues_bitwise(one_rank):
    """The state rows travel with the checkpoint: save after 3 steps, load into a fresh
    optimizer, continue; equal bitwise to the run without the break, and to the
    definition at the end."""
    W, grads = _data(12, 8, seed=31, steps=6)

    def run(break_at):
        w = nn.Parameter(W.float().clone())
        opt = _disco(w)
        for i, G in enumerate(grads):
            if i == break_at:
                sd = opt.state_dict()
                assert sd["state"][0]["rule_state"].shape == (1, 15, 11)
                c = sd["state"][0]["rule_state"][0, 12, :8]
                assert (
                    float((c - 1.0).abs().min()) > 0
                )  # the saved column scales are learned ones
                w2 = nn.Parameter(w.detach().clone())
                opt = _disco(w2)
                opt.load_state_dict(sd)
                w = w2
            w.grad = G.float().clone()
            opt.step()
        return w.detach()

    straight, resumed = run(-1), run(3)
    assert torch.equal(straight, resumed)
    eta = EtaSchedule("inverse_poly", start=1, c=0.5)
    ref = _reference(W, grads, sphere="matrix", tangent=False, k=1.0, eta=eta)[-1][0]
    assert float((straight.double() - ref).abs().max()) < 1e-4


def test_a_lost_column_state_restarts_from_one(one_rank):
    """Loading a checkpoint whose rule_state has zero column rows restarts c at 1:
    the same step as a fresh optimizer on the same weight."""
    W, grads = _data(12, 8, seed=13, steps=2)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w)
    w.grad = grads[0].float().clone()
    opt.step()
    sd = opt.state_dict()
    sd["state"][0]["rule_state"][0, 12:] = 0.0
    sd["state"][0]["rule_state"][0, :12] = 0.0
    w2 = nn.Parameter(w.detach().clone())
    a = _disco(w2)
    a.load_state_dict(sd)
    w3 = nn.Parameter(w.detach().clone())
    b = _disco(w3)
    for p, o in ((w2, a), (w3, b)):
        p.grad = grads[1].float().clone()
        for g in o.param_groups:
            g["t"] = 0
        o.step()
    assert torch.equal(w2.detach(), w3.detach())
