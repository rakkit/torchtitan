# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-unit length freeze on F's recipe (row sphere, per-row Adam lengths): `frozen_units` names unit types
(`attention.qkv_linear.wq`, `feed_forward.w3`, ...) whose row lengths stay at their start, g = M ||W0_i||,
while the other units of the same fused parameter learn theirs. A frozen unit's U-step is F's, with the
gradient taken at its fixed lengths (G_U = g G): the unit is what a magnitude_shape "none" group would do to
it (freeze.<type> at unit granularity). References in float64, written from these equations."""

import copy
import itertools
import math  # noqa: F401
import os
import socket
import sys
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout, qkv_units, stacked_units
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import AngularMuownMap, EtaSchedule
from torchtitan.optimizers.disco.rules import Compute, Rule, rule_of, StepCtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _angular_muown_s5_reference  # noqa: E402
from _angular_muown_s5_reference import S5AngularMuownMap  # noqa: E402

LR, MAG_LR, M_START = 0.05, 0.3, 4.0
ETA = EtaSchedule("inverse_poly", start=1, c=0.5)
Q, K, V, W1, W3 = (
    "attention.qkv_linear.wq",
    "attention.qkv_linear.wk",
    "attention.qkv_linear.wv",
    "feed_forward.w1",
    "feed_forward.w3",
)


@dataclass(frozen=True)
class _Polar(Compute):
    k: float = 0.5

    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return self.k * (u @ vh)


def _exact_polar(X, k):
    u, _, vh = torch.linalg.svd(X, full_matrices=False)
    return k * (u @ vh)


def _data(m, n, seed, steps=6):
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
    W, grads, *, frozen, k=0.5, M=M_START, lr=LR, mag_lr=MAG_LR, beta=0.95, eta=ETA
):
    """F's recipe: U = W0 / its row norms, g = M x those norms; G_U = g G; M <- beta M + G_U;
    D = G_U + beta M; O = k polar(D) rescaled to ||U||_F; U+ = RowNorm(U - lr eta(t) O~); g by Adam on
    <G_i, U_i> at lr mag_lr, or (frozen) g fixed."""
    start = W.norm(dim=1, keepdim=True)
    U, g = W / start, M * start
    mom, adam, out = (
        torch.zeros_like(W),
        {"m": torch.zeros_like(g), "v": torch.zeros_like(g)},
        [],
    )
    for t, G in enumerate(grads, start=1):
        GU = g * G
        mom = beta * mom + GU
        D = GU + beta * mom
        O = _exact_polar(D, k)
        V = U - lr * eta(t) * O * U.norm() / O.norm()
        U_next = V / V.norm(dim=1, keepdim=True)
        if not frozen:
            g = g - lr * mag_lr * _adam(adam, (G * U).sum(dim=1, keepdim=True), t)
        U = U_next
        out.append(g * U)
    return out


def _map(frozen_units=(Q,), **kw):
    args = dict(
        warmup=0,
        c=0.0,
        eta=ETA,
        sphere="row",
        tangent=False,
        native_norm=False,
        radius=M_START,
        magnitude_lr=MAG_LR,
        frozen_units=frozen_units,
    )
    return AngularMuownMap(Rule((), _Polar(0.5), scale="one"), **{**args, **kw})


def _run_stack(cm, Ws, grads, frozen):
    """A stack of units, `frozen` the per-unit mask DiSCO passes in the step context."""
    w = torch.stack(Ws)
    state = w.new_zeros(len(Ws), *cm.state_shape(*Ws[0].shape))
    mask = None if frozen is None else torch.tensor(frozen)
    out = []
    for t in range(len(grads[0])):
        res = cm.run(
            torch.stack([gs[t] for gs in grads]),
            w,
            state,
            StepCtx(LR, 0.0, t + 1, frozen=mask),
        )
        w, state = res.weight, res.state
        out.append((w, state))
    return out


def _stack(seed=1, m=12, n=8, units=3, steps=6):
    data = [_data(m, n, seed=seed + j, steps=steps) for j in range(units)]
    return [W for W, _ in data], [gs for _, gs in data]


# ---------------------------------------------------------------- the definition
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)])
@pytest.mark.parametrize(
    "frozen", [(False, True, False), (True, False, True), (False, False, False)]
)
def test_a_stack_with_frozen_units_is_its_definition(m, n, frozen):
    Ws, grads = _stack(m=m, n=n)
    got = _run_stack(_map(), Ws, grads, frozen)
    for j, (W, gs) in enumerate(zip(Ws, grads)):
        ref = _reference(W, gs, frozen=frozen[j])
        assert (
            max(float((r - w[j]).abs().max()) for r, (w, _) in zip(ref, got)) < 1e-10
        ), j


def test_the_definition_sees_a_learning_frozen_unit():
    W, gs = _data(12, 8, seed=3)
    a, b = _reference(W, gs, frozen=True), _reference(W, gs, frozen=False)
    assert max(float((x - y).abs().max()) for x, y in zip(a, b)) > 1e-3


@pytest.mark.parametrize("rule", ["adam", "sign"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_lengths_are_bitwise_constant_and_the_rest_is_the_unfrozen_rule(
    rule, dtype
):
    """8 steps, a lost state at step 4: the frozen unit's lengths are M ||W0_i|| (to the last bit of the
    norm) and then bitwise constant; the other units' weights and state are torch.equal to the run without
    the mask."""
    Ws, grads = _stack(steps=8)
    Ws, grads = [W.to(dtype) for W in Ws], [[G.to(dtype) for G in gs] for gs in grads]
    cm = _map(magnitude_rule=rule)
    w, state = torch.stack(Ws), torch.stack(Ws).new_zeros(3, *cm.state_shape(12, 8))
    w0, s0 = w, state
    start = M_START * Ws[1].norm(dim=-1)
    for t in range(1, 9):
        G = torch.stack([gs[t - 1] for gs in grads])
        if t == 4:
            state, s0 = torch.zeros_like(state), torch.zeros_like(s0)
        a = cm.run(
            G, w, state, StepCtx(LR, 0.0, t, frozen=torch.tensor([False, True, False]))
        )
        b = cm.run(
            G, w0, s0, StepCtx(LR, 0.0, t, frozen=torch.tensor([False, False, False]))
        )
        if t == 1:
            keep = a.state[1, :, 8]
            assert float(((keep - start) / start).abs().max()) < (
                1e-15 if dtype == torch.float64 else 1e-6
            )
        if (
            t == 4
        ):  # a lost state restarts from the current rows: the lengths it finds then stay
            assert float(((a.state[1, :, 8] - start) / start).abs().max()) < 1e-6
            keep = a.state[1, :, 8]
        assert torch.equal(a.state[1, :, 8], keep), t
        assert torch.equal(a.weight[[0, 2]], b.weight[[0, 2]]) and torch.equal(
            a.state[[0, 2]], b.state[[0, 2]]
        ), t
        w, state, w0, s0 = a.weight, a.state, b.weight, b.state
    assert float((w[1] - w0[1]).abs().max()) > 1e-4


def test_a_frozen_unit_is_the_none_shape_bitwise():
    """The frozen unit's weights are those of magnitude_shape "none" (freeze.<type>) on that unit."""
    Ws, grads = _stack()
    got = _run_stack(_map(), Ws, grads, (False, True, False))
    none = AngularMuownMap(
        Rule((), _Polar(0.5), scale="one"),
        warmup=0,
        c=0.0,
        eta=ETA,
        sphere="row",
        tangent=False,
        native_norm=False,
        radius=M_START,
        magnitude_shape="none",
    )
    ref = _run_stack(none, Ws, grads, None)
    for (wa, _), (wb, _) in zip(got, ref):
        assert torch.equal(wa[1], wb[1])


@pytest.mark.parametrize("turn_norm", ["frobenius", "spectral"])
def test_the_mask_and_the_spectral_turn_compose(turn_norm):
    Ws, grads = _stack()
    a = _run_stack(_map(turn_norm=turn_norm), Ws, grads, (True, False, False))
    b = _run_stack(_map(turn_norm=turn_norm), Ws, grads, (False, False, False))
    for (wa, sa), (wb, sb) in zip(a, b):
        assert torch.equal(wa[1:], wb[1:]) and torch.equal(sa[1:], sb[1:])
        assert torch.equal(sa[0, :, 8], a[0][1][0, :, 8])
    assert (
        float(
            ((a[0][1][0, :, 8] - M_START * Ws[0].norm(dim=-1)) / Ws[0].norm(dim=-1))
            .abs()
            .max()
        )
        < 1e-14
    )


# ---------------------------------------------------------------- settings
@pytest.mark.parametrize(
    "bad,match",
    [
        (
            dict(magnitude_shape="none", magnitude_lr=1.0),
            "frozen_units freezes per-row lengths",
        ),
        (dict(magnitude_shape="unit"), "frozen_units freezes per-row lengths"),
        (
            dict(magnitude_rule="log_rms"),
            "frozen_units runs with magnitude_rule adam or sign",
        ),
        (
            dict(magnitude_rule="polar", magnitude_lr=1.0),
            "frozen_units runs with magnitude_rule adam or sign",
        ),
        (dict(frozen_units=(Q, Q)), "frozen_units: distinct unit types"),
        (dict(frozen_units=("",)), "frozen_units: distinct unit types"),
    ],
)
def test_settings_the_freeze_does_not_use_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        _map(**bad)


def test_a_mask_is_required():
    Ws, grads = _stack(steps=1)
    with pytest.raises(ValueError, match="frozen_units needs the stack's mask"):
        _run_stack(_map(), Ws, grads, None)
    with pytest.raises(ValueError, match="frozen_units needs the stack's mask"):
        _run_stack(_map(), Ws, grads, (True, False))


def test_the_preset_builds_it():
    cm = rule_of(
        {
            "rule": "angular_muown",
            "args": {
                "compute": "svd",
                "scale": "one",
                "warmup": 0,
                "c": 0.0,
                "native_norm": False,
                "tangent": False,
                "radius": M_START,
                "magnitude_lr": MAG_LR,
                "frozen_units": [Q, W3],
            },
        }
    ).compute
    assert cm.frozen_units == (Q, W3) and cm.state_shape(12, 8) == (12, 11)


# ---------------------------------------------------------------- unchanged without it
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
    *(
        ({**_F, "sphere": s, "magnitude_shape": g, "tangent": t}, r)
        for s, g, t in itertools.product(
            ("row", "matrix"), ("unit", "row"), (False, True)
        )
        for r in ("adam", "sign")
    ),
    *(
        ({**_F, "sphere": "row", "tangent": False}, r)
        for r in (
            "log_rms",
            "induced",
            "polar",
            "polar_t",
            "v3b",
            "cosine",
            "cosine_raw",
        )
    ),
    ({**_F, "sphere": "row", "magnitude_shape": "none", "tangent": False}, "adam"),
    *(
        ({**_F, "sphere": "row", "tangent": t, "turn_norm": "spectral"}, r)
        for t in (False, True)
        for r in ("adam", "sign")
    ),
    ({**_F, "sphere": "row", "tangent": False, "turn_norm": "spectral"}, "cosine"),
    (
        {
            **_F,
            "sphere": "row",
            "magnitude_shape": "none",
            "tangent": False,
            "turn_norm": "spectral",
        },
        "adam",
    ),
]


@pytest.mark.parametrize("args,rule", _CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute", ["svd", "newtonschulz5"])
def test_every_rule_is_the_s1_rule_bitwise(args, rule, dtype, compute, monkeypatch):
    """No frozen_units (and no mask in the step context): every rule, the spectral turn included, against the
    frozen copy of the S1 class: stacks of 3 with a zero row, a lost state at step 4, 8 steps."""
    extra = {}
    if args.get("magnitude_shape") != "none":
        extra["magnitude_rule"] = rule
        lr_mag = {"adam": 0.3, "sign": 0.3, "log_rms": 0.3, "rel_rms": 0.2}.get(rule)
        if lr_mag is not None:
            extra["magnitude_lr"] = lr_mag
    live = rule_of(
        {"rule": "angular_muown", "args": {"compute": compute, **args, **extra}}
    ).compute
    assert live.frozen_units == ()
    # the copy is 9c856e1b's class (k = 5); the coordinator set k = 20 on 10-05, the class is otherwise unchanged
    monkeypatch.setattr(_angular_muown_s5_reference, "_SPECTRAL_POWER_ITERS", 20)
    c = live
    old = S5AngularMuownMap(
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
        c.turn_norm,
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


# ---------------------------------------------------------------- in DiSCO: fused parameters
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


D, H, KV, HD, FF = (
    8,
    2,
    2,
    4,
    16,
)  # q / k / v runs interleave per KV group (one state shape per wqkv)
NAMES = [
    "layers.0.attention.qkv_linear.wqkv.weight",
    "layers.0.attention.wo.weight",
    "layers.0.feed_forward.w13.weight",
    "layers.0.feed_forward.w2.weight",
]
LAYOUTS = {
    NAMES[0]: ParamLayout("matrix", units=qkv_units(H, KV, HD, D)),
    NAMES[1]: ParamLayout("matrix"),
    NAMES[2]: ParamLayout("matrix", units=stacked_units(("w1", "w3"), FF, D)),
    NAMES[3]: ParamLayout("matrix"),
}
SHAPES = [((H + 2 * KV) * HD, D), (D, H * HD), (2, FF, D), (D, FF)]


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.randn(*s, generator=g) * 0.3) for s in SHAPES]


def _disco(params, frozen_units=()):
    args = {
        "compute": "svd",
        **_F,
        "sphere": "row",
        "tangent": False,
        "magnitude_lr": MAG_LR,
    }
    if frozen_units:
        args["frozen_units"] = list(frozen_units)
    return DiSCO(
        [
            {
                "params": params,
                "param_names": NAMES,
                "momentum": 1.0,
                "nesterov": False,
                "weight_decay": 0.0,
            }
        ],
        parallel_dims=_NoParallel(),
        lr=LR,
        rule="angular_muown",
        args=args,
        param_layouts=LAYOUTS,
    )


def _steps(opt, params, seeds):
    for s in seeds:
        g = torch.Generator().manual_seed(100 + s)
        for p, shape in zip(params, SHAPES):
            p.grad = 0.1 * torch.randn(*shape, generator=g)
        opt.step()


def _rows(opt, p, name):
    """The rows of unit `name` of parameter `p` (the layout's run lists) and the unit's position."""
    e = next(x for x in opt.entries if x.param is p)
    u = next(u for u in e.units if u.name == name)
    W = p.detach().reshape(-1, p.shape[-1])
    return torch.cat([W[a : a + n] for a, n in u.rows.runs]), u.pos


@pytest.mark.parametrize("frozen", [(Q,), (K,), (V,), (W1,), (W3,), (Q, W3)])
def test_disco_freezes_exactly_the_named_units(one_rank, frozen):
    pa, pb = _params(), _params()
    a, b = _disco(pa, frozen), _disco(pb)
    start = {}
    for i, name in ((0, "wq"), (0, "wk"), (0, "wv"), (2, "w1"), (2, "w3")):
        rows, _ = _rows(a, pa[i], name)
        start[i, name] = M_START * rows.norm(dim=1)
    _steps(a, pa, range(5))
    _steps(b, pb, range(5))
    full = {Q: (0, "wq"), K: (0, "wk"), V: (0, "wv"), W1: (2, "w1"), W3: (2, "w3")}
    for t, (i, name) in full.items():
        ra, pos = _rows(a, pa[i], name)
        rb, _ = _rows(b, pb[i], name)
        g = a.state[pa[i]]["rule_state"][pos, :, D]
        if t in frozen:
            assert (
                torch.equal(g, start[i, name].float())
                or float((g - start[i, name]).abs().max()) < 1e-6
            ), t
            assert float((ra.norm(dim=1) / start[i, name] - 1).abs().max()) < 1e-6, t
            assert float((ra - rb).abs().max()) > 1e-4, t
        else:
            assert torch.equal(ra, rb), t
    for i in (1, 3):  # wo and w2 are whole parameters, never frozen here
        assert torch.equal(pa[i].detach(), pb[i].detach())


def test_disco_refuses_a_type_that_matches_no_unit(one_rank):
    with pytest.raises(ValueError, match="frozen_units .* match no unit"):
        _disco(_params(), ("attention.qkv_linear.wz",))


def test_a_resumed_run_with_frozen_units_is_the_continuous_one(one_rank):
    pa = _params()
    a = _disco(pa, (Q, W3))
    _steps(a, pa, range(3))
    saved = [p.detach().clone() for p in pa]
    sd = copy.deepcopy(a.state_dict())
    _steps(a, pa, range(3, 5))
    pb, pc = [nn.Parameter(s.clone()) for s in saved], [
        nn.Parameter(s.clone()) for s in saved
    ]
    b, c = _disco(pb, (Q, W3)), _disco(pc, (Q, W3))
    b.load_state_dict(sd)
    _steps(b, pb, range(3, 5))
    _steps(c, pc, range(3, 5))
    assert all(torch.equal(x.detach(), y.detach()) for x, y in zip(pa, pb))
    assert not all(torch.equal(x.detach(), y.detach()) for x, y in zip(pa, pc))
