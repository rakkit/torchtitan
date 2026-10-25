# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RMNP in DiSCO (PREREG r69; optimizer-study/tools/review/r69-rmnp-84m.md §2): RN(M) = each row of M over d_in to
unit l2 (`row-l2`) in place of the orthogonaliser.
- R1 native: rule muon, row-l2, scale `rms_parity` = min(1, sqrt(m/n)), so ||RN(M) rms_parity||_F = sqrt(m) min(1,
  sqrt(m/n)) = ||polar(M) sqrt(m/n)||_F (scion's spectral scale) on every shape.
- `row_direction`: hyperball / hyperball-exact / elr and angular_muown keep their sphere, angle, momentum and lengths
  with RN as the direction:
  hyperball: W+ = r N(W - eta r RN(M) / ||RN(M)||_F), r = ||W||_F; exact: the angle (W, W+) is eta.
  F row sphere (tangent off): D = F's Nesterov sum momentum of g G, O = RN(D) (||O||_F = ||U||_F, so F's rescale is
  1), U+ = RowNorm(U - turn O): each row turns by atan(turn sqrt(1 - c^2) / (1 - turn c)), c = <U_i, O_i>.
References from these equations, float64; RowL2 runs in fp32, so trajectories agree to ~1e-6."""

import dataclasses
import math
import os
import sys

import pytest
import torch

from torchtitan.optimizers.disco.presets import AngularMuownMap, FrobeniusAUS, RowL2
from torchtitan.optimizers.disco.rules import rule_of, SCALES, StepCtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rmnp_helpers import (  # noqa: E402
    _data,
    _polar,
    B1,
    BETA,
    F_ETA,
    F_ETA_DICT,
    LR,
    M_START,
)

SHAPES = [(12, 8), (8, 12), (8, 8)]
F = {
    "scale": "one",
    "warmup": 0,
    "c": 0.0,
    "native_norm": False,
    "radius": M_START,
    "tangent": False,
    "eta": F_ETA_DICT,
}


@pytest.fixture(autouse=True, scope="module")
def _return_the_compile_budget():
    """RowL2 compiles once per shape / dtype / scale here; the rules share one recompile budget (rules.py
    _COMPILE_LIMIT), so give it back to the files that run after this one."""
    yield
    torch._dynamo.reset()


def _rn(X):
    return X / X.norm(dim=-1, keepdim=True)


def _rule(name, **args):
    return rule_of({"rule": name, "args": args})


# ---------------------------------------------------------------- R1: native RMNP and its scale
@pytest.mark.parametrize("m,n", SHAPES)
def test_native_rmnp_is_rn_times_rms_parity(m, n):
    M = torch.randn(
        m, n, generator=torch.Generator().manual_seed(m * n), dtype=torch.float64
    )
    rule = _rule("muon", compute="row-l2", scale="rms_parity")
    got = rule.run_whole(M).update
    assert float((got - _rn(M) * min(1.0, math.sqrt(m / n))).abs().max()) < 1e-6
    scion = _polar(M) * math.sqrt(m / n)  # our scion: polar x spectral
    assert abs(float(got.norm()) / float(scion.norm()) - 1) < 1e-6
    assert SCALES["rms_parity"](m, n) == min(1.0, math.sqrt(m / n))


# ---------------------------------------------------------------- hyperball / hyperball-exact / elr with RN
def _hyper(exact, **kw):
    return _rule(
        "hyperball", compute="row-l2", row_direction=True, exact_angle=exact, **kw
    ).compute


@pytest.mark.parametrize("m,n", SHAPES)
def test_hyperball_with_rn_is_its_definition(m, n):
    gen = torch.Generator().manual_seed(m + n)
    W = torch.randn(3, m, n, generator=gen, dtype=torch.float64)
    M = torch.randn(3, m, n, generator=gen, dtype=torch.float64)
    cm = _hyper(False)
    assert isinstance(cm, FrobeniusAUS) and isinstance(cm.inner.compute, RowL2)
    res = cm.run(M, W, None, StepCtx(LR, 0.0, 1))
    got = res.alpha[:, None, None] * W - res.update
    for j in range(3):
        r, D = W[j].norm(), _rn(M[j])
        Y = W[j] - LR * r * D / D.norm()
        assert float((got[j] - r * Y / Y.norm()).abs().max()) < 1e-5


@pytest.mark.parametrize("m,n", SHAPES)
def test_hyperball_exact_with_rn_turns_by_eta_exactly(m, n):
    gen = torch.Generator().manual_seed(2 * m + n)
    W = torch.randn(3, m, n, generator=gen, dtype=torch.float64)
    M = (
        torch.randn(3, m, n, generator=gen, dtype=torch.float64) + 0.5 * W
    )  # RN(M) leans on W
    eta = 0.05
    for exact in (True, False):
        res = _hyper(exact).run(M, W, None, StepCtx(eta, 0.0, 1))
        new = res.alpha[:, None, None] * W - res.update
        cos = (W * new).sum(dim=(-2, -1)) / (
            W.norm(dim=(-2, -1)) * new.norm(dim=(-2, -1))
        )
        angle = torch.arccos(cos.clamp(-1, 1))
        if exact:
            assert float((angle - eta).abs().max()) < 1e-5, angle
        else:  # the approximate step turns by less when RN(M) has a radial part
            assert float((angle - eta).abs().max()) > 1e-3, angle
        assert (
            float((new.norm(dim=(-2, -1)) / W.norm(dim=(-2, -1)) - 1).abs().max())
            < 1e-6
        )


def test_elr_takes_the_flag_too():
    cm = _rule("elr", compute="row-l2", row_direction=True).compute
    assert (
        isinstance(cm, FrobeniusAUS)
        and isinstance(cm.inner.compute, RowL2)
        and cm.radius == "free"
    )


# ---------------------------------------------------------------- F row sphere (R2r frozen twin, R3 candidate)
def _f_cell(shape, **kw):
    return _rule(
        "angular_muown",
        compute="row-l2",
        row_direction=True,
        sphere="row",
        magnitude_shape=shape,
        **{**F, **kw}
    ).compute


def _ref(W0, grads, *, learn, lr=LR, mlr=0.5):
    """F's row sphere, tangent off, from its equations with O = RN(D); lengths frozen at M g0, or per-row Adam."""
    g_at = W0.norm(dim=1)
    U, g = W0 / g_at[:, None], M_START * g_at
    mom, m_, v_, out = (
        torch.zeros_like(W0),
        torch.zeros_like(g),
        torch.zeros_like(g),
        [],
    )
    for t, G in enumerate(grads, start=1):
        turn = lr * F_ETA(t)
        gamma = (G * U).sum(dim=1)
        GU = g[:, None] * G
        mom = BETA * mom + GU
        O = _rn(GU + BETA * mom)
        O = O * U.norm() / O.norm()  # F's rescale: 1 for unit rows
        V = U - turn * O
        U_new = V / V.norm(dim=1, keepdim=True)
        if learn:
            m_ = B1 * m_ + (1 - B1) * gamma
            v_ = 0.95 * v_ + 0.05 * gamma * gamma
            g = g - lr * mlr * (m_ / (1 - B1**t)) / (
                (v_ / (1 - 0.95**t)).sqrt() + 1e-8
            )
        out.append((g[:, None] * U_new, U, U_new, O, turn))
        U = U_new
    return out


def _run(cm, W, grads):
    m, n = W.shape
    w, s, out = W[None], W.new_zeros(1, *cm.state_shape(m, n)), []
    for t, G in enumerate(grads, start=1):
        res = cm.run(G[None], w, s, StepCtx(LR, 0.0, t))
        w, s = res.weight, res.state
        out.append(w[0])
    return out


@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("shape,learn", [("none", False), ("row", True)])
def test_f_row_sphere_with_rn_is_its_definition(m, n, shape, learn):
    """R2r (f-row-none-off) and R3 (f-row-row-off, Adam lengths): sphere, momentum and lengths kept."""
    W, grads = _data(m, n, seed=m + 3 * n)
    cm = _f_cell(shape, **({"magnitude_lr": 0.5} if learn else {}))
    assert isinstance(cm, AngularMuownMap) and isinstance(cm.direction.compute, RowL2)
    got = _run(cm, W, grads)
    ref = _ref(W, grads, learn=learn)
    assert max(float((a - r[0]).abs().max()) for a, r in zip(got, ref)) < 1e-5


@pytest.mark.parametrize("m,n", SHAPES)
def test_frozen_twin_rows_turn_by_the_approximate_angle(m, n):
    """Each row: angle(U_i, U+_i) = atan(turn sqrt(1 - c_i^2) / (1 - turn c_i)), c_i = <U_i, O_i>, |O_i| = 1."""
    W, grads = _data(m, n, seed=7 * m + n)
    got = _run(_f_cell("none"), W, grads)
    for w, (_, U, _, O, turn) in zip(got, _ref(W, grads, learn=False)):
        Un = w / w.norm(dim=1, keepdim=True)
        angle = torch.arccos(((U * Un).sum(dim=1) / U.norm(dim=1)).clamp(-1, 1))
        c = (U * O).sum(dim=1)
        want = torch.atan2(turn * (1 - c * c).clamp_min(0).sqrt(), 1 - turn * c)
        assert float((angle - want).abs().max()) < 1e-5


def test_the_f_corner_is_hyperball_with_rn():
    """f-mat-none-off with the flag: the preset's hyperball corner keeps RN."""
    cm = _rule(
        "angular_muown",
        compute="row-l2",
        row_direction=True,
        sphere="matrix",
        magnitude_shape="none",
        **{**F, "radius": 1.0}
    ).compute
    assert isinstance(cm, FrobeniusAUS) and isinstance(cm.inner.compute, RowL2)


# ---------------------------------------------------------------- the flag's contract
@pytest.mark.parametrize(
    "name,extra", [("hyperball", {}), ("elr", {}), ("angular_muown", F)]
)
def test_flag_on_needs_a_row_compute(name, extra):
    with pytest.raises(ValueError, match="row_direction needs a row compute"):
        _rule(name, **{**extra, "compute": "newtonschulz5", "row_direction": True})


@pytest.mark.parametrize(
    "name,extra", [("hyperball", {}), ("elr", {}), ("angular_muown", F)]
)
def test_flag_off_a_row_compute_is_still_the_whole_rule(name, extra):
    """An embedding group inheriting the preset keeps its plain row rule."""
    rule = _rule(name, **{**extra, "compute": "row-l2", "scale": "sqrt_n"})
    assert (
        rule.kind == "rows"
        and isinstance(rule.compute, RowL2)
        and rule.scale == "sqrt_n"
    )


@pytest.mark.parametrize(
    "name,args",
    [
        (
            "hyperball",
            dict(compute="svd", eta=F_ETA_DICT, radius_ratio=[[0, 1.0], [1, 4.0]]),
        ),
        ("hyperball", dict(compute="svd", exact_angle=True)),
        ("elr", dict(compute="svd")),
        (
            "angular_muown",
            dict(
                compute="svd",
                **F,
                sphere="row",
                magnitude_shape="row",
                magnitude_lr=0.5
            ),
        ),
        (
            "angular_muown",
            dict(compute="svd", **F, sphere="row", magnitude_shape="none"),
        ),
        (
            "angular_muown",
            dict(
                compute="svd",
                **{**F, "radius": 4.0},
                sphere="matrix",
                magnitude_shape="none"
            ),
        ),
        ("angular_muown", dict(compute="svd", scale="image_spectral")),
    ],
)
def test_flag_off_builds_the_same_rule(name, args):
    a, b = _rule(name, **args), _rule(name, **args, row_direction=False)
    assert a == b and dataclasses.is_dataclass(a.compute)
