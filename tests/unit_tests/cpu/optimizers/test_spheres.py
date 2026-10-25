# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""hyperball, elr and sso against their papers' equations, in float64."""

import math

import pytest
import torch

from torchtitan.optimizers.disco.presets import (
    EtaSchedule,
    FrobeniusAUS,
    SpectralSphere,
)
from torchtitan.optimizers.disco.rules import rule_of, StepCtx


def _pair(b, m, n, seed, *, align=0.0):
    """Weights with a decaying spectrum and momenta, `align` of W mixed in."""
    g = torch.Generator().manual_seed(seed)
    k = min(m, n)
    q1 = torch.linalg.qr(torch.randn(b, m, m, generator=g))[0][..., :k]
    q2 = torch.linalg.qr(torch.randn(b, n, n, generator=g))[0][..., :k]
    W = (q1 * torch.logspace(0, -1.5, k)) @ q2.mT
    M = torch.randn(b, m, n, generator=g) * 1e-2
    return W, M + align * 1e-2 * W / W.norm(dim=(-2, -1), keepdim=True) * math.sqrt(
        m * n
    )


def _rule(name, **args):
    return rule_of({"rule": name, "args": {"compute": "svd", **args}})


def _apply(res, W):
    return res.alpha[:, None, None] * W - res.update


def _polar(X):
    U, _, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    return U @ Vh


def _angle(A, B):
    c = (A * B).sum(dim=(-2, -1)) / (A.norm(dim=(-2, -1)) * B.norm(dim=(-2, -1)))
    return torch.arccos(c.clamp(-1, 1))


def test_what_they_declare():
    for name, radius in (("hyperball", True), ("elr", False)):
        r = _rule(name)
        assert isinstance(r.compute, FrobeniusAUS)
        assert (r.kind, r.needs_weight, r.has_state, r.output) == (
            "unit",
            True,
            False,
            "update_coeffs",
        )
        assert r.compute.keeps_radius is radius
    s = _rule("sso")
    assert isinstance(s.compute, SpectralSphere) and s.compute.keeps_radius
    assert s.has_state and s.compute.state_shape(24, 16) == (16,)
    # An embedding group inheriting a sphere rule keeps its row rule and scale.
    for name in ("hyperball", "elr", "sso"):
        row = _rule(name, compute="row-l2", scale="sqrt_n")
        assert row.kind == "rows" and row.scale == "sqrt_n"
        with pytest.raises(ValueError, match="Gram"):
            _rule(name, compute="gram_polar")
    with pytest.raises(ValueError, match="no effect"):
        _rule("hyperball", scale="spectral")  # N() removes it on matrices


@pytest.mark.parametrize("tangent", [False, True])
def test_hyperball_is_its_equation(tangent):
    """W+ = r N(W - lr r N(U)), U the direction of M (or of its tangent part)."""
    W, M = _pair(3, 24, 16, 1, align=0.5)
    lr = 0.02
    res = _rule("hyperball", tangent=tangent).run_whole(M, W, StepCtx(lr, 0.0, 2))
    Wd, Md = W.double(), M.double()
    if tangent:
        Wh = Wd / Wd.norm(dim=(-2, -1), keepdim=True)
        Md = Md - (Wh * Md).sum(dim=(-2, -1), keepdim=True) * Wh
    U = _polar(Md)
    r = Wd.norm(dim=(-2, -1), keepdim=True)
    B = Wd - lr * r * U / U.norm(dim=(-2, -1), keepdim=True)
    want = r * B / B.norm(dim=(-2, -1), keepdim=True)
    got = _apply(res, W).double()
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)
    torch.testing.assert_close(got.norm(dim=(-2, -1)), r.flatten(), rtol=1e-6, atol=0)


def test_tangent_is_orthogonal_and_leaves_the_momentum():
    W, M = _pair(2, 20, 12, 3, align=2.0)
    M0 = M.clone()
    _rule("hyperball", tangent=True).run_whole(M, W, StepCtx(0.01, 0.0, 2))
    assert torch.equal(M, M0)
    # The tangent part has no component along W; the plain M here does.
    Wh = W / W.norm(dim=(-2, -1), keepdim=True)
    Mt = M - (Wh * M).sum(dim=(-2, -1), keepdim=True) * Wh
    assert (Wh * Mt).sum(dim=(-2, -1)).abs().max() < 1e-7
    assert (Wh * M).sum(dim=(-2, -1)).abs().min() > 1e-3


def test_elr_is_its_equation():
    """ELR Eq. 2: W+ = (1 - lr wd) W - lr ||W||_F N(U)."""
    W, M = _pair(3, 16, 24, 4)
    lr, wd = 0.03, 0.5
    res = _rule("elr").run_whole(M, W, StepCtx(lr, wd, 2))
    Wd = W.double()
    U = _polar(M)
    want = (1 - lr * wd) * Wd - lr * Wd.norm(dim=(-2, -1), keepdim=True) * U / U.norm(
        dim=(-2, -1), keepdim=True
    )
    torch.testing.assert_close(_apply(res, W).double(), want, rtol=0, atol=2e-6)


@pytest.mark.parametrize("name,wd", [("hyperball", 0.0), ("elr", 0.5)])
def test_exact_angle_hits_the_target(name, wd):
    """The angle between W and W+ is lr, also where the direction is far from
    tangent (c = 0.5 here), where plain lr misses it."""
    W, M = _pair(4, 24, 16, 5, align=3.0)
    lr = 0.05
    exact = _rule(name, exact_angle=True).run_whole(M, W, StepCtx(lr, wd, 2))
    plain = _rule(name).run_whole(M, W, StepCtx(lr, wd, 2))
    got = _angle(W.double(), _apply(exact, W).double())
    torch.testing.assert_close(got, torch.full_like(got, lr), rtol=0, atol=1e-6)
    miss = (_angle(W.double(), _apply(plain, W).double()) - lr).abs()
    assert miss.min() > 1e-3  # the case is not trivially tangent
    if name == "hyperball":
        r = W.double().norm(dim=(-2, -1))
        torch.testing.assert_close(
            _apply(exact, W).double().norm(dim=(-2, -1)), r, rtol=1e-6, atol=0
        )


def _sso_reference(W, M, lr, R):
    """SSO per matrix from the paper: exact top pair, exact msign, the root
    of h by bisection on the bracket of their Theorem A.3."""
    out = []
    for Wi, Mi in zip(W.double(), M.double()):
        U, S, Vh = torch.linalg.svd(Wi)
        u, v, sigma = U[:, 0], Vh[0], S[0]
        Mh = Mi / Mi.norm()
        Theta = torch.outer(u, v)
        lo = -2 * torch.linalg.matrix_norm(Mh, ord="nuc")
        hi = -lo
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if (Theta * _polar(Mh + mid * Theta)).sum() < 0:
                lo = mid
            else:
                hi = mid
        Phi = _polar(Mh + 0.5 * (lo + hi) * Theta)
        out.append((R / sigma * Wi - lr * R * Phi, Theta, Phi, 0.5 * (lo + hi)))
    return out


def test_sso_is_its_equation_and_stays_on_the_sphere():
    lr = 1e-3
    W, M = _pair(3, 24, 16, 6, align=1.0)
    rule = _rule("sso", sigma_iters=60, solve_steps=40)
    res = rule.run_whole(M, W, StepCtx(lr, 0.0, 2))
    R = math.sqrt(24 / 16)
    got = _apply(res, W).double()
    for i, (want, Theta, Phi, _) in enumerate(_sso_reference(W, M, lr, R)):
        torch.testing.assert_close(got[i], want, rtol=0, atol=1e-6)
        assert abs((Theta * Phi).sum().item()) < 1e-6  # tangent
        # On the sphere to second order: |sigma(W+) - R| = O(lr^2).
        assert abs(torch.linalg.matrix_norm(got[i], ord=2).item() - R) < 10 * lr**2
    assert res.state.shape == (3, 16)


def _sso_root(rule, W, M):
    """lambda from the rule's own solve, with an exact top pair."""
    from torchtitan.optimizers.helpers.power_iteration import top_pair_warm

    _, u, v = top_pair_warm(W, None, 200)
    Mh = M / M.norm(dim=(-2, -1), keepdim=True)
    return rule.compute.solve(Mh, u, v)[1]


def test_sso_root_on_both_sides():
    """lambda* of both signs is bracketed and found."""
    rule = _rule("sso", solve_steps=40)
    signs = set()
    for seed in range(8):
        W, M = _pair(1, 16, 12, 10 + seed, align=(-1) ** seed * 2.0)
        lam = _sso_root(rule, W, M).item()
        signs.add(math.copysign(1, lam))
        ref = _sso_reference(W, M, 1e-3, math.sqrt(16 / 12))[0][3].item()
        assert abs(lam - ref) < 1e-4, (seed, lam, ref)
    assert signs == {-1.0, 1.0}


def test_sso_default_budget_meets_the_paper_tolerance():
    """With the default 12 solve steps, |lambda - lambda*| <= 2e-4 (the
    paper's tolerance) for roots from 1e-2 to 0.57: momenta from random to
    strongly aligned with W's top pair (real hidden matrices: <= 0.03)."""
    rule = _rule("sso")
    roots = []
    for align in (0.0, 0.3, 1.0, 2.0):
        for seed in range(3):
            W, M = _pair(1, 24, 16, 20 + seed, align=align)
            ref = _sso_reference(W, M, 1e-3, math.sqrt(24 / 16))[0][3].item()
            lam = _sso_root(rule, W, M).item()
            assert abs(lam - ref) <= 2e-4, (align, seed, lam, ref)
            roots.append(abs(ref))
    assert min(roots) < 0.02 and max(roots) > 0.5  # the range is covered


@pytest.mark.parametrize("name", ["hyperball", "elr", "sso"])
def test_batched_equals_single_and_zero_momentum(name):
    rule = _rule(name)
    W, M = _pair(3, 12, 20, 7)
    ctx = StepCtx(0.02, 0.0, 3)
    both = rule.run_whole(M, W, ctx)
    for i in range(3):
        one = rule.run_whole(M[i : i + 1], W[i : i + 1], ctx)
        torch.testing.assert_close(
            one.update, both.update[i : i + 1], rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(one.alpha, both.alpha[i : i + 1], rtol=1e-5, atol=0)
    zero = rule.run_whole(torch.zeros_like(M), W, ctx)
    assert torch.all(zero.update == 0)
    if name != "sso":
        assert torch.all(zero.alpha == 1.0)  # no direction: W stays


def test_decoupled_decay():
    assert StepCtx(0.1, 0.01, 1).decay == 1.0 - 0.1 * 0.01
    assert StepCtx(0.1, 0.01, 1, True).decay == 1.0 - 0.01


def test_elr_decoupled_decay():
    """Decoupled weight decay scales W by 1 - wd, not 1 - lr wd."""
    W, M = _pair(2, 16, 24, 8)
    lr, wd = 0.03, 0.01
    res = _rule("elr").run_whole(M, W, StepCtx(lr, wd, 2, True))
    Wd = W.double()
    U = _polar(M)
    want = (1 - wd) * Wd - lr * Wd.norm(dim=(-2, -1), keepdim=True) * U / U.norm(
        dim=(-2, -1), keepdim=True
    )
    torch.testing.assert_close(_apply(res, W).double(), want, rtol=0, atol=2e-6)


# eta(t) = (1 + 0.1 (t - 10))^-1 after step 10: 1 at t = 5, 1/2 at t = 20
SCHED = {"shape": "inverse_poly", "start": 10, "c": 0.1, "p": 1.0}


@pytest.mark.parametrize("name", ["hyperball", "elr"])
def test_eta_schedule_multiplies_the_angle(name):
    """With `eta` a schedule, the exact angle is lr eta(t)."""
    W, M = _pair(3, 24, 16, 9, align=1.0)
    rule = _rule(name, exact_angle=True, eta=SCHED)
    lr = 0.03
    for t, want in ((5, lr), (20, lr / 2)):
        res = rule.run_whole(M, W, StepCtx(lr, 0.0, t))
        got = _angle(W.double(), _apply(res, W).double())
        torch.testing.assert_close(got, torch.full_like(got, want), rtol=0, atol=1e-6)


def test_sso_eta_schedule_multiplies_the_step():
    """SSO's step is lr eta(t) R orth(...): spectral norm lr eta(t) R."""
    W, M = _pair(2, 24, 16, 11)
    rule = _rule("sso", eta=SCHED)
    res = rule.run_whole(M, W, StepCtx(0.008, 0.0, 20))
    R = math.sqrt(24 / 16)
    step = torch.linalg.matrix_norm(res.update.double(), ord=2)
    torch.testing.assert_close(
        step, torch.full_like(step, 0.004 * R), rtol=1e-3, atol=0
    )
    for bad in (0.1, "lr", {"kappa": 0.1}, {"shape": "inverse_poly", "total": 50}):
        with pytest.raises(ValueError, match="eta"):
            _rule("sso", eta=bad)


@pytest.mark.parametrize("wd", [0.0, 0.01])
def test_muon_eta_schedule_is_its_equation(wd):
    """muon with `eta` a schedule: W+ = (1 - lr wd) W - lr eta(t) sqrt(m/n) polar(M):
    the schedule scales the step, not the decay; with none it is the plain lr step."""
    W, M = _pair(3, 24, 16, 12)
    rule = _rule("muon", eta=SCHED)
    assert (rule.kind, rule.output, rule.needs_weight) == (
        "unit",
        "update_coeffs",
        False,
    )
    lr = 0.5
    for t, eta in ((5, 1.0), (20, 0.5), (30, 1 / 3)):
        res = rule.run_whole(M, W, StepCtx(lr, wd, t))
        want = (1 - lr * wd) * W.double() - lr * eta * math.sqrt(24 / 16) * _polar(M)
        torch.testing.assert_close(_apply(res, W).double(), want, rtol=0, atol=2e-6)
    plain = _rule("muon").run_whole(M).update.double()
    res = rule.run_whole(M, W, StepCtx(lr, 0.0, 20))
    torch.testing.assert_close(res.update.double(), lr * 0.5 * plain, rtol=0, atol=1e-7)
    res = rule.run_whole(M, W, StepCtx(lr, 0.01, 20, True))  # decoupled: 1 - wd
    torch.testing.assert_close(res.alpha, torch.full_like(res.alpha, 0.99))
    with pytest.raises(ValueError, match="whole-unit"):
        _rule("muon", eta=SCHED, block_rows=8)
    # Groups of another rule never inherit it; a row group under it keeps its lr.
    from torchtitan.optimizers.disco.rules import RULES

    assert "eta" not in RULES["muon"].inherits
    row = _rule("muon", compute="row-l2", scale="sqrt_n", eta=SCHED)
    assert row.kind == "rows" and row.output == "update" and row.scale == "sqrt_n"


def test_eta_schedule_shapes():
    """The two shapes from their definitions: 1 up to start, then
    (1 + c x)^-p, or 1 - sqrt(x / (total - start)), x = t - start."""
    inv = EtaSchedule("inverse_poly", start=100, c=1e-3, p=1.0)
    assert inv(1) == inv(100) == 1.0
    assert math.isclose(inv(1100), 0.5) and math.isclose(inv(3100), 0.25)
    sq = EtaSchedule("sqrt_decay", start=100, total=500)
    assert (
        sq(100) == 1.0
        and math.isclose(sq(200), 0.5)
        and sq(500) == 0.0
        and sq(900) == 0.0
    )
    with pytest.raises(ValueError):
        EtaSchedule("sqrt_decay", start=100, total=100)
    with pytest.raises(ValueError):
        EtaSchedule("cosine")


def test_eta_schedule_linear_and_hold():
    """linear: 1 - x / (total - start); hold freezes eta at its step-`hold`
    value; hold = 0 leaves every shape exactly as defined."""
    lin = EtaSchedule("linear", start=100, total=500)
    assert lin(100) == 1.0 and lin(300) == 0.5 and lin(500) == 0.0 and lin(900) == 0.0
    with pytest.raises(ValueError):
        EtaSchedule("linear", start=100, total=100)
    held = EtaSchedule("inverse_poly", start=100, c=1e-3, p=1.0, hold=1100)
    assert (
        math.isclose(held(1100), 0.5)
        and held(5000) == held(1100)
        and held(600) == held(600)
    )
    with pytest.raises(ValueError):
        EtaSchedule("inverse_poly", hold=100)
    c, p, T = 0.01995, 0.5148, 25600
    for t in (1, 100, 101, 2000, 20480, 25600):
        x = t - 100
        want_inv = 1.0 if x <= 0 else (1.0 + c * x) ** -p
        want_sq = 1.0 if x <= 0 else 1.0 - math.sqrt(min(1.0, x / (T - 100)))
        assert EtaSchedule("inverse_poly", c=c, p=p)(t) == want_inv
        assert EtaSchedule("sqrt_decay", total=T)(t) == want_sq
        hx = min(t, 20480) - 100
        assert EtaSchedule("inverse_poly", c=c, p=p, hold=20480)(t) == (
            1.0 if hx <= 0 else (1.0 + c * hx) ** -p
        )
