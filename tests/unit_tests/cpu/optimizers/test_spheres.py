# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""hyperball (MuonH) against its paper's equation, in float64."""

import math

import pytest
import torch

from torchtitan.optimizers.disco.presets import FrobeniusAUS, RowL2
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


def test_what_it_declares():
    r = _rule("hyperball")
    assert isinstance(r.compute, FrobeniusAUS)
    assert (r.kind, r.needs_weight, r.has_state, r.output) == (
        "unit",
        True,
        False,
        "update_coeffs",
    )
    assert r.compute.keeps_radius is True
    # A row compute is hyperball with that direction, not a plain row rule.
    row = _rule("hyperball", compute="row-l2")
    assert isinstance(row.compute, FrobeniusAUS)
    assert isinstance(row.compute.inner.compute, RowL2)
    with pytest.raises(ValueError, match="Gram"):
        _rule("hyperball", compute="gram_polar")
    with pytest.raises(ValueError, match="unknown args"):
        _rule("hyperball", scale="spectral")  # N() removes a scale: there is none


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


def test_exact_angle_hits_the_target():
    """The angle between W and W+ is lr, also where the direction is far from
    tangent (c = 0.5 here), where plain lr misses it."""
    W, M = _pair(4, 24, 16, 5, align=3.0)
    lr, wd, name = 0.05, 0.0, "hyperball"
    exact = _rule(name, exact_angle=True).run_whole(M, W, StepCtx(lr, wd, 2))
    plain = _rule(name).run_whole(M, W, StepCtx(lr, wd, 2))
    got = _angle(W.double(), _apply(exact, W).double())
    torch.testing.assert_close(got, torch.full_like(got, lr), rtol=0, atol=1e-6)
    miss = (_angle(W.double(), _apply(plain, W).double()) - lr).abs()
    assert miss.min() > 1e-3  # the case is not trivially tangent
    r = W.double().norm(dim=(-2, -1))
    torch.testing.assert_close(
        _apply(exact, W).double().norm(dim=(-2, -1)), r, rtol=1e-6, atol=0
    )


def test_batched_equals_single_and_zero_momentum():
    rule = _rule("hyperball")
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
    assert torch.all(zero.alpha == 1.0)  # no direction: W stays


def test_the_decay():
    assert StepCtx(0.1, 0.01, 1).decay == 1.0 - 0.1 * 0.01
