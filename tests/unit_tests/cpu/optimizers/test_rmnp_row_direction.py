# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RMNP in DiSCO: RN(M) = each row of M over d_in to unit l2 (`row-l2`) in place of
the orthogonaliser.
- native: rule muon, row-l2, scale `rms_parity` = min(1, sqrt(m/n)), so
  ||RN(M) rms_parity||_F = sqrt(m) min(1, sqrt(m/n)) = ||polar(M) sqrt(m/n)||_F
  (scion's spectral scale) on every shape.
- hyperball and amphione with a row compute take RN as their direction (no
  flag, no fallback to a plain row rule): hyperball W+ = r N(W - lr r RN(M) /
  ||RN(M)||_F), r = ||W||_F; with `exact_angle` the angle (W, W+) is lr.
References from these equations, float64; RowL2 runs in fp32, so they agree to ~1e-6."""

import math

import pytest
import torch

from torchtitan.optimizers.disco.presets import AmphioneMap, FrobeniusAUS, RowL2
from torchtitan.optimizers.disco.rules import rule_of, SCALES, StepCtx

LR = 0.05
SHAPES = [(12, 8), (8, 12), (8, 8)]


@pytest.fixture(autouse=True, scope="module")
def _return_the_compile_budget():
    """RowL2 compiles once per shape / dtype / scale here; the rules share one recompile budget (rules.py
    _COMPILE_LIMIT), so give it back to the files that run after this one."""
    yield
    torch._dynamo.reset()


def _polar(X):
    u, _, vh = torch.linalg.svd(X, full_matrices=False)
    return u @ vh


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


# ---------------------------------------------------------------- hyperball / hyperball-exact with RN
def _hyper(exact, **kw):
    return _rule("hyperball", compute="row-l2", exact_angle=exact, **kw).compute


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


@pytest.mark.parametrize("name", ["hyperball", "amphione"])
def test_a_row_compute_is_the_direction(name):
    """No flag: amphione and hyperball with row-l2 are themselves with RN as the
    direction (amphione + row-l2 is the conv group's rule), never a plain row rule."""
    rule = _rule(name, compute="row-l2")
    want = AmphioneMap if name == "amphione" else FrobeniusAUS
    assert isinstance(rule.compute, want), rule
    direction = rule.compute.direction if name == "amphione" else rule.compute.inner
    assert isinstance(direction.compute, RowL2) and direction.scale == "one"


@pytest.mark.parametrize("name", ["hyperball", "amphione"])
def test_row_direction_is_no_arg(name):
    with pytest.raises(ValueError, match="unknown args"):
        _rule(name, compute="row-l2", row_direction=True)
