# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`row-l2-deflate` (PREREG "RMNP follow-ups at 84M" (c); optimizer-study/tools/review/r72-read8a.md): per unit, with M
the momentum (m x n, rows = outputs, over d_in):
    v0 = normalise(1^T M);  v <- normalise(M^T (M v)) twice;  D = RowL2(M - (M v) v^T)
then the cell's scale. A unit compute (the power iteration needs every row of the unit). References from these
equations in float64; the compute runs in fp32 (as RowL2), so they agree to ~1e-6."""

import math
import os
import sys

import pytest
import torch
from torch import nn

from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import (
    AngularMuownMap,
    compute_of,
    RowL2Deflate,
)
from torchtitan.optimizers.disco.rules import rule_of, StepCtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rmnp_helpers import _data, B1, BETA, F_ETA, F_ETA_DICT, LR, M_START  # noqa: E402
from test_angular_muown_freeze_units import (  # noqa: E402,F401  # noqa: E402
    _NoParallel,
    LAYOUTS,
    NAMES,
    one_rank,
    SHAPES as FUSED_SHAPES,
)

SHAPES = [(12, 8), (8, 12), (8, 8)]


@pytest.fixture(autouse=True, scope="module")
def _return_the_compile_budget():
    """The rules share one recompile budget (rules.py _COMPILE_LIMIT): give it back to later files."""
    yield
    torch._dynamo.reset()


def _deflate(M, iters=2):
    v = M.sum(dim=0)
    v = v / v.norm()
    for _ in range(iters):
        v = M.T @ (M @ v)
        v = v / v.norm()
    Md = M - torch.outer(M @ v, v)
    n = Md.norm(dim=1, keepdim=True)
    return torch.where(n > 0, Md / n.clamp_min(1e-300), 0.0), v


def _rn(M):
    return M / M.norm(dim=1, keepdim=True)


def _top_share(X):
    s = torch.linalg.svdvals(X)
    return float(s[0] ** 2 / (s**2).sum())


def _rule(**kw):
    return rule_of({"rule": "muon", "args": {"compute": "row-l2-deflate", **kw}})


@pytest.mark.parametrize("m,n", SHAPES)
def test_the_compute_is_its_definition(m, n):
    gen = torch.Generator().manual_seed(m * n)
    M = torch.randn(3, m, n, generator=gen, dtype=torch.float64)
    M[1] += 3.0 * torch.outer(
        torch.randn(m, generator=gen, dtype=torch.float64),
        torch.randn(n, generator=gen, dtype=torch.float64),
    )
    c = compute_of("row-l2-deflate")
    assert isinstance(c, RowL2Deflate) and c.kind == "unit" and c.iters == 2
    got = _rule(scale="one").run_whole(M).update
    for j in range(3):
        assert float((got[j] - _deflate(M[j])[0]).abs().max()) < 1e-5


@pytest.mark.parametrize("m,n", [(64, 32), (32, 64)])
def test_a_rank1_spike_is_removed_and_rows_are_unit(m, n):
    """M = a u mu^T + noise: RN(M) keeps the spike (top share near 1), D falls to the noise's level."""
    gen = torch.Generator().manual_seed(m + n)
    noise = torch.randn(m, n, generator=gen, dtype=torch.float64)
    M = (
        8.0
        * torch.outer(
            torch.randn(m, generator=gen, dtype=torch.float64).abs() + 0.5,
            torch.randn(n, generator=gen, dtype=torch.float64),
        )
        + noise
    )
    D = _rule(scale="one").run_whole(M).update
    assert _top_share(_rn(M)) > 0.8
    assert _top_share(D) < 1.5 * _top_share(_rn(noise))
    assert float((D.norm(dim=1) - 1).abs().max()) < 1e-6


def test_without_a_rank1_part_it_is_rn_off_one_direction():
    """An orthogonal M: D = RowNorm(RN(M) with v projected out), D v = 0, n - 1 equal singular values."""
    n = 16
    M = torch.linalg.qr(
        torch.randn(
            n, n, generator=torch.Generator().manual_seed(3), dtype=torch.float64
        )
    )[0]
    D = _rule(scale="one").run_whole(M).update
    _, v = _deflate(M)
    P = _rn(M) - torch.outer(_rn(M) @ v, v)
    assert float((D - _rn(P)).abs().max()) < 1e-5
    assert float((D @ v).abs().max()) < 1e-5
    s = torch.linalg.svdvals(D)
    assert float(s[-1]) < 1e-5 and float(s[:-1].max() / s[:-1].min()) < 1.5


def test_a_zero_row_stays_zero():
    M = torch.randn(
        8, 12, generator=torch.Generator().manual_seed(4), dtype=torch.float64
    )
    M[3] = 0
    D = _rule(scale="one").run_whole(M).update
    assert float(D[3].abs().max()) == 0.0 and torch.isfinite(D).all()


def test_rms_parity_scales_it_once():
    M = torch.randn(
        8, 16, generator=torch.Generator().manual_seed(5), dtype=torch.float64
    )
    got = _rule(scale="rms_parity").run_whole(M).update
    assert float((got - _deflate(M)[0] * math.sqrt(8 / 16)).abs().max()) < 1e-5


# ---------------------------------------------------------------- in DiSCO: per unit of fused parameters
def test_rmnp_wd_deflates_each_unit_of_fused_parameters(one_rank):  # noqa: F811
    """Step 1 (momentum = a positive multiple of G per row): every unit's update is rms_parity D(G_unit), the
    deflation taken per unit (q / k / v of wqkv, w1 / w3 of w13), not over the fused parameter; coupled wd."""
    g = torch.Generator().manual_seed(0)
    ps = [nn.Parameter(torch.randn(*s, generator=g)) for s in FUSED_SHAPES]
    W0 = [p.detach().clone().double() for p in ps]
    lr, wd = 0.03125, 0.003906
    opt = DiSCO(
        [
            {
                "params": ps,
                "param_names": NAMES,
                "momentum": 0.05,
                "nesterov": True,
                "weight_decay": wd,
            }
        ],
        parallel_dims=_NoParallel(),
        lr=lr,
        rule="muon",
        args={"pre": "identity", "compute": "row-l2-deflate", "scale": "rms_parity"},
        param_layouts=LAYOUTS,
    )
    G = [torch.randn(*s, generator=g) for s in FUSED_SHAPES]
    for p, gg in zip(ps, G):
        p.grad = gg.clone()
    opt.step()
    for e in opt.entries:
        i = next(k for k, q in enumerate(ps) if q is e.param)
        w2, g2 = W0[i].reshape(-1, W0[i].shape[-1]), G[i].double().reshape(
            -1, G[i].shape[-1]
        )
        u = ((1 - lr * wd) * w2 - ps[i].detach().double().reshape(w2.shape)) / lr
        for un in e.units:
            rows = [r for a, k in un.rows.runs for r in range(a, a + k)]
            m, n = un.shape
            want = _deflate(g2[rows])[0] * min(1.0, math.sqrt(m / n))
            assert float((u[rows] - want).abs().max()) < 1e-4, (i, un.name)
        if len(e.units) > 1:  # per unit, not over the fused parameter
            assert (
                float(
                    (
                        u
                        - _deflate(g2)[0]
                        * min(1.0, math.sqrt(g2.shape[0] / g2.shape[1]))
                    )
                    .abs()
                    .max()
                )
                > 1e-3
            )


# ---------------------------------------------------------------- R3: the candidate's row sphere with deflation
def _f_ref(W0, grads, *, lr=LR, mlr=0.5):
    """F's row sphere, tangent off, Adam lengths, O = D(F's Nesterov sum momentum of g G)."""
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
        O = _deflate(GU + BETA * mom)[0]
        O = O * U.norm() / O.norm()
        V = U - turn * O
        U = V / V.norm(dim=1, keepdim=True)
        m_ = B1 * m_ + (1 - B1) * gamma
        v_ = 0.95 * v_ + 0.05 * gamma * gamma
        g = g - lr * mlr * (m_ / (1 - B1**t)) / ((v_ / (1 - 0.95**t)).sqrt() + 1e-8)
        out.append(g[:, None] * U)
    return out


@pytest.mark.parametrize("m,n", SHAPES)
def test_f_row_row_off_with_deflation_is_its_definition(m, n):
    """A unit direction inside angular_muown: no row_direction flag; sphere, momentum and lengths kept."""
    W, grads = _data(m, n, seed=m + 5 * n)
    cm = rule_of(
        {
            "rule": "angular_muown",
            "args": {
                "compute": "row-l2-deflate",
                "scale": "one",
                "warmup": 0,
                "c": 0.0,
                "native_norm": False,
                "radius": M_START,
                "tangent": False,
                "eta": F_ETA_DICT,
                "sphere": "row",
                "magnitude_lr": 0.5,
            },
        }
    ).compute
    assert isinstance(cm, AngularMuownMap) and isinstance(
        cm.direction.compute, RowL2Deflate
    )
    w, s = W[None], W.new_zeros(1, *cm.state_shape(m, n))
    for t, (G, ref) in enumerate(zip(grads, _f_ref(W, grads)), start=1):
        res = cm.run(G[None], w, s, StepCtx(LR, 0.0, t))
        w, s = res.weight, res.state
        assert float((w[0] - ref).abs().max()) < 1e-5, t


def test_the_row_direction_flag_stays_for_row_computes():
    with pytest.raises(ValueError, match="row_direction needs a row compute"):
        rule_of(
            {
                "rule": "angular_muown",
                "args": {"compute": "row-l2-deflate", "row_direction": True},
            }
        )
