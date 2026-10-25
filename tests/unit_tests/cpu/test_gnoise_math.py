# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gnoise_math against closed forms. Linear regression with Gaussian inputs x ~ N(0, I_d)
and y = x . w* + eps, eps ~ N(0, s^2): at w the per-token gradient (x . w - y) x has
mean a = w - w* and covariance Sigma = (|a|^2 + s^2) I + a a^T, so
tr(Sigma) = d (|a|^2 + s^2) + |a|^2 and the signal is |a|^2."""

import itertools  # noqa: F401
import math
import os
import sys

import torch

from torchtitan.optimizers import gnoise_math as gm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu", "gnoise"))
import gn_reference as ref  # noqa: E402

D, SIGMA = 8, 0.5
N_B = [3, 5, 8, 16]  # uneven small batches


def _draw(gen, a, n_b, draws):
    """Per small batch the sum of its per-token gradients, over `draws` repetitions."""
    out = []
    for n in n_b:
        x = torch.randn(draws, n, D, generator=gen, dtype=torch.float64)
        r = x @ a - SIGMA * torch.randn(draws, n, generator=gen, dtype=torch.float64)
        out.append((r[..., None] * x).sum(1))  # [draws, D]
    return out


def test_unbiased_uneven_and_equal_to_reference():
    """T2: means over 10^4 draws equal tr(Sigma) and |a|^2 within 3 SE; each draw equals
    the equation reference; the mean of ratios is not the ratio of means."""
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(D, generator=gen, dtype=torch.float64) * 0.7
    N = sum(N_B)
    sums = _draw(gen, a, N_B, 10_000)
    g = torch.stack([s / N for s in sums], dim=-2)  # [draws, K, D], g_b = sum / N
    sq = g.square().sum(-1)
    F = g.sum(-2).square().sum(-1)
    n = torch.tensor(N_B, dtype=torch.float64)
    est = gm.estimate(sq, F, n, N)
    tr_true = D * (a @ a + SIGMA**2) + a @ a
    for name, true in (("tr", tr_true), ("sig", a @ a)):
        v = est[name]
        se = v.std() / math.sqrt(v.numel())
        assert (
            abs(v.mean() - true) < 3 * se
        ), f"{name}: mean {v.mean():.5f} vs {true:.5f} (SE {se:.5f})"
    for i in range(5):
        tr, s = ref.estimate(list(g[i].unbind(0)), N_B, N)
        assert math.isclose(float(est["tr"][i]), tr, rel_tol=1e-10)
        assert math.isclose(float(est["sig"][i]), s, rel_tol=1e-10)
    ratio = est["tr"] / est["sig"]
    ok = torch.isfinite(ratio) & (est["sig"] > 0)
    mr, rm = ratio[ok].mean(), est["tr"].mean() / est["sig"].mean()
    assert abs(mr - rm) > 3 * ratio[ok].std() / math.sqrt(
        ok.sum()
    ), "mean of ratios == ratio of means"


def test_jackknife_equals_reference():
    gen = torch.Generator().manual_seed(1)
    a = torch.randn(D, generator=gen, dtype=torch.float64)
    N = sum(N_B)
    g = torch.stack([s[0] / N for s in _draw(gen, a, N_B, 1)])  # [K, D]
    # a rank-1 projection: the cross terms are products of scalars
    p = torch.randn(D, generator=gen, dtype=torch.float64)
    p = p / p.norm()
    x = g @ p
    est = gm.estimate(
        (x * x)[None],
        (x.sum() ** 2)[None],
        torch.tensor(N_B, dtype=torch.float64),
        N,
        cross=(x.sum() * x)[None],
    )
    se_tr, se_s = ref.jackknife([v[None] for v in x], N_B, N)
    assert math.isclose(float(est["se_tr"][0]), se_tr, rel_tol=1e-9)
    assert math.isclose(float(est["se_sig"][0]), se_s, rel_tol=1e-9)


def test_absent_small_batches_and_estimability():
    """n_b = 0 is left out; fewer than 2 small batches with tokens is not estimable."""
    sq = torch.tensor([[1.0, 2.0, 0.0, 3.0]], dtype=torch.float64)
    F = torch.tensor([4.0], dtype=torch.float64)
    full = gm.estimate(sq, F, torch.tensor([[2.0, 3.0, 0.0, 5.0]]), 10.0)
    kept = gm.estimate(sq[:, [0, 1, 3]], F, torch.tensor([[2.0, 3.0, 5.0]]), 10.0)
    assert torch.allclose(full["tr"], kept["tr"]) and torch.allclose(
        full["sig"], kept["sig"]
    )
    one = gm.estimate(sq, F, torch.tensor([[2.0, 0.0, 0.0, 0.0]]), 10.0)
    assert torch.isnan(one["tr"]).all() and torch.isnan(one["sig"]).all()


def test_loo_sign_known_answers():
    """T14: equal small batches agree everywhere; on Gaussian coordinates the LOO
    agreement equals p_n p_{N-n} + (1 - p_n)(1 - p_{N-n}), p_B = Phi(|mu| sqrt(B) / s)."""
    d = torch.full((4, 50), 0.3, dtype=torch.float64)
    assert torch.allclose(
        gm.loo_sign_agreement(d, d.sum(0)), torch.ones(4, dtype=torch.float64)
    )
    gen = torch.Generator().manual_seed(2)
    K, n, R, draws = 4, 6, 400, 400
    mu = torch.linspace(-0.4, 0.4, R, dtype=torch.float64)
    vals = []
    for _ in range(draws):
        tok = mu + torch.randn(K, n, R, generator=gen, dtype=torch.float64)
        dk = tok.sum(1)
        vals.append(gm.loo_sign_agreement(dk, dk.sum(0)).mean())
    v = torch.stack(vals)
    N = K * n
    want = gm.gaussian_sign_agreement(
        mu**2, torch.ones(R, dtype=torch.float64), n, N
    ).mean()
    se = v.std() / math.sqrt(draws)
    assert (
        abs(v.mean() - want) < 3 * se
    ), f"LOO {v.mean():.4f} vs Gaussian {want:.4f} (SE {se:.4f})"
    # the plain form (sign(d_b) vs sign(total)) reads high by the self-overlap
    plain = []
    for _ in range(draws // 4):
        tok = mu + torch.randn(K, n, R, generator=gen, dtype=torch.float64)
        dk = tok.sum(1)
        plain.append((torch.sign(dk) == torch.sign(dk.sum(0))).double().mean())
    assert torch.stack(plain).mean() > want + 0.05


def test_exact_polar_equals_svd_and_zeroes_null_space():
    gen = torch.Generator().manual_seed(3)
    for m, n in ((7, 7), (5, 9), (9, 4)):
        X = torch.randn(m, n, generator=gen)
        U, S, Vh = torch.linalg.svd(X.double(), full_matrices=False)
        O, rank = gm.exact_polar(X)
        assert torch.allclose(O.double(), U @ Vh, atol=1e-6) and int(rank) == min(m, n)
    # rank 2 of 5: the polar is the partial isometry U_2 V_2^T
    A = torch.randn(6, 2, generator=gen, dtype=torch.float64) @ torch.randn(
        2, 5, generator=gen, dtype=torch.float64
    )
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    O, rank = gm.exact_polar(A.float())
    assert int(rank) == 2
    assert torch.allclose(O.double(), U[:, :2] @ Vh[:2], atol=1e-5)


def test_agreement_known_answers_and_isotropic_topk():
    """T15: duplicated groups agree 1; independent noise 0 within 3 SE; the noise share in
    a fixed W's top-k directions is k / min(m, n) under isotropic noise. (Printed: the
    batch's own G as the basis; for isotropic Gaussian noise its bias cancels in A - F.)"""
    gen = torch.Generator().manual_seed(4)
    m, n = 12, 20
    X = torch.randn(m, n, generator=gen)
    O, _ = gm.exact_polar(torch.stack([X, X]))
    assert math.isclose(float(gm.lmo_agreement(O, 2)), 1.0, rel_tol=1e-6)
    vals = []
    for _ in range(100):
        O, _ = gm.exact_polar(torch.randn(2, m, n, generator=gen))
        vals.append(float(gm.lmo_agreement(O, 2)))
    v = torch.tensor(vals, dtype=torch.float64)
    assert abs(v.mean()) < 3 * v.std() / 10
    W = torch.randn(m, n, generator=gen)
    Uk_all, _, _ = gm.top_left(W, m)
    K, draws = 4, 300
    for k in (1, 4):
        own, fixed = [], []
        for _ in range(draws):
            g = torch.randn(K, m, n, generator=gen, dtype=torch.float64)
            nb = torch.full((K,), 10.0, dtype=torch.float64)
            tot = gm.estimate(
                g.square().sum((1, 2))[None], g.sum(0).square().sum()[None], nb, 40.0
            )
            for basis, sink in (
                (Uk_all[:, :k].double(), fixed),
                (torch.linalg.svd(g.sum(0))[0][:, :k], own),
            ):
                p = torch.einsum("sk,bsn->bkn", basis, g)
                est = gm.estimate(
                    p.square().sum((1, 2))[None],
                    p.sum(0).square().sum()[None],
                    nb,
                    40.0,
                )
                sink.append(float(est["tr"][0] / tot["tr"][0]))
        f = torch.tensor(fixed)
        assert abs(f.mean() - k / m) < 3 * f.std() / math.sqrt(
            draws
        ), f"k={k}: {f.mean()} vs {k / m}"
        o = torch.tensor(own)
        print(
            f"k={k}: W basis {f.mean():.4f}, G's own basis {o.mean():.4f} +- {o.std() / math.sqrt(draws):.4f}, k/m {k / m:.4f}"
        )


def test_ema_bias_correction_and_se():
    state = {}
    vals = [1.0, 3.0, 2.0, 5.0]
    out = gm.ema_update(state, {"x": torch.tensor([vals[0]])}, dt=10, tau=200)
    assert math.isclose(float(out["x"]), vals[0], rel_tol=1e-12)
    for v in vals[1:]:
        out = gm.ema_update(state, {"x": torch.tensor([v])}, dt=10, tau=200)
    beta = math.exp(-10 / 200)
    w = [beta ** (len(vals) - 1 - i) * (1 - beta) for i in range(len(vals))]
    mean = sum(wi * v for wi, v in zip(w, vals)) / sum(w)
    var = sum(wi * v * v for wi, v in zip(w, vals)) / sum(w) - mean**2
    se = math.sqrt(var * sum(wi * wi for wi in w)) / sum(w)
    assert math.isclose(float(out["x"]), mean, rel_tol=1e-12)
    assert math.isclose(float(out["x/se"]), se, rel_tol=1e-9)
    # a NaN (not estimable) leaves the average alone
    before = float(out["x"])
    out = gm.ema_update(state, {"x": torch.tensor([math.nan])}, dt=10, tau=200)
    assert math.isclose(float(out["x"]), before, rel_tol=1e-12)
