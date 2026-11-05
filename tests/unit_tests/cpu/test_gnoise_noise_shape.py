# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gnoise research's noise shape on synthetic small batches, against the equations: with
ghat_b = (N_den / n_b) g_b, Ghat = (N_den / N) sum_b g_b, d_b = ghat_b - Ghat (valid b),
the per-token noise covariance (1 / (K - 1)) sum_b n_b vec(d_b) vec(d_b)^T has the
top-k share ntop_k; C_row = (1 / (K - 1)) sum_b n_b d_b d_b^T (rows x rows),
B_S1 = (tr C_row^(1/2))^2 / ||Ghat||_S1^2. Both sides of the eigenvalue route (rows <=
K x cols and rows > K x cols) are taken."""

import math
import types

import pytest
import torch

from torchtitan.optimizers.gnoise_research import Research


def _ref(g, n, n_den):
    k = [b for b in range(len(n)) if n[b] > 0]
    K, N = len(k), sum(n[b] for b in k)
    Gh = (n_den / N) * sum(g[b] for b in k)
    d = {b: (n_den / n[b]) * g[b] - Gh for b in k}
    cov = sum(n[b] * torch.outer(d[b].reshape(-1), d[b].reshape(-1)) for b in k) / (
        K - 1
    )
    lam = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
    top = {q: float(lam[:q].sum() / lam.sum()) for q in (1, 2, 3)}
    C = sum(n[b] * d[b] @ d[b].T for b in k) / (K - 1)
    w, V = torch.linalg.eigh(C)
    root = (V * w.clamp_min(0).sqrt()) @ V.T
    noise = float(torch.trace(root)) ** 2
    grad = float(torch.linalg.matrix_norm(Gh, ord="nuc")) ** 2
    return top, noise, grad


def _call(g, n, n_den, *, K, shape):
    fake = types.SimpleNamespace(ntop=[q for q in (1, 2, 3) if q <= K - 1], bs1=K >= 8)
    it = types.SimpleNamespace(shape=shape)
    flat = torch.stack([x.reshape(-1) for x in g])
    nt = torch.tensor(n, dtype=torch.float64)
    return Research._noise_shape(
        fake, it, flat, nt, torch.tensor(float(n_den), dtype=torch.float64), nt > 0
    )


@pytest.mark.parametrize("shape", [(40, 3), (6, 10)])  # rows > K cols, rows <= K cols
@pytest.mark.parametrize("empty", [False, True])
def test_against_the_equations(shape, empty):
    gen = torch.Generator().manual_seed(sum(shape))
    K = 9 if empty else 8
    n = [float(20 + 7 * (b % 3)) for b in range(K)]
    if empty:
        n[4] = 0.0  # a small batch without tokens is left out
    mean = torch.randn(*shape, generator=gen, dtype=torch.float64)
    g = [
        (0.3 * mean + torch.randn(*shape, generator=gen, dtype=torch.float64))
        * (n[b] / 100)
        for b in range(K)
    ]
    g = [x if n[b] > 0 else torch.zeros_like(x) for b, x in enumerate(g)]
    n_den = 1.1 * sum(n)
    top, noise, grad = _ref(g, n, n_den)
    got = [float(v) for v in _call(g, n, n_den, K=K, shape=shape)]
    assert got[:3] == pytest.approx([top[1], top[2], top[3]], rel=1e-10)
    # rows > K cols: C_row is rank deficient, and each zero eigenvalue's round-off
    # (~1e-16 lambda_max) adds ~1e-8 sqrt(lambda_max) to the trace on either route
    assert got[3:] == pytest.approx([noise / grad, noise, grad], rel=1e-6)


def test_fewer_than_eight_valid_small_batches_give_nan():
    gen = torch.Generator().manual_seed(1)
    n = [10.0] * 8
    n[0] = 0.0
    g = [torch.randn(5, 4, generator=gen, dtype=torch.float64) for _ in n]
    got = [float(v) for v in _call(g, n, 80.0, K=8, shape=(5, 4))]
    assert all(math.isfinite(v) for v in got[:3]) and all(
        math.isnan(v) for v in got[3:]
    )


def test_k4_has_no_bs1_and_top3_at_most():
    gen = torch.Generator().manual_seed(2)
    n = [10.0, 12.0, 14.0, 16.0]
    g = [torch.randn(5, 4, generator=gen, dtype=torch.float64) for _ in n]
    got = _call(g, n, 60.0, K=4, shape=(5, 4))
    assert len(got) == 3 and float(got[2]) == pytest.approx(
        1.0, rel=1e-12
    )  # rank K - 1 = 3
