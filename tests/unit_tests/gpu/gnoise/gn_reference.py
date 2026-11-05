# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The gnoise estimators written from their equations, on explicit tensors.

Nothing here imports torchtitan.optimizers.gnoise*: a reference copied from the
code under test agrees with its bugs.

Small batch b: n_b tokens, gradient g_b of sum(loss) / N_den. With N = sum n_b
over the small batches that hold tokens, ghat_b = (N_den / n_b) g_b, Ghat =
(N_den / N) sum_b g_b, and for a projection P:
    A = mean_b ||P ghat_b||^2, F = ||P Ghat||^2, H = mean_b 1 / n_b,
    tr = (A - F) / (H - 1/N), S = (F H - A/N) / (H - 1/N).
"""

import math

import torch


def estimate(
    Pg: list[torch.Tensor], n: list[float], n_den: float
) -> tuple[float, float]:
    """Pg[b] = P g_b (any shape); n[b] tokens. Returns (tr, S) or (nan, nan)."""
    keep = [b for b in range(len(n)) if n[b] > 0]
    if len(keep) < 2:
        return math.nan, math.nan
    N = sum(n[b] for b in keep)
    A = sum(
        float(((n_den / n[b]) * Pg[b].double()).square().sum()) for b in keep
    ) / len(keep)
    G = sum(Pg[b].double() for b in keep)
    F = float(((n_den / N) * G).square().sum())
    H = sum(1.0 / n[b] for b in keep) / len(keep)
    den = H - 1.0 / N
    return (A - F) / den, (F * H - A / N) / den


def jackknife(
    Pg: list[torch.Tensor], n: list[float], n_den: float
) -> tuple[float, float]:
    """Delete-one jackknife SE of (tr, S): the estimator recomputed without each b."""
    keep = [b for b in range(len(n)) if n[b] > 0]
    k = len(keep)
    trs, ss = [], []
    for out in keep:
        rest = [b for b in keep if b != out]
        tr, s = estimate([Pg[b] for b in rest], [n[b] for b in rest], n_den)
        trs.append(tr)
        ss.append(s)

    def se(v):
        m = sum(v) / k
        return math.sqrt((k - 1) / k * sum((x - m) ** 2 for x in v))

    return se(trs), se(ss)


def unit_projection(W: torch.Tensor):
    """P g = <g, W> W / ||W||^2 (the Frobenius radial direction of W)."""
    w = W.double() / W.double().norm()
    return lambda g: (g.double() * w).sum() * w


def row_radial(W: torch.Tensor):
    """P g = the per-row projections onto the rows of W (rows with zero norm: 0)."""
    w = W.double()
    nrm = w.norm(dim=1, keepdim=True)
    u = torch.where(nrm > 0, w / nrm.clamp_min(1e-300), 0.0)
    return lambda g: (g.double() * u).sum(1, keepdim=True) * u


def tangent(W: torch.Tensor):
    """P g = g minus its Frobenius radial part (explicit, not total - radial)."""
    rad = unit_projection(W)
    return lambda g: g.double() - rad(g)


def loo_sign(d: list[torch.Tensor], rel_tol: float = 1e-5) -> float:
    """Mean over b of the fraction of coordinates with sign(d_b) = sign(sum_{b' != b} d_b'),
    over coordinates with d_b != 0 and |rest| > rel_tol (|d_b| + |total|)."""
    total = sum(x.double() for x in d)
    vals = []
    for x in d:
        rest = total - x.double()
        used = (x != 0) & (rest.abs() > rel_tol * (x.double().abs() + total.abs()))
        if used.any():
            vals.append(
                float(
                    (torch.sign(x.double()) == torch.sign(rest))[used].double().mean()
                )
            )
    return sum(vals) / len(vals)
