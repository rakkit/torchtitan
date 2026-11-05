# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The gradient-noise estimators as plain tensor functions (no distribution).

Small batch b holds n_b tokens; g_b is its gradient of sum(loss) / N_den, so
ghat_b = (N_den / n_b) g_b is its mean gradient and G = sum_b g_b. For a
projection P fixed with respect to the batch, with N = sum_b n_b:

    A = mean_b ||P ghat_b||^2,  F = (N_den / N)^2 ||P G||^2,  H = mean_b 1 / n_b
    tr = (A - F) / (H - 1/N),   S = (F H - A / N) / (H - 1/N)

Small batches with n_b = 0 are left out (experts). See optimizers/readme.md.
"""

import math

import torch


# Singular values below this fraction of sigma_max are mapped to 0 by the exact polar.
POLAR_REL_TOL = 1e-6


def estimate(
    sq: torch.Tensor,
    F: torch.Tensor,
    n: torch.Tensor,
    n_den: float | torch.Tensor,
    cross: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """tr and S per component, and their delete-one jackknife SE when `cross` is given.

    sq[..., K]: ||P g_b||^2; F[...]: ||P G||^2; n[..., K] or n[K]: tokens (0 = absent);
    cross[..., K]: <P G, P g_b>. All in g's raw scale. Non-estimable entries are NaN.
    """
    sq = sq.double()
    F = F.double()
    n = torch.broadcast_to(n.double(), sq.shape)
    n_den = torch.as_tensor(n_den, dtype=torch.float64, device=sq.device)
    valid = n > 0
    k = valid.sum(-1).double()
    N = n.sum(-1)
    inv = torch.where(valid, 1.0 / n.clamp_min(1.0), 0.0)
    w = torch.where(valid, (n_den * inv) ** 2, 0.0)
    A = (w * sq).sum(-1) / k.clamp_min(1.0)
    H = inv.sum(-1) / k.clamp_min(1.0)
    Fh = (n_den / N.clamp_min(1.0)) ** 2 * F
    den = H - 1.0 / N.clamp_min(1.0)
    ok = (k >= 2) & (den > 0)
    nan = torch.full_like(A, math.nan)
    safe = torch.where(ok, den, 1.0)
    out = {
        "tr": torch.where(ok, (A - Fh) / safe, nan),
        "sig": torch.where(ok, (Fh * H - A / N.clamp_min(1.0)) / safe, nan),
        "A": A,
        "F": Fh,
        "H": H,
        "N": N,
        "K": k,
    }
    if cross is None:
        return out
    cross = cross.double()
    # Delete-one: the sample without b (rows where b is absent leave it unchanged).
    km = (k[..., None] - 1.0).clamp_min(1.0)
    A_m = (A[..., None] * k[..., None] - w * sq) / km
    H_m = (H[..., None] * k[..., None] - inv) / km
    N_m = (N[..., None] - n).clamp_min(1.0)
    F_m = (n_den / N_m) ** 2 * (F[..., None] - 2.0 * cross + sq)
    den_m = H_m - 1.0 / N_m
    ok_m = valid & (k[..., None] >= 3) & (den_m > 0)
    safe_m = torch.where(ok_m, den_m, 1.0)
    tr_m = torch.where(ok_m, (A_m - F_m) / safe_m, 0.0)
    sig_m = torch.where(ok_m, (F_m * H_m - A_m / N_m) / safe_m, 0.0)
    for name, t in (("tr", tr_m), ("sig", sig_m)):
        mean = t.sum(-1) / k.clamp_min(1.0)
        dev = torch.where(ok_m, t - mean[..., None], 0.0)
        se = ((k - 1.0) / k.clamp_min(1.0) * dev.square().sum(-1)).sqrt()
        out[f"se_{name}"] = torch.where(ok & (k >= 3), se, nan)
    return out


def from_sums(
    a_sum: torch.Tensor,
    F: torch.Tensor,
    k: int,
    h: torch.Tensor,
    N: torch.Tensor,
    n_den,
) -> tuple[torch.Tensor, torch.Tensor]:
    """tr and S when only A's weighted sum over the k small batches is kept:
    a_sum = sum_b (n_den / n_b)^2 ||P g_b||^2, h = mean_b 1 / n_b, N = sum_b n_b."""
    A = a_sum.double() / torch.as_tensor(k, dtype=torch.float64).clamp_min(1.0)
    Fh = (n_den / N) ** 2 * F.double()
    den = h - 1.0 / N
    ok = (den > 0) & (torch.as_tensor(k) >= 2)
    safe = torch.where(ok, den, 1.0)
    return (
        torch.where(ok, (A - Fh) / safe, math.nan),
        torch.where(ok, (Fh * h - A / N) / safe, math.nan),
    )


def ema_update(
    state: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    dt: float,
    tau: float,
) -> dict[str, torch.Tensor]:
    """Bias-corrected EMAs of `values` after `dt` training steps with time constant `tau`.

    state holds the raw averages, `c` = 1 - prod(beta) and `c2` = the sum of squared
    weights, so the EMA's standard error is sqrt(var * c2) / c. NaN inputs (not
    estimable) leave that entry unchanged. Returns {key: value, key/se: SE}.
    """
    beta = math.exp(-dt / tau)
    out = {}
    for key, x in values.items():
        x = x.double()
        good = torch.isfinite(x)
        xv = torch.where(good, x, 0.0)
        if key not in state:
            for s in ("", "/sq", "/c", "/c2"):
                state[key + s] = torch.zeros_like(x)
        m, sq, c, c2 = (state[key + s] for s in ("", "/sq", "/c", "/c2"))
        m = torch.where(good, beta * m + (1.0 - beta) * xv, m)
        sq = torch.where(good, beta * sq + (1.0 - beta) * xv * xv, sq)
        c2 = torch.where(good, beta * beta * c2 + (1.0 - beta) ** 2, c2)
        c = torch.where(good, beta * c + (1.0 - beta), c)
        state[key], state[key + "/sq"], state[key + "/c"], state[key + "/c2"] = (
            m,
            sq,
            c,
            c2,
        )
        cc = c.clamp_min(1e-300)
        mean = torch.where(c > 0, m / cc, math.nan)
        var = (sq / cc - (m / cc) ** 2).clamp_min(0.0)
        out[key] = mean
        out[key + "/se"] = torch.where(c > 0, (var * c2).sqrt() / cc, math.nan)
    return out


LOO_REL_TOL = (
    1e-5  # the leave-one-out rest counts only above this fraction of |d_b| + |total|
)


def loo_sign_terms(
    d: torch.Tensor, total: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per coordinate of d[..., R] against total[R]: (agree, used) of `loo_sign_agreement`."""
    rest = total - d
    used = (d != 0) & (rest.abs() > LOO_REL_TOL * (d.abs() + total.abs()))
    return (torch.sign(d) == torch.sign(rest)) & used, used


def loo_sign_agreement(d: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    """Per small batch, the fraction of coordinates where sign(d_b) = sign(total - d_b).

    d[K, R] holds each small batch's coordinate, total[R] their (separately reduced) sum.
    A coordinate is left out when d_b = 0 or the rest is unresolved, |total - d_b| <=
    LOO_REL_TOL (|d_b| + |total|): a row only b touched leaves fp residue there.
    Returns [K] (NaN where none is left).
    """
    agree, used = loo_sign_terms(d, total[None, :])
    cnt = used.sum(-1).double()
    return torch.where(cnt > 0, agree.sum(-1).double() / cnt.clamp_min(1.0), math.nan)


def segment_nanquantile(
    x: torch.Tensor, seg: torch.Tensor, n_seg: int, q: torch.Tensor
) -> torch.Tensor:
    """torch.nanquantile(x[seg == s], q) (linear interpolation) for every segment s at once.

    x[n], seg[n] in [0, n_seg), every segment non-empty. Returns [len(q), n_seg]. As torch:
    sort ascending with NaN last, rank = q (count of non-NaN - 1), lerp between the
    neighbours; an all-NaN segment gives NaN."""
    order = torch.sort(x, stable=True).indices
    order = order[
        torch.sort(seg[order], stable=True).indices
    ]  # by segment, ascending within
    xs = x[order]
    cnt = torch.bincount(seg, minlength=n_seg)
    start = torch.cumsum(cnt, 0) - cnt
    # the non-NaN count per segment; an unweighted bincount is deterministic on CUDA, a weighted one is not
    valid = torch.bincount(seg[~torch.isnan(x)], minlength=n_seg).to(x.dtype)
    ranks = q.to(x.dtype)[:, None] * (valid - 1.0)[None, :]
    below = ranks.long()
    lo = xs[start[None, :] + below]
    hi = xs[start[None, :] + ranks.ceil().long()]
    return torch.lerp(lo, hi, ranks - below)


def gaussian_sign_agreement(sig: torch.Tensor, tr: torch.Tensor, n, N) -> torch.Tensor:
    """p_n p_{N-n} + (1 - p_n)(1 - p_{N-n}), p_B = Phi(sqrt(B S / tr)); Phi(0) when S <= 0."""
    ratio = torch.where((sig > 0) & (tr > 0), sig / tr.clamp_min(1e-300), 0.0).double()

    def p(B: float) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf((B * ratio).sqrt() / math.sqrt(2.0)))

    pn, pr = p(n), p(N - n)
    return pn * pr + (1.0 - pn) * (1.0 - pr)


def exact_polar(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The polar factor of X[..., m, n] from the eigendecomposition of its fp64 Gram.

    Singular values below POLAR_REL_TOL * sigma_max map to 0 (a partial isometry).
    Returns (O in fp32, rank per matrix).
    """
    per = max(1, (1 << 27) // (X.shape[-1] * X.shape[-2]))
    if (
        X.ndim > 2 and X.shape[0] > per
    ):  # a few matrices at a time: fp64 temporaries ~1 GB
        parts = [exact_polar(X[i : i + per]) for i in range(0, X.shape[0], per)]
        return torch.cat([o for o, _ in parts]), torch.cat([r for _, r in parts])
    t = X.shape[-2] < X.shape[-1]
    Y = (X.mT if t else X).double()  # [.., m >= n]
    lam, V = torch.linalg.eigh(Y.mT @ Y)
    lmax = lam[..., -1:].clamp_min(0.0)
    keep = lam > (POLAR_REL_TOL**2) * lmax
    inv = torch.where(keep, lam.clamp_min(1e-300).rsqrt(), 0.0)
    O = (Y @ (V * inv[..., None, :]) @ V.mT).float()
    return (O.mT if t else O), keep.sum(-1)


def top_left(
    W: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """W's top-k singular vectors on its smaller side and the top pair, exactly.

    Returns (U_k [s, k] on the smaller side s = min(m, n), u1 [m], v1 [n]).
    """
    m, n = W.shape
    Y = W.double()
    small_left = m <= n
    gram = Y @ Y.mT if small_left else Y.mT @ Y
    lam, V = torch.linalg.eigh(gram)
    top = V[:, -1]
    s1 = lam[-1].clamp_min(0.0).sqrt()
    if small_left:
        u1 = top
        v1 = (Y.mT @ u1) / s1.clamp_min(1e-300)
    else:
        v1 = top
        u1 = (Y @ v1) / s1.clamp_min(1e-300)
    Uk = V[:, -k:].flip(-1) if k > 0 else V[:, :0]
    return Uk.float(), u1.float(), v1.float()


def lmo_agreement(O: torch.Tensor, groups_per_partition: int) -> torch.Tensor:
    """Mean <O_i, O_j> / min(m, n) over the disjoint pairs inside each partition.

    O[P * g, m, n] stacks the LMOs of `g` disjoint group sums for each of P partitions.
    """
    g = groups_per_partition
    m, n = O.shape[-2:]
    flat = O.reshape(-1, g, m * n)
    iu = torch.triu_indices(g, g, 1)
    total = torch.zeros((), dtype=torch.float64, device=O.device)
    for p in range(flat.shape[0]):
        for i, j in zip(iu[0].tolist(), iu[1].tolist()):
            total += (flat[p, i] * flat[p, j]).sum(dtype=torch.float64)
    return total / (flat.shape[0] * iu.shape[1]) / min(m, n)


def row_agreement(
    Gs: torch.Tensor, groups_per_partition: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The row rule's LMO agreement: mean over rows of cos(rows of two group sums),
    zero-gradient rows counting as 0, averaged over disjoint pairs; plus the fraction of
    (pair, row) entries where both rows are nonzero. Gs[P * g, R, C]."""
    g = groups_per_partition
    R, C = Gs.shape[-2:]
    x = Gs.reshape(-1, g, R, C).float()
    nrm = x.double().square().sum(-1).sqrt()  # [P, g, R]
    cos_sum = torch.zeros((), dtype=torch.float64, device=Gs.device)
    both_sum = torch.zeros((), dtype=torch.float64, device=Gs.device)
    n = 0
    for p in range(x.shape[0]):
        for i in range(g):
            for j in range(i + 1, g):
                d = (x[p, i] * x[p, j]).double().sum(-1)
                ok = (nrm[p, i] > 0) & (nrm[p, j] > 0)
                cos_sum += torch.where(
                    ok, d / (nrm[p, i] * nrm[p, j]).clamp_min(1e-300), 0.0
                ).sum()
                both_sum += ok.double().sum()
                n += R
    return cos_sum / n, both_sum / n
