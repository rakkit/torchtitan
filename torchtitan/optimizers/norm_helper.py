# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os

import torch
from torch.distributed.tensor import DTensor

# cuSOLVER drivers, chosen per use rather than globally -- measured on GH200 /
# torch 2.12 at the exact matrix shapes of the 7B/8B and 30B MoE flavors
# (all of them tall-or-wide against dim=2048; none square):
#
#   matrix                     gesvd   gesvdj   gesvda
#   attn wq   [4096,2048]     157 ms   181 ms    26 ms
#   ffn w1    [6144,2048]     191 ms   187 ms    26 ms
#   moe expert [768,2048]      37 ms    35 ms     7 ms
#   embed    [201088,2048]    292 ms   393 ms    85 ms
#
# `gesvda` is 4-7x faster than the historical `gesvd` everywhere, and `gesvdj`
# is no faster (sometimes slower), which is why the obvious switch to Jacobi
# does nothing.
#
# But `gesvda` is an *approximate* driver and cannot be a blanket default:
#   - sigma_max is exact (0 to ~1e-7 relative on every shape tested),
#   - sigma_min is NOT: ~1e-2 relative on ill-conditioned input and 100-300%
#     on rank-deficient input, which lands directly on `condition_number`
#     (= S[0]/S[-1], measured 65-77% off for rank-deficient matrices),
#   - and it can fail outright -- it raised
#     `_LinAlgError: the algorithm failed to converge` on a rank-deficient
#     2048x2048 input, which would take down a training run.
#
# So: `gesvda` only where sigma_max alone is consumed, with a fallback to
# `gesvd` if it throws; `gesvd` whenever the whole spectrum is used.
SVDVALS_DRIVER: str = os.environ.get("DISCO_SVDVALS_DRIVER", "gesvd")
SVDVALS_DRIVER_SIGMA_ONLY: str = os.environ.get(
    "DISCO_SVDVALS_DRIVER_SIGMA_ONLY", "gesvda"
)


def _svdvals(W: torch.Tensor, sigma_only: bool = False) -> torch.Tensor:
    """`torch.linalg.svdvals` with the driver appropriate to the consumer.

    `sigma_only=True` promises that only `S[0]` will be read, which unlocks the
    much faster approximate driver (see the table above). It falls back to the
    accurate driver if the approximate one fails to converge, so a pathological
    parameter degrades to "slower" rather than "crashes training".

    `driver=` is only accepted for CUDA inputs -- passing it on a CPU tensor
    raises outright, which is why every 2-D norm in this module used to be
    CPU-unusable. Selecting it by device here makes the module work on CPU
    (needed for unit tests and naive_param_norm's debug path) with no change to
    CUDA behaviour.
    """
    if not W.is_cuda:
        return torch.linalg.svdvals(W)
    if sigma_only:
        try:
            return torch.linalg.svdvals(W, driver=SVDVALS_DRIVER_SIGMA_ONLY)
        except Exception:
            pass
    return torch.linalg.svdvals(W, driver=SVDVALS_DRIVER)


# Backend for the FULL singular-value spectrum. "gram" computes it as
# sqrt(eigvalsh(W W^T)) in float64, orienting so the Gram matrix is over the
# smaller dimension; "svd" is the historical `torch.linalg.svdvals` in float32.
#
# "gram" is both faster and MORE accurate than the fp32 SVD it replaces.
# Measured on GH200 / torch 2.12 against a float64 SVD as ground truth,
# [768,2048], relative error on sigma_min / condition_number:
#
#   conditioning     fp32 svdvals        fp64 gram
#   well-cond.       4.9e-08              1.9e-13
#   kappa 1e3        2.8e-07              2.7e-11
#   kappa 1e6        3.0e-04              2.2e-05
#   kappa 1e8        3.9e-02              7.0e-03
#   rank-deficient   9.4e-01              3.6e-02
#
# and on cost: [768,2048] 37.5 -> 7.1 ms, [6144,2048] 182 -> 23 ms,
# embedding [201088,2048] 296 -> 54 ms. Batched it is far better still: the
# 282 [768,2048] matrices one rank owns for qwen30b-a3b at EP=64 go from
# 11.15 s (svdvals cannot batch -- it serialises internally) to 0.31 s.
#
# float64 is essential and nearly free. Note it cannot be traded for a
# normalization trick: rescaling W (by its Frobenius norm, by its max-abs
# entry) or rescaling the Gram before `eigvalsh` was measured and changes
# nothing -- sigma_min error stays ~1e-4 at kappa 1e2, ~1e-2 at kappa 1e3 and
# total at kappa 1e6 for every variant. kappa(W W^T) = kappa(W)^2 is
# scale-invariant, so normalization only guards against overflow, which is not
# the failure mode here. (gram_helper's `_safe_sym_eigvalsh` rescales for a
# different reason: helping LAPACK resolve close eigenvalues at large
# magnitude.) Essential: forming the Gram squares the
# condition number, so a float32 Gram is catastrophic -- the same table with a
# float32 GEMM gives 1.0e+00 sigma_min error (1e14 on condition_number) from
# kappa 1e6 upward. Nearly free, for two reasons:
#
#   * The Gram GEMM in fp64 is actually *faster* than in true fp32 on this
#     hardware -- consistently 0.79-0.82x, CUDA-event timed. On Hopper/GH200,
#     FP64 has dedicated tensor cores while non-TF32 FP32 does not and falls
#     back to CUDA cores: measured on a 68.7 GFLOP Gram, true fp32 49.7 TFLOPS
#     vs fp64 61.8 TFLOPS (TF32 434.8, bf16 841.1, both unusable here -- TF32
#     has ~10 mantissa bits against a product that already squares the
#     condition number). Using fp64 also makes this immune to a globally
#     enabled TF32 flag silently destroying an fp32 Gram.
#   * The GEMM is not the cost anyway. `eigvalsh` on the [k,k] Gram dominates
#     (7.0 vs 0.06 ms at k=768; 22.9 vs 0.7 ms at k=2048) and is only 1.03-1.07x
#     slower in fp64.
#
# This is a genuine numerics change to every spectrum-derived metric
# (condition_number, effective_rank*, stable_rank, rms_to_rms, spectrum). The
# values get *more* accurate, but they do move -- set DISCO_SPECTRUM_BACKEND=svd
# to keep the old ones for continuity with an in-flight run.
SPECTRUM_BACKEND: str = os.environ.get("DISCO_SPECTRUM_BACKEND", "gram")

# Budget for the float64 copy of W while accumulating the Gram. Bounds the
# temporary both for a very long reduction dimension (the 201088-row embedding
# would otherwise need a 3.3 GiB fp64 copy) and for a large batch.
_GRAM_FP64_BUDGET_BYTES: int = 1 << 30  # 1 GiB


def gram_batch_capacity(m: int, n: int, budget_bytes: int = 2 << 30) -> int:
    """How many `[m, n]` matrices to put through `_gram_spectrum` at once.

    The float64 copy of W is already bounded inside `_gram_spectrum`, so what
    limits the batch is the `[k, k]` Gram itself plus eigvalsh workspace. Sizing
    by a byte budget rather than a fixed count matters: the previous fixed cap
    of 128 came from `gesvda`'s batch limit and does not apply to this path, and
    it was measurably throttling it -- 5376 [384,1024] matrices took 1236 ms at
    chunk 128 versus 849 ms unchunked (1.45x).
    """
    k = min(int(m), int(n))
    per_matrix = 3 * k * k * 8  # Gram + eigenvalue/workspace headroom
    return max(1, min(4096, budget_bytes // max(per_matrix, 1)))


def fp64_gram(W: torch.Tensor) -> torch.Tensor:
    """The symmetrised float64 Gram `W W^T`, oriented over the smaller dim.

    `W` is `[..., m, n]`; returns `[..., k, k]` with `k = min(m, n)`, always
    float64.

    float64 is the point, not an incidental detail: `kappa(W W^T) ==
    kappa(W)^2`, and that squaring is scale-invariant, so normalising `W`
    first does not rescue it -- in float32 the product genuinely loses the
    small end of the spectrum and can trip `eigh`'s convergence. In float64
    the squared condition number is affordable, and the result is BOTH faster
    than an SVD of `W` and more accurate than one done in float32 (measured:
    5.9x faster for eigenvalues at `[768,2048]`, with the smallest eigenvalue
    34-500x closer to a float64 SVD reference). Hopper helps here -- its
    float64 tensor cores (61.8 TFLOPS) beat non-TF32 float32 (49.7), which has
    no tensor-core path at all.

    The reduction is chunked so the float64 copy of `W` stays under
    `_GRAM_FP64_BUDGET_BYTES` regardless of batch size. float32 -> float64 is
    exact, so chunking changes nothing numerically; it only bounds the
    temporary, which matters for shapes like the `[201088, 2048]` lm_head
    where an unchunked float64 copy would be several GB.

    Shared with `gram_helper`, which needs the same matrix for its eigen
    paths -- keep it in one place so the numerics and the memory bound do not
    diverge between the two.
    """
    m, n = W.shape[-2], W.shape[-1]
    # Reduce over the longer axis so the Gram is [k, k] with k = min(m, n).
    Wo = W if m <= n else W.transpose(-2, -1)
    L = Wo.shape[-1]
    k = Wo.shape[-2]
    batch = 1
    for d in Wo.shape[:-2]:
        batch *= d
    per_col = max(batch * k * 8, 1)
    chunk = max(1, min(L, _GRAM_FP64_BUDGET_BYTES // per_col))
    if chunk < L:
        G = torch.zeros(*Wo.shape[:-2], k, k, device=W.device, dtype=torch.float64)
        for i in range(0, L, chunk):
            x = Wo[..., i : i + chunk].to(torch.float64)
            G += x @ x.transpose(-2, -1)
            del x
    else:
        Wd = Wo.to(torch.float64)
        G = Wd @ Wd.transpose(-2, -1)
        del Wd
    # Symmetric in exact arithmetic; forcing it removes any asymmetry the GEMM
    # introduced before eigvalsh/eigh sees it.
    return 0.5 * (G + G.transpose(-2, -1))


def _gram_spectrum(W: torch.Tensor) -> torch.Tensor:
    """Full descending singular-value spectrum via the float64 Gram matrix.

    `W` is `[..., m, n]`; returns `[..., min(m, n)]` in `W`'s float dtype.
    Orients so the Gram is over the smaller dimension, and for a 2-D input with
    a very long reduction dimension accumulates the Gram in chunks so the
    float64 temporary stays bounded.
    """
    out_dtype = torch.promote_types(W.dtype, torch.float32)
    G = fp64_gram(W)
    ev = torch.linalg.eigvalsh(G)
    # Ascending eigenvalues of W W^T are sigma^2; tiny negatives are round-off.
    return ev.clamp_min(0).sqrt().flip(-1).to(out_dtype)


def gram_top_singular_pair(
    W: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact leading singular triple `(sigma, u1, v1)` from the float64 Gram.

    `W` is `[..., m, n]`; returns `[...]`, `[..., m]`, `[..., n]`.

    This replaces what would otherwise be a power iteration, and it is both
    exact and cheaper. `_gram_spectrum` already forms `W W^T` in float64 and
    calls `eigvalsh`; asking for `eigh` instead returns the eigenvectors of the
    same matrix for 5-19% more, and the leading one *is* `u1`, with
    `v1 = W^T u1 / sigma` following in one matvec. No iteration, no warm-start
    state to carry between logging steps, no convergence to worry about.

    Measured against a float64 SVD: sigma error 5e-14 to 6e-13, `u1` residual
    ~1e-15, `v1` residual ~5e-17 -- i.e. exact to working precision, where a
    warm-started power iteration was landing at 1e-2 to 1e-3 on a cold start.

    Cost, batched (the shape that matters, since the expert path decomposes
    hundreds at once): eigh vs eigvalsh is 1.19x on 1344 `[384,1024]`
    (+47 ms total) and 1.11x on 282 `[768,2048]` (+32 ms).

    Orientation note: the Gram is taken over the smaller dimension, so when
    `m > n` the roles of `u1` and `v1` swap internally; they are swapped back
    before returning, so the contract is always `W v1 = sigma u1`.
    """
    out_dtype = torch.promote_types(W.dtype, torch.float32)
    m, n = W.shape[-2], W.shape[-1]
    flipped = m > n
    Wo = W.transpose(-2, -1) if flipped else W

    # `fp64_gram`, not an inline float64 copy: it chunks the reduction to keep
    # the float64 temporary under `_GRAM_FP64_BUDGET_BYTES`. That budget exists
    # for exactly the shape this function is now called with on the embedding
    # path -- an unchunked float64 copy of `[201088, 2048]` is ~3.3 GB, and
    # several are live at once.
    G = fp64_gram(Wo)
    ev, evec = torch.linalg.eigh(G)
    sigma = ev[..., -1].clamp_min(0).sqrt()
    tiny = torch.finfo(torch.float64).tiny
    a = evec[..., -1]  # leading left vector
    # v1 = W^T u1 / sigma, computed against the original (non-float64) operand
    # and promoted only for the contraction, so no full float64 copy is needed.
    b = torch.einsum("...mn,...m->...n", Wo.to(torch.float64), a) / sigma.clamp_min(
        tiny
    ).unsqueeze(-1)
    del G, ev, evec

    # `a` spans the row space of the oriented matrix; undo the orientation so
    # the caller always gets (u1 in R^m, v1 in R^n).
    u1, v1 = (b, a) if flipped else (a, b)
    return sigma.to(out_dtype), u1.to(out_dtype), v1.to(out_dtype)


def _full_spectrum(W: torch.Tensor) -> torch.Tensor:
    """Full spectrum through the configured backend, with a safe fallback."""
    if SPECTRUM_BACKEND == "gram":
        try:
            return _gram_spectrum(W)
        except Exception:
            # eigvalsh can fail to converge on pathological input; a slower
            # accurate answer beats taking down a training run.
            pass
    if W.ndim > 2:
        return torch.linalg.svdvals(W)
    return _svdvals(W)


# Norms computable without any singular value. Everything else in
# NORM_FUNCTIONS needs the spectrum: "rms_to_rms" and "stable_rank" need only
# sigma_max, while "condition_number", "effective_rank" and
# "effective_rank_squared" need all of it.
_SVD_FREE_NORMS: frozenset[str] = frozenset(
    {
        "l1_to_rms",
        "rms_to_inf",
        "supremum",
        "frobenius_norm",
        "average_entry_size",
    }
)

# Norms that need a decomposition but read only its largest singular value.
# Requesting nothing outside `_SVD_FREE_NORMS | _SIGMA_ONLY_NORMS` (and no
# spectrum) selects the fast approximate driver -- 4-7x, exact for sigma_max.
# `condition_number`, `effective_rank` and `effective_rank_squared` are the
# ones that force the accurate driver, and `condition_number` is in the
# "default" set, so trimming `norms_to_log` is what unlocks this.
_SIGMA_ONLY_NORMS: frozenset[str] = frozenset({"rms_to_rms", "stable_rank"})


@torch.no_grad()
def rms_to_rms_norm(W):
    """
    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_uniform_(w.T, ...)``.
    """
    assert W.ndim == 2, "operator norm can only be applied to matrices"
    norm = torch.linalg.norm(W.to(torch.float32), ord=2, dtype=torch.float32)
    fan_out, fan_in = W.shape
    scale = math.sqrt(fan_in / fan_out)
    norm *= scale
    return norm


@torch.no_grad()
def l1_to_rms_norm(W):
    assert W.ndim == 2, "operator norm can only be applied to matrices"
    norm = torch.max(
        torch.linalg.norm(W.to(torch.float32), ord=2, dim=0, dtype=torch.float32)
    )
    scale = torch.sqrt(torch.tensor(W.shape[0], dtype=W.dtype, device=W.device))
    norm /= scale
    return norm


@torch.no_grad()
def rms_to_inf_norm(W):
    assert W.ndim == 2, "operator norm can only be applied to matrices"
    norm = torch.max(
        torch.linalg.norm(W.to(torch.float32), ord=2, dim=1, dtype=torch.float32)
    )
    scale = torch.sqrt(torch.tensor(W.shape[1], dtype=W.dtype, device=W.device))
    norm *= scale
    return norm


@torch.no_grad()
def supremum_norm(x):
    return x.abs().max()


@torch.no_grad()
def condition_number(W):
    assert W.ndim == 2, "condition number calculation can only be applied to matrices"
    S = _svdvals(W.to(torch.float32))
    return S[0] / S[-1]


@torch.no_grad()
def frobenius_norm(W):
    return torch.linalg.norm(W.float(), ord="fro")


@torch.no_grad()
def average_entry_size(W):
    # https://docs.modula.systems/examples/weight-erasure/
    return frobenius_norm(W) / math.sqrt(W.numel())


@torch.no_grad()
def stable_rank(W):
    # https://docs.modula.systems/examples/weight-erasure/
    S = _svdvals(W.to(torch.float32))
    spec = S[0]
    if spec == 0:
        return torch.tensor(0.0, device=W.device)
    frob_norm = frobenius_norm(W)
    return (frob_norm**2) / (spec**2)


@torch.no_grad()
def effective_rank(W):
    # https://docs.modula.systems/examples/weight-erasure/
    S = _svdvals(W.to(torch.float32))
    p = (S / (S.sum() + 1e-12)).clamp_min(1e-12)
    return torch.exp(-(p * p.log()).sum())


@torch.no_grad()
def effective_rank_normalized(W):
    # Same as `effective_rank`, but divided by min(fan_in, fan_out) so the
    # result lies in (0, 1] and is comparable across parameters/models of
    # different sizes.
    S = _svdvals(W.to(torch.float32))
    p = (S / (S.sum() + 1e-12)).clamp_min(1e-12)
    erank = torch.exp(-(p * p.log()).sum())
    return erank / min(W.shape[-2], W.shape[-1])


@torch.no_grad()
def effective_rank_squared(W):
    # Same as `effective_rank`, but the probability distribution is over
    # squared singular values (spectral "energy", same p_i as the
    # cumulative spectral energy E(k) = sum_{i<=k} sigma_i^2 / sum_j sigma_j^2
    # in optimizers/spectrum_logging.py) rather than raw singular values:
    #   p_i = sigma_i^2 / sum_j sigma_j^2,  r_eff^2 = exp(-sum_i p_i log p_i)
    S = _svdvals(W.to(torch.float32))
    S_sq = S * S
    p = (S_sq / (S_sq.sum() + 1e-12)).clamp_min(1e-12)
    return torch.exp(-(p * p.log()).sum())


NORM_FUNCTIONS = {
    "rms_to_rms": rms_to_rms_norm,
    "l1_to_rms": l1_to_rms_norm,
    "rms_to_inf": rms_to_inf_norm,
    "supremum": supremum_norm,
    "condition_number": condition_number,
    "frobenius_norm": frobenius_norm,
    "average_entry_size": average_entry_size,
    "stable_rank": stable_rank,
    "effective_rank": effective_rank,
    # "effective_rank_normalized": effective_rank_normalized,
    "effective_rank_squared": effective_rank_squared,
}


# NOT torch.compile'd, deliberately. `torch.linalg.svdvals` cannot be lowered
# by inductor, so wrapping this in torch.compile graph-breaks around it and --
# measured on GH200 / torch 2.12 -- costs about as much again as the
# decomposition itself. Same math, eager vs compiled:
#
#     [768,2048]      40.1 ms  vs   83.1 ms   (2.08x)
#     [4096,2048]    177.8 ms  vs  360.4 ms   (2.03x)
#     [6144,2048]    180.8 ms  vs  381.7 ms   (2.11x)
#     [201088,2048]  302.4 ms  vs  576.3 ms   (1.91x)
#
# The non-SVD arithmetic here is a handful of reductions costing ~0.06 ms, so
# there was never much for the compiler to win, and the graph break around the
# decomposition more than eats it. `fused_metrics_no_svd` keeps its decorator:
# it has no decomposition to break on, and there compile genuinely helps at the
# large shapes (2.10 ms vs 3.20 ms on the embedding).
@torch.no_grad()
def fused_metrics(W, eps=1e-20):
    if W.ndim < 2:
        # Operator norms require a matrix.
        return {"supremum": W.abs().max(), "spectrum": W.abs()}

    Wf = W.float()
    Wf_square = Wf * Wf
    fan_out, fan_in = Wf.shape

    sup = Wf.abs().amax()
    rowsqsum = Wf_square.sum(1)
    colsqsum = Wf_square.sum(0)

    row_l2 = rowsqsum.sqrt()
    col_l2 = colsqsum.sqrt()

    l1_to_rms = col_l2.max() / math.sqrt(fan_out)
    rms_to_inf = row_l2.max() * math.sqrt(fan_in)

    S = _full_spectrum(Wf)

    spec = S[0] * math.sqrt(fan_in / fan_out)

    cond = S[0] / (S[-1] + eps)
    cond = cond.clamp_min(eps)

    frob_norm = row_l2.norm(p=2)

    spec_unscaled = S[0]
    srank = (frob_norm**2) / (spec_unscaled**2 + eps)
    srank = srank.clamp_min(eps)

    p = (S / (S.sum() + eps)).clamp_min(eps)
    erank = torch.exp(-(p * p.log()).sum())
    erank_norm = erank / min(fan_out, fan_in)

    S_sq = S * S
    p_sq = (S_sq / (S_sq.sum() + eps)).clamp_min(eps)
    erank_sq = torch.exp(-(p_sq * p_sq.log()).sum())

    avg_entry = frob_norm / math.sqrt(fan_out * fan_in)

    return {
        "rms_to_rms": spec,
        "l1_to_rms": l1_to_rms,
        "rms_to_inf": rms_to_inf,
        "supremum": sup,
        "condition_number": cond,
        "frobenius_norm": frob_norm,
        "average_entry_size": avg_entry,
        "stable_rank": srank,
        "effective_rank": erank,
        # "effective_rank_normalized": erank_norm,
        "effective_rank_squared": erank_sq,
        "spectrum": S,
        # See diag_metrics' note -- kept out of the norms_to_log projection in
        # calculate_norm, handed out as its own key.
        "sigma_max": S[0],
    }


# here we use allow dynamic is to support the DDP running
@torch.no_grad()
@torch.compile(dynamic=True)
def fused_metrics_no_svd(W, eps=1e-20):
    """The subset of `fused_metrics` that needs no singular values.

    Same expressions, same order of operations as `fused_metrics`, so the
    values are bit-identical to taking these keys out of its result -- this is
    a pure "skip the SVD" variant, not a re-derivation. Selected by
    `calculate_norm` when neither the requested norms nor the caller's spectrum
    request need `S` (see `_SVD_FREE_NORMS`).
    """
    if W.ndim < 2:
        return {"supremum": W.abs().max()}

    Wf = W.float()
    Wf_square = Wf * Wf
    fan_out, fan_in = Wf.shape

    sup = Wf.abs().amax()
    rowsqsum = Wf_square.sum(1)
    colsqsum = Wf_square.sum(0)

    row_l2 = rowsqsum.sqrt()
    col_l2 = colsqsum.sqrt()

    l1_to_rms = col_l2.max() / math.sqrt(fan_out)
    rms_to_inf = row_l2.max() * math.sqrt(fan_in)

    frob_norm = row_l2.norm(p=2)
    avg_entry = frob_norm / math.sqrt(fan_out * fan_in)

    return {
        "l1_to_rms": l1_to_rms,
        "rms_to_inf": rms_to_inf,
        "supremum": sup,
        "frobenius_norm": frob_norm,
        "average_entry_size": avg_entry,
    }


# Also not compiled -- same graph-break reason as `fused_metrics`. The margin
# is smaller here (the approximate driver dominates) but it is still the wrong
# side: 7.71 ms eager vs 8.07 ms compiled at [768,2048].
@torch.no_grad()
def fused_metrics_sigma_only(W, eps=1e-20):
    """`fused_metrics` restricted to what `S[0]` alone can produce.

    Same expressions as `fused_metrics`, so the shared keys are identical; the
    difference is the decomposition underneath, which uses the fast approximate
    driver (see `_svdvals`). Selected by `calculate_norm` when no requested
    norm reads past the largest singular value.
    """
    if W.ndim < 2:
        return {"supremum": W.abs().max()}

    Wf = W.float()
    Wf_square = Wf * Wf
    fan_out, fan_in = Wf.shape

    sup = Wf.abs().amax()
    rowsqsum = Wf_square.sum(1)
    colsqsum = Wf_square.sum(0)
    row_l2 = rowsqsum.sqrt()
    col_l2 = colsqsum.sqrt()

    l1_to_rms = col_l2.max() / math.sqrt(fan_out)
    rms_to_inf = row_l2.max() * math.sqrt(fan_in)

    s_max = _svdvals(Wf, sigma_only=True)[0]
    spec = s_max * math.sqrt(fan_in / fan_out)

    frob_norm = row_l2.norm(p=2)
    srank = ((frob_norm**2) / (s_max**2 + eps)).clamp_min(eps)
    avg_entry = frob_norm / math.sqrt(fan_out * fan_in)

    return {
        "rms_to_rms": spec,
        "l1_to_rms": l1_to_rms,
        "rms_to_inf": rms_to_inf,
        "supremum": sup,
        "frobenius_norm": frob_norm,
        "average_entry_size": avg_entry,
        "stable_rank": srank,
        "sigma_max": s_max,
    }


@torch.no_grad()
@torch.compile(dynamic=True)
def diag_metrics(v, eps=1e-20):
    """Closed form of `fused_metrics(torch.diag_embed(v))`, without ever
    building the matrix.

    `calculate_norm` expands any 1-D parameter into a `[d, d]` diagonal matrix
    so that the operator norms are defined at all, then runs a full SVD on it.
    Only `numel() == 1` parameters are routed away to `step_scalar`, so every
    RMSNorm weight `[d_model]` in the model takes that path -- twice per
    logging step (update and weight) -- materialising a `d^2` float32 matrix
    and running the slowest cuSOLVER driver over it. For d = 4096 that is a
    64 MiB allocation and an O(d^3) decomposition, to recover numbers that are
    all available from `sort(|v|)` in O(d log d).

    Every quantity is exact, not approximated: for `W = diag(v)` each row and
    each column has a single non-zero entry, so `row_l2 == col_l2 == |v|`, the
    singular values are `sort(|v|, descending)`, and `fan_in == fan_out == d`
    makes `rms_to_rms`'s `sqrt(fan_in/fan_out)` scale exactly 1. Verified
    against the diag_embed path in scratchpad/verify_norm_helper.py.

    One inherited oddity is preserved deliberately: `average_entry_size`
    divides by `sqrt(fan_out * fan_in) == d`, i.e. it averages over the `d^2`
    entries of the embedded matrix, `d^2 - d` of which are structurally zero,
    rather than over the `d` real ones. That is what the current code logs, and
    changing it would put a discontinuity in every in-flight run's series.
    """
    vf = v.reshape(-1).float()
    d = vf.shape[0]
    a = vf.abs()

    S = torch.sort(a, descending=True).values
    s_max = S[0]
    s_min = S[-1]

    # frob_norm mirrors fused_metrics' `row_l2.norm(p=2)` where row_l2 == |v|.
    frob_norm = a.norm(p=2)

    sup = a.amax()
    l1_to_rms = s_max / math.sqrt(d)
    rms_to_inf = s_max * math.sqrt(d)
    # spec = S[0] * sqrt(fan_in / fan_out) and fan_in == fan_out == d.
    spec = s_max

    cond = (s_max / (s_min + eps)).clamp_min(eps)
    srank = ((frob_norm**2) / (s_max**2 + eps)).clamp_min(eps)

    p = (S / (S.sum() + eps)).clamp_min(eps)
    erank = torch.exp(-(p * p.log()).sum())

    S_sq = S * S
    p_sq = (S_sq / (S_sq.sum() + eps)).clamp_min(eps)
    erank_sq = torch.exp(-(p_sq * p_sq.log()).sum())

    avg_entry = frob_norm / d

    return {
        "rms_to_rms": spec,
        "l1_to_rms": l1_to_rms,
        "rms_to_inf": rms_to_inf,
        "supremum": sup,
        "condition_number": cond,
        "frobenius_norm": frob_norm,
        "average_entry_size": avg_entry,
        "stable_rank": srank,
        "effective_rank": erank,
        "effective_rank_squared": erank_sq,
        "spectrum": S,
        # Exposed separately from "spectrum" so radial_helper's spectral
        # metrics can keep using it when spectrum *logging* is off: S[0] is
        # already computed here, whereas shipping the whole vector is what
        # costs packing/all-gather/CPU-transfer. See calculate_norm.
        "sigma_max": S[0],
    }


# cusolverDnSgesvdaStridedBatched takes a batch size but not an unlimited one:
# a batch of 282 [768,2048] matrices raised CUSOLVER_STATUS_INVALID_VALUE, while
# 192 worked. Chunked at 128 for margin -- measured throughput is already within
# 5% of the best observed chunk (1.08 vs 1.03 ms/matrix), so the headroom is
# free.
_GESVDA_MAX_BATCH: int = 128


def _svdvals_batched(W: torch.Tensor, sigma_only: bool = False) -> torch.Tensor:
    """Batched `svdvals` for `[B, m, n]` -> `[B, k]`, chunked and with fallback.

    Batching is what makes the expert path affordable. For qwen30b-a3b at
    EP=64 a rank owns 47 MoE layers x 3 matrices x 2 local experts = 282
    matrices of [768,2048] per logging step. Measured on GH200 / torch 2.12:

        gesvd,  one at a time   42.9 ms each  -> 12.10 s     (what it was)
        gesvda, one at a time    7.4 ms each  ->  2.08 s
        gesvda, batched (128)    1.1 ms each  ->  0.30 s     (~40x)

    Note the batch has to be formed **across layers**, not within a parameter:
    at 1-2 local experts per rank, batching a single param's expert axis buys
    nothing (measured 1.00x at E=1, and 0.56x at E=2 -- actively worse). Note
    also that `gesvd` gains nothing at all from batching, it serialises
    internally, so this win exists only together with the approximate driver.
    """
    if not W.is_cuda:
        return torch.linalg.svdvals(W)
    if sigma_only:
        try:
            B = W.shape[0]
            if B <= _GESVDA_MAX_BATCH:
                return torch.linalg.svdvals(W, driver=SVDVALS_DRIVER_SIGMA_ONLY)
            return torch.cat(
                [
                    torch.linalg.svdvals(
                        W[i : i + _GESVDA_MAX_BATCH],
                        driver=SVDVALS_DRIVER_SIGMA_ONLY,
                    )
                    for i in range(0, B, _GESVDA_MAX_BATCH)
                ],
                dim=0,
            )
        except Exception:
            # Same contract as _svdvals: degrade to slower-but-accurate rather
            # than take down a training run.
            pass
    return torch.linalg.svdvals(W, driver=SVDVALS_DRIVER)


@torch.no_grad()
@torch.compile(dynamic=True)
def _batched_axis_reductions(Wf):
    """Everything `fused_metrics` derives without singular values, for a batch.

    Kept separate from the decomposition so the chunking/fallback logic in
    `_svdvals_batched` stays outside the compiled region (a try/except around a
    cuSOLVER call would graph-break anyway).
    """
    Wf_square = Wf * Wf
    sup = Wf.abs().amax(dim=(-2, -1))
    row_l2 = Wf_square.sum(-1).sqrt()
    col_l2 = Wf_square.sum(-2).sqrt()
    return sup, row_l2.amax(-1), col_l2.amax(-1), row_l2.norm(p=2, dim=-1)


@torch.no_grad()
def calculate_norm_batched(
    W: torch.Tensor,
    norms_to_log: list[str] | None = None,
    transpose: bool = False,
    want_spectrum: bool = True,
    eps: float = 1e-20,
) -> dict[str, torch.Tensor]:
    """`calculate_norm` for a stack of equally-shaped matrices.

    `W` is `[B, m, n]`; every returned scalar metric is `[B]` and "spectrum" is
    `[B, k]`, so entry `i` is exactly what `calculate_norm(W[i], ...)` returns.
    That equivalence is asserted in
    tests/unit_tests/disco_metrics/verify_norm_helper.py.

    Exists for the expert path, which is the only place with many identically
    shaped matrices in scope at once -- see `_svdvals_batched` for why that is
    worth ~40x there. Every expression below is the same one `fused_metrics`
    uses, just with the reductions taken over the trailing two dims.
    """
    if norms_to_log is None:
        norms_to_log = list(NORM_FUNCTIONS.keys())
    if isinstance(W, torch.nn.Parameter):
        W = W.data
    if isinstance(W, DTensor):
        W = W.to_local()
    if W.ndim != 3:
        raise ValueError(
            f"calculate_norm_batched expects [B, m, n], got {tuple(W.shape)}"
        )

    if transpose:
        W = W.transpose(-2, -1)
    Wf = W.float()
    fan_out, fan_in = Wf.shape[-2], Wf.shape[-1]

    requested = set(norms_to_log)
    needs_svd = want_spectrum or not requested.issubset(_SVD_FREE_NORMS)
    needs_full_spectrum = want_spectrum or not requested.issubset(
        _SVD_FREE_NORMS | _SIGMA_ONLY_NORMS
    )

    sup, row_max, col_max, frob_norm = _batched_axis_reductions(Wf)

    out: dict[str, torch.Tensor] = {
        "supremum": sup,
        "l1_to_rms": col_max / math.sqrt(fan_out),
        "rms_to_inf": row_max * math.sqrt(fan_in),
        "frobenius_norm": frob_norm,
        "average_entry_size": frob_norm / math.sqrt(fan_out * fan_in),
    }

    if needs_svd:
        S = (
            _full_spectrum(Wf)
            if needs_full_spectrum
            else _svdvals_batched(Wf, sigma_only=True)
        )
        s_max = S[..., 0]
        out["rms_to_rms"] = s_max * math.sqrt(fan_in / fan_out)
        out["stable_rank"] = ((frob_norm**2) / (s_max**2 + eps)).clamp_min(eps)
        if needs_full_spectrum:
            out["condition_number"] = (
                (S[..., -1] + eps).reciprocal().mul(s_max).clamp_min(eps)
            )
            p = (S / (S.sum(-1, keepdim=True) + eps)).clamp_min(eps)
            out["effective_rank"] = torch.exp(-(p * p.log()).sum(-1))
            S_sq = S * S
            p_sq = (S_sq / (S_sq.sum(-1, keepdim=True) + eps)).clamp_min(eps)
            out["effective_rank_squared"] = torch.exp(-(p_sq * p_sq.log()).sum(-1))
        norms = {name: out[name] for name in norms_to_log}
        if want_spectrum:
            norms["spectrum"] = S
        else:
            norms["sigma_max"] = s_max
        return norms

    return {name: out[name] for name in norms_to_log}


def get_norms_to_log(norms_to_log: str | list[str]) -> list[str]:
    """
    Return a list of norms to log.
    The following contents in `norms_to_log` are special:
    - "default": replaced by
                 ["rms_to_rms", "l1_to_rms", "rms_to_inf", "supremum", "condition_number"]
    - "all" or "everything": log all norms
    """
    if isinstance(norms_to_log, str):
        norms_to_log = [norms_to_log]
    # Remove duplicates while keeping order.
    norms_to_log = list(dict.fromkeys(norms_to_log))

    if "all" in norms_to_log or "everything" in norms_to_log:
        return list(NORM_FUNCTIONS.keys())

    if "default" in norms_to_log:
        # Replace the "default" entry.
        update_index = norms_to_log.index("default")
        norms_to_log = (
            norms_to_log[:update_index]
            + [
                "rms_to_rms",
                "l1_to_rms",
                "rms_to_inf",
                "supremum",
                "condition_number",
            ]
            + norms_to_log[update_index + 1 :]
        )

    return norms_to_log


def calculate_norm(
    W: torch.Tensor,
    norms_to_log: list[str] | None = None,
    transpose: bool = False,
    use_fused_metrics: bool = True,
    want_spectrum: bool = True,
) -> dict[str, torch.Tensor]:
    """
    It is important to note that the order of the norms is the same
    as the order of `norms_to_log`.

    we expect the weights to be [D_out, D_in], otherwise, we need to set `transpose` to True

    `want_spectrum` controls whether the full singular-value vector is returned
    under the "spectrum" key. It used to be injected unconditionally, but the
    only consumer (optimizers/spectrum_logging.py) pops every `track_spectrum_*`
    entry and discards it unless `enable_spectrum_plot` or
    `enable_spectrum_export` is set -- both default to False. Since disco.py
    packs, all-gathers and moves those vectors to CPU on the way there, and
    they are ~99% of that gather's payload, the flag is threaded from the
    optimizer config rather than assumed. When it is False AND no requested
    norm needs singular values, the SVD is skipped entirely.

    Note that callers must use `norms.pop("spectrum", None)` -- the key is
    absent, not None, when `want_spectrum=False`.
    """
    if norms_to_log is None:
        norms_to_log = list(NORM_FUNCTIONS.keys())

    # Unwrap Parameter first (its .data may still be a DTensor)
    if isinstance(W, torch.nn.Parameter):
        W = W.data
    # Then strip DTensor to get a plain local Tensor for torch.compile
    if isinstance(W, DTensor):
        W = W.to_local()

    requested = set(norms_to_log)
    needs_svd = want_spectrum or not requested.issubset(_SVD_FREE_NORMS)
    # Full spectrum only when something actually reads past S[0].
    needs_full_spectrum = want_spectrum or not requested.issubset(
        _SVD_FREE_NORMS | _SIGMA_ONLY_NORMS
    )

    # sigma_max is handed out separately from the full spectrum: radial_helper
    # only needs the leading value, and it is already computed whenever the SVD
    # tier runs, whereas the *vector* is what costs packing, all-gathering and
    # a device-to-host copy. So `want_spectrum=False` still yields sigma_max.
    if W.ndim == 1:
        # A 1-D parameter is conceptually diag(v) -- that is what gives its
        # operator norms a meaning at all. Computed in closed form from
        # sort(|v|) rather than by materialising the d x d matrix and running
        # an O(d^3) SVD over it; `diag_metrics` documents the exact
        # correspondence. `transpose` is a no-op on a diagonal matrix.
        if use_fused_metrics:
            all_norms = diag_metrics(W)
            norms = {name: all_norms[name] for name in norms_to_log}
            if want_spectrum:
                norms["spectrum"] = all_norms["spectrum"]
            else:
                norms["sigma_max"] = all_norms["sigma_max"]
            return norms
        # Non-fused reference path keeps the historical diag_embed behaviour.
        if W.numel() > 1:
            W = torch.diag_embed(W)

    if transpose:
        W = W.transpose(0, 1)

    if use_fused_metrics:
        if not needs_svd:
            all_norms = fused_metrics_no_svd(W)
        elif needs_full_spectrum:
            all_norms = fused_metrics(W)
        else:
            all_norms = fused_metrics_sigma_only(W)
        norms = {norm_name: all_norms[norm_name] for norm_name in norms_to_log}
        if want_spectrum:
            norms["spectrum"] = all_norms["spectrum"]
        elif "sigma_max" in all_norms:
            # Absent in the SVD-free tier, which genuinely cannot produce it.
            norms["sigma_max"] = all_norms["sigma_max"]
    else:
        norms = {norm_name: NORM_FUNCTIONS[norm_name](W) for norm_name in norms_to_log}
        if want_spectrum:
            norms["spectrum"] = (
                _svdvals(W.to(torch.float32)) if W.ndim >= 2 else W.abs()
            )

    return norms
