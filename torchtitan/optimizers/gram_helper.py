# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gram-based metrics comparing a weight's before/after-update geometry against
its raw momentum -- see gram_matrix.md (repo root) for the full math
writeup this ports; formulas below follow that spec's reference
`get_gram_metrics` implementation directly.

Three tensors, all the same shape:
  - `W_before` -- the weight before this step's update.
  - `V_raw`    -- the RAW effective gradient/momentum (whatever's fed into
                  `AbstractDiSCO.lmo()`), not the LMO-processed update.
  - `W_after`  -- the weight after this step's update. disco.py passes the
                  cheap `pseudo_w` approximation here (the same tensor
                  already used for track_param_* norms), not a fresh real
                  post-update read.

`U = W_after - W_before` (the exact realised displacement) and `A = -U` are
derived internally and used throughout -- see gram_matrix.md's "which tensor
answers which question" table for the intuition (raw momentum `V` studies
emergent optimizer-state geometry; the realised displacement `U` studies the
actual trajectory that moved the weights).

Three cumulative levels, gated by a single `level: int` argument:
  0: nothing (returns {} immediately -- the cheap no-op every disco.py call
     site relies on).
  1: O(m^2) entrywise geometry (row-dominance ratios, off-diagonal
     correlation stats, weight/momentum/update self- and cross-alignment,
     exact weight-Gram change).
  2: adds O(m^3) spectral structure (Gram/correlation eigenspectra for all
     4 tensors, subspace overlaps, weight-eigenbasis energy).
  3: adds whitened/generalised geometry (relative-motion and relative-flow
     eigenspectra, cross-Gram singular values, canonical correlations) --
     most expensive, most numerically sensitive. The reference's optional
     `include_cross_svd` extras (cross-Gram singular values, canonical
     correlations) are folded in unconditionally here rather than exposed
     as a separate flag -- both only ever run at level 3.

Every vector-valued metric has length `m` (the row-count after orientation
is applied) -- every Gram/correlation matrix here is square `m x m` (built
as `X @ Y.T`, never `Y.T @ X`). Orientation is decided unconditionally by
shape, inside `calculate_gram_metrics`: rows are always transposed to be
<= cols (`m = min(D_out, D_in)`), regardless of any caller-supplied
`transpose` argument -- see that function's docstring for why (numerical:
guarantees a reduced SVD spans the complete eigenspace; and a fix for the
large-vocab OOM that name-based `need_T` matching missed for `output`).

Where a full vector is itself returned, its generic percentile/mean/min/max
summary is intentionally NOT also returned (redundant, reconstructable
post-hoc from the logged vector); named *nonlinear* reductions are kept
alongside their vector. Two vectors from the reference are dropped as
trivially derivable from an already-logged vector: `raw_trace_normalised`/
`actual_trace_normalised` (a simple `/ sum()` of `K_V_eigenvalues`/
`K_U_eigenvalues`, both already returned).
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch.distributed.tensor import DTensor

from torchtitan.optimizers.norm_helper import _gram_spectrum, fp64_gram

_DEFAULT_GRAM_EPS: float = 1e-6
_DEFAULT_GRAM_TOPK: int = 8


def _prep(X: torch.Tensor) -> torch.Tensor:
    if isinstance(X, torch.nn.Parameter):
        X = X.data
    if isinstance(X, DTensor):
        X = X.to_local()
    if X.ndim == 1 and X.numel() > 1:
        X = torch.diag_embed(X)
    return X


def gram_matrix_is_transposed(shape: tuple[int, ...]) -> bool:
    """Whether Gram metrics orient the final two matrix dimensions as ``X.T``."""
    return len(shape) >= 2 and int(shape[-2]) > int(shape[-1])


def gram_vector_len(shape: tuple[int, ...], transpose: bool = False) -> int:
    """
    Static, shape-derived length of every vector-valued gram metric for a
    parameter of this local shape -- for disco.py's per-param offset-table
    precompute (DDP/FSDP/experts). `transpose` is accepted only for
    call-site compatibility and is ignored: `calculate_gram_metrics`
    decides orientation itself from the tensor's actual shape (always
    `rows <= cols`, see its docstring), so this must independently track
    `min(shape[-2], shape[-1])` regardless of what's passed here -- a
    mismatch between the two would corrupt disco.py's packed offset
    tables.
    """
    if len(shape) == 1:
        return max(int(shape[0]), 1)
    return min(int(shape[-2]), int(shape[-1]))


def _gram(X: torch.Tensor) -> torch.Tensor:
    return X @ X.T


def _relative_floor(reference: torch.Tensor, eps_rel: float) -> torch.Tensor:
    # Scale-relative floor for a denominator that's structurally unbounded
    # relative to its numerator (can be exactly 0 while the numerator is
    # positive -- e.g. perfectly orthogonal rows, or a rank-deficient Gram
    # matrix) -- keeps the resulting *ratio* capped at a fixed,
    # scale-independent ceiling (1 / eps_rel) instead of blowing up
    # arbitrarily as the true denominator approaches 0. An absolute eps
    # can't do this: it doesn't scale with the input, so it either swamps
    # small-but-legitimate values (business-scale eps) or lets the ratio
    # blow up unpredictably near 0 (any fixed small eps). Backstopped with
    # an absolute machine-tiny floor for the fully-degenerate case where
    # `reference` itself is exactly 0 (avoids a literal 0/0).
    tiny = torch.finfo(reference.dtype).tiny
    return (eps_rel * reference).clamp_min(tiny)


def _row_normalise(X: torch.Tensor) -> torch.Tensor:
    # Floor each row by its OWN norm only, never a whole-matrix reference:
    # a matrix-wide floor under-normalises any row that's disproportionately
    # smaller than the rest of the matrix (e.g. a near-dead neuron sitting
    # alongside normal-scale rows), since the floor would then reflect the
    # OTHER rows' scale, not this row's. A row's norm is either exactly 0
    # (undefined direction) or some positive value that normalises
    # correctly on its own terms regardless of other rows' scale, so a bare
    # machine-tiny floor is both correct and sufficient.
    norm = X.norm(dim=1, keepdim=True)
    tiny = torch.finfo(X.dtype).tiny
    normalised = X / norm.clamp_min(tiny)
    return torch.where(norm > 0, normalised, torch.zeros_like(normalised))


def _corr(X: torch.Tensor) -> torch.Tensor:
    X_hat = _row_normalise(X)
    return X_hat @ X_hat.T


def _cross_corr(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return _row_normalise(X) @ _row_normalise(Y).T


def _offdiag(A: torch.Tensor) -> torch.Tensor:
    return A - torch.diag_embed(torch.diagonal(A))


def _dominance_ratio(diagonal: torch.Tensor, off_mean: torch.Tensor) -> torch.Tensor:
    # diagonal / off_mean, meant to be scale-invariant (both are reductions
    # of the same matrix) -- shared by _row_dominance and _cross_summary's
    # specificity, which were the same formula duplicated inline.
    eps_rel = torch.finfo(diagonal.dtype).eps
    denominator = torch.maximum(off_mean, _relative_floor(diagonal, eps_rel))
    # `torch.where`, not boolean-mask indexing. `result[m] = a[m] / b[m]`
    # runs `nonzero()` for each of the three masked accesses, and every
    # `nonzero()` is a device->host sync because the output shape is
    # data-dependent. This helper is called ~12x per calculate_gram_metrics,
    # so that was ~36 syncs per call on a path that is already
    # dispatch-bound. `where` selects with no sync and no shape dependence.
    #
    # The unselected branch may be inf/NaN (denominator can be 0 exactly
    # where diagonal is 0), but `where` discards it rather than propagating
    # -- same deterministic-0 sentinel convention as radial_helper's guards.
    safe = denominator.clamp_min(torch.finfo(diagonal.dtype).tiny)
    return torch.where(diagonal > 0, diagonal / safe, torch.zeros_like(diagonal))


def _row_dominance(A: torch.Tensor, m: int) -> torch.Tensor:
    # Sum the off-diagonal entries directly (mask the diagonal out first)
    # rather than `row_sum - diagonal` -- the subtraction form suffers
    # catastrophic cancellation exactly when the matrix is highly
    # diagonal-dominant (the regime this metric is meant to detect):
    # row_sum ~= diagonal there, so their difference loses most of its
    # precision instead of correctly coming out small.
    abs_A = A.abs()
    diagonal = torch.diagonal(abs_A)
    off_A = abs_A.clone()
    off_A.fill_diagonal_(0)
    off_mean = off_A.sum(dim=1) / max(m - 1, 1)
    return _dominance_ratio(diagonal, off_mean)


def _offdiag_stats(
    C: torch.Tensor, m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Zero the diagonal in a copy rather than gathering with a boolean mask.
    # `C[off_mask]` allocates an m x m bool mask AND runs a data-dependent
    # `nonzero()` gather producing m^2 - m elements (589k at m=768) plus a
    # device->host sync. Zeroing is one kernel and keeps everything on device.
    #
    # Deliberately NOT computed as (total - diagonal): that is the same
    # catastrophic cancellation `_row_dominance` documents avoiding, and it
    # bites exactly in the diagonally-dominant regime these metrics exist to
    # detect. Zeroed entries contribute nothing to a sum of non-negatives, and
    # `max` is unaffected because `ax` is non-negative -- if every off-diagonal
    # is 0 the true max is 0 anyway.
    count = max(m * m - m, 1)
    ax = C.abs()
    ax.fill_diagonal_(0)
    sq = C.square()
    sq.fill_diagonal_(0)
    return ax.sum() / count, (sq.sum() / count).sqrt(), ax.max()


def _cross_summary(C_XY: torch.Tensor, m: int) -> dict[str, torch.Tensor]:
    diagonal = torch.diagonal(C_XY)
    # Same off-diagonal-masking fix as _row_dominance -- avoid
    # `row_sum - diagonal`'s catastrophic cancellation under high diagonal
    # dominance.
    abs_C = C_XY.abs()
    off_C = abs_C.clone()
    off_C.fill_diagonal_(0)
    off_mean = off_C.sum(dim=1) / max(m - 1, 1)
    specificity = _dominance_ratio(diagonal.abs(), off_mean)
    indices = torch.arange(m, device=C_XY.device)
    # diagonal_energy_fraction's numerator is a strict energy subset of its
    # denominator (diagonal^2 <= sum of all entries^2), so it's already
    # bounded in [0, 1] -- a small absolute floor (not a relative one) is
    # enough to avoid 0/0 without risking any blow-up.
    tiny = torch.finfo(C_XY.dtype).tiny
    return {
        "diagonal": diagonal,
        "row_specificity": specificity,
        "row_top1_identity": (
            (C_XY.abs().argmax(dim=1) == indices).to(C_XY.dtype).mean()
        ),
        "column_top1_identity": (
            (C_XY.abs().argmax(dim=0) == indices).to(C_XY.dtype).mean()
        ),
        "diagonal_energy_fraction": (
            diagonal.square().sum() / C_XY.square().sum().clamp_min(tiny)
        ),
    }


def _matrix_cosine(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Cauchy-Schwarz bounds this in [-1, 1] regardless of A/B's scale, so a
    # tiny absolute floor (not a relative one) is enough to avoid 0/0.
    tiny = torch.finfo(A.dtype).tiny
    return (A * B).sum() / (A.norm() * B.norm()).clamp_min(tiny)


def _effective_rank(eigenvalues: torch.Tensor) -> torch.Tensor:
    values = eigenvalues.clamp_min(0)
    total = values.sum()
    # The probability sum is bounded (each value <= the sum of all
    # non-negative values), so a tiny absolute floor suffices here -- an
    # absolute business-scale eps would otherwise stop `probabilities` from
    # summing to ~1 whenever the whole spectrum is uniformly small,
    # corrupting the entropy below. `xlogy` handles the p=0 entropy term
    # exactly (0 * log(0) := 0 by definition) with no eps-in-the-log fudge
    # needed, and without an eps distorting log(p) for small-but-positive p
    # the way `log(p + eps)` would.
    tiny = torch.finfo(values.dtype).tiny
    probabilities = values / total.clamp_min(tiny)
    entropy = -torch.xlogy(probabilities, probabilities).sum()
    effective_rank = torch.exp(entropy)
    # A fully zero spectrum spans no directions -- effective_rank should be
    # 0, not exp(0)=1 (which the formula above would otherwise give: every
    # probability is 0/tiny=0, xlogy(0,0)=0, so entropy=0).
    return torch.where(total > 0, effective_rank, torch.zeros_like(effective_rank))


def _spectral_summary(eigenvalues: torch.Tensor, topk: int) -> dict[str, torch.Tensor]:
    # Expects descending-sorted, non-negative-clamped eigenvalues.
    eigenvalues = eigenvalues.clamp_min(0)
    total = eigenvalues.sum()
    k = min(topk, eigenvalues.numel())
    eps_rel = torch.finfo(eigenvalues.dtype).eps
    tiny = torch.finfo(eigenvalues.dtype).tiny
    return {
        "effective_rank": _effective_rank(eigenvalues),
        "largest": eigenvalues[0],
        "smallest": eigenvalues[-1],
        # Condition number is genuinely unbounded (smallest eigenvalue can
        # be exactly 0 for a rank-deficient matrix) -- same relative-floor
        # treatment as row_dominance, not a tiny absolute floor.
        "condition_regularized": eigenvalues[0]
        / torch.maximum(eigenvalues[-1], _relative_floor(eigenvalues[0], eps_rel)),
        "topk_energy_fraction": eigenvalues[:k].sum() / total.clamp_min(tiny),
    }


def _inverse_sqrt_from_eigh(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    matrix_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Scale-relative floor (NOT a bare eps clamp) -- keeps this
    # scale-invariant across params of very different magnitude. The old
    # `matrix_scale.clamp_min(eps)` broke exactly that for matrix_scale <
    # eps: clamping matrix_scale itself up to eps first made the floor
    # collapse to a fixed eps^2 regardless of how much smaller matrix_scale
    # actually was -- the same absolute-floor swamping bug fixed elsewhere
    # in this file, hiding here too. Floor purely proportionally instead,
    # backstopped only by machine-tiny for the literal matrix_scale == 0
    # case.
    tiny = torch.finfo(eigenvalues.dtype).tiny
    scale = matrix_scale.clamp_min(0)
    floor = (eps * scale).clamp_min(tiny)
    # A fully zero-scale matrix has no direction to whiten relative to --
    # define its inverse-sqrt as the zero matrix rather than an arbitrary
    # huge value from flooring near-zero eigenvalues up to `tiny`.
    active = (scale > 0).to(eigenvalues.dtype)
    inv_sqrt_values = eigenvalues.clamp_min(floor).rsqrt() * active
    return (eigenvectors * inv_sqrt_values.unsqueeze(0)) @ eigenvectors.T


def _gram_eigh_from_factor(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ascending eigenpairs of X @ X.T via SVD of X directly -- avoids
    explicitly forming X @ X.T, which squares the condition number
    (kappa(X @ X.T) = kappa(X)^2) and is measurably more likely to trip up
    eigh's convergence (observed in practice: a bf16-all-to-all-derived
    matrix, upcast to fp32, hit "eigh: the algorithm failed to converge
    because the input matrix is ill-conditioned or has too many repeated
    eigenvalues" on exactly this kind of explicitly-formed product).
    Assumes rows <= cols (guaranteed by calculate_gram_metrics's
    orientation policy -- see its docstring), so the Gram is over the
    smaller dimension and spans the complete eigenspace, with no
    rank-deficient dimensions to pad.

    NOTE the docstring above used to argue for the SVD *because* forming
    X @ X.T squares the condition number. That reasoning is right in
    float32 and is exactly why this now uses `norm_helper.fp64_gram`:
    in float64 the squared condition number is affordable, and the result
    is both faster (4.4x with eigenvectors at [768,2048]) and MORE accurate
    than the float32 SVD it replaces (smallest eigenvalue 34-500x closer to
    a float64 reference). `eigh` returns ascending eigenpairs directly, so
    no flip is needed. Eigenvector signs may differ from the SVD's, which
    is fine: every downstream use is of the form V diag(f(w)) V^T (see
    `_inverse_sqrt_from_eigh`), which is sign-invariant.
    """
    w, V = torch.linalg.eigh(fp64_gram(X))
    return w.clamp_min(0).to(X.dtype), V.to(X.dtype)


def _svd_eigenvalues(X: torch.Tensor) -> torch.Tensor:
    """
    Descending eigenvalues of X @ X.T, values only (no eigenvectors) --
    same float64-Gram reasoning as `_gram_eigh_from_factor`, and cheaper
    still when eigenvectors aren't needed (5.9x over `svdvals` at
    [768,2048]).
    """
    return torch.linalg.eigvalsh(fp64_gram(X)).clamp_min(0).flip(-1).to(X.dtype)


def _safe_sym_eigvalsh(A: torch.Tensor) -> torch.Tensor:
    """
    Signed eigenvalues in ascending order, for matrices that are NOT a
    single factor's Gram product (delta_GW, J -- differences of PSD
    matrices, genuinely indefinite, so the SVD-of-factor trick above
    doesn't apply). Rescales to unit max-abs-entry before eigvalsh (helps
    LAPACK resolve close eigenvalues -- fp32's *absolute* precision
    degrades at large magnitude even though its *relative* precision
    doesn't, and genuinely-distinct eigenvalues can round together at
    large scale) and forces exact numerical symmetry, then rescales the
    result back.
    """
    # Symmetrize BEFORE measuring scale, not after -- scale must reflect
    # the actual matrix being normalised and decomposed (A_sym), not a
    # possibly-slightly-different unsymmetrized A. Matches
    # _safe_psd_eigh/_safe_psd_eigvalsh's ordering, which already got this
    # right.
    A_sym = 0.5 * (A + A.T)
    scale = A_sym.abs().amax()
    if scale == 0:
        return torch.zeros(A.shape[0], device=A.device, dtype=A.dtype)
    return torch.linalg.eigvalsh(A_sym / scale) * scale


def _safe_psd_eigh(
    G: torch.Tensor, factor: Callable[[], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ascending eigenpairs of a PSD matrix G (== factor @ factor.T). Tries a
    cheap, rescaled+symmetrized `eigh` on the already-formed G first (fast,
    and numerically robust for the vast majority of inputs); falls back to
    the more expensive but bulletproof SVD-of-factor
    (`_gram_eigh_from_factor`) only if `eigh` actually raises. `factor` is
    a zero-arg callable, not a tensor -- SVD-of-factor is measurably more
    expensive than eigh for wide matrices (scales with the larger
    dimension rather than collapsing it away via the Gram product), so it
    must stay unevaluated unless the fallback path is actually taken.
    """
    G_sym = 0.5 * (G + G.T)
    scale = G_sym.abs().amax()
    n = G_sym.shape[0]
    if scale == 0:
        return (
            torch.zeros(n, device=G.device, dtype=G.dtype),
            torch.eye(n, device=G.device, dtype=G.dtype),
        )
    try:
        values, vectors = torch.linalg.eigh(G_sym / scale)
        return values.clamp_min(0) * scale, vectors
    except torch._C._LinAlgError:
        return _gram_eigh_from_factor(factor())


def _safe_psd_eigvalsh(
    G: torch.Tensor, factor: Callable[[], torch.Tensor]
) -> torch.Tensor:
    """
    Descending eigenvalues of a PSD matrix G (== factor @ factor.T),
    values only. Same try-cheap-eigh-first-fall-back-to-SVD-of-factor
    strategy as `_safe_psd_eigh`, and the same lazy-`factor` contract.
    """
    G_sym = 0.5 * (G + G.T)
    scale = G_sym.abs().amax()
    n = G_sym.shape[0]
    if scale == 0:
        return torch.zeros(n, device=G.device, dtype=G.dtype)
    try:
        values = torch.linalg.eigvalsh(G_sym / scale)
        return values.clamp_min(0).flip(0) * scale
    except torch._C._LinAlgError:
        return _svd_eigenvalues(factor())


@dataclass
class _GramCore:
    m: int
    W_before: torch.Tensor
    V_raw: torch.Tensor
    W_after: torch.Tensor
    U_actual: torch.Tensor
    A_actual: torch.Tensor
    G_Wm: torch.Tensor
    G_Wp: torch.Tensor
    G_V: torch.Tensor
    G_U: torch.Tensor
    C_Wm: torch.Tensor
    C_Wp: torch.Tensor
    C_V: torch.Tensor
    C_U: torch.Tensor
    C_WV: torch.Tensor
    C_WU: torch.Tensor
    C_VA: torch.Tensor
    delta_GW: torch.Tensor
    delta_CW: torch.Tensor
    naive_delta_GW: torch.Tensor
    # Row-normalized factors (C_X = X_hat @ X_hat.T) -- stored so level 2's
    # SVD-based C_* eigenvalues (_svd_eigenvalues(X_hat)) reuse the same
    # normalization computed here, instead of row-normalizing a second time.
    Wm_hat: torch.Tensor
    Wp_hat: torch.Tensor
    V_hat: torch.Tensor
    U_hat: torch.Tensor


def _build_gram_core(
    W_before: torch.Tensor, V_raw: torch.Tensor, W_after: torch.Tensor
) -> _GramCore:
    m = W_before.shape[0]
    U_actual = W_after - W_before
    A_actual = -U_actual

    G_Wm, G_Wp, G_V, G_U = (
        _gram(W_before),
        _gram(W_after),
        _gram(V_raw),
        _gram(U_actual),
    )
    Wm_hat, Wp_hat, V_hat, U_hat = (
        _row_normalise(W_before),
        _row_normalise(W_after),
        _row_normalise(V_raw),
        _row_normalise(U_actual),
    )
    C_Wm, C_Wp, C_V, C_U = (
        Wm_hat @ Wm_hat.T,
        Wp_hat @ Wp_hat.T,
        V_hat @ V_hat.T,
        U_hat @ U_hat.T,
    )
    C_WV = _cross_corr(W_before, V_raw)
    C_WU = _cross_corr(W_before, U_actual)
    C_VA = _cross_corr(V_raw, A_actual)

    # Exact algebraic identity: G_Wp - G_Wm == W_before@U^T + U@W_before^T +
    # U@U^T (U = W_after - W_before). Used as the PRIMARY delta_GW (not
    # just a sanity-check reconstruction): G_Wp/G_Wm can both be large in
    # magnitude while their true difference is small (a small update
    # relative to the weight's own scale -- the common case), so computing
    # delta_GW as a direct subtraction of two large matrices is
    # catastrophic-cancellation-prone. This product-based form only
    # involves the already-small U_actual, so it doesn't suffer the same
    # precision loss -- and it's what feeds G_W_change_eigenvalues's eigh
    # and J's construction below, exactly the numerically-sensitive path
    # that benefits most. `naive_delta_GW` (the direct subtraction) is kept
    # separately, purely so gram_identity_residual stays a real
    # precision-loss diagnostic instead of comparing a value to itself.
    delta_GW = W_before @ U_actual.T + U_actual @ W_before.T + G_U
    delta_CW = C_Wp - C_Wm
    naive_delta_GW = G_Wp - G_Wm

    return _GramCore(
        m=m,
        W_before=W_before,
        V_raw=V_raw,
        W_after=W_after,
        U_actual=U_actual,
        A_actual=A_actual,
        G_Wm=G_Wm,
        G_Wp=G_Wp,
        G_V=G_V,
        G_U=G_U,
        C_Wm=C_Wm,
        C_Wp=C_Wp,
        C_V=C_V,
        C_U=C_U,
        C_WV=C_WV,
        C_WU=C_WU,
        C_VA=C_VA,
        delta_GW=delta_GW,
        delta_CW=delta_CW,
        naive_delta_GW=naive_delta_GW,
        Wm_hat=Wm_hat,
        Wp_hat=Wp_hat,
        V_hat=V_hat,
        U_hat=U_hat,
    )


def _level1_metrics(core: _GramCore) -> dict[str, torch.Tensor]:
    m = core.m
    # Cross-tensor ratios below (numerator/denominator from genuinely
    # different tensors, e.g. update norm vs. weight norm) have no natural
    # same-tensor relative reference -- a weight/momentum/etc. can be
    # legitimately exactly 0, so full scale-invariance isn't achievable.
    # Just avoid literal 0/0 with a tiny absolute floor, same as any other
    # structurally-bounded-elsewhere ratio.
    tiny = torch.finfo(core.W_before.dtype).tiny

    V_mean_abs, V_rms, V_max_abs = _offdiag_stats(core.C_V, m)
    U_mean_abs, U_rms, U_max_abs = _offdiag_stats(core.C_U, m)
    Wm_mean_abs, Wm_rms, Wm_max_abs = _offdiag_stats(core.C_Wm, m)
    Wp_mean_abs, Wp_rms, Wp_max_abs = _offdiag_stats(core.C_Wp, m)

    # Sign-convention note: VA is built from A_actual (= -U_actual, the
    # descent-oriented direction), but WU is built from U_actual itself (the
    # literal displacement W_after - W_before) -- these are DIFFERENT sign
    # conventions relative to each other by design (WU_diagonal answers "did
    # this row grow/shrink", VA_diagonal answers "does momentum point the
    # way weights actually moved"). If you want a descent-oriented
    # counterpart of WU (cos(w_before, a) instead of cos(w_before, u)), it's
    # a pure sign flip -- row_normalise(-X) == -row_normalise(X), so
    # cross_corr(W_before, A_actual) == -C_WU exactly (not just its
    # diagonal) -- WA_diagonal = -WU["diagonal"], WA_row_specificity ==
    # WU["row_specificity"] unchanged (abs()-based), etc. Not returned as a
    # separate metric since it's trivially derivable from what's already
    # logged (same "drop what's a linear/simple transform of an
    # already-logged vector" rule as principal_angles_radians before).
    VA = _cross_summary(core.C_VA, m)
    WV = _cross_summary(core.C_WV, m)
    WU = _cross_summary(core.C_WU, m)

    U_relative_step_fro = core.U_actual.norm() / core.W_before.norm().clamp_min(tiny)
    update_to_momentum_isotropy_ratio = U_rms / V_rms.clamp_min(tiny)
    gram_geometry_alignment = _matrix_cosine(_offdiag(core.C_V), _offdiag(core.C_U))
    global_direction_alignment = _matrix_cosine(core.V_raw, core.A_actual)

    gram_change_relative = core.delta_GW.norm() / core.G_Wm.norm().clamp_min(tiny)
    correlation_change_per_row = core.delta_CW.norm() / (m**0.5)
    relational_change_fraction = _offdiag(core.delta_GW).square().sum() / (
        core.delta_GW.square().sum().clamp_min(tiny)
    )
    gram_identity_residual = (
        core.naive_delta_GW - core.delta_GW
    ).norm() / core.delta_GW.norm().clamp_min(tiny)

    return {
        "V_R_raw": _row_dominance(core.G_V, m),
        "V_R_cos": _row_dominance(core.C_V, m),
        "U_R_raw": _row_dominance(core.G_U, m),
        "U_R_cos": _row_dominance(core.C_U, m),
        "Wm_R_raw": _row_dominance(core.G_Wm, m),
        "Wm_R_cos": _row_dominance(core.C_Wm, m),
        "Wp_R_raw": _row_dominance(core.G_Wp, m),
        "Wp_R_cos": _row_dominance(core.C_Wp, m),
        "VA_diagonal": VA["diagonal"],
        "VA_row_specificity": VA["row_specificity"],
        "WV_diagonal": WV["diagonal"],
        "WV_row_specificity": WV["row_specificity"],
        "WU_diagonal": WU["diagonal"],
        "WU_row_specificity": WU["row_specificity"],
        "V_offdiag_mean_abs": V_mean_abs,
        "V_offdiag_rms": V_rms,
        "V_offdiag_max_abs": V_max_abs,
        "U_offdiag_mean_abs": U_mean_abs,
        "U_offdiag_rms": U_rms,
        "U_offdiag_max_abs": U_max_abs,
        "U_relative_step_fro": U_relative_step_fro,
        "Wm_offdiag_mean_abs": Wm_mean_abs,
        "Wm_offdiag_rms": Wm_rms,
        "Wm_offdiag_max_abs": Wm_max_abs,
        "Wp_offdiag_mean_abs": Wp_mean_abs,
        "Wp_offdiag_rms": Wp_rms,
        "Wp_offdiag_max_abs": Wp_max_abs,
        "update_to_momentum_isotropy_ratio": update_to_momentum_isotropy_ratio,
        "gram_geometry_alignment": gram_geometry_alignment,
        "global_direction_alignment": global_direction_alignment,
        "VA_row_top1_identity": VA["row_top1_identity"],
        "VA_column_top1_identity": VA["column_top1_identity"],
        "VA_diagonal_energy_fraction": VA["diagonal_energy_fraction"],
        "WV_row_top1_identity": WV["row_top1_identity"],
        "WV_column_top1_identity": WV["column_top1_identity"],
        "WV_diagonal_energy_fraction": WV["diagonal_energy_fraction"],
        "WU_row_top1_identity": WU["row_top1_identity"],
        "WU_column_top1_identity": WU["column_top1_identity"],
        "WU_diagonal_energy_fraction": WU["diagonal_energy_fraction"],
        "gram_change_relative": gram_change_relative,
        "correlation_change_per_row": correlation_change_per_row,
        "relational_change_fraction": relational_change_fraction,
        # Compares the direct-subtraction naive_delta_GW against the
        # numerically-stable, product-based delta_GW used as the real
        # delta_GW everywhere else -- expected to be small (this identity
        # holds exactly in exact arithmetic, regardless of how faithful
        # W_after/pseudo_w is to a true post-update read) but is NOT
        # tautologically zero: it's now a genuine (if usually tiny)
        # measure of the naive subtraction's cancellation error, not just
        # floating-point noise between two equally-precise paths. Still a
        # dtype/precision diagnostic, not a training-dynamics signal.
        "gram_identity_residual": gram_identity_residual,
    }


@dataclass
class _Level2Extras:
    # Ascending order, straight from eigh -- level 3 reuses these directly
    # rather than re-flipping to descending and back: U @ diag(f(eigvals)) @
    # U.T is invariant to eigenpair ordering, so this is a pure
    # simplification, not a behavior change.
    gwm_asc: torch.Tensor
    UWm_asc: torch.Tensor
    gv_asc: torch.Tensor
    UV_asc: torch.Tensor
    gu_asc: torch.Tensor
    UU_asc: torch.Tensor


def _level2_metrics(
    core: _GramCore, topk: int
) -> tuple[dict[str, torch.Tensor], _Level2Extras]:
    m = core.m

    gwm_asc, UWm_asc = _safe_psd_eigh(core.G_Wm, lambda: core.W_before)
    gwp_asc, UWp_asc = _safe_psd_eigh(core.G_Wp, lambda: core.W_after)
    gv_asc, UV_asc = _safe_psd_eigh(core.G_V, lambda: core.V_raw)
    gu_asc, UU_asc = _safe_psd_eigh(core.G_U, lambda: core.U_actual)

    gwm, UWm = gwm_asc.flip(0), UWm_asc.flip(1)
    gwp, UWp = gwp_asc.flip(0), UWp_asc.flip(1)
    gv, UV = gv_asc.flip(0), UV_asc.flip(1)
    gu, UU = gu_asc.flip(0), UU_asc.flip(1)

    cwm = _safe_psd_eigvalsh(core.C_Wm, lambda: core.Wm_hat)
    cwp = _safe_psd_eigvalsh(core.C_Wp, lambda: core.Wp_hat)
    cv = _safe_psd_eigvalsh(core.C_V, lambda: core.V_hat)
    cu = _safe_psd_eigvalsh(core.C_U, lambda: core.U_hat)

    k = min(topk, m)

    def overlap(UX: torch.Tensor, UY: torch.Tensor) -> torch.Tensor:
        return (UX[:, :k].T @ UY[:, :k]).square().sum() / k

    q_V = torch.diagonal(UWm.T @ core.G_V @ UWm).clamp_min(0)
    q_U = torch.diagonal(UWm.T @ core.G_U @ UWm).clamp_min(0)
    # Bounded distributions (each entry <= the sum of all non-negative
    # entries) -- a tiny absolute floor suffices, same reasoning as
    # _effective_rank's probabilities.
    q_tiny = torch.finfo(q_V.dtype).tiny
    q_V_dist = q_V / q_V.sum().clamp_min(q_tiny)
    q_U_dist = q_U / q_U.sum().clamp_min(q_tiny)

    scalars: dict[str, torch.Tensor] = {}
    for prefix, eig in (
        ("G_Wm", gwm),
        ("G_Wp", gwp),
        ("G_V", gv),
        ("G_U", gu),
        ("C_Wm", cwm),
        ("C_Wp", cwp),
        ("C_V", cv),
        ("C_U", cu),
    ):
        for name, val in _spectral_summary(eig, topk).items():
            scalars[f"{prefix}_{name}"] = val

    scalars.update(
        {
            "C_Wm_deviation_from_identity": (cwm - 1).square().mean().sqrt(),
            "C_Wp_deviation_from_identity": (cwp - 1).square().mean().sqrt(),
            "C_V_deviation_from_identity": (cv - 1).square().mean().sqrt(),
            "C_U_deviation_from_identity": (cu - 1).square().mean().sqrt(),
            "overlap_Wm_V": overlap(UWm, UV),
            "overlap_Wm_U": overlap(UWm, UU),
            "overlap_V_U": overlap(UV, UU),
            "overlap_Wm_Wp": overlap(UWm, UWp),
            "G_W_effective_rank_delta": (_effective_rank(gwp) - _effective_rank(gwm)),
        }
    )

    # gwp - gwm compares the i-th LARGEST eigenvalue of G_Wp against the
    # i-th largest of G_Wm -- a rank/position-wise comparison of two
    # independently-sorted spectra, NOT a per-eigenvector-tracked change
    # (sorting can reshuffle which actual eigenvector lands at position i
    # between the two matrices) -- named accordingly, not just
    # "eigenvalue_delta". Contrast with G_W_change_eigenvalues below, the
    # eigenvalues of the actual difference matrix delta_GW (the
    # numerically-stable product-based formula, mathematically equal to
    # G_Wp - G_Wm -- see _build_gram_core) itself -- a more principled
    # measure of the Gram change's own spectral
    # content, unaffected by any eigenvector reshuffling between G_Wp/G_Wm.
    # Unlike G_W/C_W eigenvalues (always >=0, real Gram/correlation
    # matrices), delta_GW is a difference of two PSD matrices and generally
    # indefinite -- signed, not clamped, same treatment as level 3's
    # J_eigenvalues (delta_GW's whitened counterpart).
    G_W_rankwise_eigenvalue_delta = gwp - gwm
    G_W_change_eigenvalues = _safe_sym_eigvalsh(core.delta_GW).flip(0)

    vectors = {
        "G_Wm_eigenvalues": gwm,
        "G_Wp_eigenvalues": gwp,
        "G_V_eigenvalues": gv,
        "G_U_eigenvalues": gu,
        "C_Wm_eigenvalues": cwm,
        "C_Wp_eigenvalues": cwp,
        "C_V_eigenvalues": cv,
        "C_U_eigenvalues": cu,
        "energy_V_in_Wm_basis": q_V,
        "energy_U_in_Wm_basis": q_U,
        "energy_V_in_Wm_basis_distribution": q_V_dist,
        "energy_U_in_Wm_basis_distribution": q_U_dist,
        "G_W_rankwise_eigenvalue_delta": G_W_rankwise_eigenvalue_delta,
        "G_W_change_eigenvalues": G_W_change_eigenvalues,
    }

    extras = _Level2Extras(
        gwm_asc=gwm_asc,
        UWm_asc=UWm_asc,
        gv_asc=gv_asc,
        UV_asc=UV_asc,
        gu_asc=gu_asc,
        UU_asc=UU_asc,
    )
    return {**scalars, **vectors}, extras


def _log_rate_spread(rates: torch.Tensor) -> torch.Tensor:
    # log(rates + eps) has the same absolute-eps swamping problem as
    # everywhere else in this file: a genuine eigendirection the update
    # doesn't touch at all gives rate == 0 exactly (not just floating-point
    # noise), and if the OTHER rates are uniformly small too (a small
    # update relative to the weight, a real training regime), an absolute
    # eps makes every log(rate + eps) collapse toward the same log(eps)
    # constant, corrupting the spread. Floor each rate relative to the
    # largest rate instead -- unlike _row_normalise (where referencing
    # other rows was wrong, since rows are logically independent), the
    # rates being compared against each other via max_rate is exactly what
    # "spread" means here, so this is the right reference.
    eps_rel = torch.finfo(rates.dtype).eps
    max_rate = rates.max()
    floor = _relative_floor(max_rate, eps_rel)
    log_rates = torch.log(torch.maximum(rates, floor))
    spread = log_rates.std()
    return torch.where(max_rate > 0, spread, torch.zeros_like(spread))


def _level3_metrics(
    core: _GramCore, eps: float, extras: _Level2Extras
) -> dict[str, torch.Tensor]:
    W_scale = torch.diagonal(core.G_Wm).mean()
    V_scale = torch.diagonal(core.G_V).mean()
    U_scale = torch.diagonal(core.G_U).mean()

    GW_inv_sqrt = _inverse_sqrt_from_eigh(extras.gwm_asc, extras.UWm_asc, W_scale, eps)
    GV_inv_sqrt = _inverse_sqrt_from_eigh(extras.gv_asc, extras.UV_asc, V_scale, eps)
    GU_inv_sqrt = _inverse_sqrt_from_eigh(extras.gu_asc, extras.UU_asc, U_scale, eps)

    # K_V = GW_inv_sqrt @ G_V @ GW_inv_sqrt = factor @ factor.T for
    # factor = GW_inv_sqrt @ V_raw (GW_inv_sqrt is exactly symmetric by
    # construction). Try the cheap eigh on the already-available G_V
    # sandwich first; the factor (an extra matmul, wasted if eigh
    # succeeds) is only actually computed on the rare fallback path.
    K_V = GW_inv_sqrt @ core.G_V @ GW_inv_sqrt
    K_U = GW_inv_sqrt @ core.G_U @ GW_inv_sqrt
    eig_KV = _safe_psd_eigvalsh(K_V, lambda: GW_inv_sqrt @ core.V_raw)
    eig_KU = _safe_psd_eigvalsh(K_U, lambda: GW_inv_sqrt @ core.U_actual)

    # J = GW_inv_sqrt @ delta_GW @ GW_inv_sqrt is genuinely indefinite
    # (delta_GW is a difference of two PSD matrices), so the factor trick
    # above doesn't apply -- hardened eigvalsh instead.
    J = GW_inv_sqrt @ core.delta_GW @ GW_inv_sqrt
    eig_J = _safe_sym_eigvalsh(J).flip(0)  # signed -- no clamp

    K_V_rates = eig_KV.sqrt()
    K_U_rates = eig_KU.sqrt()
    K_U_log_rate_spread = _log_rate_spread(K_U_rates)

    # Bounded distributions/fractions (each entry, or each signed part, is
    # <= the sum of all non-negative magnitudes) -- a tiny absolute floor
    # suffices, same reasoning as _effective_rank's probabilities.
    eig_tiny = torch.finfo(eig_KV.dtype).tiny
    norm_KV = eig_KV / eig_KV.sum().clamp_min(eig_tiny)
    norm_KU = eig_KU / eig_KU.sum().clamp_min(eig_tiny)
    relative_spectrum_l1_distance = (norm_KV - norm_KU).abs().sum()

    J_abs_sum = eig_J.abs().sum().clamp_min(torch.finfo(eig_J.dtype).tiny)
    J_positive_fraction = eig_J.clamp_min(0).sum() / J_abs_sum
    J_negative_fraction = (-eig_J.clamp_max(0)).sum() / J_abs_sum

    Q_WV = GW_inv_sqrt @ (core.W_before @ core.V_raw.T) @ GV_inv_sqrt
    Q_WU = GW_inv_sqrt @ (core.W_before @ core.U_actual.T) @ GU_inv_sqrt
    Q_VA = GV_inv_sqrt @ (core.V_raw @ core.A_actual.T) @ GU_inv_sqrt

    # All six singular-value sets below are [m, m] and the same shape, so they
    # go through ONE batched float64-Gram decomposition instead of six separate
    # `svdvals` calls. That was 288 ms of level 3's 799 ms at [768,2048] --
    # measured 207 ms -> 28 ms (7.4x) and 6800x more accurate (max relative
    # error 5.9e-8 vs 4.05e-4 against a float64 SVD reference).
    #
    # Batching `svdvals` itself buys nothing (measured 1.00x) -- cuSOLVER
    # serialises it internally, the same reason batching the expert norms was
    # a no-op in pass 1. The win is the float64 Gram, which turns each SVD
    # into an eigvalsh of an [m, m] matrix that DOES batch.
    _sv = _gram_spectrum(
        torch.stack([core.C_WV, core.C_WU, core.C_VA, Q_WV, Q_WU, Q_VA])
    )
    c_wv_sv, c_wu_sv, c_va_sv, q_wv_sv, q_wu_sv, q_va_sv = _sv.unbind(0)

    return {
        "K_V_eigenvalues": eig_KV,
        "K_V_rates": K_V_rates,
        "K_U_eigenvalues": eig_KU,
        "K_U_rates": K_U_rates,
        "J_eigenvalues": eig_J,
        # C_WV/C_WU/C_VA are plain cross-correlation singular values, NOT
        # bounded by 1 (e.g. near-duplicate rows can push these up toward
        # ~m) -- no clamp.
        "C_WV_singular_values": c_wv_sv,
        "C_WU_singular_values": c_wu_sv,
        "C_VA_singular_values": c_va_sv,
        # Q_WV/Q_WU/Q_VA are canonical correlations (classic whitened
        # cross-Gram CCA construction) -- mathematically guaranteed in
        # [0, 1] by Cauchy-Schwarz, unlike the plain singular values above.
        # Any value outside that range is floating-point noise from the
        # whitening transforms (GW_inv_sqrt/GV_inv_sqrt/GU_inv_sqrt), not
        # real signal -- clamp to enforce the known bound, same rationale
        # as clamping PSD eigenvalues to >= 0 elsewhere in this file.
        "Q_WV_canonical_correlations": q_wv_sv.clamp(0, 1),
        "Q_WU_canonical_correlations": q_wu_sv.clamp(0, 1),
        "Q_VA_canonical_correlations": q_va_sv.clamp(0, 1),
        "K_U_log_rate_spread": K_U_log_rate_spread,
        "relative_spectrum_l1_distance": relative_spectrum_l1_distance,
        "J_positive_fraction": J_positive_fraction,
        "J_negative_fraction": J_negative_fraction,
    }


GRAM_SCALAR_NAMES_BY_LEVEL: dict[int, list[str]] = {
    1: [
        "V_offdiag_mean_abs",
        "V_offdiag_rms",
        "V_offdiag_max_abs",
        "U_offdiag_mean_abs",
        "U_offdiag_rms",
        "U_offdiag_max_abs",
        "U_relative_step_fro",
        "Wm_offdiag_mean_abs",
        "Wm_offdiag_rms",
        "Wm_offdiag_max_abs",
        "Wp_offdiag_mean_abs",
        "Wp_offdiag_rms",
        "Wp_offdiag_max_abs",
        "update_to_momentum_isotropy_ratio",
        "gram_geometry_alignment",
        "global_direction_alignment",
        "VA_row_top1_identity",
        "VA_column_top1_identity",
        "VA_diagonal_energy_fraction",
        "WV_row_top1_identity",
        "WV_column_top1_identity",
        "WV_diagonal_energy_fraction",
        "WU_row_top1_identity",
        "WU_column_top1_identity",
        "WU_diagonal_energy_fraction",
        "gram_change_relative",
        "correlation_change_per_row",
        "relational_change_fraction",
        "gram_identity_residual",
    ],
}
GRAM_SCALAR_NAMES_BY_LEVEL[2] = (
    GRAM_SCALAR_NAMES_BY_LEVEL[1]
    + [
        f"{prefix}_{field}"
        for prefix in ("G_Wm", "G_Wp", "G_V", "G_U", "C_Wm", "C_Wp", "C_V", "C_U")
        for field in (
            "effective_rank",
            "largest",
            "smallest",
            "condition_regularized",
            "topk_energy_fraction",
        )
    ]
    + [
        "C_Wm_deviation_from_identity",
        "C_Wp_deviation_from_identity",
        "C_V_deviation_from_identity",
        "C_U_deviation_from_identity",
        "overlap_Wm_V",
        "overlap_Wm_U",
        "overlap_V_U",
        "overlap_Wm_Wp",
        "G_W_effective_rank_delta",
    ]
)
GRAM_SCALAR_NAMES_BY_LEVEL[3] = GRAM_SCALAR_NAMES_BY_LEVEL[2] + [
    "K_U_log_rate_spread",
    "relative_spectrum_l1_distance",
    "J_positive_fraction",
    "J_negative_fraction",
]

GRAM_VECTOR_NAMES_BY_LEVEL: dict[int, list[str]] = {
    1: [
        "V_R_raw",
        "V_R_cos",
        "U_R_raw",
        "U_R_cos",
        "Wm_R_raw",
        "Wm_R_cos",
        "Wp_R_raw",
        "Wp_R_cos",
        "VA_diagonal",
        "VA_row_specificity",
        "WV_diagonal",
        "WV_row_specificity",
        "WU_diagonal",
        "WU_row_specificity",
    ],
}
GRAM_VECTOR_NAMES_BY_LEVEL[2] = GRAM_VECTOR_NAMES_BY_LEVEL[1] + [
    "G_Wm_eigenvalues",
    "G_Wp_eigenvalues",
    "G_V_eigenvalues",
    "G_U_eigenvalues",
    "C_Wm_eigenvalues",
    "C_Wp_eigenvalues",
    "C_V_eigenvalues",
    "C_U_eigenvalues",
    "energy_V_in_Wm_basis",
    "energy_U_in_Wm_basis",
    "energy_V_in_Wm_basis_distribution",
    "energy_U_in_Wm_basis_distribution",
    "G_W_rankwise_eigenvalue_delta",
    "G_W_change_eigenvalues",
]
GRAM_VECTOR_NAMES_BY_LEVEL[3] = GRAM_VECTOR_NAMES_BY_LEVEL[2] + [
    "K_V_eigenvalues",
    "K_V_rates",
    "K_U_eigenvalues",
    "K_U_rates",
    "J_eigenvalues",
    "C_WV_singular_values",
    "C_WU_singular_values",
    "C_VA_singular_values",
    "Q_WV_canonical_correlations",
    "Q_WU_canonical_correlations",
    "Q_VA_canonical_correlations",
]


def gram_scalar_names(level: int) -> list[str]:
    return GRAM_SCALAR_NAMES_BY_LEVEL.get(level, [])


def gram_vector_names(level: int) -> list[str]:
    return GRAM_VECTOR_NAMES_BY_LEVEL.get(level, [])


@torch.no_grad()
def calculate_gram_metrics(
    W_before: torch.Tensor,
    V_raw: torch.Tensor,
    W_after: torch.Tensor,
    level: int = 0,
    eps: float = _DEFAULT_GRAM_EPS,
    topk: int = _DEFAULT_GRAM_TOPK,
) -> dict[str, torch.Tensor]:
    """
    Cheap no-op ({}) for level <= 0 -- the single early-return point; every
    disco.py call site stays unconditional (see module docstring). `V_raw`
    should be the raw effective grad/momentum (whatever's fed into
    AbstractDiSCO.lmo()); `W_after` should be the post-update weight
    (disco.py passes `pseudo_w`) -- see readme.md.

    Mirrors norm_helper.calculate_norm's unwrap contract for all three
    tensors (Parameter/DTensor -> local tensor, 1-D -> diag_embed), then
    upcasts each to float32 if in fp16/bf16 (Gram/eigh/svd are unreliable
    in half precision).

    `transpose` is accepted for call-site compatibility (disco.py's
    embedding path passes its name-based `need_T`) but is IGNORED here:
    orientation is instead decided unconditionally from shape -- rows are
    always transposed to be <= cols. This both (a) guarantees the reduced
    SVD used internally (see `_gram_eigh_from_factor`) always spans the
    complete eigenspace, and (b) fixes the large-vocab OOM that `need_T`'s
    name-based matching missed for `output`/lm_head (same pathological
    shape as `tok_embeddings`, different param name). This is a deliberate
    semantic choice, not just a numerical nicety: any parameter with
    `D_out > D_in` (e.g. an FFN up-projection) now tracks input-feature-wise
    dynamics instead of output-channel-wise -- see readme.md.

    NaN/Inf is replaced element-wise with 0 before any computation -- this
    is a best-effort diagnostic feature and must never be able to crash
    training. `W_before`/`W_after` are sanitized as a PAIR (zeroed together
    wherever EITHER is non-finite at a position), not independently: since
    `U_actual = W_after - W_before`, independently zeroing just the
    non-finite side would fabricate a fake update at that position (e.g.
    NaN-before + finite-after would read as "jumped from 0 to
    W_after[i,j]", an update that never happened) rather than correctly
    recording "no valid update data here". `V_raw` has no such pairing
    concern (nothing downstream derives a delta from it against another
    tensor the same way) and is sanitized independently.

    Returns a dict whose key set is a deterministic function of `level`
    alone (gram_scalar_names(level) + gram_vector_names(level)),
    independent of parameter shape -- disco.py's DDP/FSDP/experts packing
    code relies on this fixed arity. Degenerate shapes (m < 2) return {}
    rather than ill-defined values, same as level <= 0.
    """
    if level <= 0:
        return {}
    W_before = _prep(W_before)
    V_raw = _prep(V_raw)
    W_after = _prep(W_after)
    if (
        W_before.ndim < 2
        or V_raw.ndim < 2
        or W_after.ndim < 2
        or W_before.shape != V_raw.shape
        or W_before.shape != W_after.shape
        or W_before.shape[0] < 2
    ):
        return {}
    if W_before.dtype in (torch.float16, torch.bfloat16):
        W_before = W_before.float()
    if V_raw.dtype in (torch.float16, torch.bfloat16):
        V_raw = V_raw.float()
    if W_after.dtype in (torch.float16, torch.bfloat16):
        W_after = W_after.float()

    weight_pair_valid = torch.isfinite(W_before) & torch.isfinite(W_after)
    W_before = torch.where(weight_pair_valid, W_before, torch.zeros_like(W_before))
    W_after = torch.where(weight_pair_valid, W_after, torch.zeros_like(W_after))

    V_valid = torch.isfinite(V_raw)
    V_raw = torch.where(V_valid, V_raw, torch.zeros_like(V_raw))

    if gram_matrix_is_transposed(tuple(W_before.shape)):
        W_before = W_before.transpose(0, 1)
        V_raw = V_raw.transpose(0, 1)
        W_after = W_after.transpose(0, 1)

    core = _build_gram_core(W_before, V_raw, W_after)
    out: dict[str, torch.Tensor] = dict(_level1_metrics(core))
    extras = None
    if level >= 2:
        lvl2, extras = _level2_metrics(core, topk)
        out.update(lvl2)
    if level >= 3:
        out.update(_level3_metrics(core, eps, extras))
    return out
