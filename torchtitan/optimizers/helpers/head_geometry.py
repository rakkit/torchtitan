# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LM-head update geometry: eta, rho_RI, gamma_H.

The question: is the head's *functional* step a better coordinate than the raw
LR? For an update direction `U` (unit in its own norm) and head input `h`,

    rho_RI = ||[U h]||_q-inf / ( ||U||_rms->inf * ||h||_rms )
    gamma_H = ||U||_rms->q-inf / ||U||_rms->inf

where `[z]` is the softmax-quotient class, `||[z]||_q-inf = (max z - min z)/2`,
the norm the softmax actually sees (adding a constant to every logit is a no-op).

These two factor the thing we care about:

    rho_RI = gamma_H * a_H

with `a_H` the update-activation alignment. `gamma_H` answers "is rms->inf a loose
bound on the head's true norm?" and `a_H` answers "is the update misaligned with
the activations?". The realised logit step is `eps_z = eta * rho_RI`; both it and
`a_H` are derived offline, so only the primitives are logged.

Geometry, for reference:

    ||[U h]||_q-inf = 1/2 * max_{i,j} (u_i - u_j)^T h
    => ||U||_rms->q-inf = (sqrt(d)/2) * D(U),  D(U) = max_{i,j} ||u_i - u_j||_2
       ||U||_rms->inf   = sqrt(d) * max_i ||u_i||_2
       D(U) <= 2 max_i ||u_i||  =>  gamma_H <= 1

The sqrt(d) cancels in gamma_H, leaving pure row geometry:
`gamma_H = D(U) / (2 max_i ||u_i||)`.

## Why each probe is itself a gamma_H bound, and what that costs us

`D(U) = max_{||w||=1} (max_i u_i^T w - min_i u_i^T w)`, so *any* unit direction
gives a lower bound -- including the probe activations, which are already being
pushed through the head for rho_RI. Writing that out for probe `p`, with
`||h_p||_2 = sqrt(d) * h_rms_p`:

    spread of U along h_p/||h_p||  =  2 * qsup_p / ||h_p||_2
    => its gamma_H bound           =  qsup_p / (max_row * ||h_p||_2)  =  rho_p

**The probe's contribution to gamma_H is exactly rho_p.** So folding probes in is
`max(gamma_sweep, rho.amax())` -- no second [V,P] matmul, which is what an earlier
version of this file did, at 211 GFLOP and 412 MB at V=201088.

It also makes the consequence unmissable: `gamma_H >= max_p rho_p` *by
construction*, so `a_H_max = rho.amax() / gamma_H <= 1` is an identity, not a
test. It cannot detect an under-converged diameter. `gamma_H_sweep` (probes
excluded) is reported alongside precisely so `a_H_sweep_max > 1` remains a check
that can actually fire.

Everything here is a pure function of `(h, U, eta)`, so it is testable without an
optimizer or a training loop, and it is deterministic given `seed`.
"""

from __future__ import annotations

import torch

__all__ = [
    "head_geometry",
    "diameter_lower_bound",
    "quotient_sup_norm",
    "product_frobenius_sq",
    "HEAD_METRIC_NAMES",
]

# `head_geometry`'s keys, in its return order. The optimizer packs them into a
# fixed-size vector for the collective that carries them to the storing rank,
# so this tuple and the dict at the end of `head_geometry` must stay in step.
HEAD_METRIC_NAMES: tuple[str, ...] = (
    "eta",
    "rho_mean",
    "rho_rms",
    "rho_max",
    "gamma_H",
    "gamma_H_sweep",
    "a_H_max",
    "a_H_sweep_max",
    "U_rms_to_inf",
    "h_rms_mean",
    "lar_ratio_u",
    "lar_u",
    "lar_ratio_w",
    "lar_w",
)

_TINY = 1e-30


@torch.no_grad()
def quotient_sup_norm(
    h: torch.Tensor,
    U: torch.Tensor,
    vocab_chunk: int,
    *,
    with_frob_sq: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-probe `||[U h]||_q-inf`, streaming over the vocabulary.

    `z = h @ U.T` is `[P, V]`; at V = 201088 and P = 512 that is 412 MB, against
    a training run that already reserves ~94% of the card. Chunking the vocab
    into a handful of pieces keeps it around 100 MB. The loop is short (a few
    iterations), so it unrolls cleanly if this is ever compiled -- the thing to
    avoid would be a ~100-iteration loop.

    `with_frob_sq` also accumulates `||U h^T||_F^2` (for LAR) from the very same
    `z` blocks. That is the cheap way to get it: no second matmul, no second
    pass, just one more reduction over a block already resident -- measured
    +0.10 ms at V=50272 d=512 and +0.40 ms at V=201088 d=2048.

    Returns `[P]`, or `([P], scalar)` when `with_frob_sq`.
    """
    P = h.shape[0]
    hi = torch.full((P,), float("-inf"), device=h.device, dtype=torch.float32)
    lo = torch.full((P,), float("inf"), device=h.device, dtype=torch.float32)
    frob_sq = torch.zeros((), device=h.device, dtype=torch.float32)
    for start in range(0, U.shape[0], vocab_chunk):
        z = h @ U[start : start + vocab_chunk].T  # [P, |C|], transient
        hi = torch.maximum(hi, z.amax(dim=-1))
        lo = torch.minimum(lo, z.amin(dim=-1))
        if with_frob_sq:
            # vector_norm is a fused reduction; `z.pow(2).sum()` would allocate
            # a second [P, |C|] block (103 MB at P=512, chunk=50272) first.
            frob_sq += torch.linalg.vector_norm(z).square()
        del z
    qsup = (hi - lo) / 2.0
    return (qsup, frob_sq) if with_frob_sq else qsup


@torch.no_grad()
def product_frobenius_sq(
    h: torch.Tensor, M: torch.Tensor, vocab_chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(||M h^T||_F^2, ||M||_F^2)` for a `[V, d]` matrix `M` and `[P, d]` probes.

    Two routes, dispatched on shape, because the flop counts cross over:

        direct  stream [P, |C|] blocks      P*V*d flops,  O(P * chunk) memory
        gram    <h^T h, M^T M>              V*d^2 flops,  O(d^2) memory

    so gram wins iff `d < P`. Measured (P = 512): 2.58x faster at d=256, 1.26x
    at d=512, 0.27x at d=2048 -- i.e. the rule is worth following in both
    directions. Gram also holds ~6x less memory, and `tr(M^T M) = ||M||_F^2`
    falls out of the same contraction for free.

    Both routes agree with a float64 reference to ~1e-8 relative.
    """
    P, d = h.shape
    if d < P:
        G = M.t() @ M  # [d, d]
        return ((h.t() @ h) * G).sum(), G.diagonal().sum()
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    for start in range(0, M.shape[0], vocab_chunk):
        z = h @ M[start : start + vocab_chunk].T
        total += torch.linalg.vector_norm(z).square()  # fused; no [P,|C|] copy
        del z
    return total, torch.linalg.vector_norm(M).square()


@torch.no_grad()
def diameter_lower_bound(
    U: torch.Tensor,
    iters: int,
    *,
    row_sq: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Lower bound on `D(U) = max_{i,j} ||u_i - u_j||_2`, from `U` alone.

    Exact is O(V^2 d) -- impossible at V = 201088. Uses
    `D(U) = max_{||w||=1} (max_i u_i^T w - min_i u_i^T w)`, so any direction
    gives a lower bound: a principal-direction probe, then a farthest-point
    sweep (pick a row, jump to the row farthest from it, repeat).

    Probe activations are deliberately NOT accepted here. They are a lower bound
    too, but folding them in is free at the call site (see the module docstring)
    and doing it here would make this function unable to answer "how good is the
    sweep on its own?", which is the only remaining check on `a_H`.

    `iters` is a fixed count, deliberately: a convergence test would be
    data-dependent control flow and would break `fullgraph`. It sets both the
    power-iteration count and the number of sweep restarts.

    Memory is O(V + d) beyond `U` itself: the centring and the distances are
    both folded into matvecs rather than materialising `[V, d]` temporaries.

    **Measured accuracy** against exact `torch.cdist` on random unit-row matrices
    (V up to 20000, d up to 512, iters=10): recovers **~0.98 of the true
    diameter**, stable to +/-0.003 across seeds. So `gamma_H` reads ~2% low and
    `a_H` ~2% high, systematically. That bias is stable, so trends in `gamma_H`
    over training are trustworthy; treat its absolute level as a slight
    under-estimate. Raising `iters` past ~20 does not help much -- the residual
    gap is the sweep missing the exact extremal pair, not slow convergence.
    """
    V, d = U.shape
    if row_sq is None:
        # vector_norm is a fused reduction; `U.pow(2).sum(-1)` would materialise
        # a whole [V, d] temporary first -- 1.6 GB at V=201088, d=2048.
        row_sq = torch.linalg.vector_norm(U, dim=-1).square()
    n = max(iters // 2, 1)

    # Top principal direction of the CENTRED rows -- the diameter-achieving pair
    # lies near the axis of greatest spread. Centring never materialises
    # `U - mean`: for mu = mean(U),
    #     centred @ w   == U @ w - (mu @ w)
    #     centred.T @ y == U.T @ y - mu * y.sum()
    # both exact, both O(V + d) beyond U.
    mu = U.mean(dim=0)
    w = torch.randn(d, device=U.device, dtype=U.dtype, generator=generator)
    w = w / w.norm().clamp_min(_TINY)
    for _ in range(n):
        y = U @ w - (mu @ w)
        w = U.t() @ y - mu * y.sum()
        w = w / w.norm().clamp_min(_TINY)
    # The constant `mu @ w` cancels in a max-minus-min, so project raw U.
    proj = U @ w
    best = proj.amax() - proj.amin()
    del proj, y

    # Farthest-point sweep, with RESTARTS. A single sweep settles into a 2-cycle
    # between two mutually-farthest points after ~2 steps, so iterating one chain
    # buys nothing past that -- measured 96.6% of exact at 10 chained steps, a
    # systematic 3.4% under-estimate that would bias gamma_H. Independent
    # restarts from random rows explore different far pairs instead.
    starts = torch.randint(0, V, (n,), device=U.device, generator=generator)
    for r in range(n):
        idx = starts[r].reshape(1)
        for _ in range(2):  # 2 steps is where a chain saturates
            # ||u_i - a||^2 = ||u_i||^2 - 2 u_i.a + ||a||^2, in O(V) memory
            # instead of a [V, d] difference. The expansion's cancellation only
            # bites for pairs that nearly coincide; we read off the FARTHEST
            # point, the opposite end, where it is well conditioned.
            anchor = U.index_select(0, idx).squeeze(0)  # [d]
            dist_sq = row_sq - 2.0 * (U @ anchor) + row_sq.index_select(0, idx)
            best = torch.maximum(best, dist_sq.amax().clamp_min(0).sqrt())
            idx = dist_sq.argmax().reshape(1)
    return best


@torch.no_grad()
def head_geometry(
    h: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    eta: float | torch.Tensor,
    vocab_chunk: int = 50272,
    diameter_iters: int = 10,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """The logged primitives plus their health checks.

    Args:
        h: `[P, d]` head inputs (post final norm) for P probe tokens.
        U: `[V, d]` unit update direction -- for DiSCO this is `lmo(g)`, i.e.
            the applied direction *before* the `-lr` scaling.
        W: `[V, d]` pre-update head weight, for LAR.
        eta: the head's learning rate, float or 0-dim tensor.
        seed: seeds a LOCAL generator for the diameter estimator. The default
            fixed value makes the estimator deterministic and gives it common
            random numbers across steps, so step-to-step moves in `gamma_H` are
            signal rather than estimator jitter (measured at 1.9% spread when
            the draws were allowed to vary). It is a local `torch.Generator` on
            purpose: drawing from the global stream inside `optimizer.step()`
            would let a diagnostic perturb training the moment any stochastic op
            (dropout, router noise, stochastic rounding) is enabled.

    Returns tensors (not floats): every value stays on device so the caller
    decides when to sync. A degenerate `U` (all rows zero) yields the 0 sentinel
    throughout, matching `radial_helper`'s convention rather than dividing by a
    clamp floor and logging an astronomical number.
    """
    if U.ndim != 2:
        raise ValueError(f"U must be [V, d]; got {tuple(U.shape)}")
    if h.ndim != 2 or h.shape[-1] != U.shape[-1]:
        raise ValueError(
            f"h must be [P, d] matching U's d={U.shape[-1]}; got {tuple(h.shape)}"
        )
    if W.shape != U.shape:
        raise ValueError(
            f"W must match U's shape {tuple(U.shape)}; got {tuple(W.shape)}"
        )

    if h.dtype != torch.float32:
        h = h.float()
    Uf = U if U.dtype == torch.float32 else U.float()
    d = Uf.shape[1]
    zero = torch.zeros((), device=Uf.device, dtype=torch.float32)

    # Fused reduction: `Uf.pow(2).sum(-1)` would allocate a full [V, d] first.
    row_norms = torch.linalg.vector_norm(Uf, dim=-1)
    row_sq = row_norms.square()
    max_row = row_norms.amax()
    valid = max_row > 0
    # ||U||_rms->inf = sqrt(d) * max_i ||u_i||, matching norm_helper.rms_to_inf_norm
    u_rms_to_inf = max_row * (d**0.5)
    # ||h||_rms per probe; exactly 1 under a parameter-free final RMSNorm, but
    # computed rather than assumed so a norm-type change shows up instead of
    # silently rescaling every metric.
    h_rms = h.pow(2).mean(dim=-1).sqrt()

    qsup, uh_frob_sq = quotient_sup_norm(h, Uf, vocab_chunk, with_frob_sq=True)
    rho = torch.where(
        valid,
        qsup / (u_rms_to_inf * h_rms).clamp_min(_TINY),
        torch.zeros_like(qsup),
    )
    rho_max = rho.amax()

    gen = torch.Generator(device=Uf.device)
    gen.manual_seed(int(seed))
    d_sweep = diameter_lower_bound(Uf, diameter_iters, row_sq=row_sq, generator=gen)
    gamma_sweep = torch.where(valid, d_sweep / (2.0 * max_row).clamp_min(_TINY), zero)
    # Each probe's own bound on gamma_H is exactly rho_p (module docstring), so
    # folding the probes in is a max over already-computed numbers. Note what
    # this makes explicit: gamma_H >= rho_max always, hence a_H_max <= 1 always.
    gamma_H = torch.maximum(gamma_sweep, rho_max)

    # ---- LAR (Shehper & Vaswani, arXiv:2605.28975) ------------------------
    #   LAR = log_n( ||M X||_RMS / (||M||_RMS ||X||_RMS) ),  X = h^T, n = d
    # with ||.||_RMS the ELEMENTWISE rms (||.||_F / sqrt(numel)) -- NOT this
    # codebase's rms->rms operator norm, which is a different quantity.
    #
    # Equivalent spectral form: LAR = 1 + (1/2) log_n(sum_i p_i q_i), with
    # p_i = s_i^2/sum_j s_j^2 and q_i = sum_x (v_i^T x)^2 / sum_x ||x||^2. The
    # equivalence is exact: sum_i p_i q_i = ||M X||_F^2 / (||M||_F^2 ||X||_F^2),
    # and converting F -> elementwise RMS for a [V, d] matrix against a [d, P]
    # batch contributes sqrt(V d) sqrt(d P) / sqrt(V P) = d, which is where the
    # `1 +` and the base come from. Note V and P both cancel: LAR does not care
    # that the head is extremely rectangular.
    #
    # The raw ratio is logged alongside because it is base-free. `n = d` varies
    # along a width sweep, so LAR values on such a sweep are in DIFFERENT bases
    # by construction; keeping the ratio means the convention can be revisited
    # offline without a rerun.
    n_t = torch.as_tensor(float(d), device=Uf.device)
    log_n = n_t.log()
    h_rms_all = h_rms.pow(2).mean().sqrt()  # ||h||_F / sqrt(P d), batch-wide

    def _lar(prod_frob_sq, m_frob_sq):
        num = (prod_frob_sq / float(Uf.shape[0] * h.shape[0])).clamp_min(0).sqrt()
        den = (m_frob_sq / float(Uf.shape[0] * d)).clamp_min(0).sqrt() * h_rms_all
        ratio = torch.where(den > 0, num / den.clamp_min(_TINY), zero)
        if d <= 1:  # log base 1 is undefined; the ratio is still meaningful
            return ratio, zero
        lar = torch.where(ratio > 0, ratio.clamp_min(_TINY).log() / log_n, zero)
        return ratio, lar

    lar_ratio_u, lar_u = _lar(uh_frob_sq, row_sq.sum())
    Wf = W if W.dtype == torch.float32 else W.float()
    wh_frob_sq, w_frob_sq = product_frobenius_sq(h, Wf, vocab_chunk)
    lar_ratio_w, lar_w = _lar(wh_frob_sq, w_frob_sq)

    eta_t = (
        eta
        if isinstance(eta, torch.Tensor)
        else torch.as_tensor(float(eta), device=Uf.device)
    )

    return {
        "eta": eta_t.to(device=Uf.device, dtype=torch.float32),
        "rho_mean": rho.mean(),
        "rho_rms": rho.pow(2).mean().sqrt(),
        "rho_max": rho_max,
        "gamma_H": gamma_H,
        # Probes excluded, so this one is NOT pinned above rho_max.
        "gamma_H_sweep": gamma_sweep,
        # <= 1 identically -- kept for continuity with runs logged before
        # gamma_H_sweep existed, not as a diagnostic.
        "a_H_max": torch.where(gamma_H > 0, rho_max / gamma_H.clamp_min(_TINY), zero),
        # The check that can fire: > 1 means the sweep under-converged, since
        # a_H <= 1 is a theorem about the true gamma_H.
        "a_H_sweep_max": torch.where(
            gamma_sweep > 0, rho_max / gamma_sweep.clamp_min(_TINY), zero
        ),
        "U_rms_to_inf": u_rms_to_inf,
        "h_rms_mean": h_rms.mean(),
        # LAR: the base-free ratio, and the paper's log_d of it.
        "lar_ratio_u": lar_ratio_u,
        "lar_u": lar_u,
        "lar_ratio_w": lar_ratio_w,
        "lar_w": lar_w,
    }
