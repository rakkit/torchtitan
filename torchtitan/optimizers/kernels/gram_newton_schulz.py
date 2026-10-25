# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The Gram Newton-Schulz iteration below is adapted from
#   https://github.com/Dao-AILab/gram-newton-schulz  (MIT licensed)
#   Jack Zhang, Noah Amsel, Berlin Chen, Tri Dao,
#   "Gram Newton-Schulz: A Fast, Hardware-Aware Newton-Schulz Algorithm for Muon"
# Vendored rather than imported so training does not depend on a checkout
# outside this repository. Coefficients here are this repo's own polar-express
# set (see zeropower.zeropower_via_polar_express), not the upstream defaults,
# so this is a numerical near-match to the existing backend rather than a
# different iteration -- see the accuracy note below.

"""
Polar decomposition by Newton-Schulz iterated on the Gram matrix.

Mathematically the same iteration as `zeropower_via_polar_express`, but instead
of iterating on `X` (n x m) it iterates on `R = X X^T` (n x n, symmetric) and
applies the accumulated polynomial to `X` once at the end. For the shapes DiSCO
feeds the LMO that is a strict FLOP reduction, and the square symmetric products
map onto better GEMM kernels.

Register name `gram_polar_express` (a rule's `compute`; the opt_moe recipes
use it with 6 steps). Unlike norm_helper/radial_helper, this sits on the
*update path*: its output is the direction the optimizer applies, so switching
to or from it changes the training trajectory.

The numbers below are like-for-like at 8 iterations for every backend (they
were taken when `polar_express_triton` ignored `steps` and always ran 8).

Measured on GH200 / torch 2.12, CUDA-event timed, 20 warmup + 5 trials x 20
reps, at the two batched expert shapes DiSCO computes per optimizer step for
qwen30b-a3b at EP=64 (`[188,768,2048]` and `[94,2048,768]`):

    backend                          err vs UV^T    expert LMO/step
    polar_express_triton (default)      0.0064          24.31 ms
    this, float16 + quack kernels       0.0028          20.17 ms   (1.21x)
    this, float16 + torch backend       0.0027          27.14 ms   (slower)

So with the kernels this is ~2.3x more accurate and ~1.2x faster than
`polar_express_triton`; without quack it is slower.

Upstream's default coefficients use a safety factor of 1.05 where this repo uses
1.01; that difference, not the algorithm or the working precision, made
upstream's defaults look less accurate (0.0967) in a first comparison.
"""

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

from .floors import NORM_FLOOR, SUMSQ_FLOOR
from .zeropower import (
    _POLAR_EXPRESS_BASE,
    polar_express_coefficients,
    zeropower_backends,
)

# Restarting the accumulated polynomial partway through keeps the Gram
# iteration stable; upstream restarts once, before iteration 2.
_RESET_ITERATIONS = frozenset({2})

# The polar-express triples are evaluated at x / safety. 1.01: this repo's
# X-iteration value (bitwise today's kernel); upstream Gram-NS uses 1.05.
GRAM_SAFETY_DEFAULT = 1.01


def gram_coefficients(
    steps: int, safety: float = GRAM_SAFETY_DEFAULT
) -> list[tuple[float, float, float]]:
    """First `steps` triples as p(x / safety): (a / f, b / f^3, c / f^5), the
    limit triple unscaled and repeated."""
    if safety == GRAM_SAFETY_DEFAULT:
        return polar_express_coefficients(steps)
    f = safety
    table = [(a / f, b / f**3, c / f**5) for a, b, c in _POLAR_EXPRESS_BASE[:-1]]
    table.append(_POLAR_EXPRESS_BASE[-1])
    return (table + [table[-1]] * max(0, steps - len(table)))[:steps]


_QUACK = None


def _quack_ops():
    """Symmetric-GEMM kernels if available, else plain torch matmuls."""
    global _QUACK
    if _QUACK is None:
        try:
            from quack.gemm_interface import gemm, gemm_symmetric

            _QUACK = (
                gemm_symmetric,
                lambda A, B, C, alpha=1.0, beta=1.0: gemm_symmetric(
                    A, B, C=C, alpha=alpha, beta=beta
                ),
                lambda A, B: gemm(A, B, tuned=False),
            )
        except Exception as exc:
            _QUACK = False
            # Loud, once. Without the symmetric-GEMM kernels this backend is
            # SLOWER than the default `polar_express_triton` it would be
            # replacing (27.1 vs 24.3 ms/step measured), so silently degrading
            # to the torch path would quietly cost throughput for the whole
            # run. Selecting this backend without quack installed is a
            # misconfiguration, not a soft fallback.
            logger.warning(
                "[DiSCO] compute='gram_polar_express' selected but the "
                "quack symmetric-GEMM kernels are unavailable (%s). Falling back "
                "to plain torch matmuls, which is SLOWER than "
                "'polar_express_triton'. Install quack-kernels, or switch the "
                "compute back.",
                type(exc).__name__,
            )
    if _QUACK is False:
        return (
            lambda A, B: A @ B,
            lambda A, B, C, alpha=1.0, beta=1.0: torch.baddbmm(
                C, A, B, alpha=alpha, beta=beta
            ),
            lambda A, B: A @ B,
        )
    return _QUACK


@torch.no_grad()
def gram_polar_express(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    use_kernels: bool = True,
    work_dtype: torch.dtype = torch.float16,
    safety: float = GRAM_SAFETY_DEFAULT,
) -> Tensor:
    """Drop-in for `zeropower_via_polar_express` with the Gram iteration.

    Same signature as every other entry in `zeropower.zeropower_backends`:
    `(G, steps=..., eps=...)`, 2-D or 3-D input, output in `G`'s dtype.

    `work_dtype` defaults to float16, not the bfloat16 that
    `zeropower_via_polar_express` uses. That is deliberate and measured: the
    Gram iteration squares its operand every step, so it is more sensitive to
    mantissa width, and float16's 3 extra mantissa bits are worth far more than
    bfloat16's extra range here. At 8 iterations on `[188,768,2048]`, error
    against the exact polar factor is 0.0132 in bfloat16, 0.0027 in float16 and
    0.0017 in float32 (float32 being ~8x slower and not worth it).
    """
    # The quack symmetric-GEMM kernels accept only the half dtypes they were
    # built for; handing them float32 segfaults the process rather than raising,
    # so the dtype is checked here instead of being discovered at runtime.
    if work_dtype not in (torch.bfloat16, torch.float16):
        use_kernels = False
    sym_mm, sym_baddbmm, mm = (
        _quack_ops()
        if use_kernels
        else (
            lambda A, B: A @ B,
            lambda A, B, C, alpha=1.0, beta=1.0: torch.baddbmm(
                C, A, B, alpha=alpha, beta=beta
            ),
            lambda A, B: A @ B,
        )
    )
    coeffs = gram_coefficients(steps, safety)

    is_2d = G.dim() == 2
    if is_2d:
        G = G.unsqueeze(0)
    assert G.dim() == 3, f"expected a 2-D or 3-D tensor, got {tuple(G.shape)}"
    original_dtype = G.dtype

    # Iterate on whichever side gives the smaller Gram matrix.
    tall = G.size(-2) > G.size(-1)
    X = G.to(torch.float32)
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / torch.where(norm >= NORM_FLOOR, norm * 1.01 + eps, torch.inf)
    X = X.to(work_dtype)

    R = sym_mm(X.mT, X) if tall else sym_mm(X, X.mT)
    I = (
        torch.eye(R.size(-1), device=X.device, dtype=X.dtype)
        .unsqueeze(0)
        .expand(R.size(0), -1, -1)
        .contiguous()
    )
    Q = None
    n = len(coeffs)
    for i, (a, b, c) in enumerate(coeffs):
        if i in _RESET_ITERATIONS and i != 0:
            X = mm(X, Q) if tall else mm(Q, X)
            R = sym_mm(X.mT, X) if tall else sym_mm(X, X.mT)
            Q = None
        Z = sym_baddbmm(R, R, C=R, alpha=c, beta=b)
        if i == 0 or i in _RESET_ITERATIONS:
            Q = Z + a * I
        else:
            Q = sym_baddbmm(Q, Z, C=Q, beta=a)
        if i < n - 1 and (i + 1) not in _RESET_ITERATIONS:
            RZ = sym_baddbmm(R, Z, C=R, beta=a)
            R = sym_baddbmm(Z, RZ, C=RZ, beta=a)

    X = mm(X, Q) if tall else mm(Q, X)
    if is_2d:
        X = X.squeeze(0)
    return X.to(original_dtype)


@torch.no_grad()
def gram_polar_factor(
    R: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    work_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """`Q` with `X @ Q = polar(X)` from `R = X^T X` alone (`[n, n]` or
    `[B, n, n]`), so a row-sharded tall `X` needs one all-reduce of its partial
    Grams and no all-to-all: each rank then applies `rows @ Q`. Same recurrence
    as `gram_polar_express`; a reset re-forms the Gram as `Q^T R Q` instead of
    from `X`. Returns `Q` in fp32."""
    coeffs = polar_express_coefficients(steps)
    is_2d = R.dim() == 2
    if is_2d:
        R = R.unsqueeze(0)
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0)
    scale = torch.where(trace >= SUMSQ_FLOOR, trace.sqrt() * 1.01 + eps, torch.inf)
    scale2 = scale * scale
    R0 = R.to(torch.float32) / scale2[:, None, None]  # fp32: the resets re-form from it
    R = R0.to(work_dtype)
    I = torch.eye(R.size(-1), device=R.device, dtype=R.dtype).expand_as(R).contiguous()
    Q = None
    Qtot = None  # product of the Q's up to the last reset
    n = len(coeffs)
    for i, (a, b, c) in enumerate(coeffs):
        if i in _RESET_ITERATIONS and i != 0:
            Qtot = Q if Qtot is None else Qtot @ Q
            # From the fp32 Gram: an fp16 one leaves negative eigenvalues in a
            # rank-deficient unit's null space, which the polynomial blows up.
            Qf = Qtot.to(torch.float32)
            R = (Qf.mT @ R0 @ Qf).to(work_dtype)
            Q = None
        Z = torch.baddbmm(R, R, R, alpha=c, beta=b)
        if i == 0 or i in _RESET_ITERATIONS:
            Q = Z + a * I
        else:
            Q = torch.baddbmm(Q, Q, Z, beta=a)
        if i < n - 1 and (i + 1) not in _RESET_ITERATIONS:
            RZ = torch.baddbmm(R, R, Z, beta=a)
            R = torch.baddbmm(RZ, Z, RZ, beta=a)
    Q = Q if Qtot is None else Qtot @ Q
    Q = Q.to(torch.float32) / scale[:, None, None]
    return Q.squeeze(0) if is_2d else Q


# Registered from here, not from `zeropower`: this module imports that one, so
# on the `gram_newton_schulz`-first path `zeropower` would be reaching into a
# module that has not defined `gram_polar_express` yet. The dict exists by the
# time this line runs either way. It is on the *update path*, so selecting it
# changes the training trajectory -- see this module's docstring.
zeropower_backends["gram_polar_express"] = gram_polar_express
