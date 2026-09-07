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
# set (see muon_utils.zeropower_via_polar_express), not the upstream defaults,
# so this is a numerical near-match to the existing backend rather than a
# different iteration -- see the accuracy note below.

"""
Polar decomposition by Newton-Schulz iterated on the Gram matrix.

Mathematically the same iteration as `zeropower_via_polar_express`, but instead
of iterating on `X` (n x m) it iterates on `R = X X^T` (n x n, symmetric) and
applies the accumulated polynomial to `X` once at the end. For the shapes DiSCO
feeds the LMO that is a strict FLOP reduction, and the square symmetric products
map onto better GEMM kernels.

**Opt in only.** Register name `gram_polar_express`; the default
`zeropower_backend` is unchanged. Unlike everything in norm_helper/radial_helper,
this sits on the *update path* -- its output is the direction the optimizer
applies -- so switching it changes the training trajectory and is a deliberate
choice, not a free speedup.

A note on iteration counts before the numbers: `polar_express_triton` iterates
its whole coefficient table and so runs **8** Newton-Schulz iterations
regardless of the `steps` argument, while `zeropower_via_polar_express` honours
`steps` and runs 5 by default. Any comparison between them at "steps=5" is
therefore 8 iterations against 5. Everything below is like-for-like at 8.

Measured on GH200 / torch 2.12, CUDA-event timed, 20 warmup + 5 trials x 20
reps, at the two batched shapes step_experts produces per optimizer step for
qwen30b-a3b at EP=64 (`[188,768,2048]` and `[94,2048,768]`):

    backend                          err vs UV^T    expert LMO/step
    polar_express_triton (default)      0.0064          24.31 ms
    this, float16 + quack kernels       0.0028          20.17 ms   (1.21x)
    this, float16 + torch backend       0.0027          27.14 ms   (slower)

So with the kernels this is simultaneously ~2.3x more accurate and ~1.2x
faster than the current default. Without quack it is slower than the default
and not worth enabling.

i.e. indistinguishable, and marginally closer. The two outputs differ from each
other by ~2%, which is the expected spread between two ~9%-accurate
approximations of the same quantity. Note the upstream default coefficients use
a safety factor of 1.05 where this repo uses 1.01; that difference -- not the
algorithm and not the working precision -- is what made upstream's defaults look
less accurate (0.0967) in a first comparison.
"""

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

from .muon_utils import polar_express_coefficients

# Restarting the accumulated polynomial partway through keeps the Gram
# iteration stable; upstream restarts once, before iteration 2.
_RESET_ITERATIONS = frozenset({2})

_QUACK = None


def _quack_ops():
    """Symmetric-GEMM kernels if available, else plain torch matmuls."""
    global _QUACK
    if _QUACK is None:
        try:
            from quack.gemm_interface import gemm, gemm_add, gemm_symmetric

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
                "[DiSCO] zeropower_backend='gram_polar_express' selected but the "
                "quack symmetric-GEMM kernels are unavailable (%s). Falling back "
                "to plain torch matmuls, which is SLOWER than "
                "'polar_express_triton'. Install quack-kernels, or switch the "
                "backend back.",
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
) -> Tensor:
    """Drop-in for `zeropower_via_polar_express` with the Gram iteration.

    Same signature as every other entry in `muon_utils.zeropower_backends`:
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
    coeffs = polar_express_coefficients(steps)

    is_2d = G.dim() == 2
    if is_2d:
        G = G.unsqueeze(0)
    assert G.dim() == 3, f"expected a 2-D or 3-D tensor, got {tuple(G.shape)}"
    original_dtype = G.dtype

    # Iterate on whichever side gives the smaller Gram matrix.
    tall = G.size(-2) > G.size(-1)
    X = G.to(torch.float32)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + eps)
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
