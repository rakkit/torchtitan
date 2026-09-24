# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import repeat

import torch

from .floors import NORM_FLOOR
from .newton_schulz_triton import newton_schulz_triton, polar_express_triton


def zeropower_via_svd(G, **kwargs):
    original_dtype = G.dtype
    is_2d = G.dim() == 2
    if is_2d:
        G = G.unsqueeze(0)  # batch of 1

    assert G.dim() == 3, f"Expected 2D or 3D tensor, got {G.shape}"

    G32 = G.to(torch.float32)  # SVD does not support bfloat16 reliably
    rows, cols = G32.shape[-2], G32.shape[-1]
    transposed = rows > cols
    if transposed:
        G32 = G32.transpose(-2, -1)

    X_list = []
    for b in range(G32.shape[0]):
        U, S, V = G32[b].svd()  # per-matrix SVD (matches your API choice)
        # Pseudo-polar: a singular direction at the noise floor carries no
        # information and its singular vectors are arbitrary, so send it to 0
        # as the iterative backends do -- `U @ V.t()` would send it to 1.
        # Tolerance is LAPACK's rank rule, max(m, n) * eps * sigma_max.
        tol = S[..., :1] * (max(G32.shape[-2:]) * torch.finfo(torch.float32).eps)
        Xb = (U * (S > tol)) @ V.t()
        Xb = Xb * (torch.linalg.vector_norm(G32[b]) >= NORM_FLOOR)  # as the others
        X_list.append(Xb)

    X = torch.stack(X_list, dim=0)

    if transposed:
        X = X.transpose(-2, -1)
    if is_2d:
        X = X.squeeze(0)

    return X.to(original_dtype).contiguous()


# Polar Express
# Polar-express coefficients (https://arxiv.org/abs/2505.16932), with the
# 1/1.01 stabiliser applied to every triple except the limit. Factored out of
# `zeropower_via_polar_express` so `gram_newton_schulz.gram_polar_express`
# iterates on literally the same numbers -- the two backends should differ only
# in how the polynomial is evaluated, never in which polynomial it is.
_POLAR_EXPRESS_BASE = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.948690853482295, -2.908902115962949, 0.5518191394370137),
    (3.318419657370602, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.668903984574749, 0.4188073119525673),
    (1.891301407787398, -1.267995827194587, 0.3768040894852483),
    (1.875001480853448, -1.250001645399949, 0.3750001645474248),
    (1.875000000000000, -1.250000000000000, 0.375000000000000),  # limit
]
_POLAR_EXPRESS_STABILISED = [
    (a / 1.01, b / (1.01**3), c / (1.01**5))
    for (a, b, c) in _POLAR_EXPRESS_BASE[:-1]
] + [_POLAR_EXPRESS_BASE[-1]]


def polar_express_coefficients(steps: int) -> list[tuple[float, float, float]]:
    """First `steps` polar-express triples, repeating the limit triple if
    `steps` exceeds the table."""
    coeffs = _POLAR_EXPRESS_STABILISED + list(
        repeat(
            _POLAR_EXPRESS_STABILISED[-1],
            max(0, steps - len(_POLAR_EXPRESS_STABILISED)),
        )
    )
    return coeffs[:steps]


@torch.compile
def zeropower_via_polar_express(G, steps=5, eps=1e-7):
    # https://arxiv.org/abs/2505.16932
    # Support 2D (unsqueezed to batch=1) and 3D inputs.
    is_2d = G.dim() == 2
    if is_2d:
        G = G.unsqueeze(0)
    assert (
        G.dim() == 3
    ), f"Please make sure gradients are 2D or 3D tensors, got shape: {G.shape}"

    coeffs = polar_express_coefficients(steps)

    original_dtype = G.dtype
    X = G.bfloat16()

    # Transpose whole batch if rows > cols (same rule as your 2D version).
    rows, cols = X.shape[-2], X.shape[-1]
    transposed = False
    if rows > cols:
        X = X.transpose(1, 2)
        transposed = True

    # Per-matrix normalisation; matches your scalar normalisation when batch=1.
    norm = torch.linalg.norm(X, dim=(-2, -1), keepdim=True)
    X = X / torch.where(norm >= NORM_FLOOR, norm * 1.01 + eps, torch.inf)

    # Main loop (batched matmuls).
    for k in range(steps):
        a, b, c = coeffs[k]
        A = torch.bmm(X, X.transpose(1, 2))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)

    if transposed:
        X = X.transpose(1, 2)
    if is_2d:
        X = X.squeeze(0)

    return X.to(original_dtype)


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """

    assert (
        len(G.shape) == 2 or len(G.shape) == 3
    ), f"Please make sure gradients are 2D tensors to use NS, got shape: {G.shape}"

    is_2d = len(G.shape) == 2
    if is_2d:
        G = G.unsqueeze(0)

    assert (
        len(G.shape) == 3
    ), f"Please make sure gradients are 2D or 3D tensors, got shape: {G.shape}"

    a, b, c = (3.4445, -4.7750, 2.0315)
    original_dtype = G.dtype
    X = G.bfloat16()

    # Determine if we need to transpose the matrices based on their dimensions
    rows, cols = X.shape[-2], X.shape[-1]
    transposed = False
    if rows > cols:
        transposed = True
        X = X.transpose(1, 2)

    # Normalize each matrix in the batch individually
    norm = torch.linalg.norm(X, dim=(-2, -1), keepdim=True)
    X = X / torch.where(norm >= NORM_FLOOR, norm + eps, torch.inf)

    # Perform the iteration using batched matrix multiplication (torch.bmm)
    for _ in range(steps):
        # A = X @ X.T for a batch
        A = torch.bmm(X, X.transpose(1, 2))
        # B = b*A + c*A@A for a batch
        B = b * A + c * torch.bmm(A, A)
        # X = a*X + B@X for a batch
        X = a * X + torch.bmm(B, X)

    # Transpose back if we did it at the beginning
    if transposed:
        X = X.transpose(1, 2)

    # If the original input was a 2D tensor, squeeze the batch dimension out
    if is_2d:
        X = X.squeeze(0)

    return X.to(original_dtype)


zeropower_backends = dict(
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5,
    polar_express=zeropower_via_polar_express,
    newtonschulz5_triton=newton_schulz_triton,
    polar_express_triton=polar_express_triton,
    identity=lambda x, **kwargs: x,
)

# `gram_polar_express` adds itself to the dict at the end of its own module,
# which is the only order that works both ways round: it needs
# `polar_express_coefficients` from here, and whichever of the two is imported
# first leaves the other partially initialised. Imported here for that side
# effect, so the backend exists however the caller reached this module.
from . import gram_newton_schulz as _gram_newton_schulz  # noqa: E402,F401
