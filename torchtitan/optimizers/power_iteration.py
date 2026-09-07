# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Exact leading singular triple (sigma_max, u1, v1) of a batch of matrices.

Why this module exists: norm_helper.calculate_norm is only ever called on the
update `-lr*u` and on the post-update weight `pseudo_w`, so sigma_max is free
for those two. It is NOT called on the *pre-update* weight, and radial_helper's
radiality metrics need the leading singular *vectors* of that weight, not just
its largest singular value.

Why it is not a power iteration, despite the name. It started as one, on the
assumption that a full decomposition of W_before would be prohibitive. That
assumption was overturned by the norm_helper work: computing a spectrum via the
float64 Gram (`sqrt(eigvalsh(W W^T))`) turned out to be 5-8x *faster* than
`torch.linalg.svdvals` and more accurate. Once the Gram is being formed anyway,
asking for `eigh` instead of `eigvalsh` returns the eigenvectors of that same
matrix for 5-19% more, and the leading one *is* u1, with `v1 = W^T u1 / sigma`
following in a single matvec.

So this is exact, and it is cheaper than the iteration it replaces:

    method                          sigma error       notes
    warm-started power iteration    1e-2 .. 1e-3      cold start; needs
                                                      per-parameter warm-start
                                                      state carried across steps
    float64 Gram + eigh             5e-14 .. 6e-13    no state, no iteration,
                                                      no convergence question

and batched -- the shape that matters, since the expert path decomposes
hundreds of matrices at once -- `eigh` costs 1.19x `eigvalsh` on 1344
`[384,1024]` (+47 ms in total) and 1.11x on 282 `[768,2048]` (+32 ms).

The iteration-shaped arguments (`v0`, `n_iter`, `n_iter_cold`, `tol`,
`early_exit`) are accepted and ignored, so callers that were written against
the iterative contract keep working; `v0` in particular is harmless to keep
passing, there is simply no warm start to carry any more.
"""

import torch

from .norm_helper import gram_top_singular_pair

__all__ = [
    "top_singular_pair",
    "spectral_norm",
    "IS_STUB",
    "DEFAULT_N_ITER",
    "DEFAULT_N_ITER_COLD",
]

# No longer a stub: the values below are exact. Kept as a public flag because
# disco.py branches on it to decide whether the spectral radial metrics are
# meaningful enough to compute.
IS_STUB: bool = False

# Retained for signature compatibility; unused.
DEFAULT_N_ITER = 2
DEFAULT_N_ITER_COLD = 24


@torch.no_grad()
def top_singular_pair(
    W: torch.Tensor,
    v0: torch.Tensor | None = None,
    n_iter: int = DEFAULT_N_ITER,
    n_iter_cold: int = DEFAULT_N_ITER_COLD,
    tol: float = 1e-7,
    early_exit: bool = False,
    backend: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Leading singular triple of `W`.

    Args:
        W: `[..., m, n]`. Arbitrary leading batch dimensions -- a bare `[m, n]`
            matrix, a `[E, d_out, d_in]` expert stack, or a stack of
            same-shaped dense params all work with the same call.
        v0, n_iter, n_iter_cold, tol, early_exit: accepted and ignored, see the
            module docstring. There is no iteration and no warm start.
        backend: reserved for a fused implementation.

    Returns:
        `(sigma, u1, v1)` with shapes `[...]`, `[..., m]`, `[..., n]`, float32,
        satisfying `W v1 = sigma u1`. `u1`/`v1` are unit-norm. For an all-zero
        or non-finite slice, `sigma` is 0 with deterministic unit vectors, so
        callers that divide by sigma stay finite.
    """
    if backend != "torch":
        impl = _BACKENDS.get(backend)
        if impl is None:
            raise ValueError(
                f"unknown backend {backend!r}; available: {sorted(_BACKENDS)}"
            )
        return impl(W)

    if W.ndim < 2:
        raise ValueError(
            f"top_singular_pair needs a matrix, got shape {tuple(W.shape)}"
        )

    m, n = W.shape[-2], W.shape[-1]
    finite = torch.isfinite(W).all(dim=-1).all(dim=-1)
    Wc = torch.where(finite[..., None, None], W, torch.zeros_like(W))
    sigma, u1, v1 = gram_top_singular_pair(Wc)

    tiny = torch.finfo(torch.float32).tiny
    # `eigh` on an all-zero Gram returns an arbitrary orthonormal basis; pin the
    # degenerate case to a deterministic sentinel instead, and renormalise the
    # rest (v1 comes out of a division and is only unit-norm up to round-off).
    degenerate = ~finite | (Wc.abs().amax(dim=(-2, -1)) <= tiny)
    u1 = u1 / u1.norm(dim=-1, keepdim=True).clamp_min(tiny)
    v1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(tiny)
    # Unconditional `torch.where`, NOT `if bool(degenerate.any())`: reading a
    # device tensor into a Python bool forces a device-to-host sync on every
    # call, which is exactly the stall disco.py goes to some length to avoid
    # elsewhere (see `_materialize_gathered`). The two zero-vectors are tiny
    # and the `where` is cheap, so paying them always is far better than
    # paying a sync always.
    e_u = torch.zeros(m, device=W.device, dtype=u1.dtype)
    e_v = torch.zeros(n, device=W.device, dtype=v1.dtype)
    e_u[0] = 1.0
    e_v[0] = 1.0
    d = degenerate.unsqueeze(-1)
    u1 = torch.where(d, e_u.expand_as(u1), u1)
    v1 = torch.where(d, e_v.expand_as(v1), v1)
    sigma = torch.where(degenerate, torch.zeros_like(sigma), sigma)
    return sigma.abs(), u1, v1


@torch.no_grad()
def spectral_norm(
    W: torch.Tensor,
    v0: torch.Tensor | None = None,
    n_iter: int = DEFAULT_N_ITER,
    n_iter_cold: int = DEFAULT_N_ITER_COLD,
) -> torch.Tensor:
    """`sigma_max(W)` only, for callers that do not need the singular vectors."""
    return top_singular_pair(W)[0]


_BACKENDS: dict = {}
