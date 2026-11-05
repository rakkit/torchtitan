# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""sigma_1 / ||X||_F of the tensor gram_polar_express receives, per DiSCO unit,
on armed (norm-logging, spectral) steps. DiSCO names the stack's units around
each rule call (`units`); the rule's call sites record their orthogonaliser
input (`record`). Inert while not armed."""

import contextlib
import dataclasses
import logging

import torch

logger = logging.getLogger(__name__)

BAND = (0.94, 0.955)  # the Gram 1.01 failure band (tools/handoff/nanprobe-20261014.md)

_armed = False
_keys: list[str] | None = None
_out: dict[str, list[torch.Tensor]] = {}
_warned = False


def is_armed() -> bool:
    return _armed


def arm() -> None:
    global _armed, _out
    _armed, _out = True, {}


def disarm() -> dict[str, list[torch.Tensor]]:
    """The recorded values, per unit key: one 0-d tensor per call, in call order."""
    global _armed, _out
    out, _armed, _out = _out, False, {}
    return out


@contextlib.contextmanager
def units(keys: list[str] | None):
    """The unit keys of the stack the enclosed rule call computes, in stack order
    (None: nothing is recorded for it)."""
    global _keys
    if not _armed:
        yield
        return
    prev, _keys = _keys, None if keys is None else list(keys)
    try:
        yield
    finally:
        _keys = prev


def uses_gram(compute) -> bool:
    """`compute` (or the compute of a Chain) is gram_polar_express."""
    from .presets import Orthogonalise

    main = getattr(compute, "main", compute)
    return isinstance(main, Orthogonalise) and main.backend == "gram_polar_express"


# Power iterations for sigma_1: 200x faster than CUDA's batched fp32 SVD (measured ~1 s per
# [16, 1536, 512] stack); a lower bound, within 1e-4 when sigma_1 dominates (the band), a few % low
# on flat spectra (far from the band).
POWER_ITERS = 30


@torch.no_grad()
def s1f(x: torch.Tensor) -> torch.Tensor:
    """sigma_1 / ||x||_F per stack element ([B, m, n] or [m, n] -> [B]), fp32,
    sigma_1 by POWER_ITERS power iterations from the all-ones vector."""
    x = x.float()
    if x.ndim == 2:
        x = x[None]
    a = x if x.shape[-2] >= x.shape[-1] else x.mT
    v = torch.ones(a.shape[0], a.shape[-1], device=a.device, dtype=a.dtype)
    v = v / v.norm(dim=-1, keepdim=True)
    for _ in range(POWER_ITERS):
        v = torch.einsum("bmn,bm->bn", a, torch.einsum("bmn,bn->bm", a, v))
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    s = torch.einsum("bmn,bn->bm", a, v).norm(dim=-1)
    f = x.norm(dim=(-2, -1))
    return torch.where(
        f > 0, s / f.clamp_min(torch.finfo(torch.float32).tiny), torch.zeros_like(f)
    )


def record(compute, x: torch.Tensor) -> None:
    """`x`: the stack `compute` (gram_polar_express) is about to receive."""
    if not _armed or _keys is None or not uses_gram(compute):
        return
    vals = s1f(x)
    if vals.shape[0] != len(_keys):
        global _warned
        if not _warned:
            logger.warning(
                f"[DiSCO] gram input stack of {vals.shape[0]} for {len(_keys)} named units: not logged"
            )
            _warned = True
        return
    for k, v in zip(_keys, vals):
        _out.setdefault(k, []).append(v)


def has_gram(obj) -> bool:
    """A rule (or compute) tree contains a gram_polar_express Orthogonalise."""
    from .presets import Orthogonalise

    if isinstance(obj, Orthogonalise):
        return obj.backend == "gram_polar_express"
    if isinstance(obj, tuple):
        return any(has_gram(o) for o in obj)
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return False
    return any(
        has_gram(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init
    )
