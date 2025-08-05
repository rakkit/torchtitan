# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch.distributed.tensor import DTensor


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
def rms_to_l1_norm(W):
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
    S = torch.linalg.svdvals(W.to(torch.float32), driver="gesvd")
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
    S = torch.linalg.svdvals(W.to(torch.float32), driver="gesvd")
    spec = S[0]
    if spec == 0:
        return torch.tensor(0.0, device=W.device)
    frob_norm = frobenius_norm(W)
    return (frob_norm**2) / (spec**2)


@torch.no_grad()
def effective_rank(W):
    # https://docs.modula.systems/examples/weight-erasure/
    S = torch.linalg.svdvals(W.to(torch.float32), driver="gesvd")
    p = (S / (S.sum() + 1e-12)).clamp_min(1e-12)
    return torch.exp(-(p * p.log()).sum())


NORM_FUNCTIONS = {
    "rms_to_rms": rms_to_rms_norm,
    "l1_to_rms": l1_to_rms_norm,
    "rms_to_l1": rms_to_l1_norm,
    "supremum": supremum_norm,
    "condition_number": condition_number,
    "frobenius_norm": frobenius_norm,
    "average_entry_size": average_entry_size,
    "stable_rank": stable_rank,
    "effective_rank": effective_rank,
}


@torch.no_grad()
@torch.compile(fullgraph=True)
def fused_metrics(W, eps=1e-20):
    if W.ndim < 2:
        # Operator norms require a matrix.
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
    rms_to_l1 = row_l2.max() * math.sqrt(fan_in)

    S = torch.linalg.svdvals(Wf, driver="gesvd")

    spec = S[0] * math.sqrt(fan_in / fan_out)

    cond = S[0] / (S[-1] + eps)
    cond = cond.clamp_min(eps)

    frob_norm = row_l2.norm(p=2)

    spec_unscaled = S[0]
    srank = (frob_norm**2) / (spec_unscaled**2 + eps)
    srank = srank.clamp_min(eps)

    p = (S / (S.sum() + eps)).clamp_min(eps)
    erank = torch.exp(-(p * p.log()).sum())

    avg_entry = frob_norm / math.sqrt(fan_out * fan_in)

    return {
        "rms_to_rms": spec,
        "l1_to_rms": l1_to_rms,
        "rms_to_l1": rms_to_l1,
        "supremum": sup,
        "condition_number": cond,
        "frobenius_norm": frob_norm,
        "average_entry_size": avg_entry,
        "stable_rank": srank,
        "effective_rank": erank,
    }


def calculate_norm(
    W: torch.Tensor,
    transpose: bool = False,
    use_fused_metrics: bool = True,
) -> dict[str, torch.Tensor]:
    """
    It is important to note that the order of the norms is the same
    as the order of `NORM_FUNCTIONS.keys()`.
    """
    W = W.to_local() if isinstance(W, DTensor) else W
    if transpose:
        W = W.transpose(0, 1)
    if use_fused_metrics:
        norms = fused_metrics(W)
    else:
        norms = {
            norm_name: norm_fn(W) for (norm_name, norm_fn) in NORM_FUNCTIONS.items()
        }

    return norms
