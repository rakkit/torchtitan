# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The MoE's EP-1 token dispatch and combine as row kernels: one program per
token moves its K routed rows (each a contiguous D-wide row). Inductor's fused
gathers for the same ops ran at ~0.75 TB/s in the l44 profile (64k tokens x top
8 x 2048; ~3 ms per layer each way).

``slot_row_N`` (N = T * K, token-major): the routed row of each (token, k) slot;
every routed row belongs to at most one slot (a padded layout leaves rows to no
slot: zero). Sums run in fp32 in k order with separate multiply and add (no FMA),
so a plain loop over k in eager PyTorch reproduces them bitwise.

Legend: T tokens, K top-k, N = T * K slots, M routed rows (N, or more when
padded), D model dim.
"""

import torch
import triton
import triton.language as tl

_NO_FMA = {"enable_fp_fusion": False}


@triton.jit
def _scatter_rows_kernel(
    x_ptr, slot_ptr, out_ptr, D, K: tl.constexpr, BLOCK_D: tl.constexpr
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D
    row = tl.load(x_ptr + token * D + cols, mask=mask)
    for k in tl.static_range(K):
        dst = tl.load(slot_ptr + token * K + k)
        tl.store(out_ptr + dst * D + cols, row, mask=mask)


@triton.jit
def _gather_sum_kernel(
    rows_ptr,
    slot_ptr,
    scores_ptr,
    out_ptr,
    D,
    K: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(K):
        src = tl.load(slot_ptr + token * K + k)
        row = tl.load(rows_ptr + src * D + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_SCORES:
            row = row * tl.load(scores_ptr + token * K + k).to(tl.float32)
        acc = acc + row
    tl.store(out_ptr + token * D + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gather_sum_backward_kernel(
    rows_ptr,
    slot_ptr,
    scores_ptr,
    grad_ptr,
    grad_rows_ptr,
    grad_scores_ptr,
    D,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    grad = tl.load(grad_ptr + token * D + cols, mask=mask, other=0.0).to(tl.float32)
    for k in tl.static_range(K):
        src = tl.load(slot_ptr + token * K + k)
        score = tl.load(scores_ptr + token * K + k).to(tl.float32)
        row = tl.load(rows_ptr + src * D + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_scores_ptr + token * K + k, tl.sum(row * grad, axis=0))
        tl.store(
            grad_rows_ptr + src * D + cols,
            (grad * score).to(grad_rows_ptr.dtype.element_ty),
            mask=mask,
        )


_BLOCK_D = 1024


@torch.library.custom_op("torchtitan::moe_scatter_rows", mutates_args=())
def scatter_rows(
    x_TD: torch.Tensor, slot_row_N: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """``out_MD[slot_row_N[t * K + k]] = x_TD[t]``; rows of no slot are zero."""
    num_tokens, dim = x_TD.shape
    top_k = slot_row_N.numel() // num_tokens
    if num_rows == slot_row_N.numel():
        out_MD = x_TD.new_empty(num_rows, dim)
    else:
        out_MD = x_TD.new_zeros(num_rows, dim)
    if not x_TD.is_cuda:
        out_MD[slot_row_N] = x_TD.repeat_interleave(top_k, dim=0)
        return out_MD
    _scatter_rows_kernel[(num_tokens, triton.cdiv(dim, _BLOCK_D))](
        x_TD.contiguous(), slot_row_N, out_MD, dim, K=top_k, BLOCK_D=_BLOCK_D
    )
    return out_MD


@scatter_rows.register_fake
def _(x_TD, slot_row_N, num_rows):
    return x_TD.new_empty(num_rows, x_TD.shape[1])


@torch.library.custom_op("torchtitan::moe_gather_sum", mutates_args=())
def gather_sum(
    rows_MD: torch.Tensor,
    slot_row_N: torch.Tensor,
    scores_TK: torch.Tensor | None,
    top_k: int,
) -> torch.Tensor:
    """``out_TD[t] = sum_k scores_TK[t, k] * rows_MD[slot_row_N[t * K + k]]`` in fp32
    (no scores: a plain sum), cast to ``rows_MD``'s dtype."""
    num_tokens, dim = slot_row_N.numel() // top_k, rows_MD.shape[1]
    if not rows_MD.is_cuda:
        acc = torch.promote_types(rows_MD.dtype, torch.float32)
        rows_TKD = rows_MD[slot_row_N].view(num_tokens, top_k, dim).to(acc)
        if scores_TK is not None:
            rows_TKD = rows_TKD * scores_TK.float().to(acc).unsqueeze(-1)
        out_TD = rows_TKD[:, 0]
        for k in range(1, top_k):
            out_TD = out_TD + rows_TKD[:, k]
        return out_TD.to(rows_MD.dtype)
    out_TD = rows_MD.new_empty(num_tokens, dim)
    _gather_sum_kernel[(num_tokens, triton.cdiv(dim, _BLOCK_D))](
        rows_MD.contiguous(),
        slot_row_N,
        scores_TK.contiguous() if scores_TK is not None else slot_row_N,
        out_TD,
        dim,
        K=top_k,
        HAS_SCORES=scores_TK is not None,
        BLOCK_D=_BLOCK_D,
        **_NO_FMA,
    )
    return out_TD


@gather_sum.register_fake
def _(rows_MD, slot_row_N, scores_TK, top_k):
    return rows_MD.new_empty(slot_row_N.numel() // top_k, rows_MD.shape[1])


@torch.library.custom_op("torchtitan::moe_gather_sum_backward", mutates_args=())
def _gather_sum_backward(
    rows_MD: torch.Tensor,
    slot_row_N: torch.Tensor,
    scores_TK: torch.Tensor,
    grad_TD: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, top_k = scores_TK.shape
    dim = rows_MD.shape[1]
    if rows_MD.shape[0] == slot_row_N.numel():
        grad_rows_MD = rows_MD.new_empty(rows_MD.shape)
    else:
        grad_rows_MD = rows_MD.new_zeros(rows_MD.shape)
    if not rows_MD.is_cuda:
        acc = torch.promote_types(rows_MD.dtype, torch.float32)
        rows_TKD = rows_MD[slot_row_N].view(num_tokens, top_k, dim).to(acc)
        grad_TD = grad_TD.to(acc)
        grad_scores_TK = (rows_TKD * grad_TD.unsqueeze(1)).sum(-1)
        grad_rows_MD[slot_row_N] = (
            (grad_TD.unsqueeze(1) * scores_TK.float().to(acc).unsqueeze(-1))
            .reshape(-1, dim)
            .to(rows_MD.dtype)
        )
        return grad_rows_MD, grad_scores_TK.float().to(scores_TK.dtype)
    grad_scores_TK = torch.empty(
        num_tokens, top_k, device=scores_TK.device, dtype=torch.float32
    )
    _gather_sum_backward_kernel[(num_tokens,)](
        rows_MD.contiguous(),
        slot_row_N,
        scores_TK.contiguous(),
        grad_TD.contiguous(),
        grad_rows_MD,
        grad_scores_TK,
        dim,
        K=top_k,
        BLOCK_D=triton.next_power_of_2(dim),
        **_NO_FMA,
    )
    return grad_rows_MD, grad_scores_TK.to(scores_TK.dtype)


@_gather_sum_backward.register_fake
def _(rows_MD, slot_row_N, scores_TK, grad_TD):
    return rows_MD.new_empty(rows_MD.shape), scores_TK.new_empty(scores_TK.shape)


def _scatter_rows_setup(ctx, inputs, output):
    _, slot_row_N, _ = inputs
    ctx.save_for_backward(slot_row_N)
    ctx.top_k = slot_row_N.numel() // inputs[0].shape[0]


def _scatter_rows_backward(ctx, grad_MD):
    (slot_row_N,) = ctx.saved_tensors
    return gather_sum(grad_MD, slot_row_N, None, ctx.top_k), None, None


scatter_rows.register_autograd(
    _scatter_rows_backward, setup_context=_scatter_rows_setup
)


def _gather_sum_setup(ctx, inputs, output):
    rows_MD, slot_row_N, scores_TK, top_k = inputs
    ctx.save_for_backward(rows_MD, slot_row_N, scores_TK)
    ctx.top_k = top_k


def _gather_sum_backward_fn(ctx, grad_TD):
    rows_MD, slot_row_N, scores_TK = ctx.saved_tensors
    if scores_TK is None:
        grad_rows_MD = scatter_rows(grad_TD, slot_row_N, rows_MD.shape[0])
        return grad_rows_MD, None, None, None
    grad_rows_MD, grad_scores_TK = _gather_sum_backward(
        rows_MD, slot_row_N, scores_TK, grad_TD
    )
    return grad_rows_MD, None, grad_scores_TK, None


gather_sum.register_autograd(_gather_sum_backward_fn, setup_context=_gather_sum_setup)
