# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Top-k of each row of a small last dimension (MoE router scores, E <= a few
hundred): one load per row, k max / argmax passes in registers. torch.topk
(sbtopk) took 1.4 ms per layer at 64k tokens x 256 experts in the l44 profile
(0.9 ms alone; this kernel 0.13 ms).
Values descending; equal values: lowest index first (torch.topk leaves that
order unspecified). No autograd: the routers use the indices (and the scores for
statistics only)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_topk_kernel(
    x_ptr,
    values_ptr,
    indices_ptr,
    num_rows,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, E_POW2)
    row_ok = rows < num_rows
    x = tl.load(
        x_ptr + rows[:, None] * E + cols[None, :],
        mask=row_ok[:, None] & (cols[None, :] < E),
        other=-float("inf"),
    )
    for k in tl.static_range(K):
        value, index = tl.max(
            x, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        tl.store(values_ptr + rows * K + k, value, mask=row_ok)
        tl.store(indices_ptr + rows * K + k, index.to(tl.int64), mask=row_ok)
        x = tl.where(cols[None, :] == index[:, None], -float("inf"), x)


@torch.library.custom_op("torchtitan::row_topk", mutates_args=())
def _row_topk(x_TE: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows, num_cols = x_TE.shape
    values_TK = x_TE.new_empty(num_rows, k)
    indices_TK = torch.empty(num_rows, k, device=x_TE.device, dtype=torch.int64)
    # 8 rows of 256 / 4 of 384 (padded to 512) per program: the fastest tile in
    # t10/bench_router_topk.py on GH200 (6.8x / 5.1x torch.topk at 64k rows)
    block_rows = max(1, 2048 // triton.next_power_of_2(num_cols))
    _row_topk_kernel[(triton.cdiv(num_rows, block_rows),)](
        x_TE.contiguous(),
        values_TK,
        indices_TK,
        num_rows,
        E=num_cols,
        E_POW2=triton.next_power_of_2(num_cols),
        K=k,
        BLOCK_ROWS=block_rows,
    )
    return values_TK, indices_TK


@_row_topk.register_fake
def _(x_TE, k):
    return (
        x_TE.new_empty(x_TE.shape[0], k),
        x_TE.new_empty(x_TE.shape[0], k, dtype=torch.int64),
    )


def row_topk(x_TE: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k along the last dimension of a 2-D tensor, values descending."""
    if not x_TE.is_cuda:
        return torch.topk(x_TE, k, dim=-1, sorted=True)
    return _row_topk(x_TE.detach(), k)
