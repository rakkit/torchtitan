# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 reduce-scatter copy-in for bf16 gradients into an fp32 buffer.

With ``param_dtype=bf16`` and ``reduce_dtype=fp32`` FSDP packs every gradient
of a unit into one fp32 buffer viewed as ``(world_size, -1)``: row ``r`` holds
each gradient's padded dim-0 shard ``r``, cast to fp32. FSDP's kernel for this
(``chunk_cat``) runs at ~2.0 TB/s on an H100-class GPU; one 2-D Triton launch
per gradient writes the same bytes at ~3.0 TB/s (contiguous copy floor ~3.55).
The output is bit-identical (a bf16 -> fp32 cast is exact). Any other case
(other dtypes, CPU) uses FSDP's own copy-in.
"""

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _default_reduce_scatter_input_fn,
)


@triton.jit
def _shard_rows_to_fp32(
    src, dst, numel, shard_numel, row_len, col, BLOCK: tl.constexpr
):
    rank = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_shard = offs < shard_numel
    # dim-0 shards are contiguous; the last ranks' shards may run past numel (zero padding)
    x = tl.load(
        src + rank * shard_numel + offs,
        mask=in_shard & (rank * shard_numel + offs < numel),
        other=0.0,
    )
    tl.store(dst + rank * row_len + col + offs, x.to(tl.float32), mask=in_shard)


def bf16_fp32_reduce_scatter_input_fn(
    unsharded_grads: list[torch.Tensor], *, shard_dims: list[int], world_size: int
) -> Callable[[torch.Tensor], None]:
    """FSDP ``set_reduce_scatter_input_fn`` callback (see the module docstring)."""
    default_copy_in = _default_reduce_scatter_input_fn(
        unsharded_grads, shard_dims=shard_dims, world_size=world_size
    )
    if not all(
        g.is_cuda and g.dtype == torch.bfloat16 and g.is_contiguous()
        for g in unsharded_grads
    ):
        return default_copy_in

    def copy_in(out: torch.Tensor) -> None:
        if out.dtype != torch.float32:
            default_copy_in(out)
            return
        row_len, col = out.numel() // world_size, 0
        for g in unsharded_grads:
            rows = -(-g.shape[0] // world_size)  # dim-0 rows per (padded) shard
            shard_numel = rows * (g.numel() // max(g.shape[0], 1))
            block = min(8192, triton.next_power_of_2(max(shard_numel, 1)))
            _shard_rows_to_fp32[(world_size, triton.cdiv(shard_numel, block))](
                g, out, g.numel(), shard_numel, row_len, col, BLOCK=block, num_warps=8
            )
            col += shard_numel

    return copy_in
