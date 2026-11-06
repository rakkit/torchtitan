# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The attention prologue of an opt_moe layer in one launch: from the qkv
projection's output to the q / k / v the attention takes.

    q heads:   q -> RMSNorm -> RoPE
    KV heads:  k, v -> into the conv state; x + conv(x) over the last W inputs
               (this forward's rows or the state); k -> RMSNorm -> RoPE

as vLLM's Inkling prologue (``models/inkling/nvidia/ops/qkvr_prep.py``: k / v
read in place from the projection, the conv-state insert and the conv, k's
norm), plus RoPE (neox rotate-half over the first ``rope_dim`` dims; none on
NoPE layers) and without Inkling's relative projection or KV-cache store (the
attention writes its cache). The conv state is ``[blocks, 2 G, N, K]``: this
rank's k heads, then its v heads (``OptMoeConvState``), addressed by
positions through the block table as Inkling's sconv kernel.

Numerics: x + conv(x) in fp32, rounded once (as ``fused_sconv``); RMSNorm and
RoPE in fp32 from the rounded values, each rounded to the model dtype as
vLLM's ``RMSNorm`` / ``get_rope`` native paths.

Suffixes: ``T`` tokens (``BT`` per program), ``K`` head dim, ``R`` half the
rotated width."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _conv_slice(
    qkv_ptr,
    stride_qkv_t,
    col,
    cache_ptr,
    cache_head,
    d0,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    weight_ptr,
    w_row,
    tokens,
    positions,
    requests,
    slots,
    qstart,
    valid,
    row_mask,
    block_table_ptr,
    stride_bt,
    max_blocks,
    block_size,
    W: tl.constexpr,
    BT: tl.constexpr,
    S: tl.constexpr,
):
    """``x + conv(x)`` (fp32) of ``S`` channels of one head: the projection's
    columns ``col ..``, the state head ``cache_head``'s dims ``d0 ..`` (this
    forward's raw ``x`` inserted first)."""
    dims = tl.arange(0, S)
    x = tl.load(
        qkv_ptr + tokens[:, None] * stride_qkv_t + col + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    safe_slot = tl.maximum(slots, 0)
    tl.store(
        cache_ptr
        + (safe_slot // block_size)[:, None] * stride_c_blk
        + cache_head * stride_c_h
        + (safe_slot % block_size)[:, None] * stride_c_n
        + d0
        + dims[None, :],
        x,
        mask=valid[:, None],
    )
    acc = tl.zeros([BT, S], dtype=tl.float32)
    for tap in tl.static_range(W):
        src = positions - (W - 1) + tap
        row = tokens - (W - 1) + tap
        in_window = valid & (src >= 0)
        intra = in_window & (row >= qstart)
        cached = in_window & (row < qstart)
        xt = tl.load(
            qkv_ptr + tl.maximum(row, 0)[:, None] * stride_qkv_t + col + dims[None, :],
            mask=intra[:, None],
            other=0.0,
        ).to(tl.float32)
        safe_src = tl.maximum(src, 0)
        block = tl.load(
            block_table_ptr
            + requests * stride_bt
            + tl.minimum(safe_src // block_size, max_blocks - 1),
            mask=cached,
            other=0,
        ).to(tl.int64)
        xc = tl.load(
            cache_ptr
            + block[:, None] * stride_c_blk
            + cache_head * stride_c_h
            + (safe_src % block_size)[:, None] * stride_c_n
            + d0
            + dims[None, :],
            mask=cached[:, None],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight_ptr + (w_row + dims) * W + tap).to(tl.float32)
        acc += (xt + xc) * w[None, :]
    return acc + x.to(tl.float32)


@triton.jit
def _norm_rope_store(
    a,
    b,
    c,
    out_ptr,
    tokens,
    row_mask,
    out_col,
    stride_out_t,
    norm_w_ptr,
    cos_sin_ptr,
    stride_cs,
    positions,
    eps,
    K: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    QK_NORM: tl.constexpr,
    NORM_WEIGHT: tl.constexpr,
    ROPE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """One head's rounded values as its slices ``a`` [.., R], ``b`` [.., R]
    (the two rotated halves) and ``c`` [.., C] (the rest): RMSNorm over the
    head, then RoPE on ``a`` / ``b``; stored at ``out_col``."""
    ra = tl.arange(0, R)
    a = a.to(tl.float32)
    b = b.to(tl.float32)
    if C > 0:
        rc = tl.arange(0, C)
        c = c.to(tl.float32)
    if QK_NORM:
        ss = tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)
        if C > 0:
            ss += tl.sum(c * c, axis=1)
        rstd = tl.rsqrt(ss / K + eps)[:, None]
        a = (a * rstd).to(DTYPE).to(tl.float32)
        b = (b * rstd).to(DTYPE).to(tl.float32)
        if C > 0:
            c = (c * rstd).to(DTYPE).to(tl.float32)
        if NORM_WEIGHT:
            a = (
                (a * tl.load(norm_w_ptr + ra).to(tl.float32)[None, :])
                .to(DTYPE)
                .to(tl.float32)
            )
            b = (
                (b * tl.load(norm_w_ptr + R + ra).to(tl.float32)[None, :])
                .to(DTYPE)
                .to(tl.float32)
            )
            if C > 0:
                c = (
                    (c * tl.load(norm_w_ptr + 2 * R + rc).to(tl.float32)[None, :])
                    .to(DTYPE)
                    .to(tl.float32)
                )
    if ROPE:
        cs = cos_sin_ptr + positions[:, None] * stride_cs + ra[None, :]
        cos = tl.load(cs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        sin = tl.load(cs + R, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a, b = a * cos - b * sin, b * cos + a * sin
    base = out_ptr + tokens[:, None] * stride_out_t + out_col
    tl.store(base + ra[None, :], a.to(DTYPE), mask=row_mask[:, None])
    tl.store(base + R + ra[None, :], b.to(DTYPE), mask=row_mask[:, None])
    if C > 0:
        tl.store(base + 2 * R + rc[None, :], c.to(DTYPE), mask=row_mask[:, None])


@triton.jit(do_not_specialize=["T", "max_blocks", "block_size"])
def _prologue_kernel(
    qkv_ptr,
    stride_qkv_t,
    conv_w_ptr,
    q_norm_w_ptr,
    k_norm_w_ptr,
    cos_sin_ptr,
    stride_cs,
    cache_ptr,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    positions_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    stride_bt,
    max_blocks,
    qstart_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    T,
    eps,
    block_size,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    QK_NORM: tl.constexpr,
    NORM_WEIGHT: tl.constexpr,
    ROPE: tl.constexpr,
    BT: tl.constexpr,
):
    tokens = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1)  # q heads 0 .. H - 1, then KV heads
    row_mask = tokens < T
    positions = tl.load(positions_ptr + tokens, mask=row_mask, other=0)
    DTYPE = q_out_ptr.dtype.element_ty
    ra = tl.arange(0, R)
    if head < H:
        col = head * K
        base = qkv_ptr + tokens[:, None] * stride_qkv_t + col
        m = row_mask[:, None]
        a = tl.load(base + ra[None, :], mask=m, other=0.0)
        b = tl.load(base + R + ra[None, :], mask=m, other=0.0)
        c = a
        if C > 0:
            c = tl.load(base + 2 * R + tl.arange(0, C)[None, :], mask=m, other=0.0)
        _norm_rope_store(
            a,
            b,
            c,
            q_out_ptr,
            tokens,
            row_mask,
            col,
            H * K,
            q_norm_w_ptr,
            cos_sin_ptr,
            stride_cs,
            positions,
            eps,
            K,
            R,
            C,
            QK_NORM,
            NORM_WEIGHT,
            ROPE,
            DTYPE,
        )
    else:
        g = head - H
        requests = tl.load(seq_idx_ptr + tokens, mask=row_mask, other=0)
        slots = tl.load(slot_ptr + tokens, mask=row_mask, other=-1)
        qstart = tl.load(qstart_ptr + tokens, mask=row_mask, other=0)
        valid = row_mask & (slots >= 0)
        k_col = H * K + g * K
        v_col = H * K + G * K + g * K
        # k by slices: the norm needs the whole head, RoPE pairs a with b
        ka = _conv_slice(
            qkv_ptr,
            stride_qkv_t,
            k_col,
            cache_ptr,
            g,
            0,
            stride_c_blk,
            stride_c_h,
            stride_c_n,
            conv_w_ptr,
            g * K,
            tokens,
            positions,
            requests,
            slots,
            qstart,
            valid,
            row_mask,
            block_table_ptr,
            stride_bt,
            max_blocks,
            block_size,
            W,
            BT,
            R,
        ).to(DTYPE)
        kb = _conv_slice(
            qkv_ptr,
            stride_qkv_t,
            k_col + R,
            cache_ptr,
            g,
            R,
            stride_c_blk,
            stride_c_h,
            stride_c_n,
            conv_w_ptr,
            g * K + R,
            tokens,
            positions,
            requests,
            slots,
            qstart,
            valid,
            row_mask,
            block_table_ptr,
            stride_bt,
            max_blocks,
            block_size,
            W,
            BT,
            R,
        ).to(DTYPE)
        kc = ka
        if C > 0:
            kc = _conv_slice(
                qkv_ptr,
                stride_qkv_t,
                k_col + 2 * R,
                cache_ptr,
                g,
                2 * R,
                stride_c_blk,
                stride_c_h,
                stride_c_n,
                conv_w_ptr,
                g * K + 2 * R,
                tokens,
                positions,
                requests,
                slots,
                qstart,
                valid,
                row_mask,
                block_table_ptr,
                stride_bt,
                max_blocks,
                block_size,
                W,
                BT,
                C,
            ).to(DTYPE)
        _norm_rope_store(
            ka,
            kb,
            kc,
            k_out_ptr,
            tokens,
            row_mask,
            g * K,
            G * K,
            k_norm_w_ptr,
            cos_sin_ptr,
            stride_cs,
            positions,
            eps,
            K,
            R,
            C,
            QK_NORM,
            NORM_WEIGHT,
            ROPE,
            DTYPE,
        )
        dims = tl.arange(0, K)
        v = _conv_slice(
            qkv_ptr,
            stride_qkv_t,
            v_col,
            cache_ptr,
            G + g,
            0,
            stride_c_blk,
            stride_c_h,
            stride_c_n,
            conv_w_ptr,
            G * K + g * K,
            tokens,
            positions,
            requests,
            slots,
            qstart,
            valid,
            row_mask,
            block_table_ptr,
            stride_bt,
            max_blocks,
            block_size,
            W,
            BT,
            K,
        )
        tl.store(
            v_out_ptr + tokens[:, None] * (G * K) + g * K + dims[None, :],
            v.to(DTYPE),
            mask=row_mask[:, None],
        )


def attention_prologue(
    qkv_TC: torch.Tensor,
    conv_weight_CW: torch.Tensor,
    q_norm_weight: torch.Tensor | None,
    k_norm_weight: torch.Tensor | None,
    qk_norm: bool,
    eps: float,
    cos_sin_cache: torch.Tensor | None,
    rope_dim: int,
    conv_cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q ``[T, H K]``, k / v ``[T, G K]`` (contiguous) from the projection
    output ``qkv_TC`` ``[T, (H + 2 G) K]`` (any token stride); ``rope_dim`` 0:
    no RoPE (NoPE layer)."""
    T = qkv_TC.shape[0]
    H, G, K = num_heads, num_kv_heads, head_dim
    q = torch.empty((T, H * K), dtype=qkv_TC.dtype, device=qkv_TC.device)
    k = torch.empty((T, G * K), dtype=qkv_TC.dtype, device=qkv_TC.device)
    v = torch.empty_like(k)
    if T == 0:
        return q, k, v
    assert qkv_TC.stride(1) == 1 and conv_weight_CW.is_contiguous()
    assert conv_cache.stride(3) == 1 and conv_cache.shape[1] == 2 * G
    rotated = rope_dim if rope_dim else K  # NoPE: the same slices, no rotation
    R, C = rotated // 2, K - rotated
    assert R & (R - 1) == 0 and (C == 0 or C & (C - 1) == 0), (rope_dim, K)
    weight = q_norm_weight is not None
    BT = 4
    grid = (triton.cdiv(T, BT), H + G)
    _prologue_kernel[grid](
        qkv_TC,
        qkv_TC.stride(0),
        conv_weight_CW,
        q_norm_weight if weight else qkv_TC,
        k_norm_weight if weight else qkv_TC,
        cos_sin_cache if rope_dim else qkv_TC,
        cos_sin_cache.stride(0) if rope_dim else 0,
        conv_cache,
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        query_start,
        q,
        k,
        v,
        T,
        eps,
        conv_cache.shape[2],
        H=H,
        G=G,
        K=K,
        W=conv_weight_CW.shape[1],
        R=R,
        C=C,
        QK_NORM=qk_norm,
        NORM_WEIGHT=weight,
        ROPE=rope_dim > 0,
        BT=BT,
    )
    return q, k, v
