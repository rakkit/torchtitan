# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Generic, Literal, TypeVar

import numpy as np
import spmd_types as spmd

import torch
import torch.distributed as dist
from torch.export import register_dataclass

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.context_parallel import (
    cp_all_gather_tensors,
    get_token_fragments,
    MAX_CP_RANK_FRAGMENTS,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from .attention import (
    create_attention_mask,
    FlexAttentionMetadata,
    FlexInnerAttention,
    InnerAttention,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)

__all__ = [
    "CPInnerAttention",
    "AllGatherKVFlex",
    "AllGatherKVVarlen",
    "AllGatherKVFlexInnerAttention",
    "AllGatherKVSlidingWindowFlexInnerAttention",
    "AllGatherKVSlidingWindowVarlenInnerAttention",
    "AllGatherKVVarlenInnerAttention",
    "Ulysses",
    "UlyssesInnerAttention",
    "UlyssesFlexInnerAttention",
    "UlyssesSlidingWindowFlexInnerAttention",
    "UlyssesSlidingWindowVarlenInnerAttention",
    "UlyssesVarlenInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1


_GlobalAttentionMetadataT = TypeVar("_GlobalAttentionMetadataT")
_LocalAttentionMetadataT = TypeVar("_LocalAttentionMetadataT")


class CPInnerAttention(
    InnerAttention,
    ABC,
    Generic[_GlobalAttentionMetadataT, _LocalAttentionMetadataT],
):
    """Inner attention that owns CP execution and metadata preparation.

    Subclasses implement the CP attention algorithm and prepare its context
    metadata for rank-local execution.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    @staticmethod
    @abstractmethod
    def prepare_cp_metadata(
        attention_metadata: _GlobalAttentionMetadataT,
        *,
        permutation: torch.Tensor | None,
    ) -> _LocalAttentionMetadataT:
        """Prepare this backend's metadata for rank-local CP execution."""
        raise NotImplementedError


class AllGatherKVFlex(CPInnerAttention[FlexAttentionMetadata, FlexAttentionMetadata]):
    """Base for K/V all-gather CP attention using FlexAttention masks."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: FlexAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> FlexAttentionMetadata:
        """Prepare a rank-local BlockMask for K/V all-gather CP.

        The returned mask covers the current rank's Q shard and the global K/V
        sequence. Its mask function maps local Q and global K/V positions from
        the permuted sequence back to their original global indices before
        applying the input mask function. The input block size and full-block
        representation are preserved.

        Args:
            attention_metadata: BlockMask for the unsharded global sequence.
            permutation: Global token permutation applied before CP sharding,
                with shape ``[1, seq_len]`` or ``[batch, seq_len]``. ``None``
                selects contiguous sharding without token reordering.

        Returns:
            A local-Q/global-KV BlockMask for the current CP rank.

        Raises:
            NotImplementedError: If the global Q length cannot be evenly
                divided into complete Q blocks across CP ranks.
            ValueError: If ``permutation`` has an invalid shape or sequence
                length.
        """
        block_mask = attention_metadata
        global_q_len, global_kv_len = block_mask.seq_lengths
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP metadata preparation requires an active multi-rank CP mesh axis."
            )
        cp_size = cp_group.size()
        cp_rank = dist.get_rank(cp_group)
        q_block_size, _ = block_mask.BLOCK_SIZE
        if global_q_len % (cp_size * q_block_size) != 0:
            raise NotImplementedError(
                f"Global Q length ({global_q_len}) must be divisible by CP size "
                f"({cp_size}) * Q block size ({q_block_size})."
            )

        if permutation is not None:
            if permutation.ndim != 2:
                raise ValueError(
                    "CP permutation must have shape [1, seq_len] or "
                    f"[batch, seq_len], but got {tuple(permutation.shape)}."
                )
            if permutation.shape[1] != global_q_len:
                raise ValueError(
                    f"CP permutation length ({permutation.shape[1]}) must match "
                    f"global Q length ({global_q_len})."
                )

        local_q_len = global_q_len // cp_size
        mask_mod = block_mask.mask_mod

        def _unpermute_idx(
            batch_idx: torch.Tensor, post_permutation_idx: torch.Tensor
        ) -> torch.Tensor:
            if permutation is None:
                return post_permutation_idx
            if permutation.shape[0] == 1:
                return permutation[0][post_permutation_idx]
            return permutation[batch_idx][post_permutation_idx]

        def _local_q_idx_to_global_q_idx(local_q_idx: torch.Tensor) -> torch.Tensor:
            local_block_idx, local_block_offset = (
                local_q_idx // q_block_size,
                local_q_idx % q_block_size,
            )
            num_local_blocks = local_q_len // q_block_size
            global_block_idx = num_local_blocks * cp_rank + local_block_idx
            return global_block_idx * q_block_size + local_block_offset

        def _local_mask_mod(
            batch_idx: torch.Tensor,
            head_idx: torch.Tensor,
            local_q_idx: torch.Tensor,
            global_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            # The original mask_mod expects indices in the pre-permutation order.
            return mask_mod(
                batch_idx,
                head_idx,
                _unpermute_idx(
                    batch_idx,
                    _local_q_idx_to_global_q_idx(local_q_idx),
                ),
                _unpermute_idx(batch_idx, global_kv_idx),
            )

        return create_attention_mask(
            _local_mask_mod,
            block_mask.kv_num_blocks.shape[0],
            block_mask.kv_num_blocks.shape[1],
            local_q_len,
            global_kv_len,
            device=block_mask.kv_num_blocks.device,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            # Preserve whether the global mask stores full blocks separately.
            separate_full_blocks=block_mask.full_kv_num_blocks is not None,
        )


class AllGatherKVFlexInnerAttention(
    AllGatherKVFlex,
    FlexInnerAttention,
):
    """FlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        AllGatherKVFlex.Config,
        FlexInnerAttention.Config,
    ):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_THK, v_THV = cp_all_gather_tensors(
            (k_THK, v_THV), reduce_dtype=self.reduce_dtype
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)


class AllGatherKVSlidingWindowFlexInnerAttention(
    AllGatherKVFlex, SlidingWindowFlexInnerAttention
):
    """SlidingWindowFlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        AllGatherKVFlex.Config,
        SlidingWindowFlexInnerAttention.Config,
    ):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_THK, v_THV = cp_all_gather_tensors(
            (k_THK, v_THV), reduce_dtype=self.reduce_dtype
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)


@dataclass(frozen=True, slots=True, kw_only=True)
class AllGatherKVVarlenAttentionMetadata(VarlenAttentionMetadata):
    """Global document offsets; with a CP permutation, this rank's query pieces
    (``pieces``: one varlen sequence per document piece of each local token
    range) and the K/V rows of each piece (``kv_index`` into the gathered K/V,
    or for a sliding window into [local K/V; the gathered fragment tails], the
    local rows each rank publishes being ``tail_index``)."""

    pieces: VarlenAttentionMetadata | None
    kv_index: torch.Tensor | None
    tail_index: torch.Tensor | None


register_dataclass(AllGatherKVVarlenAttentionMetadata)


def _fragment_pieces(
    cu_seqlens: list[int],
    fragments: list[list[tuple[int, int]]],
    *,
    cp_rank: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Host-side piece layout for `AllGatherKVVarlen` under a permutation:
    (cu_seq_q, cu_seq_k, kv_index, tail_index). Every local token range is cut
    at document starts into pieces; a piece's keys are its document's tokens
    up to its last query (the last ``window + 1`` with a sliding window), so one
    varlen call with bottom-right causal alignment covers every piece.

    Full attention reads the keys from the gathered K/V (rank-major: rank r's
    ranges at r x T / CP). A sliding window gathers only each range's last
    ``window`` rows (right-aligned in ``window``-row slots, `MAX_CP_RANK_FRAGMENTS`
    per rank) and reads [local rows; gathered tails]: the ``window`` keys
    before a range start are always in the tails of the ranges before it."""
    num_tokens = cu_seqlens[-1]
    cp_size = len(fragments)
    local = num_tokens // cp_size
    if max(len(f) for f in fragments) > MAX_CP_RANK_FRAGMENTS:
        raise ValueError(
            f"A CP permutation gives a rank more than {MAX_CP_RANK_FRAGMENTS} token "
            "ranges; varlen K/V all-gather CP supports at most that many."
        )
    cu = np.asarray(cu_seqlens, dtype=np.int64)
    source = np.full(num_tokens, -1, dtype=np.int64)
    tail_index = None
    if window < 0:
        for rank, ranges in enumerate(fragments):  # gathered rows, rank-major
            offset = rank * local
            for a, b in ranges:
                source[a:b] = np.arange(offset, offset + b - a)
                offset += b - a
    else:
        slot = window
        for rank, ranges in enumerate(fragments):
            for f, (a, b) in enumerate(ranges):
                n = min(slot, b - a)
                start = local + (rank * MAX_CP_RANK_FRAGMENTS + f + 1) * slot - n
                source[b - n : b] = np.arange(start, start + n)
        tail_index = np.zeros(MAX_CP_RANK_FRAGMENTS * slot, dtype=np.int64)
        offset = 0
        for f, (a, b) in enumerate(fragments[cp_rank]):
            n = min(slot, b - a)
            tail_index[(f + 1) * slot - n : (f + 1) * slot] = np.arange(
                offset + b - a - n, offset + b - a
            )
            offset += b - a
    q_lengths, k_lengths, kv_rows = [], [], []
    offset = 0
    for a, b in fragments[cp_rank]:
        first = int(np.searchsorted(cu, a, side="right")) - 1
        for doc in range(first, len(cu) - 1):
            s, e = int(cu[doc]), int(cu[doc + 1])
            if s >= b:
                break
            qa, qb = max(a, s), min(b, e)
            if qb <= qa:
                continue
            ks = s if window < 0 else max(s, qa - window)
            if window < 0:
                rows = source[ks:qb]
            else:  # before the range: the tails; inside it: local rows
                inside = max(ks, a)
                rows = np.concatenate(
                    [source[ks:inside], np.arange(offset + inside - a, offset + qb - a)]
                )
            assert (rows >= 0).all(), (a, b, ks, qb)
            q_lengths.append(qb - qa)
            k_lengths.append(qb - ks)
            kv_rows.append(rows)
        offset += b - a
    cu_q = np.concatenate([[0], np.cumsum(q_lengths)]).astype(np.int32)
    cu_k = np.concatenate([[0], np.cumsum(k_lengths)]).astype(np.int32)
    return cu_q, cu_k, np.concatenate(kv_rows), tail_index


class AllGatherKVVarlen(
    CPInnerAttention[VarlenAttentionMetadata, AllGatherKVVarlenAttentionMetadata],
):
    """Share K/V all-gather CP logic across the varlen variants."""

    reduce_dtype: torch.dtype

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: VarlenAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> AllGatherKVVarlenAttentionMetadata:
        """Prepare global document offsets and K/V reordering indices."""
        if not isinstance(attention_metadata, VarlenAttentionMetadata):
            raise ValueError(
                "K/V all-gather varlen CP requires VarlenAttentionMetadata, "
                f"but got {type(attention_metadata).__name__}."
            )
        left, right = attention_metadata.window_size
        if right != 0 or left < -1:
            raise ValueError("K/V all-gather varlen CP requires a causal window.")
        if attention_metadata.cu_seq_k is not attention_metadata.cu_seq_q:
            raise ValueError("Varlen K/V all-gather CP supports only self-attention.")
        pieces = kv_index = tail_index = None
        if permutation is not None:
            if permutation.ndim != 2 or permutation.shape[0] != 1:
                raise ValueError(
                    "Varlen K/V all-gather CP requires a permutation with shape "
                    "(1, seq_len)."
                )
            group = spmd_mesh_group(MeshAxisName.CP)
            assert group is not None
            cu_seqlens = attention_metadata.cu_seq_q.tolist()
            device = attention_metadata.cu_seq_q.device
            cu_q, cu_k, rows, tails = _fragment_pieces(
                cu_seqlens,
                get_token_fragments(
                    cu_seqlens[-1], cp_size=group.size(), permutation=permutation
                ),
                cp_rank=dist.get_rank(group),
                window=left,
            )
            pieces = VarlenAttentionMetadata(
                cu_seq_q=torch.from_numpy(cu_q).to(device),
                cu_seq_k=torch.from_numpy(cu_k).to(device),
                # the local row count, a bound on every piece: the varlen
                # dispatcher branches on max_q (<= 128, == 1), and a rank whose
                # longest piece crossed 128 would recompile alone
                max_q=int(cu_q[-1]),
                max_k=int(np.diff(cu_k).max()),
                window_size=attention_metadata.window_size,
            )
            kv_index = torch.from_numpy(rows).to(device)
            if tails is not None:
                tail_index = torch.from_numpy(tails).to(device)
        return AllGatherKVVarlenAttentionMetadata(
            **{
                field.name: getattr(attention_metadata, field.name)
                for field in fields(VarlenAttentionMetadata)
            },
            pieces=pieces,
            kv_index=kv_index,
            tail_index=tail_index,
        )

    @staticmethod
    def _chunk_metadata(
        varlen_metadata: VarlenAttentionMetadata,
        *,
        q_start: int,
        end: int,
        kv_start: int,
    ) -> VarlenAttentionMetadata:
        """Describe one fixed query chunk and its contiguous K/V range."""
        global_cu_seq = varlen_metadata.cu_seq_q
        if spmd.is_type_checking():
            # Document offsets vary across DP but are shared across CP and TP.
            spmd.assert_type(
                global_cu_seq,
                spmd.R,
                {MeshAxisName.DP: spmd.V},
                partition_spec=spmd.PartitionSpec(MeshAxisName.DP),
            )
        cu_seq_q = global_cu_seq.clamp(min=q_start, max=end) - q_start
        cu_seq_k = global_cu_seq.clamp(min=kv_start, max=end) - kv_start
        if spmd.is_type_checking():
            # cu_seq_q and cu_seq_k have different values across CP ranks.
            for cu_seq in (cu_seq_q, cu_seq_k):
                spmd.mutate_type(cu_seq, MeshAxisName.CP, src=spmd.R, dst=spmd.V)
        chunk_len = end - q_start
        return VarlenAttentionMetadata(
            cu_seq_q=cu_seq_q,
            cu_seq_k=cu_seq_k,
            max_q=min(varlen_metadata.max_q, chunk_len),
            max_k=min(varlen_metadata.max_k, end - kv_start),
            window_size=varlen_metadata.window_size,
        )

    def _forward_window_halo(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: AllGatherKVVarlenAttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Sliding-window CP with contiguous shards: each rank all-gathers only
        the last ``left`` K/V rows of every rank's shard and runs its shard
        against [the previous rank's rows; its own K/V]."""
        assert isinstance(self, VarlenInnerAttention)
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        assert cp_group is not None
        cp_rank = dist.get_rank(cp_group)
        left = attention_metadata.window_size[0]
        chunk_len = q_THK.shape[0]
        assert chunk_len >= left, (chunk_len, left)
        # [left] rows per rank: the end of its shard
        with spmd.no_typecheck():
            tails = [x[chunk_len - left : chunk_len] for x in (k_THK, v_THV)]
        if spmd.is_type_checking():
            for tail, x in zip(tails, (k_THK, v_THV), strict=True):
                spmd.assert_type_like(tail, x)
        halo_k, halo_v = cp_all_gather_tensors(
            tuple(tails), reduce_dtype=self.reduce_dtype
        )

        # The first rank has no predecessor: an empty halo slice, so the
        # gathered halos stay in its autograd graph and its backward joins the
        # reduce-scatter every other rank runs.
        q_start = cp_rank * chunk_len
        with spmd.no_typecheck():
            prev = slice((cp_rank - 1) * left, cp_rank * left) if cp_rank else slice(0)
            k_chunk = torch.cat((halo_k[prev], k_THK))
            v_chunk = torch.cat((halo_v[prev], v_THV))
        if spmd.is_type_checking():
            # as the full gather's slices: the K/V of this chunk's range
            spmd.assert_type_like(k_chunk, halo_k)
            spmd.assert_type_like(v_chunk, halo_v)
        metadata = self._chunk_metadata(
            attention_metadata,
            q_start=q_start,
            end=q_start + chunk_len,
            kv_start=q_start - (left if cp_rank else 0),
        )
        return self._forward_varlen(
            q_THK, k_chunk, v_chunk, attention_metadata=metadata, **kwargs
        )

    def _forward_pieces(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: AllGatherKVVarlenAttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """A permuted layout (`_fragment_pieces`): one varlen call over this
        rank's document pieces, their keys gathered by ``kv_index`` from the
        all-gathered K/V (full attention) or from [local K/V; every rank's
        range tails] (sliding window)."""
        assert isinstance(self, VarlenInnerAttention)
        assert attention_metadata.pieces is not None
        tail_index = attention_metadata.tail_index
        if tail_index is None:
            k_src, v_src = cp_all_gather_tensors(
                (k_THK, v_THV), reduce_dtype=self.reduce_dtype
            )
        else:
            with spmd.no_typecheck():
                tails = tuple(x.index_select(0, tail_index) for x in (k_THK, v_THV))
            halo_k, halo_v = cp_all_gather_tensors(
                tails, reduce_dtype=self.reduce_dtype
            )
            with spmd.no_typecheck():
                k_src, v_src = torch.cat((k_THK, halo_k)), torch.cat((v_THV, halo_v))
        with spmd.no_typecheck():
            k_buf = k_src.index_select(0, attention_metadata.kv_index)
            v_buf = v_src.index_select(0, attention_metadata.kv_index)
        return self._forward_varlen(
            q_THK, k_buf, v_buf, attention_metadata=attention_metadata.pieces, **kwargs
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: AllGatherKVVarlenAttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, AllGatherKVVarlenAttentionMetadata):
            raise ValueError(
                f"{type(self).__name__} requires AllGatherKVVarlenAttentionMetadata, "
                f"but got {type(attention_metadata).__name__}."
            )
        assert isinstance(self, VarlenInnerAttention)
        left = attention_metadata.window_size[0]
        if attention_metadata.pieces is not None:
            out_THV = self._forward_pieces(
                q_THK, k_THK, v_THV, attention_metadata=attention_metadata, **kwargs
            )
        elif left != -1:
            # Sliding window: a shard needs only its own K/V and the previous
            # `left` tokens (the end of the shard before it): gather those
            # halos, not the whole sequence.
            return self._forward_window_halo(
                q_THK, k_THK, v_THV, attention_metadata=attention_metadata, **kwargs
            )
        else:
            # No load balancer: Q is one contiguous shard; it needs at most one
            # document length of earlier keys.
            k_THK, v_THV = cp_all_gather_tensors(
                (k_THK, v_THV), reduce_dtype=self.reduce_dtype
            )
            cp_group = spmd_mesh_group(MeshAxisName.CP)
            assert cp_group is not None
            q_start = dist.get_rank(cp_group) * q_THK.shape[0]
            end = q_start + q_THK.shape[0]
            kv_start = max(0, q_start - attention_metadata.max_k)
            metadata = self._chunk_metadata(
                attention_metadata, q_start=q_start, end=end, kv_start=kv_start
            )
            out_THV = self._forward_varlen(
                q_THK,
                k_THK[kv_start:end],
                v_THV[kv_start:end],
                attention_metadata=metadata,
                **kwargs,
            )
        if kwargs.get("out_transform") is None and spmd.is_type_checking():
            spmd.assert_type(
                out_THV,
                spmd.get_local_type(q_THK),
                spmd.get_partition_spec(q_THK),
            )
        return out_THV


class AllGatherKVVarlenInnerAttention(
    AllGatherKVVarlen,
    VarlenInnerAttention[AllGatherKVVarlenAttentionMetadata],
):
    """Varlen attention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        AllGatherKVVarlen.Config,
        VarlenInnerAttention.Config,
    ):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]


class AllGatherKVSlidingWindowVarlenInnerAttention(
    AllGatherKVVarlen,
    SlidingWindowVarlenInnerAttention[AllGatherKVVarlenAttentionMetadata],
):
    """Sliding-window varlen attention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        AllGatherKVVarlen.Config,
        SlidingWindowVarlenInnerAttention.Config,
    ):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]


class Ulysses(CPInnerAttention[_GlobalAttentionMetadataT, _GlobalAttentionMetadataT]):
    """Base for Ulysses CP attention with global attention metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: _GlobalAttentionMetadataT,
        *,
        permutation: torch.Tensor | None,
    ) -> _GlobalAttentionMetadataT:
        if permutation is not None:
            raise ValueError(
                "Ulysses CP requires contiguous token shards without a permutation."
            )
        return attention_metadata


class UlyssesInnerAttention(Ulysses[_GlobalAttentionMetadataT]):
    """Move CP sharding from tokens to heads while keeping metadata global."""

    @dataclass(kw_only=True, slots=True)
    class Config(Ulysses.Config):
        pass

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP attention requires an active multi-rank CP mesh axis."
            )
        # Shard heads instead of tokens: (T/cp, H, *) -> (T, H/cp, *).
        q_THK, k_THK, v_THV = (
            spmd.redistribute(
                x,
                cp_group,
                src=spmd.S(_TOKEN_DIM),
                dst=spmd.S(_HEAD_DIM),
            )
            for x in (q_THK, k_THK, v_THV)
        )
        # super() follows the concrete class MRO to its inner attention.
        out_THV = super().forward(q_THK, k_THK, v_THV, **kwargs)
        # Back to sharded tokens: (T, H/cp, V) -> (T/cp, H, V); an
        # out_transform may return (output, lse), lse [T, H/cp] -> [T/cp, H].
        outputs = out_THV if isinstance(out_THV, tuple) else (out_THV,)
        outputs = tuple(
            spmd.redistribute(
                x,
                cp_group,
                src=spmd.S(_HEAD_DIM),
                dst=spmd.S(_TOKEN_DIM),
            )
            for x in outputs
        )
        return outputs if isinstance(out_THV, tuple) else outputs[0]


class UlyssesFlexInnerAttention(
    UlyssesInnerAttention[FlexAttentionMetadata], FlexInnerAttention
):
    """FlexInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesInnerAttention.Config, FlexInnerAttention.Config):
        pass


class UlyssesSlidingWindowFlexInnerAttention(
    UlyssesInnerAttention[FlexAttentionMetadata], SlidingWindowFlexInnerAttention
):
    """SlidingWindowFlexInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        UlyssesInnerAttention.Config,
        SlidingWindowFlexInnerAttention.Config,
    ):
        pass


class UlyssesVarlenInnerAttention(
    UlyssesInnerAttention[VarlenAttentionMetadata], VarlenInnerAttention
):
    """VarlenInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesInnerAttention.Config, VarlenInnerAttention.Config):
        pass


class UlyssesSlidingWindowVarlenInnerAttention(
    UlyssesInnerAttention[VarlenAttentionMetadata],
    SlidingWindowVarlenInnerAttention,
):
    """Sliding-window varlen attention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        UlyssesInnerAttention.Config,
        SlidingWindowVarlenInnerAttention.Config,
    ):
        pass
