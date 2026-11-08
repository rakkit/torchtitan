# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for K/V all-gather varlen context parallelism."""

import contextlib
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import spmd_types as spmd
import torch
import torch.distributed as dist
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.accelerator_graph import AcceleratorGraphInputSpec
from torchtitan.distributed.context_parallel import (
    document_cp_fragments,
    get_token_fragments,
)
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.attention import (
    cp_attention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVSlidingWindowVarlenInnerAttention,
    AllGatherKVVarlen,
    AllGatherKVVarlenInnerAttention,
)


def _metadata(offsets: list[int]) -> VarlenAttentionMetadata:
    cu_seq = torch.tensor(offsets, dtype=torch.int32)
    max_seq = int(torch.diff(cu_seq).max().item())
    return VarlenAttentionMetadata(
        cu_seq_q=cu_seq, cu_seq_k=cu_seq, max_q=max_seq, max_k=max_seq
    )


def _layouts(offsets: list[int], *, cp_size: int) -> dict:
    """Each rank's global token ranges under head-tail and the document
    layout (the longest document split per document)."""
    num_tokens = offsets[-1]
    head_tail = _HeadTailLoadBalancer(
        num_tokens, cp_size, torch.device("cpu")
    )._generate_indices()
    return {
        "head_tail": get_token_fragments(
            num_tokens, cp_size=cp_size, permutation=head_tail
        ),
        "document": document_cp_fragments(
            offsets, cp_size=cp_size, min_document_length=8
        ),
    }


def _check_pieces(test, offsets, fragments, window) -> None:
    num_tokens, cp_size = offsets[-1], len(fragments)
    local = [[g for a, b in ranges for g in range(a, b)] for ranges in fragments]
    tails = []
    for rank in range(cp_size):
        _, _, _, tail_index = cp_attention._fragment_pieces(
            offsets, fragments, cp_rank=rank, window=window
        )
        if tail_index is not None:
            tails += [local[rank][i] for i in tail_index.tolist()]
    for rank in range(cp_size):
        cu_q, cu_k, rows, _ = cp_attention._fragment_pieces(
            offsets, fragments, cp_rank=rank, window=window
        )
        # the key source: the gathered rows (full) or [local; tails] (window)
        source = (
            [g for ranks in local for g in ranks] if window < 0 else local[rank] + tails
        )
        test.assertEqual(int(cu_q[-1]), num_tokens // cp_size)
        for i in range(len(cu_q) - 1):
            queries = local[rank][cu_q[i] : cu_q[i + 1]]
            keys = [source[j] for j in rows[cu_k[i] : cu_k[i + 1]]]
            first, last = queries[0], queries[-1]
            test.assertEqual(queries, list(range(first, last + 1)))
            doc_start = max(o for o in offsets if o <= first)
            key_start = doc_start if window < 0 else max(doc_start, first - window)
            test.assertEqual(keys, list(range(key_start, last + 1)))


def _reference_varlen(q, k, v, *, attention_metadata, **kwargs):
    """Varlen attention in fp64, causal aligned to each sequence's last key."""
    meta = attention_metadata
    left = meta.window_size[0]
    out = torch.zeros_like(q)
    cu_q, cu_k = meta.cu_seq_q.tolist(), meta.cu_seq_k.tolist()
    for i in range(len(cu_q) - 1):
        qs, ks = q[cu_q[i] : cu_q[i + 1]], k[cu_k[i] : cu_k[i + 1]]
        vs = v[cu_k[i] : cu_k[i + 1]]
        if not len(qs):
            continue
        lq, lk = len(qs), len(ks)
        pos_q = torch.arange(lq).unsqueeze(1) + (lk - lq)
        pos_k = torch.arange(lk).unsqueeze(0)
        allowed = pos_k <= pos_q
        if left >= 0:
            allowed &= pos_k >= pos_q - left
        scores = torch.einsum("qhd,khd->hqk", qs, ks).masked_fill(~allowed, -torch.inf)
        out[cu_q[i] : cu_q[i + 1]] = torch.einsum(
            "hqk,khd->qhd", scores.softmax(-1), vs
        )
    return out


@contextlib.contextmanager
def _emulated_cp_rank(*, rank: int, cp_size: int, gather=None):
    """Run cp_attention as CP rank ``rank`` of ``cp_size`` in one process;
    ``gather`` stands in for the K/V all-gather."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                cp_attention,
                "spmd_mesh_group",
                return_value=SimpleNamespace(size=lambda: cp_size),
            )
        )
        stack.enter_context(
            mock.patch.object(cp_attention.dist, "get_rank", return_value=rank)
        )
        if gather is not None:
            stack.enter_context(
                mock.patch.object(cp_attention, "cp_all_gather_tensors", gather)
            )
        yield


class TestDocumentLayout(TestCase):
    def test_longest_document_head_tail_rest_contiguous(self) -> None:
        # CP 2: the 80-token document in 4 chunks of 20 (rank 0: chunks 0 and
        # 3, rank 1: chunks 1 and 2); the other 48 tokens in row order, 24 each
        offsets = [0, 10, 30, 110, 128]
        fragments = document_cp_fragments(offsets, cp_size=2, min_document_length=64)
        self.assertEqual(
            fragments,
            [[(0, 24), (30, 50), (90, 110)], [(24, 30), (50, 90), (110, 128)]],
        )
        for cp_size in (2, 4):
            fragments = document_cp_fragments(
                offsets, cp_size=cp_size, min_document_length=64
            )
            covered = sorted(g for f in fragments for a, b in f for g in range(a, b))
            self.assertEqual(covered, list(range(128)))
            for ranges in fragments:
                self.assertEqual(sum(b - a for a, b in ranges), 128 // cp_size)
                self.assertLessEqual(len(ranges), 4)
        # no document long enough: contiguous shards
        self.assertEqual(
            document_cp_fragments(offsets, cp_size=2, min_document_length=81),
            [[(0, 64)], [(64, 128)]],
        )


class TestAllGatherKVVarlenPreparation(TestCase):
    def test_pieces_hold_each_querys_keys(self) -> None:
        # every local query row lands in exactly one piece whose keys are its
        # document's rows up to it (the last window + 1 with a sliding window)
        offsets = [0, 3, 20, 28, 32]
        for name, fragments in _layouts(offsets, cp_size=2).items():
            for window in (-1, 3):
                with self.subTest(layout=name, window=window):
                    _check_pieces(self, offsets, fragments, window)

    def test_builds_chunk_metadata(self) -> None:
        metadata = AllGatherKVVarlen.prepare_cp_metadata(
            _metadata([0, 3, 7, 8]), permutation=None
        )
        self.assertIsNone(metadata.pieces)
        for kv_start, expected_k in ((0, [0, 3, 6, 6]), (2, [0, 1, 4, 4])):
            with self.subTest(kv_start=kv_start):
                chunk = AllGatherKVVarlen._chunk_metadata(
                    metadata, q_start=4, end=6, kv_start=kv_start
                )
                self.assertEqual(
                    chunk.cu_seq_q, torch.tensor([0, 0, 2, 2], dtype=torch.int32)
                )
                self.assertEqual(
                    chunk.cu_seq_k, torch.tensor(expected_k, dtype=torch.int32)
                )
                self.assertEqual(chunk.max_q, 2)
                self.assertEqual(chunk.max_k, 4)

    def test_chunk_metadata_spmd_types(self) -> None:
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))
        global_metadata = _metadata([0, 3, 7, 8])
        metadata = AllGatherKVVarlen.prepare_cp_metadata(
            global_metadata, permutation=None
        )
        with set_current_spmd_mesh(mesh), typecheck():
            chunk = AllGatherKVVarlen._chunk_metadata(
                metadata, q_start=4, end=6, kv_start=2
            )
            spmd.assert_type(
                global_metadata.cu_seq_q,
                {"dp": spmd.V, "cp": spmd.R, "tp": spmd.R},
                partition_spec=spmd.PartitionSpec("dp"),
            )
            spmd.assert_type(
                (chunk.cu_seq_q, chunk.cu_seq_k),
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                partition_spec=spmd.PartitionSpec("dp"),
            )

    def test_rejects_invalid_permutation(self) -> None:
        for permutation in (torch.arange(8), torch.arange(8).repeat(2, 1)):
            with self.subTest(shape=permutation.shape):
                with self.assertRaisesRegex(ValueError, "requires a permutation"):
                    AllGatherKVVarlen.prepare_cp_metadata(
                        _metadata([0, 8]), permutation=permutation
                    )

    def test_rejects_cross_attention(self) -> None:
        metadata = VarlenAttentionMetadata(
            cu_seq_q=torch.tensor([0, 4, 8], dtype=torch.int32),
            cu_seq_k=torch.tensor([0, 3, 8], dtype=torch.int32),
            max_q=4,
            max_k=5,
        )
        with self.assertRaisesRegex(ValueError, "self-attention"):
            AllGatherKVVarlen.prepare_cp_metadata(metadata, permutation=None)

    def test_rejects_noncausal_windows(self) -> None:
        for window_size in ((-1, -1), (2, 1), (-2, 0)):
            with self.subTest(window_size=window_size):
                metadata = replace(_metadata([0, 8]), window_size=window_size)
                with self.assertRaisesRegex(ValueError, "requires a causal window"):
                    AllGatherKVVarlen.prepare_cp_metadata(metadata, permutation=None)


class TestVarlenMetadataCudaGraphInputs(TestCase):
    def test_metadata_inputs_are_stable(self) -> None:
        for use_cp, head_tail in ((False, False), (True, False), (True, True)):
            with self.subTest(use_cp=use_cp, head_tail=head_tail):
                captured_metadata = _metadata([0, 4])
                rebuilt_metadata = _metadata([0, 4])
                if use_cp:
                    permutation = (
                        torch.arange(4, dtype=torch.int32).unsqueeze(0)
                        if head_tail
                        else None
                    )
                    with _emulated_cp_rank(rank=0, cp_size=2):
                        captured_metadata = AllGatherKVVarlen.prepare_cp_metadata(
                            captured_metadata, permutation=permutation
                        )
                        rebuilt_metadata = AllGatherKVVarlen.prepare_cp_metadata(
                            rebuilt_metadata, permutation=permutation
                        )
                spec = AcceleratorGraphInputSpec({"metadata": captured_metadata})
                captured = spec.flatten({"metadata": captured_metadata})
                rebuilt = spec.flatten({"metadata": rebuilt_metadata})
                self.assertEqual(
                    spec.unflatten(rebuilt), {"metadata": rebuilt_metadata}
                )
                tensors = [leaf for leaf in captured if isinstance(leaf, torch.Tensor)]
                # offsets; with a permutation also the pieces' offsets and K/V rows
                self.assertEqual(len(tensors), 5 if head_tail else 2)
                for first, second in zip(captured, rebuilt, strict=True):
                    if not isinstance(first, torch.Tensor):
                        self.assertEqual(first, second)


class TestKVAllGatherVarlenInnerAttention(TestCase):
    def test_matches_reference_attention_on_every_rank(self) -> None:
        # CP 2 emulated in one process: each rank's output equals the global
        # attention at its tokens, for contiguous, head-tail and document
        # layouts, full and sliding-window attention (fp64 reference kernel)
        offsets, cp_size = [0, 3, 20, 28, 32], 2
        num_tokens, local = offsets[-1], offsets[-1] // cp_size
        torch.manual_seed(0)
        q, k, v = (torch.randn(num_tokens, 2, 4, dtype=torch.float64) for _ in range(3))
        layouts = {
            "contiguous": [[(r * local, (r + 1) * local)] for r in range(cp_size)]
        }
        layouts.update(_layouts(offsets, cp_size=cp_size))
        for name, fragments in layouts.items():
            ids = [
                torch.tensor([g for a, b in f for g in range(a, b)]) for f in fragments
            ]
            permutation = None if name == "contiguous" else torch.cat(ids).unsqueeze(0)
            for window in (-1, 3):
                metadata = replace(_metadata(offsets), window_size=(window, 0))
                expected = _reference_varlen(q, k, v, attention_metadata=metadata)
                prepared = []
                for rank in range(cp_size):
                    with _emulated_cp_rank(rank=rank, cp_size=cp_size):
                        prepared.append(
                            AllGatherKVVarlen.prepare_cp_metadata(
                                metadata, permutation=permutation
                            )
                        )

                def gather(tensors, *, reduce_dtype, _window=window, _prep=prepared):
                    out = []
                    for x, full in zip(tensors, (k, v), strict=False):
                        parts = []
                        for rank in range(cp_size):
                            rows = full[ids[rank]]
                            if x.shape[0] == local:  # whole local shards
                                parts.append(rows)
                            elif _prep[rank].tail_index is not None:
                                parts.append(rows[_prep[rank].tail_index])
                            else:  # contiguous halo: each shard's last rows
                                parts.append(rows[local - _window :])
                        out.append(torch.cat(parts))
                    return tuple(out)

                config = (
                    AllGatherKVVarlenInnerAttention.Config()
                    if window == -1
                    else AllGatherKVSlidingWindowVarlenInnerAttention.Config(
                        sliding_window_size=window + 1
                    )
                )
                attention = config.build()
                for rank in range(cp_size):
                    with self.subTest(layout=name, window=window, rank=rank):
                        with _emulated_cp_rank(
                            rank=rank, cp_size=cp_size, gather=gather
                        ), mock.patch.object(
                            VarlenInnerAttention,
                            "_forward_varlen",
                            lambda self, q, k, v, **kw: _reference_varlen(
                                q, k, v, **kw
                            ),
                        ):
                            actual = attention(
                                q[ids[rank]],
                                k[ids[rank]],
                                v[ids[rank]],
                                attention_metadata=prepared[rank],
                            )
                        torch.testing.assert_close(actual, expected[ids[rank]])


if __name__ == "__main__":
    run_tests()
