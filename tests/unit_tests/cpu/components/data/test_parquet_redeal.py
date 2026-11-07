# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Re-dealing an epoch's remainder after a data-parallel degree change: the
unread rows of every old rank's stream (rest of the open spans, then the
unopened ones), dealt to the new ranks, cover those rows exactly once, in
balanced contiguous chunks."""

import random

import pytest

from torchtitan.components.data.parquet_manifest import (
    redeal_spans,
    RowSpan,
    unread_spans,
)


def _rows(spans):
    return [(s.shard, s.row_group, r) for s in spans for r in range(s.start, s.stop)]


def _random_stream(rng, shard):
    spans = [
        RowSpan(shard, g, rng.randrange(0, 50), 0) for g in range(rng.randrange(1, 8))
    ]
    spans = [
        RowSpan(s.shard, s.row_group, s.start, s.start + rng.randrange(1, 300))
        for s in spans
    ]
    next_span = rng.randrange(0, len(spans) + 1)
    opened = list(range(next_span))
    width = rng.randrange(1, 4)
    slots = []
    for i in opened[-width:]:
        slots.append(
            {"span_index": i, "row_in_span": rng.randrange(0, spans[i].num_rows + 1)}
        )
    slots += [None] * (width - len(slots))
    return spans, slots, next_span


def test_unread_spans_are_the_rest_of_open_spans_then_unopened():
    spans = [RowSpan("a", 0, 10, 20), RowSpan("a", 1, 0, 5), RowSpan("b", 0, 0, 7)]
    slots = [{"span_index": 0, "row_in_span": 4}, None]
    assert unread_spans(spans, slots, 1) == [
        RowSpan("a", 0, 14, 20),
        RowSpan("a", 1, 0, 5),
        RowSpan("b", 0, 0, 7),
    ]


@pytest.mark.parametrize("old,new", [(4, 2), (2, 4), (3, 5), (5, 3), (1, 7)])
def test_redeal_covers_the_unread_rows_once_in_balanced_chunks(old, new):
    rng = random.Random(old * 31 + new)
    unread = []
    for rank in range(old):
        spans, slots, next_span = _random_stream(rng, f"r{rank}")
        unread += unread_spans(spans, slots, next_span)
    chunks = redeal_spans(unread, new)
    assert len(chunks) == new
    assert _rows([s for c in chunks for s in c]) == _rows(unread)
    sizes = [sum(s.num_rows for s in c) for c in chunks]
    assert max(sizes) - min(sizes) <= 1


def _stream(root, dp_rank, dp_world_size, seed=0, width=3):
    from torchtitan.components.data.parquet_stream import ParquetStreamSource
    from torchtitan.components.data.types import DatasetIterationPolicy

    cfg = ParquetStreamSource.Config(
        path=str(root), columns=("text",), num_concurrent_spans=width
    )
    policy = DatasetIterationPolicy(
        seed=seed,
        shuffle=False,
        repeat=True,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        streaming_shuffle_buffer_size=0,
    )
    return cfg, iter(cfg.build().build_iter_dataset(dataset_iteration_policy=policy))


@pytest.mark.parametrize("old,new,reads", [(3, 2, (5, 0, 13)), (2, 3, (9, 20))])
def test_a_dp_change_reads_the_rest_of_the_pass_once(tmp_path, old, new, reads):
    """Old ranks read part of the pass, then the job restarts at another DP degree:
    the rows read before plus each new rank's re-dealt remainder cover the pass
    exactly once; the next pass is the new rank's normal share."""
    from torchtitan.components.data.parquet_manifest import rank_row_range
    from torchtitan.components.data.parquet_stream import redealt_state

    from tests.unit_tests.cpu.components.data.test_parquet_stream import _corpus

    root = _corpus(tmp_path / "ds", shards=3, groups=3, rows=7)  # 63 rows, 9 groups
    seen, states = [], []
    for rank, n in enumerate(reads):
        _, it = _stream(root, rank, old)
        seen += [next(it)["text"] for _ in range(n)]
        states.append(it.get_state())
    unread = 63 - len(seen)
    for rank in range(new):
        cfg, it = _stream(root, rank, new)
        it.set_state(
            redealt_state(
                cfg, states, dp_rank=rank, dp_world_size=new, fresh_state=it.get_state()
            )
        )
        lo, hi = rank_row_range(unread, rank, new)
        seen += [next(it)["text"] for _ in range(hi - lo)]
        state = it.get_state()
        assert state["epoch"] == 0
        _, own = _stream(root, rank, new)
        share = rank_row_range(63, rank, new)
        nxt = sorted(next(it)["text"] for _ in range(share[1] - share[0]))
        own_rows = sorted(next(own)["text"] for _ in range(share[1] - share[0]))
        assert it.get_state()["epoch"] == 1
        assert {r[4:] for r in nxt} <= {f"{i:06d}" for i in range(63)}
        assert len(nxt) == len(own_rows) == len(set(nxt))
    assert sorted(seen) == [f"row-{i:06d}" for i in range(63)]


def test_a_cursor_keeps_its_seed_when_the_stream_is_rebuilt_with_another(tmp_path):
    """A stream rebuilt with another seed (e.g. a changed loader seed) resumes the
    saved cursor in the span order it was taken under."""
    from tests.unit_tests.cpu.components.data.test_parquet_stream import _corpus

    root = _corpus(tmp_path / "ds", shards=3, groups=3, rows=7)
    _, it = _stream(root, 0, 1, seed=5)
    for _ in range(10):
        next(it)
    state = it.get_state()
    expected = [next(it)["text"] for _ in range(80)]
    _, moved = _stream(root, 0, 1, seed=6)
    moved.set_state(state)
    assert [next(moved)["text"] for _ in range(80)] == expected
