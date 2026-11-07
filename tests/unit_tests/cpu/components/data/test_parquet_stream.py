# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Streaming parquet source: same rows as random access, a fraction of the memory.

The correctness bar is unchanged -- ranks must read disjoint, complete, in-order data
and resume exactly -- so these tests assert on the rows actually produced. The memory
claim is checked separately by asserting the resident batch is far smaller than a row
group.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torchtitan.components.data.parquet_manifest import (
    build_manifest,
    MANIFEST_FILENAME,
    write_manifest,
)
from torchtitan.components.data.parquet_stream import ParquetStreamSource
from torchtitan.components.data.types import DatasetIterationPolicy


def _corpus(root: Path, shards: int, groups: int, rows: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    counter = 0
    for shard in range(shards):
        writer = pq.ParquetWriter(
            root / f"part-{shard:05d}.parquet", pa.table({"text": ["x"]}).schema
        )
        for _ in range(groups):
            writer.write_table(
                pa.table({"text": [f"row-{counter + i:06d}" for i in range(rows)]})
            )
            counter += rows
        writer.close()
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)
    return root


def _policy(dp_rank=0, dp_world_size=1, repeat=False):
    return DatasetIterationPolicy(
        seed=0,
        shuffle=False,
        repeat=repeat,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        streaming_shuffle_buffer_size=0,
    )


def _source(root: Path, dp_rank=0, dp_world_size=1, repeat=False, **kwargs):
    cfg = ParquetStreamSource.Config(path=str(root), columns=("text",), **kwargs)
    return cfg.build().build_iter_dataset(  # upstream's StreamingDataSource
        dataset_iteration_policy=_policy(dp_rank, dp_world_size, repeat)
    )


def test_disk_order_is_still_available(tmp_path):
    """num_concurrent_spans=1 plus reshuffle off is plain sequential reading."""
    root = _corpus(tmp_path / "ds", shards=2, groups=3, rows=10)
    rows = [
        r["text"]
        for r in _source(root, num_concurrent_spans=1, reshuffle_spans_per_epoch=False)
    ]
    assert rows == [f"row-{i:06d}" for i in range(60)]


def test_concurrent_spans_read_every_row_exactly_once(tmp_path):
    """Interleaving reorders; it must not drop, duplicate, or invent rows."""
    root = _corpus(tmp_path / "ds", shards=2, groups=3, rows=10)
    rows = [r["text"] for r in _source(root, num_concurrent_spans=4)]
    assert sorted(rows) == [f"row-{i:06d}" for i in range(60)]
    assert rows != [f"row-{i:06d}" for i in range(60)], "it should actually interleave"


def test_concurrent_spans_mix_distant_parts_of_the_corpus(tmp_path):
    """The property a sliding window cannot give.

    A window of W can only move a row W positions, so with sequential input on-disk
    neighbours stay neighbours. Interleaving draws from spans the per-epoch permutation
    scattered across the rank's whole range, so early output already contains rows from
    far apart on disk.
    """
    root = _corpus(tmp_path / "ds", shards=4, groups=4, rows=25)  # 400 rows, 16 spans
    head = [r["text"] for r in _source(root, num_concurrent_spans=8)][:40]
    indices = [int(t.split("-")[1]) for t in head]

    assert (
        max(indices) - min(indices) > 200
    ), f"first 40 rows span only {max(indices) - min(indices)} of 400 rows on disk"


def test_ranks_read_disjoint_and_complete(tmp_path):
    root = _corpus(tmp_path / "ds", shards=2, groups=3, rows=10)
    world = 7

    seen: list[str] = []
    for rank in range(world):
        seen.extend(r["text"] for r in _source(root, rank, world))

    assert sorted(seen) == [f"row-{i:06d}" for i in range(60)]
    assert len(set(seen)) == 60, "no row read twice"


def test_rows_per_rank_differ_by_at_most_one(tmp_path):
    """Equal rows per rank is the balancing argument for pre-shuffled corpora."""
    root = _corpus(tmp_path / "ds", shards=2, groups=3, rows=10)
    world = 7
    counts = [sum(1 for _ in _source(root, r, world)) for r in range(world)]
    assert sum(counts) == 60
    assert max(counts) - min(counts) <= 1


def test_repeat_cycles_the_dataset(tmp_path):
    """Exhaustion policy: loop and keep the weight, so the mixture ratio holds."""
    root = _corpus(tmp_path / "ds", shards=1, groups=1, rows=5)
    iterator = iter(_source(root, repeat=True, num_concurrent_spans=1))
    rows = [next(iterator)["text"] for _ in range(12)]
    assert rows[:5] == rows[5:10], "the second pass covers the same rows"


def test_without_repeat_it_stops(tmp_path):
    root = _corpus(tmp_path / "ds", shards=1, groups=1, rows=5)
    assert sum(1 for _ in _source(root, repeat=False)) == 5


@pytest.mark.parametrize("width", [1, 4])
def test_resume_continues_exactly_where_it_stopped(tmp_path, width):
    """Every row exactly once across the seam, at any interleave width."""
    root = _corpus(tmp_path / "ds", shards=2, groups=2, rows=8)

    first = iter(_source(root, num_concurrent_spans=width))
    consumed = [next(first)["text"] for _ in range(13)]
    state = first.get_state()

    second = iter(_source(root, num_concurrent_spans=width))
    second.set_state(state)
    rest = [row["text"] for row in second]

    assert sorted(consumed + rest) == [f"row-{i:06d}" for i in range(32)]
    assert not set(consumed) & set(rest), "resume must not repeat a row"


@pytest.mark.parametrize("width", [1, 4])
def test_resume_reproduces_the_uninterrupted_order(tmp_path, width):
    """The draw sequence is recomputed from (seed, epoch, block), not replayed, so a
    resumed run must emit exactly what an uninterrupted one would."""
    root = _corpus(tmp_path / "ds", shards=2, groups=2, rows=8)

    straight = [r["text"] for r in _source(root, num_concurrent_spans=width)]

    first = iter(_source(root, num_concurrent_spans=width))
    consumed = [next(first)["text"] for _ in range(13)]
    second = iter(_source(root, num_concurrent_spans=width))
    second.set_state(first.get_state())

    assert consumed + [r["text"] for r in second] == straight


def test_resume_across_a_span_boundary(tmp_path):
    """The interesting case: the cursor lands exactly at the end of a row group."""
    root = _corpus(tmp_path / "ds", shards=1, groups=4, rows=5)

    first = iter(_source(root, num_concurrent_spans=1))
    consumed = [next(first)["text"] for _ in range(10)]  # exactly two spans
    state = first.get_state()

    second = iter(_source(root, num_concurrent_spans=1))
    second.set_state(state)
    rest = [row["text"] for row in second]

    assert sorted(consumed + rest) == [f"row-{i:06d}" for i in range(20)]


def test_a_width_change_is_refused_rather_than_silently_wrong(tmp_path):
    """The slot cursors are positional; restoring them into a different number of
    slots would resume the wrong spans."""
    root = _corpus(tmp_path / "ds", shards=2, groups=2, rows=8)
    first = iter(_source(root, num_concurrent_spans=4))
    for _ in range(5):
        next(first)

    second = iter(_source(root, num_concurrent_spans=2))
    with pytest.raises(ValueError, match="num_concurrent_spans must match"):
        second.set_state(first.get_state())


def test_state_is_plain_integers(tmp_path):
    """Inspectable and diffable, unlike an opaque library blob."""
    root = _corpus(tmp_path / "ds", shards=1, groups=2, rows=6)
    iterator = iter(_source(root))
    for _ in range(7):
        next(iterator)

    state = iterator.get_state()
    assert set(state) == {
        "epoch",
        "slots",
        "next_span",
        "draw_index",
        "next_index",
        "seed",
        "spans_fingerprint",
        "remainder",
    }
    assert state["remainder"] is None, "set only by a DP-degree re-deal"
    assert state["next_index"] == 7, "next_index counts documents emitted"
    for key in ("epoch", "next_span", "draw_index", "next_index"):
        assert isinstance(state[key], int)
    for slot in state["slots"]:
        assert slot is None or all(isinstance(v, int) for v in slot.values())
    assert all(isinstance(v, int) for v in state["spans_fingerprint"])


def test_batch_size_bounds_the_resident_set(tmp_path):
    """The reason this source exists: hold a batch, not a row group.

    A random-access source must cache whole decoded row groups (measured 111-127 MiB on
    real corpora, x46 datasets = ~5.7 GiB/rank). Streaming holds one batch.
    """
    root = _corpus(tmp_path / "ds", shards=1, groups=1, rows=4096)
    parquet_file = pq.ParquetFile(root / "part-00000.parquet")
    whole_group = parquet_file.read_row_group(0, columns=["text"]).nbytes

    batch_bytes = max(
        batch.nbytes
        for batch in parquet_file.iter_batches(batch_size=256, columns=["text"])
    )

    assert batch_bytes * 4 < whole_group, (
        f"a 256-row batch ({batch_bytes}B) must be far smaller than the row group "
        f"({whole_group}B)"
    )


def test_missing_manifest_refuses_to_discover(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    pq.write_table(pa.table({"text": ["a"]}), root / "part-00000.parquet")
    with pytest.raises(FileNotFoundError, match="build_py_config_dataset_manifests"):
        _source(root)


def test_rank_with_no_rows_raises(tmp_path):
    root = _corpus(tmp_path / "ds", shards=1, groups=1, rows=4)
    with pytest.raises(ValueError, match="was assigned no rows"):
        _source(root, dp_rank=7, dp_world_size=8)


def test_stale_manifest_read_failure_names_the_manifest(tmp_path):
    """Raw pyarrow says "index out of bounds" / FileNotFoundError, which points nowhere.

    Coverage moved here when the random-access source was deleted; the streaming source
    has the same error path and the same need to name the manifest as the culprit.
    """
    root = _corpus(tmp_path / "ds", shards=2, groups=1, rows=8)
    source = _source(root)

    # Remove a shard behind the manifest's back.
    (root / "part-00001.parquet").unlink()

    with pytest.raises(RuntimeError, match="manifest is out of date"):
        list(source)


def test_added_shards_are_reported_as_stale(tmp_path):
    """The silent case: new data appears, the manifest is not rebuilt, and training
    quietly continues on the old subset."""
    import os
    import time

    from torchtitan.components.data.parquet_manifest import check_freshness

    root = _corpus(tmp_path / "ds", shards=2, groups=1, rows=4)
    manifest = build_manifest(root)
    assert check_freshness(manifest, root) is None, "a fresh manifest must be silent"

    time.sleep(1.1)  # directory mtime has 1s granularity on some filesystems
    pq.write_table(pa.table({"text": ["new"]}), root / "part-00009.parquet")
    os.utime(root, None)

    message = check_freshness(manifest, root)
    assert message is not None and "Rebuild" in message


def test_concurrent_iteration_is_safe(tmp_path):
    """Grain converts map->iter with ReadOptions(num_threads=N); several iterators may
    run at once. Each holds its own cursor and file handle, so they must not interfere."""
    import threading

    root = _corpus(tmp_path / "ds", shards=2, groups=2, rows=16)
    results: dict[int, list[str]] = {}
    errors: list[str] = []

    def worker(k: int) -> None:
        try:
            results[k] = [row["text"] for row in _source(root)]
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors[:2]
    expected = [f"row-{i:06d}" for i in range(64)]
    for rows in results.values():
        assert sorted(rows) == expected, "each iterator must see the whole shard"
    first = next(iter(results.values()))
    assert all(
        rows == first for rows in results.values()
    ), "iterators built the same way must agree; a shared cursor would diverge"


def test_next_index_keeps_counting_across_epochs(tmp_path: Path):
    """`data_docs/{alias}` must be monotone, not a saw-tooth.

    `next_index` is what `documents_consumed` reports per dataset. Counting only within
    the current epoch would reset it to zero every time a corpus wrapped -- and the
    corpora that wrap are the small, high-weight ones, i.e. exactly the ones being
    watched.
    """
    source = _source(_corpus(tmp_path / "c", shards=1, groups=1, rows=5), repeat=True)
    iterator = iter(source)

    for _ in range(12):  # two full passes plus two rows
        next(iterator)

    state = iterator.get_state()
    assert state["epoch"] == 2
    assert state["next_index"] == 12


def test_resume_after_an_epoch_wrap_restores_the_cumulative_count(tmp_path: Path):
    """The counter must survive a checkpoint, not restart from the epoch's own offset."""
    root = _corpus(tmp_path / "c", shards=1, groups=1, rows=5)
    first = iter(_source(root, repeat=True))
    for _ in range(12):
        next(first)
    state = first.get_state()

    second = iter(_source(root, repeat=True))
    second.set_state(state)

    assert second.get_state()["next_index"] == 12
    next(second)
    assert second.get_state()["next_index"] == 13


def test_resuming_into_a_changed_corpus_is_refused(tmp_path: Path):
    """A cursor is only meaningful against the span list it was taken from.

    `span_index` indexes a permutation of the rank's spans. Adding shards and rebuilding
    the manifest -- the documented remedy for a stale manifest -- renumbers them, so the
    saved integers name different rows. `check_freshness` cannot catch this: rebuilding
    re-records the directory mtime, so the manifest reads as fresh.
    """
    root = _corpus(tmp_path / "ds", shards=2, groups=2, rows=8)
    first = iter(_source(root, repeat=True))
    for _ in range(5):
        next(first)
    state = first.get_state()

    # The corpus grows and the manifest is rebuilt.
    pq.write_table(
        pa.table({"text": [f"row-{i:06d}" for i in range(900, 916)]}),
        root / "part-00002.parquet",
    )
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)

    second = iter(_source(root, repeat=True))
    with pytest.raises(ValueError, match="corpus changed since the checkpoint"):
        second.set_state(state)


@pytest.mark.parametrize("width", [1, 3])
def test_a_redealt_remainder_is_read_once_then_normal_passes(tmp_path, width):
    """After a DP-degree change a rank reads its re-dealt remainder as the rest of
    the pass (resumable mid-way), then its own share in the next passes."""
    from torchtitan.components.data.parquet_manifest import RowSpan
    from torchtitan.components.data.parquet_stream import remainder_state

    root = _corpus(tmp_path / "ds", shards=2, groups=2, rows=8)  # 32 rows, 4 groups
    remainder = [
        RowSpan("part-00000.parquet", 1, 3, 8),
        RowSpan("part-00001.parquet", 0, 0, 4),
    ]
    want = [f"row-{i:06d}" for i in [*range(11, 16), *range(16, 20)]]
    fresh = iter(_source(root, repeat=True, num_concurrent_spans=width))
    state = remainder_state(fresh.get_state(), remainder, epoch=2, emitted=40)
    it = iter(_source(root, repeat=True, num_concurrent_spans=width))
    it.set_state(state)
    head = [next(it)["text"] for _ in range(4)]
    mid = it.get_state()
    assert mid["remainder"] is not None and mid["epoch"] == 2
    resumed = iter(_source(root, repeat=True, num_concurrent_spans=width))
    resumed.set_state(mid)
    tail = [next(resumed)["text"] for _ in range(5)]
    assert sorted(head + tail) == want
    after = [next(resumed)["text"] for _ in range(32)]  # the next pass: every row
    assert sorted(after) == [f"row-{i:06d}" for i in range(32)]
    state = resumed.get_state()
    assert state["remainder"] is None and state["epoch"] == 3
    assert state["next_index"] == 40 + 9 + 32
