# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Manifest build, row-range assignment, and the properties the design rests on.

The load-bearing claims, each tested rather than asserted in a docstring:

  * every rank gets a contiguous row range, ranges differ by at most one row, and
    together they partition the dataset exactly -- no gaps, no overlap;
  * a rank opens only the shards overlapping its range;
  * assignment still works when there are FEWER row groups than ranks, which is the
    real shape of 23 of 40 measured corpora and would starve any file- or
    row-group-granular scheme;
  * a mistyped column fails at startup instead of silently yielding nothing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torchtitan.components.data.parquet_manifest import (
    assign_row_spans,
    build_manifest,
    check_freshness,
    MANIFEST_FILENAME,
    rank_row_range,
    read_manifest,
    row_spans_for_range,
    RowSpan,
    sharing_factor,
    split_spans_for_interleave,
    validate_column,
    write_manifest,
)


def _make_corpus(
    root: Path, num_shards: int, rows_per_group: list[int], text: str = "text"
) -> Path:
    """Write a parquet corpus with an exact, known row-group layout."""
    root.mkdir(parents=True, exist_ok=True)
    counter = 0
    for shard in range(num_shards):
        tables = []
        for group_rows in rows_per_group:
            tables.append(
                pa.table(
                    {
                        text: [f"row-{counter + i}" for i in range(group_rows)],
                        "extra": list(range(counter, counter + group_rows)),
                    }
                )
            )
            counter += group_rows
        combined = pa.concat_tables(tables)
        # row_group_size fixes the layout only when groups are uniform; otherwise write
        # each group explicitly so the footer matches `rows_per_group` exactly.
        writer = pq.ParquetWriter(root / f"part-{shard:05d}.parquet", combined.schema)
        for table in tables:
            writer.write_table(table)
        writer.close()
    return root


def _all_rows(manifest) -> int:
    return manifest["num_rows"]


# --------------------------------------------------------------------------- build


def test_build_manifest_records_exact_row_group_layout(tmp_path):
    root = _make_corpus(tmp_path / "ds", num_shards=3, rows_per_group=[5, 7, 2])
    manifest = build_manifest(root)

    assert manifest["num_shards"] == 3
    assert manifest["num_rows"] == 3 * 14
    assert [s["row_groups"] for s in manifest["shards"]] == [[5, 7, 2]] * 3
    # Deterministic order: the manifest defines the training order.
    assert [s["path"] for s in manifest["shards"]] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    assert set(manifest["columns"]) == {"text", "extra"}


def test_manifest_round_trip_and_version_check(tmp_path):
    root = _make_corpus(tmp_path / "ds", num_shards=2, rows_per_group=[4])
    manifest = build_manifest(root)
    path = write_manifest(manifest, root / MANIFEST_FILENAME)
    assert read_manifest(path) == manifest

    bad = json.loads(path.read_text())
    bad["version"] = 999
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="manifest version"):
        read_manifest(path)


def test_spans_reconstruct_the_requested_rows(tmp_path):
    root = _make_corpus(tmp_path / "ds", num_shards=3, rows_per_group=[5, 7, 2])
    manifest = build_manifest(root)

    for start, stop in [(0, 1), (0, 42), (3, 9), (13, 15), (41, 42), (14, 28)]:
        spans = row_spans_for_range(manifest, start, stop)
        assert sum(s.num_rows for s in spans) == stop - start
        for span in spans:
            assert (
                0
                <= span.start
                < span.stop
                <= manifest["shards"][0]["row_groups"][span.row_group]
            )


def test_all_ranks_together_cover_every_row_once(tmp_path):
    root = _make_corpus(tmp_path / "ds", num_shards=3, rows_per_group=[5, 7, 2])
    manifest = build_manifest(root)
    world = 5

    seen: list[tuple[str, int, int]] = []
    for rank in range(world):
        for span in assign_row_spans(manifest, rank, world):
            seen.extend(
                (span.shard, span.row_group, row)
                for row in range(span.start, span.stop)
            )

    assert len(seen) == _all_rows(manifest), "every row assigned exactly once"
    assert len(set(seen)) == len(seen), "no row assigned to two ranks"


def test_fewer_row_groups_than_ranks_still_feeds_every_rank(tmp_path):
    """The case that breaks file- and row-group-granular assignment.

    Mirrors PDF-HQ-topic-split/*: 256 shards x 1 row group = 256 units, measured on 23
    of 40 real corpora. At DP=1024 a per-unit scheme starves 768 ranks; row ranges do
    not.
    """
    root = _make_corpus(tmp_path / "ds", num_shards=16, rows_per_group=[64])
    manifest = build_manifest(root)
    world = 64  # 4x more ranks than row groups

    assert sharing_factor(manifest, world) == 4
    per_rank = [len(assign_row_spans(manifest, r, world)) for r in range(world)]
    assert all(count > 0 for count in per_rank), "no rank may be starved"

    rows = [
        sum(s.num_rows for s in assign_row_spans(manifest, r, world))
        for r in range(world)
    ]
    assert sum(rows) == manifest["num_rows"]
    assert max(rows) - min(rows) <= 1


def test_rank_opens_only_overlapping_shards(tmp_path):
    """The property that removes the filesystem storm."""
    root = _make_corpus(tmp_path / "ds", num_shards=8, rows_per_group=[10])
    manifest = build_manifest(root)
    world = 8

    for rank in range(world):
        touched = {span.shard for span in assign_row_spans(manifest, rank, world)}
        assert len(touched) == 1, f"rank {rank} should open exactly one of 8 shards"

    all_touched = {
        span.shard
        for rank in range(world)
        for span in assign_row_spans(manifest, rank, world)
    }
    assert len(all_touched) == 8, "collectively the ranks still cover the corpus"


def test_no_sharing_when_row_groups_are_plentiful(tmp_path):
    root = _make_corpus(tmp_path / "ds", num_shards=8, rows_per_group=[10] * 4)
    manifest = build_manifest(root)
    assert sharing_factor(manifest, dp_world_size=8) == 1


# ---------------------------------------------------------------------- validation


def test_missing_column_fails_with_the_available_names(tmp_path):
    root = _make_corpus(tmp_path / "ds", num_shards=1, rows_per_group=[4])
    manifest = build_manifest(root)

    validate_column(manifest, "text", root)  # present: no raise
    with pytest.raises(ValueError, match="not found"):
        validate_column(manifest, "content", root)


def test_build_on_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        build_manifest(empty)


# --------------------------------------------------------- adversarial / regressions


def _multi_group_corpus(root: Path, groups: int, rows: int) -> Path:
    """A corpus with a known number of row groups in ONE file."""
    root.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        root / "part-00000.parquet", pa.table({"text": ["x"]}).schema
    )
    for group in range(groups):
        writer.write_table(pa.table({"text": [f"g{group}-r{i}" for i in range(rows)]}))
    writer.close()
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)
    return root


def test_added_shards_are_reported_as_stale(tmp_path):
    """The silent case: new data appears, the manifest is not rebuilt, and training
    quietly uses the old subset."""
    import os
    import time

    from torchtitan.components.data.parquet_manifest import check_freshness

    root = _make_corpus(tmp_path / "ds", num_shards=2, rows_per_group=[4])
    manifest = build_manifest(root)
    assert check_freshness(manifest, root) is None, "fresh manifest must be silent"

    time.sleep(1.1)  # directory mtime has 1s granularity on some filesystems
    pq.write_table(
        pa.table({"text": ["new"], "extra": [0]}), root / "part-00009.parquet"
    )
    os.utime(root, None)

    message = check_freshness(manifest, root)
    assert message is not None and "Rebuild" in message


def test_corrupt_manifest_says_what_to_do(tmp_path):
    root = _make_corpus(tmp_path / "ds", num_shards=1, rows_per_group=[4])
    path = root / MANIFEST_FILENAME
    write_manifest(build_manifest(root), path)
    path.write_text("{not json")

    with pytest.raises(ValueError, match="not valid JSON"):
        read_manifest(path)


def test_a_freshly_written_manifest_is_not_reported_stale(tmp_path):
    """The manifest lives in the directory it describes, so writing it bumps that
    directory's mtime. Recording the mtime only during the walk made every new manifest
    warn on its first read -- observed on a real run, and exactly the way to train
    people to ignore the warning that matters."""
    root = tmp_path / "corpus"
    root.mkdir()
    pq.write_table(pa.table({"text": ["a", "b"]}), root / "p.parquet")

    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)

    assert check_freshness(read_manifest(root / MANIFEST_FILENAME), root) is None


def test_adding_a_shard_is_still_reported_stale(tmp_path):
    """The freshness check must keep catching the case it exists for."""
    root = tmp_path / "corpus"
    root.mkdir()
    pq.write_table(pa.table({"text": ["a"]}), root / "p0.parquet")
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)

    time.sleep(1.1)  # the check has a 1 s tolerance
    pq.write_table(pa.table({"text": ["b"]}), root / "p1.parquet")

    stale = check_freshness(read_manifest(root / MANIFEST_FILENAME), root)
    assert stale is not None and "NOT read" in stale


def test_split_spans_gives_the_interleave_something_to_work_with():
    """At high DP a rank's whole range can sit inside ONE row group.

    Then span permutation permutes a list of one and the concurrent-span width caps at
    one, so every epoch replays the identical order. Splitting the range restores both.
    """
    one = [RowSpan(shard="p.parquet", row_group=3, start=1000, stop=5096)]

    split = split_spans_for_interleave(one, 8)

    assert len(split) == 8
    # The rows are partitioned: no gap, no overlap, same total.
    assert split[0].start == 1000 and split[-1].stop == 5096
    for a, b in zip(split, split[1:]):
        assert a.stop == b.start
    assert sum(s.num_rows for s in split) == 4096
    # Same shard and row group throughout: splitting opens no new file.
    assert {(s.shard, s.row_group) for s in split} == {("p.parquet", 3)}


def test_split_spans_refuses_to_shard_below_the_floor():
    """Past the floor the repeated decode of the shared row group stops paying."""
    tiny = [RowSpan(shard="p.parquet", row_group=0, start=0, stop=100)]

    split = split_spans_for_interleave(tiny, 8)

    assert len(split) == 1, "100 rows must not become 8 spans of 12"
    assert split[0] == tiny[0]


def test_split_spans_is_a_no_op_when_there_are_already_enough():
    spans = [
        RowSpan(shard=f"p{i}.parquet", row_group=0, start=0, stop=1000)
        for i in range(8)
    ]
    assert split_spans_for_interleave(spans, 8) == spans
