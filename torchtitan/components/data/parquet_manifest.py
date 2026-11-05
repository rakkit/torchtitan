# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parquet manifests: describe what is on disk so ranks never have to discover it.

WHY THIS EXISTS
---------------
Without a manifest, every data-parallel rank independently discovers the corpus.
``datasets.load_dataset`` on a directory runs, per rank: repeated recursive globs to
infer the split layout, then ``fs.info()`` on every file to read mtimes. At 2048 shards
x 1000+ ranks on a shared filesystem that is a metadata storm that hangs the cluster.

A manifest turns startup into a single small file read. **Exactly one process ever walks
the directory** -- that is the invariant that matters, not the file itself.

WHAT IS AND IS NOT IN HERE
--------------------------
The manifest describes *what is on disk*. The training config decides *what to train
on*. Nothing here expresses intent:

  manifest  -> shard paths, row-group row counts, available columns, totals
  config    -> which column to read, mixing weight, dataset id/alias

In particular there are deliberately **no token counts**. A ``token_count`` column
exists in many corpora but is neither guaranteed present nor guaranteed accurate, so
nothing here depends on it. Balancing is by ROW COUNT, which is sound because the
corpora are pre-shuffled: equal rows per rank means roughly equal tokens per rank
without ever needing to know token counts.

ASSIGNMENT IS BY ROW RANGE, NOT BY FILE OR ROW GROUP
----------------------------------------------------
Measured over 40 real corpora (2026-09-10): **23 of 40 have exactly one row group per
shard**, and 16 of 40 have fewer than 2048 row groups in total -- the 13
``PDF-HQ-topic-split/*`` datasets have 256 shards x 1 row group, which would starve 768
of 1024 ranks under any file- or row-group-granular assignment.

So a rank takes a contiguous *row* range and reads only the row groups overlapping it:

  * many row groups  -> each rank gets whole, distinct groups; no sharing.
  * few row groups   -> a group is shared by ceil(W / num_row_groups) ranks, each
                        decompressing it and keeping its slice. Bounded and small.

Either way a rank opens only the files overlapping its range, never all of them.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1

#: Never split a span below this many rows. The cost of a sub-span is the prefix of its
#: row group that must be decoded to reach it, so splitting is cheap in absolute terms
#: for the small per-rank shares this exists for; the floor only stops it degenerating
#: into per-row seeking. 64 rows is still ~64 documents of mixing per stream.
_MIN_SPAN_ROWS = 64


@dataclass(frozen=True)
class RowSpan:
    """A contiguous slice of one parquet row group assigned to one rank."""

    shard: str
    """Shard path, relative to the dataset directory."""
    row_group: int
    """Row-group index within that shard."""
    start: int
    """First row to take, relative to the start of the row group."""
    stop: int
    """One past the last row to take, relative to the start of the row group."""

    @property
    def num_rows(self) -> int:
        return self.stop - self.start


def _dir_mtime(dataset_dir: Path) -> float:
    try:
        return Path(dataset_dir).stat().st_mtime
    except OSError:
        return 0.0


def check_freshness(manifest: dict[str, Any], dataset_dir: Path) -> str | None:
    """Cheap staleness signal: one stat, no directory listing.

    Returns a message when the directory has changed since the manifest was built,
    otherwise None. Adding shards is the case that is otherwise SILENT -- the extra data
    is simply never read, and training quietly uses the old subset.
    """
    recorded = manifest.get("source_mtime")
    if not recorded:
        return None  # built before this field existed; nothing to compare
    current = _dir_mtime(dataset_dir)
    if current and abs(current - recorded) > 1.0:
        return (
            f"{dataset_dir}: directory changed since {MANIFEST_FILENAME} was built "
            f"({manifest.get('built_at')}). Shards added after the build are NOT read "
            "and removed shards will fail at read time. Rebuild with "
            "build_py_config_dataset_manifests.py."
        )
    return None


def _read_shard_footer(args: tuple[Path, Path]) -> dict[str, Any]:
    """Read one parquet footer. No data pages are touched."""
    import pyarrow.parquet as pq

    path, dataset_dir = args
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    return {
        "path": str(path.relative_to(dataset_dir)),
        "row_groups": [
            metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)
        ],
        "_columns": {
            name: str(dtype)
            for name, dtype in zip(
                parquet_file.schema_arrow.names, parquet_file.schema_arrow.types
            )
        },
    }


def build_manifest(dataset_dir: Path, *, max_workers: int = 32) -> dict[str, Any]:
    """Walk ``dataset_dir`` once and describe every parquet shard in it.

    Only footers are read, so cost is metadata-bound rather than data-bound. Shards are
    sorted by path so the manifest -- and therefore the training order -- is
    deterministic.
    """
    dataset_dir = Path(dataset_dir)
    files = sorted(p for p in dataset_dir.rglob("*.parquet") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {dataset_dir}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        shards = list(pool.map(_read_shard_footer, ((f, dataset_dir) for f in files)))

    # Columns are recorded so the loader can validate the configured text key at startup
    # instead of silently yielding nothing. Take them from the first shard; a corpus with
    # a non-uniform schema is a data bug we want to surface, not paper over.
    columns = shards[0].pop("_columns")
    for shard in shards[1:]:
        shard.pop("_columns", None)

    return {
        "version": MANIFEST_VERSION,
        "source_dir": str(dataset_dir),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Directory mtime changes when a shard is added or removed, so one stat() at
        # startup detects a stale manifest without listing the directory -- listing is
        # exactly the storm this file exists to avoid.
        "source_mtime": _dir_mtime(dataset_dir),
        "columns": columns,
        "num_rows": sum(sum(s["row_groups"]) for s in shards),
        "num_shards": len(shards),
        "shards": shards,
    }


def _write_json(manifest: dict[str, Any], path: Path) -> None:
    """Write atomically, so a concurrent reader never sees a half-written manifest."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(path)


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Write the manifest, then re-record the directory mtime the write itself changed.

    The manifest normally lives INSIDE the directory it describes, so creating it bumps
    that directory's mtime -- past the `source_mtime` recorded moments earlier during the
    walk. Without the second pass every freshly built manifest is born "stale" and warns
    on the first read of every run, which trains people to ignore the one warning that
    catches silently-unread shards.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest, path)

    source_dir = Path(manifest.get("source_dir", path.parent))
    settled = _dir_mtime(source_dir)
    if settled and settled != manifest.get("source_mtime"):
        manifest["source_mtime"] = settled
        _write_json(manifest, path)
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON ({exc}). It may be truncated from an interrupted "
            "build; delete it and rebuild with "
            "build_py_config_dataset_manifests.py."
        ) from exc
    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest version {version!r}, expected {MANIFEST_VERSION}. "
            "Rebuild it with reproduce_cfgs/helper_scripts/build_py_config_dataset_manifests.py"
        )
    return manifest


def rank_row_range(num_rows: int, dp_rank: int, dp_world_size: int) -> tuple[int, int]:
    """Split ``num_rows`` into contiguous per-rank ranges differing by at most one row.

    Equal row counts are the whole balancing story: the corpora are pre-shuffled, so
    equal rows per rank gives roughly equal tokens per rank without knowing token counts.
    """
    if not 0 <= dp_rank < dp_world_size:
        raise ValueError(f"dp_rank {dp_rank} out of range for size {dp_world_size}")
    per_rank, remainder = divmod(num_rows, dp_world_size)
    start = dp_rank * per_rank + min(dp_rank, remainder)
    stop = start + per_rank + (1 if dp_rank < remainder else 0)
    return start, stop


def row_spans_for_range(
    manifest: dict[str, Any], start: int, stop: int
) -> list[RowSpan]:
    """Map a global row range onto the row groups that overlap it.

    Row groups entirely outside the range are skipped, so the caller opens only the
    shards it actually needs.
    """
    spans: list[RowSpan] = []
    cursor = 0
    for shard in manifest["shards"]:
        shard_rows = sum(shard["row_groups"])
        if cursor + shard_rows <= start:  # whole shard is before the range
            cursor += shard_rows
            continue
        if cursor >= stop:  # this and every later shard is past the range
            break
        for row_group_idx, group_rows in enumerate(shard["row_groups"]):
            group_start, group_stop = cursor, cursor + group_rows
            cursor = group_stop
            if group_stop <= start or group_start >= stop:
                continue
            spans.append(
                RowSpan(
                    shard=shard["path"],
                    row_group=row_group_idx,
                    start=max(start, group_start) - group_start,
                    stop=min(stop, group_stop) - group_start,
                )
            )
    return spans


def assign_row_spans(
    manifest: dict[str, Any], dp_rank: int, dp_world_size: int
) -> list[RowSpan]:
    """The rank's share of the dataset, as row-group-aligned spans."""
    start, stop = rank_row_range(manifest["num_rows"], dp_rank, dp_world_size)
    return row_spans_for_range(manifest, start, stop)


def split_spans_for_interleave(spans: list[RowSpan], target: int) -> list[RowSpan]:
    """Subdivide spans so a rank has at least `target` of them to interleave.

    At high DP a rank's whole row range can fall inside ONE row group, and then both
    shuffle mechanisms go inert: `_epoch_spans` permutes a list of one, and the
    concurrent-span width caps at the span count. Every epoch then replays the identical
    order -- exactly what `reshuffle_spans_per_epoch` exists to prevent, on precisely the
    small corpora that wrap most often. Measured on the production mixture at DP=1024,
    17 of 44 corpora land here.

    A span is a (shard, row_group, start, stop) slice, so splitting it is arithmetic on
    the row range and opens no extra file. The cost is that sibling sub-spans of the same
    row group each decode from the group's start to reach their slice, so a group is
    decompressed more than once per epoch. That is affordable exactly where it is needed:
    these corpora are small (a rank holds far less than one group) and low weight. Spans
    are never split below `_MIN_SPAN_ROWS`, which bounds the trade.
    """
    if target <= len(spans):
        return list(spans)

    per_span = -(-target // max(1, len(spans)))
    out: list[RowSpan] = []
    for span in spans:
        pieces = min(per_span, max(1, span.num_rows // _MIN_SPAN_ROWS))
        if pieces <= 1:
            out.append(span)
            continue
        edges = [span.start + (span.num_rows * i) // pieces for i in range(pieces + 1)]
        out.extend(
            RowSpan(shard=span.shard, row_group=span.row_group, start=lo, stop=hi)
            for lo, hi in zip(edges, edges[1:])
            if hi > lo
        )
    return out


def sharing_factor(manifest: dict[str, Any], dp_world_size: int) -> int:
    """How many ranks share one row group, worst case.

    1 means every rank reads whole distinct row groups. Larger values mean that many
    ranks each decompress the same group and keep a slice of it -- correct, but worth
    surfacing so the overlap is visible rather than silent.
    """
    num_row_groups = sum(len(shard["row_groups"]) for shard in manifest["shards"])
    if num_row_groups == 0:
        return 0
    return max(1, -(-dp_world_size // num_row_groups))


def validate_column(manifest: dict[str, Any], column: str, dataset_dir: Any) -> None:
    """Fail at startup on a missing column, instead of yielding an empty dataset.

    The legacy loader wrapped sample processing in a bare ``except Exception: continue``,
    so a mistyped key produced no rows, no counter and no log.
    """
    columns = manifest.get("columns") or {}
    if column not in columns:
        raise ValueError(
            f"Column {column!r} not found in {dataset_dir}. "
            f"Available columns: {sorted(columns)}"
        )
