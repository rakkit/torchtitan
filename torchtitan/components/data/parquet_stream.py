# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Streaming parquet source: read the rank's rows in order, hold only a batch.

WHY NOT RANDOM ACCESS -- IT IS A THROUGHPUT ARGUMENT, NOT A MEMORY ONE
----------------------------------------------------------------------
Parquet's atomic read unit is the ROW GROUP: one row cannot be decoded without
decompressing the whole group. MEASURED on a real shard (Nemotron-CC-v2/High-Quality,
32768 rows per group): decoding one group costs 0.17 s and yields 111.5 MiB.

A ``RandomAccessDataSource`` must serve ``__getitem__(i)`` for arbitrary ``i``, and
grain's map path hands it a GLOBALLY SHUFFLED index (``dataset.py:144-146`` shuffles the
whole index space, then slices a DP shard). Consecutive requests therefore land in
different row groups, so a per-dataset group cache almost never hits. Measured on the
same shard, 200 rows at uniformly random indices with a one-group cache:

    iter_batches, sequential      342,808 rows/s
    random access, 1-group cache       11.3 rows/s   (9 of 200 cache hits)

That is a **30,000x** slowdown, and it is the whole reason for this source. It is not
fixable by caching harder: with 46 datasets interleaved and a shuffled index inside each,
the working set is the corpus.

The memory number often quoted alongside this (one decoded group per dataset, 46 x ~124
MiB ~= 5.7 GiB per rank) is a SIDE EFFECT of the cache, not the problem. On a GH200 node
each rank has its own ~120 GB Grace CPU, so 5.7 GiB is about 5% of one rank's RAM --
affordable, and no reason on its own to avoid random access.

Streaming removes the throughput problem by construction: reads are sequential, so one
group decode is amortized over its 32768 rows instead of serving one. Holding only a
batch is a bonus:

    read_row_group(0)          111.5 MiB
    iter_batches(bs=1024)        4.5 MiB
    iter_batches(bs=4096)       14.2 MiB

MULTI-EPOCH SHUFFLING WITHOUT RANDOM ACCESS
-------------------------------------------
Streaming does give up grain's map-level index shuffle. It does NOT give up shuffling.
Two orthogonal mechanisms live here, and the measurements say you want both (12800-row
corpus, 64 spans; "disk span" is how much of the corpus a 512-row stretch of OUTPUT
covers; "epoch rho" is the order correlation between two consecutive epochs, averaged
over 3 seeds -- 0 means the epochs are independently ordered):

    configuration                        disk span   #spans   epoch rho
    window 1024 only                          8.0%      6.1   +0.983
    span permutation only                    61.1%      3.5   -0.062
    span perm + window 1024                  78.4%      6.1   -0.080
    span perm + 8 concurrent spans           82.0%     10.2   -0.041
    span perm + 8 spans + window 1024        85.4%     12.7   -0.061

A sliding window ALONE is the bad option, and visibly so: it can move a row at most
`window` positions, so epoch 2 replays epoch 1 almost exactly (rho +0.98) and any output
stretch is still one contiguous piece of disk.

  * `reshuffle_spans_per_epoch` is what decorrelates EPOCHS (+0.98 -> ~0.0). It permutes
    which row-group spans are read in which order, for a few tens to hundreds of spans,
    so it is free.
  * `num_concurrent_spans` is what breaks up LOCAL adjacency, by reading N permuted spans
    at once and drawing rows between them. No buffer of rows is held -- just N decode
    batches -- and reads stay sequential within each span.

Neither needs random access, and neither reads a row group twice.
"""

from __future__ import annotations
import logging

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import grain.python as grain

from torchtitan.components.data.parquet_manifest import (
    assign_row_spans,
    check_freshness,
    MANIFEST_FILENAME,
    read_manifest,
    RowSpan,
    sharing_factor,
    split_spans_for_interleave,
    validate_column,
)
from torchtitan.components.data.types import DatasetIterationPolicy
from torchtitan.config import Configurable


class ParquetStreamSource(Configurable):
    """Streams this rank's row range from a manifest-described parquet corpus."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        path: str
        """Dataset directory holding the shards and manifest.json."""
        manifest_path: str | None = None
        """Override, for a corpus whose directory is not writable. Normally unset."""
        columns: tuple[str, ...] = ()
        """Columns that MUST exist. Validated against the manifest at startup, so a typo
        fails immediately instead of yielding empty samples. Empty means all."""
        optional_columns: tuple[str, ...] = ()
        """Columns to decode IF the corpus has them, ignored otherwise. SFT uses this for
        `tools` / `enable_thinking`, which some corpora omit."""
        batch_size: int = 1024
        """Rows decoded at a time. This is the resident set: ~4.5 MiB at 1024 rows of
        web text, ~14 MiB at 4096. Multiply by the number of datasets in the mix."""
        num_concurrent_spans: int = 8
        """How many row-group spans to read from at once, drawing rows between them.

        This is the shuffle that matters for a corpus seen more than once, and it is
        strictly stronger than a sliding window. A window of W can only move a row W
        positions, so with a sequential input it keeps on-disk neighbours neighbours.
        Interleaving instead draws each row from N spans that the per-epoch permutation
        scattered across the rank's whole range, so consecutive emitted rows come from N
        widely separated places in the corpus.

        Reads stay sequential WITHIN each span, so the parquet access pattern is
        unchanged; the cost is N open files and N decode buffers instead of one
        (~4.5 MiB each at the default batch_size).

        1 restores plain sequential reading."""
        reshuffle_spans_per_epoch: bool = True
        """Permute the order of this rank's row-group spans on each pass.

        Without it a repeating dataset replays the IDENTICAL document order -- and
        therefore identical packing boundaries -- every epoch, which matters for any
        corpus seen more than once. Permuting spans is essentially free: it reorders a
        list of tens-to-hundreds of spans, not rows, and reads stay sequential WITHIN
        each span, so the parquet access pattern is unchanged.

        This is deliberately independent of `shuffle`: `shuffle=False` means "the corpus
        is already shuffled, do not shuffle rows", not "replay the same order forever"."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_iter_dataset(
        self, *, dataset_iteration_policy: DatasetIterationPolicy
    ) -> grain.IterDataset:
        """Upstream's StreamingDataSource: the stream of this rank's rows under
        `dataset_iteration_policy` (rank, world, seed, repeat)."""
        return _ParquetStream(
            self.config, dataset_iteration_policy=dataset_iteration_policy
        )


class _ParquetStream(grain.IterDataset):
    """This rank's row range of the corpus as a grain IterDataset."""

    def __init__(
        self,
        config: ParquetStreamSource.Config,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        super().__init__()
        self._dir = Path(config.path)
        manifest_path = (
            Path(config.manifest_path)
            if config.manifest_path
            else self._dir / MANIFEST_FILENAME
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No manifest at {manifest_path}. Build it with:\n"
                f"  python reproduce_cfgs/helper_scripts/"
                f"build_py_config_dataset_manifests.py <config.py>\n"
                "Refusing to fall back to per-rank directory discovery, which is what "
                "stalls a shared filesystem at scale."
            )
        self._manifest = read_manifest(manifest_path)

        rank = dataset_iteration_policy.dp_rank
        world = dataset_iteration_policy.dp_world_size
        if rank == 0:
            stale = check_freshness(self._manifest, self._dir)
            if stale:
                logger.warning("%s", stale)

        for column in config.columns:
            validate_column(self._manifest, column, self._dir)
        available = set(self._manifest.get("columns") or {})
        present_optional = [c for c in config.optional_columns if c in available]
        if config.optional_columns and rank == 0:
            missing = [c for c in config.optional_columns if c not in available]
            if missing:
                logger.info(
                    "%s: optional column(s) %s not in this corpus; skipping them.",
                    self._dir.name,
                    ", ".join(missing),
                )
        self._columns = (list(config.columns) + present_optional) or None
        self._batch_size = max(1, config.batch_size)

        self._spans: list[RowSpan] = assign_row_spans(self._manifest, rank, world)
        if not self._spans:
            # Upstream's "fewer rows than ranks" guard (dataset.py:148-152) never fires
            # for a streaming source. Without this, a corpus smaller than the DP degree
            # silently feeds zero rows to some ranks and the mixture proportions quietly
            # differ per rank.
            raise ValueError(
                f"{self._dir}: rank {rank} of {world} was assigned no rows. The dataset "
                f"has {self._manifest['num_rows']} rows, fewer than the {world} "
                "data-parallel ranks, so it cannot feed them all. Use a larger corpus, "
                "a smaller DP degree, or drop this dataset from the mix."
            )
        self._repeat = dataset_iteration_policy.repeat
        self._reshuffle_spans = config.reshuffle_spans_per_epoch
        self._num_concurrent_spans = max(1, config.num_concurrent_spans)
        self._seed = dataset_iteration_policy.seed + rank

        shared = sharing_factor(self._manifest, world)
        if rank == 0 and shared > 1:
            logger.warning(
                "%s: %d ranks share each row group at dp_world_size=%d, so each of them "
                "decompresses the whole group and keeps a slice. Correct, but the read "
                "amplification is real; a corpus with more row groups avoids it.",
                self._dir.name,
                shared,
                world,
            )

        # At high DP a rank's whole range can sit inside one row group, which would make
        # BOTH shuffle mechanisms no-ops (permuting a list of one, and a width capped at
        # the span count). Subdivide so the interleave has something to work with.
        if self._reshuffle_spans and len(self._spans) < config.num_concurrent_spans:
            split = split_spans_for_interleave(self._spans, config.num_concurrent_spans)
            if rank == 0 and len(split) != len(self._spans):
                logger.info(
                    "%s: rank share is %d span(s) at dp_world_size=%d; split into %d so "
                    "span permutation and interleaving still mix. Sub-spans of one row "
                    "group each decode from its start, so the group is decompressed "
                    "more than once per epoch.",
                    self._dir.name,
                    len(self._spans),
                    world,
                    len(split),
                )
            self._spans = split

        if rank == 0:
            logger.info(
                "%s: %d rows over %d shards; rank 0 streams %d rows from %d shard(s)",
                self._dir.name,
                self._manifest["num_rows"],
                self._manifest["num_shards"],
                sum(span.num_rows for span in self._spans),
                len({span.shard for span in self._spans}),
            )

    def __iter__(self) -> grain.DatasetIterator:
        return _ParquetSpanIterator(
            self._dir,
            self._spans,
            columns=self._columns,
            batch_size=self._batch_size,
            repeat=self._repeat,
            reshuffle_spans=self._reshuffle_spans,
            num_concurrent_spans=self._num_concurrent_spans,
            seed=self._seed,
        )


class _ParquetSpanIterator(grain.DatasetIterator):
    """Reads `num_concurrent_spans` spans at once, drawing one row at a time between
    them.

    The checkpoint cursor is plain integers -- which spans are open, how far into each,
    and how many draws have been made -- so the state is inspectable and diffable
    rather than an opaque library blob.
    """

    #: Draws are generated in blocks so a resume recomputes one block instead of
    #: replaying every draw made since the epoch began.
    _DRAW_BLOCK = 8192

    def __init__(
        self,
        dataset_dir: Path,
        spans: list[RowSpan],
        *,
        columns: list[str] | None,
        batch_size: int,
        repeat: bool,
        reshuffle_spans: bool = False,
        num_concurrent_spans: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._dir = dataset_dir
        self._base_spans = spans
        self._reshuffle_spans = reshuffle_spans
        self._seed = seed
        self._columns = columns
        self._batch_size = batch_size
        self._repeat = repeat
        self._width = max(1, min(num_concurrent_spans, len(spans)))
        # Rows in one full pass. Span COUNT and total rows are permutation-invariant, so
        # this is the same for every epoch.
        self._rows_per_epoch = sum(span.num_rows for span in self._base_spans)
        self._epoch = 0
        self._emitted = 0  # cumulative across epochs; drives data_docs/{alias}
        # `span_index` indexes a permutation of THIS span list. If the corpus gained or
        # lost shards and the manifest was rebuilt, the same integers name different
        # rows, and a resume silently re-reads some and skips others. `check_freshness`
        # cannot catch it -- rebuilding the manifest is the documented remedy, and doing
        # so re-records the directory mtime. So pin the shape the cursor was taken
        # against.
        self._fingerprint = [len(self._base_spans), self._rows_per_epoch]
        self._start_epoch(0)

    # -- span order -----------------------------------------------------------
    def _epoch_spans(self, epoch: int) -> list[RowSpan]:
        """Span order for one pass. Derived purely from (seed, epoch), so every rank
        reproduces it on resume without storing the permutation."""
        if not self._reshuffle_spans:
            return list(self._base_spans)
        # Epoch 0 is permuted too. Exempting it made the first pass the one pass whose
        # concurrent spans were adjacent on disk (slots take spans in order), so the
        # very epoch every short run consists of had the weakest mixing. Disk order has
        # no special status here -- the corpora are pre-shuffled -- and
        # reshuffle_spans_per_epoch=False remains the way to ask for it.
        spans = list(self._base_spans)
        # Multiply-and-add rather than hash((seed, epoch)): tuple hashing of ints is
        # stable today but is an implementation detail, and this is reproduced on every
        # resume.
        random.Random(self._seed * 1_000_003 + epoch).shuffle(spans)
        return spans

    def _start_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._spans = self._epoch_spans(epoch)
        self._next_span = 0
        self._draw_index = 0
        self._draws: list[int] = []
        self._draw_block = -1
        # Slot i holds (span_index, row_in_span) or None once its span is exhausted and
        # no span is left to replace it.
        self._slots: list[dict[str, int] | None] = [None] * self._width
        self._rows: list[Iterator[dict[str, Any]] | None] = [None] * self._width
        for slot in range(self._width):
            self._fill_slot(slot)

    def _fill_slot(self, slot: int) -> None:
        if self._next_span >= len(self._spans):
            self._slots[slot] = None
            self._rows[slot] = None
            return
        self._slots[slot] = {"span_index": self._next_span, "row_in_span": 0}
        self._rows[slot] = None
        self._next_span += 1

    # -- draws ----------------------------------------------------------------
    def _draw(self) -> int:
        """Slot to read from next. A pure function of (seed, epoch, draw_index), so a
        resume recomputes it rather than replaying the RNG."""
        block, offset = divmod(self._draw_index, self._DRAW_BLOCK)
        if block != self._draw_block:
            rng = random.Random(
                (self._seed * 1_000_003 + self._epoch) * 1_000_003 + block
            )
            self._draws = [rng.randrange(self._width) for _ in range(self._DRAW_BLOCK)]
            self._draw_block = block
        self._draw_index += 1
        return self._draws[offset]

    # -- reading --------------------------------------------------------------
    def _span_rows(self, span: RowSpan, skip: int) -> Iterator[dict[str, Any]]:
        """Stream one span, resuming `skip` rows in.

        `iter_batches` decodes incrementally, so peak memory is one batch rather than
        the whole row group -- the entire point of this source.
        """
        import pyarrow.parquet as pq

        try:
            parquet_file = pq.ParquetFile(self._dir / span.shard)
            batches = parquet_file.iter_batches(
                batch_size=self._batch_size,
                row_groups=[span.row_group],
                columns=self._columns,
            )
        except (FileNotFoundError, OSError, IndexError) as exc:
            raise RuntimeError(
                f"{self._dir}: failed to read row group {span.row_group} of "
                f"{span.shard} ({type(exc).__name__}: {exc}). The manifest is out of "
                "date with the shards on disk; rebuild it with "
                "build_py_config_dataset_manifests.py."
            ) from exc

        # `span` is a slice of the row group; `skip` resumes inside that slice.
        position = 0
        wanted_start, wanted_stop = span.start + skip, span.stop
        for batch in batches:
            batch_stop = position + batch.num_rows
            if batch_stop > wanted_start and position < wanted_stop:
                lo = max(wanted_start - position, 0)
                hi = min(wanted_stop - position, batch.num_rows)
                for row in batch.slice(lo, hi - lo).to_pylist():
                    yield row
            position = batch_stop
            if position >= wanted_stop:
                return

    def _active_slot(self) -> int | None:
        """The drawn slot, or the next live one after it. Scanning rather than
        re-drawing keeps the draw sequence independent of which slots are alive, so a
        resume reproduces it exactly."""
        if all(s is None for s in self._slots):
            return None
        start = self._draw()
        for step in range(self._width):
            slot = (start + step) % self._width
            if self._slots[slot] is not None:
                return slot
        return None

    def __next__(self) -> dict[str, Any]:
        while True:
            slot = self._active_slot()
            if slot is None:
                if not self._repeat:
                    raise StopIteration
                # Exhaustion policy: loop the dataset and keep its weight, so the
                # configured mixture ratio holds for the whole stage.
                self._start_epoch(self._epoch + 1)
                continue
            cursor = self._slots[slot]
            assert cursor is not None
            if self._rows[slot] is None:
                self._rows[slot] = self._span_rows(
                    self._spans[cursor["span_index"]], cursor["row_in_span"]
                )
            try:
                row = next(self._rows[slot])
            except StopIteration:
                self._fill_slot(slot)
                continue
            cursor["row_in_span"] += 1
            self._emitted += 1
            return row

    # -- checkpointing --------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "slots": [None if s is None else dict(s) for s in self._slots],
            "next_span": self._next_span,
            "draw_index": self._draw_index,
            # Documents emitted so far, CUMULATIVE across epochs, so per-dataset
            # consumption metrics need no separate accumulator. Counting within the
            # epoch instead would make `data_docs/{alias}` saw-tooth back to zero every
            # time a small corpus wrapped -- exactly the corpora that wrap most often.
            "next_index": self._emitted,
            "spans_fingerprint": list(self._fingerprint),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        # The span permutation and the draw sequence are pure functions of
        # (seed, epoch [, block]), so both are recomputed rather than stored.
        self._start_epoch(int(state.get("epoch", 0)))
        slots = state.get("slots")
        if slots is None:
            raise ValueError(
                "dataloader state predates concurrent-span reading and cannot be "
                "restored; start the stage fresh"
            )
        fingerprint = state.get("spans_fingerprint")
        if fingerprint is not None and list(fingerprint) != self._fingerprint:
            raise ValueError(
                f"checkpoint was taken against {fingerprint[0]} spans / "
                f"{fingerprint[1]} rows, but this run sees {self._fingerprint[0]} / "
                f"{self._fingerprint[1]}. The corpus changed since the checkpoint, so "
                "the saved cursor points at different rows; start this dataset fresh "
                "(rename its alias) rather than resuming into a shifted traversal."
            )
        if len(slots) != self._width:
            raise ValueError(
                f"checkpoint has {len(slots)} concurrent spans but this run reads "
                f"{self._width}; num_concurrent_spans must match to resume"
            )
        self._slots = [None if s is None else dict(s) for s in slots]
        self._rows = [None] * self._width  # reopened lazily at the restored positions
        self._next_span = int(state["next_span"])
        self._draw_index = int(state["draw_index"])
        self._emitted = int(state.get("next_index", 0))

logger = logging.getLogger(__name__)
