# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The record-based dataset list, end to end through upstream's grain mix.

Two things are load-bearing and therefore tested against behaviour, not shape:

  * mixing several manifest-backed corpora produces samples from all of them, in the
    configured proportions, with each DP rank reading only its own rows;
  * `RankLocalDatasetConfig` prevents upstream's global-shuffle-then-slice from
    scattering a rank's rows across every shard -- the failure that would silently undo
    the manifest.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import grain.python as grain
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torchtitan.components.data.mix import build_mix, DatasetSpec

from torchtitan.components.data.parquet_manifest import (
    build_manifest,
    MANIFEST_FILENAME,
    write_manifest,
)
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = [ord(char) % 250 + 10 for char in text]
        return [self.bos_id] * add_bos + tokens + [self.eos_id] * add_eos


CONTEXT = DatasetBuildContext(
    tokenizer=FakeTokenizer(),
    max_context_length=9,
    num_tokens_per_microbatch=18,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


def _policy(dp_rank=0, dp_world_size=1, **overrides):
    base = DatasetIterationPolicy(
        seed=0,
        shuffle=False,
        repeat=False,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        streaming_shuffle_buffer_size=0,
    )
    return replace(base, **overrides) if overrides else base


def _corpus(root: Path, tag: str, rows: int, column: str = "text") -> Path:
    """A corpus whose rows are self-identifying, so provenance is checkable."""
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({column: [f"{tag}{i}" for i in range(rows)]}),
        root / "part-00000.parquet",
    )
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)
    return root


def _decode(sample) -> str:
    """Recover the source text from a TextSequence, to check which corpus it came from."""
    # FakeTokenizer maps char -> ord % 250 + 10, and TextProcessor drops the last token
    # from input_ids; the first character survives and is enough to identify the corpus.
    return chr((int(sample.input_ids[1]) - 10) % 250)


def test_mix_draws_from_every_dataset(tmp_path):
    a = _corpus(tmp_path / "a", "a", rows=64)
    b = _corpus(tmp_path / "b", "b", rows=64)

    mix = build_mix(
        [
            DatasetSpec(alias="a", path=str(a), weight=1.0),
            DatasetSpec(alias="b", path=str(b), weight=1.0),
        ],
        by_tokens=True,
    )
    dataset = mix.build(context=CONTEXT, dataset_iteration_policy=_policy())
    tags = [_decode(s) for s in list(dataset)[:40] if s is not None]

    assert set(tags) == {"a", "b"}, "both corpora must appear in the mixture"


def test_mix_respects_relative_weights(tmp_path):
    a = _corpus(tmp_path / "a", "a", rows=400)
    b = _corpus(tmp_path / "b", "b", rows=400)

    mix = build_mix(
        [
            DatasetSpec(alias="a", path=str(a), weight=3.0),
            DatasetSpec(alias="b", path=str(b), weight=1.0),
        ],
        by_tokens=True,
    )
    dataset = mix.build(context=CONTEXT, dataset_iteration_policy=_policy())
    tags = [_decode(s) for s in list(dataset)[:200] if s is not None]

    ratio = tags.count("a") / max(tags.count("b"), 1)
    assert 2.0 < ratio < 4.5, f"expected roughly 3:1, got {ratio:.2f}"


def test_each_dataset_can_use_a_different_column(tmp_path):
    """Production mixes `text` and `content` keys; a record carries its own."""
    a = _corpus(tmp_path / "a", "a", rows=32, column="text")
    b = _corpus(tmp_path / "b", "b", rows=32, column="content")

    mix = build_mix(
        [
            DatasetSpec(alias="a", path=str(a), text_key="text"),
            DatasetSpec(alias="b", path=str(b), text_key="content"),
        ],
        by_tokens=True,
    )
    dataset = mix.build(context=CONTEXT, dataset_iteration_policy=_policy())
    tags = [_decode(s) for s in list(dataset)[:40] if s is not None]
    assert set(tags) == {"a", "b"}


def test_a_wrong_column_fails_at_startup(tmp_path):
    a = _corpus(tmp_path / "a", "a", rows=8, column="text")
    mix = build_mix(
        [DatasetSpec(alias="a", path=str(a), text_key="content")], by_tokens=True
    )
    with pytest.raises(ValueError, match="not found"):
        mix.build(context=CONTEXT, dataset_iteration_policy=_policy())


def test_ranks_partition_the_mix_without_overlap(tmp_path):
    a = _corpus(tmp_path / "a", "a", rows=64)
    b = _corpus(tmp_path / "b", "b", rows=64)
    specs = [
        DatasetSpec(alias="a", path=str(a)),
        DatasetSpec(alias="b", path=str(b)),
    ]
    world = 4

    lengths = []
    for rank in range(world):
        mix = build_mix(specs, by_tokens=True)
        dataset = mix.build(
            context=CONTEXT,
            dataset_iteration_policy=_policy(dp_rank=rank, dp_world_size=world),
        )
        lengths.append(sum(1 for s in dataset if s is not None))

    assert all(n > 0 for n in lengths), "every rank must get data"
    # 128 rows over 4 ranks; the mix stops at its first exhausted child, so assert the
    # ranks are balanced rather than that they sum to the corpus.
    assert max(lengths) - min(lengths) <= 2, lengths


def test_streaming_source_is_not_resharded_by_upstream(tmp_path):
    """Replaces the old RankLocalDatasetConfig guard.

    A streaming source takes `_build_iter_dataset`, which does NOT shard -- the source
    already did. If it were treated as random-access, `_build_map_dataset` would shuffle
    globally and slice again, leaving the rank 1/16 of the data instead of 1/4.
    """
    from torchtitan.components.data.dataset import SingleDatasetConfig
    from torchtitan.components.data.parquet_stream import ParquetStreamSource
    from torchtitan.components.data.text_datasets import TextProcessor

    a = _corpus(tmp_path / "a", "a", rows=64)
    world = 4

    config = SingleDatasetConfig(
        source=ParquetStreamSource.Config(path=str(a), columns=("text",)),
        processor=TextProcessor.Config(),
    )
    got = sum(
        1
        for s in config.build(
            context=CONTEXT,
            dataset_iteration_policy=_policy(dp_rank=0, dp_world_size=world),
        )
        if s is not None
    )

    assert got == 64 // world, f"expected {64 // world} rows, got {got}"
