# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-dataset metrics on the grain path, and the trainer wiring that reaches them.

The 0.4.0 scheduler reached into `dataloader.dataset` to
mutate a live weights tensor. A grain loader has no such attribute, so the trainer used
to die at construction with `'KeyedMixDataLoader' object has no attribute 'dataset'` --
after the dataloader had already been built, i.e. only ever at run time on a real
config. These tests keep that path exercised from CPU unit tests.

The reporting half is worth keeping: `data_docs/{alias}` has no upstream equivalent and
is the only signal that the mixture is actually being consumed in the configured
proportions.
"""

from __future__ import annotations

from pathlib import Path

import grain.python as grain
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torchtitan.components.data.keyed_loader import KeyedMixDataLoader
from torchtitan.components.data.mix import DatasetSpec, make_pretrain_dataloader_config
from torchtitan.components.data.parquet_manifest import (
    build_manifest,
    MANIFEST_FILENAME,
    write_manifest,
)
from torchtitan.observability.data_consumption import (
    build_data_mix_metrics,
    DataMixMetrics,
)


class _Tokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        return [1] * add_bos + [ord(c) % 250 + 10 for c in text] + [2] * add_eos


def _loader(tmp_path: Path) -> KeyedMixDataLoader:
    specs = []
    for name, weight in (("High-Quality", 3.0), ("v1-Code", 1.0)):
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"text": [f"{name}-{i}" for i in range(64)]}), root / "p.parquet"
        )
        write_manifest(build_manifest(root), root / MANIFEST_FILENAME)
        specs.append(DatasetSpec(alias=name, path=str(root), weight=weight))

    config = make_pretrain_dataloader_config(
        specs,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        num_prefetch_microbatches=1,
    )
    return KeyedMixDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=_Tokenizer(),
        max_context_length=16,
        num_tokens_per_microbatch=32,
    )


def test_the_trainer_can_build_metrics_for_a_grain_loader(tmp_path: Path):
    """The regression: this used to raise AttributeError at trainer construction."""
    loader = _loader(tmp_path)
    try:
        metrics = build_data_mix_metrics(loader)
        assert isinstance(metrics, DataMixMetrics)
        assert metrics.get_log_dict_at_step(0)[0]
    finally:
        loader.close()


def test_mixing_weights_are_reported_normalized(tmp_path: Path):
    """3:1 configured must read as 0.75/0.25, matching 0.4.0's `data_mixing/{alias}`."""
    loader = _loader(tmp_path)
    try:
        weights = loader.mixing_weights()
    finally:
        loader.close()

    assert weights == {
        "data_mixing/High-Quality": 0.75,
        "data_mixing/v1-Code": 0.25,
    }


def test_documents_consumed_is_reported_per_dataset(tmp_path: Path):
    """`data_docs/{alias}`, straight off the cursors -- no separate accumulator."""
    loader = _loader(tmp_path)
    try:
        iterator = iter(loader)
        for _ in range(4):
            next(iterator)
        metrics = build_data_mix_metrics(loader)
        _, data_docs = metrics.get_log_dict_at_step(4)
    finally:
        loader.close()

    # the token-weighted mix (concat_then_split) adds its own token counts per dataset
    assert set(data_docs) == {
        "data_docs/High-Quality",
        "data_docs/v1-Code",
        "data_tokens/High-Quality",
        "data_tokens/v1-Code",
    }
    assert sum(int(v) for k, v in data_docs.items() if k.startswith("data_docs/")) > 0
    assert len(metrics.get_log_dict_at_step(4)) == 2


def test_document_counts_are_not_silently_zero_under_a_window_shuffle(tmp_path: Path):
    """An undecodable cursor must RAISE, not report zero documents.

    `WindowShuffleIterDataset` names its parent state `parent_window_start_state`, so a
    walk that only follows `parent` falls off the end. Returning 0 there is
    indistinguishable from "this dataset has read nothing", which is how every
    `data_docs/{alias}` series read zero for the entire SFT path without anyone noticing.
    """
    from torchtitan.components.data.keyed_mix_state import cursor_position

    windowed = {
        "parent_window_start_state": {"parent": {"next_index": 17}},
        "window_size": 1000,
    }
    assert cursor_position(windowed) == 17

    with pytest.raises(ValueError, match="no 'next_index' found"):
        cursor_position({"some_future_transform_state": {"next_index": 5}})


def test_aliases_that_collide_after_metric_normalization_are_refused(tmp_path: Path):
    """ "/" is a wandb namespace separator, so aliases are normalized to "_" for metrics.

    Two aliases differing only by a slash would otherwise pass the uniqueness check and
    then collapse into one series, where the survivor reports the OTHER dataset's weight.
    Production aliases come from paths like `PDF-HQ-topic-split/*`, so this is reachable.
    """
    from torchtitan.components.data.mix import build_mix, DatasetSpec

    with pytest.raises(ValueError, match="both normalize to"):
        build_mix(
            [
                DatasetSpec(alias="PDF-HQ/law", path="/nonexistent", weight=3.0),
                DatasetSpec(alias="PDF-HQ_law", path="/nonexistent", weight=1.0),
            ],
            by_tokens=True,
        )
