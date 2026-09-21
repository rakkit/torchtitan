# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The factory a training config calls, and the validator's `replace` contract.

`Validator.__init__` does `replace(config.dataloader, repeat=...)` (validate.py:146) and
then `.build(...)`. That is easy to break silently -- a factory that returned something
non-`replace`-able, or that dropped `dataset_ids` on replace, would fail only when
validation first runs. Both are pinned here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torchtitan.components.data.keyed_loader import KeyedMixDataLoader
from torchtitan.components.data.mix import DatasetSpec, make_pretrain_dataloader_config
from torchtitan.components.data.packing import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
)
from torchtitan.components.data.parquet_manifest import (
    build_manifest,
    MANIFEST_FILENAME,
    write_manifest,
)


def _corpus(root: Path, tag: str, rows: int = 32) -> str:
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"text": [f"{tag}{i}" for i in range(rows)]}), root / "p.parquet"
    )
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)
    return str(root)


def _specs(tmp_path: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            alias="High-Quality", path=_corpus(tmp_path / "hq", "a"), weight=12.1
        ),
        DatasetSpec(
            alias="v1-Code",
            path=_corpus(tmp_path / "code", "b"),
            weight=7.76,
            text_key="text",
        ),
    ]


def test_factory_defaults_to_the_lossless_packer(tmp_path):
    """concat_then_split, because first_fit drops oversized documents (60.7% of tokens
    on a real long-document corpus at 8192)."""
    config = make_pretrain_dataloader_config(_specs(tmp_path))
    assert isinstance(config.dataset, ConcatThenSplitPackingConfig)


def test_factory_can_select_first_fit_with_more_bins(tmp_path):
    config = make_pretrain_dataloader_config(
        _specs(tmp_path), packing="first_fit", num_packing_bins=32
    )
    assert isinstance(config.dataset, FirstFitPackingConfig)
    assert config.dataset.num_packing_bins == 32


def test_unknown_packing_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown packing"):
        make_pretrain_dataloader_config(_specs(tmp_path), packing="best_fit_maybe")


def test_dataset_ids_are_carried_for_checkpoint_keying(tmp_path):
    config = make_pretrain_dataloader_config(_specs(tmp_path))
    assert config.dataset_ids == ("High-Quality", "v1-Code")


def test_validator_replace_preserves_dataset_ids(tmp_path):
    """Validator does replace(config.dataloader, repeat=...); ids must survive it."""
    config = make_pretrain_dataloader_config(_specs(tmp_path))

    validator_config = replace(config, repeat=False)

    assert validator_config.dataset_ids == ("High-Quality", "v1-Code")
    assert validator_config.repeat is False
    assert validator_config.dataset is config.dataset


def test_validator_config_builds_and_iterates(tmp_path):
    """The end the validator actually exercises: repeat=False at dp_world_size=1."""
    import grain.python as grain

    class FakeTokenizer:
        bos_id = 1
        eos_id = 2

        def encode(self, text, add_bos=False, add_eos=False):
            return [1] * add_bos + [ord(c) % 250 + 10 for c in text] + [2] * add_eos

    config = replace(
        make_pretrain_dataloader_config(
            _specs(tmp_path),
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
            num_prefetch_microbatches=1,
        ),
        repeat=False,
    )
    loader = KeyedMixDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        max_context_length=16,
        num_tokens_per_microbatch=32,
    )
    try:
        # #4773 replaced the batch dict with TokenizedTrainingMicrobatch.
        batch = next(iter(loader))
        assert batch.input.numel() == 32
        assert batch.labels is not None
        assert batch.num_valid_tokens > 0
    finally:
        loader.close()


def test_alias_is_the_checkpoint_key(tmp_path):
    """One field, two jobs: the metric label and the checkpoint key are the same string.

    An earlier draft carried both `id` and `alias`, which meant two near-identical
    strings on every dataset. Neither exists upstream, so there was no compatibility
    reason to keep both.
    """
    specs = _specs(tmp_path)
    config = make_pretrain_dataloader_config(specs)
    assert config.dataset_ids == tuple(d.alias for d in specs)


def test_trainer_must_not_forward_seed_into_a_grain_dataloader(tmp_path):
    """The startup crash this pins is total, not subtle.

    `Configurable.build()` raises when a build kwarg names a config field
    (configurable.py:152-158), and every grain loader owns `seed` as a CONFIG field.
    trainer.py therefore passes no seed at all and the loader reads `dataloader.seed`.
    Forwarding `--debug.seed` here, as the trainer once did, made every config built by
    this factory fail at startup.
    """
    config = make_pretrain_dataloader_config(_specs(tmp_path))
    with pytest.raises(ValueError, match="overlap with config fields"):
        config.build(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=None,
            max_context_length=16,
            num_tokens_per_microbatch=32,
            seed=42,
        )
    # The kwargs trainer.py actually passes must all be absent from the config.
    field_names = {f.name for f in dataclasses.fields(config)}
    trainer_kwargs = {
        "dp_world_size",
        "dp_rank",
        "tokenizer",
        "max_context_length",
        "num_tokens_per_batch",
    }
    assert not (trainer_kwargs & field_names), trainer_kwargs & field_names


def test_read_in_order_pins_the_source_to_disk_order(tmp_path):
    """A validator must score the SAME held-out tokens on every run.

    Without this, a change to the TRAINING path moves val loss because the validator
    sampled different tokens -- which is exactly how an A/B in this migration reported a
    spurious +0.027 that vanished once both arms scored the same tokens.
    """
    pinned = make_pretrain_dataloader_config(_specs(tmp_path), read_in_order=True)
    default = make_pretrain_dataloader_config(_specs(tmp_path))

    for child in pinned.dataset.dataset.datasets:
        source = child.dataset.source
        assert source.num_concurrent_spans == 1
        assert source.reshuffle_spans_per_epoch is False

    # The training default is the opposite: interleave and reshuffle per epoch.
    for child in default.dataset.dataset.datasets:
        source = child.dataset.source
        assert source.num_concurrent_spans > 1
        assert source.reshuffle_spans_per_epoch is True
