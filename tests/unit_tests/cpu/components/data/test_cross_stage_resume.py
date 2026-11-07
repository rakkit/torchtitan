# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end cross-stage resume, on real parquet through the real loader.

This is the gate for the whole refactor. Stage 1 trains on A-E and checkpoints; stage 2
trains on A-I and must CONTINUE A-E without re-reading a single row, while F-I start
from zero. The rows are self-identifying so the assertion is about actual consumed data,
not about state dictionaries agreeing with themselves.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import grain.python as grain
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.keyed_loader import KeyedMixDataLoader
from torchtitan.components.data.keyed_mix_state import cursor_position, find_mix_node
from torchtitan.components.data.mix import (
    build_mix,
    DatasetSpec,
    make_pretrain_dataloader_config,
)
from torchtitan.components.data.parquet_manifest import (
    build_manifest,
    MANIFEST_FILENAME,
    write_manifest,
)


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        # Encode the row's numeric suffix so a consumed row is identifiable downstream.
        return (
            [self.bos_id] * add_bos
            + [ord(c) % 250 + 10 for c in text]
            + [self.eos_id] * add_eos
        )


def _corpus(root: Path, tag: str, rows: int) -> str:
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"text": [f"{tag}-{i:04d}" for i in range(rows)]}),
        root / "part-00000.parquet",
    )
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)
    return str(root)


def _loader(tmp_path: Path, ids: list[str], rows: int = 256):
    specs = [DatasetSpec(alias=i, path=_corpus(tmp_path / i, i, rows)) for i in ids]
    # Deliberately the REAL factory, so the graph carries a packer and the mix state is
    # nested exactly as it is in production. Hand-building a packer-less graph here is
    # what let a broken `state_dict()` pass this suite.
    config = make_pretrain_dataloader_config(
        specs,
        shuffle=False,
        repeat=True,
        seed=0,
        num_prefetch_microbatches=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )
    return KeyedMixDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        max_context_length=8,
        num_tokens_per_microbatch=16,
    )


def _rank_state(checkpoint: dict) -> dict:
    """The keyed state of rank 0: one opaque item in the checkpoint."""
    return pickle.loads(checkpoint["dp_rank_0"])


def _consume(loader, batches: int) -> int:
    iterator = iter(loader)
    for _ in range(batches):
        next(iterator)
    return batches


def test_stage_two_resumes_old_datasets_and_starts_new_ones_fresh(tmp_path):
    stage1_ids = ["A", "B", "C", "D", "E"]
    stage2_ids = [*stage1_ids, "F", "G", "H", "I"]

    stage1 = _loader(tmp_path, stage1_ids)
    _consume(stage1, batches=6)
    checkpoint = stage1.state_dict()
    stage1.close()

    # The checkpoint names its datasets -- that is the whole point.
    rank_state = _rank_state(checkpoint)
    assert set(rank_state["datasets"]) == set(stage1_ids)
    saved_cursors = {k: cursor_position(v) for k, v in rank_state["datasets"].items()}
    assert any(
        c > 0 for c in saved_cursors.values()
    ), "stage 1 must have read something"

    stage2 = _loader(tmp_path, stage2_ids)
    stage2.load_state_dict(checkpoint)
    resumed = stage2._iterator.get_state()
    stage2.close()

    cursors = dict(
        zip(
            stage2_ids,
            [cursor_position(p) for p in find_mix_node(resumed)["parents"]],
        )
    )
    for dataset_id in stage1_ids:
        assert cursors[dataset_id] == saved_cursors[dataset_id], (
            f"{dataset_id} must resume where stage 1 stopped, "
            f"got {cursors[dataset_id]} want {saved_cursors[dataset_id]}"
        )
    for dataset_id in ("F", "G", "H", "I"):
        assert cursors[dataset_id] == 0, f"{dataset_id} is new and must start fresh"


def test_reordering_between_stages_does_not_move_cursors(tmp_path):
    ids = ["A", "B", "C"]
    stage1 = _loader(tmp_path, ids)
    _consume(stage1, batches=5)
    checkpoint = stage1.state_dict()
    saved = {
        k: cursor_position(v) for k, v in _rank_state(checkpoint)["datasets"].items()
    }
    stage1.close()

    reordered = ["C", "A", "B"]
    stage2 = _loader(tmp_path, reordered)
    stage2.load_state_dict(checkpoint)
    cursors = dict(
        zip(
            reordered,
            [
                cursor_position(p)
                for p in find_mix_node(stage2._iterator.get_state())["parents"]
            ],
        )
    )
    stage2.close()

    assert cursors == saved, "a cursor must follow its dataset, not its position"


def test_same_config_resume_is_exact(tmp_path):
    ids = ["A", "B"]
    first = _loader(tmp_path, ids)
    _consume(first, batches=4)
    checkpoint = first.state_dict()
    before = first._iterator.get_state()
    first.close()

    second = _loader(tmp_path, ids)
    second.load_state_dict(checkpoint)
    after = second._iterator.get_state()
    second.close()

    assert after == before, "an unchanged config must restore bit-identically"


def test_legacy_state_version_is_refused(tmp_path):
    loader = _loader(tmp_path, ["A"])
    try:
        with pytest.raises(ValueError, match="unsupported dataloader state version"):
            loader.load_state_dict(
                {"version": 1, "dp_world_size": 1, "dp_rank_0": {"next_index": 3}}
            )
    finally:
        loader.close()


def test_dataset_ids_are_required(tmp_path):
    """Without ids the checkpoint cannot name its cursors; fail loudly at build."""
    specs = [DatasetSpec(alias="A", path=_corpus(tmp_path / "A", "A", 32))]
    config = KeyedMixDataLoader.Config(
        dataset=build_mix(specs),
        collator=TextCollator.Config(),
        dataset_ids=(),
        shuffle=False,
        repeat=True,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )
    with pytest.raises(ValueError, match="requires dataset_ids"):
        KeyedMixDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=FakeTokenizer(),
            max_context_length=8,
            num_tokens_per_microbatch=16,
        )


def test_per_dataset_document_counts_come_from_the_cursors(tmp_path):
    """Replaces 0.4.0's shared-memory num_docs_sampled, exactly and for free."""
    from torchtitan.components.data.keyed_mix_state import documents_consumed

    ids = ["A", "B"]
    loader = _loader(tmp_path, ids)
    _consume(loader, batches=6)

    metrics = loader.consumption_metrics()
    keyed = _rank_state(loader.state_dict())
    counts = documents_consumed(keyed)
    loader.close()

    assert set(counts) == {"A", "B"}
    assert sum(counts.values()) > 0, "something must have been consumed"
    # The counter is the cursor, so it agrees with the checkpoint by construction.
    for dataset_id, value in counts.items():
        assert value == cursor_position(keyed["datasets"][dataset_id])

    # Aliases label the series; "/" is a wandb namespace separator and must be scrubbed.
    assert "data_docs/A" in metrics and "data_docs/B" in metrics
    assert metrics["data_docs/A"] == counts["A"]


def test_consumption_counts_survive_a_resume(tmp_path):
    """The old counter lived in shared memory and went stale; this one cannot."""
    ids = ["A", "B"]
    first = _loader(tmp_path, ids)
    _consume(first, batches=5)
    checkpoint = first.state_dict()
    before = first.consumption_metrics()
    first.close()

    second = _loader(tmp_path, ids)
    second.load_state_dict(checkpoint)
    after = second.consumption_metrics()
    second.close()

    assert after == before, "resumed counts must match what was checkpointed"


def test_stage_two_resumes_through_dcp(tmp_path):
    """The checkpointer's real path: dcp.save / dcp.load of {"dataloader": loader}.
    DCP flattens nested dicts into one item per leaf and loads only the keys the
    LIVE state asks for, so a nested rank state cannot resume a stage that adds
    datasets, and a 44-dataset mixture made ~1,243 items per rank."""
    stage1_ids = ["A", "B", "C", "D", "E"]
    stage1 = _loader(tmp_path, stage1_ids)
    _consume(stage1, batches=6)
    saved = {
        k: cursor_position(v)
        for k, v in _rank_state(stage1.state_dict())["datasets"].items()
    }
    flat, _ = flatten_state_dict({"dataloader": stage1.state_dict()})
    assert len(flat) == 3, sorted(flat)  # version, dp_world_size, the rank's state
    checkpoint_id = str(tmp_path / "ckpt")
    dcp.save({"dataloader": stage1}, checkpoint_id=checkpoint_id, no_dist=True)
    stage1.close()

    stage2_ids = [*stage1_ids, "F", "G", "H", "I"]
    stage2 = _loader(tmp_path, stage2_ids)
    dcp.load({"dataloader": stage2}, checkpoint_id=checkpoint_id, no_dist=True)
    resumed = find_mix_node(stage2._iterator.get_state())["parents"]
    stage2.close()
    cursors = dict(zip(stage2_ids, [cursor_position(p) for p in resumed]))
    assert {k: cursors[k] for k in stage1_ids} == saved
    assert all(cursors[k] == 0 for k in ("F", "G", "H", "I"))
