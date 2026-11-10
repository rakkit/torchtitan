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
import torch
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


PACKINGS = ["concat_then_split", "concat_then_split_per_dataset"]


def _corpus(root: Path, tag: str, rows: int) -> str:
    return _corpus_of(root, [f"{tag}-{i:04d}" for i in range(rows)])


def _corpus_of(root: Path, texts: list[str]) -> str:
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": texts}), root / "part-00000.parquet")
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)
    return str(root)


def _loader(
    tmp_path: Path,
    ids: list[str],
    rows: int = 256,
    packing: str = "concat_then_split",
):
    specs = [DatasetSpec(alias=i, path=_corpus(tmp_path / i, i, rows)) for i in ids]
    return _build(specs, packing)


def _build(
    specs: list[DatasetSpec],
    packing: str,
    *,
    dp_world_size: int = 1,
    dp_rank: int = 0,
):
    # Deliberately the REAL factory, so the graph carries a packer and the mix state is
    # nested exactly as it is in production. Hand-building a packer-less graph here is
    # what let a broken `state_dict()` pass this suite.
    config = make_pretrain_dataloader_config(
        specs,
        packing=packing,
        shuffle=False,
        repeat=True,
        seed=0,
        num_prefetch_microbatches=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )
    return KeyedMixDataLoader(
        config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
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


@pytest.mark.parametrize("packing", PACKINGS)
def test_stage_two_resumes_old_datasets_and_starts_new_ones_fresh(tmp_path, packing):
    stage1_ids = ["A", "B", "C", "D", "E"]
    stage2_ids = [*stage1_ids, "F", "G", "H", "I"]

    stage1 = _loader(tmp_path, stage1_ids, packing=packing)
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

    stage2 = _loader(tmp_path, stage2_ids, packing=packing)
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


@pytest.mark.parametrize("packing", PACKINGS)
def test_same_config_resume_is_exact(tmp_path, packing):
    ids = ["A", "B"]
    first = _loader(tmp_path, ids, packing=packing)
    iterator = iter(first)
    for _ in range(4):
        next(iterator)
    checkpoint = first.state_dict()
    before = first._iterator.get_state()
    expected = [next(iterator).input.clone() for _ in range(5)]
    first.close()

    second = _loader(tmp_path, ids, packing=packing)
    second.load_state_dict(checkpoint)
    after = second._iterator.get_state()
    iterator = iter(second)
    resumed = [next(iterator).input.clone() for _ in range(5)]
    second.close()

    assert after == before, "an unchanged config must restore bit-identically"
    assert all(
        (a == b).all() for a, b in zip(resumed, expected, strict=True)
    ), "the resumed loader must yield the batches the uninterrupted one did"


def test_both_packings_make_weights_count_tokens(tmp_path):
    # equal weights, documents of 2 vs 40 characters: a document-count mix would give the
    # long corpus ~95% of the tokens; both packings give each corpus ~half
    short_id, long_id = 10 + ord("s") % 250, 10 + ord("L") % 250
    specs = [
        DatasetSpec(alias="short", path=_corpus_of(tmp_path / "short", ["ss"] * 4000)),
        DatasetSpec(alias="long", path=_corpus_of(tmp_path / "long", ["L" * 40] * 400)),
    ]

    def microbatches(packing: str) -> list[torch.Tensor]:
        loader = _build(specs, packing)
        iterator = iter(loader)
        out = [next(iterator).input.flatten().clone() for _ in range(400)]
        loader.close()
        return out

    # documents mixed by token weight: half of ALL tokens each (each document's BOS
    # included), so the letters split (2/3) / (2/3 + 40/41) = 0.406 : 0.594
    tokens = torch.cat(microbatches("concat_then_split"))
    num_short = int((tokens == short_id).sum())
    assert 0.39 < num_short / (num_short + int((tokens == long_id).sum())) < 0.42

    # packed per corpus: every row (max_context_length 8) is one corpus, half of the
    # rows each, and a microbatch (16 tokens = 2 rows) can hold both corpora
    rows = [
        row
        for mb in microbatches("concat_then_split_per_dataset")
        for row in mb.split(8)
    ]
    is_short = [bool((row == short_id).any()) for row in rows]
    assert not any(s and bool((row == long_id).any()) for s, row in zip(is_short, rows))
    assert 0.4 < sum(is_short) / len(rows) < 0.6


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
        dataset=build_mix(specs, by_tokens=True),
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


def _rank_loader(tmp_path: Path, ids: list[str], rank: int, world: int):
    specs = [
        DatasetSpec(alias=i, path=str(tmp_path / i))
        if (tmp_path / i).exists()
        else DatasetSpec(alias=i, path=_corpus(tmp_path / i, i, 256))
        for i in ids
    ]
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
        dp_world_size=world,
        dp_rank=rank,
        tokenizer=FakeTokenizer(),
        max_context_length=8,
        num_tokens_per_microbatch=16,
    )


def _load_like_the_checkpointer(loader, checkpoint_id: str) -> None:
    from torchtitan.components.checkpointer import DATALOADER
    from torchtitan.components.checkpointer.dcp import CheckpointManager

    checkpointer = CheckpointManager.__new__(CheckpointManager)
    checkpointer._load_checkpoint(
        {DATALOADER: loader}, checkpoint_id, from_hf=False, from_quantized=False
    )


def test_a_dp_degree_change_redeals_every_old_rank_through_the_checkpointer(tmp_path):
    """3 data-parallel ranks checkpoint, the job resumes on 2: each new rank loads
    all 3 old rank items and reads its share of the rows the old ranks left
    unread in the pass; a same-degree resume still reads only its own item."""
    from torchtitan.components.data.keyed_mix_state import stream_state

    ids = ["A", "B"]
    merged, unread = {}, {alias: 0 for alias in ids}
    for rank, batches in enumerate((3, 7, 5)):
        loader = _rank_loader(tmp_path, ids, rank, 3)
        _consume(loader, batches)
        state = loader.state_dict()
        merged.update(state)
        for alias, cursor in _rank_state_of(state, rank)["datasets"].items():
            stream = stream_state(cursor)
            assert stream["epoch"] == 0
            unread[alias] += stream["spans_fingerprint"][1] - stream["next_index"]
        loader.close()
    checkpoint_id = str(tmp_path / "ckpt")
    dcp.save({"dataloader": merged}, checkpoint_id=checkpoint_id, no_dist=True)

    remainder = {alias: 0 for alias in ids}
    for rank in range(2):
        loader = _rank_loader(tmp_path, ids, rank, 2)
        _load_like_the_checkpointer(loader, checkpoint_id)
        assert set(loader.state_dict()) == {
            "version",
            "dp_world_size",
            f"dp_rank_{rank}",
        }
        parents = find_mix_node(loader._iterator.get_state())["parents"]
        for alias, cursor in zip(ids, parents):
            stream = stream_state(cursor)
            remainder[alias] += sum(
                stop - start for _, _, start, stop in stream["remainder"]
            )
        _consume(loader, 2)
        loader.close()
    assert remainder == unread

    same = _rank_loader(tmp_path, ids, 1, 3)
    same.request_saved_ranks(["dp_rank_0", "dp_rank_1", "dp_rank_2"])
    assert set(same.state_dict()) == {"version", "dp_world_size", "dp_rank_1"}
    _load_like_the_checkpointer(same, checkpoint_id)
    same.close()


def _rank_state_of(checkpoint: dict, rank: int) -> dict:
    return pickle.loads(checkpoint[f"dp_rank_{rank}"])


@pytest.mark.parametrize("packing", ["concat_then_split_per_dataset"])
def test_rank_start_survives_stage_switch_and_dp_change(tmp_path, packing):
    # Each DP rank starts its walk through the mix at dp_rank * stride (stride = rows per
    # microbatch for per-dataset rows), so the ranks of a step draw different datasets. A
    # restore that resets the mix index to 0 puts every rank on the same dataset again: the
    # stage switch (changed dataset list) and the DP change must keep each rank's own start.
    stride = 2  # 16 tokens / 8 per row
    specs = [DatasetSpec(alias=i, path=_corpus(tmp_path / i, i, 256)) for i in "ABC"]
    checkpoint = {}
    for rank in range(4):
        loader = _build(specs, packing, dp_world_size=4, dp_rank=rank)
        assert find_mix_node(loader._iterator.get_state())["index"] == rank * stride
        _consume(loader, batches=3)
        checkpoint.update(loader.state_dict())
        loader.close()

    def index_after_load(new_specs, world, rank, state):
        loader = _build(new_specs, packing, dp_world_size=world, dp_rank=rank)
        loader.load_state_dict(state)
        index = find_mix_node(loader._iterator.get_state())["index"]
        loader.close()
        return index

    # same datasets, same DP degree: the exact resume continues from the saved index
    saved = find_mix_node(_rank_state_of(checkpoint, 3)["iterator"])["index"]
    assert saved > 3 * stride
    assert index_after_load(specs, 4, 3, checkpoint) == saved
    # stage switch (a dataset added): the mix restarts at this rank's own start
    staged = [*specs, DatasetSpec(alias="D", path=_corpus(tmp_path / "D", "D", 256))]
    assert index_after_load(staged, 4, 3, checkpoint) == 3 * stride
    # DP change 4 -> 2: the same, at the new rank's start
    assert index_after_load(specs, 2, 1, checkpoint) == 1 * stride


def test_token_counts_resume_exactly_and_restart_on_a_stage_switch(tmp_path):
    # The token-weighted mix keeps one token count per dataset next to the cursors. An
    # exact resume restores them; a stage switch (changed dataset list) or a DP change
    # starts this rank's counts at 0 for the new mix while the cursors carry over.
    specs = [DatasetSpec(alias=i, path=_corpus(tmp_path / i, i, 256)) for i in "ABC"]
    checkpoint = {}
    for rank in range(2):
        loader = _build(specs, "concat_then_split", dp_world_size=2, dp_rank=rank)
        _consume(loader, batches=5)
        checkpoint.update(loader.state_dict())
        loader.close()
    saved = find_mix_node(_rank_state_of(checkpoint, 1)["iterator"])["tokens"]
    assert all(t > 0 for t in saved)

    def tokens_after_load(new_specs, world, rank):
        loader = _build(
            new_specs, "concat_then_split", dp_world_size=world, dp_rank=rank
        )
        loader.load_state_dict(checkpoint)
        tokens = find_mix_node(loader._iterator.get_state())["tokens"]
        loader.close()
        return tokens

    assert tokens_after_load(specs, 2, 1) == saved
    staged = [*specs, DatasetSpec(alias="D", path=_corpus(tmp_path / "D", "D", 256))]
    assert tokens_after_load(staged, 2, 1) == [0, 0, 0, 0]
    assert tokens_after_load(specs, 1, 0) == [0, 0, 0]
