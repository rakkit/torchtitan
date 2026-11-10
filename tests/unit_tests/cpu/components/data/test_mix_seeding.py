# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A dataset's RNG must follow its ALIAS, not its position in the list.

Upstream seeds mix children by position (`seed=policy.seed + index`,
`dataset.py:236-240`, with its own comment: "Inserting or reordering a child reseeds
every later child"). That is the wrong rule for the stage workflow: stage 2 adds
datasets, and if an insertion re-randomizes every later dataset's span order, the
cursors restored for those datasets point into a different traversal than the one that
produced them. The cursor survives and the ORDER silently does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grain.python as grain
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torchtitan.components.data.dataset import WeightedDataset
from torchtitan.components.data.mix import (
    alias_seed,
    build_mix,
    DatasetSpec,
    KeyedDatasetMixConfig,
)
from torchtitan.components.data.parquet_manifest import (
    build_manifest,
    MANIFEST_FILENAME,
    write_manifest,
)
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy


class _Tokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        return [1] * add_bos + [ord(c) % 250 + 10 for c in text] + [2] * add_eos


CONTEXT = DatasetBuildContext(
    tokenizer=_Tokenizer(),
    max_context_length=32,
    num_tokens_per_microbatch=32,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)
POLICY = DatasetIterationPolicy(
    seed=7,
    shuffle=False,
    repeat=True,
    dp_rank=0,
    dp_world_size=1,
    streaming_shuffle_buffer_size=0,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class _SeedRecorder:
    """A dataset config that records the seed it is built with."""

    seen: list

    def build(self, *, context, dataset_iteration_policy) -> Any:
        del context
        self.seen.append(dataset_iteration_policy.seed)
        return grain.MapDataset.source([{"x": 0}]).to_iter_dataset()


def _mix(aliases: tuple[str, ...], seen: dict[str, list]) -> KeyedDatasetMixConfig:
    return KeyedDatasetMixConfig(
        datasets=tuple(
            WeightedDataset(dataset=_SeedRecorder(seen=seen.setdefault(a, [])))
            for a in aliases
        ),
        aliases=aliases,
        by_tokens=False,
    )


def test_a_datasets_seed_does_not_depend_on_its_position():
    """The property that makes cross-stage resume actually resume the same traversal."""
    stage1: dict[str, list] = {}
    _mix(("alpha", "beta"), stage1).build(
        context=CONTEXT, dataset_iteration_policy=POLICY
    )
    # Stage 2 inserts a dataset BEFORE the existing ones -- the worst case for
    # position-based seeding.
    stage2: dict[str, list] = {}
    _mix(("gamma", "alpha", "beta"), stage2).build(
        context=CONTEXT, dataset_iteration_policy=POLICY
    )

    assert stage1["alpha"] == stage2["alpha"]
    assert stage1["beta"] == stage2["beta"]


def test_distinct_aliases_get_distinct_seeds():
    """Otherwise every dataset would replay the same permutation."""
    seeds = {alias_seed(7, name) for name in ("alpha", "beta", "gamma", "delta")}
    assert len(seeds) == 4


def test_renaming_an_alias_changes_its_seed():
    """The documented rule: a rename restarts the cursor AND re-randomizes the order.
    One rule for both, rather than two fields that can disagree."""
    assert alias_seed(7, "alpha") != alias_seed(7, "alpha-v2")


def test_alias_seed_is_stable_across_processes():
    """It must be, or a resumed run would traverse differently than the run that saved.

    `hash()` on a str is salted per process (PYTHONHASHSEED), which is exactly the trap
    this avoids by using blake2b.
    """
    assert alias_seed(0, "High-Quality") == alias_seed(0, "High-Quality")
    assert alias_seed(0, "High-Quality") == 1753986520


def test_build_mix_carries_the_aliases(tmp_path: Path):
    """The seeding is only position-independent if the aliases reach the mix."""
    specs = []
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        pq.write_table(pa.table({"text": ["x"] * 4}), root / "p.parquet")
        write_manifest(build_manifest(root), root / MANIFEST_FILENAME)
        specs.append(DatasetSpec(alias=name, path=str(root)))

    mix = build_mix(specs, by_tokens=True)

    assert mix.aliases == ("a", "b")


# grain's IterDataset.mix picks the child of the k-th element from k and the weights only. Without a per-rank start,
# every DP rank draws the same child at every step: with one packed microbatch per step and per-dataset packing, a
# whole global batch came from ONE dataset (the 8B study's train loss jumped between dataset loss levels, 11-10).


@dataclass(frozen=True)
class _ConstantChild:
    child_id: int

    def build(self, *, context, dataset_iteration_policy):
        no_threads = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
        return (
            grain.MapDataset.range(10**9)
            .map(lambda _: self.child_id)
            .to_iter_dataset(read_options=no_threads)
        )


def _rank_iterator(
    weights: list[float], dp_rank: int, dp_world_size: int, stride: int = 1
):
    from types import SimpleNamespace

    from torchtitan.components.data.dataset import DatasetMixConfig

    mix = DatasetMixConfig(
        datasets=tuple(
            WeightedDataset(dataset=_ConstantChild(i), weight=w)
            for i, w in enumerate(weights)
        )
    )
    policy = DatasetIterationPolicy(
        seed=0,
        shuffle=False,
        repeat=True,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        streaming_shuffle_buffer_size=1,
    )
    context = SimpleNamespace(read_options=grain.ReadOptions(), mix_rank_stride=stride)
    return iter(mix.build(context=context, dataset_iteration_policy=policy))


def test_each_step_mixes_children_across_dp_ranks():
    weights = [0.4, 0.3, 0.2, 0.1]
    world = 64
    iterators = [_rank_iterator(weights, r, world) for r in range(world)]
    for _ in range(20):  # 20 steps, one element per rank per step
        step = [next(it) for it in iterators]
        for child, w in enumerate(weights):
            assert abs(step.count(child) - w * world) <= 2, (child, step.count(child))


def test_rank_offset_survives_resume():
    weights = [0.5, 0.3, 0.2]
    it = _rank_iterator(weights, dp_rank=5, dp_world_size=8)
    for _ in range(7):
        next(it)
    state = it.get_state()
    expected = [next(it) for _ in range(11)]
    resumed = _rank_iterator(weights, dp_rank=5, dp_world_size=8)
    resumed.set_state(state)
    assert [next(resumed) for _ in range(11)] == expected


@pytest.mark.parametrize("stride", [1, 4])
def test_small_children_follow_their_weight_over_ten_steps(stride):
    # 64 ranks x `stride` elements per microbatch (4 = per-dataset rows of 16k in a 64k
    # microbatch). The ranks' n-th microbatches take block n of the one sequence, so a
    # child due 1.3 elements per step gets 1 or 2 as due, and any 10 steps are within
    # ~1 element of 10 x its due. (Ranks walking consecutive positions from
    # rank * stride shared a window moving `stride` positions per step: 10-step totals
    # off by up to 12.8 elements for the 0.5% children.)
    weights = [30, 20, 12, 10, 8, 6, 4, 3, 2, 1.5, 1, 0.8, 0.7, 0.5, 0.5]
    world = 64
    due = [w / sum(weights) * world * stride for w in weights]
    iterators = [_rank_iterator(weights, r, world, stride) for r in range(world)]
    steps = []
    for _ in range(60):
        step = [next(it) for it in iterators for _ in range(stride)]
        steps.append([step.count(child) for child in range(len(weights))])
    for counts in steps:
        for child, count in enumerate(counts):
            assert abs(count - due[child]) < 1.5, (child, count, due[child])
    for start in range(0, 51, 10):
        for child in range(len(weights)):
            total = sum(counts[child] for counts in steps[start : start + 10])
            assert abs(total - 10 * due[child]) < 2, (start, child, total)
