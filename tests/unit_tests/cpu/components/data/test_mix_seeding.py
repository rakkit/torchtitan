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


def _rank_iterator(weights: list[float], dp_rank: int, dp_world_size: int):
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
    context = SimpleNamespace(read_options=grain.ReadOptions(), mix_rank_stride=1)
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
