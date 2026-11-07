# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fixed validation slice: every rank count scores the same sequences."""

import pytest

from torchtitan.components.validator import slice_plan, Validator


def _covered(shards, sequences, microbatch, world):
    """(shard, sequence) pairs each rank scores, from its plan."""
    out = []
    for rank in range(world):
        mine = []
        for shard, skip, take in slice_plan(shards, sequences, microbatch, world, rank):
            first = skip * microbatch
            mine += [(shard, i) for i in range(first, first + take * microbatch)]
        out.append(mine)
    return out


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16, 32, 64, 128])
def test_every_world_scores_the_whole_slice_once(world):
    ranks = _covered(4, 1024, 32, world)
    want = [(s, i) for s in range(4) for i in range(1024)]
    assert [x for r in ranks for x in r] == want  # in order, each once
    assert len({len(r) for r in ranks}) == 1  # equal steps on every rank


def test_a_slice_that_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="evenly"):
        slice_plan(4, 1024, 32, 256, 0)  # 16 sequences per rank < one microbatch
    with pytest.raises(ValueError, match="evenly"):
        slice_plan(4, 1000, 32, 4, 0)


def test_config_takes_all_three_or_none():
    Validator.Config(slice_shards=4, slice_sequences=1024, slice_microbatch=32)
    with pytest.raises(ValueError, match="all"):
        Validator.Config(slice_shards=4, slice_sequences=1024)
    with pytest.raises(ValueError, match="steps"):
        Validator.Config(
            steps=12, slice_shards=4, slice_sequences=1024, slice_microbatch=32
        )


def test_the_slice_is_read_once_and_replayed():
    """The slice is the same at every validation: the second validation must not
    rebuild loaders and re-read (at 2k ranks a rank skipped up to ~1k sequences
    of its shard every time)."""
    from types import SimpleNamespace

    builds = []

    class Loader:
        def __init__(self, shard):
            self.items = [f"s{shard}-m{i}" for i in range(10)]

        def __iter__(self):
            return iter(self.items)

        def close(self):
            pass

    class DLConfig:
        def build(self, *, dp_rank, **kwargs):
            builds.append(dp_rank)
            return Loader(dp_rank)

    v = Validator.__new__(Validator)
    v.config = SimpleNamespace(slice_shards=2, slice_microbatch=1)
    v.plan = [(0, 3, 2), (1, 0, 1)]
    v.dl_config = DLConfig()
    v.tokenizer = None
    v.seq_len = 8
    first = list(v._slice_microbatches())
    second = list(v._slice_microbatches())
    assert first == second == ["s0-m3", "s0-m4", "s1-m0"]
    assert builds == [0, 1]
