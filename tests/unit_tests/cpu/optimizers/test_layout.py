# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run-list algebra and the layout vocabulary, no process group needed."""

import pytest
import torch

from torchtitan.models.common.param_layout import by_rank, KINDS, named
from torchtitan.optimizers.disco.layout import (
    matrix_view,
    ParamLayout,
    resolve_units,
    RunList,
)


def test_runlist_rejects_overlap_and_empty_runs():
    with pytest.raises(ValueError):
        RunList(((0, 4), (2, 3)))
    with pytest.raises(ValueError):
        RunList(((0, 0),))
    assert RunList.contiguous(5, 0).runs == ()


def test_merged_joins_adjacent_runs():
    assert RunList.merged([(4, 2), (0, 4), (6, 1), (10, 2)]).runs == ((0, 7), (10, 2))


def test_intersect_splits_at_both_boundaries():
    a = RunList(((0, 10), (20, 5)))
    b = RunList(((5, 10), (22, 10)))
    assert a.intersect(b).runs == ((5, 5), (22, 3))
    assert a.intersect(RunList(())).runs == ()


def test_positions_within_enumerates_the_container():
    # A rank holds rows [0,4) and [8,12) of a [2,F=8,D] stacked param under
    # Shard(1); unit w3 is rows [8,16). The intersection sits at local rows 4..7.
    local = RunList(((0, 4), (8, 4)))
    unit = RunList.contiguous(8, 8)
    inter = local.intersect(unit)
    assert inter.runs == ((8, 4),)
    assert inter.positions_within(local).runs == ((4, 4),)
    assert inter.positions_within(unit).runs == ((0, 4),)


def test_positions_within_refuses_straddling_runs():
    with pytest.raises(ValueError):
        RunList.contiguous(2, 6).positions_within(RunList(((0, 4), (8, 4))))


def test_plain_layouts_by_rank_and_the_kind_vocabulary():
    assert by_rank(torch.Size([])).kind == "scalar"
    assert by_rank(torch.Size([1])).kind == "scalar"
    assert by_rank(torch.Size([7])).kind == "vector"
    assert by_rank(torch.Size([3, 4])).kind == "matrix"
    with pytest.raises(ValueError, match="explicit layout"):
        by_rank(torch.Size([2, 3, 4]))  # rank-3+ is never a plain parameter
    assert {"embedding", "unembedding", "router", "expert_batch"} <= set(KINDS)
    with pytest.raises(ValueError, match="unknown layout kind"):
        ParamLayout("embed")


def test_named_finds_a_parameter_by_identity():
    m = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 4))
    assert named(m, m[1].weight) == "1.weight"
    with pytest.raises(ValueError):
        named(m[0], m[1].weight)


def test_expert_batch_units_are_one_matrix_per_expert():
    units = resolve_units("w1", torch.Size([4, 32, 64]), ParamLayout("expert_batch"))
    assert [u.name for u in units] == ["0", "1", "2", "3"]
    assert units[2].rows.runs == ((64, 32),)
    assert units[2].shape == (32, 64)


def test_matrix_view_flattens_the_leading_dims():
    assert matrix_view(torch.Size([2, 8, 5])) == (16, 5)
    assert matrix_view(torch.Size([7])) == (1, 7)
