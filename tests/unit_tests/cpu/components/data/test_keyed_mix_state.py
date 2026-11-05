# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-stage resume: the contract the env-var mapping used to approximate.

Stage 1 trains on A-E, stage 2 on A-I. A-E must resume exactly; F-I must start fresh.
The adversarial cases matter as much as the happy path, because each one is something
the positional/env-var scheme gets silently wrong: reordering the list, renaming an
alias, moving a path, removing a dataset.

Every state here is NESTED, because that is what a real loader produces. An earlier
version of these tests handed in a flat `{"parents": [...]}`, which no shipped
configuration ever emits -- the packer alone wraps the mix one level deeper -- so the
suite passed while `state_dict()` raised on the first checkpoint of any packed run.
"""

from __future__ import annotations

import pytest

from torchtitan.components.data.keyed_mix_state import (
    find_mix_node,
    from_keyed,
    KEYED_SCHEMA,
    to_keyed,
)


def _loader_state(cursors: list[int], index: int = 0) -> dict:
    """The shape the shipped graph produces: mix -> packer -> batch -> prefetch."""
    return {
        "remainder_slices": {"input_ids": (0, 0)},
        "elements_from_buffer_after_checkpoint": 1,
        "parent_state": {
            "parents": [
                {"index_for_rng": c, "parent": {"next_index": c}} for c in cursors
            ],
            "index": index,
            "stop": False,
        },
    }


def _cursors(state: dict) -> list[int]:
    return [p["parent"]["next_index"] for p in find_mix_node(state)["parents"]]


def _mix_index(state: dict) -> int:
    return find_mix_node(state)["index"]


def test_the_mix_node_is_found_below_the_packer():
    """The bug this whole module tripped on: `parents` is not at the top level."""
    state = _loader_state([1, 2])

    assert "parents" not in state
    assert find_mix_node(state) is state["parent_state"]


def test_round_trip_is_identity_when_nothing_changes():
    ids = ["a", "b", "c"]
    original = _loader_state([5, 7, 9], index=21)

    restored, plan = from_keyed(to_keyed(original, ids), ids)

    assert restored == original, "an unchanged config must resume the state verbatim"
    assert plan.restored == ("a", "b", "c")
    assert plan.fresh == () and plan.orphaned == ()


def test_same_config_resume_keeps_the_packers_partial_document():
    """The reason the raw state is stored alongside the keyed view.

    Per-dataset cursors cannot express "600 tokens into a 1000-token document"; that
    lives in the packer's `remainder_slices`. Rebuilding from cursors alone would silently
    re-read the document from its start.
    """
    ids = ["a"]
    original = _loader_state([5])
    original["remainder_slices"] = {"input_ids": (600, 1001)}

    restored, _ = from_keyed(to_keyed(original, ids), ids)

    assert restored["remainder_slices"] == {"input_ids": (600, 1001)}


def test_stage_one_to_stage_two_resumes_old_and_starts_new_fresh():
    """The actual use case: A-E continue, F-I begin at zero."""
    stage1_ids = ["A", "B", "C", "D", "E"]
    keyed = to_keyed(_loader_state([10, 20, 30, 40, 50], index=150), stage1_ids)

    stage2_ids = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    restored, plan = from_keyed(keyed, stage2_ids, _loader_state([0] * 9))

    assert _cursors(restored) == [10, 20, 30, 40, 50, 0, 0, 0, 0]
    assert plan.restored == tuple(stage1_ids)
    assert plan.fresh == ("F", "G", "H", "I")
    assert plan.orphaned == ()


def test_reordering_the_dataset_list_is_a_no_op():
    """Positional state gets this silently wrong; keyed state cannot."""
    keyed = to_keyed(_loader_state([1, 2, 3]), ["a", "b", "c"])

    restored, plan = from_keyed(keyed, ["c", "a", "b"], _loader_state([0, 0, 0]))

    assert _cursors(restored) == [3, 1, 2], "cursors must follow their dataset"
    assert plan.fresh == () and plan.orphaned == ()


def test_inserting_in_the_middle_does_not_shift_later_datasets():
    """The failure the env-var mapping existed to hand-patch."""
    keyed = to_keyed(_loader_state([11, 22, 33]), ["a", "b", "c"])

    restored, plan = from_keyed(keyed, ["a", "new", "b", "c"], _loader_state([0] * 4))

    assert _cursors(restored) == [11, 0, 22, 33]
    assert plan.fresh == ("new",)


def test_removing_a_dataset_reports_it_as_orphaned():
    keyed = to_keyed(_loader_state([1, 2, 3]), ["a", "b", "c"])

    restored, plan = from_keyed(keyed, ["a", "c"], _loader_state([0, 0]))

    assert _cursors(restored) == [1, 3]
    assert plan.orphaned == ("b",)


def test_alias_is_independent_of_path():
    """Moving a corpus between clusters must not disturb resume.

    That broke the old (key, alias, path) identity matching; the alias is the only thing
    that travels in the checkpoint.
    """
    keyed = to_keyed(_loader_state([7, 8]), ["High-Quality", "v1-Code"])

    restored, plan = from_keyed(keyed, ["High-Quality", "v1-Code"])

    assert _cursors(restored) == [7, 8]
    assert plan.restored == ("High-Quality", "v1-Code")


def test_mix_index_is_reset_when_the_child_set_changes():
    """`index` drives the interleave and is only valid for the set that produced it."""
    keyed = to_keyed(_loader_state([5, 5], index=10), ["a", "b"])

    unchanged, _ = from_keyed(keyed, ["a", "b"])
    assert _mix_index(unchanged) == 10, "unchanged set keeps its phase"

    changed, _ = from_keyed(keyed, ["a", "b", "c"], _loader_state([0, 0, 0], index=99))
    assert _mix_index(changed) == 0, "changed set must not resume out of phase"


def test_a_changed_dataset_list_needs_the_live_fresh_state():
    """Guessing the cursor shape is what made this fail with a bare KeyError before."""
    keyed = to_keyed(_loader_state([1]), ["a"])
    with pytest.raises(ValueError, match="fresh state"):
        from_keyed(keyed, ["a", "b"])


def test_state_without_per_child_cursors_is_rejected():
    """A map-level mix has one fused cursor; refuse it rather than corrupt a resume."""
    with pytest.raises(ValueError, match="ITER children"):
        to_keyed({"next_index": 10}, ["a"])


def test_id_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="2 children but 3 dataset ids"):
        to_keyed(_loader_state([1, 2]), ["a", "b", "c"])


def test_foreign_schema_is_rejected():
    with pytest.raises(ValueError, match="unsupported dataloader state schema"):
        from_keyed({"datasets": {}}, ["a"])
    assert KEYED_SCHEMA == "keyed-mix/2"
