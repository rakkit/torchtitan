# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-dataset checkpoint state keyed by dataset id.

THE PROBLEM THIS SOLVES
-----------------------
Stage 1 trains on datasets A-E. Stage 2 trains on A-I and must CONTINUE A-E from where
stage 1 stopped, while F-I start from zero. Nothing upstream expresses that:

  * a MAP-level mix stores one fused ``{"next_index": N}`` -- no per-child cursor exists,
    so the request cannot even be represented. This is why every child is an ITER dataset
    (``ParquetStreamSource`` builds a ``grain.IterDataset``, so ``SingleDatasetConfig`` takes
    the iter path and grain builds an iter-level mix);
  * an ITER-level mix stores ``{"parents": [cursor, cursor], "index": N}`` -- positional,
    and NESTED under every transform between it and the loader. Restoring a 2-child state
    into a 3-child mix SUCCEEDS SILENTLY, because grain zips parents against states
    non-strictly. Appending happens to work; inserting, removing or reordering silently
    resumes the wrong datasets at the wrong offsets.

The fork's answer until now was an env var, ``TORCHTITAN_MIX_STATE_MAPPING='0-0|1-1|
18-17|...'``, hand-derived by re-executing two config files and matching on
``(key, alias, path)`` -- so moving a corpus between clusters broke it, and correctness
depended on a shell guard on ``SLURM_ARRAY_TASK_ID``.

THE FIX
-------
Store the dataset ids IN the checkpoint next to the cursors. Then restoration is a
lookup rather than a reconstruction, and reordering, renaming an alias or moving a path
are all non-events. Nothing outside the checkpoint is consulted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

KEYED_SCHEMA = "keyed-mix/2"


@dataclass(frozen=True)
class RestorePlan:
    """What a load did, so it can be logged and asserted on."""

    restored: tuple[str, ...]
    """Datasets found in the checkpoint and resumed at their saved cursor."""
    fresh: tuple[str, ...]
    """Datasets in the config but not the checkpoint; they start at zero."""
    orphaned: tuple[str, ...]
    """Datasets in the checkpoint but no longer in the config; dropped."""

    def summary(self) -> str:
        return (
            f"dataloader restore: {len(self.restored)} resumed, "
            f"{len(self.fresh)} fresh, {len(self.orphaned)} orphaned"
        )


def find_mix_node(state: Any) -> dict[str, Any] | None:
    """Locate the mix's own state inside the loader's nested iterator state.

    The mix is NOT at the top level. Every transform between it and the loader wraps its
    parent's state, so with the shipped pretrain graph
    (mix -> ConcatThenSplit -> batch -> ThreadPrefetch) the loader's state looks like::

        {"remainder_slices": ..., "elements_from_buffer_after_checkpoint": 1,
         "parent_state": {"index": 6, "stop": False, "parents": [<cursor>, <cursor>]}}

    Reading `state["parents"]` therefore finds nothing and raises, which is what happened
    for every packed configuration -- i.e. every real one. Searching breadth-first finds
    the shallowest node carrying per-child cursors, and stays correct when a transform is
    added or removed, which is the whole reason not to index a fixed depth.
    """
    queue: list[Any] = [state]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            if isinstance(node.get("parents"), list):
                return node
            queue.extend(
                value for value in node.values() if isinstance(value, (dict, list))
            )
        elif isinstance(node, list):
            queue.extend(item for item in node if isinstance(item, (dict, list)))
    return None


def to_keyed(iterator_state: dict[str, Any], dataset_ids: list[str]) -> dict[str, Any]:
    """Name the per-dataset cursors, keeping the raw state for an exact resume.

    Both halves are stored on purpose. `iterator` is the loader's untouched state, which
    restores a SAME-CONFIG resume bit-exactly -- including the packer's partially
    consumed document, which no per-dataset cursor can express. `datasets` is the keyed
    view used when the dataset list has changed and an exact resume is not meaningful.
    """
    node = find_mix_node(iterator_state)
    if node is None:
        raise ValueError(
            "no per-dataset cursors found in the dataloader state; the mix must be "
            "built from ITER children (a map-level mix fuses them into one index)"
        )
    parents = node["parents"]
    if len(parents) != len(dataset_ids):
        raise ValueError(
            f"mix has {len(parents)} children but {len(dataset_ids)} dataset ids"
        )
    return {
        "schema": KEYED_SCHEMA,
        "dataset_ids": list(dataset_ids),
        "iterator": iterator_state,
        "datasets": dict(zip(dataset_ids, parents)),
    }


def from_keyed(
    keyed_state: dict[str, Any],
    dataset_ids: list[str],
    fresh_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], RestorePlan]:
    """Rebuild an iterator state for *this* config's datasets.

    Present in both -> resume. In the config only -> fresh. In the checkpoint only ->
    dropped. That is exactly the stage-1 -> stage-2 contract, with no external mapping.
    """
    schema = keyed_state.get("schema")
    if schema != KEYED_SCHEMA:
        raise ValueError(
            f"unsupported dataloader state schema {schema!r}, expected {KEYED_SCHEMA!r}"
        )
    saved: dict[str, Any] = keyed_state.get("datasets", {})
    plan = RestorePlan(
        restored=tuple(i for i in dataset_ids if i in saved),
        fresh=tuple(i for i in dataset_ids if i not in saved),
        orphaned=tuple(i for i in saved if i not in set(dataset_ids)),
    )

    # Same datasets in the same order: restore the raw state and lose nothing. Order
    # matters because the cursors inside it are positional.
    if list(keyed_state.get("dataset_ids", ())) == list(dataset_ids):
        return keyed_state["iterator"], plan

    # The dataset list changed, so the surrounding state (the packer's half-consumed
    # document, the mix's draw counter) describes a traversal that no longer exists.
    # Start from the LIVE iterator's own fresh state -- by definition the right shape for
    # this pipeline -- and splice the surviving cursors into it.
    if fresh_state is None:
        raise ValueError(
            "the dataset list changed since the checkpoint, which needs the live "
            "iterator's fresh state to restore into"
        )
    # deepcopy: `get_state()` hands back the iterator's internal dict BY REFERENCE for
    # some grain transforms, so writing into it here would corrupt the live iterator
    # (and, at construction, grain's shared step-zero state).
    fresh_state = copy.deepcopy(fresh_state)
    node = find_mix_node(fresh_state)
    if node is None:
        raise ValueError("no per-dataset cursors found in the fresh dataloader state")
    fresh_parents = list(node["parents"])
    node["parents"] = [
        saved.get(dataset_id, fresh_parents[index])
        for index, dataset_id in enumerate(dataset_ids)
    ]
    # `index` counts emitted samples and drives which child the mix draws from next. It
    # is only meaningful for the child set and weights that produced it, so a changed
    # dataset list must not inherit it -- otherwise the interleave resumes out of phase.
    node["index"] = 0
    return fresh_state, plan


def documents_consumed(keyed_state: dict[str, Any]) -> dict[str, int]:
    """Documents read per dataset, taken straight from the checkpointed cursors.

    Replaces 0.4.0's ``num_docs_sampled`` shared-memory counter (logged as
    ``data_docs/{alias}``). No separate accumulator is needed: each child streams rows and
    maintains its own ``next_index``, which IS the number of rows it has emitted --
    cumulative across epochs, so a corpus seen twice keeps counting up. That makes the
    metric exact by construction and automatically consistent with resume; the old
    counter lived in shared memory precisely so a scheduler in another process could
    read it, which left the main-process copy permanently stale.

    Note this counts DOCUMENTS, not tokens. Token counts are deliberately not tracked
    anywhere in this design: a ``token_count`` column is neither guaranteed present nor
    guaranteed accurate, and with pre-shuffled corpora document counts carry the same
    information about mixture proportions.
    """
    schema = keyed_state.get("schema")
    if schema != KEYED_SCHEMA:
        raise ValueError(
            f"unsupported dataloader state schema {schema!r}, expected {KEYED_SCHEMA!r}"
        )
    return {
        dataset_id: cursor_position(cursor)
        for dataset_id, cursor in keyed_state.get("datasets", {}).items()
    }


#: Grain transforms do not agree on what to call their parent's state.
#: `WindowShuffleIterDataset` uses `parent_window_start_state`; most use `parent`.
_PARENT_KEYS = ("parent", "parent_window_start_state", "parent_state")


def cursor_position(cursor: Any) -> int:
    """Documents emitted, found wherever the transform stack put it.

    A child's state is not flat: `random_map` (the sample processor) and the filters each
    wrap their parent, so the source's own cursor ends up nested, e.g.
    ``{"index_for_rng": 3, "parent": {"next_index": 3, "span_index": 0, ...}}``.
    Reading a fixed depth would silently break whenever a transform is added or removed,
    so walk the parent chain instead and take the first `next_index` found.

    Raises rather than returning 0 when the chain cannot be followed. Returning 0 is
    indistinguishable from "this dataset has read nothing", and that is exactly how a
    whole `data_docs/{alias}` series silently read zero for every run with `shuffle=True`
    -- `WindowShuffleIterDataset` names its parent key `parent_window_start_state`, the
    walk fell off the end, and the fallback reported 0 documents forever.
    """
    seen = 0
    while isinstance(cursor, dict) and seen < 32:  # bounded: states are shallow
        if "next_index" in cursor:
            return int(cursor["next_index"])
        for key in _PARENT_KEYS:
            if key in cursor:
                cursor = cursor[key]
                break
        else:
            break
        seen += 1
    raise ValueError(
        "no 'next_index' found in a dataset cursor; a transform in the pipeline names "
        f"its parent state something outside {_PARENT_KEYS}. Add it there rather than "
        "letting the per-dataset document counts silently read zero."
    )
