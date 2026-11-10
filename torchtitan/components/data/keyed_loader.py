# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A GrainDataLoader whose checkpoint state names its datasets.

Upstream stores the mix cursors positionally, so resuming a changed dataset list needs
an external index mapping -- historically a hand-maintained env var. This subclass
writes the dataset ids into the checkpoint beside the cursors, which makes restoration a
lookup instead of a reconstruction. See keyed_mix_state for the contract.

Everything else -- rank keying, the DP-degree guard, iteration -- is inherited unchanged.
"""

from __future__ import annotations

import copy
import logging
import pickle

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import tyro

from torchtitan.components.data.keyed_mix_state import (
    documents_consumed,
    find_mix_node,
    from_keyed,
    RestorePlan,
    stream_state,
    to_keyed,
)
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.parquet_stream import ParquetStreamSource, redealt_state

KEYED_STATE_VERSION = 4


class KeyedMixDataLoader(GrainDataLoader):
    """GrainDataLoader that checkpoints per-dataset cursors by dataset id."""

    @dataclass(kw_only=True, slots=True)
    class Config(GrainDataLoader.Config):
        dataset_ids: Annotated[tuple[str, ...], tyro.conf.Suppress] = ()
        """Aliases of the mix children, in order. Set by the config builder alongside the
        dataset list; these are what the checkpoint is keyed by."""
        dataset_weights: Annotated[tuple[float, ...], tyro.conf.Suppress] = ()
        """Configured mixing weights, in the same order. Carried only so the metrics
        logger can report `data_mixing/{alias}` -- grain fixes the weights inside the
        built graph and offers no way to read them back, and no way to change them
        (`DatasetMixConfig` resolves them at build time)."""

    def __init__(self, config: Config, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._dp_rank = kwargs["dp_rank"]
        self._dataset_config = config.dataset
        # Old ranks' items the next load reads (set by the checkpointer; see
        # `request_saved_ranks`).
        self._saved_rank_ids: tuple[str, ...] = ()
        self._dataset_ids = list(config.dataset_ids)
        self._dataset_weights = list(config.dataset_weights)
        if self._dataset_weights and len(self._dataset_weights) != len(
            self._dataset_ids
        ):
            raise ValueError(
                f"{len(self._dataset_weights)} dataset_weights for "
                f"{len(self._dataset_ids)} dataset_ids"
            )
        if not self._dataset_ids:
            raise ValueError(
                "KeyedMixDataLoader requires dataset_ids; without them the checkpoint "
                "cannot name its cursors and cross-stage resume degrades to positional"
            )

    def state_dict(self) -> dict[str, Any]:
        # The rank's state is ONE opaque item: DCP turns every nested leaf into its
        # own checkpoint item (~1,200 per rank for a 44-dataset mix) and loads only
        # the keys the live state asks for, which a stage adding datasets would not
        # find in the checkpoint.
        state = super().state_dict()
        state[self._rank_id] = pickle.dumps(
            to_keyed(state[self._rank_id], self._dataset_ids)
        )
        state["version"] = KEYED_STATE_VERSION
        for rank_id in self._saved_rank_ids:
            state.setdefault(rank_id, b"")
        return state

    def request_saved_ranks(self, rank_ids: Sequence[str]) -> None:
        """The checkpointer, before a load, names the rank items the checkpoint
        holds. After a data-parallel degree change every rank re-deals all old
        ranks' cursors, so the next load asks for all of them; with the same degree
        each rank reads only its own item."""
        if len(rank_ids) != self._dp_world_size:
            self._saved_rank_ids = tuple(rank_ids)

    def mixing_weights(self) -> dict[str, float]:
        """Configured weight per dataset, normalized to sum to 1.

        Reported as `data_mixing/{alias}`. Constants: grain resolves the mix
        proportions at build time.
        """
        total = sum(self._dataset_weights)
        if not total:
            return {}
        return {
            f"data_mixing/{alias.replace('/', '_')}": weight / total
            for alias, weight in zip(self._dataset_ids, self._dataset_weights)
        }

    def consumption_metrics(self) -> dict[str, int]:
        """Per-dataset documents consumed, ready for the metrics logger.

        Reported as `data_docs/{alias}`; the alias is the key.
        """
        counts = documents_consumed(
            to_keyed(self._iterator.get_state(), self._dataset_ids)
        )
        return {
            # "/" is a namespace separator in wandb, so it must not appear in a label.
            f"data_docs/{alias.replace('/', '_')}": value
            for alias, value in counts.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._saved_rank_ids = ()  # requested for this load only
        if not state_dict:
            return
        version = state_dict.get("version")
        if version != KEYED_STATE_VERSION:
            raise ValueError(
                f"unsupported dataloader state version {version}, expected "
                f"{KEYED_STATE_VERSION}: start the stage fresh."
            )
        if state_dict["dp_world_size"] != self._dp_world_size:
            restored_state, plan = self._redealt(state_dict)
        else:
            if not state_dict.get(self._rank_id):
                raise ValueError(
                    f"checkpoint is missing dataloader state for {self._rank_id}"
                )
            # Capture the untouched state first: it is the correct "fresh" cursor for
            # any dataset this stage adds, in whatever shape the transform stack
            # produces.
            fresh_state = self._iterator.get_state()
            restored_state, plan = from_keyed(
                pickle.loads(state_dict[self._rank_id]), self._dataset_ids, fresh_state
            )
        # Log on every rank's rank 0 only; a wrong stage config should be visible at
        # step 0 rather than inferred from a loss curve later.
        if self._rank_id == "dp_rank_0":
            logger.info("%s", plan.summary())
            if plan.fresh:
                logger.info("  starting fresh: %s", ", ".join(plan.fresh))
            if plan.orphaned:
                logger.info(
                    "  in checkpoint but not in this config (dropped): %s",
                    ", ".join(plan.orphaned),
                )
        try:
            self._iterator.set_state(restored_state)
        except Exception:
            self.close()
            raise

    def _redealt(
        self, state_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], RestorePlan]:
        """This rank's iterator state after a data-parallel degree change: per
        dataset, the rows every old rank had not read in its current pass are
        re-dealt over the new ranks (`redealt_state`). The packer's partial
        document and the mix's draw counter of the old ranks are dropped: the mix
        and its weights restart from this rank's fresh state (its own start position)."""
        old_world = int(state_dict["dp_world_size"])
        old = []
        for rank in range(old_world):
            item = state_dict.get(f"dp_rank_{rank}")
            if not item:
                raise ValueError(
                    f"resuming {old_world} data-parallel ranks' dataloader state on "
                    f"{self._dp_world_size} needs every old rank's state; dp_rank_{rank} "
                    "was not loaded (the checkpointer requests them all)"
                )
            old.append(pickle.loads(item)["datasets"])
        sources = _parquet_sources(self._dataset_config)
        # deepcopy: get_state() can return the iterator's internal dicts by reference.
        state = copy.deepcopy(self._iterator.get_state())
        node = find_mix_node(state)
        if node is None:
            raise ValueError(
                "no per-dataset cursors found in the fresh dataloader state"
            )
        restored = []
        for index, alias in enumerate(self._dataset_ids):
            if any(alias not in cursors for cursors in old):
                continue
            target = stream_state(node["parents"][index])
            redealt = redealt_state(
                sources[alias],
                [stream_state(cursors[alias]) for cursors in old],
                dp_rank=self._dp_rank,
                dp_world_size=self._dp_world_size,
                fresh_state=target,
            )
            target.clear()
            target.update(redealt)
            restored.append(alias)
        # the mix restarts at this rank's fresh index (its start position, dp_rank *
        # mix_rank_stride), never at 0 on every rank
        saved = {alias for cursors in old for alias in cursors}
        plan = RestorePlan(
            restored=tuple(restored),
            fresh=tuple(i for i in self._dataset_ids if i not in restored),
            orphaned=tuple(sorted(saved - set(self._dataset_ids))),
        )
        if self._rank_id == "dp_rank_0":
            logger.info(
                "dataloader: %d data-parallel ranks' unread rows re-dealt over %d",
                old_world,
                self._dp_world_size,
            )
        return state, plan


def _parquet_sources(dataset_config: Any) -> dict[str, ParquetStreamSource.Config]:
    """Each mix child's parquet source config, by alias: the keyed mix inside the
    packing configs (`make_pretrain_dataloader_config`)."""
    node = dataset_config
    while not hasattr(node, "aliases"):
        node = getattr(node, "dataset", None)
        if node is None:
            raise ValueError(
                "re-dealing after a data-parallel degree change needs the keyed "
                "dataset mix (make_pretrain_dataloader_config)"
            )
    sources = {}
    for alias, item in zip(node.aliases, node.datasets):
        child = (
            item.dataset
        )  # per-dataset packing wraps each child in its packer config
        while not hasattr(child, "source") and hasattr(child, "dataset"):
            child = child.dataset
        source = getattr(child, "source", None)
        if not isinstance(source, ParquetStreamSource.Config):
            raise ValueError(
                f"dataset {alias!r}: re-dealing after a data-parallel degree change "
                "needs a parquet stream source"
            )
        sources[alias] = source
    return sources


logger = logging.getLogger(__name__)
