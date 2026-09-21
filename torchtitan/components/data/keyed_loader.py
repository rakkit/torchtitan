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
import logging

from dataclasses import dataclass
from typing import Annotated, Any

import tyro

from torchtitan.components.data.keyed_mix_state import (
    documents_consumed,
    from_keyed,
    to_keyed,
)
from torchtitan.components.data.loader import GrainDataLoader

KEYED_STATE_VERSION = 2


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
        state = super().state_dict()
        state[self._rank_id] = to_keyed(state[self._rank_id], self._dataset_ids)
        state["version"] = KEYED_STATE_VERSION
        return state

    def mixing_weights(self) -> dict[str, float]:
        """Configured weight per dataset, normalized to sum to 1.

        Reported as `data_mixing/{alias}`, matching 0.4.0. These are CONSTANTS on this
        loader: grain resolves mix proportions at build time, so unlike 0.4.0 there is no
        step at which they change.
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

        Mirrors 0.4.0's `data_docs/{alias}` series. No id->alias translation is needed:
        the alias is the key.
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
        if not state_dict:
            return
        version = state_dict.get("version")
        if version != KEYED_STATE_VERSION:
            raise ValueError(
                f"unsupported dataloader state version {version}, expected "
                f"{KEYED_STATE_VERSION}. A checkpoint written by the legacy loader "
                "cannot be resumed onto this one; start the stage fresh."
            )
        if state_dict["dp_world_size"] != self._dp_world_size:
            raise ValueError(
                "cannot resume after changing the effective data-parallel degree"
            )
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )

        # Capture the untouched state first: it is the correct "fresh" cursor for any
        # dataset this stage adds, in whatever shape the transform stack produces.
        fresh_state = self._iterator.get_state()
        restored_state, plan = from_keyed(
            state_dict[self._rank_id], self._dataset_ids, fresh_state
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

logger = logging.getLogger(__name__)
