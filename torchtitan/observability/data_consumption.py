# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-dataset data metrics. Reports; does not schedule.

WHAT WAS HERE BEFORE
--------------------
A 420-line `DataMixScheduler` that reweighted the mixture during training by reaching
into `dataloader.dataset.weights` and mutating a live tensor, plus the percentage-keyed
schedule parser, the shared-memory counter walk and two fallback scheduler classes.

None of it can work on the grain path and none of it is reachable: grain's
`DatasetMixConfig` resolves mix proportions into the built graph
(`components/data/dataset.py:250-259`) and exposes no way to read or change them
afterwards, and no dataloader in the tree has a `.dataset` attribute any more.

That is not a lost capability in practice. A schedule over the mixture is expressed as
separate stage configs resumed from a checkpoint, which the keyed per-dataset cursors
(`components/data/keyed_mix_state.py`) support directly and inspectably -- unlike the old
scheme, which needed a hand-maintained `TORCHTITAN_MIX_STATE_MAPPING` env var to line the
stages up.

WHAT IS KEPT
------------
The reporting half, which has no upstream equivalent: `data_docs/{alias}` per dataset,
read straight from the checkpointed cursors, and the constant `data_mixing/{alias}`
weights. `data_tokens/*` is deliberately absent -- token counts are not tracked anywhere
in this design, because a `token_count` column is neither guaranteed present nor
guaranteed accurate.
"""

import torch

# BaseDataLoader is UPSTREAM's (components/data/loader.py).
from torchtitan.components.data.loader import BaseDataLoader

__all__ = ["DataMixMetrics", "build_data_mix_metrics"]


class DataMixMetrics:
    """Reports per-dataset consumption for a loader that keys its cursors by alias."""

    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader

    def get_weights_at_step(self, current_step: int):
        del current_step
        return list(getattr(self.dataloader, "_dataset_weights", []))

    def get_log_dict_at_step(self, current_step: int):
        del current_step
        mixing = {
            key: torch.tensor(value)
            for key, value in self.dataloader.mixing_weights().items()
        }
        docs = {
            key: torch.tensor(value)
            for key, value in self.dataloader.consumption_metrics().items()
        }
        return mixing, docs, {}

    def convert_mixing_configs_to_json(self):
        return {"0": self.get_weights_at_step(0)}

    def step(self, current_step: int) -> None:
        pass

    def dump_mixing_configs(self, dump_folder: str) -> None:
        pass


class NoDataMixMetrics:
    """For a loader with no per-dataset cursors (a plain single-source GrainDataLoader).

    Reports nothing rather than reporting zeros under an invented alias, which is what
    the old fallback did and what made `data_docs/not_mixed_datasets` show up in runs.
    """

    def get_weights_at_step(self, current_step: int):
        del current_step
        return [1]

    def get_log_dict_at_step(self, current_step: int):
        del current_step
        return {}, {}, {}

    def convert_mixing_configs_to_json(self):
        return {"0": [1]}

    def step(self, current_step: int) -> None:
        pass

    def dump_mixing_configs(self, dump_folder: str) -> None:
        pass


def build_data_mix_metrics(
    dataloader: BaseDataLoader,
    mixing_scheduler_configs: str | None = None,
):
    if mixing_scheduler_configs:
        raise ValueError(
            "data_mixing_scheduler_configs is set, but the grain dataloader fixes its "
            "mix weights at build time and cannot reweight during a run. Express the "
            "schedule as separate stage configs resumed from a checkpoint instead."
        )
    if hasattr(dataloader, "consumption_metrics"):
        return DataMixMetrics(dataloader)
    return NoDataMixMetrics()
