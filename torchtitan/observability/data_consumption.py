# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-dataset data metrics of the keyed-mix loader (no upstream equivalent):
``data_docs/{alias}``, the documents consumed per dataset, read straight from
the checkpointed cursors, ``data_tokens/{alias}`` from a token-weighted mix's own
counts, and the constant ``data_mixing/{alias}`` weights."""

import torch

from torchtitan.components.data.keyed_loader import KeyedMixDataLoader
from torchtitan.components.data.loader import BaseDataLoader

__all__ = [
    "DataMixMetrics",
    "DataMixMetricsLike",
    "NoDataMixMetrics",
    "build_data_mix_metrics",
]


class DataMixMetrics:
    """Reports per-dataset consumption for a loader that keys its cursors by alias."""

    def __init__(self, dataloader: KeyedMixDataLoader) -> None:
        self.dataloader = dataloader

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
        return mixing, docs


class NoDataMixMetrics:
    """For a loader with no per-dataset cursors (a plain single-source
    GrainDataLoader): reports nothing."""

    def get_log_dict_at_step(self, current_step: int):
        del current_step
        return {}, {}


# What `build_data_mix_metrics` returns: the real reporter, or the null object
# used when the loader has no per-dataset cursors.
DataMixMetricsLike = DataMixMetrics | NoDataMixMetrics


def build_data_mix_metrics(dataloader: BaseDataLoader) -> DataMixMetricsLike:
    if isinstance(dataloader, KeyedMixDataLoader):
        return DataMixMetrics(dataloader)
    return NoDataMixMetrics()
