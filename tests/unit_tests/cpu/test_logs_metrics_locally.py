# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DiSCO skips its logging gather only when every shard rank has a logger."""

import pytest

from torchtitan.observability import metrics

from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import logs_metrics_locally


@pytest.mark.parametrize(
    "all_shards, all_ranks, wandb, tensorboard, local",
    [
        (True, True, True, False, True),
        (True, True, False, True, True),
        (True, True, False, False, False),  # no logger at all
        (True, False, True, False, False),  # narrowed to one rank's logger
        (False, True, True, False, False),
    ],
)
def test_logs_metrics_locally(all_shards, all_ranks, wandb, tensorboard, local):
    m = MetricsProcessor.Config(
        save_all_shard_ranks=all_shards,
        save_for_all_ranks=all_ranks,
        loggers=([metrics.WandBLogger.Config()] if wandb else [])
        + ([metrics.TensorBoardLogger.Config()] if tensorboard else []),
    )
    assert logs_metrics_locally(m) is local
