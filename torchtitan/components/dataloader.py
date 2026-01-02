# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.tools.logging import logger


# NOTE: This class deliberately inherits from `Exception` and not `StopIteration`.
# According to PEP 479, raising a `StopIteration` or its subclass from within a
# generator will wrap it in a `RuntimeError`. Since this exception is designed
# to be raised from a generator-based dataloader and caught by the training loop,
# inheriting from `StopIteration` would make it uncatchable and would crash the
# program.
# See: https://peps.python.org/pep-0479/
class DataloaderExhaustedError(Exception):
    """An exception that indicates dataloader exhaustion."""

    pass


class BaseDataLoader(Stateful, ABC):
    """Base class for all dataloaders.

    This is used to enforce that all dataloaders have the methods defined in ``Stateful``,
    ``state_dict()`` and ``load_state_dict()``.
    """

    @abstractmethod
    def __iter__(self): ...


class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
        collate_fn: Optional function to collate samples in a batch.
        data_loader_kwargs: Additional arguments to pass to the
            ``torchdata.stateful_dataloader.StatefulDataLoader`` constructor.
    """

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        collate_fn: Callable | None = None,
        **data_loader_kwargs,
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        super().__init__(
            dataset, batch_size, collate_fn=collate_fn, **data_loader_kwargs
        )
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                "expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to
        # keep this for backward compatibility.

        rank_state = pickle.loads(state_dict[self._rank_id])

        snap = rank_state["_snapshot"]
        main = snap["_main_snapshot"]
        w0 = snap["_worker_snapshots"]["worker_0"]

        datasets_state = w0["dataset_state"]["datasets"]

        if isinstance(datasets_state, dict):
            # reset dataloader progress
            snap["_snapshot_step"] = 0
            main["_sampler_iter_yielded"] = 0
            if (
                main.get("_sampler_iter_state")
                and "samples_yielded" in main["_sampler_iter_state"]
            ):
                main["_sampler_iter_state"]["samples_yielded"] = 0

            # reset outer bookkeeping (often matters for “hang on first batch”)
            rank_state["_steps_since_snapshot"] = 0
            rank_state["_iterator_finished"] = False

            # reset worker fetcher state (if present)
            if "fetcher_state" in w0:
                w0["fetcher_state"]["dataset_iter_state"] = None
                w0["fetcher_state"]["fetcher_ended"] = False

            logger.info(
                f"Dataloader checkpoint state for dp rank {self.dp_rank} has been monkey patched"
            )

        super().load_state_dict(rank_state)
