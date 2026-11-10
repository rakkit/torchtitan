# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable dataset recipes backed by Grain."""

import heapq
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast, Protocol, TypeAlias

import grain.python as grain
import numpy as np
from grain._src.python.dataset.transformations import mix as _grain_mix

from torchtitan.components.data.sources import (
    RandomAccessDataSource,
    SourceConfig,
    StreamingDataSource,
)
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.config import Configurable


GrainDataset: TypeAlias = grain.MapDataset | grain.IterDataset


class DatasetConfig(Protocol):
    """Builds one node of a Grain dataset graph."""

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSequence:
    """Next-token-aligned text preserved through composition and packing.

    NOTE: It is the dataset's processor responsibility to shift tokens into
    aligned input and label pairs. The trainer does not do it.
    """

    input_ids: np.ndarray
    """Tokens provided to the model."""
    labels: np.ndarray
    """Target for each input token, with `IGNORE_INDEX` where loss is disabled."""
    positions: np.ndarray | None = None
    """Per-token positions; `None` until packing or collation materializes them."""
    padding_mask: np.ndarray | None = None
    """Per-token mask that is true for padding; `None` before padding."""

    def __post_init__(self) -> None:
        lengths = [len(self.input_ids), len(self.labels)]
        if self.positions is not None:
            lengths.append(len(self.positions))
        if self.padding_mask is not None:
            lengths.append(len(self.padding_mask))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"TextSequence fields must have equal lengths, got {lengths}"
            )


class SampleProcessor(Configurable, ABC):
    """Configured row processor using Grain-provided deterministic randomness."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(
        self,
        sample: Any,
        rng: np.random.Generator,
    ) -> Any:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class SingleDatasetConfig:
    """Build one dataset from a source and row-level transforms.

    `pre_filters` run before `processor`; `post_filters` run afterward.
    This node owns leaf shuffle, repeat, and effective-DP sharding.
    """

    source: SourceConfig
    pre_filters: tuple[Callable[[Any], bool], ...] = ()
    processor: SampleProcessor.Config | None = None
    post_filters: tuple[Callable[[Any], bool], ...] = ()

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        source = self.source.build()
        if isinstance(source, RandomAccessDataSource):
            dataset: GrainDataset = grain.MapDataset.source(source)
        elif isinstance(source, StreamingDataSource):
            dataset = source.build_iter_dataset(
                dataset_iteration_policy=dataset_iteration_policy
            )
        else:
            raise TypeError(
                "source must build a RandomAccessDataSource or StreamingDataSource"
            )

        if isinstance(dataset, grain.MapDataset):
            return self._build_map_dataset(
                dataset,
                context=context,
                dataset_iteration_policy=dataset_iteration_policy,
            )
        return self._build_iter_dataset(
            dataset,
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )

    def _build_map_dataset(
        self,
        dataset: grain.MapDataset,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.MapDataset:
        """Process globally indexed rows before shuffle, sharding, and repeat."""
        # Filter raw rows.
        for filter_fn in self.pre_filters:
            dataset = dataset.filter(filter_fn)

        # Process rows into training samples.
        if self.processor is not None:
            dataset = dataset.random_map(
                self.processor.build(context=context),
                seed=dataset_iteration_policy.seed,
            )

        # Filter processed samples.
        for filter_fn in self.post_filters:
            dataset = dataset.filter(filter_fn)

        # Shuffle globally, then give each DP rank a disjoint slice.
        if dataset_iteration_policy.shuffle:
            dataset = dataset.shuffle(seed=dataset_iteration_policy.seed)
        if len(dataset) < dataset_iteration_policy.dp_world_size:
            raise ValueError(
                f"dataset has {len(dataset)} rows, fewer than "
                f"dp_world_size={dataset_iteration_policy.dp_world_size}"
            )
        shard_size, remainder = divmod(
            len(dataset), dataset_iteration_policy.dp_world_size
        )
        shard_start = dataset_iteration_policy.dp_rank * shard_size + min(
            dataset_iteration_policy.dp_rank, remainder
        )
        shard_stop = shard_start + shard_size
        if dataset_iteration_policy.dp_rank < remainder:
            shard_stop += 1
        dataset = dataset[shard_start:shard_stop]
        if dataset_iteration_policy.repeat:
            # Grain preserves the epoch through sliced map indices, so the
            # upstream shuffle uses seed + epoch on each repeat.
            dataset = dataset.repeat()
        return dataset

    def _build_iter_dataset(
        self,
        dataset: grain.IterDataset,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.IterDataset:
        """Shuffle streaming rows before row processing."""
        # Shuffle raw stream rows.
        if dataset_iteration_policy.shuffle:
            dataset = grain.experimental.WindowShuffleIterDataset(
                dataset,
                window_size=dataset_iteration_policy.streaming_shuffle_buffer_size,
                seed=dataset_iteration_policy.seed,
            )

        # Filter and process rows in stream order.
        for filter_fn in self.pre_filters:
            dataset = dataset.filter(filter_fn)
        if self.processor is not None:
            dataset = dataset.random_map(
                self.processor.build(context=context),
                seed=dataset_iteration_policy.seed + dataset_iteration_policy.dp_rank,
            )
        for filter_fn in self.post_filters:
            dataset = dataset.filter(filter_fn)
        return dataset


@dataclass(frozen=True, kw_only=True, slots=True)
class WeightedDataset:
    """A dataset and its relative selection weight."""

    dataset: DatasetConfig
    weight: float = 1.0
    tokens_per_document: float | None = None
    """Mean tokens per document; only `TokenWeightedMixConfig` reads it (start phase)."""


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetMixConfig:
    """Interleave weighted children.

    `MapDataset.filter` leaves rejected indices as `None`, so the all-map path
    weights attempted indices rather than accepted samples. Otherwise, weights
    select elements emitted by each iterable child. The child defines the
    element: mixing `TextSequence` children weights documents; mixing packed
    fixed-length children weights physical tokens.

    With `repeat=True` the mix is infinite. With `repeat=False` the mix stops
    at the first exhausted child, so larger children are not fully covered.
    """

    datasets: tuple[WeightedDataset, ...]

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        if not self.datasets or any(
            not math.isfinite(item.weight) or item.weight <= 0 for item in self.datasets
        ):
            raise ValueError(
                "DatasetMixConfig requires finite, positive-weight datasets"
            )
        children = [
            item.dataset.build(
                context=context,
                # Inserting or reordering a child reseeds every later child.
                dataset_iteration_policy=replace(
                    dataset_iteration_policy,
                    seed=dataset_iteration_policy.seed + index,
                ),
            )
            for index, item in enumerate(self.datasets)
        ]
        weights = [item.weight for item in self.datasets]
        # TODO(data-token-weighted-mix): Support source weights by token count,
        # not only emitted examples or packed rows. Checkpoint running estimates.
        if all(isinstance(child, grain.MapDataset) for child in children):
            return grain.MapDataset.mix(
                cast(list[grain.MapDataset], children),
                weights=weights,
            )
        children = [
            child.to_iter_dataset(read_options=context.read_options)
            if isinstance(child, grain.MapDataset)
            else child
            for child in children
        ]
        if len(children) == 1:
            return grain.IterDataset.mix(children, weights=weights)
        return _StridedMixIterDataset(
            children,
            weights=weights,
            block=context.mix_rank_stride,
            ranks=dataset_iteration_policy.dp_world_size,
            rank=dataset_iteration_policy.dp_rank,
        )


class _StridedMixIterDataset(grain.IterDataset):
    """grain's weighted interleave (`IterDataset.mix`), each DP rank at its own
    positions of the one sequence.

    grain picks the child of the k-th element from k and the weights only. This
    rank's i-th element takes position (i // block) * block * ranks + rank * block
    + i % block (block = elements a rank takes per microbatch), so the ranks' n-th
    microbatches together take block n of block * ranks consecutive positions, and
    every round of microbatches follows the weights to within one element per
    child. (A rank walking consecutive positions from rank * block made the ranks
    of a step share a window that moved only `block` positions per step: a child due
    1.3 elements per step got 1 for ~`ranks` steps in a row.) One rank = grain's
    mix. Only the choice of child changes: each rank reads its own shard of every
    child, in order. The state keeps grain's layout; "index" counts this rank's
    elements.
    """

    def __init__(
        self,
        parents: list[grain.IterDataset],
        *,
        weights: list[float],
        block: int,
        ranks: int,
        rank: int,
    ):
        super().__init__(parents)
        self._proportions = tuple(_grain_mix._float_to_int_proportions(weights))
        self._block, self._ranks, self._rank = block, ranks, rank

    def __iter__(self):
        return _StridedMixIterator(
            [parent.__iter__() for parent in self._parents],
            self._proportions,
            block=self._block,
            ranks=self._ranks,
            rank=self._rank,
        )


class _StridedMixIterator(grain.DatasetIterator):
    def __init__(
        self,
        parents: list[grain.DatasetIterator],
        proportions: tuple[int, ...],
        *,
        block: int,
        ranks: int,
        rank: int,
    ):
        super().__init__(parents)
        self._proportions = proportions
        self._block, self._ranks, self._rank = block, ranks, rank
        self._index = 0
        self._stop = False

    def __next__(self):
        if self._stop:
            raise StopIteration
        group, offset = divmod(self._index, self._block)
        position = (group * self._ranks + self._rank) * self._block + offset
        child, _ = _grain_mix._dataset_and_key_of_next_element(
            position, self._proportions
        )
        self._index += 1
        try:
            return next(self._parents[child])
        except Exception:
            self._stop = True  # as grain's mix: stop at the first exhausted child
            raise

    def get_state(self) -> dict[str, Any]:
        return {
            "parents": [parent.get_state() for parent in self._parents],
            "index": self._index,
            "stop": self._stop,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        for parent, parent_state in zip(self._parents, state["parents"], strict=True):
            parent.set_state(parent_state)
        self._index = state["index"]
        self._stop = state["stop"]


@dataclass(frozen=True, kw_only=True, slots=True)
class TokenWeightedMixConfig:
    """Interleave weighted children so that each child's share of the emitted TOKENS
    follows its weight, one PIECE at a time.

    Each child's documents are cut where the packer cuts them anyway (at every
    document start and every ``max_context_length`` tokens after it, as
    `_text_sequence_to_packing_input` assigns positions), and the next piece comes
    from the child whose emitted tokens divided by its weight is smallest (ties: the
    first child). Every rank's shares then track the weights to within one piece per
    child, whatever the document lengths: a 500k-token document is spread over many
    microbatches instead of filling one rank's microbatches back to back. A packer
    sees exactly the pieces whole documents would have given it. The state is the
    children's cursors (at the start of a partly emitted document), the offset into
    that document, one token count per child and this rank's phase.

    Example, weights 1 : 1, max_context_length 4, child A's documents 2 tokens,
    child B's 10 (pieces 4, 4, 2)::

        A B A A B A B ...   (A2 B4 A2 A2 B4 A2 B2: equal tokens)

    The choice depends only on the token counts. Every rank emits the same tokens
    per step, so ranks starting alike would reach a small child's turn at the same
    step (all ranks on it in one step, none in the next ones). So DP rank r starts
    with a PHASE: the counts the same choice reaches after r microbatches' tokens
    (`_rank_phase`), each child's pieces sized min(max_context_length, its
    `tokens_per_document`) so that its start is off by at most one of its own pieces;
    the phase is added to the rank's own counts when it chooses. The ranks of one
    step then sit at consecutive microbatches of one schedule, which follow the
    weights (as `_StridedMixIterDataset` does for the per-dataset packing). Each
    rank still reads its own shard of every child from its start. With
    `repeat=False` the mix stops at the first exhausted child.
    """

    datasets: tuple[WeightedDataset, ...]

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        if not self.datasets or any(
            not math.isfinite(item.weight) or item.weight <= 0 for item in self.datasets
        ):
            raise ValueError(
                "TokenWeightedMixConfig requires finite, positive-weight datasets"
            )
        children = [
            item.dataset.build(
                context=context,
                dataset_iteration_policy=replace(
                    dataset_iteration_policy,
                    seed=dataset_iteration_policy.seed + index,
                ),
            )
            for index, item in enumerate(self.datasets)
        ]
        children = [
            child.to_iter_dataset(read_options=context.read_options)
            if isinstance(child, grain.MapDataset)
            else child
            for child in children
        ]
        weights = [item.weight for item in self.datasets]
        return _TokenWeightedMixIterDataset(
            children,
            weights=weights,
            max_context_length=context.max_context_length,
            phase=_rank_phase(
                weights,
                tokens=dataset_iteration_policy.dp_rank
                * context.num_tokens_per_microbatch,
                piece_tokens=[
                    max(
                        1,
                        round(
                            min(context.max_context_length, item.tokens_per_document)
                        ),
                    )
                    if item.tokens_per_document
                    else context.max_context_length
                    for item in self.datasets
                ],
            ),
        )


def _rank_phase(
    weights: list[float], *, tokens: int, piece_tokens: list[int]
) -> list[int]:
    """Per child, the tokens the token-weighted choice gives it until `tokens` tokens
    are emitted, child i's pieces `piece_tokens[i]` long: a DP rank's start in the
    common schedule (`TokenWeightedMixConfig`)."""
    counts = [0] * len(weights)
    heap = [(0.0, index) for index in range(len(weights))]  # (count / weight, index)
    emitted = 0
    while emitted < tokens:
        _, index = heapq.heappop(heap)
        counts[index] += piece_tokens[index]
        emitted += piece_tokens[index]
        heapq.heappush(heap, (counts[index] / weights[index], index))
    return counts


def _piece_starts(document: TextSequence, max_context_length: int) -> np.ndarray:
    """Where the packer restarts positions: every document start and every
    max_context_length tokens after it (`_text_sequence_to_packing_input`)."""
    token_indices = np.arange(len(document.input_ids), dtype=np.int64)
    document_starts = 0
    if document.positions is not None:
        document_starts = np.maximum.accumulate(
            np.where(np.asarray(document.positions) == 0, token_indices, 0)
        )
    return np.flatnonzero((token_indices - document_starts) % max_context_length == 0)


def _slice(document: TextSequence, start: int, end: int) -> TextSequence:
    def cut(field: np.ndarray | None) -> np.ndarray | None:
        return None if field is None else np.asarray(field)[start:end]

    return TextSequence(
        input_ids=cut(document.input_ids),
        labels=cut(document.labels),
        positions=cut(document.positions),
        padding_mask=cut(document.padding_mask),
    )


class _TokenWeightedMixIterDataset(grain.IterDataset):
    def __init__(
        self,
        parents: list[grain.IterDataset],
        *,
        weights: list[float],
        max_context_length: int,
        phase: list[int],
    ):
        super().__init__(parents)
        self._weights = weights
        self._max_context_length = max_context_length
        self._phase = phase

    def __iter__(self):
        return _TokenWeightedMixIterator(
            [parent.__iter__() for parent in self._parents],
            self._weights,
            self._max_context_length,
            self._phase,
        )


class _TokenWeightedMixIterator(grain.DatasetIterator):
    # The state keeps grain's mix layout ("parents" = the children's cursors), so
    # keyed_mix_state finds and re-keys the cursors as for grain's own mix.
    def __init__(
        self,
        parents: list[grain.DatasetIterator],
        weights: list[float],
        max_context_length: int,
        phase: list[int],
    ):
        super().__init__(parents)
        self._weights = weights
        self._max_context_length = max_context_length
        self._tokens = [0] * len(parents)  # emitted by this rank (data_tokens/*)
        self._phase = list(phase)  # this rank's start in the schedule
        # per child: its partly emitted document, the cursor before it, the next
        # piece's index into that document's piece starts
        self._document: list[TextSequence | None] = [None] * len(parents)
        self._starts: list[np.ndarray | None] = [None] * len(parents)
        self._cursor: list[Any] = [None] * len(parents)
        self._piece = [0] * len(parents)
        self._stop = False

    def _fetch(self, index: int) -> None:
        cursor = self._parents[index].get_state()
        document = next(self._parents[index])
        while len(document.input_ids) == 0:
            cursor = self._parents[index].get_state()
            document = next(self._parents[index])
        self._document[index], self._cursor[index] = document, cursor
        self._starts[index] = _piece_starts(document, self._max_context_length)
        self._piece[index] = 0

    def __next__(self) -> TextSequence:
        if self._stop:
            raise StopIteration
        index = min(
            range(len(self._parents)),
            key=lambda i: (self._tokens[i] + self._phase[i]) / self._weights[i],
        )
        if self._document[index] is None:
            try:
                self._fetch(index)
            except StopIteration:
                self._stop = True
                raise
        document, starts, piece = (
            self._document[index],
            self._starts[index],
            self._piece[index],
        )
        start = int(starts[piece])
        end = (
            int(starts[piece + 1])
            if piece + 1 < len(starts)
            else len(document.input_ids)
        )
        if piece + 1 < len(starts):
            self._piece[index] = piece + 1
        else:
            self._document[index] = self._starts[index] = self._cursor[index] = None
        self._tokens[index] += end - start
        return _slice(document, start, end)

    def get_state(self) -> dict[str, Any]:
        return {
            "parents": [
                parent.get_state() if self._document[i] is None else self._cursor[i]
                for i, parent in enumerate(self._parents)
            ],
            "pieces": [
                0 if self._document[i] is None else self._piece[i]
                for i in range(len(self._parents))
            ],
            "tokens": list(self._tokens),
            "phase": list(self._phase),
            "stop": self._stop,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        for index, (parent, parent_state) in enumerate(
            zip(self._parents, state["parents"], strict=True)
        ):
            parent.set_state(parent_state)
            self._document[index] = self._starts[index] = self._cursor[index] = None
            if state["pieces"][index]:  # re-read the partly emitted document
                self._fetch(index)
                self._piece[index] = state["pieces"][index]
        self._tokens = list(state["tokens"])
        self._phase = list(state["phase"])
        self._stop = state["stop"]


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetConcatConfig:
    """Concatenates finite children before global shuffle and DP sharding."""

    datasets: tuple[DatasetConfig, ...]

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.MapDataset:
        child_iteration_policy = replace(
            dataset_iteration_policy,
            shuffle=False,
            repeat=False,
            dp_rank=0,
            dp_world_size=1,
        )
        children = [
            dataset.build(
                context=context,
                # As in DatasetMixConfig, each child gets its own seed so that
                # a repeated child does not repeat its random draws.
                # Inserting or reordering a child reseeds every later child.
                dataset_iteration_policy=replace(
                    child_iteration_policy,
                    seed=dataset_iteration_policy.seed + index,
                ),
            )
            for index, dataset in enumerate(self.datasets)
        ]

        if not children or not all(
            isinstance(child, grain.MapDataset) for child in children
        ):
            raise TypeError("DatasetConcatConfig requires map-style children")

        dataset = grain.MapDataset.concatenate(cast(list[grain.MapDataset], children))

        if dataset_iteration_policy.shuffle:
            dataset = dataset.shuffle(seed=dataset_iteration_policy.seed)

        if len(dataset) < dataset_iteration_policy.dp_world_size:
            raise ValueError(
                f"dataset has {len(dataset)} rows, fewer than "
                f"dp_world_size={dataset_iteration_policy.dp_world_size}"
            )

        shard_size, remainder = divmod(
            len(dataset), dataset_iteration_policy.dp_world_size
        )
        shard_start = dataset_iteration_policy.dp_rank * shard_size + min(
            dataset_iteration_policy.dp_rank, remainder
        )
        shard_stop = shard_start + shard_size
        if dataset_iteration_policy.dp_rank < remainder:
            shard_stop += 1
        dataset = dataset[shard_start:shard_stop]

        if dataset_iteration_policy.repeat:
            dataset = dataset.repeat()
        return dataset
