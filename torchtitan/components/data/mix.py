# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One list of dataset records -> an upstream grain mix.

A dataset is one ``DatasetSpec`` record carrying its own fields. Its ``alias``
labels the metrics (``data_docs/{alias}``), keys its checkpoint cursor across
stages and seeds its RNG: renaming an alias restarts that dataset's cursor and
changes its shuffle order; reordering the list or moving the corpus to another
filesystem does not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from torchtitan.components.data.collators import TextCollator

from torchtitan.components.data.dataset import (
    DatasetMixConfig,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
    WeightedDataset,
)
from torchtitan.components.data.keyed_loader import KeyedMixDataLoader
from torchtitan.components.data.packing import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
)
from torchtitan.components.data.parquet_stream import ParquetStreamSource
from torchtitan.components.data.text_datasets import TextProcessor
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy


@dataclass(kw_only=True, slots=True)
class DatasetSpec:
    """One corpus in a training mixture. Used by pretrain and SFT alike."""

    alias: str
    """Names the dataset. Unique, and stable across stages: it labels the metrics
    (`data_docs/{alias}`), keys this dataset's checkpoint cursor, and seeds its RNG.
    Renaming it makes the dataset start from zero; moving its path or reordering the
    list does not."""
    path: str
    """Dataset directory: parquet shards plus manifest.json."""
    weight: float = 1.0
    """Relative selection weight. Normalized against the other datasets, so only ratios
    matter. Selects documents."""
    text_key: str = "text"
    """Column to read. Validated against the manifest at startup."""
    manifest_path: str | None = None
    """Override for a corpus whose directory is not writable. Normally unset."""
    columns: tuple[str, ...] = ()
    """Columns that MUST exist, validated against the manifest at startup. Defaults to
    just `text_key`."""
    optional_columns: tuple[str, ...] = ()
    """Columns decoded only if the corpus has them. SFT uses `tools` /
    `enable_thinking`, which some corpora omit."""
    row_adapter: Any = None
    """SFT only: how to turn THIS corpus's row into `(messages, tools, enable_thinking)`.

    Corpora disagree about layout -- a `messages` list vs a prompt/response column pair,
    tools in their own column vs nested in `messages[0]`, everything as JSON strings vs
    real structs; see `data/sft.py` for the shipped adapters. Ignored by pretrain."""

    def __post_init__(self) -> None:
        if not self.alias:
            raise ValueError("DatasetSpec.alias is required (it keys checkpoint state)")
        if not self.path:
            raise ValueError(f"DatasetSpec {self.alias!r}: path is required")
        if not (self.weight > 0):
            raise ValueError(
                f"DatasetSpec {self.alias!r}: weight must be positive, got {self.weight}"
            )
        if not self.columns:
            self.columns = (self.text_key,)


def alias_seed(base_seed: int, alias: str) -> int:
    """Derive a dataset's RNG seed from its alias rather than its list position.

    Upstream seeds mix children BY POSITION (`seed=policy.seed + index`,
    `dataset.py:236-240`, with its own comment "Inserting or reordering a child reseeds
    every later child"). That is fine when the dataset list is fixed, and wrong for the
    stage workflow this fork exists to support: stage 2 inserts datasets ahead of
    existing ones, which would silently re-randomize the span order and the sample RNG of
    every dataset after the insertion point -- datasets whose cursors were just carefully
    restored.

    Hashing the alias makes the seed a property of the DATASET, so the only thing that
    changes a dataset's data order is renaming it, which is the same rule that governs
    its cursor. `blake2b` rather than `hash()` because the builtin is salted for `str`.
    """
    digest = hashlib.blake2b(alias.encode("utf-8"), digest_size=4).digest()
    return (base_seed + int.from_bytes(digest, "big")) % (2**31 - 1)


@dataclass(frozen=True, kw_only=True, slots=True)
class KeyedDatasetMixConfig(DatasetMixConfig):
    """A mix whose children are seeded by alias instead of by list position.

    Everything else -- weighting, the map/iter dispatch, the exhaustion rules -- is
    upstream's. Only the seed derivation is replaced; see `alias_seed`.
    """

    aliases: tuple[str, ...]

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> Any:
        if len(self.aliases) != len(self.datasets):
            raise ValueError(
                f"{len(self.aliases)} aliases for {len(self.datasets)} datasets"
            )
        base_seed = dataset_iteration_policy.seed
        children = [
            WeightedDataset(
                dataset=_SeededDataset(
                    dataset=item.dataset,
                    seed=alias_seed(base_seed, alias),
                ),
                weight=item.weight,
            )
            for item, alias in zip(self.datasets, self.aliases)
        ]
        # Delegate to upstream with children that pin their own seed, so upstream's
        # `seed + index` lands on a value the child then ignores.
        return DatasetMixConfig(datasets=tuple(children)).build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _SeededDataset:
    """Wraps a dataset config so it builds with a fixed seed, whatever it is handed."""

    dataset: Any
    seed: int

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> Any:
        return self.dataset.build(
            context=context,
            dataset_iteration_policy=replace(dataset_iteration_policy, seed=self.seed),
        )


def _text_reader(column: str):
    """Read one named column. Bound per dataset so a mixture can span schemas."""

    def read(sample: dict[str, Any]) -> str:
        return sample[column]

    return read


def build_mix(
    datasets: list[DatasetSpec],
    *,
    processor_for: Any = None,
    read_in_order: bool = False,
) -> KeyedDatasetMixConfig:
    """Turn the record list into a weighted, alias-seeded grain mix.

    `processor_for(dataset) -> SampleProcessor.Config` overrides tokenization per
    dataset; the default reads `text_key` and tokenizes with the run's tokenizer.
    """
    if not datasets:
        raise ValueError("build_mix requires at least one dataset")

    # Uniqueness is checked on the METRIC name, not the raw alias: "/" is a namespace
    # separator in wandb, so keyed_loader normalizes it to "_". Two aliases differing
    # only by a slash would otherwise pass this check and then collapse into one metric
    # series, where the survivor reports the other dataset's weight.
    seen: dict[str, str] = {}
    for dataset in datasets:
        key = dataset.alias.replace("/", "_")
        if key in seen:
            raise ValueError(
                f"dataset aliases {seen[key]!r} and {dataset.alias!r} both normalize to "
                f"{key!r}; the alias keys checkpoint state and names the metric series, "
                "so it must be unique after '/' is replaced with '_'"
            )
        seen[key] = dataset.alias

    children = []
    for dataset in datasets:
        processor: SampleProcessor.Config = (
            processor_for(dataset)
            if processor_for is not None
            else TextProcessor.Config(text_fn=_text_reader(dataset.text_key))
        )
        children.append(
            WeightedDataset(
                dataset=SingleDatasetConfig(
                    # A SampleProcessor returns None to REJECT a sample -- SFT does it
                    # on five paths (empty conversation, no assistant turn, nothing
                    # trainable, too short, over length). Grain's map passes None
                    # through untouched, so without this filter the first rejected
                    # sample reaches the packer as `None.input_ids` and kills the run.
                    # Upstream's own test pins this idiom
                    # (test_grain_data.py::test_single_dataset_post_filter_removes_none).
                    post_filters=(lambda sample: sample is not None,),
                    source=ParquetStreamSource.Config(
                        path=dataset.path,
                        manifest_path=dataset.manifest_path,
                        columns=dataset.columns,
                        optional_columns=dataset.optional_columns,
                        **(
                            {
                                "num_concurrent_spans": 1,
                                "reshuffle_spans_per_epoch": False,
                            }
                            if read_in_order
                            else {}
                        ),
                    ),
                    processor=processor,
                ),
                weight=dataset.weight,
            )
        )
    return KeyedDatasetMixConfig(
        datasets=tuple(children),
        aliases=tuple(dataset.alias for dataset in datasets),
    )


def make_pretrain_dataloader_config(
    datasets: list[DatasetSpec],
    *,
    packing: str = "concat_then_split",
    num_packing_bins: int = 8,
    shuffle: bool = False,
    repeat: bool = True,
    read_in_order: bool = False,
    seed: int = 0,
    **loader_kwargs: Any,
) -> KeyedMixDataLoader.Config:
    """Build a complete dataloader config from the dataset record list.

    ``concat_then_split`` mixes documents and then packs them, so a weight selects
    DOCUMENTS: a corpus of long documents takes more tokens than its weight.
    ``concat_then_split_per_dataset`` packs each dataset into its own rows of
    ``max_context_length`` tokens and mixes those rows, so a weight is that dataset's
    share of TOKENS (the data README's "pack each child before mixing"); a microbatch
    joins consecutive mixed rows, and the DP ranks of a step take consecutive rows of
    the mix (`PerDatasetRowsMixConfig`), so every step follows the weights.

    ``packing`` defaults to ``concat_then_split`` deliberately. Measured on a real
    long-document corpus at num_tokens_per_batch=8192, ``first_fit`` silently DROPS every
    document longer than ``max_context_length`` -- 41% of documents and 60.7% of all
    tokens -- while concat-then-split splits them across rows and loses nothing, with
    zero padding. Choose ``first_fit`` only when documents must stay whole (SFT
    conversations), and then size ``max_context_length`` above the corpus's long tail and
    raise ``num_packing_bins`` well above the default 8 (8 bins -> 7.7% padding, 32 ->
    2.4%).

    ``shuffle=False`` is correct for pre-shuffled corpora and keeps each rank's parquet
    reads sequential. Note it does NOT mean "replay the same order every epoch": the
    source permutes its span order per epoch regardless (`reshuffle_spans_per_epoch`).

    ``read_in_order=True`` turns that off as well, so the corpus is traversed in disk
    order. **Use it for every validator.** A validator is a measuring stick: it must score
    the same held-out tokens on every run, or a change to the training path appears to
    move val loss when all that moved was which tokens were sampled. That happened here --
    an A/B whose two arms had differently-configured validators reported a spurious
    +0.027 that vanished once both scored the same tokens.
    """
    mix = build_mix(datasets, read_in_order=read_in_order)
    if packing == "concat_then_split":
        packed: Any = ConcatThenSplitPackingConfig(dataset=mix)
    elif packing == "concat_then_split_per_dataset":
        packed = PerDatasetRowsMixConfig(
            dataset=replace(
                mix,
                datasets=tuple(
                    replace(
                        item, dataset=ConcatThenSplitPackingConfig(dataset=item.dataset)
                    )
                    for item in mix.datasets
                ),
            )
        )
    elif packing == "first_fit":
        packed = FirstFitPackingConfig(dataset=mix, num_packing_bins=num_packing_bins)
    elif packing == "none":
        packed = mix
    else:
        raise ValueError(
            f"unknown packing {packing!r}; expected concat_then_split, "
            "concat_then_split_per_dataset, first_fit or none"
        )

    return KeyedMixDataLoader.Config(
        dataset=packed,
        collator=TextCollator.Config(),
        dataset_ids=tuple(dataset.alias for dataset in datasets),
        dataset_weights=tuple(float(dataset.weight) for dataset in datasets),
        shuffle=shuffle,
        repeat=repeat,
        seed=seed,
        **loader_kwargs,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class PerDatasetRowsMixConfig:
    """Mix rows packed per dataset, then join each rank's rows into microbatches.

    The children of ``dataset`` pack their own documents into rows of
    ``max_context_length`` tokens; the mix interleaves those rows by weight, and every
    ``num_tokens_per_microbatch / max_context_length`` consecutive rows become one
    microbatch. Mixing whole microbatches instead put one dataset in each microbatch,
    and with grain's index-only selection every DP rank drew the same dataset at the
    same step: a whole global batch from one dataset, and a train loss that jumped
    between the datasets' loss levels (8B study, 11-10). Each row starts its own
    document segment (positions restart at 0), as rows do in any packed microbatch.

    Example, 4 rows per microbatch, 2 DP ranks, mix sequence A A B C A D B A ...::

        step 1: rank 0 rows A A B C   rank 1 rows A D B A
    """

    dataset: DatasetMixConfig

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> Any:
        rows_per_microbatch, rest = divmod(
            context.num_tokens_per_microbatch, context.max_context_length
        )
        if context.max_num_documents is not None:
            raise ValueError(
                "per-dataset rows do not support max_num_documents (a per-microbatch "
                "cap; each row would need its share of it)"
            )
        if rest or rows_per_microbatch < 1:
            raise ValueError(
                f"num_tokens_per_microbatch {context.num_tokens_per_microbatch} must be a "
                f"multiple of max_context_length {context.max_context_length}"
            )
        rows = self.dataset.build(
            context=replace(
                context,
                num_tokens_per_microbatch=context.max_context_length,
                mix_rank_stride=rows_per_microbatch,
            ),
            dataset_iteration_policy=dataset_iteration_policy,
        )
        return rows.batch(rows_per_microbatch, drop_remainder=True, batch_fn=_join_rows)


def _join_rows(rows: list[TextSequence]) -> TextSequence:
    def join(field: str) -> np.ndarray | None:
        parts = [getattr(row, field) for row in rows]
        return None if parts[0] is None else np.concatenate(parts)

    return TextSequence(
        input_ids=join("input_ids"),
        labels=join("labels"),
        positions=join("positions"),
        padding_mask=join("padding_mask"),
    )


def replace_columns(
    dataset: DatasetSpec,
    columns: tuple[str, ...],
    optional_columns: tuple[str, ...] = (),
) -> DatasetSpec:
    """Copy a record with a different column selection.

    `columns` must exist and are validated at startup; `optional_columns` are decoded
    only when present, so a corpus without `tools` or `enable_thinking` still works while
    a missing or mistyped `messages` still fails immediately.
    """
    return DatasetSpec(
        alias=dataset.alias,
        path=dataset.path,
        weight=dataset.weight,
        text_key=dataset.text_key,
        manifest_path=dataset.manifest_path,
        columns=columns,
        optional_columns=optional_columns,
        row_adapter=dataset.row_adapter,
    )
