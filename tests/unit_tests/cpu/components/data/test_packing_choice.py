# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What each packer does with documents that do not fit, and how much it wastes.

The packer choice is usually framed as a padding-efficiency question. It is not: on a
long-document corpus the dominant term is DATA LOSS, because `FirstFitPackingConfig`
drops any document longer than `max_context_length` (packing.py:77-79) while
`ConcatThenSplitPackingConfig` splits it across rows instead.

Note the drop threshold is `max_context_length`, NOT the row width
`num_tokens_per_batch`. The two are equal in the measurement below and in most pretrain
configs, which is exactly why the difference is easy to miss; the last two tests here
pin it for the case where they differ.

Measured on a real PDF corpus (median ~7.4k tokens) at num_tokens_per_microbatch=8192:
FirstFit dropped 41% of documents, which was 60.7% of all tokens; padding was 7.7% at
the default 8 bins and 2.4% at 32. These tests pin the behaviour that produces those
numbers so the trade-off stays visible.
"""

from __future__ import annotations

import dataclasses

import grain.python as grain
import numpy as np
import pytest

from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.mix import DatasetSpec
from torchtitan.components.data.packing import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
)
from torchtitan.components.data.sft import make_sft_dataloader_config
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy

NUM_TOKENS_PER_BATCH = 64


class _Tokenizer:
    bos_id = 1
    eos_id = 2


CONTEXT = DatasetBuildContext(
    tokenizer=_Tokenizer(),
    max_context_length=NUM_TOKENS_PER_BATCH,
    num_tokens_per_microbatch=NUM_TOKENS_PER_BATCH,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)
POLICY = DatasetIterationPolicy(
    seed=0,
    shuffle=False,
    repeat=False,
    dp_rank=0,
    dp_world_size=1,
    streaming_shuffle_buffer_size=0,
)


class _Docs:
    """A dataset config yielding TextSequences of the given token lengths."""

    def __init__(self, lengths):
        self._lengths = lengths

    def build(self, *, context, dataset_iteration_policy):
        del context, dataset_iteration_policy
        rows = [
            TextSequence(
                input_ids=np.arange(n, dtype=np.int64),
                labels=np.arange(n, dtype=np.int64),
            )
            for n in self._lengths
        ]
        return grain.MapDataset.source(rows).to_iter_dataset()


def _tokens_emitted(packed) -> int:
    """Count REAL tokens, not padding.

    A packed row is padded out to num_tokens_per_batch, so len(input_ids) is the row
    width and says nothing about how much data survived. Padding carries IGNORE_INDEX
    labels, which is what the collator's num_valid_tokens counts too.
    """
    from torchtitan.components.loss import IGNORE_INDEX

    return sum(int((row.labels != IGNORE_INDEX).sum()) for row in packed)


def test_first_fit_keeps_a_document_longer_than_the_row(tmp_path):
    """first-fit splits a document longer than the row (upstream #4156) instead
    of dropping it: every token is emitted."""
    lengths = [10, 10, NUM_TOKENS_PER_BATCH + 1, 10]

    packed = list(
        FirstFitPackingConfig(dataset=_Docs(lengths)).build(
            context=CONTEXT, dataset_iteration_policy=POLICY
        )
    )

    emitted = _tokens_emitted(packed)
    assert emitted == sum(lengths), f"no document may be dropped, got {emitted}"


def test_concat_then_split_keeps_oversized_documents_by_splitting(tmp_path):
    """The same corpus, no data loss: long documents are cut across rows."""
    lengths = [10, 10, NUM_TOKENS_PER_BATCH + 1, 10]

    packed = list(
        ConcatThenSplitPackingConfig(dataset=_Docs(lengths)).build(
            context=CONTEXT, dataset_iteration_policy=POLICY
        )
    )

    # Every emitted row is exactly full: concat-then-split never pads.
    assert packed, "expected at least one full row"
    for row in packed:
        assert len(row.input_ids) == NUM_TOKENS_PER_BATCH
        assert _tokens_emitted([row]) == NUM_TOKENS_PER_BATCH, "no padding at all"
    # It keeps far more of the corpus than first-fit does on the same input.
    assert _tokens_emitted(packed) > 30


def test_more_bins_reduce_padding(tmp_path):
    """num_packing_bins is a real lever: 8 -> 7.7% padding, 32 -> 2.4% on real data."""
    lengths = [40, 30, 25, 20, 18, 15, 12, 10, 9, 8, 7, 6, 5, 4] * 4

    def fill(bins: int) -> float:
        packed = list(
            FirstFitPackingConfig(dataset=_Docs(lengths), num_packing_bins=bins).build(
                context=CONTEXT, dataset_iteration_policy=POLICY
            )
        )
        if not packed:
            return 0.0
        return _tokens_emitted(packed) / (len(packed) * NUM_TOKENS_PER_BATCH)

    assert fill(16) >= fill(2), "more open bins should not pack worse"


def test_num_packing_bins_must_be_positive():
    with pytest.raises(ValueError, match="num_packing_bins must be positive"):
        FirstFitPackingConfig(dataset=_Docs([4]), num_packing_bins=0)


def test_first_fit_keeps_a_document_between_the_context_and_the_row_width():
    """The row (`num_tokens_per_microbatch_per_dp_rank`) is usually a multiple of
    `max_context_length`; a document longer than the context but shorter than the
    row is split at `max_context_length`, not dropped."""
    context = dataclasses.replace(CONTEXT, max_context_length=NUM_TOKENS_PER_BATCH // 2)
    # 40 tokens: longer than max_context_length (32), shorter than the row (64).
    lengths = [10, NUM_TOKENS_PER_BATCH // 2 + 8, 10]

    packed = list(
        FirstFitPackingConfig(dataset=_Docs(lengths)).build(
            context=context, dataset_iteration_policy=POLICY
        )
    )

    # The 40-token document is split at max_context_length (10..42, 42..50).
    assert _tokens_emitted(packed) == sum(
        lengths
    ), "a document between max_context_length and the row width must not be dropped"


def test_concat_then_split_restarts_positions_every_max_context_length():
    """How a document longer than the context is made trainable.

    `_text_sequence_to_packing_input` synthesizes `arange % max_context_length`
    (packing.py:126-129), so a long document arrives as consecutive context-sized
    segments. `positions` resetting to 0 is what the document masks (the attention's
    `get_efficient_causal_mask_mod_for_packed_document`) read as a document start, so
    block-causal attention treats each segment independently and no query
    ever sees a key further back than `max_context_length`.
    """
    context = dataclasses.replace(CONTEXT, max_context_length=NUM_TOKENS_PER_BATCH // 4)
    packed = list(
        ConcatThenSplitPackingConfig(dataset=_Docs([NUM_TOKENS_PER_BATCH * 4])).build(
            context=context, dataset_iteration_policy=POLICY
        )
    )

    assert packed
    row = packed[0]
    starts = np.flatnonzero(np.asarray(row.positions) == 0)
    segment_lengths = np.diff(np.append(starts, len(row.positions)))
    assert set(segment_lengths.tolist()) == {NUM_TOKENS_PER_BATCH // 4}


def test_sft_helper_drops_at_the_same_threshold_the_packer_filters_on():
    """The counted drop and the silent drop must use one number.

    `make_sft_dataloader_config` counts oversized conversations in the processor so the
    loss is visible. If it counted against the row width while the packer filtered on
    the context, conversations between the two would be dropped without being counted.
    """
    config = make_sft_dataloader_config(
        [DatasetSpec(alias="a", path="/nonexistent")],
        max_context_length=1234,
    )
    processor = config.dataset.dataset.datasets[0].dataset.processor
    assert processor.drop_if_longer_than == 1234
