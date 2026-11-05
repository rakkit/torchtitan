# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offset-aware validation data for offline loss and BPB evaluation."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info, IterableDataset

from torchtitan.components.data.mix import DatasetSpec
from torchtitan.components.data.parquet_stream import ParquetStreamSource
from torchtitan.components.data.types import DatasetIterationPolicy
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.evaluation.config import EvaluationConfigError


CharSpan = tuple[int, int]
PackedSample = tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class TokenizedDocument:
    """Token ids plus raw UTF-8 bytes attributed to every token's source span.

    A byte length is zero for special tokens and for tokenizer tokens whose
    character offset is empty. Positive entries come only from that token's
    non-overlapping raw-text character span.
    """

    token_ids: np.ndarray
    token_byte_lengths: np.ndarray

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1 or self.token_byte_lengths.ndim != 1:
            raise EvaluationConfigError(
                "token ids and byte lengths must be one-dimensional"
            )
        if len(self.token_ids) != len(self.token_byte_lengths):
            raise EvaluationConfigError(
                "token ids and byte lengths must have the same number of entries"
            )
        if np.any(self.token_byte_lengths < 0):
            raise EvaluationConfigError("token byte lengths must be non-negative")


Document = TokenizedDocument


def tokenize_document_with_byte_spans(
    tokenizer: BaseTokenizer, text: str
) -> TokenizedDocument:
    """Tokenize text and assign every token its exact raw UTF-8 span length.

    The normal TorchTitan tokenizer interface returns ids only. Offline BPB
    additionally needs character offsets. Standard ``HuggingFaceTokenizer``
    instances expose the underlying Rust ``tokenizers`` encoder, whose offsets
    are character positions in the original input. An evaluation-specific
    tokenizer may alternatively provide ``encode_with_offsets`` returning
    ``(token_ids, [(start_char, end_char), ...])``.
    """

    token_ids, char_offsets = _encode_with_char_offsets(tokenizer, text)
    if len(token_ids) != len(char_offsets):
        raise EvaluationConfigError(
            "tokenizer returned a different number of token ids and character offsets"
        )

    byte_boundaries = [0]
    for character in text:
        byte_boundaries.append(byte_boundaries[-1] + len(character.encode("utf-8")))

    # Byte-level BPE tokens don't have to align to character boundaries: a
    # token's raw bytes can start partway through one character and run into
    # the next (very common for CJK/rare-script text, where one character
    # frequently isn't a single vocabulary token). Character-granular offsets
    # can't express "half a character", so the tokenizer reports such a
    # token's span as every character its bytes touch -- which legitimately
    # overlaps its neighbors' spans. To get an exact, non-double-counting
    # byte total we do a coverage sweep: each token is credited only for the
    # *new* character territory beyond the highest point already credited to
    # an earlier token. This is exact (every byte of the document is
    # credited to exactly one token) and handles both an exact-repeat span
    # (a whole character split across N tokens) and a partial-overlap span
    # (a token straddling a character boundary) with the same rule.
    token_byte_lengths: list[int] = []
    last_covered_end = 0
    for offset in char_offsets:
        start, end = _validate_char_offset(offset, text_length=len(text))
        if start == end or end <= last_covered_end:
            token_byte_lengths.append(0)
            continue
        credit_start = max(start, last_covered_end)
        token_byte_lengths.append(byte_boundaries[end] - byte_boundaries[credit_start])
        last_covered_end = end

    return TokenizedDocument(
        token_ids=np.asarray(token_ids, dtype=np.int64),
        token_byte_lengths=np.asarray(token_byte_lengths, dtype=np.int64),
    )


def _encode_with_char_offsets(
    tokenizer: BaseTokenizer, text: str
) -> tuple[list[int], list[CharSpan]]:
    custom_encoder = getattr(tokenizer, "encode_with_offsets", None)
    if callable(custom_encoder):
        encoded = custom_encoder(text, add_bos=True, add_eos=True)
        if not isinstance(encoded, tuple) or len(encoded) != 2:
            raise EvaluationConfigError(
                "tokenizer.encode_with_offsets must return (token_ids, character_offsets)"
            )
        token_ids, char_offsets = encoded
        return list(token_ids), list(char_offsets)

    backend = getattr(tokenizer, "tokenizer", None)
    if backend is None or not callable(getattr(backend, "encode", None)):
        raise EvaluationConfigError(
            "offline BPB evaluation requires tokenizer character offsets; use the "
            "standard HuggingFaceTokenizer or implement encode_with_offsets"
        )

    backend_encoding = backend.encode(text)
    backend_ids = getattr(backend_encoding, "ids", None)
    backend_offsets = getattr(backend_encoding, "offsets", None)
    if backend_ids is None or backend_offsets is None:
        raise EvaluationConfigError(
            "tokenizer backend does not expose character offsets required for BPB"
        )

    # Match the exact ids used by the regular training dataloader. The wrapper
    # can add BOS/EOS around the backend encoding; those inserted special tokens
    # deliberately have an empty raw-text span.
    token_ids = list(tokenizer.encode(text, add_bos=True, add_eos=True))
    return token_ids, _align_backend_offsets(
        token_ids, list(backend_ids), list(backend_offsets)
    )


def _align_backend_offsets(
    token_ids: list[int], backend_ids: list[int], backend_offsets: list[Any]
) -> list[CharSpan]:
    if len(backend_ids) != len(backend_offsets):
        raise EvaluationConfigError("tokenizer backend returned malformed offsets")

    offsets: list[CharSpan] = []
    backend_index = 0
    for token_id in token_ids:
        if backend_index < len(backend_ids) and token_id == backend_ids[backend_index]:
            offset = backend_offsets[backend_index]
            if not isinstance(offset, (tuple, list)) or len(offset) != 2:
                raise EvaluationConfigError(
                    "tokenizer backend returned malformed offsets"
                )
            offsets.append((offset[0], offset[1]))
            backend_index += 1
        else:
            # Only wrapper-added special tokens may be absent from the backend
            # sequence. They have no source-text bytes by definition.
            offsets.append((0, 0))
    if backend_index != len(backend_ids):
        raise EvaluationConfigError(
            "could not align training tokenizer ids with backend offsets; exact BPB "
            "accounting is unavailable for this tokenizer"
        )
    return offsets


def _validate_char_offset(offset: Any, *, text_length: int) -> CharSpan:
    if not isinstance(offset, (tuple, list)) or len(offset) != 2:
        raise EvaluationConfigError("tokenizer returned a malformed character offset")
    start, end = offset
    if not isinstance(start, Integral) or not isinstance(end, Integral):
        raise EvaluationConfigError("tokenizer character offsets must be integers")
    start, end = int(start), int(end)
    if not 0 <= start <= end <= text_length:
        raise EvaluationConfigError(
            f"tokenizer offset {(start, end)} is outside source text length {text_length}"
        )
    return start, end


class RawTextParquetDataset(IterableDataset[Document]):
    """Yield offset-aware tokenized documents from one manifest-described corpus.

    Reads through `ParquetStreamSource`, the same source the training path uses, so
    evaluation and training agree about sharding and about what "this corpus" means.
    Only the byte attribution below is evaluation-specific -- BPB needs per-token byte
    lengths, which no packer in the training path produces.
    """

    def __init__(
        self,
        *,
        dataset: DatasetSpec,
        tokenizer: BaseTokenizer,
        dp_rank: int,
        dp_world_size: int,
    ) -> None:
        self.dataset_name = dataset.alias
        self.dataset_path = dataset.path
        self._text_key = dataset.text_key
        self._tokenizer = tokenizer
        # repeat=False: evaluation must terminate. Disk order, because a validation set
        # is a measuring stick and must score the same tokens on every run.
        self._source = ParquetStreamSource.Config(
            path=dataset.path,
            manifest_path=dataset.manifest_path,
            columns=(dataset.text_key,),
            num_concurrent_spans=1,
            reshuffle_spans_per_epoch=False,
        ).build().build_iter_dataset(
            dataset_iteration_policy=DatasetIterationPolicy(
                seed=0,
                shuffle=False,
                repeat=False,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                streaming_shuffle_buffer_size=0,
            )
        )

    def __iter__(self) -> Iterator[Document]:
        data_iter = iter(self._source)
        worker_info = get_worker_info()
        if worker_info is not None:
            data_iter = itertools.islice(
                data_iter, worker_info.id, None, worker_info.num_workers
            )

        for sample in data_iter:
            sample_text = sample[self._text_key]
            if not isinstance(sample_text, str):
                raise EvaluationConfigError(
                    f"validation dataset {self.dataset_name!r} produced non-string text"
                )
            if not sample_text.strip():
                continue
            yield tokenize_document_with_byte_spans(self._tokenizer, sample_text)


class ByteTrackingGreedyPackedDataset(IterableDataset[PackedSample]):
    """Mirror greedy token packing while retaining bytes of scored target spans."""

    def __init__(
        self,
        dataset: Iterable[Document],
        *,
        seq_len: int,
        drop_long_samples: bool,
    ) -> None:
        if seq_len <= 0:
            raise EvaluationConfigError("seq_len must be positive")
        self.dataset = dataset
        self.seq_len = seq_len
        self.drop_long_samples = drop_long_samples

    @property
    def _max_len(self) -> int:
        return self.seq_len + 1

    @staticmethod
    def _emit(tokens: np.ndarray, token_byte_lengths: np.ndarray) -> PackedSample:
        return (
            {"input": torch.from_numpy(tokens[:-1])},
            torch.from_numpy(tokens[1:].copy()),
            torch.tensor(token_byte_lengths[1:].sum(), dtype=torch.int64),
        )

    def __iter__(self) -> Iterator[PackedSample]:
        token_buffer: list[int] = []
        byte_buffer: list[int] = []
        for document in self.dataset:
            if self.drop_long_samples and len(document.token_ids) > self._max_len:
                continue
            token_buffer.extend(document.token_ids.tolist())
            byte_buffer.extend(document.token_byte_lengths.tolist())

            while len(token_buffer) >= self._max_len:
                tokens = np.asarray(token_buffer[: self._max_len], dtype=np.int64)
                token_bytes = np.asarray(byte_buffer[: self._max_len], dtype=np.int64)
                del token_buffer[: self._max_len]
                del byte_buffer[: self._max_len]
                yield self._emit(tokens, token_bytes)


def build_evaluation_dataloader(
    dataset: DatasetSpec,
    *,
    tokenizer: BaseTokenizer,
    dp_rank: int,
    dp_world_size: int,
    seq_len: int,
    local_batch_size: int,
    drop_long_samples: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """Build a finite offset-aware loader for greedy packing.

    One corpus per named validation set: scoring a MIXTURE would report a single number
    over an interleave whose proportions are a training-time decision, which is not what
    a validation set is for.
    """
    if num_workers > 0:
        # Workers shard DOCUMENTS round-robin inside the source, but the greedy packer
        # sits above them, so each worker packs its own subsequence and drops its own
        # tail: the scored token set changes with num_workers. A validation set is a
        # measuring stick, so refuse rather than let a throughput knob move the number.
        # (Each worker also decodes the rank's entire range and discards (N-1)/N of it.)
        raise EvaluationConfigError(
            f"num_workers={num_workers} changes which tokens an evaluation set scores, "
            "because packing happens above the per-worker document sharding. Use 0."
        )

    source = RawTextParquetDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
    packed: Iterable[PackedSample] = ByteTrackingGreedyPackedDataset(
        source,
        seq_len=seq_len,
        drop_long_samples=drop_long_samples,
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": local_batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(packed, **loader_kwargs)
