# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offset-aware validation data for offline loss and BPB evaluation."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import torch
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, get_worker_info, IterableDataset

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.evaluation.config import EvaluationConfigError
from torchtitan.hf_datasets.text_datasets import (
    _coerce_to_list,
    _normalize_list,
    _prepared_data_files,
    _replace_none_with_literal,
    _validate_dataset,
    HuggingFaceTextDataLoader,
)


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


class RawTextHuggingFaceDataset(IterableDataset[Document]):
    """Yield offset-aware tokenized documents from a finite distributed source."""

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        dp_rank: int,
        dp_world_size: int,
        dataset_inner_name: str | None,
        dataset_files: str | Sequence[str] | None,
        dataset_split: str,
        dataset_streaming: bool,
        dataset_key: str,
    ) -> None:
        dataset_name = dataset_name.lower()
        path, dataset_loader, sample_processor = _validate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_inner_name=dataset_inner_name,
            dataset_files=dataset_files,
            dataset_split=dataset_split,
            dataset_streaming=dataset_streaming,
            dataset_key=dataset_key,
        )
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self._data = split_dataset_by_node(dataset_loader(path), dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self._sample_processor = sample_processor

    def __iter__(self) -> Iterator[Document]:
        data_iter = iter(self._data)
        worker_info = get_worker_info()
        if worker_info is not None:
            data_iter = itertools.islice(
                data_iter, worker_info.id, None, worker_info.num_workers
            )

        for sample in data_iter:
            try:
                sample_text = self._sample_processor(sample)
            except Exception:
                continue
            if not isinstance(sample_text, str):
                raise EvaluationConfigError(
                    f"validation dataset {self.dataset_name!r} produced non-string text"
                )
            if not sample_text.strip():
                continue
            try:
                yield tokenize_document_with_byte_spans(self._tokenizer, sample_text)
            except EvaluationConfigError:
                raise
            except Exception:
                continue


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
    config: HuggingFaceTextDataLoader.Config,
    *,
    tokenizer: BaseTokenizer,
    dp_rank: int,
    dp_world_size: int,
    seq_len: int,
    local_batch_size: int,
) -> DataLoader:
    """Build a finite offset-aware loader for greedy packing."""

    if config.infinite:
        raise EvaluationConfigError(
            "offline validation requires dataloader.infinite=False"
        )
    if config.pack_strategy != "greedy":
        raise EvaluationConfigError(
            "offline validation supports dataloader.pack_strategy='greedy' only"
        )
    if config.dataset_mix_in_seq:
        raise EvaluationConfigError(
            "offline validation does not support dataset_mix_in_seq; define one named set "
            "per source dataset"
        )

    dataset_names = _coerce_to_list(config.dataset)
    if dataset_names is None or len(dataset_names) != 1:
        raise EvaluationConfigError(
            "offline validation supports exactly one source dataset per named validation set"
        )
    dataset_paths = _replace_none_with_literal(config.dataset_path)
    inner_names = _replace_none_with_literal(config.dataset_inner_name)
    dataset_splits = _normalize_list(
        _coerce_to_list(config.dataset_split), 1, duplicate=True
    )
    dataset_keys = _normalize_list(
        _coerce_to_list(config.dataset_key), 1, duplicate=True
    )
    if dataset_splits is None or dataset_keys is None:
        raise EvaluationConfigError("dataset_split and dataset_key must be configured")
    dataset_paths = _normalize_list(dataset_paths, 1)
    inner_names = _normalize_list(inner_names, 1)
    dataset_path = dataset_paths[0]
    if dataset_path is not None and not isinstance(dataset_path, str):
        raise EvaluationConfigError("dataset_path must be a string or None")
    dataset_files = (
        None
        if dataset_path is None
        else _prepared_data_files(
            dataset_path,
            config.dataset_files,
            dataset_splits[0],
            config.dataset_streaming,
        )
    )

    source = RawTextHuggingFaceDataset(
        dataset_name=dataset_names[0],
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        dataset_inner_name=inner_names[0],
        dataset_files=dataset_files,
        dataset_split=dataset_splits[0],
        dataset_streaming=config.dataset_streaming,
        dataset_key=dataset_keys[0],
    )
    packed: Iterable[PackedSample] = ByteTrackingGreedyPackedDataset(
        source,
        seq_len=seq_len,
        drop_long_samples=config.drop_long_samples,
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": local_batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(packed, **loader_kwargs)
