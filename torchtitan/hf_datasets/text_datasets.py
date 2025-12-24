# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from random import Random
from typing import Any, Callable

import torch
from datasets import Dataset, Features, Value, interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def _process_simple_text(sample: dict[str, Any], key: str) -> str:
    """Process a simple custom dataset's sample text."""
    return sample[key]


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return _process_simple_text(sample, "text")


def _load_simple_dataset(
    dataset_path: str,
    dataset_name: str | None,
    dataset_files: str | Sequence[str] | None,
    dataset_split: str,
    dataset_streaming: bool,
):
    """Load a simple custom dataset with its configuration."""
    return load_dataset(
        dataset_path,
        name=dataset_name,
        data_files=dataset_files,
        split=dataset_split,
        streaming=dataset_streaming,
    )


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return _load_simple_dataset(
        dataset_path,
        dataset_name="en",
        dataset_files=None,
        dataset_split=split,
        dataset_streaming=True,
    )


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, dataset_split="train"),
        text_processor=partial(_process_simple_text, key="text"),
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=partial(_load_simple_dataset, dataset_split="train"),
        text_processor=partial(_process_simple_text, key="text"),
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, dataset_split="validation"),
        text_processor=partial(_process_simple_text, key="text"),
    ),
    "simple_custom": None,
}


def _validate_dataset(
    dataset_name: str,
    dataset_path: str | None,
    dataset_inner_name: str | None,
    dataset_files: str | Sequence[str] | None,
    dataset_split: str,
    dataset_streaming: bool,
    dataset_key: str,
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    if config is None:
        # that goes to simple_custom, we need to read everything from the config
        assert dataset_path is not None
        config = DatasetConfig(
            path=dataset_path,
            loader=lambda path: _load_simple_dataset(
                path,
                dataset_inner_name,
                dataset_files,
                dataset_split,
                dataset_streaming,
            ),
            text_processor=lambda sample: _process_simple_text(sample, dataset_key),
        )
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: list[str] | str,
        job_config: JobConfig,
        dataset_path: list[str] | str | None,
        tokenizer: BaseTokenizer,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        dataset_inner_name: list[str] | str | None = None,
        dataset_files: str | Sequence[str] | None = None,
        dataset_split: list[str] | str = "train",
        dataset_streaming: bool = False,
        dataset_key: list[str] | str = "text",
    ) -> None:

        dataset_name = (
            dataset_name if isinstance(dataset_name, list) else [dataset_name]
        )
        # Force lowercase for consistent comparison
        dataset_name = [dn.lower() for dn in dataset_name]
        dataset_path = (
            dataset_path if isinstance(dataset_path, list) else [dataset_path]
        )
        dataset_inner_name = (
            dataset_inner_name
            if isinstance(dataset_inner_name, list)
            else [dataset_inner_name]
        )
        dataset_split = (
            dataset_split if isinstance(dataset_split, list) else [dataset_split]
        )
        dataset_key = dataset_key if isinstance(dataset_key, list) else [dataset_key]

        assert (
            len(dataset_name)
            == len(dataset_path)
            == len(dataset_inner_name)
            == len(dataset_split)
            == len(dataset_key)
        )
        ds_list = []
        text_processor_list = []
        for dataset in range(len(dataset_name)):
            (
                d_path,
                load_fn,
                text_processor,
            ) = _validate_dataset(
                dataset_name[dataset],
                dataset_path[dataset],
                dataset_inner_name[dataset],
                dataset_files,
                dataset_split[dataset],
                dataset_streaming,
                dataset_key[dataset],
            )
            ds_list.append(load_fn(d_path))
            text_processor_list.append(text_processor)

        dataset_weights = job_config.training.dataset_weights
        dataset_weights = (
            [1.0] * len(dataset_path)
            if dataset_weights is None
            # Convert to floats.
            else list(map(float, dataset_weights))
        )

        # Define the explicit schema
        new_features = ds_list[0].features.copy()
        new_features["text"] = Value("large_string")

        # Apply to all
        ds_list = [ds.cast(new_features) for ds in ds_list]

        ds = interleave_datasets(
            ds_list,
            probabilities=dataset_weights,
            seed=job_config.training.dataset_seed,
            stopping_strategy="all_exhausted",
        )

        logger.info("Splitting dataset by data parallel rank and world size")
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.infinite = infinite
        # TODO Need to pick one processor since after interleaving
        self._text_processor = text_processor[0]

        # Variables for checkpointing
        self._sample_idx = 0

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        while True:
            num_yielded = 0
            for sample in self._get_data_iter():
                self._sample_idx += 1
                # Use the dataset-specific text processor
                try:
                    sample_text = self._text_processor(sample)
                except Exception:
                    # bad row / missing key / load error -> skip, but state is correct
                    continue

                # Skip None / empty
                if sample_text is None:
                    continue
                if isinstance(sample_text, str) and not sample_text.strip():
                    continue

                try:
                    sample_tokens = self._tokenizer.encode(
                        sample_text, add_bos=True, add_eos=True
                    )
                except Exception:
                    # tokenization error -> skip, state is correct
                    continue

                num_yielded += 1
                yield sample_tokens

            if not self.infinite:
                logger.warning(
                    f"HuggingFaceDataset {self.dataset_name} from {self.dataset_path} has run out of data"
                )
                break
            elif num_yielded == 0:
                # "HuggingFaceDataset (shard on this rank) yielded 0 samples. Stopping iteration to prevent infinite loop."
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(
                    f"HuggingFaceDataset {self.dataset_name} from {self.dataset_path} is being re-looped"
                )
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


class GreedyPackedDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset: IterableDataset,
        seq_len: int = 2048,
        infinite: bool = False,
        num_mtp_tokens: int = 0,
    ) -> None:
        self._data = dataset
        self.seq_len = seq_len
        self.infinite = infinite
        self.num_mtp_tokens = num_mtp_tokens

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    @property
    def dataset_name(self):
        return self._data.dataset_name

    @property
    def dataset_path(self):
        return self._data.dataset_path

    def _get_data_iter(self):
        # We don't use the sample index because we defer skipping to the
        # sub-dataset.
        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len + self.num_mtp_tokens

        while True:
            num_yielded = 0
            for sample_tokens in self._get_data_iter():
                num_yielded += 1
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(
                    f"GreedyPackedDataset {self.dataset_name} from {self.dataset_path} has run out of data"
                )
                break
            elif num_yielded == 0:
                # "GreedyPackedDataset (shard on this rank) yielded 0 samples. Stopping iteration to prevent infinite loop."
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(
                    f"GreedyPackedDataset {self.dataset_name} from {self.dataset_path} is being re-looped"
                )
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._token_buffer = state_dict["token_buffer"]
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {
            "token_buffer": self._token_buffer,
            "sample_idx": self._sample_idx,
            "dataset": self._data.state_dict(),
        }


class WindowShuffledDataset(IterableDataset, Stateful):
    # Implementation highly inspired by
    # `torch.utils.data.datapipes.iter.ShufflerIterDataPipe`.

    def __init__(
        self,
        dataset: IterableDataset,
        *,
        buffer_size: int = 10000,
        seed: int | None = 0,
    ) -> None:
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.dataset = dataset
        self._buffer = []
        self.buffer_size = buffer_size
        self._enabled = True
        self._initial_seed = seed
        self._rng = Random(self._initial_seed)

    def set_shuffle(self, shuffle: bool = True):
        self._enabled = shuffle
        return self

    def set_initial_seed(self, seed: int | None = None):
        self._initial_seed = seed
        self._rng.seed(self._initial_seed)
        return self

    def __iter__(self):
        if not self._enabled:
            yield from self.dataset
        else:
            for x in self.dataset:
                if len(self._buffer) >= self.buffer_size:
                    idx = self._rng.randint(0, len(self._buffer) - 1)
                    val, self._buffer[idx] = self._buffer[idx], x
                    yield val
                else:
                    self._buffer.append(x)
            while self._buffer:
                idx = self._rng.randint(0, len(self._buffer) - 1)
                yield self._buffer.pop(idx)

    def reset(self) -> None:
        self._buffer = []
        self._rng.seed(self._initial_seed)

    def load_state_dict(self, state_dict):
        def list_tree_to_tuple(obj):
            if isinstance(obj, list):
                return tuple(list_tree_to_tuple(x) for x in obj)
            return obj

        # This should not be required and doesn't pop up during testing,
        # but we add it for safety.
        state_dict["rng_state"] = list_tree_to_tuple(state_dict["rng_state"])

        self._buffer = state_dict["shuffle_buffer"]
        self._initial_seed = state_dict["initial_seed"]
        self._enabled = state_dict["enabled"]
        self._rng.setstate(state_dict["rng_state"])
        self.dataset.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {
            "shuffle_buffer": self._buffer,
            "initial_seed": self._initial_seed,
            "enabled": self._enabled,
            "rng_state": self._rng.getstate(),
            "dataset": self.dataset.state_dict(),
        }


def _normalize_list(
    xs: list[str | None] | None,
    length: int,
    duplicate: bool = False,
) -> list[str | None]:
    if xs is None:
        xs = [None] * length
    elif duplicate and len(xs) == 1:
        xs = [xs[0] for _ in range(length)]
    return xs


def _replace_none_with_literal(xs: list[str] | None) -> list[str | None] | None:
    if xs is None:
        xs = None
    else:
        xs = [None if x == "None" else x for x in xs]
    return xs


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = _replace_none_with_literal(job_config.training.dataset_path)
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    rng = torch.Generator()
    dataset_streaming = job_config.training.dataset_streaming

    if job_config.training.dataset_seed is not None:
        rng.manual_seed(job_config.training.dataset_seed)

    if job_config.training.running_sft_training:
        from torchtitan.hf_datasets.sft_text_datasets import SFTDataset

        sft_data_config = job_config.sft_data_config
        # TODO: Improving the dataset loading, its easy to fix
        dataset_split = sft_data_config.split
        dataset_subset = sft_data_config.dataset_subset
        dataset_path = (
            dataset_path[0] if isinstance(dataset_path, list) else dataset_path
        )
        dataset = load_dataset(
            dataset_path,
            dataset_subset,
            split=dataset_split,
            streaming=dataset_streaming,
        )
        hf_ds = SFTDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            sft_data_config=sft_data_config,
        )
        collate_fn = hf_ds.collate_fn
        return ParallelAwareDataloader(
            dataset=hf_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=batch_size,
            num_workers=job_config.training.dataset_num_workers,
            pin_memory=job_config.training.dataset_pin_memory,
            generator=rng,
            collate_fn=collate_fn,
        )

    num_mtp_tokens = job_config.training.num_mtp_tokens
    dataset_weights = job_config.training.dataset_weights
    dataset_mix_in_seq = job_config.training.dataset_mix_in_seq
    dataset_inner_name = _replace_none_with_literal(
        job_config.training.dataset_inner_name
    )
    dataset_files = job_config.training.dataset_files
    dataset_split = job_config.training.dataset_split
    dataset_key = job_config.training.dataset_key

    normed_list_length = len(dataset_name)
    dataset_path = _normalize_list(dataset_path, normed_list_length)
    dataset_inner_name = _normalize_list(dataset_inner_name, normed_list_length)
    dataset_split = _normalize_list(dataset_split, normed_list_length)
    dataset_key = _normalize_list(dataset_key, normed_list_length)
    dataset_weights = (
        [1.0] * normed_list_length
        if dataset_weights is None
        # Convert to floats.
        else list(map(float, dataset_weights))
    )

    if len(dataset_name) > 1:
        assert (
            dataset_files is None
        ), "cannot supply dataset files when using multiple datasets"
    for d in [
        dataset_path,
        dataset_inner_name,
        dataset_split,
        dataset_key,
        dataset_weights,
    ]:
        assert (
            len(d) == normed_list_length
        ), f"list {d} does not match length of list of datasets (length = {normed_list_length})"

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        job_config=job_config,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        dataset_inner_name=dataset_inner_name,
        dataset_files=dataset_files,
        dataset_split=dataset_split,
        dataset_streaming=dataset_streaming,
        dataset_key=dataset_key,
    )

    # First pack, then mix → data is only mixed in batch dimension.
    # First mix, then pack → data is also mixed inside packed sample.
    if not dataset_mix_in_seq:
        hf_ds = GreedyPackedDataset(
            dataset=hf_ds,
            seq_len=seq_len,
            infinite=infinite,
            num_mtp_tokens=num_mtp_tokens,
        )

    if job_config.training.dataset_seed is None:
        job_config.training.dataset_seed = job_config.debug.seed

    if job_config.training.dataset_shuffle_buffer_size:
        hf_ds = WindowShuffledDataset(
            hf_ds,
            buffer_size=job_config.training.dataset_shuffle_buffer_size,
            seed=job_config.training.dataset_seed,
        )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=job_config.training.dataset_num_workers,
        pin_memory=job_config.training.dataset_pin_memory,
        generator=rng,
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets."""
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len
    dataset_inner_name = job_config.validation.dataset_inner_name
    dataset_files = job_config.validation.dataset_files
    dataset_split = job_config.validation.dataset_split
    dataset_streaming = job_config.validation.dataset_streaming
    dataset_key = job_config.validation.dataset_key

    collate_fn = None
    if not job_config.training.running_sft_training:
        hf_ds = HuggingFaceDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            dataset_inner_name=dataset_inner_name,
            dataset_files=dataset_files,
            dataset_split=dataset_split,
            dataset_streaming=dataset_streaming,
            dataset_key=dataset_key,
        )

        hf_ds = GreedyPackedDataset(
            dataset=hf_ds,
            seq_len=seq_len,
            infinite=False,
        )
    else:
        from torchtitan.hf_datasets.sft_text_datasets import SFTDataset

        sft_data_config = job_config.sft_data_config
        # TODO: Improving the dataset loading, its easy to fix
        dataset_split = sft_data_config.split
        dataset_subset = sft_data_config.dataset_subset
        dataset = load_dataset(
            dataset_path,
            dataset_subset,
            split=dataset_split,
            streaming=dataset_streaming,
        )
        hf_ds = SFTDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            sft_data_config=sft_data_config,
        )
        collate_fn = hf_ds.collate_fn

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=job_config.validation.dataset_num_workers,
        pin_memory=job_config.validation.dataset_pin_memory,
        collate_fn=collate_fn,
    )
