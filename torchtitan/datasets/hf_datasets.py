# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return _load_simple_dataset(
        dataset_path,
        dataset_name="en",
        dataset_files=None,
        dataset_split=split,
        dataset_streaming=True,
    )


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


def _process_simple_text(sample: dict[str, Any], key: str) -> str:
    """Process a simple custom dataset's sample text."""
    return sample[key]


@dataclass
class DatasetArgs:
    path: str
    name: str | None
    files: str | Sequence[str] | None
    split: str
    streaming: bool
    key: str


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetArgs(
        path="allenai/c4",
        name="en",
        files=None,
        split="train",
        streaming=True,
        key="text",
    ),
    "c4_test": DatasetArgs(
        path="tests/assets/c4_test",
        name=None,
        files=None,
        split="train",
        streaming=False,
        key="text",
    ),
    "c4_validation": DatasetArgs(
        path="allenai/c4",
        name="en",
        files=None,
        split="validation",
        streaming=True,
        key="text",
    ),
    "fineweb": DatasetArgs(
        path="HuggingFaceFW/fineweb",
        name="default",
        files=None,
        split="train",
        streaming=True,
        key="text",
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
        assert dataset_path is not None
        config = DatasetArgs(
            path=dataset_path,
            name=dataset_inner_name,
            files=dataset_files,
            split=dataset_split,
            streaming=dataset_streaming,
            key=dataset_key,
        )
    if not isinstance(config, DatasetConfig):
        assert isinstance(config, DatasetArgs)
        old_config = config
        config = DatasetConfig(
            path=old_config.path,
            loader=lambda path: _load_simple_dataset(
                path,
                old_config.name,
                old_config.files,
                old_config.split,
                old_config.streaming,
            ),
            text_processor=lambda sample: _process_simple_text(sample, old_config.key),
        )

    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        dataset_inner_name: str | None = None,
        dataset_files: str | Sequence[str] | None = None,
        dataset_split: str = "train",
        dataset_streaming: bool = False,
        dataset_key: str = "text",
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_inner_name=dataset_inner_name,
            dataset_files=dataset_files,
            dataset_split=dataset_split,
            dataset_streaming=dataset_streaming,
            dataset_key=dataset_key,
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.infinite = infinite
        self._text_processor = text_processor

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
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._sample_idx += 1
                yield sample_tokens

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
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


class MixedDataset(IterableDataset, Stateful):
    def __init__(self, datasets: list[IterableDataset], weights: list[float] | None):
        self.datasets = datasets
        self.weights = [1.0] * len(self.datasets) if weights is None else weights

        self.num_sampled_per_dataset = [0] * len(self.datasets)
        self._dataset_indices = list(range(len(self.datasets)))
        self._sample_idx = 0
        self._data_iters = None
        self._rng = Random(self._sample_idx)

    @property
    def normed_weights(self):
        weights_sum = sum(self.weights)
        return [w / weights_sum for w in self.weights]

    def _init_data_iters(self):
        self._data_iters = [iter(dataset) for dataset in self.datasets]

    def _sample_dataset(self, sample_idx: int):
        self._rng.seed(sample_idx)
        dataset_index = self._rng.choices(self._dataset_indices, weights=self.weights)[
            0
        ]
        return dataset_index

    def _get_next(self, dataset_index: int):
        data_iter = self._data_iters[dataset_index]
        try:
            return next(data_iter)
        except StopIteration:
            dataset = self.datasets[dataset_index]
            logger.warning(f"Removing {dataset.dataset_name} from data mix.")
            self.weights[dataset_index] = 0.0
            return None

    def __iter__(self):
        if self._data_iters is None:
            self._init_data_iters()
        while True:
            sample = None
            # Handle exhausted data iterators.
            while sample is None:
                dataset_index = self._sample_dataset(self._sample_idx)
                sample = self._get_next(dataset_index)

            self.num_sampled_per_dataset[dataset_index] += 1
            self._sample_idx += 1
            yield sample

            if all(w == 0.0 for w in self.weights):
                logger.warning(
                    "Data mix is empty (all sampling weights have been set to zero); "
                    "stopping iteration."
                )
                break
        # Unset data iterators so they will be re-initialized.
        self._data_iters = None

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self.weights = state_dict["weights"]
        self.num_sampled_per_dataset = state_dict["num_sampled_per_dataset"]

        # Restore sub-datasets.
        dataset_dicts = state_dict["datasets"]
        for dataset in self.datasets:
            dataset.load_state_dict(dataset_dicts[dataset.dataset_name])

        # Unset data iterators so they will be re-initialized.
        self._data_iters = None

    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
            "weights": self.weights,
            "num_sampled_per_dataset": self.num_sampled_per_dataset,
            "datasets": {
                dataset.dataset_name: dataset.state_dict() for dataset in self.datasets
            },
        }


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

    def _get_data_iter(self):
        # We don't use the sample index because we defer skipping to the
        # sub-dataset.
        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len + self.num_mtp_tokens

        while True:
            for sample_tokens in self._get_data_iter():
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
                    f"Packed dataset {self.dataset_name} has run out of data"
                )
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Packed dataset {self.dataset_name} is being re-looped")
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


def build_hf_dataloader(
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
    num_mtp_tokens = job_config.training.num_mtp_tokens
    dataset_weights = job_config.training.dataset_weights
    dataset_mix_in_seq = job_config.training.dataset_mix_in_seq
    dataset_inner_name = _replace_none_with_literal(
        job_config.training.dataset_inner_name
    )
    dataset_files = job_config.training.dataset_files
    dataset_split = job_config.training.dataset_split
    dataset_streaming = job_config.training.dataset_streaming
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
    hf_datasets = []
    for (d_name, d_path, d_inner_name, d_split, d_key) in zip(
        dataset_name,
        dataset_path,
        dataset_inner_name,
        dataset_split,
        dataset_key,
    ):
        hf_ds = HuggingFaceDataset(
            dataset_name=d_name,
            dataset_path=d_path,
            tokenizer=tokenizer,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            dataset_inner_name=d_inner_name,
            dataset_files=dataset_files,
            dataset_split=d_split,
            dataset_streaming=dataset_streaming,
            dataset_key=d_key,
        )
        if not dataset_mix_in_seq:
            hf_ds = GreedyPackedDataset(
                dataset=hf_ds,
                seq_len=seq_len,
                infinite=infinite,
                num_mtp_tokens=num_mtp_tokens,
            )
        hf_datasets.append(hf_ds)

    # First pack, then mix → data is only mixed in batch dimension.
    # First mix, then pack → data is also mixed inside packed sample.
    hf_ds = MixedDataset(hf_datasets, dataset_weights)
    if dataset_mix_in_seq:
        hf_ds = GreedyPackedDataset(
            dataset=hf_ds,
            seq_len=seq_len,
            infinite=infinite,
            num_mtp_tokens=num_mtp_tokens,
        )

    if job_config.training.dataset_seed is None:
        job_config.training.dataset_seed = job_config.training.seed

    if job_config.training.dataset_shuffle_buffer_size:
        hf_ds = WindowShuffledDataset(
            hf_ds,
            buffer_size=job_config.training.dataset_shuffle_buffer_size,
            seed=job_config.training.dataset_seed,
        )

    rng = torch.Generator()
    if job_config.training.dataset_seed is not None:
        rng.manual_seed(job_config.training.dataset_seed)
    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=job_config.training.dataset_num_workers,
        pin_memory=job_config.training.dataset_pin_memory,
        generator=rng,
    )


def build_hf_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
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

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=False,
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

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=job_config.validation.dataset_num_workers,
        pin_memory=job_config.validation.dataset_pin_memory,
    )
