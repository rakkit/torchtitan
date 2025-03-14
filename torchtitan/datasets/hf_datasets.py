# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass

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
        seq_len: int = 2048,
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
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

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
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
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
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    dataset_inner_name = job_config.training.dataset_inner_name
    dataset_files = job_config.training.dataset_files
    dataset_split = job_config.training.dataset_split
    dataset_streaming = job_config.training.dataset_streaming
    dataset_key = job_config.training.dataset_key

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        dataset_inner_name=dataset_inner_name,
        dataset_files=dataset_files,
        dataset_split=dataset_split,
        dataset_streaming=dataset_streaming,
        dataset_key=dataset_key,
    )

    rng = torch.Generator()
    if job_config.training.seed is not None:
        rng.manual_seed(job_config.training.seed)
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
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=False,
        dataset_inner_name=dataset_inner_name,
        dataset_files=dataset_files,
        dataset_split=dataset_split,
        dataset_streaming=dataset_streaming,
        dataset_key=dataset_key,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=job_config.validation.dataset_num_workers,
        pin_memory=job_config.validation.dataset_pin_memory,
    )
