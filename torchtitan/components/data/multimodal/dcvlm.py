# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DataComp-VLM's WebDataset shards (``mlfoundations/dcvlm-baseline-*``): a
DP-sharded tar reader and the sample processor.

One sample is a group of tar members sharing a key::

    r00260_000008735.conversations.txt   human:<question> <image> <image><EOCL>
                                         gpt:<answer><EOCL> ...
    r00260_000008735.0.jpg, .1.jpg       the N-th ``<image>`` -> N.jpg
    r00260_000008735.ds_name.txt         source dataset (provenance)

``process_dcvlm_sample`` turns it into our chat template: ``human`` -> user,
``gpt`` -> assistant, each ``<image>`` -> ``<|vision_start|> <|image_pad|> x n
<|vision_end|>`` with n = 256 per InternVL tile (``internvl_tiling.py``), loss on
the assistant turns only (located like ``MultiTurnChatProcessor``: the whole
conversation is rendered once, each turn's span by re-rendering prefixes).
InternVL's placeholder rule (``internvl_chat_finetune.py``): a one-image sample
without ``<image>`` gets ``<image>\\n`` before its first turn; any other count
mismatch drops the sample.
"""

from __future__ import annotations

import glob
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grain
import numpy as np
import torch

from torchtitan.components.data.types import DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import Configurable

logger = logging.getLogger(__name__)

_ROLES = {"human": "user", "gpt": "assistant"}
_IMAGE = "<image>"


class DCVLMShardSource(Configurable):
    """Streams DataComp-VLM tar shards; DP rank r reads shards r, r + W, ...
    (the mix is globally shuffled, so contiguous shards are a random slice).
    Checkpoints the shard position and the samples read from it."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        shard_pattern: str
        """Glob of the tar shards, e.g. ``.../shards/*/*.tar``."""

        max_shards: int | None = None
        """Use only the first ``max_shards`` shards in sorted order (a slice of
        the shuffled mix); None: all."""

    def __init__(self, config: Config) -> None:
        paths = sorted(glob.glob(config.shard_pattern))
        if not paths:
            raise FileNotFoundError(f"no shards match {config.shard_pattern!r}")
        if config.max_shards is not None:
            if len(paths) < config.max_shards:
                raise ValueError(
                    f"{len(paths)} shards found, {config.max_shards} requested"
                )
            paths = paths[: config.max_shards]
            # By name, not by count: a shard missing below max_shards (a download
            # still running) would shift the sorted list, and a restart would map
            # every rank's checkpointed (shard, sample) state to other files.
            expected = [f"{i:06d}.tar" for i in range(config.max_shards)]
            if [Path(p).name for p in paths] != expected:
                missing = sorted(set(expected) - {Path(p).name for p in paths})
                raise ValueError(
                    f"shards missing below {config.max_shards}: {missing[:5]}"
                )
        self._paths = tuple(str(Path(p).resolve()) for p in paths)

    def build_iter_dataset(
        self, *, dataset_iteration_policy: DatasetIterationPolicy
    ) -> grain.IterDataset:
        rank, world = (
            dataset_iteration_policy.dp_rank,
            dataset_iteration_policy.dp_world_size,
        )
        mine = self._paths[rank::world]
        if not mine:
            raise ValueError(
                f"DP rank {rank} of {world} gets no shard ({len(self._paths)} shards)"
            )
        return _ShardIterDataset(mine, repeat=dataset_iteration_policy.repeat)


class _ShardIterDataset(grain.IterDataset):
    def __init__(self, paths: tuple[str, ...], *, repeat: bool) -> None:
        super().__init__()
        self._paths = paths
        self._repeat = repeat

    def __iter__(self) -> grain.DatasetIterator:
        return _ShardIterator(self._paths, repeat=self._repeat)


class _ShardIterator(grain.DatasetIterator):
    def __init__(self, paths: tuple[str, ...], *, repeat: bool) -> None:
        super().__init__()
        self._paths = paths
        self._repeat = repeat
        self._shard = 0  # position in self._paths (grows past len when repeating)
        self._sample = 0  # samples already returned from the current shard
        self._samples = None

    def _open(self) -> None:
        if self._shard >= len(self._paths) and not self._repeat:
            raise StopIteration
        self._samples = _read_shard(self._paths[self._shard % len(self._paths)])
        for _ in range(self._sample):  # resume: skip what was returned before
            next(self._samples)

    def __next__(self) -> dict[str, Any]:
        while True:
            if self._samples is None:
                self._open()
            try:
                sample = next(self._samples)
            except StopIteration:
                self._samples, self._shard, self._sample = None, self._shard + 1, 0
                continue
            self._sample += 1
            return sample

    def get_state(self) -> dict[str, Any]:
        return {"shard": self._shard, "sample": self._sample}

    def set_state(self, state: dict[str, Any]) -> None:
        self._shard, self._sample, self._samples = state["shard"], state["sample"], None


def _read_shard(path: str):
    """Yield ``{"key", "conversations", "images", "ds_name"}`` per sample of one
    tar; members of a sample are contiguous (WebDataset), images ordered by N."""

    def finish(key: str, members: dict[str, bytes]) -> dict[str, Any] | None:
        if "conversations.txt" not in members:
            return None
        image_ids = sorted(
            int(ext.split(".")[0]) for ext in members if ext.split(".")[0].isdigit()
        )
        images = [
            members[next(e for e in members if e.startswith(f"{i}."))]
            for i in image_ids
        ]
        return {
            "key": key,
            "conversations": members["conversations.txt"].decode("utf-8", "replace"),
            "images": images,
            "ds_name": members.get("ds_name.txt", b"")
            .decode("utf-8", "replace")
            .strip(),
        }

    key, members = None, {}
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.rsplit("/", 1)[-1]
            member_key, _, ext = name.partition(".")
            if member_key != key:
                if key is not None and (sample := finish(key, members)) is not None:
                    yield sample
                key, members = member_key, {}
            members[ext] = tar.extractfile(member).read()
    if key is not None and (sample := finish(key, members)) is not None:
        yield sample


def parse_conversation(text: str) -> list[dict[str, str]] | None:
    """``human:...<EOCL>\\ngpt:...<EOCL>`` -> chat messages; None if a turn has
    no known role."""
    messages = []
    for turn in text.split("<EOCL>"):
        turn = turn.strip("\n")
        if not turn.strip():
            continue
        role, sep, content = turn.partition(":")
        if not sep or role.strip() not in _ROLES:
            return None
        messages.append({"role": _ROLES[role.strip()], "content": content})
    return messages or None


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def process_dcvlm_sample(
    sample: dict[str, Any],
    tokenizer: Any,
    vision_processor: Any,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """One DataComp-VLM sample -> ``input_ids``, ``labels`` (assistant turns
    only), ``positions`` and the images' tiles (``pixel_values``); None drops it."""
    messages = parse_conversation(sample["conversations"])
    if (
        messages is None
        or messages[0]["role"] != "user"
        or not any(m["role"] == "assistant" for m in messages)
    ):
        return None
    if any(
        "<|" in m["content"] for m in messages
    ):  # text that would tokenize as our special tokens
        return None
    images = sample["images"]
    placeholders = sum(m["content"].count(_IMAGE) for m in messages)
    if placeholders == 0 and len(images) == 1:
        messages[0]["content"] = f"{_IMAGE}\n" + messages[0]["content"]
    elif placeholders != len(images):
        return None

    pixel_values, image_tokens = [], []
    for image in images:
        try:
            pixels, grid = vision_processor.resize_and_normalize(
                image, num_images_in_sample=len(images)
            )
        except Exception as e:  # unreadable image: drop the sample
            logger.warning(f"Dropping DCVLM sample {sample.get('key')}: {e}")
            return None
        pixel_values.append(pixels)
        image_tokens.append(vision_processor.num_placeholder_tokens(grid))

    block = f"{tokenizer.vision_start_token}{tokenizer.image_token}{tokenizer.vision_end_token}"
    for m in messages:
        m["content"] = m["content"].replace(_IMAGE, block)

    def render(msgs: list[dict[str, str]], **extra: Any) -> list[int]:
        text = tokenizer.apply_chat_template(
            msgs, enable_thinking=False, truncate_history_thinking=False, **extra
        )
        return tokenizer.encode(text, add_bos=False, add_eos=False)

    try:  # the template refuses some conversations (e.g. no user query): drop, never crash
        ids = render(messages)
        trained = np.zeros(len(ids), dtype=bool)
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            start = _common_prefix(
                render(messages[:i], add_generation_prompt=True), ids
            )
            stop = _common_prefix(render(messages[: i + 1]), ids)
            trained[start:stop] = True
    except Exception as e:
        logger.warning(f"Dropping DCVLM sample {sample.get('key')}: chat template: {e}")
        return None

    # Expand each image's single <|image_pad|> to its tile tokens (never trained).
    pad_id = tokenizer.image_id
    expanded, mask, image = [], [], 0
    for token, is_trained in zip(ids, trained):
        if token == pad_id:
            expanded.extend([pad_id] * image_tokens[image])
            mask.extend([False] * image_tokens[image])
            image += 1
        else:
            expanded.append(token)
            mask.append(bool(is_trained))
    if image != len(images) or not any(mask[1:]):
        return None

    input_ids = torch.tensor(expanded[:-1])
    labels = torch.tensor(expanded[1:])
    trained_T = torch.tensor(mask[1:])
    labels[~trained_T] = IGNORE_INDEX
    # DataComp-VLM / InternVL "square averaging" (Molmo2: 4 / sqrt(n)): every
    # trained token of the sample weighs 1 / sqrt(trained tokens of the sample).
    loss_weights = trained_T.float() / float(trained_T.sum()) ** 0.5
    return {
        "input_ids": input_ids,
        "labels": labels,
        "positions": torch.arange(len(input_ids)),
        "pixel_values": pixel_values,
        "loss_weights": loss_weights,
    }
