# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RL prompts in NeMo Gym's jsonl format (the NVIDIA Nemotron RL datasets).

Each row carries its opening messages in ``responses_create_params.input``, the
environment that scores it in ``agent_ref.name`` and that environment's fields
(``expected_answer``, ``schema_str``, ``instruction_id_list`` + ``kwargs``, ...).
One dataset mixes rows of several environments; the rubric scores each row with
its own environment's grader.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from renderers import Message

from torchtitan.rl.rollout.data import RLDataset


@dataclass(frozen=True, kw_only=True, slots=True)
class NemoGymSample:
    """One prompt and what its environment needs to score the answer."""

    agent: str
    """The scoring environment (``agent_ref.name``), e.g. ``mcqa_simple_agent``."""

    source: str
    """The row's ``dataset`` field (falls back to ``agent``): per-source metrics."""

    messages: list[Message]
    """Opening messages (``responses_create_params.input``)."""

    fields: dict[str, Any] = field(default_factory=dict)
    """The row's other fields (expected answer, schema, constraints, ...)."""


def read_nemo_gym_rows(
    paths: list[str], agents: list[str]
) -> tuple[list[NemoGymSample], dict[str, int]]:
    """Samples of the rows whose ``agent_ref.name`` is in ``agents``.

    Rows still holding a placeholder (``_hf_placeholder``,
    ``_hf_question_placeholder``: prompt text not filled in from their source
    dataset) are skipped. Returns the samples and the number of rows skipped
    per reason.
    """
    wanted = set(agents)
    samples: list[NemoGymSample] = []
    skipped = {"other_agent": 0, "placeholder": 0}
    for path in paths:
        with open(path) as rows:
            for line in rows:
                row = json.loads(line)
                agent = (row.get("agent_ref") or {}).get("name", "")
                if agent not in wanted:
                    skipped["other_agent"] += 1
                    continue
                if any(k.startswith("_hf_") and k.endswith("placeholder") for k in row):
                    skipped["placeholder"] += 1
                    continue
                params = row.pop("responses_create_params")
                row.pop("agent_ref")
                samples.append(
                    NemoGymSample(
                        agent=agent,
                        source=str(row.get("dataset") or agent),
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in params["input"]
                        ],
                        fields=row,
                    )
                )
    return samples, skipped


class NemoGymDataset(RLDataset):
    """An endless, resumable stream over NeMo Gym rows of the chosen environments."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        paths: list[str]
        """Local NeMo Gym jsonl files (compute nodes are offline)."""

        agents: list[str]
        """Environments to keep (``agent_ref.name``); the rubric must grade each."""

        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        self._samples, _ = read_nemo_gym_rows(config.paths, config.agents)
        if not self._samples:
            raise ValueError(f"no rows of {config.agents} in {config.paths}")
        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._position = 0

    def __iter__(self) -> Iterator[NemoGymSample]:
        return self

    def __next__(self) -> NemoGymSample:
        if self._position == len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._position = 0
        sample = self._samples[self._order[self._position]]
        self._position += 1
        return sample

    def state_dict(self) -> dict:
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "position": self._position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._position = state_dict["position"]
