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

    tools: list[dict[str, Any]] = field(default_factory=list)
    """Tools the prompt offers (``responses_create_params.tools``), in the chat format
    ``{"type": "function", "function": {"name", "description", "parameters"}}`` with
    ``parameters`` a mapping, as vLLM passes them to the chat template; empty for most rows."""

    fields: dict[str, Any] = field(default_factory=dict)
    """The row's other fields (expected answer, schema, constraints, ...)."""

    @property
    def domain(self) -> str:
        """The environment: picks the teacher under multi-teacher distillation."""
        return self.agent

    @property
    def row_id(self) -> str | None:
        """The row's id, saved with each rollout to join it back to the dataset: its uuid,
        else the preparation tool's ``_row_id`` (source file + line)."""
        row_id = self.fields.get("uuid") or self.fields.get("_row_id")
        return None if row_id is None else str(row_id)


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
                tools = list(params.get("tools") or [])
                for tool in tools:
                    function = tool.get("function") if isinstance(tool, dict) else None
                    if not isinstance(function, dict) or not isinstance(
                        function.get("parameters"), dict
                    ):
                        raise ValueError(
                            f"{path}: a tool is not in the chat format with a parameters mapping "
                            f"({str(tool)[:120]}); convert it in data preparation"
                        )
                samples.append(
                    NemoGymSample(
                        agent=agent,
                        source=str(row.get("dataset") or agent),
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in params["input"]
                        ],
                        tools=tools,
                        fields=row,
                    )
                )
    return samples, skipped


class NemoGymDataset(RLDataset):
    """An endless, resumable stream over NeMo Gym rows of the chosen environments.

    Without ``weights`` the mix follows the rows' own proportions (one shuffled pass after
    another). With ``weights`` each sample first draws its environment by weight, then takes
    that environment's next row (each environment cycles through its own shuffled rows).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        paths: list[str]
        """Local NeMo Gym jsonl files (compute nodes are offline)."""

        agents: list[str]
        """Environments to keep (``agent_ref.name``); the rubric must grade each."""

        weights: dict[str, float] | None = None
        """Sampling weight per environment (normalized; one positive weight per agent);
        None: the mix follows the row counts."""

        seed: int = 42
        shuffle: bool = True

        def __post_init__(self) -> None:
            if self.weights is not None and (
                set(self.weights) != set(self.agents)
                or any(w <= 0 for w in self.weights.values())
            ):
                raise ValueError(
                    f"weights need one positive weight per agent {self.agents}, got {self.weights}"
                )

    def __init__(self, config: Config) -> None:
        self._samples, _ = read_nemo_gym_rows(config.paths, config.agents)
        if not self._samples:
            raise ValueError(f"no rows of {config.agents} in {config.paths}")
        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        if config.weights is None:
            streams = {"": list(range(len(self._samples)))}
            self._weights = {"": 1.0}
        else:
            streams = {agent: [] for agent in config.weights}
            for index, sample in enumerate(self._samples):
                streams[sample.agent].append(index)
            empty = [agent for agent, rows in streams.items() if not rows]
            if empty:
                raise ValueError(f"no rows of {empty} in {config.paths}")
            self._weights = dict(config.weights)
        for order in streams.values():
            if self._shuffle:
                self._rng.shuffle(order)
        self._orders = streams
        self._positions = dict.fromkeys(streams, 0)

    def __iter__(self) -> Iterator[NemoGymSample]:
        return self

    def __next__(self) -> NemoGymSample:
        names = list(self._weights)
        name = (
            names[0]
            if len(names) == 1
            else self._rng.choices(names, weights=[self._weights[n] for n in names])[0]
        )
        order = self._orders[name]
        if self._positions[name] == len(order):
            if self._shuffle:
                self._rng.shuffle(order)
            self._positions[name] = 0
        sample = self._samples[order[self._positions[name]]]
        self._positions[name] += 1
        return sample

    def state_dict(self) -> dict:
        return {
            "rng_state": self._rng.getstate(),
            "orders": {name: list(order) for name, order in self._orders.items()},
            "positions": dict(self._positions),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._orders = {
            name: list(order) for name, order in state_dict["orders"].items()
        }
        self._positions = dict(state_dict["positions"])
