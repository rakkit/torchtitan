# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-Bench-style tasks (a Terminus-2 agent in a terminal, graded by the task's
own tests) on titan_eval's framework-free ``envs/terminal`` (``TerminusEnv.for_task``):
one Apptainer sandbox per rollout from the task's image (read-only) and its prepared
writable folders. The assistant's Terminus-2 JSON turns are plain assistant text, the
terminal screens come back as user messages, and after the last turn the task's hidden
tests run (``grade``) before the sandbox closes. The reward is the grade (0 or 1).

A task row (jsonl)::

    {"task": <name>, "task_dir": <instruction.md, task.toml, tests/>, "image": <image.sif>,
     "prep_dir": <tb_prepare output: task.env + template>, "grade_image": <optional>}

A TMax row (titan_eval tasksets/tmax: the shared stack image + the task's template) adds
``"layout": "tmax"`` with ``prep_dir`` = the template folder (``TerminusEnv.for_tmax``).
"""

from __future__ import annotations

import json
import os
import random
import socket
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from renderers import Message

from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rollout.data import RLDataset
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.rl.rollout.rubric import RewardFn

REWARD_KEY = "terminal/reward"


@dataclass(frozen=True, kw_only=True, slots=True)
class TerminalTaskSample:
    """One task: where its files, image and prepared folders are."""

    task: str
    task_dir: str
    image: str
    prep_dir: str
    grade_image: str | None = None
    layout: Literal["tb2", "tmax"] = "tb2"
    """``tb2`` (task image + tb_prepare folders) or ``tmax`` (stack image + template)."""
    task_set: str = "terminal"

    def __post_init__(self) -> None:
        if self.layout not in ("tb2", "tmax"):
            raise ValueError(
                f"task {self.task}: layout {self.layout!r} is neither 'tb2' nor 'tmax'"
            )

    @property
    def domain(self) -> str:
        return self.task_set

    @property
    def row_id(self) -> str | None:
        return f"{self.task_set}/{self.task}"


class TerminalTaskDataset(RLDataset):
    """An endless, resumable stream over task rows (one shuffled pass after another)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        paths: list[str]
        """Task-row jsonl files."""

        task_set: str = "terminal"
        """Name of the set for per-set metrics (e.g. tb2, tmax)."""

        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        self._samples = [
            TerminalTaskSample(task_set=config.task_set, **json.loads(line))
            for path in config.paths
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        if not self._samples:
            raise ValueError(f"no task rows in {config.paths}")
        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._position = 0

    def __iter__(self) -> Iterator[TerminalTaskSample]:
        return self

    def __next__(self) -> TerminalTaskSample:
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


class TerminalTaskEnv(MessageEnv):
    """``TerminusEnv.for_task`` per rollout. Its own limits end the episode (upstream's
    recipe: 120 turns, 7,200 s; set TokenEnv's ``max_num_turns`` above them); the step
    that ends it grades the task (tests staged only now) and reports ``terminal/reward``
    plus the episode's counters."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        titan_eval_lib: str
        """titan_eval's ``lib`` folder; the framework-free env is imported from it as
        ``titan_eval_envs.terminal``."""

        log_dir: str
        """Grading records: ONE JSONL file per actor process, <log_dir>/grades-<host>-<pid>.jsonl,
        one line per rollout (task, reward, per-test outcomes, output tails). No folder per
        rollout: at RL scale that is millions of files on the shared file system (11-11 limit)."""

    def __init__(self, config: Config, *, env_input: TerminalTaskSample) -> None:
        if config.titan_eval_lib not in sys.path:
            sys.path.insert(0, config.titan_eval_lib)
        from titan_eval_envs.terminal.env import TerminusEnv

        name = f"term-{uuid.uuid4().hex[:16]}"
        self._grade_image = env_input.grade_image
        log_file = str(
            Path(config.log_dir) / f"grades-{socket.gethostname()}-{os.getpid()}.jsonl"
        )
        if env_input.layout == "tmax":
            self._env = TerminusEnv.for_tmax(
                env_input.task_dir,
                env_input.prep_dir,
                None,
                name=name,
                log_file=log_file,
            )
        else:
            self._env = TerminusEnv.for_task(
                env_input.task_dir,
                env_input.image,
                env_input.prep_dir,
                None,
                name=name,
                log_file=log_file,
            )

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(init_prompt_messages=await self._env.init())

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        env_messages, done = await self._env.step(
            {"role": "assistant", "content": completion_message.get("content") or ""}
        )
        rewards = {
            f"terminal/{k}": float(v)
            for k, v in self._env.info.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if done:
            rewards[REWARD_KEY] = await self._env.grade(self._grade_image)
        return MessageEnvStepOutput(
            env_messages=env_messages, done=done, env_rewards=rewards
        )

    async def close(self) -> None:
        await self._env.close()


class RewardTerminalTask(RewardFn):
    """The task's grade from the last turn (0 when the episode never reached grading)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        if not rollout.turns:
            return 0.0
        return float(rollout.turns[-1].env_rewards.get(REWARD_KEY, 0.0))
