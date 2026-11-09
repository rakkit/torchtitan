# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graders for NeMo Gym rows that need no sandbox, dispatched on the row's environment.

``grade_mcqa`` and ``grade_structured_output`` are plain functions (no TitanRL
imports) so an eval harness can use the same graders.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import jsonschema

from torchtitan.rl.examples.dapo_math.math_verify_pool import MathVerifyPool
from torchtitan.rl.examples.nemo_gym.data import NemoGymSample
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rollout.rubric import RewardFn

MATH = "math_with_judge_simple_agent"
MCQA = "mcqa_simple_agent"
STRUCTURED_OUTPUTS = "structured_outputs_simple_agent"

_ANSWER_LINE = re.compile(r"Answer\s*:\s*\(?\s*([A-Za-z])\s*\)?", re.IGNORECASE)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def grade_mcqa(response: str, expected_answer: str) -> float:
    """1.0 when the last ``Answer: X`` line names the expected letter (the rows' prompt
    asks for that last line)."""
    found = _ANSWER_LINE.findall(response)
    return float(bool(found) and found[-1].upper() == expected_answer.strip().upper())


def grade_structured_output(response: str, schema_str: str) -> float:
    """1.0 when the answer is JSON (bare or in a code fence) valid against the schema."""
    fenced = _JSON_FENCE.findall(response)
    text = (fenced[-1] if fenced else response).strip()
    try:
        jsonschema.validate(json.loads(text), json.loads(schema_str))
    except (json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError):
        return 0.0
    return 1.0


class RewardNemoGym(RewardFn):
    """Scores each row with its environment's grader (math, multiple choice, JSON)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        math_timeout_seconds: float = 60.0
        """Math-Verify limit per answer (as RewardMathVerify)."""

        math_num_processes: int = 4
        """Math-Verify processes per rollout worker."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._math = MathVerifyPool(
            num_processes=config.math_num_processes,
            timeout_seconds=config.math_timeout_seconds,
        )

    async def __call__(self, rollout: Rollout, env_input: NemoGymSample) -> float:
        if not rollout.turns:
            return 0.0
        message = rollout.turns[-1].completion_message
        response = (message.get("content") or "") if message else ""
        fields = env_input.fields
        if env_input.agent == MATH:
            result = await self._math.score(
                response=response, ground_truth=str(fields["expected_answer"])
            )
            reward = result.score
        elif env_input.agent == MCQA:
            reward = grade_mcqa(response, str(fields["expected_answer"]))
        elif env_input.agent == STRUCTURED_OUTPUTS:
            reward = grade_structured_output(response, fields["schema_str"])
        else:
            raise ValueError(f"no grader for NeMo Gym environment {env_input.agent!r}")
        rollout.turns[-1].metrics.append(
            m.Metric(f"nemo_gym/{env_input.agent}/reward", m.Mean(reward))
        )
        return reward
