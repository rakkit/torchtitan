# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graders for NeMo Gym rows that need no sandbox, dispatched on the row's environment.

Each grader follows the matching NeMo Gym resources server (``mcqa``,
``structured_outputs``, ``math_with_judge`` without its LLM judge). The
``grade_*`` functions are plain functions (no TitanRL imports) so an eval
harness can use the same graders.
"""

from __future__ import annotations

import json
import re
import unicodedata
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

_BOXED = "\\boxed{"
_TEXT_WRAPPER = re.compile(r"^\\(?:text|textbf|mathrm)\{(.*)\}$", re.DOTALL)
_LEADING_LETTER = re.compile(r"^\(?([A-Za-z])\)?(?:$|[\s:.)\-])")


def grade_mcqa(
    response: str, expected_answer: str, output_regex: str | list[str] | None = None
) -> float:
    """1.0 when the response names the expected letter.

    The letter comes from the last match of the row's ``output_regex`` (its
    prompt template's answer format, case-insensitive), else from the last
    ``\\boxed{...}``, as NeMo Gym's mcqa server does.
    """
    letter = None
    patterns = [output_regex] if isinstance(output_regex, str) else output_regex or []
    for pattern in patterns:
        found = re.findall(pattern, response, re.IGNORECASE)
        if found:
            letter = unicodedata.normalize("NFKC", found[-1]).strip().upper()
            break
    if letter is None:
        letter = _boxed_letter(response)
    return float(letter is not None and letter == expected_answer.strip().upper())


def _boxed_letter(response: str) -> str | None:
    inner = _last_boxed_content(response)
    if inner is None:
        return None
    inner = inner.strip()
    if wrapped := _TEXT_WRAPPER.match(inner):
        inner = wrapped.group(1).strip()
    found = _LEADING_LETTER.match(inner)
    return found.group(1).upper() if found else None


def _last_boxed_content(text: str) -> str | None:
    start = text.rfind(_BOXED)
    if start == -1:
        return None
    depth = 1
    for index in range(start + len(_BOXED), len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start + len(_BOXED) : index]
    return None


def grade_structured_output(response: str, schema_str: str) -> float:
    """1.0 when the whole response is JSON valid against the schema made strict
    (every property required, no others), as NeMo Gym's structured_outputs server."""
    schema = json.loads(schema_str)
    _make_strict(schema)
    try:
        jsonschema.validate(json.loads(response.strip()), schema)
    except (json.JSONDecodeError, jsonschema.ValidationError):
        return 0.0
    return 1.0


def _make_strict(schema: object) -> None:
    if isinstance(schema, dict):
        if "properties" in schema:
            schema["required"] = list(schema["properties"])
            schema["additionalProperties"] = False
        for value in schema.values():
            _make_strict(value)
    elif isinstance(schema, list):
        for value in schema:
            _make_strict(value)


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
                response=response,
                ground_truth=str(fields["expected_answer"]),
            )
            reward = result.score
        elif env_input.agent == MCQA:
            template = fields.get("template_metadata") or {}
            reward = grade_mcqa(
                response, str(fields["expected_answer"]), template.get("output_regex")
            )
        elif env_input.agent == STRUCTURED_OUTPUTS:
            reward = grade_structured_output(response, fields["schema_str"])
        else:
            raise ValueError(f"no grader for NeMo Gym environment {env_input.agent!r}")
        rollout.turns[-1].metrics.append(
            m.Metric(f"nemo_gym/{env_input.agent}/reward", m.Mean(reward))
        )
        return reward
