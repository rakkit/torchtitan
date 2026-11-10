# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment for NeMo Gym rows: the row's messages in, one answer out. Rows of
``ns_tools_simple_agent`` (math with a stateful Python tool) run multi-turn instead, each
rollout in its own Apptainer sandbox (titan_eval ``envs/python_tool``; set
``NemoGymEnv.Config.python_tool``)."""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass

from renderers import Message

from torchtitan.rl.examples.nemo_gym.data import NemoGymSample
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

NS_TOOLS = "ns_tools_simple_agent"


@dataclass(kw_only=True, slots=True)
class PythonToolConfig:
    """The sandboxed stateful Python tool of ``ns_tools_simple_agent`` rows."""

    titan_eval_lib: str
    """titan_eval's ``lib`` folder; the framework-free tool is imported from it as
    ``titan_eval_envs.python_tool``."""

    image: str
    """The tool's Apptainer image (python-sci ``image.sif``)."""


class NemoGymEnv(MessageEnv):
    """Opens with the row's messages; the first answer ends the rollout, except for
    ``ns_tools_simple_agent`` rows: their tool calls run in a sandbox until an answer
    without a tool call. The turn limit is `TokenEnv`'s ``max_num_turns`` (a truncated
    rollout scores the truncation reward); the rubric grades the last answer."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        python_tool: PythonToolConfig | None = None
        """The sandboxed Python tool; None: ``ns_tools_simple_agent`` rows raise."""

    def __init__(self, config: Config, *, env_input: NemoGymSample) -> None:
        self._messages = env_input.messages
        self._tools = env_input.tools
        self._tool_env = None
        if env_input.agent == NS_TOOLS:
            if config.python_tool is None:
                raise ValueError(f"{NS_TOOLS} rows need NemoGymEnv.Config.python_tool")
            if config.python_tool.titan_eval_lib not in sys.path:
                sys.path.insert(0, config.python_tool.titan_eval_lib)
            from titan_eval_envs.python_tool.env import PythonToolEnv

            row = {
                "responses_create_params": {
                    "input": list(self._messages),
                    "tools": list(self._tools),
                },
                "expected_answer": env_input.fields["expected_answer"],
            }
            # one sandbox per rollout (siblings share the row); no turn limit of its own
            self._tool_env = PythonToolEnv(
                row,
                config.python_tool.image,
                name=f"pytool-{uuid.uuid4().hex[:16]}",
                max_turns=sys.maxsize,
            )

    async def init(self) -> MessageEnvInitOutput:
        if self._tool_env is not None:
            messages, tools = await self._tool_env.init()
            return MessageEnvInitOutput(init_prompt_messages=messages, tools=tools)
        return MessageEnvInitOutput(
            init_prompt_messages=list(self._messages), tools=list(self._tools)
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        if self._tool_env is None:
            return MessageEnvStepOutput(done=True)
        message = dict(completion_message)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                _openai_tool_call(call, i)
                for i, call in enumerate(message["tool_calls"])
            ]
        env_messages, done = await self._tool_env.step(message)
        # tool counters so far: the "tool/" keys are COUNTERS, never rewards; only the
        # rubric reads them, as nemo_gym/tool/* metrics
        return MessageEnvStepOutput(
            env_messages=env_messages,
            done=done,
            env_rewards=_tool_counts(self._tool_env),
        )

    async def close(self) -> None:
        if self._tool_env is not None:
            await self._tool_env.close()


def _openai_tool_call(call: object, index: int) -> dict:
    """A renderers ParsedToolCall (how `TokenEnv` parses a reply) as the OpenAI-style dict
    the framework-free tool env reads. A call that did not parse keeps its raw text as the
    arguments, so the env answers with a tool error instead of the rollout failing."""
    if isinstance(call, dict):
        return call
    status = getattr(call, "status", None)
    ok = status is None or getattr(status, "name", str(status)) == "OK"
    return {
        "id": getattr(call, "id", None) or f"call_{index}",
        "type": "function",
        "function": {
            "name": getattr(call, "name", None),
            "arguments": getattr(call, "arguments", None) if ok else call.raw,
        },
    }


def _tool_counts(tool_env) -> dict[str, float]:
    """The rollout's tool counters (calls, bad / capped calls, seconds, and the sandbox
    session's timeouts, memory stops, dead sessions, refused forks, cut outputs)."""
    counts = {
        k: v
        for k, v in tool_env.info.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if tool_env.session is not None:
        counts.update({f"session_{k}": v for k, v in tool_env.session.stats.items()})
    return {f"tool/{k}": float(v) for k, v in counts.items()}
