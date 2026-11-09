# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-turn environment for NeMo Gym rows: the row's messages in, one answer out."""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.rl.examples.nemo_gym.data import NemoGymSample
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)


class NemoGymEnv(MessageEnv):
    """Opens with the row's messages; the first answer ends the rollout."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: NemoGymSample) -> None:
        self._messages = env_input.messages
        self._tools = env_input.tools

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=list(self._messages), tools=list(self._tools)
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        return MessageEnvStepOutput(done=True)
