# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-teacher on-policy distillation: teacher config rules and routing samples to teachers."""

import asyncio
from types import SimpleNamespace

import pytest

from torchtitan.rl.controller.controller import Controller, TeacherConfig


def _teacher(name, domains=(), *, parallelism="tp1", checkpointer="hf"):
    return TeacherConfig(
        name=name,
        generator=SimpleNamespace(checkpointer=checkpointer, parallelism=parallelism),
        hf_assets_path=f"/assets/{name}",
        domains=list(domains),
    )


def _validate(*teachers):
    Controller.Config._validate_teachers(SimpleNamespace(teachers=list(teachers)))


def test_teacher_rules():
    _validate(_teacher("a", ["math"]), _teacher("b", ["mcqa", "json"]), _teacher("c"))
    _validate(_teacher("only"))
    with pytest.raises(ValueError, match="unique"):
        _validate(_teacher("a", ["math"]), _teacher("a", ["mcqa"]))
    with pytest.raises(ValueError, match="claimed by teachers 'a' and 'b'"):
        _validate(_teacher("a", ["math"]), _teacher("b", ["math"]))
    with pytest.raises(ValueError, match="at most one teacher may have no domains"):
        _validate(_teacher("a"), _teacher("b"))
    with pytest.raises(ValueError, match="same generator parallelism"):
        _validate(_teacher("a", ["math"]), _teacher("b", parallelism="tp2"))
    with pytest.raises(ValueError, match="checkpointer"):
        _validate(_teacher("a", checkpointer=None))


def _score_fn(*teachers):
    async def call_one(token_ids, *, metrics_prefix, **kwargs):
        return SimpleNamespace(prompt_token_logprobs=[metrics_prefix])

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(teachers=list(teachers))
    controller._sampling = SimpleNamespace(stop_token_ids=[])
    controller.teacher_routers = {
        t.name: SimpleNamespace(generate=SimpleNamespace(call_one=call_one))
        for t in teachers
    }
    score = controller._make_score_fn()
    return lambda domain: asyncio.run(
        score([1, 2], request_id="r", group_id=0, domain=domain)
    )[0]


def test_each_sample_goes_to_its_domains_teacher():
    score = _score_fn(_teacher("instruct", ["math"]), _teacher("thinking", ["mcqa"]))
    assert score("math") == "teacher/instruct"
    assert score("mcqa") == "teacher/thinking"
    with pytest.raises(ValueError, match="no teacher scores domain 'json'"):
        score("json")
    with pytest.raises(ValueError, match="no teacher scores domain None"):
        score(None)
    # a teacher without domains takes everything the others do not claim
    score = _score_fn(_teacher("instruct", ["math"]), _teacher("default"))
    assert score("math") == "teacher/instruct"
    assert score("json") == "teacher/default"
    assert score(None) == "teacher/default"
