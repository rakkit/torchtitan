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
    Controller.Config._validate_teachers(
        SimpleNamespace(teachers=list(teachers), judge=None)
    )


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

    def run(domain):
        scores = asyncio.run(score([1, 2], request_id="r", group_id=0, domain=domain))
        # the router that answered (its metrics prefix) and the teacher name returned agree
        assert scores.token_logprobs[0] == f"teacher/{scores.teacher}"
        return scores.token_logprobs[0]

    return run


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


def test_judge_gets_token_ids_from_its_own_chat_template(tmp_path):
    # transformers 5 returns a BatchEncoding from apply_chat_template(tokenize=True):
    # list() of it gave the keys ["input_ids", "attention_mask"], not token ids
    import json
    import shutil

    from transformers import AutoTokenizer

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(f"tests/assets/tokenizer/{name}", tmp_path / name)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 7}))
    sent = []

    async def call_one(token_ids, *, sampling_config, **kwargs):
        sent.append((token_ids, sampling_config))
        return SimpleNamespace(
            token_ids=tokenizer.encode("Rating: [[9]]"), finish_reason="stop"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path))
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        judge=SimpleNamespace(
            hf_assets_path=str(tmp_path), temperature=0.0, max_tokens=8
        )
    )
    controller.judge_router = SimpleNamespace(
        generate=SimpleNamespace(call_one=call_one)
    )
    controller._judge_tokenizer = tokenizer
    judge_fn = controller._make_judge_fn()
    reply = asyncio.run(
        judge_fn([{"role": "user", "content": "rate this"}], request_id="j", group_id=0)
    )
    ((token_ids, sampling),) = sent
    assert all(isinstance(t, int) for t in token_ids)
    assert token_ids == tokenizer.apply_chat_template(
        [{"role": "user", "content": "rate this"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    assert sampling.stop_token_ids == [7] and sampling.max_tokens == 8
    # the reply is the judge's tokens decoded (the test tokenizer drops spaces when decoding)
    assert reply == tokenizer.decode(
        tokenizer.encode("Rating: [[9]]"), skip_special_tokens=True
    )
    # with the completion's finish reason and token count (a reply cut at max_tokens shows)
    assert reply.finish_reason == "stop"
    assert reply.num_tokens == len(tokenizer.encode("Rating: [[9]]"))
