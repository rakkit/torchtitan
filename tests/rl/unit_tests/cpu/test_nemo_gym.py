# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NeMo Gym rows: loading, environment filtering, and the sandbox-free graders."""

import json

from torchtitan.rl.examples.nemo_gym import (
    grade_mcqa,
    grade_structured_output,
    read_nemo_gym_rows,
)


def test_mcqa_uses_the_rows_answer_format():
    # rows carry their prompt template's regex (about 30 formats in the Nano blend)
    boxed = r"\\boxed\{\s*([A-Za-z0-9])\s*\}"
    double_brackets = r"\[\[\s*([A-Za-z0-9])\s*\]\]"
    assert grade_mcqa("Reasoning...\nAnswer: \\boxed{D}", "D", boxed) == 1.0
    assert grade_mcqa("The correct answer is: [[c]]", "C", double_brackets) == 1.0
    assert grade_mcqa("[[A]] at first, then [[C]]", "A", double_brackets) == 0.0
    assert grade_mcqa("Answer: C", "C", double_brackets) == 0.0
    # no regex match: the letter of the last \boxed{...}
    assert grade_mcqa("so \\boxed{\\text{E}}", "E", double_brackets) == 1.0
    assert grade_mcqa("no answer", "A") == 0.0


def test_structured_output_is_strict_json():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "string"}},
        }
    )
    assert grade_structured_output('{"x": 3, "y": "a"}', schema) == 1.0
    # every property is required and no others are allowed, the whole answer must be JSON
    assert grade_structured_output('{"x": 3}', schema) == 0.0
    assert grade_structured_output('{"x": 3, "y": "a", "z": 1}', schema) == 0.0
    assert grade_structured_output('```json\n{"x": 3, "y": "a"}\n```', schema) == 0.0
    assert grade_structured_output('{"x": "three", "y": "a"}', schema) == 0.0


def test_rows_are_filtered_by_environment_and_placeholders_skipped(tmp_path):
    def row(agent, **extra):
        return {
            "responses_create_params": {"input": [{"role": "user", "content": "q"}]},
            "agent_ref": {"type": "responses_api_agents", "name": agent},
            **extra,
        }

    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                row("mcqa_simple_agent", expected_answer="B", dataset="stem"),
                row("math_with_judge_simple_agent", _hf_placeholder={"row": 3}),
                row(
                    "math_with_judge_simple_agent", _hf_question_placeholder={"row": 4}
                ),
                row("code_gen_simple_agent"),
            )
        )
    )
    samples, skipped = read_nemo_gym_rows(
        [str(path)], ["mcqa_simple_agent", "math_with_judge_simple_agent"]
    )
    assert [(s.agent, s.source, s.fields["expected_answer"]) for s in samples] == [
        ("mcqa_simple_agent", "stem", "B")
    ]
    assert samples[0].messages == [{"role": "user", "content": "q"}]
    assert skipped == {"other_agent": 1, "placeholder": 2}
