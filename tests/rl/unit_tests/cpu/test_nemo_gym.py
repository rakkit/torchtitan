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


def test_mcqa_reads_the_last_answer_line():
    assert grade_mcqa("Reasoning...\nAnswer: F", "F") == 1.0
    assert grade_mcqa("Answer: (b)", "B") == 1.0
    assert grade_mcqa("Answer: A\n...\nAnswer: C", "A") == 0.0  # the last line counts
    assert grade_mcqa("no final line", "A") == 0.0


def test_structured_output_validates_against_the_schema():
    schema = json.dumps(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    )
    assert grade_structured_output('```json\n{"x": 3}\n```', schema) == 1.0
    assert grade_structured_output('{"x": 3}', schema) == 1.0
    assert grade_structured_output('{"x": "three"}', schema) == 0.0
    assert grade_structured_output("not json", schema) == 0.0


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
    assert skipped == {"other_agent": 1, "placeholder": 1}
