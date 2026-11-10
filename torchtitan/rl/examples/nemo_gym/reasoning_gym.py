# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""reasoning_gym grader: a port of NeMo Gym's resources_servers/reasoning_gym (Apache-2.0).
Each row is one procedurally generated task (``metadata["source_dataset"]``, e.g.
``zebra_puzzles``); the reasoning_gym package's own scorer for that task grades the answer:
the last ``<answer>...</answer>``, else the first ``\\boxed{...}``, else the whole reply.
Reward 1 only when that score is 1, else 0: reasoning_gym's default scorer gives len(gold) /
len(answer) when the gold is a substring of a WRONG answer ("112" for 12, "12, 7"), >= 0.5 on
about half of the rows (review TASK 16); NeMo Gym passes the partial score through."""

from __future__ import annotations

import re

_ANSWER = re.compile(r"<answer>\s?(.*?)\s?</answer>", re.DOTALL)
_BOXED = re.compile(r"\\boxed\{([^}]+)\}")


def extract_reasoning_gym_answer(response: str) -> str:
    """NeMo Gym's _extract_answer_from_response on the reply text."""
    tagged = _ANSWER.findall(response)
    if tagged:
        return tagged[-1].strip()
    boxed = _BOXED.search(response)
    if boxed:
        return boxed.group(1).strip()
    return response.strip()


def grade_reasoning_gym(
    response: str, question: str, answer: str | None, metadata: dict
) -> float:
    """The task's own score (reasoning_gym.get_score_answer_fn) of the extracted answer; a
    scorer error scores 0 (as NeMo Gym). A row without ``source_dataset`` raises (data error)."""
    task = metadata.get("source_dataset")
    if not task:
        raise ValueError(
            f"reasoning_gym row without metadata.source_dataset: {metadata}"
        )
    if not response.strip():
        return 0.0
    import reasoning_gym  # heavy (many task modules): only when a row needs it

    entry = {"question": question, "answer": answer, "metadata": metadata}
    try:
        score_fn = reasoning_gym.get_score_answer_fn(task)
        score = float(
            score_fn(answer=extract_reasoning_gym_answer(response), entry=entry)
        )
    except Exception:  # NeMo Gym: an answer the task's scorer cannot read scores 0
        return 0.0
    return 1.0 if score >= 1.0 else 0.0
