# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for torchtitan/rl/examples/terminal_bench/env.py task rows: a TMax row
(``"layout": "tmax"``, titan_eval tasksets/tmax/rows.py) builds its episode with
``TerminusEnv.for_tmax`` (stack image + template), a Terminal-Bench row with
``TerminusEnv.for_task``. titan_eval's env is replaced by a recorder (no sandbox)."""

import json
import sys
import types

import pytest

pytest.importorskip("renderers")

from torchtitan.rl.examples.terminal_bench.env import (
    TerminalTaskDataset,
    TerminalTaskEnv,
    TerminalTaskSample,
)


class _RecordingTerminusEnv:
    calls: list = []

    @classmethod
    def for_task(cls, task_dir, image, prep_dir, log_dir, *, name):
        cls.calls.append(("for_task", task_dir, image, prep_dir, log_dir, name))
        return cls()

    @classmethod
    def for_tmax(cls, task_dir, template, log_dir, *, name):
        cls.calls.append(("for_tmax", task_dir, template, log_dir, name))
        return cls()


@pytest.fixture
def recorder(monkeypatch):
    _RecordingTerminusEnv.calls = []
    pkg = types.ModuleType("titan_eval_envs")
    term = types.ModuleType("titan_eval_envs.terminal")
    env = types.ModuleType("titan_eval_envs.terminal.env")
    env.TerminusEnv = _RecordingTerminusEnv
    monkeypatch.setitem(sys.modules, "titan_eval_envs", pkg)
    monkeypatch.setitem(sys.modules, "titan_eval_envs.terminal", term)
    monkeypatch.setitem(sys.modules, "titan_eval_envs.terminal.env", env)
    return _RecordingTerminusEnv


def _rows(tmp_path):
    rows = [
        # the shape titan_eval tasksets/tmax/rows.py writes
        {
            "task": "task_000001_abc",
            "task_dir": "/src/task_000001_abc",
            "image": "/stack/0c4e/image.sif",
            "prep_dir": "/prepared/aarch64/tmax/task_000001_abc",
            "layout": "tmax",
        },
        # a Terminal-Bench row: no layout key (default tb2)
        {
            "task": "chess-best-move",
            "task_dir": "/tb2/chess-best-move",
            "image": "/tb2/chess-best-move.sif",
            "prep_dir": "/prepared/aarch64/tb2/chess-best-move",
        },
    ]
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_rows_keep_their_layout(tmp_path) -> None:
    ds = TerminalTaskDataset(
        TerminalTaskDataset.Config(
            paths=[str(_rows(tmp_path))], shuffle=False, task_set="tmax"
        )
    )
    a, b = next(ds), next(ds)
    assert (a.task, a.layout, a.task_set) == ("task_000001_abc", "tmax", "tmax")
    assert (b.task, b.layout) == ("chess-best-move", "tb2")


def test_tmax_row_uses_the_template_and_tb2_row_the_task_image(
    tmp_path, recorder
) -> None:
    cfg = TerminalTaskEnv.Config(
        titan_eval_lib=str(tmp_path / "lib"), log_dir=str(tmp_path / "logs")
    )
    ds = TerminalTaskDataset(
        TerminalTaskDataset.Config(paths=[str(_rows(tmp_path))], shuffle=False)
    )
    TerminalTaskEnv(cfg, env_input=next(ds))
    TerminalTaskEnv(cfg, env_input=next(ds))
    (k1, *tmax), (k2, *tb2) = recorder.calls
    assert k1 == "for_tmax"
    assert tmax[:2] == [
        "/src/task_000001_abc",
        "/prepared/aarch64/tmax/task_000001_abc",
    ]
    assert tmax[2].startswith(str(tmp_path / "logs" / "task_000001_abc")) and tmax[
        3
    ].startswith("term-")
    assert k2 == "for_task"
    assert tb2[:3] == [
        "/tb2/chess-best-move",
        "/tb2/chess-best-move.sif",
        "/prepared/aarch64/tb2/chess-best-move",
    ]


def test_unknown_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="neither 'tb2' nor 'tmax'"):
        TerminalTaskSample(
            task="t", task_dir="/d", image="/i.sif", prep_dir="/p", layout="other"
        )
