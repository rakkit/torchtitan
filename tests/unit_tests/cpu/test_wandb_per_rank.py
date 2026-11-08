# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""When several ranks write W&B runs, each needs its own id -- the offline runs
of one id overwrite each other on `wandb sync` -- and all of them one group,
which the W&B UI folds into one row. One logger keeps the launcher's run."""

import sys
import types
from types import SimpleNamespace

import pytest

from torchtitan.observability import metrics


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.kwargs = None

    def init(self, **kwargs):
        self.kwargs = kwargs

    def define_metric(self, *args, **kwargs):
        pass


@pytest.fixture
def wandb(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    for k in ("WANDB_RUN_NAME", "WANDB_NAME", "WANDB_RUN_ID", "WANDB_RUN_GROUP"):
        monkeypatch.delenv(k, raising=False)
    return fake


def _run(wandb):
    kw = wandb.kwargs
    return kw["id"], kw["name"], kw["group"]


def test_one_logger_keeps_the_launchers_run(wandb, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_RUN_ID", "run-7")
    metrics.WandBLogger(metrics.WandBLogger.Config(), log_dir=str(tmp_path))
    assert _run(wandb) == ("run-7", None, None)


def test_each_rank_gets_its_own_run_in_one_group(wandb, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_NAME", "run-7")
    monkeypatch.setenv("WANDB_RUN_ID", "run-7")
    metrics.WandBLogger(
        metrics.WandBLogger.Config(),
        log_dir=str(tmp_path),
        rank=3,
        default_group="dump",
    )
    assert _run(wandb) == ("run-7-rank3", "run-7-rank3", "run-7")


def test_the_launchers_group_wins(wandb, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_NAME", "run-7")
    monkeypatch.setenv("WANDB_RUN_GROUP", "run")
    metrics.WandBLogger(
        metrics.WandBLogger.Config(),
        log_dir=str(tmp_path),
        rank=0,
        default_group="dump",
    )
    assert _run(wandb) == (None, "run-7-rank0", "run")


def test_without_launcher_names_the_dump_folder_groups(wandb, tmp_path):
    metrics.WandBLogger(
        metrics.WandBLogger.Config(),
        log_dir=str(tmp_path),
        rank=1,
        default_group="dump",
    )
    assert _run(wandb) == (None, "dump-rank1", "dump")


@pytest.mark.parametrize(
    "for_all, shards, first_dp_tp, per_rank",
    [
        (False, False, False, False),  # one global rank logs
        (True, True, False, True),  # every shard rank
        (True, True, True, True),  # shard ranks win over first-dp-and-tp
        (True, False, True, False),  # one rank of the loss mesh
        (True, False, False, True),  # every rank
    ],
)
def test_which_configs_write_a_run_per_rank(for_all, shards, first_dp_tp, per_rank):
    cfg = SimpleNamespace(
        save_for_all_ranks=for_all,
        save_all_shard_ranks=shards,
        save_first_dp_and_tp=first_dp_tp,
    )
    assert metrics.wandb_run_per_rank(cfg, pp_enabled=False) is per_rank


def test_pp_stages_each_log_under_first_dp_and_tp():
    """One logger per PP stage: they must not share the launcher's run id."""
    cfg = SimpleNamespace(
        save_for_all_ranks=True, save_all_shard_ranks=False, save_first_dp_and_tp=True
    )
    assert metrics.wandb_run_per_rank(cfg, pp_enabled=True) is True
