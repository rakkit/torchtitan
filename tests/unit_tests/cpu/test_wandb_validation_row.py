# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation logs at the step whose training metrics were just logged. With a
real offline W&B run, both must reach that training step (each `log` call is
its own row keyed by `trainer/step`), and the last row must survive `close`."""

import glob
import json

import pytest

from torchtitan.observability import metrics

wandb = pytest.importorskip("wandb")


def _history(log_dir):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    (path,) = glob.glob(f"{log_dir}/wandb/offline-run-*/*.wandb")
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    rows = {}
    while (data := ds.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            row = {
                ("/".join(i.nested_key) or i.key): json.loads(i.value_json)
                for i in rec.history.item
            }
            rows.setdefault(int(row["trainer/step"]), {}).update(row)
    return rows


def test_validation_joins_the_training_row(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    for k in ("WANDB_RUN_NAME", "WANDB_NAME", "WANDB_RUN_ID", "WANDB_RUN_GROUP"):
        monkeypatch.delenv(k, raising=False)
    logger = metrics.WandBLogger(metrics.WandBLogger.Config(), log_dir=str(tmp_path))
    logger.log({"loss": 3.0}, step=100)
    logger.log({"validation_metrics/loss": 3.5}, step=100)
    logger.log({"loss": 2.0}, step=101)
    logger.close()

    rows = _history(tmp_path)
    assert rows[100]["loss"] == 3.0
    assert rows[100]["validation_metrics/loss"] == 3.5
    assert rows[101]["loss"] == 2.0
