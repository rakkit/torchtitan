# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The column gains c of angular_muown col_scale runs, logged with log_gain_stats (ps2n L1).

Per unit (one matrix): col_gain_rms, col_gain_mean_log (mean log|c|), col_gain_q10 / q50 / q90
(of c) and col_gain_rel_update (||c - c_before|| / ||c_before||, c_before = 1 on the first step),
read from the optimizer state after the step. The references are recomputed with numpy from the
CHECKPOINT of the same step (state_dict), not from the code under test. Reading must not change
the run: the weights and the state after N steps are bitwise equal with and without the reads."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.optimizers.disco.optimizer import _key

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_angular_muown_col_scale as base  # noqa: E402
from test_angular_muown_col_scale import one_rank  # noqa: E402,F401

STATS = ("rms", "mean_log", "q10", "q50", "q90", "rel_update")
M, N = 12, 8


def _ckpt_c(opt):
    rs = opt.state_dict()["state"][0]["rule_state"]
    return rs[0, M, :N].double().numpy()


def _expected(c, c0):
    return {
        "rms": float(np.sqrt(np.mean(c**2))),
        "mean_log": float(np.mean(np.log(np.abs(c)))),
        "q10": float(np.quantile(c, 0.1)),
        "q50": float(np.quantile(c, 0.5)),
        "q90": float(np.quantile(c, 0.9)),
        "rel_update": float(np.linalg.norm(c - c0) / np.linalg.norm(c0)),
    }


def _steps(n_steps, read, w0, grads, opt_factory=base._disco):
    w = nn.Parameter(w0.float().clone())
    opt = opt_factory(w)
    logged = []
    for G in grads[:n_steps]:
        before = opt.col_gain_snapshot() if read else None
        w.grad = G.float().clone()
        opt.step()
        logged.append(opt.col_gain_metrics(before) if read else None)
    return w.detach(), opt, logged


def test_logged_equals_the_checkpoint_c(one_rank):  # noqa: F811
    W, grads = base._data(M, N, seed=81, steps=5)
    w = nn.Parameter(W.float().clone())
    opt = base._disco(w)
    e = opt.entries[0]
    key = _key(e, e.units[0])
    c0 = np.ones(N)
    for step, G in enumerate(grads, start=1):
        before = opt.col_gain_snapshot()
        w.grad = G.float().clone()
        opt.step()
        got = opt.col_gain_metrics(before)
        c = _ckpt_c(opt)
        want = _expected(c, c0)
        assert sorted(got) == sorted(f"col_gain_{s}/{key}" for s in STATS), sorted(got)
        for s in STATS:
            assert got[f"col_gain_{s}/{key}"] == pytest.approx(
                want[s], rel=1e-5, abs=1e-6
            ), (step, s)
        assert float(np.abs(c - 1).max()) > 0  # a learned c, not the start
        c0 = c


def test_reading_does_not_change_the_run_bitwise(one_rank):  # noqa: F811
    W, grads = base._data(M, N, seed=83, steps=6)
    w_off, opt_off, _ = _steps(6, False, W, grads)
    w_on, opt_on, logged = _steps(6, True, W, grads)
    assert all(logged) and torch.equal(w_off, w_on)
    a, b = (
        opt_off.state_dict()["state"][0]["rule_state"],
        opt_on.state_dict()["state"][0]["rule_state"],
    )
    assert torch.equal(a, b)


def test_a_run_without_col_scale_logs_nothing(one_rank):  # noqa: F811
    from torchtitan.models.common.param_layout import ParamLayout
    from torchtitan.optimizers.disco.optimizer import DiSCO

    W, grads = base._data(M, N, seed=85, steps=2)
    w = nn.Parameter(W.float().clone())
    args = {**base._ARGS, "col_scale": False}
    opt = DiSCO(
        [
            {
                "params": [w],
                "param_names": ["w"],
                "momentum": 1.0,
                "nesterov": False,
                "weight_decay": 0.0,
            }
        ],
        parallel_dims=base._NoParallel(),
        lr=base.LR,
        rule="angular_muown",
        args=args,
        param_layouts={"w": ParamLayout("matrix")},
    )
    assert opt.col_gain_snapshot() is None
    w.grad = grads[0].float().clone()
    opt.step()
    assert opt.col_gain_metrics(None) == {}


def _container(opts, *, log_gain_stats=True):
    c = MagicMock(spec=DiSCOOptimizersContainer)
    c.model_parts = [nn.Linear(2, 2)]  # no RMSNorm: the model gains are an empty family
    c.parallel_dims = None
    c.log_residual_stats, c.log_gain_stats = False, log_gain_stats
    c.log_sink_stats = False
    c.log_norm_freq, c.dense_norm_steps = 10, 500
    c.optimizers = opts
    c._norms_armed = c._model_stats_armed = False
    c._gains_before = c._col_before = None
    for name in ("wants_norms_at", "_arm_model_stats", "_collect_model_stats"):
        setattr(c, name, getattr(DiSCOOptimizersContainer, name).__get__(c))
    return c


def test_the_container_reads_c_on_the_gain_cadence_once(one_rank):  # noqa: F811
    W, grads = base._data(M, N, seed=87, steps=4)
    w = nn.Parameter(W.float().clone())
    opt = base._disco(w)
    c = _container([opt])
    key = _key(opt.entries[0], opt.entries[0].units[0])
    for step, G in enumerate(grads[:2], start=1):
        c._arm_model_stats(
            step, should_log=True, is_final_step=False
        )  # steps 1..10 are dense: armed
        assert c._model_stats_armed
        w.grad = G.float().clone()
        opt.step()
        got = c._collect_model_stats()
        assert f"col_gain_rms/{key}" in got and f"col_gain_rel_update/{key}" in got
        want = _expected(_ckpt_c(opt), np.ones(N) if step == 1 else prev)
        assert got[f"col_gain_rel_update/{key}"] == pytest.approx(
            want["rel_update"], rel=1e-5
        )
        prev = _ckpt_c(opt)
        assert c._collect_model_stats() == {}  # one read per armed step
    off = _container([opt], log_gain_stats=False)
    off._arm_model_stats(1, should_log=True, is_final_step=False)
    assert off._col_before is None and off._collect_model_stats() == {}


def test_every_unit_of_fused_parameters_is_logged_under_its_own_name(
    one_rank,  # noqa: F811
):
    """q / k / v of one wqkv and w1 / w3 of one w13 are separate units with their own c."""
    import test_angular_muown_freeze_units as fz
    from torchtitan.optimizers.disco.optimizer import DiSCO

    params = fz._params(seed=3)
    args = {
        "compute": "svd",
        **fz._F,
        "sphere": "row",
        "tangent": False,
        "magnitude_lr": fz.MAG_LR,
        "col_scale": True,
    }
    opt = DiSCO(
        [
            {
                "params": params,
                "param_names": fz.NAMES,
                "momentum": 1.0,
                "nesterov": False,
                "weight_decay": 0.0,
            }
        ],
        parallel_dims=fz._NoParallel(),
        lr=fz.LR,
        rule="angular_muown",
        args=args,
        param_layouts=fz.LAYOUTS,
    )
    before = opt.col_gain_snapshot()
    assert len(before) == 4 and [b.shape[0] for b in before] == [3, 1, 2, 1]
    fz._steps(opt, params, range(3))
    got = opt.col_gain_metrics(before)
    names = sorted(
        {k.split("/", 1)[1] for k in got},
    )
    assert names == sorted(
        [
            "layers.0.attention.qkv_linear.wq",
            "layers.0.attention.qkv_linear.wk",
            "layers.0.attention.qkv_linear.wv",
            "layers.0.attention.wo",
            "layers.0.feed_forward.w1",
            "layers.0.feed_forward.w3",
            "layers.0.feed_forward.w2",
        ]
    )
    for e in opt.entries:
        rs = opt.state[e.param]["rule_state"]
        for un in e.units:
            c = rs[un.pos, -3, :-3].double().numpy()
            assert abs(float(np.abs(c - 1).max())) > 0
            k = _key(e, un)
            assert got[f"col_gain_rms/{k}"] == pytest.approx(
                float(np.sqrt(np.mean(c**2))), rel=1e-5
            )
            assert got[f"col_gain_q50/{k}"] == pytest.approx(
                float(np.quantile(c, 0.5)), rel=1e-5
            )
            assert got[f"col_gain_mean_log/{k}"] == pytest.approx(
                float(np.mean(np.log(np.abs(c)))), rel=1e-4, abs=1e-6
            )
