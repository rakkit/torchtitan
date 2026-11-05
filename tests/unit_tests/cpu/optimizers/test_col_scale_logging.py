# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The column scales c of amphione `column_scales` runs, logged with log_gain_stats.

Per unit (one matrix): col_scale_rms, col_scale_mean_log (mean log|c|), col_scale_q10 / q50 / q90
(of c) and col_scale_rel_update (||c - c_before|| / ||c_before||, c_before = 1 on the first step),
read from the optimizer state after the step. The references are recomputed with numpy from the
CHECKPOINT of the same step (state_dict), not from the code under test. Reading must not change
the run: the weights and the state after N steps are bitwise equal with and without the reads.
Units outside `column_scale_units` keep c = 1 and are logged as such (what to log: wave-2 diagnostics)."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout, qkv_units, stacked_units
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.optimizers.disco.optimizer import _key, DiSCO

STATS = ("rms", "mean_log", "q10", "q50", "q90", "rel_update")
M, N, LR = 12, 8, 0.05
NAME = (
    "layers.0.feed_forward.w1.weight"  # a type in amphione's default column_scale_units
)


@pytest.fixture(scope="module")
def one_rank():
    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


def _data(m, n, seed, steps):
    gen = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(
        torch.randn(max(m, n), max(m, n), generator=gen, dtype=torch.float64)
    )[0][:m, :n]
    W = W * torch.logspace(-0.3, 0.3, m, dtype=torch.float64)[:, None]
    W = W * torch.logspace(0.2, -0.2, n, dtype=torch.float64)[None, :]
    return W, [
        0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]


def _amphione(params, names, layouts, *, columns=True, column_scale_units=None):
    args = {"compute": "svd", "column_scales": columns}
    if column_scale_units is not None:
        args["column_scale_units"] = column_scale_units
    return DiSCO(
        [
            {
                "params": params,
                "param_names": names,
                "momentum": 0.0,
                "nesterov": False,
                "weight_decay": 0.0,
            }
        ],
        parallelism_context=_NoParallel(),
        lr=LR,
        rule="amphione",
        args=args,
        param_layouts=layouts,
    )


def _disco(w, columns=True):
    """One matrix: its column_scale_units name exactly its own type (every type must name a unit)."""
    return _amphione(
        [w],
        [NAME],
        {NAME: ParamLayout("matrix")},
        columns=columns,
        column_scale_units=["feed_forward.w1"],
    )


def _ckpt_c(opt):
    rs = opt.state_dict()["state"][0][
        "rule_state"
    ]  # per unit [m + 2, n + 2]: c is row m
    return rs["0"][M, :N].double().numpy()


def _expected(c, c0):
    return {
        "rms": float(np.sqrt(np.mean(c**2))),
        "mean_log": float(np.mean(np.log(np.abs(c)))),
        "q10": float(np.quantile(c, 0.1)),
        "q50": float(np.quantile(c, 0.5)),
        "q90": float(np.quantile(c, 0.9)),
        "rel_update": float(np.linalg.norm(c - c0) / np.linalg.norm(c0)),
    }


def _steps(n_steps, read, w0, grads):
    w = nn.Parameter(w0.float().clone())
    opt = _disco(w)
    logged = []
    for G in grads[:n_steps]:
        before = opt.col_scale_snapshot() if read else None
        w.grad = G.float().clone()
        opt.step()
        logged.append(opt.col_scale_metrics(before) if read else None)
    return w.detach(), opt, logged


def test_logged_equals_the_checkpoint_c(one_rank):  # noqa: F811
    W, grads = _data(M, N, seed=81, steps=5)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w)
    e = opt.entries[0]
    key = _key(e, e.units[0])
    c0 = np.ones(N)
    for step, G in enumerate(grads, start=1):
        before = opt.col_scale_snapshot()
        w.grad = G.float().clone()
        opt.step()
        got = opt.col_scale_metrics(before)
        c = _ckpt_c(opt)
        want = _expected(c, c0)
        assert sorted(got) == sorted(f"col_scale_{s}/{key}" for s in STATS), sorted(got)
        for s in STATS:
            assert got[f"col_scale_{s}/{key}"] == pytest.approx(
                want[s], rel=1e-5, abs=1e-6
            ), (step, s)
        assert float(np.abs(c - 1).max()) > 0  # a learned c, not the start
        c0 = c


def test_reading_does_not_change_the_run_bitwise(one_rank):  # noqa: F811
    W, grads = _data(M, N, seed=83, steps=6)
    w_off, opt_off, _ = _steps(6, False, W, grads)
    w_on, opt_on, logged = _steps(6, True, W, grads)
    assert all(logged) and torch.equal(w_off, w_on)
    a, b = (
        opt_off.state_dict()["state"][0]["rule_state"],
        opt_on.state_dict()["state"][0]["rule_state"],
    )
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_a_run_without_columns_logs_nothing(one_rank):  # noqa: F811
    W, grads = _data(M, N, seed=85, steps=2)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w, columns=False)
    assert opt.col_scale_snapshot() is None
    w.grad = grads[0].float().clone()
    opt.step()
    assert opt.col_scale_metrics(None) == {}


def _container(opts, *, log_gain_stats=True):
    c = MagicMock(spec=DiSCOOptimizersContainer)
    c.model_parts = [nn.Linear(2, 2)]  # no RMSNorm: the model gains are an empty family
    c.parallelism_context = None
    c.log_residual_stats, c.log_gain_stats = False, log_gain_stats
    c.log_sink_stats = False
    c.log_norm_freq = 1  # every step logs (the container's cadence)
    c.optimizers = opts
    c._norms_armed = c._model_stats_armed = False
    c._gains_before = c._col_before = None
    for name in ("wants_norms_at", "_arm_model_stats", "_collect_model_stats"):
        setattr(c, name, getattr(DiSCOOptimizersContainer, name).__get__(c))
    return c


def test_the_container_reads_c_on_the_gain_cadence_once(one_rank):  # noqa: F811
    W, grads = _data(M, N, seed=87, steps=4)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w)
    c = _container([opt])
    key = _key(opt.entries[0], opt.entries[0].units[0])
    prev = None
    for step, G in enumerate(grads[:2], start=1):
        c._arm_model_stats(
            step, should_log=True, is_final_step=False
        )  # every step is a norm-logging step here: armed
        assert c._model_stats_armed
        w.grad = G.float().clone()
        opt.step()
        got = c._collect_model_stats()
        assert f"col_scale_rms/{key}" in got and f"col_scale_rel_update/{key}" in got
        want = _expected(_ckpt_c(opt), np.ones(N) if step == 1 else prev)
        assert got[f"col_scale_rel_update/{key}"] == pytest.approx(
            want["rel_update"], rel=1e-5
        )
        prev = _ckpt_c(opt)
        assert c._collect_model_stats() == {}  # one read per armed step
    off = _container([opt], log_gain_stats=False)
    off._arm_model_stats(1, should_log=True, is_final_step=False)
    assert off._col_before is None and off._collect_model_stats() == {}


D, H, KV, HD, FF = 8, 2, 2, 4, 16  # MHA: one state shape per wqkv
NAMES = [
    "layers.0.attention.qkv_linear.wqkv.weight",
    "layers.0.attention.wo.weight",
    "layers.0.feed_forward.w13.weight",
    "layers.0.feed_forward.w2.weight",
]
LAYOUTS = {
    NAMES[0]: ParamLayout("matrix", units=qkv_units(H, KV, HD, D)),
    NAMES[1]: ParamLayout("matrix"),
    NAMES[2]: ParamLayout("matrix", units=stacked_units(("w1", "w3"), FF, D)),
    NAMES[3]: ParamLayout("matrix"),
}
SHAPES = [((H + 2 * KV) * HD, D), (D, H * HD), (2, FF, D), (D, FF)]
LEARNED = {"wq", "wk", "wv", "w1", "w3"}  # amphione's default column_scale_units


def test_every_unit_of_fused_parameters_is_logged_under_its_own_name(
    one_rank,  # noqa: F811
):
    """q / k / v of one wqkv and w1 / w3 of one w13 are separate units with their
    own c; wo and w2 are outside column_scale_units, so their c stays exactly 1."""
    g = torch.Generator().manual_seed(3)
    params = [nn.Parameter(torch.randn(*s, generator=g) * 0.3) for s in SHAPES]
    opt = _amphione(params, NAMES, LAYOUTS)
    before = opt.col_scale_snapshot()
    assert len(before) == 4 and [b.shape[0] for b in before] == [3, 1, 2, 1]
    for s in range(3):
        g = torch.Generator().manual_seed(100 + s)
        for p, shape in zip(params, SHAPES):
            p.grad = 0.1 * torch.randn(*shape, generator=g)
        opt.step()
    got = opt.col_scale_metrics(before)
    names = sorted({k.split("/", 1)[1] for k in got})
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
            c = rs[str(un.pos)][-2, :-2].double().numpy()
            k = _key(e, un)
            if k.rsplit(".", 1)[-1] in LEARNED:
                assert float(np.abs(c - 1).max()) > 0, k
            else:
                assert np.array_equal(c, np.ones_like(c)), k
            assert got[f"col_scale_rms/{k}"] == pytest.approx(
                float(np.sqrt(np.mean(c**2))), rel=1e-5
            )
            assert got[f"col_scale_q50/{k}"] == pytest.approx(
                float(np.quantile(c, 0.5)), rel=1e-5
            )
            assert got[f"col_scale_mean_log/{k}"] == pytest.approx(
                float(np.mean(np.log(np.abs(c)))), rel=1e-4, abs=1e-6
            )
