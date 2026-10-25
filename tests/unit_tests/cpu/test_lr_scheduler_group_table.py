# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A param group's own `lr_table`: knots ((step, factor), ...), the group's lr
at the optimizer's step t (1-based) is lr x the piecewise-linear interpolation
at t, the end values outside the knots. It replaces the shared schedule for
that group. A loaded scheduler writes its lrs into the groups."""

import pytest
import torch

from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer

T = 40


def _trainer(group_b=None, *, decay_ratio=0.2, warmup_steps=0):
    a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    groups = [
        {"params": [a], "lr": 0.1},
        {"params": [b], "lr": 0.04, **(group_b or {})},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    opt._opt_called = True
    cfg = LRSchedulersContainer.Config(
        warmup_steps=warmup_steps,
        decay_ratio=decay_ratio,
        decay_type="linear",
        min_lr_factor=0.0,
    )
    return opt, cfg.build(optimizers=[opt], training_steps=T)


def _lrs(opt, sched, steps):
    out = []
    for _ in range(steps):
        out.append([g["lr"] for g in opt.param_groups])
        sched.step()
    return out


def _interp(knots, t):
    """The table at step t, written from its definition."""
    if t <= knots[0][0]:
        return knots[0][1]
    if t >= knots[-1][0]:
        return knots[-1][1]
    for (s0, v0), (s1, v1) in zip(knots, knots[1:]):
        if s0 <= t <= s1:
            return v0 + (v1 - v0) * (t - s0) / (s1 - s0)
    raise AssertionError


def test_a_group_follows_its_table_and_the_other_group_the_trainer():
    knots = [[0, 0.0], [5, 1.0], [20, 0.5], [35, 0.25]]
    opt, sched = _trainer({"lr_table": knots}, warmup_steps=4)
    plain_opt, plain_sched = _trainer(warmup_steps=4)
    got, plain = _lrs(opt, sched, T), _lrs(plain_opt, plain_sched, T)
    for t, ((lr_a, lr_b), (p_a, _)) in enumerate(zip(got, plain), start=1):
        assert lr_a == p_a, t
        assert abs(lr_b - 0.04 * _interp(knots, t)) < 1e-15, t
    assert got[-1][1] == 0.04 * 0.25  # flat after the last knot


def test_a_table_written_as_the_trainers_wsd_is_bitwise_the_trainers():
    """warmup 4, stable, linear to 0 over the last 20% (8 of 40), ending at the
    virtual step T + 1: the same lrs to the bit."""
    knots = [[0, 0.0], [4, 1.0], [T + 1 - 8, 1.0], [T + 1, 0.0]]
    opt, sched = _trainer({"lr_table": knots}, warmup_steps=4)
    plain_opt, plain_sched = _trainer(warmup_steps=4)
    assert _lrs(opt, sched, T) == _lrs(plain_opt, plain_sched, T)


@pytest.mark.parametrize(
    "bad",
    [
        [[0, 1.0]],  # one knot
        [[0, 1.0], [0, 0.5]],  # steps not increasing
        [[0, 1.0], [5, float("nan")]],
        [[0, 1.0], [5, -0.1]],
        [[0, 1.0, 2.0], [5, 1.0]],
    ],
)
def test_a_bad_table_is_refused(bad):
    with pytest.raises(ValueError, match="lr_table"):
        _trainer({"lr_table": bad})


@pytest.mark.parametrize(
    "key", ["warmup_steps", "decay_ratio", "decay_type", "min_lr_factor"]
)
def test_a_table_does_not_mix_with_the_other_group_keys(key):
    value = {
        "warmup_steps": 3,
        "decay_ratio": 0.5,
        "decay_type": "cosine",
        "min_lr_factor": 0.1,
    }[key]
    with pytest.raises(ValueError, match="lr_table"):
        _trainer({"lr_table": [[0, 1.0], [10, 0.0]], key: value})


def test_a_loaded_scheduler_runs_the_new_table_from_the_next_step():
    """A branch: the checkpoint (state and lrs) of a run on table A at step 12,
    loaded under table B. Step 13 already runs B (the group lr is rewritten on
    load), and the other group is unchanged."""
    a_knots = [[0, 1.0], [T, 1.0]]
    b_knots = [[0, 1.0], [12, 1.0], [T, 0.0]]
    opt, sched = _trainer({"lr_table": a_knots})
    _lrs(opt, sched, 12)
    osd, ssd = opt.state_dict(), sched.state_dict()
    opt2, sched2 = _trainer({"lr_table": b_knots})
    opt2.load_state_dict(osd)  # brings table A's lr for step 13
    sched2.load_state_dict(ssd)
    got = _lrs(opt2, sched2, T - 12)
    plain_opt, plain_sched = _trainer()
    plain = _lrs(plain_opt, plain_sched, T)[12:]
    for t, ((lr_a, lr_b), (p_a, _)) in enumerate(zip(got, plain), start=13):
        assert lr_a == p_a, t
        assert abs(lr_b - 0.04 * _interp(b_knots, t)) < 1e-15, t


def test_a_plain_resume_is_bitwise():
    opt, sched = _trainer({"lr_table": [[0, 0.0], [5, 1.0], [T, 0.1]]}, warmup_steps=3)
    full = _lrs(opt, sched, T)
    opt2, sched2 = _trainer(
        {"lr_table": [[0, 0.0], [5, 1.0], [T, 0.1]]}, warmup_steps=3
    )
    _lrs(opt2, sched2, 17)
    osd, ssd = opt2.state_dict(), sched2.state_dict()
    opt3, sched3 = _trainer(
        {"lr_table": [[0, 0.0], [5, 1.0], [T, 0.1]]}, warmup_steps=3
    )
    opt3.load_state_dict(osd)
    sched3.load_state_dict(ssd)
    assert _lrs(opt3, sched3, T - 17) == full[17:]
