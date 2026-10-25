# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule knots for lr_table / eta tables, checked against their definitions."""

import pytest
import torch

from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.schedule_tables import (
    cooldown_knots,
    power_base_knots,
    table_at,
    wsd_knots,
)


@pytest.mark.parametrize(
    "warmup,total", [(100, 25600), (0, 25600), (13, 1600), (200, 12800)]
)
def test_wsd_knots_are_the_trainers_wsd_to_the_bit(warmup, total):
    a, b = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW(
        [
            {"params": [a], "lr": 0.1},
            {"params": [b], "lr": 0.1, "lr_table": wsd_knots(warmup, total)},
        ]
    )
    opt._opt_called = True
    cfg = LRSchedulersContainer.Config(
        warmup_steps=warmup, decay_ratio=0.2, decay_type="linear", min_lr_factor=0.0
    )
    sched = cfg.build(optimizers=[opt], training_steps=total)
    for _ in range(total):
        assert opt.param_groups[0]["lr"] == opt.param_groups[1]["lr"]
        sched.step()


def test_power_base_follows_its_definition_at_the_knots():
    k = power_base_knots(
        warmup=100, tokens_per_step=128 * 4096, d1_tokens=52e6, q=0.15, until=12800
    )
    assert k[:2] == [[0, 0.0], [100, 1.0]]
    for t, v in k[2:]:
        d = (t - 100) * 128 * 4096
        assert v == pytest.approx((1 + d / 52e6) ** -0.15, rel=1e-12)
    assert k[-1][0] == 12800 and all(b[0] > a[0] for a, b in zip(k, k[1:]))
    flat = power_base_knots(
        warmup=0, tokens_per_step=1, d1_tokens=1.0, q=0.0, until=1000
    )
    assert all(v == 1.0 for _, v in flat)
    # between knots the error of the linear reading is small (every 100 steps at B 128)
    t = 5050
    exact = (1 + (t - 100) * 128 * 4096 / 52e6) ** -0.15
    assert abs(table_at(k, t) - exact) < 1e-5


def test_cooldown_is_the_base_times_a_linear_decay():
    base = power_base_knots(
        warmup=100, tokens_per_step=128 * 4096, d1_tokens=52e6, q=0.15, until=12800
    )
    c = cooldown_knots(base, start=10240, end=12800, every=10)
    for t in (50, 100, 5000, 10240):
        assert table_at(c, t) == pytest.approx(table_at(base, t), rel=1e-12)
    for t in range(10250, 12801, 10):
        assert table_at(c, t) == pytest.approx(
            table_at(base, t) * (12800 - t) / 2560, rel=1e-12, abs=1e-15
        )
    assert c[-1] == [12800, 0.0]
    assert all(b[0] > a[0] for a, b in zip(c, c[1:]))


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        power_base_knots(warmup=10, tokens_per_step=1, d1_tokens=1.0, q=-1.0, until=100)
    with pytest.raises(ValueError):
        cooldown_knots([[0, 1.0], [10, 1.0]], start=10, end=5)


def test_a_cooldown_of_an_exact_eta_schedule():
    from torchtitan.optimizers.disco.presets import EtaSchedule

    eta = EtaSchedule("inverse_poly", start=100, c=128 * 4096 / 52e6, p=0.15)
    c = cooldown_knots(eta, start=10240, end=12800, every=10)
    for t in range(10240, 12801, 10):
        assert table_at(c, t) == pytest.approx(
            eta(t) * (12800 - t) / 2560, rel=1e-12, abs=1e-15
        )
    table = EtaSchedule("table", points=tuple((int(a), float(b)) for a, b in c))
    for t in range(10240, 12801, 10):
        assert table(t) == pytest.approx(table_at(c, t), rel=1e-12, abs=1e-15)
