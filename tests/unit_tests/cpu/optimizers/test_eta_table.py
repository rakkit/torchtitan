# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EtaSchedule shape `table`: piecewise-linear (step, value) points given by the cell. Expected values are written
from the definition (linear interpolation, the first / last value held outside the points), and from sqrt_decay's
own formula 1 - sqrt(x / (total - start))."""

import math

import pytest

from torchtitan.optimizers.disco.presets import eta_at, eta_of, EtaSchedule


def _sqrt_decay_points(start, total, upto):
    """The sqrt_decay value at every integer step 0..upto, from its definition."""
    return [
        [
            t,
            1.0
            if t - start <= 0
            else 1.0 - math.sqrt(min(1.0, (t - start) / (total - start))),
        ]
        for t in range(upto + 1)
    ]


@pytest.mark.parametrize("start", [0, 100])
def test_a_table_of_sqrt_decay_values_is_sqrt_decay_bitwise(start):
    total = 25600
    ref = eta_of("hyperball", {"shape": "sqrt_decay", "start": start, "total": total})
    tab = eta_of(
        "hyperball", {"shape": "table", "points": _sqrt_decay_points(start, total, 600)}
    )
    assert all(tab(t) == ref(t) for t in range(601))  # bitwise at every knot
    assert tab(5000) == tab(600)  # held past the last point


def test_linear_between_points_and_held_outside():
    e = eta_of(
        "hyperball", {"shape": "table", "points": [[100, 1.0], [300, 0.5], [500, 0.0]]}
    )
    assert (
        e(0) == 1.0
        and e(100) == 1.0
        and e(300) == 0.5
        and e(500) == 0.0
        and e(9999) == 0.0
    )
    assert e(200) == 0.75 and e(400) == 0.25
    assert abs(e(101) - (1.0 - 0.5 * 1 / 200)) < 1e-15
    assert abs(e(499) - 0.5 * 1 / 200) < 1e-15


def test_json_lists_become_the_same_schedule_as_tuples():
    a = eta_of("hyperball", {"shape": "table", "points": [[0, 1.0], [10, 0.5]]})
    b = eta_of("hyperball", {"shape": "table", "points": ((0, 1.0), (10, 0.5))})
    assert a == b and hash(a) == hash(b) and isinstance(a, EtaSchedule)


@pytest.mark.parametrize(
    "points",
    [
        [[0, 1.0]],
        [[0, 1.0], [0, 0.5]],
        [[5, 1.0], [3, 0.5]],
        [[0, 1.0], [1, float("nan")]],
        [[0, 1.0], [1, 2.0, 3.0]],
    ],
)
def test_a_bad_table_is_refused(points):
    with pytest.raises(ValueError):
        eta_of("hyperball", {"shape": "table", "points": points})


def test_a_table_takes_no_other_shape_keys():
    with pytest.raises(ValueError, match="does not use"):
        eta_of(
            "hyperball", {"shape": "table", "points": [[0, 1.0], [1, 0.5]], "hold": 5}
        )
    with pytest.raises(ValueError, match="does not use"):
        eta_of(
            "hyperball",
            {
                "shape": "sqrt_decay",
                "start": 0,
                "total": 10,
                "points": [[0, 1.0], [1, 0.5]],
            },
        )


def test_the_step_size_is_lr_times_the_table():
    class _Ctx:
        lr, t = 0.02, 150

    e = eta_of("hyperball", {"shape": "table", "points": [[100, 1.0], [200, 0.0]]})
    assert eta_at(e, _Ctx) == 0.02 * 0.5


def test_a_table_eta_sets_the_exact_angle_of_hyperball():
    """hyperball with exact_angle turns by lr eta(t): with a table, by lr x the interpolated value."""
    import torch

    from torchtitan.optimizers.disco.rules import rule_of, StepCtx

    g = torch.Generator().manual_seed(3)
    W = torch.randn(3, 24, 16, generator=g, dtype=torch.float64)
    M = torch.randn(3, 24, 16, generator=g, dtype=torch.float64)
    rule = rule_of(
        {
            "rule": "hyperball",
            "args": {
                "compute": "svd",
                "exact_angle": True,
                "eta": {"shape": "table", "points": [[10, 1.0], [30, 0.25], [50, 0.0]]},
            },
        }
    )
    lr = 0.03
    for t, eta in (
        (5, 1.0),
        (10, 1.0),
        (20, 0.625),
        (30, 0.25),
        (40, 0.125),
        (60, 0.0),
    ):
        res = rule.run_whole(M, W, StepCtx(lr, 0.0, t))
        new = res.alpha[:, None, None] * W - res.update
        c = (W * new).sum(dim=(-2, -1)) / (
            W.norm(dim=(-2, -1)) * new.norm(dim=(-2, -1))
        )
        got = torch.arccos(c.clamp(-1, 1))
        torch.testing.assert_close(
            got, torch.full_like(got, lr * eta), rtol=0, atol=1e-6
        )
