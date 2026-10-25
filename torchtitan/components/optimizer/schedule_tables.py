# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Knots ((step, factor), ...) on the optimizer's step t (1-based) for a param
group's `lr_table` and a rule's eta `table`: the trainer's WSD, the
horizon-free base (1 + D / D1)^-q in tokens D, and a cooldown branch of any
table. Linear between knots, as both consumers read them."""

import bisect
import math

__all__ = ["wsd_knots", "power_base_knots", "cooldown_knots", "table_at"]


def table_at(knots, t: float) -> float:
    knots = [tuple(k) for k in knots]
    i = bisect.bisect_right(knots, (t, math.inf))
    if i == 0:
        return knots[0][1]
    if i == len(knots):
        return knots[-1][1]
    (s0, v0), (s1, v1) = knots[i - 1], knots[i]
    return v0 if s0 == t else v0 + (v1 - v0) * (t - s0) / (s1 - s0)


def wsd_knots(warmup: int, total: int, decay_ratio: float = 0.2) -> list[list[float]]:
    """The trainer's linear warmup, stable, linear decay over the last
    `decay_ratio` of `total` steps to 0 at the virtual step total + 1: equal
    to its lr to the bit (tests/unit_tests/cpu/test_lr_scheduler_group_table.py)."""
    decay = min(round(total * decay_ratio), total - warmup)
    head = [[0, 0.0], [warmup, 1.0]] if warmup else [[0, 1.0]]
    return head + [[total + 1 - decay, 1.0], [total + 1, 0.0]]


def power_base_knots(
    *,
    warmup: int,
    tokens_per_step: int,
    d1_tokens: float,
    q: float,
    until: int,
    every: int = 100
) -> list[list[float]]:
    """Linear warmup to 1 over `warmup` steps, then (1 + D / D1)^-q with D the
    tokens since the warmup's end, knots every `every` steps to `until`
    (q = 0: flat)."""
    if (
        warmup < 0
        or tokens_per_step <= 0
        or d1_tokens <= 0
        or q < 0
        or until <= warmup
        or every < 1
    ):
        raise ValueError(
            "power_base_knots: warmup >= 0, tokens_per_step, d1 > 0, q >= 0, until > warmup, every >= 1"
        )
    knots = [[0, 0.0], [warmup, 1.0]] if warmup else [[0, 1.0]]
    steps = list(range(warmup + every, until, every)) + [until]
    for t in steps:
        d = (t - warmup) * tokens_per_step
        knots.append([t, (1.0 + d / d1_tokens) ** -q])
    return knots


def cooldown_knots(base, *, start: int, end: int, every: int = 10) -> list[list[float]]:
    """`base` (knots, or a function of t such as an `EtaSchedule`) up to
    `start`, then times a linear decay to 0 at `end` (knots every `every`
    steps over the cooldown). Before `start` only the knots of a table are
    kept; a branch starts at `start`, so only t >= start is ever read."""
    if not 0 < start < end or every < 1:
        raise ValueError("cooldown_knots: 0 < start < end, every >= 1")
    f = base if callable(base) else (lambda t: table_at(base, t))
    head = [] if callable(base) else [list(k) for k in base if k[0] < start]
    head = (head or [[0, f(0)]]) + [[start, f(start)]]
    tail = []
    for t in list(range(start + every, end, every)) + [end]:
        tail.append([t, f(t) * (end - t) / (end - start)])
    return head + tail
