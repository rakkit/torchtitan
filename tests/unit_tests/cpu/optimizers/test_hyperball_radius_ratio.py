# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hyperball with a prescribed radius path (`radius_ratio`): after step t the
radius is c(t) / c(0) times the starting radius, the direction is plain
hyperball's, and c = 1 is plain hyperball bitwise."""

import math

import pytest
import torch

from torchtitan.optimizers.disco.presets import FrobeniusAUS
from torchtitan.optimizers.disco.rules import rule_of, StepCtx

LR = 0.05
PATH = [[0, 1.0], [1, 0.7], [4, 2.0], [7, 5.0]]


def _c(t):
    """The path read by hand: 1 at 0, 0.7 at 1, linear to 2 at 4, linear to 5 at 7, flat after."""
    if t <= 0:
        return 1.0
    if t <= 1:
        return 1.0 - 0.3 * t
    if t <= 4:
        return 0.7 + 1.3 * (t - 1) / 3
    if t <= 7:
        return 2.0 + 3.0 * (t - 4) / 3
    return 5.0


def _rule(**args):
    return rule_of({"rule": "hyperball", "args": {"compute": "svd", **args}})


def _run(W, grads, **args):
    rule, w, out = _rule(**args), W[None].clone(), []
    for t, G in enumerate(grads, start=1):
        res = rule.run_whole(G[None], w, StepCtx(LR, 0.0, t))
        w = res.alpha[:, None, None] * w - res.update
        out.append(w[0].clone())
    return out


def _data(seed=0, steps=10):
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(12, 8, generator=gen, dtype=torch.float64) / math.sqrt(8)
    return W, [
        torch.randn(12, 8, generator=gen, dtype=torch.float64) for _ in range(steps)
    ]


def test_ratio_one_is_plain_hyperball_bitwise():
    W, grads = _data()
    plain = _run(W, grads)
    one = _run(W, grads, radius_ratio=[[0, 1.0]])
    assert all(torch.equal(a, b) for a, b in zip(plain, one))


def test_radius_follows_the_path_and_the_direction_is_plain_hyperball():
    W, grads = _data(seed=1)
    r0 = float(W.norm())
    plain = _run(W, grads)
    path = _run(W, grads, radius_ratio=PATH)
    for t, (p, q) in enumerate(zip(plain, path), start=1):
        # FrobeniusAUS works in float32, so even plain hyperball's radius drifts ~3e-8
        assert abs(float(q.norm()) - r0 * _c(t)) < 1e-6 * r0 * _c(t), t
        torch.testing.assert_close(q / q.norm(), p / p.norm(), rtol=1e-7, atol=1e-7)


def test_the_comparison_sees_a_wrong_path():
    W, grads = _data(seed=2)
    r0 = float(W.norm())
    q = _run(W, grads, radius_ratio=[[0, 1.0], [1, 0.7], [4, 2.1], [7, 5.0]])
    assert abs(float(q[3].norm()) - r0 * _c(4)) > 1e-3 * r0


@pytest.mark.parametrize(
    "pts", [[[3, 1.0], [2, 2.0]], [[0, 1.0], [0, 2.0]], [[0, 1.0], [5, 0.0]]]
)
def test_bad_paths_are_refused(pts):
    with pytest.raises(ValueError, match="radius_ratio"):
        _rule(radius_ratio=pts)


def test_a_free_radius_takes_no_path():
    inner = _rule().compute.inner
    with pytest.raises(ValueError, match="radius='keep'"):
        FrobeniusAUS(inner, "free", radius_ratio=((0, 1.0),))
