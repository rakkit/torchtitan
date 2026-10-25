# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The AUS rule: what it declares, the map's invariants, the eta schedule."""

import math  # noqa: F401

import pytest
import torch

from torchtitan.optimizers.disco.presets import AUSMap, EtaSchedule
from torchtitan.optimizers.disco.rules import rule_of, StepCtx


def _mats(b, m, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, m, n, generator=g)


def _aus(**args):
    return rule_of(
        {
            "rule": "aus",
            "args": {"compute": "svd", "scale": "spectral", **args},
        }
    )


def test_what_aus_declares_and_group_selection():
    r = rule_of(
        dict(
            rule="aus",
            args=dict(compute="newtonschulz5", steps=5, scale="spectral"),
        )
    )
    assert isinstance(r.compute, AUSMap)
    assert (r.kind, r.needs_weight, r.has_state, r.output) == (
        "unit",
        True,
        True,
        "update_coeffs",
    )
    assert r.tail == ()  # the scale lives inside the map, before the rotation
    assert r.compute.inner.output == "update"  # the plain direction it rotates along
    # A group inheriting `aus` with row arguments stays that row rule.
    row = rule_of(dict(rule="aus", args=dict(compute="sign", scale="inv_n")))
    assert row.kind == "rows" and row.output == "update"
    with pytest.raises(ValueError):
        rule_of(dict(rule="aus", args={"bogus": 1}))


def test_map_matches_its_equations_and_keeps_the_radius():
    eta = 0.05  # lr 0.1 times eta(20) = 1/2
    rule = _aus(sigma_iters=30, eta={"shape": "inverse_poly", "start": 10, "c": 0.1})
    W, G = _mats(3, 24, 16, 1), _mats(3, 24, 16, 2) * 1e-2
    res = rule.run_whole(G, W, StepCtx(lr=0.1, wd=0.0, t=20))
    W_new = res.alpha[:, None, None] * W - res.update
    U = rule.compute.inner.run_whole(G).update
    # sigma(W_new) = sigma(W - eta U) by construction with exact sigmas; the
    # warm power iteration is a lower bound on both sides, so a few percent.
    s_new = torch.linalg.matrix_norm(W_new, ord=2)
    s_ref = torch.linalg.matrix_norm(W - eta * U, ord=2)
    torch.testing.assert_close(s_new, s_ref, rtol=5e-2, atol=0)
    # The direction moved by about eta (the relative displacement of W / ||W||).
    q0, q1 = W / s_ref[:, None, None], W_new / s_new[:, None, None]
    rel = torch.linalg.matrix_norm(q1 - q0, ord=2)
    assert torch.all(rel > 0.5 * eta) and torch.all(rel < 2.0 * eta)
    assert res.state.shape == (3, 16)
    # Weight decay scales W before the sigmas: alpha carries it.
    decayed = rule.run_whole(G, W, StepCtx(lr=0.1, wd=5.0, t=20))  # decay 0.5
    torch.testing.assert_close(decayed.alpha, res.alpha * 0.5, rtol=5e-2, atol=0)
    # No schedule: the step is the group's lr.
    by_lr = _aus(sigma_iters=30).run_whole(G, W, StepCtx(eta, 0.0, 7))
    torch.testing.assert_close(by_lr.alpha, res.alpha, rtol=1e-6, atol=0)


def test_batched_call_equals_per_matrix_calls():
    rule = _aus(sigma_iters=8)
    W, G = _mats(2, 12, 20, 5), _mats(2, 12, 20, 6)
    ctx = StepCtx(0.02, 0.0, 3)
    both = rule.run_whole(G, W, ctx)
    for i in range(2):
        one = rule.run_whole(G[i : i + 1], W[i : i + 1], ctx)
        torch.testing.assert_close(
            one.update, both.update[i : i + 1], rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(one.alpha, both.alpha[i : i + 1], rtol=1e-5, atol=0)


def test_eta_schedule_is_checked_when_the_rule_is_made():
    assert _aus().compute.eta is None  # no schedule: the step is lr
    sched = _aus(eta={"shape": "sqrt_decay", "start": 100, "total": 500}).compute.eta
    assert isinstance(sched, EtaSchedule) and sched.total == 500
    for bad in (
        {"kappa": 0.2},
        {"shape": "nope"},
        {"shape": "sqrt_decay", "c": 0.1, "total": 5},
    ):
        with pytest.raises(ValueError):
            _aus(eta=bad)
