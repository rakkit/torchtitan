# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Gram rules: Q from the Gram alone is the polar factor, and a unit split
over ranks gives the same update as the whole unit on one rank."""

import torch

from torchtitan.optimizers.disco.presets import Centre, GramPolar
from torchtitan.optimizers.disco.rules import rule_of
from torchtitan.optimizers.kernels.gram_newton_schulz import (
    gram_polar_express,
    gram_polar_factor,
)


def _tall(seed=0, m=96, n=24):
    return torch.randn(m, n, generator=torch.Generator().manual_seed(seed))


def _phases_on_parts(phases, parts, m):
    """What the optimizer does on row-sharded parts: a reducing phase takes the
    sum of every part's statistic, the others act on each part alone."""
    for ph in phases:
        stat = sum(ph.stat(p) for p in parts) if ph.reduce else None
        parts = [ph(p, stat, m) for p in parts]
    return parts


def _split_run(rule, x, cut):
    """`rule` on `x` held as two row-shards, the way the local family runs it."""
    m = x.shape[0]
    parts = _phases_on_parts(rule.pre, [x[:cut], x[cut:]], m)
    if rule.kind == "gram":
        R = sum(rule.compute.gram(p) for p in parts)  # the all-reduced Gram
        parts = [rule.compute.finish(p, R, m) for p in parts]
        post = rule.tail
    else:
        parts = [rule.call_fused(p, m) for p in parts]
        post = rule.rest_tail
    return torch.cat(_phases_on_parts(post, parts, m))


def test_gram_polar_factor_matches_the_full_kernel():
    x = _tall()
    for steps in (5, 8):
        want = gram_polar_express(
            x, steps=steps, eps=1e-7, use_kernels=False, work_dtype=torch.float32
        )
        got = x @ gram_polar_factor(
            x.mT @ x, steps=steps, eps=1e-7, work_dtype=torch.float32
        )
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    # and it is a polar factor: orthonormal columns
    q = x @ gram_polar_factor(x.mT @ x, steps=8, eps=1e-7, work_dtype=torch.float32)
    torch.testing.assert_close(q.mT @ q, torch.eye(x.shape[1]), rtol=0, atol=2e-3)


def test_right_polar_split_rows_agree_with_the_whole():
    rule = rule_of(
        dict(
            rule="right_polar",
            args={"steps": 8, "scale": "spectral", "work_dtype": "float32"},
        )
    )
    assert rule.kind == "gram" and isinstance(rule.compute, GramPolar)
    x = _tall(3)
    torch.testing.assert_close(
        _split_run(rule, x, 40), rule.reference(x).update, rtol=1e-4, atol=1e-5
    )


def test_centring_and_the_lm_head_rule_agree_on_split_rows():
    x = _tall(5, 80, 16)
    # col-center as a pre-norm: reduce over rows, then subtract the mean
    mu = rule_of(dict(rule="muon", args={"pre": "col-center"}))
    assert mu.pre == (Centre(),) and mu.pre[0].reduce
    (whole,) = _phases_on_parts(mu.pre, [x], 80)
    parts = torch.cat(_phases_on_parts(mu.pre, [x[:30], x[30:]], 80))
    torch.testing.assert_close(parts, whole, rtol=1e-6, atol=1e-6)
    assert torch.allclose(whole.mean(0), torch.zeros(16), atol=1e-6)
    # the LM-head rule: centre -> right polar (or none) -> scale -> centre
    for spectral in (True, False):
        rule = rule_of(
            dict(
                rule="lm_head",
                args={"steps": 8, "spectral": spectral, "work_dtype": "float32"},
            )
        )
        assert rule.kind == ("gram" if spectral else "rows")
        assert rule.pre == (Centre(),) and rule.post[-1] == Centre()
        whole = rule.reference(x).update
        assert torch.allclose(whole.mean(0), torch.zeros(16), atol=1e-5)  # re-projected
        torch.testing.assert_close(_split_run(rule, x, 30), whole, rtol=1e-4, atol=1e-5)


def test_an_all_zero_unit_gives_a_zero_update_not_nan():
    """`EPS` is 1e-30 and the Gram iteration divides by the squared scale, which
    underflows to 0 in fp32 for an all-zero unit unless it is floored."""
    zero = torch.zeros(96, 24)
    for group in (
        {"rule": "right_polar", "args": {"steps": 8, "work_dtype": "float32"}},
        {"rule": "lm_head", "args": {"steps": 8, "work_dtype": "float32"}},
    ):
        u = rule_of(group).reference(zero).update
        assert torch.isfinite(u).all(), group["rule"]
        assert torch.equal(u, torch.zeros_like(u)), group["rule"]


def test_a_decayed_moment_gives_no_blow_up():
    """A unit whose gradient stopped (an unseen token's row, an idle expert)
    decays by the momentum factor each step. Once its squares underflow in
    fp32, `x / (norm + EPS)` became `x / 1e-30`: row rules returned updates
    ~1e6 and every orthogonaliser NaN. Below the floor the update is now 0."""
    cuda = torch.cuda.is_available()  # gram_polar_express runs on CUDA only
    device = "cuda" if cuda else "cpu"
    backends = ("newtonschulz5", "polar_express") + (
        ("gram_polar_express",) if cuda else ()
    )
    groups = [
        {"rule": "sgd", "args": {"compute": "row-l2", "scale": "sqrt_n"}},
        {"rule": "right_polar", "args": {"steps": 8, "work_dtype": "float32"}},
    ] + [
        {"rule": "muon", "args": {"compute": b, "steps": 6, "scale": "spectral"}}
        for b in backends
    ]
    torch.manual_seed(0)
    for group in groups:
        rule = rule_of(group)
        healthy = rule.reference(torch.randn(64, 32, device=device)).update.abs().max()
        for a in (1e-21, 1e-23, 1e-25, 1e-30, 1e-36):
            u = rule.reference(torch.randn(64, 32, device=device) * a).update
            assert torch.isfinite(u).all(), (group["rule"], group["args"], a)
            assert u.abs().max() <= healthy, (group["rule"], group["args"], a)


def test_a_rank_one_unit_gives_a_finite_polar_factor():
    """The resets re-formed the Gram from its fp16 copy: on a rank-deficient
    unit the rounding left negative eigenvalues that the polynomial blew up
    to inf (4 of 4 seeds on [512, 64] of rank 1)."""
    rule = rule_of({"rule": "right_polar", "args": {}})
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        u, v = torch.randn(512, 1, generator=g), torch.randn(1, 64, generator=g)
        got = rule.reference(u @ v).update
        assert torch.isfinite(got).all(), seed
        want = (u / u.norm()) @ (v / v.norm())  # the polar factor of u v^T
        cos = (got * want).sum() / (got.norm() * want.norm())
        assert cos > 0.95, (seed, float(cos))
