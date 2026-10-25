# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gram_polar_express's safety factor as a rule arg `safety` next to `steps`:
1.01 (the default) is today's kernel to the bit; another value scales the
polar-express triples as p(x / f), and reaches the kernel from the config."""

import pytest
import torch

from torchtitan.optimizers.disco import presets  # noqa: F401  (registers the rules)
from torchtitan.optimizers.disco.rules import rule_of
from torchtitan.optimizers.kernels import zeropower
from torchtitan.optimizers.kernels.gram_newton_schulz import (
    gram_coefficients,
    gram_polar_express,
)

BASE = zeropower._POLAR_EXPRESS_BASE


@pytest.fixture(autouse=True)
def _fresh_compiled_rules():
    """The spy backend gets traced into the cached compiled rule functions;
    drop them around each test so no other test reuses a spy's graph."""
    from torchtitan.optimizers.disco import presets as _presets, rules as _rules

    def clear():
        _rules._COMPILED.clear()
        _presets._SHIFTED.clear()
        torch._dynamo.reset()

    clear()
    yield
    clear()


def test_the_default_triples_are_todays_to_the_bit():
    for steps in (5, 6, 7, 9):
        assert gram_coefficients(steps, 1.01) == zeropower.polar_express_coefficients(
            steps
        )


def test_another_safety_scales_every_triple_but_the_limit():
    f = 1.05
    got = gram_coefficients(7, f)
    want = [(a / f, b / f**3, c / f**5) for a, b, c in BASE[:-1]][:7]
    assert got == want
    assert gram_coefficients(9, f)[-1] == BASE[-1]


def test_the_default_kernel_is_todays_kernel_to_the_bit():
    g = torch.Generator().manual_seed(0)
    G = torch.randn(3, 64, 32, generator=g)
    a = gram_polar_express(G, steps=6, use_kernels=False, work_dtype=torch.float32)
    b = gram_polar_express(
        G, steps=6, use_kernels=False, work_dtype=torch.float32, safety=1.01
    )
    assert torch.equal(a, b)
    c = gram_polar_express(
        G, steps=6, use_kernels=False, work_dtype=torch.float32, safety=1.05
    )
    assert not torch.equal(a, c)


@pytest.mark.parametrize(
    "group",
    [
        {
            "rule": "muon",
            "args": {
                "compute": "gram_polar_express",
                "steps": 7,
                "safety": 1.05,
                "scale": "one",
            },
        },
        {
            "rule": "angular_muown",
            "args": {"compute": "gram_polar_express", "steps": 7, "safety": 1.05},
        },
        {
            "rule": "hyperball",
            "args": {"compute": "gram_polar_express", "steps": 7, "safety": 1.05},
        },
    ],
)
def test_the_safety_reaches_the_kernel(group, monkeypatch):
    seen = []

    def spy(x, steps=5, eps=1e-7, safety=1.01):
        seen.append((steps, safety))
        return torch.zeros_like(x)

    monkeypatch.setitem(zeropower.zeropower_backends, "gram_polar_express", spy)
    rule = rule_of(group)
    x = torch.randn(1, 16, 8)
    w = torch.randn(1, 16, 8)
    from torchtitan.optimizers.disco.rules import StepCtx

    ctx = StepCtx(0.01, 0.0, 5, False)
    state = torch.zeros(1, *rule.compute.state_shape(16, 8)) if rule.has_state else None
    try:
        rule.run_whole(x, w, ctx, state)
    except Exception:
        pass  # a zero direction may not be a valid step; only the call matters
    assert seen and all(s == (7, 1.05) for s in seen), seen


def test_safety_without_a_gram_compute_is_refused():
    with pytest.raises(ValueError, match="safety"):
        rule_of(
            {
                "rule": "muon",
                "args": {"compute": "newtonschulz5", "steps": 5, "safety": 1.05},
            }
        )
    with pytest.raises(ValueError, match="safety"):
        rule_of(
            {
                "rule": "muon",
                "args": {"compute": "gram_polar_express", "steps": 5, "safety": 0.9},
            }
        )


def test_an_inheriting_row_rule_ignores_the_safety():
    """An embedding group inherits the default group's args: its row-l2 compute has no Gram iteration."""
    rule = rule_of(
        {
            "rule": "muon",
            "args": {"compute": "row-l2", "scale": "sqrt_n", "safety": 1.05},
        }
    )
    assert rule.kind == "rows"
