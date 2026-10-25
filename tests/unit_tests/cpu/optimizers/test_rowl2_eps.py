# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RowL2's division guard `eps`: the row map x / (||x|| + eps), written from the equation. The default is the
module constant EPS and builds the same compute as before; `eps` is a muon arg of the row-l2 compute only and is
never inherited from the default group."""

import pytest
import torch

from torchtitan.optimizers.disco.presets import EPS, RowL2
from torchtitan.optimizers.disco.rules import rule_of, RULES


def _x(seed=0):
    g = torch.randn(64, 32, generator=torch.Generator().manual_seed(seed))
    return g * torch.logspace(-14, 2, 64).unsqueeze(1)  # row norms over 16 decades


def _want(x, eps):
    n = x.double().norm(dim=-1, keepdim=True)
    return (x.double() / (n + eps)).float()


def test_default_eps_is_the_module_constant_and_the_old_map_bitwise():
    x = _x()
    n = x.float().pow(2).sum(dim=-1, keepdim=True).sqrt()
    old = (x.float() / (n + 1e-30)).to(x.dtype)  # the pre-eps formula
    assert EPS == 1e-30 and RowL2().eps == EPS and RowL2() == RowL2(eps=EPS)
    assert torch.equal(RowL2().core(x), old)
    assert torch.equal(RowL2(eps=1e-30).core(x), old)


@pytest.mark.parametrize("eps", [1e-6, 1e-4, 1e-2])
def test_eps_is_the_guard_in_x_over_norm_plus_eps(eps):
    x = _x(1)
    got = RowL2(eps=eps).core(x)
    torch.testing.assert_close(got, _want(x, eps), rtol=1e-5, atol=0)
    n = x.norm(dim=-1)
    out = got.norm(dim=-1)
    big, small = n > 1e3 * eps, n < 1e-3 * eps
    assert big.any() and small.any()
    torch.testing.assert_close(out[big], torch.ones_like(out[big]), rtol=2e-3, atol=0)
    torch.testing.assert_close(
        out[small], n[small] / eps, rtol=2e-3, atol=0
    )  # a row far below eps keeps its size / eps


def test_muon_eps_builds_the_row_compute_with_that_guard():
    got = rule_of(
        {
            "rule": "muon",
            "args": {
                "compute": "row-l2",
                "scale": "sqrt_n",
                "pre": "identity",
                "eps": 1e-5,
            },
        }
    )
    assert got.compute == RowL2(eps=1e-5) and got.kind == "rows"
    default = rule_of(
        {
            "rule": "muon",
            "args": {"compute": "row-l2", "scale": "sqrt_n", "pre": "identity"},
        }
    )
    assert default.compute == RowL2()
    x = _x(2)
    assert not torch.equal(
        got.reference(x).update, default.reference(x).update
    )  # rows below 1e-5 are shrunk
    assert torch.equal(
        default.reference(x).update,
        rule_of(
            {
                "rule": "muon",
                "args": {
                    "compute": "row-l2",
                    "scale": "sqrt_n",
                    "pre": "identity",
                    "eps": 1e-30,
                },
            }
        )
        .reference(x)
        .update,
    )


def test_eps_is_never_inherited_from_the_default_group():
    assert (
        "eps" in RULES["muon"].args and "eps" not in RULES["muon"].inherits
    )  # the default's Adam eps (mbetas.fant) must not reach the row rule


@pytest.mark.parametrize(
    "args", [{"compute": "newtonschulz5"}, {"compute": "row-l2", "block_rows": 8}]
)
def test_eps_is_for_the_row_l2_compute_only(args):
    with pytest.raises(ValueError, match="row-l2 compute"):
        rule_of({"rule": "muon", "args": {**args, "eps": 1e-5}})
