# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RowL2's division guard: the row map x / (||x|| + EPS), EPS = 1e-30 the module
constant (not an arg: `test_rules.test_eps_is_a_constant_not_an_arg`), and the
shared guard `_norm(x, dim, eps)` written from its equation."""

import pytest
import torch

from torchtitan.optimizers.disco.presets import _norm, EPS, RowL2


def _x(seed=0):
    g = torch.randn(64, 32, generator=torch.Generator().manual_seed(seed))
    return g * torch.logspace(-14, 2, 64).unsqueeze(1)  # row norms over 16 decades


def _want(x, eps):
    n = x.double().norm(dim=-1, keepdim=True)
    return (x.double() / (n + eps)).float()


def test_row_l2_is_x_over_norm_plus_the_constant_bitwise():
    x = _x()
    n = x.float().pow(2).sum(dim=-1, keepdim=True).sqrt()
    assert EPS == 1e-30
    assert torch.equal(RowL2().core(x), (x.float() / (n + 1e-30)).to(x.dtype))


@pytest.mark.parametrize("eps", [1e-6, 1e-4, 1e-2])
def test_the_guard_is_x_over_norm_plus_eps(eps):
    x = _x(1)
    got = _norm(x, -1, eps)
    torch.testing.assert_close(got, _want(x, eps), rtol=1e-5, atol=0)
    n = x.norm(dim=-1)
    out = got.norm(dim=-1)
    big, small = n > 1e3 * eps, n < 1e-3 * eps
    assert big.any() and small.any()
    torch.testing.assert_close(out[big], torch.ones_like(out[big]), rtol=2e-3, atol=0)
    torch.testing.assert_close(
        out[small], n[small] / eps, rtol=2e-3, atol=0
    )  # a row far below eps keeps its size / eps
