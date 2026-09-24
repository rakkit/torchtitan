# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`left_polar`: the interchangeable-rows rule for wide units (MoE routers).

The claim is that adding a constant vector to every row is invisible to the
softmax downstream, so the rule projects that direction out of the gradient
and again out of the update, and everything between is ordinary Muon.
"""

import pytest
import torch

from torchtitan.optimizers.disco.presets import Centre, Orthogonalise, RowNorm
from torchtitan.optimizers.disco.rules import rule_of, Scale


def _g(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * 1e-2


def _left(**args):
    return rule_of({"rule": "left_polar", "args": {"compute": "svd", **args}})


def test_rule_selection_and_what_it_reduces():
    r = _left(steps=5)
    assert r.pre == (Centre(),) and isinstance(r.compute, Orthogonalise)
    assert r.kind == "unit" and r.post[-1] == Centre()
    # the owner has the unit whole, so the re-centring runs there, then the scale
    assert r.rest_tail == (Centre(), Scale("spectral"))


def test_centring_is_the_rule_not_an_arg():
    with pytest.raises(ValueError, match="pre"):
        rule_of({"rule": "left_polar", "args": {"pre": "row-l2"}})


def test_update_has_no_all_ones_component():
    """Both projections: the update's column means are zero, whatever the
    gradient's are."""
    g = _g(4, 64) + 0.5  # a large constant offset the rule must remove
    u = _left(steps=5).reference(g).update
    assert u.shape == g.shape
    assert u.mean(dim=-2).norm() < 1e-6 * u.norm()


def test_matches_muon_on_the_centred_gradient():
    """Nothing between the two projections is new: it is Muon."""
    args = {"compute": "svd", "steps": 5, "scale": "spectral"}
    left = rule_of({"rule": "left_polar", "args": args})
    muon = rule_of({"rule": "muon", "args": args})
    g = _g(4, 64, seed=3)
    want = muon.reference(g - g.mean(dim=-2, keepdim=True)).update
    want = want - want.mean(dim=-2, keepdim=True)
    torch.testing.assert_close(left.reference(g).update, want, rtol=1e-6, atol=1e-7)


def test_split_rows_reduce_to_the_whole():
    """What the optimizer does on a row-sharded unit: the pre statistic summed
    over the holders, each centres its own rows, the owner gets the unit whole
    and runs compute and every post phase there."""
    r = _left(steps=5)
    g = _g(8, 32, seed=5)
    whole = r.reference(g).update
    (ph,) = r.pre
    stat = sum(ph.stat(g[i : i + 4]) for i in (0, 4))
    centred = torch.cat([ph(g[i : i + 4], stat, 8) for i in (0, 4)])
    torch.testing.assert_close(r.run_whole(centred).update, whole, rtol=1e-6, atol=1e-7)


def test_hybrid_polar_is_a_muon_config():
    """The SwiGLU rule of the same paper (row-norm, then spectral) needs no
    class: a row-wise pre phase (no reduction) and the usual whole-unit compute."""
    r = rule_of(
        {
            "rule": "muon",
            "args": {"pre": "row-l2", "compute": "svd", "steps": 5},
        }
    )
    assert r.pre == (RowNorm(),) and not r.pre[0].reduce and r.kind == "unit"
    g = _g(32, 16, seed=7)
    rows = g / g.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(r.pre[0].whole(g), rows, rtol=1e-5, atol=1e-7)
