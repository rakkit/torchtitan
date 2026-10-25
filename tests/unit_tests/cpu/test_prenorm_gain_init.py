# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``OPTMoETransformerBlock.Config.prenorm_gain_init``: the constant init of the attention_norm / ffn_norm gains.

The default (1.0) must leave the model exactly as before: the norms keep their stock ``reset_parameters`` and the init RNG stream
is untouched. A value x != 1 sets only the two pre-norm gains of every layer to x (a constant: no RNG), and nothing else changes.
"""

import functools
import re

import pytest
import torch

from torchtitan.models.opt_moe import model_registry

PRE = re.compile(r"(^|\.)(attention_norm|ffn_norm)\.weight$")


def _tiny(*, prenorm_gain_init=None, norm_type="rmsnorm", seed=0, n_layers=3):
    cfg = model_registry("gauge-d8-w256")
    cfg.n_layers, cfg.vocab_size = n_layers, 512
    if norm_type == "rmsnorm":
        cfg.layer.norm_type = cfg.layer.attention.norm_type = cfg.norm_type = "rmsnorm"
    else:
        cfg.layer.norm_type = norm_type
    if prenorm_gain_init is not None:
        cfg.layer.prenorm_gain_init = prenorm_gain_init
    cfg._expand_layers()
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device("cpu"))
    return model


def _params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_the_default_leaves_the_stock_norm_init_in_place():
    a, b = _tiny(), _tiny(prenorm_gain_init=1.0)
    pa, pb = _params(a), _params(b)
    assert pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa)
    for model in (a, b):
        for layer in model.layers.values():
            assert (
                layer.attention_norm._param_init is None
                and layer.ffn_norm._param_init is None
            )
    assert all(
        torch.equal(v, torch.ones_like(v)) for k, v in pa.items() if PRE.search(k)
    )


def test_a_pre_norm_gain_init_sets_only_the_two_pre_norm_gains():
    base, moved = _params(_tiny()), _params(_tiny(prenorm_gain_init=4.0))
    assert base.keys() == moved.keys()
    pre = [k for k in base if PRE.search(k)]
    assert len(pre) == 2 * 3 and all(
        torch.equal(moved[k], torch.full_like(moved[k], 4.0)) for k in pre
    )
    others = [k for k in base if k not in pre]
    assert others and all(torch.equal(base[k], moved[k]) for k in others), [
        k for k in others if not torch.equal(base[k], moved[k])
    ]
    norms = [k for k in others if k.endswith("norm.weight")]
    assert norms and all(
        torch.equal(moved[k], torch.ones_like(moved[k])) for k in norms
    ), norms  # q/k-norm and the final norm stay at 1


def test_the_init_is_a_constant_that_consumes_no_rng():
    torch.manual_seed(5)
    _tiny(prenorm_gain_init=4.0, seed=7)
    after_a = torch.rand(3)
    torch.manual_seed(5)
    _tiny(seed=7)
    after_b = torch.rand(3)
    assert torch.equal(after_a, after_b)


def test_the_value_needs_a_gain_to_initialise():
    with pytest.raises(ValueError, match="prenorm_gain_init"):
        _tiny(prenorm_gain_init=4.0, norm_type="np_rmsnorm")
    _tiny(prenorm_gain_init=1.0, norm_type="np_rmsnorm")  # the default is always fine


def test_the_param_init_is_a_constant_fill():
    model = _tiny(prenorm_gain_init=0.5)
    init = next(iter(model.layers.values())).attention_norm._param_init["weight"]
    assert (
        isinstance(init, functools.partial)
        and init.func is torch.nn.init.constant_
        and init.keywords == {"val": 0.5}
    )
