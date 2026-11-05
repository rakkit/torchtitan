# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.opt_moe.norm_moe import NormRouter
from torchtitan.models.opt_moe.utils.inits import make_param_init

E, K, D, T = 16, 2, 32, 2048


def _router(**kw) -> NormRouter:
    score_func = kw.pop("score_func", "sigmoid")
    return NormRouter.Config(
        num_experts=E,
        top_k=K,
        score_func=score_func,
        gate=HiMidLoLinear.Config(
            backward_mode="hi_mid_lo",
            in_features=D,
            out_features=E,
            param_init={"weight": make_param_init("scion_normal_output", 1.0)},
        ),
        **kw,
    ).build()


def test_off_by_default_and_routing_is_unchanged():
    torch.manual_seed(0)
    plain = _router()
    torch.manual_seed(0)
    explicit_off = _router(quantile_balance_bins=None)
    assert plain.quantile_balancer is None

    x, bias = torch.randn(T, D), torch.randn(E) * 0.01
    plain.train()
    explicit_off.train()
    a, b = plain(x, bias), explicit_off(x, bias)
    assert torch.equal(a.topk_expert_ids_TK, b.topk_expert_ids_TK)
    assert torch.equal(a.topk_scores_TK, b.topk_scores_TK)


def test_one_update_beats_the_sign_nudge():
    """The histogram solves for the quantile, so balance improves in one step."""
    torch.manual_seed(0)
    r = _router(quantile_balance_bins=64)
    r.train()
    x, bias = torch.randn(T, D), torch.zeros(E)

    out = r(x, bias)
    counts = torch.bincount(out.topk_expert_ids_TK.reshape(-1), minlength=E).float()
    before = ((counts.max() - counts.mean()) / counts.mean()).item()

    hist = r.quantile_balancer.required_bias_histogram_EB
    new_bias = r.quantile_balancer.estimate_expert_bias(hist, bias)
    hist.zero_()

    out2 = r(x, new_bias)
    counts2 = torch.bincount(out2.topk_expert_ids_TK.reshape(-1), minlength=E).float()
    after = ((counts2.max() - counts2.mean()) / counts2.mean()).item()

    assert after < before / 2, f"maxvio {before:.4f} -> {after:.4f}"
    # estimate_expert_bias mean-centres, so the bias cannot drift.
    assert abs(new_bias.mean().item()) < 1e-5


def test_histogram_counts_only_unmasked_tokens():
    """opt_moe's loss_mask marks VALID tokens; observe() takes a padding mask."""
    torch.manual_seed(0)
    r = _router(quantile_balance_bins=32)
    r.train()
    valid = 100
    mask = torch.zeros(T, dtype=torch.bool)
    mask[:valid] = True
    r(torch.randn(T, D), torch.zeros(E), loss_mask=mask)
    assert int(r.quantile_balancer.required_bias_histogram_EB.sum()) == valid * E


def test_eval_falls_back_to_plain_topk():
    torch.manual_seed(0)
    q = _router(quantile_balance_bins=32)
    torch.manual_seed(0)
    plain = _router()
    x, bias = torch.randn(T, D), torch.randn(E) * 0.01
    q.eval()
    plain.eval()
    assert torch.equal(q(x, bias).topk_expert_ids_TK, plain(x, bias).topk_expert_ids_TK)
    assert int(q.quantile_balancer.required_bias_histogram_EB.sum()) == 0


def test_requires_an_expert_bias():
    r = _router(quantile_balance_bins=32)
    r.train()
    with pytest.raises(ValueError, match="requires an expert bias"):
        r(torch.randn(T, D), None)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"score_func": "softmax"}, "sigmoid"),
        ({"_debug_force_load_balance": True}, "_debug_force_load_balance"),
        ({"num_expert_groups": 4, "num_limited_groups": 2}, "node-limited"),
    ],
)
def test_rejects_incompatible_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _router(quantile_balance_bins=8, **kwargs)
