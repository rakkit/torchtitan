# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pin the routing semantics we inherit from upstream's TokenChoiceTopKRouter.

We import upstream's router rather than vendoring one, which means upstream's
*defaults* silently become our semantics. This test re-derives routing from the
DeepSeek-V3 equations independently of the router implementation, so a changed
upstream default (or a config field we forgot to set) fails here instead of in a
training curve.
"""

import pytest
import torch

from torchtitan.models.opt_moe.flavors import make_norm_moe_config


def reference_routing(x, gate_w, bias, top_k, route_scale):
    """DeepSeek-V3 routing, written from the equations, not from the code."""
    scores = torch.sigmoid((x.float() @ gate_w.float().t()))
    # The bias steers selection only; gate values come from the raw scores.
    choice = scores if bias is None else scores + bias
    idx = torch.topk(choice, k=top_k, dim=-1, sorted=False).indices
    top = scores.gather(-1, idx)
    top = top / (top.sum(dim=-1, keepdim=True) + 1e-20)  # Eq 19-style renorm
    return top * route_scale, idx, scores


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("with_bias", [False, True])
def test_routing_follows_the_deepseek_v3_equations(top_k, with_bias):
    torch.manual_seed(0)
    dim, num_experts, T = 64, 8, 128
    router = make_norm_moe_config(
        dim=dim, hidden_dim=128, num_experts=num_experts, top_k=top_k
    ).router.build()
    router.eval()
    x = torch.randn(T, dim)
    bias = torch.randn(num_experts) if with_bias else None
    with torch.no_grad():
        top_s, top_i, _ = router(x, bias)
        ref_s, ref_i, _ = reference_routing(
            x, router.gate.weight, bias, top_k, router.route_scale
        )
    # The same experts per token (topk(sorted=False) order is arbitrary) ...
    assert torch.equal(
        torch.sort(top_i, dim=-1).values, torch.sort(ref_i, dim=-1).values
    )
    # ... with the same gate values, aligned by expert id.
    dense = torch.zeros(T, num_experts).scatter_(-1, top_i, top_s.float())
    ref_dense = torch.zeros(T, num_experts).scatter_(-1, ref_i, ref_s.float())
    assert (dense - ref_dense).abs().max().item() < 1e-5
    # route_norm: the gate weights sum to route_scale.
    s = top_s.sum(dim=-1)
    torch.testing.assert_close(
        s, torch.full_like(s, router.route_scale), atol=1e-4, rtol=0
    )
