# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OptMoE's fast path must match a naive per-token, per-expert reference.

The production path routes with argsort, runs experts through a grouped GEMM,
and scatter-adds the results back. The reference below does the obvious thing
instead -- loop over tokens, loop over their selected experts, apply that
expert's weights, accumulate. It is far too slow to train with, but it is easy
to read and check by eye, so agreement between the two is real evidence rather
than two implementations sharing an assumption.

"""

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config


def _moe_config(*, dim, hidden_dim, num_experts, top_k, **kw):
    """One OptMoE layer's config: gauge-d8-w256 built with a single MoE layer
    (no depth scaling, as one layer alone)."""
    return (
        build_model_config(
            "gauge-d8-w256",
            parallelism=ParallelismConfig(),
            n_layers=1,
            dim=dim,
            n_dense_layers=0,
            depth_init="identity",
            moe_hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            **kw,
        )
        .layers[0]
        .feed_forward
    )


def reference_moe(moe, x_TD):
    """Naive MoE: for each token, apply each of its top-k experts directly."""
    router = moe.router
    experts = moe.routed_experts

    scores_TE = torch.sigmoid(router.gate(x_TD))
    bias = moe.expert_bias_E
    choice = scores_TE if bias is None else scores_TE + bias
    topk_ids = torch.topk(choice, k=router.top_k, dim=-1, sorted=False).indices
    topk_scores = scores_TE.gather(-1, topk_ids)
    # DeepSeek-V3 renormalises the selected weights before scaling. This must be
    # derived from the equations, not copied from the model: an earlier version
    # of this reference omitted it and so agreed with a buggy model to 0 ULP.
    topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)
    topk_scores = topk_scores * router.route_scale

    # w13 is [E, 2, F, D]: the gate (w1) and up (w3) projections, stacked.
    w1, w3 = experts.w13.weight.unbind(1)
    w2 = experts.w2.weight
    out = torch.zeros_like(x_TD, dtype=torch.float32)
    for t in range(x_TD.shape[0]):
        xt = x_TD[t].bfloat16()
        for k in range(router.top_k):
            e = int(topk_ids[t, k])
            # Same casts as the grouped GEMM, so any difference is routing or
            # accumulation, not precision.
            h = torch.nn.functional.silu(xt @ w1[e].bfloat16().t())
            h = h * (xt @ w3[e].bfloat16().t())
            y = h @ w2[e].bfloat16().t()
            out[t] += y.float() * topk_scores[t, k].float()

    if moe.shared_experts is not None:
        out = out + moe.shared_experts(x_TD).float()
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the grouped GEMM needs CUDA")
@pytest.mark.parametrize("num_shared_experts, top_k", [(0, 1), (0, 2), (1, 2), (1, 4)])
def test_moe_matches_the_naive_reference(num_shared_experts, top_k):
    torch.manual_seed(0)
    cfg = _moe_config(
        dim=64,
        hidden_dim=32,
        num_experts=6,
        top_k=top_k,
        num_shared_experts=num_shared_experts,
        load_balance_coeff=1e-3,
    )
    moe = cfg.build().cuda()
    with torch.no_grad():
        moe.init_states(buffer_device=torch.device("cuda"))
    moe.eval()
    x = torch.randn(48, 64, device="cuda")
    with torch.no_grad():
        fast = moe(x)
        ref = reference_moe(moe, x)
    d = (fast.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert d < 5e-2 * scale, (d, scale)
