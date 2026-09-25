# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NormMoE's fast path must match a naive per-token, per-expert reference.

The production path routes with argsort, runs experts through a grouped GEMM,
and scatter-adds the results back. The reference below does the obvious thing
instead -- loop over tokens, loop over their selected experts, apply that
expert's weights, accumulate. It is far too slow to train with, but it is easy
to read and check by eye, so agreement between the two is real evidence rather
than two implementations sharing an assumption.

Run: python tests/unit_tests/opt_moe/test_moe_reference.py
"""

import torch

from torchtitan.models.opt_moe.norm_moe import make_norm_moe_config


def reference_moe(moe, x_TD, *, mid_norm: bool):
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
            if mid_norm:  # parameter-free RMS norm over the hidden dim
                hf = h.float()
                h = (hf * torch.rsqrt(hf.pow(2).mean() + 1e-30)).to(h.dtype)
            y = h @ w2[e].bfloat16().t()
            out[t] += y.float() * topk_scores[t, k].float()

    if moe.shared_experts is not None:
        out = out + moe.shared_experts(x_TD).float()
    return out


def run_case(*, norm_everywhere, num_shared_experts, top_k, label):
    torch.manual_seed(0)
    cfg = make_norm_moe_config(
        dim=64,
        hidden_dim=32,
        num_experts=6,
        top_k=top_k,
        num_shared_experts=num_shared_experts,
        norm_everywhere=norm_everywhere,
        load_balance_coeff=1e-3,
    )
    moe = cfg.build().cuda()
    with torch.no_grad():
        moe.init_states(buffer_device=torch.device("cuda"))
    moe.eval()

    x = torch.randn(48, 64, device="cuda")
    with torch.no_grad():
        fast = moe(x)
        ref = reference_moe(moe, x, mid_norm=norm_everywhere)

    d = (fast.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    ok = d < 5e-2 * scale
    print(
        f"  {'OK  ' if ok else 'FAIL'} {label}: max|d|={d:.3e} "
        f"scale={scale:.3f} rel={d/scale:.2e}"
    )
    return not ok


def main() -> int:
    failures = 0
    failures += run_case(
        norm_everywhere=False,
        num_shared_experts=0,
        top_k=1,
        label="top_k=1, no shared, no mid-norm",
    )
    failures += run_case(
        norm_everywhere=False,
        num_shared_experts=0,
        top_k=2,
        label="top_k=2, no shared, no mid-norm",
    )
    failures += run_case(
        norm_everywhere=True,
        num_shared_experts=0,
        top_k=2,
        label="top_k=2, no shared, mid-norm",
    )
    failures += run_case(
        norm_everywhere=True,
        num_shared_experts=1,
        top_k=2,
        label="top_k=2, shared expert, mid-norm",
    )
    failures += run_case(
        norm_everywhere=True,
        num_shared_experts=1,
        top_k=4,
        label="top_k=4, shared expert, mid-norm",
    )
    print()
    print("MATCHES REFERENCE" if failures == 0 else f"{failures} CASE(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
