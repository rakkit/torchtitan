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

import torch

from torchtitan.models.opt_moe.norm_moe import make_norm_moe_config


def reference_routing(x, gate_w, bias, top_k, route_scale):
    """DeepSeek-V3 routing, written from the equations, not from the code."""
    scores = torch.sigmoid((x.float() @ gate_w.float().t()))
    # The bias steers selection only; gate values come from the raw scores.
    choice = scores if bias is None else scores + bias
    idx = torch.topk(choice, k=top_k, dim=-1, sorted=False).indices
    top = scores.gather(-1, idx)
    top = top / (top.sum(dim=-1, keepdim=True) + 1e-20)  # Eq 19-style renorm
    return top * route_scale, idx, scores


def main():
    torch.manual_seed(0)
    dim, num_experts, T = 64, 8, 128
    failures = []

    for top_k in (1, 2, 4):
        for with_bias in (False, True):
            cfg = make_norm_moe_config(
                dim=dim, hidden_dim=128, num_experts=num_experts, top_k=top_k
            )
            router = cfg.router.build()
            router.eval()

            x = torch.randn(T, dim)
            bias = torch.randn(num_experts) if with_bias else None
            with torch.no_grad():
                out = router(x, bias)
                top_s, top_i, scores = (
                    out.topk_scores_TK,
                    out.topk_expert_ids_TK,
                    out.scores_TE,
                )
                ref_s, ref_i, ref_scores = reference_routing(
                    x, router.gate.weight, bias, top_k, router.route_scale
                )

            tag = f"top_k={top_k} bias={with_bias}"
            # Compare as sets per token: topk(sorted=False) order is arbitrary.
            same_set = (
                (torch.sort(top_i, dim=-1).values == torch.sort(ref_i, dim=-1).values)
                .all()
                .item()
            )
            # Align scores by expert id before comparing.
            dense = torch.zeros(T, num_experts).scatter_(-1, top_i, top_s.float())
            ref_dense = torch.zeros(T, num_experts).scatter_(-1, ref_i, ref_s.float())
            d = (dense - ref_dense).abs().max().item()

            ok = same_set and d < 1e-5
            print(
                f"{'OK  ' if ok else 'FAIL'} {tag}: same_experts={same_set} "
                f"max|d_gate|={d:.3e} sum_top={top_s.sum(-1).mean().item():.4f}"
            )
            if not ok:
                failures.append(tag)

            # route_norm must make the gate weights sum to route_scale.
            s = top_s.sum(dim=-1)
            if not torch.allclose(s, torch.full_like(s, router.route_scale), atol=1e-4):
                print(
                    f"FAIL {tag}: gate weights do not sum to route_scale "
                    f"({s.mean().item():.4f} vs {router.route_scale:.4f}) "
                    "-- route_norm is off"
                )
                failures.append(tag + "/route_norm")

    print("ROUTER OK" if not failures else f"ROUTER FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
