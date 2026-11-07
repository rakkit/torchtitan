# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GatherTokenDispatcher (EP 1): dispatch and combine as row gathers. Same math
as the scatter-add LocalTokenDispatcher (fp64), and in bf16 at least as close to
the fp64 result: the combine sums each token's K score-weighted rows in fp32 and
rounds once, the scatter-add adds them into a bf16 accumulator one by one."""

import torch

from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher
from torchtitan.models.opt_moe.moe import GatherTokenDispatcher

T, D, E, K = 256, 32, 8, 4


def _experts(routed_ND, counts_E, weight_EDD):
    rows, start = [], 0
    for e, n in enumerate(counts_E.tolist()):
        rows.append(routed_ND[start : start + n] @ weight_EDD[e].T)
        start += n
    return torch.cat(rows)


def _run(dispatcher_cls, dtype):
    g = torch.Generator().manual_seed(0)
    x_TD = torch.randn(T, D, generator=g, dtype=torch.float64).to(dtype)
    logits_TE = torch.randn(T, E, generator=g, dtype=torch.float64)
    weight_EDD = (torch.randn(E, D, D, generator=g, dtype=torch.float64) / D**0.5).to(
        dtype
    )
    grad_TD = torch.randn(T, D, generator=g, dtype=torch.float64).to(dtype)
    x_TD.requires_grad_(True)
    weight_EDD.requires_grad_(True)
    scores_TK, ids_TK = torch.sigmoid(logits_TE).topk(K, dim=-1)
    scores_TK.requires_grad_(True)
    routing_TE = torch.zeros(T, E, dtype=torch.bool).scatter_(1, ids_TK, True)
    dispatcher = dispatcher_cls(dispatcher_cls.Config(num_experts=E, top_k=K))
    routed_ND, counts_E, metadata = dispatcher.dispatch(
        x_TD, scores_TK, ids_TK, routing_TE
    )
    out_TD = dispatcher.combine(
        _experts(routed_ND, counts_E, weight_EDD), metadata, x_TD
    )
    out_TD.backward(grad_TD)
    return [t.double() for t in (out_TD, x_TD.grad, scores_TK.grad, weight_EDD.grad)]


def test_gather_dispatch_is_the_scatter_add_math():
    for got, want in zip(
        _run(GatherTokenDispatcher, torch.float64),
        _run(LocalTokenDispatcher, torch.float64),
    ):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_gather_combine_in_bf16_is_at_least_as_accurate():
    reference = _run(LocalTokenDispatcher, torch.float64)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    gather = [
        rel(a, b)
        for a, b in zip(_run(GatherTokenDispatcher, torch.bfloat16), reference)
    ]
    scatter = [
        rel(a, b) for a, b in zip(_run(LocalTokenDispatcher, torch.bfloat16), reference)
    ]
    assert all(g <= s * 1.05 for g, s in zip(gather, scatter)), (gather, scatter)
