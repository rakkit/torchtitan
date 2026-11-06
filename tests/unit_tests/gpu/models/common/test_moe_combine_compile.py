# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ``moe_score_and_scatter_add`` region (the local token dispatcher's
combine): compiled, the forward and the expert-output gradient are the eager
results bitwise (the score multiply fuses, the scatter-add stays the
deterministic custom op), the score gradient is a fused reduction within
rounding of eager's, a compiled step repeats bitwise (no atomic adds), and two
token counts plus a no_grad pass stay within the region's graph budget."""

import unittest

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher

E, K, D = 64, 8, 2048


def _inputs(num_tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids_TK = torch.rand(num_tokens, E, device="cuda", generator=g).topk(K).indices
    order_N = torch.argsort(ids_TK.view(-1), stable=True)
    scores_N = torch.rand(num_tokens * K, device="cuda", generator=g)
    routed_ND = torch.randn(
        num_tokens * K, D, device="cuda", dtype=torch.bfloat16, generator=g
    )
    grad_TD = torch.randn(
        num_tokens, D, device="cuda", dtype=torch.bfloat16, generator=g
    )
    x_TD = torch.zeros(num_tokens, D, device="cuda", dtype=torch.bfloat16)
    return routed_ND, scores_N, order_N // K, x_TD, grad_TD


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMoECombineLocalCompile(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.dispatcher = LocalTokenDispatcher.Config(num_experts=E, top_k=K).build()

    def tearDown(self):
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    def _step(self, routed_ND, scores_N, tokens_N, x_TD, grad_TD):
        routed_ND = routed_ND.clone().requires_grad_()
        scores_N = scores_N.clone().requires_grad_()
        out_TD = self.dispatcher._score_and_scatter_add(
            routed_ND, scores_N, tokens_N, x_TD
        )
        grads = torch.autograd.grad(out_TD, (routed_ND, scores_N), grad_TD)
        return out_TD.detach(), *grads

    def test_compiled_combine_is_eager_and_repeats_bitwise(self):
        inputs = _inputs(4096, seed=0)
        LocalCompileConfig().apply()
        eager = self._step(*inputs)
        LocalCompileConfig(regions=["moe_score_and_scatter_add"]).apply()
        torch._dynamo.utils.counters.clear()
        first, second = self._step(*inputs), self._step(*inputs)
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)
        out, grad_routed, grad_scores = first
        torch.testing.assert_close(out, eager[0], rtol=0, atol=0)
        torch.testing.assert_close(grad_routed, eager[1], rtol=0, atol=0)
        torch.testing.assert_close(grad_scores, eager[2], rtol=1e-5, atol=1e-5)
        for got, want in zip(second, first, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_token_counts_and_no_grad_stay_within_the_graph_budget(self):
        LocalCompileConfig(regions=["moe_score_and_scatter_add"]).apply()
        torch._dynamo.utils.counters.clear()
        for num_tokens in (4096, 3000, 1234):
            self._step(*_inputs(num_tokens, seed=num_tokens))
        with torch.no_grad():
            routed_ND, scores_N, tokens_N, x_TD, _ = _inputs(2048, seed=1)
            self.dispatcher._score_and_scatter_add(routed_ND, scores_N, tokens_N, x_TD)
        # static shapes, then one dynamic graph for any token count, then no_grad
        frames = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        self.assertLessEqual(frames, 3)


if __name__ == "__main__":
    unittest.main()
