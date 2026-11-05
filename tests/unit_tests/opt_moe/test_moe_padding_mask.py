# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Upstream's padding mask reaches every MoE layer: the padding tokens stay out
of the router's counts and the gate entropy."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.moe import MoE
from torchtitan.models.opt_moe import build_model_config

TOP_K = 2


def _config():
    return build_model_config(
        "gauge-d8-w256",
        parallelism=ParallelismConfig(),
        seq_len=64,
        n_layers=2,
        n_dense_layers=0,
        moe_hidden_dim=64,
        num_experts=8,
        top_k=TOP_K,
    )


class _CausalSDPA(nn.Module):
    def forward(self, q, k, v, **kwargs):
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True
        )
        return out.transpose(0, 1)


class _StubExperts(nn.Module):
    def forward(self, x_TD, *args, **kwargs):
        return torch.zeros_like(x_TD)


def test_forward_hands_the_padding_mask_to_every_moe_layer():
    with torch.device("meta"):
        model = _config().build()
    model.to_empty(device="cpu")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    moes = []
    for layer in model.layers.values():
        layer.attention.inner_attention = _CausalSDPA()
        assert isinstance(layer.feed_forward, MoE)
        layer.feed_forward.routed_experts = _StubExperts()
        moes.append(layer.feed_forward)
    model.train()
    tokens = torch.randint(0, 100, (32,))
    padding = torch.arange(32) >= 20
    with torch.no_grad():
        model(tokens, padding_mask=padding)
    for moe in moes:
        assert int(moe.router.tokens_per_expert_E.sum()) == 20 * TOP_K
        assert float(moe.router_entropy) > 0
