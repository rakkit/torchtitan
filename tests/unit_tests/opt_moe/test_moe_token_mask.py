# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The MoE's padding mask, from the batch to every MoE layer: upstream's padding
mask by default; with ``enable_token_mask_for_moe`` every token whose label is
ignored."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.models.opt_moe import build_model_config

TOP_K = 2


def _config(token_mask: bool):
    return build_model_config(
        "gauge-d8-w256",
        parallelism=ParallelismConfig(),
        seq_len=64,
        n_layers=2,
        n_dense_layers=0,
        moe=dict(hidden_dim=64, num_experts=8, top_k=TOP_K),
        enable_token_mask_for_moe=token_mask,
    )


@pytest.mark.parametrize("token_mask", [False, True])
def test_preprocess_sets_the_moe_padding_mask(token_mask):
    with torch.device("meta"):
        model = _config(token_mask).build()
    labels = torch.tensor([5, IGNORE_INDEX, 7, IGNORE_INDEX, 9, 3])
    upstream_padding = torch.tensor([False, False, False, False, False, True])
    upstream = (
        torch.zeros(6, dtype=torch.long),
        labels,
        {"padding_mask": upstream_padding},
        {},
    )
    with patch.object(Decoder, "preprocess_inputs", return_value=upstream):
        _, _, model_kwargs, _ = model.preprocess_inputs(
            {}, parallelism_context=None, parallelism=ParallelismConfig()
        )
    expected = labels == IGNORE_INDEX if token_mask else upstream_padding
    assert torch.equal(model_kwargs["padding_mask"], expected)
    assert "loss_mask" not in model_kwargs


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
        model = _config(False).build()
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
