# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.common.attention import (
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.qwen3_5.vision_encoder import FullSequenceVarlenInnerAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("lengths", [[256, 256, 256], [256, 128, 256]])
def test_matches_varlen(lengths):
    """Full sequences take the dense kernel, mixed lengths the varlen one; both
    equal VarlenInnerAttention within bf16 rounding, output and gradients."""
    torch.manual_seed(0)
    config = dict(sdpa_backends=["FLASH_ATTENTION"])
    full = FullSequenceVarlenInnerAttention.Config(**config).build()
    varlen = VarlenInnerAttention.Config(**config).build()
    cu_seq = torch.tensor([0, *lengths], device="cuda").cumsum(0).int()
    metadata = VarlenAttentionMetadata(
        cu_seq_q=cu_seq,
        cu_seq_k=cu_seq,
        max_q=max(lengths),
        max_k=max(lengths),
        window_size=(-1, -1),
    )
    q, k, v = (
        torch.randn(
            sum(lengths), 4, 72, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    grad_out = torch.randn_like(q)
    results = []
    for attention in (full, varlen):
        out = attention(q, k, v, attention_metadata=metadata)
        results.append((out, *torch.autograd.grad(out, (q, k, v), grad_out)))
    for got, expected in zip(*results):
        torch.testing.assert_close(got.float(), expected.float(), atol=0.05, rtol=0.05)
