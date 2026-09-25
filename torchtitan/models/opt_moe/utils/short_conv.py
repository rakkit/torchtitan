# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The short convolution of KDA / Gated DeltaNet on a token-flat stream."""

import torch
import torch.nn.functional as F


def causal_short_conv(
    x_TC: torch.Tensor, weight_C1K: torch.Tensor, positions_T: torch.Tensor | None
) -> torch.Tensor:
    """`silu` of a depthwise causal conv (`nn.Conv1d` orientation: `w[..., -1]`
    multiplies the current token), accumulated in fp32. Tap `j` reaches `j`
    tokens back only where `positions >= j`: positions restart at every packed
    document, so a document never sees the previous one."""
    k = weight_C1K.shape[-1]
    x = x_TC.float()
    w = weight_C1K.squeeze(1).float()
    if positions_T is None:
        positions_T = torch.arange(x.shape[0], device=x.device)
    y = x * w[:, k - 1]
    for j in range(1, k):
        back = F.pad(x[:-j], (0, 0, j, 0))
        y = y + torch.where((positions_T >= j).unsqueeze(-1), back, 0.0) * w[:, k - 1 - j]
    return F.silu(y).to(x_TC.dtype)
