# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The short convolution of KDA / Gated DeltaNet and of Canon / Inkling on a token-flat stream."""

import torch
import torch.nn.functional as F


def causal_short_conv(
    x_TC: torch.Tensor,
    weight_C1K: torch.Tensor,
    positions_T: torch.Tensor | None,
    conv_format: str,
) -> torch.Tensor:
    """Depthwise causal conv (`nn.Conv1d` orientation: `w[..., -1]` multiplies
    the current token), accumulated in fp32. Tap `j` reaches `j` tokens back only
    where `positions >= j`: positions restart at every packed document, so a
    document never sees the previous one. `conv_format`: "kda" returns
    `silu(conv(x))` (it replaces x); "canon" returns `conv(x) + x` (Canon /
    Inkling: residual, no activation)."""
    if conv_format not in ("kda", "canon"):
        raise ValueError(f"conv_format must be kda | canon, got {conv_format!r}")
    k = weight_C1K.shape[-1]
    x = x_TC.float()
    w = weight_C1K.squeeze(1).float()
    if positions_T is None:
        positions_T = torch.arange(x.shape[0], device=x.device)
    y = x * w[:, k - 1]
    for j in range(1, k):
        back = F.pad(x[:-j], (0, 0, j, 0))
        y = y + torch.where((positions_T >= j).unsqueeze(-1), back, 0.0) * w[:, k - 1 - j]
    y = F.silu(y) if conv_format == "kda" else y + x
    return y.to(x_TC.dtype)
