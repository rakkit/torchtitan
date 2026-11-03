# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention sinks of opt_moe: a learned logit per head in the softmax denominator.

``apply_attention_sink_rescale`` is a copy of ``apply_attention_sink_rescale`` of
``torchtitan/models/gpt_oss/model.py`` (same math), kept here so that opt_moe does not
import gpt_oss code.
"""

import torch


def apply_attention_sink_rescale(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """Rescale attention output by the learned per-head sink term:
    ``out * sigmoid(lse - s)`` with ``s`` the head's sink logit, which equals the softmax
    over the keys plus one extra logit ``s`` that carries no value."""
    sinks = sinks.view(*([1] * (lse.ndim - 1)), -1)
    sink_scale = torch.sigmoid(lse - sinks).unsqueeze(-1)
    return out * sink_scale.to(out.dtype)
