# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU


class MidNormActivation(BinaryActivationFn):
    """`inner(gate, up)`, then a parameter-free norm over the hidden dim: the
    FFN / expert mid-norm, kept for testing it. An activation holds no
    parameters, so only the `np_*` norms are available."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        inner: BinaryActivationFn.Config = field(default_factory=SwiGLU.Config)
        norm_type: str = "np_rmsnorm"
        eps: float = 1e-30

        def __post_init__(self) -> None:
            if self.norm_type not in ("np_rmsnorm", "np_layernorm"):
                raise ValueError(
                    "the mid-norm needs a parameter-free norm (np_rmsnorm or "
                    f"np_layernorm), got {self.norm_type!r}"
                )

    def __init__(self, config: Config) -> None:
        self.inner = config.inner.build()
        self.norm_type, self.eps = config.norm_type, config.eps

    def __call__(
        self, gate: torch.Tensor, up: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        h = self.inner(gate, up, **kwargs)
        if self.norm_type == "np_rmsnorm":
            return F.rms_norm(h, (h.shape[-1],), eps=self.eps)
        return F.layer_norm(h, (h.shape[-1],), eps=self.eps)


def ffn_activation(
    activation_fn: BinaryActivationFn.Config,
    *,
    mid_norm: bool,
    norm_type: str,
    norm_eps: float,
) -> BinaryActivationFn.Config:
    """What an FFN or expert block applies between `w13` and `w2`:
    `activation_fn`, wrapped in the mid-norm when it is on."""
    if not mid_norm:
        return activation_fn
    return MidNormActivation.Config(
        inner=activation_fn, norm_type=norm_type, eps=norm_eps
    )
