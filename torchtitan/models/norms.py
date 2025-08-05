# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleRMSNorm(nn.Module):
    """AKA SSNorm, from https://arxiv.org/abs/2506.19697."""

    __constants__ = ["normalized_shape", "eps"]
    normalized_shape: tuple[int, ...]
    eps: float | None

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-6,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps

        self.ssnorm_scale = nn.Parameter(torch.empty((1,), device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.rms_norm(input, self.normalized_shape, eps=self.eps)
        return self.ssnorm_scale * out

    def reset_parameters(self) -> None:
        nn.init.ones_(self.ssnorm_scale)

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


NORM_LAYERS = {
    "layernorm": lambda dim, eps: nn.LayerNorm(dim, eps=eps, bias=False),
    "np_layernorm": lambda dim, eps: nn.LayerNorm(
        dim,
        eps=eps,
        elementwise_affine=False,
        bias=False,
    ),
    "rmsnorm": lambda dim, eps: nn.RMSNorm(dim, eps=eps),
    "np_rmsnorm": lambda dim, eps: nn.RMSNorm(dim, eps=eps, elementwise_affine=False),
    "ss_rmsnorm": lambda dim, eps: SingleScaleRMSNorm(
        dim, eps=eps, dtype=torch.float32
    ),
}


def build_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Builds the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to build.
            Supported types: layernorm, np_layernorm, rmsnorm, np_rmsnorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The built normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    norm_layer_fn = NORM_LAYERS.get(norm_type)
    if norm_layer_fn is not None:
        return norm_layer_fn(dim, eps=eps)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")
