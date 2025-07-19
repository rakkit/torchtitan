# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from typing import Union, Sequence, Tuple
import torch
import torch.nn.functional as F


class SingleScalerRMSNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps"]

    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        eps: float = 1e-6,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: Tuple[int, ...] = tuple(normalized_shape)
        self.eps = eps

        self.ssnorm_scale = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.rms_norm(input, self.normalized_shape, eps=self.eps)
        return self.ssnorm_scale * out

    def reset_parameters(self) -> None:
        nn.init.ones_(self.ssnorm_scale)

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


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

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    elif norm_type == "np_rmsnorm":
        return nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
    elif norm_type == "ss_rmsnorm":
        return SingleScalerRMSNorm(dim, eps=eps)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")
