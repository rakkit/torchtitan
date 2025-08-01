# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


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
