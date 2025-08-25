# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch


def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.square(torch.nn.functional.relu(x))


def approx_gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.nn.functional.sigmoid(1.702 * x)


ACTIVATION_FUNCTIONS = {
    "silu": torch.nn.functional.silu,
    "squared_relu": squared_relu,
    "elu": torch.nn.functional.elu,
    "relu": torch.nn.functional.relu,
    "selu": torch.nn.functional.selu,
    "gelu": torch.nn.functional.gelu,
    "approx_gelu": approx_gelu,
    "quick_gelu": quick_gelu,
    "sigmoid": torch.nn.functional.sigmoid,
}


def build_activation(activation_type: str) -> Callable:
    """
    Builds the specified activation function based on the
    activation_type.

    Args:
        activation_type (str): The type of activation layer to build.

    Returns:
        The built activation function.

    Raises:
        NotImplementedError: If an unknown `activation_type` is
            provided.
    """
    activation_type = activation_type.lower()  # Normalize to lowercase

    activation_fn = ACTIVATION_FUNCTIONS.get(activation_type)
    if activation_fn is not None:
        return activation_fn
    else:
        raise NotImplementedError(f"Unknown activation_type: '{activation_type}'")
