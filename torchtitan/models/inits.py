# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math

import torch
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, DTensor

INIT_FN_TYPES = [
    "trunc_normal",
    "normal",
    "orthogonal",
    "scaled_orthogonal",
    "scion_normal",
    "scion_normal_input",
    "scion_normal_output",
    "image_orthogonal",
]


# Deliberately throw away `mean` and `std` arguments.
def _wrap_ignore_mean_std(fn):
    @functools.wraps(fn)
    def wrapped_fn(
        tensor: torch.Tensor,
        mean: float | None = None,
        std: float | None = None,
        *args,
        **kwargs,
    ):
        return fn(tensor, *args, **kwargs)

    return wrapped_fn


# Deliberately throw away the `generator` argument.
def _wrap_ignore_generator(fn):
    @functools.wraps(fn)
    def wrapped_fn(
        tensor: torch.Tensor,
        *args,
        generator: torch.Generator | None = None,
        **kwargs,
    ):
        return fn(tensor, *args, **kwargs)

    return wrapped_fn


# Deliberately throw away the `mean` argument and pass the `std`
# argument as `gain` to the wrapped function.
def _wrap_orthogonal(fn):
    @functools.wraps(fn)
    def wrapped_fn(
        tensor: torch.Tensor,
        mean: float | None = None,
        std: float | None = 1,
        *args,
        **kwargs,
    ):
        return fn(tensor, gain=std, *args, **kwargs)

    return wrapped_fn


def orthogonal_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
):
    with torch.no_grad():
        if not isinstance(tensor.data, DTensor):
            return nn.init.orthogonal_(tensor, gain=gain, generator=generator)

        temp_tensor = torch.empty(tensor.shape, device=tensor.device)  # full shape
        torch.nn.init.orthogonal_(temp_tensor, gain=gain, generator=generator)

        tensor_data = distribute_tensor(
            temp_tensor,
            placements=tensor.placements,
            device_mesh=tensor.device_mesh,
        )

        # Copy values to original `DTensor`
        tensor.copy_(tensor_data)
        return tensor


def scaled_orthogonal_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
):
    """
    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_uniform_(w.T, ...)``.
    """
    with torch.no_grad():
        assert (
            tensor.ndim == 2
        ), "Fan in and fan out can not be computed for tensor with other than 2 dimensions"
        fan_out, fan_in = tensor.shape
        scale = math.sqrt(fan_out / fan_in)
        gain *= scale

    return orthogonal_(tensor, gain, generator)


def image_orthogonal_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
):
    """Image domain initialization as specified in the Scion paper."""
    with torch.no_grad():
        assert (
            tensor.ndim == 2
        ), "Fan in and fan out can not be computed for tensor with other than 2 dimensions"
        fan_out, fan_in = tensor.shape
        scale = max(math.sqrt(fan_out / fan_in), 1.0)
        gain *= scale

    return orthogonal_(tensor, gain, generator)


def scion_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    norm_axis: int = 1,
    eps: float = 1e-12,
    scale_type: str | None = None,
    generator: torch.Generator | None = None,
):
    assert tensor.ndim == 2, "Tensor for scion_normal_ init must have 2 dimensions"
    nn.init.normal_(
        tensor,
        mean=mean,
        std=std,
        generator=generator,
    )
    if scale_type is None:
        scale = 1.0
    elif scale_type == "input":
        scale = math.sqrt(tensor.shape[norm_axis])
    elif scale_type == "output":
        scale = 1 / math.sqrt(tensor.shape[norm_axis])
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    with torch.no_grad():
        scale = scale / (
            torch.sqrt(tensor.pow(2).sum(axis=norm_axis, keepdim=True)) + eps
        )
        tensor.mul_(scale)


def build_init_fn(init_fn_type: str):
    """
    Builds the specified initialization function based on `init_fn_type`.

    Args:
        init_fn_type (str): The type of normalization layer to build.
            Supported types: trunc_normal, normal

    Returns:
        The built initialization function.

    Raises:
        NotImplementedError: If an unknown `init_fn_type` is provided.
    """
    init_fn_type = init_fn_type.lower()  # Normalize to lowercase

    if init_fn_type == "trunc_normal":
        return nn.init.trunc_normal_
    elif init_fn_type == "normal":
        return nn.init.normal_
    elif init_fn_type == "zeros":
        return _wrap_ignore_generator(_wrap_ignore_mean_std(nn.init.zeros_))
    elif init_fn_type == "orthogonal":
        return _wrap_orthogonal(orthogonal_)
    elif init_fn_type == "scaled_orthogonal":
        return _wrap_orthogonal(scaled_orthogonal_)
    elif init_fn_type == "image_orthogonal":
        return _wrap_orthogonal(image_orthogonal_)
    elif init_fn_type == "scion_normal":
        return scion_normal_
    elif init_fn_type == "scion_normal_input":
        return functools.partial(
            scion_normal_,
            scale_type="input",
            norm_axis=1,
        )
    elif init_fn_type == "scion_normal_output":
        return functools.partial(
            scion_normal_,
            scale_type="output",
            norm_axis=1,
        )
    else:
        raise NotImplementedError(f"Unknown `init_fn_type`: '{init_fn_type}'")
