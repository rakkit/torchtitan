# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, DTensor

INIT_FN_TYPES = ["trunc_normal", "normal", "orthogonal", "scion_normal"]


# Deliberately throw away `mean` and `std` arguments.
def _wrap_ignore_mean_std(fn):
    @functools.wraps(fn)
    def wrapped_fn(tensor, mean=None, std=None, *args, **kwargs):
        return fn(tensor, *args, **kwargs)

    return wrapped_fn


# Deliberately throw away the `generator` argument.
def _wrap_ignore_generator(fn):
    @functools.wraps(fn)
    def wrapped_fn(tensor, *args, generator=None, **kwargs):
        return fn(tensor, *args, **kwargs)

    return wrapped_fn


def orthogonal_(param, gain: float = 1.0, generator: torch.Generator | None = None):
    with torch.no_grad():
        if not isinstance(param.data, DTensor):
            return nn.init.orthogonal_(param, gain=gain, generator=generator)

        temp_tensor = torch.empty(param.shape, device=param.device)  # full shape
        torch.nn.init.orthogonal_(temp_tensor, gain=gain, generator=generator)

        params_data = distribute_tensor(
            temp_tensor,
            placements=param.placements,
            device_mesh=param.device_mesh,
        )

        # Copy values to original `DTensor`
        param.copy_(params_data)
        return param


def scion_normal_(
    tensor,
    mean: float = 0.0,
    std: float = 1.0,
    norm_axis: int = 1,
    eps: float = 1e-12,
    generator: torch.Generator | None = None,
):
    nn.init.normal_(
        tensor,
        mean=mean,
        std=std,
        generator=generator,
    )
    with torch.no_grad():
        divisor = torch.rsqrt(tensor.pow(2).sum(axis=norm_axis, keepdim=True) + eps)
        tensor.mul_(divisor)


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
        return _wrap_ignore_mean_std(orthogonal_)
    elif init_fn_type == "scion_normal":
        return scion_normal_
    else:
        raise NotImplementedError(f"Unknown `init_fn_type`: '{init_fn_type}'")
