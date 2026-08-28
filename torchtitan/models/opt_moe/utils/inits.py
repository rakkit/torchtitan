# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


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

        """
        Impl-1 uses `distribute_tensor`, which explicitly does
        communication that distributes the tensor weights from `src=0`
        to the other ranks.
        This can make sure that weights across "dp_replicate" are
        initialized exactly the same.
        [?is it really? will the communication cause the weights to be
        different?]
        Impl-1 works both when the (global) generator doesn't match
        across ranks and when it does.
        ---
        Impl-2 uses `DTensor.from_local`, so that there will be no
        communication.
        But it maybe would cause the weights to be different across
        `dp_replicate` due to indeterministic stuff.
        Gonna use impl-2 for now to avoid the barrier.
        # TODO(JSC): We shall do a benchmark later to see which one is
        #            better.
        Impl-2 does not works when the (global) generator doesn't match
        across ranks.
        """

        # ##########################################
        # Implementation-1: Use `distribute_tensor`
        # tensor_data = distribute_tensor(
        #     temp_tensor, placements=tensor.placements, device_mesh=tensor.device_mesh,
        # )
        # tensor.copy_(tensor_data)

        # ##########################################
        # Implementation-2: Use `DTensor.from_local`
        chunk = tensor.__create_chunk_list__()[0]  # ChunkStorageMetadata
        offs, sizes = chunk.offsets, chunk.sizes  # torch.Size objects
        for dim, (o, s) in enumerate(zip(offs, sizes)):
            temp_tensor = temp_tensor.narrow(dim, o, s)

        tensor.data.to_local().copy_(temp_tensor)

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
        std=1.0,
        generator=generator,
    )
    if scale_type is None:
        scale = 1.0
    elif scale_type == "input":
        scale = math.sqrt(tensor.shape[norm_axis])
    elif scale_type == "output":
        scale = 1.0 / math.sqrt(tensor.shape[norm_axis])
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    with torch.no_grad():
        norm = torch.sqrt(tensor.pow(2).sum(dim=norm_axis, keepdim=True))

        tensor.mul_(scale * std / (norm + eps))


def sign_(tensor: torch.Tensor, generator: torch.Generator | None = None):
    with torch.no_grad():
        nn.init.normal_(tensor, generator=generator)
        tensor.sign_()
        return tensor


def scion_sign_(
    tensor: torch.Tensor,
    norm_axis: int = 1,
    generator: torch.Generator | None = None,
):
    assert tensor.ndim == 2, "Tensor for scion_sign_ init must have 2 dimensions"
    with torch.no_grad():
        nn.init.normal_(tensor, generator=generator)
        tensor.sign_()
        tensor.div_(tensor.shape[norm_axis])
        return tensor


INIT_FN_MAP = {
    "trunc_normal": nn.init.trunc_normal_,
    "normal": nn.init.normal_,
    "sign": _wrap_ignore_mean_std(sign_),
    "zeros": _wrap_ignore_generator(_wrap_ignore_mean_std(nn.init.zeros_)),
    "orthogonal": _wrap_orthogonal(orthogonal_),
    "scaled_orthogonal": _wrap_orthogonal(scaled_orthogonal_),
    "image_orthogonal": _wrap_orthogonal(image_orthogonal_),
    "scion_normal": scion_normal_,
    "scion_sign": _wrap_ignore_mean_std(scion_sign_),
    "scion_normal_input": functools.partial(
        scion_normal_,
        scale_type="input",
        norm_axis=1,
    ),
    "scion_normal_output": functools.partial(
        scion_normal_,
        scale_type="output",
        norm_axis=1,
    ),
}
INIT_FN_TYPES = list(INIT_FN_MAP.keys())


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

    init_fn = INIT_FN_MAP.get(init_fn_type)
    if init_fn is not None:
        return init_fn
    else:
        raise NotImplementedError(f"Unknown `init_fn_type`: '{init_fn_type}'")


def parse_depth_init(depth_init):
    if depth_init in ["true", "relative_depth"]:
        depth_init = "relative_depth"
    elif depth_init in ["false", "total_depth"]:
        depth_init = "total_depth"
    elif depth_init in ["none", "null", "identity"]:
        depth_init = None
    else:
        raise ValueError(f"Unknown depth_init: {depth_init}")
    return depth_init


def setup_depth_init(depth_init: str | None, layer_id: int, n_layers: int):
    residual_div_attn = 1.0
    residual_div_ffn = 1.0
    match depth_init:
        case "relative_depth":
            residual_div_attn = (2 * (layer_id + 1)) ** 0.5
            residual_div_ffn = (2 * (layer_id + 2)) ** 0.5
        case "total_depth":
            residual_div_attn = (2 * n_layers) ** 0.5
            residual_div_ffn = (2 * n_layers) ** 0.5
        case None:
            residual_div_attn = 1.0
            residual_div_ffn = 1.0
        case _:
            raise ValueError(f"Invalid depth_init: {depth_init}")
    return residual_div_attn, residual_div_ffn


def setup_residual_scale(residual_scale: str, n_layers: int):
    block_scale, identity_scale = 1.0, 1.0
    match residual_scale:
        case "depth_scale":
            total_depth = 2 * n_layers
            block_scale = 1 / total_depth
            identity_scale = (total_depth - 1) / total_depth
        case "complete_p":
            total_depth = 2 * n_layers
            block_scale = 1 / total_depth
            identity_scale = 1.0
        case "identity":
            block_scale = 1.0
            identity_scale = 1.0
        case _:
            raise ValueError(f"Invalid residual_scale: {residual_scale}")
    return block_scale, identity_scale
