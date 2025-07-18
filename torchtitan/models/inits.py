import functools
import math
from typing import Optional

import torch
from torch.distributed.tensor import DTensor, distribute_tensor
import torch.nn as nn


def zeros_(param, **kwargs):
    return nn.init.zeros_(param)


def orthogonal_(
    param,
    std: float = 1.0,  # this is the "GAIN"
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    with torch.no_grad():
        if not isinstance(param.data, DTensor):
            return nn.init.orthogonal_(param, gain=std, generator=generator)

        temp_tensor = torch.empty(
            param.shape, device=param.device, dtype=torch.float32
        )  # float32 is used to avoid precision issue
        torch.nn.init.orthogonal_(temp_tensor, gain=std, generator=generator)

        # ##########################################
        # Implementation-1: Use `distribute_tensor`
        # params_data = distribute_tensor(
        #     temp_tensor,
        #     placements=param.placements,
        #     device_mesh=param.device_mesh,
        # )
        # torch.distributed.barrier()
        # param.copy_(params_data.to(param.dtype))

        # ##########################################
        # Implementation-2: Use `DTensor.from_local`
        chunk = param.__create_chunk_list__()[0]  # ChunkStorageMetadata
        offs, sizes = chunk.offsets, chunk.sizes  # torch.Size objects
        for dim, (o, s) in enumerate(zip(offs, sizes)):
            temp_tensor = temp_tensor.narrow(dim, o, s)

        param.data.to_local().copy_(temp_tensor)

        """
        impl-1 use `distribute_tensor`, its explicitly do communication 
        that distribute the tensor weights (from src=rank0) to other ranks.
        This can make sure that weights across "dp_replicate" are initialized exactly the same. 
        [?is it really? will the communication cause the weights to be different?] -> lets use fp32 here
        It find to be useful to add a barrier after the communication.
        ---
        impl-2 use `DTensor.from_local`, that there will be no communication.
        But it maybe would cause the weights to be different across "dp_replicate" due to indeterministic stuff.

        Gonna use impl-2 for now to avoid the barrier.
        # TODO(JSC): We shall do a benchmark later to see which one is better.
        """
        return param


def scaled_orthogonal_(
    param, std: float = 1.0, generator: Optional[torch.Generator] = None, **kwargs
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
            param.ndim == 2
        ), "Fan in and fan out can not be computed for tensor with other than 2 dimensions"
        fan_out, fan_in = param.shape
        scale = math.sqrt(fan_out / fan_in)
        std *= scale

    return orthogonal_(param, std, generator)


def image_orthogonal_(
    param,
    std: float = 1.0,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    """Image domain initialization as specified in the Scion paper."""
    with torch.no_grad():
        assert (
            param.ndim == 2
        ), "Fan in and fan out can not be computed for tensor with other than 2 dimensions"
        fan_out, fan_in = param.shape
        scale = max(math.sqrt(fan_out / fan_in), 1.0)
        std *= scale

    return orthogonal_(param, std, generator)


def scion_normal_(
    tensor,
    mean: float = 0.0,
    std: float = 1.0,
    norm_axis: int = 1,
    eps: float = 1e-12,
    scale_type: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs,
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


INIT_FN_MAP = {
    "trunc_normal": nn.init.trunc_normal_,
    "normal": nn.init.normal_,
    "zeros": zeros_,
    "orthogonal": orthogonal_,
    "scaled_orthogonal": scaled_orthogonal_,
    "image_orthogonal": image_orthogonal_,
    "scion_normal": scion_normal_,
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

    if init_fn_type in INIT_FN_MAP:
        return INIT_FN_MAP[init_fn_type]
    else:
        raise NotImplementedError(f"Unknown `init_fn_type`: '{init_fn_type}'")
