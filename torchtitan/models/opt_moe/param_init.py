# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
import re

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

        # A sharded parameter: every rank draws the full matrix (the same
        # generator state on every rank) and keeps its own shard, no communication.
        full = torch.empty(tensor.shape, device=tensor.device)
        torch.nn.init.orthogonal_(full, gain=gain, generator=generator)
        return _copy_full_into(tensor, full)


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


def build_init_fn(init_fn_type: str):
    """The initializer ``INIT_FN_MAP[init_fn_type]`` (case-insensitive), called
    as ``fn(tensor, mean=..., std=..., generator=...)``."""
    init_fn = INIT_FN_MAP.get(init_fn_type.lower())
    if init_fn is None:
        raise NotImplementedError(
            f"Unknown init_fn_type {init_fn_type!r}; available: {', '.join(INIT_FN_MAP)}"
        )
    return init_fn


def setup_depth_init(depth_init: str, layer_id: int, n_layers: int):
    """The depth-init divisors (attention, FFN) of layer ``layer_id`` of
    ``n_layers``: ``total_depth_{R,N}`` (sqrt of the residual-branch count R =
    2 N, or of N, for every layer), ``relative_depth_{R,N}`` (by the layer's
    depth) or ``identity`` (1, no depth scaling)."""
    match depth_init:
        case "relative_depth_R":
            return (2 * (layer_id + 1)) ** 0.5, (2 * (layer_id + 2)) ** 0.5
        case "relative_depth_N":
            return (layer_id + 1) ** 0.5, (layer_id + 1) ** 0.5
        case "total_depth_R":
            return (2 * n_layers) ** 0.5, (2 * n_layers) ** 0.5
        case "total_depth_N":
            return n_layers**0.5, n_layers**0.5
        case "identity":
            return 1.0, 1.0
        case _:
            raise ValueError(
                "depth_init must be total_depth_R | total_depth_N | relative_depth_R | "
                f"relative_depth_N | identity, got {depth_init!r}"
            )


def setup_residual_scale(residual_scale: str, n_layers: int):
    block_scale, identity_scale = 1.0, 1.0

    # N = number of Transformer layers
    # R = 2 * N = number of attention/FFN residual sub-blocks
    N = n_layers
    R = 2 * n_layers

    match residual_scale:
        case "identity":
            block_scale = 1.0
            identity_scale = 1.0

        case "depth_scale_R":
            block_scale = 1 / R
            identity_scale = (R - 1) / R

        case "depth_scale_N":
            block_scale = 1 / N
            identity_scale = (N - 1) / N

        case "complete_p_R":
            block_scale = 1 / R
            identity_scale = 1.0

        case "complete_p_N":
            block_scale = 1 / N
            identity_scale = 1.0

        case _ if re.fullmatch(r"complete_p_base[1-9][0-9]*", residual_scale):
            # CompleteP alpha=1 (arXiv 2505.01618 Table 1), base depth K: K / L
            block_scale = int(residual_scale[len("complete_p_base") :]) / N
            identity_scale = 1.0

        case _:
            raise ValueError(f"Invalid residual_scale: {residual_scale}")

    return block_scale, identity_scale


CONV_INIT_FFT = 4096  # zero-padded FFT grid of the random_opnorm normalisation


def _copy_full_into(tensor: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
    """Copy a full-shape tensor into a plain or DTensor parameter (its local chunk)."""
    with torch.no_grad():
        if not isinstance(tensor.data, DTensor):
            return tensor.copy_(full.to(tensor.device, tensor.dtype))
        chunk = tensor.__create_chunk_list__()[0]
        for dim, (o, s) in enumerate(zip(chunk.offsets, chunk.sizes)):
            full = full.narrow(dim, o, s)
        tensor.data.to_local().copy_(full.to(tensor.device, tensor.dtype))
        return tensor


def _conv_init(tensor: torch.Tensor, kind: str, gain: float) -> torch.Tensor:
    """Fill a depthwise conv weight [C, 1, K]. "delay" / "random_opnorm": every
    channel's operator norm max_w |H_c(w)| is `gain`. "torch_default": `gain` x the
    `nn.Conv1d` default draw, kaiming_uniform_(a=sqrt(5)) = U(-1/sqrt(K), 1/sqrt(K))
    (the same CPU draw as the module itself). The full tensor is drawn on CPU from
    the global generator, so the values do not depend on the sharding."""
    if tensor.ndim != 3 or tensor.shape[1] != 1:
        raise ValueError(
            f"conv init needs a depthwise weight [C, 1, K], got {tuple(tensor.shape)}"
        )
    channels, _, k = tensor.shape
    if kind == "torch_default":
        bound = 1.0 / math.sqrt(k)  # fan_in = K for a depthwise weight
        return _copy_full_into(
            tensor, torch.empty(channels, 1, k).uniform_(-bound, bound) * gain
        )
    if (
        kind == "delay"
    ):  # one tap uniform in 0..K-1, sign uniform: |H| = gain at every w
        tap = torch.randint(0, k, (channels,))
        sign = torch.randint(0, 2, (channels,)) * 2 - 1
        w = torch.zeros(channels, k, dtype=torch.float64)
        w[torch.arange(channels), tap] = sign.double() * gain
    else:  # random_opnorm: gaussian direction scaled to max_w |H_c| = gain on the FFT grid
        w = torch.randn(channels, k, dtype=torch.float64)
        w *= (gain / torch.fft.rfft(w, n=CONV_INIT_FFT).abs().amax(dim=-1))[:, None]
    return _copy_full_into(tensor, w.unsqueeze(1))


def conv_init_fn(kind: str, gain: float) -> "functools.partial":
    """Single-argument initializer of a depthwise conv weight for `param_init`.

    kind: "delay" (one +-gain tap per channel), "random_opnorm" (random filter with
    operator norm gain) or "torch_default" (gain x the nn.Conv1d default draw). gain > 0."""
    if kind not in ("delay", "random_opnorm", "torch_default"):
        raise ValueError(
            f"conv_init_fn_type must be delay | random_opnorm | torch_default, got {kind!r}"
        )
    if not gain > 0:
        raise ValueError(f"conv_init_gain must be > 0, got {gain!r}")
    return functools.partial(_conv_init, kind=kind, gain=float(gain))


def _init_experts_different(init_fn, w, std: float, slot: int, layer_id: int) -> None:
    """Draw every expert of a stacked ``[E, ...]`` weight from its own seed, a
    function of (layer, slot, expert): the experts start with equal norms, not
    equal weights, whatever the sharding over dim 0."""
    total_experts = w.shape[0]
    if isinstance(w, DTensor):  # sharded on dim 0 (whole experts per rank)
        local_tensor = w.to_local()
        offset = w.__create_chunk_list__()[0].offsets[0]
    else:
        local_tensor, offset = w, 0
    for e in range(local_tensor.shape[0]):
        # 3 slots per expert: w1, w2, w3
        seed = layer_id * (3 * total_experts) + slot * total_experts + e + offset
        rng = None
        if local_tensor.device.type != "meta":
            rng = torch.Generator(device=local_tensor.device)
            rng.manual_seed(seed)
        # In place through the alias: to_local() shares the DTensor's storage.
        init_fn(local_tensor[e], mean=0.0, std=std, generator=rng)


def make_expert_param_init(
    init_fn_type: str, std: float, *, slot: int, layer_id: int
) -> "functools.partial":
    """Single-argument initializer of a stacked expert weight (``slot``: 0 w1,
    1 w2, 2 w3) for ``param_init``: every expert drawn with its own seed."""
    return functools.partial(
        _init_experts_different,
        build_init_fn(init_fn_type),
        std=std,
        slot=slot,
        layer_id=layer_id,
    )


def make_param_init(
    init_fn_type: str,
    std: float,
    residual_div: float = 1.0,
) -> "functools.partial":
    """A single-argument initializer for upstream's ``param_init`` dict
    (``Module._init_param`` calls ``param_init[name](param)``).

    Args:
        init_fn_type: Key into ``INIT_FN_MAP`` (e.g. ``"scaled_orthogonal"``).
        std: Base standard deviation before depth scaling.
        residual_div: Depth-init divisor from ``setup_depth_init``. Applied to
            output projections so deeper residual branches start smaller.
    """
    init_fn = build_init_fn(init_fn_type)
    return functools.partial(init_fn, mean=0.0, std=std / residual_div)
