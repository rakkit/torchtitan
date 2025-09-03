# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this file implements FP8BitNet b1.58 https://arxiv.org/abs/2402.17764
# a reference implementation is available at
# https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import nn, Tensor
from torch.distributed._tensor import DTensor
from torch.utils._triton import has_triton

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import TorchAOBaseTensor


aten = torch.ops.aten


class FP8BitNetTrainingLinearWeight(TorchAOBaseTensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor, precomputed_scale: Optional[Tensor] = None):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor, precomputed_scale: Optional[Tensor] = None):
        self._data = data
        self._precomputed_scale = precomputed_scale

    def __tensor_flatten__(self):
        if self._precomputed_scale is not None:
            return ["_data", "_precomputed_scale"], []
        else:
            return ["_data"], []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            tensor_data_dict["_data"],
            tensor_data_dict.get("_precomputed_scale", None),
            *tensor_attributes,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self._data})"

    # adapated from FP8 implementation of WeightWithDynamicFloat8CastTensor
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        out = func(
            *pytree.tree_map_only(cls, lambda x: x._data, args),
            **pytree.tree_map_only(cls, lambda x: x._data, kwargs),
        )

        # NOTE: _precomputed_scale does not propagate through any ops
        if func is aten.copy_.default:
            # return original object
            return args[0]
        elif func in {
            aten.t.default,
            aten.detach.default,
            aten.empty_like.default,
            aten.new_zeros.default,
            aten.slice.Tensor,
            aten.view.default,
            aten.as_strided.default,
            aten._to_copy.default,
            aten._pin_memory.default,
            aten.split.Tensor,
            aten.clone.default,
        }:
            # return new wrapped object
            return pytree.tree_map_only(Tensor, lambda x: cls(x), out)
        else:
            # return new unwrapped object
            return out

    # FSDP all-gather extension v1
    def fsdp_pre_all_gather(self, mesh):
        # quantize and pack into 2-bit to save comm bandwidth
        if self._precomputed_scale is not None:
            scale = self._precomputed_scale

        else:
            scale = get_fp8_bitnet_scale(self._data)
            dist.all_reduce(scale, op=dist.ReduceOp.AVG)

        # NOTE: scale is in FP32
        data_fp8 = quantize_fp8_bitnet_weight(self._data, scale)
        data_f2 = _pack_f2_in_f8(data_fp8)
        return (data_f2,), (scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Tensor] = None,
    ):
        (data_f2,) = all_gather_outputs
        (scale,) = metadata
        scale = scale.to(param_dtype)
        if out is not None:
            assert isinstance(out, FP8BitNetPacked2bitLinearWeight)
            out.scale = scale
            return
        return FP8BitNetPacked2bitLinearWeight(data_f2, scale), all_gather_outputs


@FP8BitNetTrainingLinearWeight.implements(F.linear)
def _(func, types, args, kwargs):
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_gpu_dtype()
        args = tuple(x.to(dtype) if x is not None else x for x in args)
    return _FP8BitNetTrainingLinear.apply(*args, **kwargs)


def get_fp8_bitnet_scale(x: Tensor):
    "Tensor-wise abs-mean. Always return FP32."
    return x.float().abs().mean()


def quantize_fp8_bitnet_weight(w: Tensor, scale: Tensor, eps: float = 1e-5) -> Tensor:
    w = w.float() / scale.clip(eps)
    w = w.round().clip(-1, 1).to(torch.int8)
    return w


def _quantize_rowwise_to_fp8(x: Tensor, eps: float = 1e-5):
    """Quantize a 2D tensor row-wise into float8 using INT8-like calibration.

    Returns (x_fp8, scale_a), where:
      - x_fp8: torch.float8_e4m3fn with values approximately in [-128, 127]
      - scale_a: torch.float32 of shape (M, 1), contiguous
    """
    x2d = x.view(-1, x.shape[-1]).float()
    scale = (x2d.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(eps).to(torch.float32).contiguous()
    q = (x2d / scale).round().clamp(-128, 127)
    x_fp8 = q.to(torch.float8_e4m3fn)
    return x_fp8, scale


def _quantize_tensorwise_bitnet_to_fp8(w: Tensor, eps: float = 1e-5):
    """BitNet-style tensorwise quantization to float8: q in {-1, 0, 1}.

    Returns (w_fp8, s_w32), where s_w32 is FP32 scalar.
    """
    s_w = w.float().abs().mean().clamp_min(eps)
    q = (w.float() / s_w).round().clamp(-1, 1)
    w_fp8 = q.to(torch.float8_e4m3fn)
    return w_fp8, s_w.to(torch.float32)


def _quantize_colwise_to_fp8(x: Tensor, eps: float = 1e-5):
    """Quantize a 2D tensor column-wise into float8 using INT8-like calibration.

    Returns (x_fp8, scale_b), where:
      - x_fp8: torch.float8_e4m3fn
      - scale_b: torch.float32 of shape (1, N), contiguous
    """
    x2d = x.view(-1, x.shape[-1]).float()
    scale = (x2d.abs().amax(dim=0, keepdim=True) / 127.0).clamp_min(eps).to(torch.float32).contiguous()
    q = (x2d / scale).round().clamp(-128, 127)
    x_fp8 = q.to(torch.float8_e4m3fn)
    return x_fp8, scale


def _as_column_major(B: Tensor) -> Tensor:
    """Return a 2D tensor with the same shape/values but column-major strides.
    If already column-major (stride(0) == 1) return as-is; otherwise use T.contiguous().T trick.
    """
    assert B.dim() == 2
    if B.stride(0) == 1:
        return B
    return B.t().contiguous().t()


@torch.no_grad()
def precompute_fp8_bitnet_scale_for_fsdp(module: nn.Module):
    """Calculate scale for all FP8BitNetTrainingLinearWeight parameters.
    This should be run after the optimizer step. It performs a single all-reduce for all
    parameters to reduce overhead.
    """
    fp8_bitnet_params = [
        p
        for p in module.parameters()
        if isinstance(p, DTensor)
        and isinstance(p._local_tensor, FP8BitNetTrainingLinearWeight)
    ]
    if len(fp8_bitnet_params) == 0:
        return

    # NOTE: use torch.compile to save memory and increase speed?
    fp8_bitnet_scales = [
        get_fp8_bitnet_scale(x) for x in fp8_bitnet_params
    ]  # local absmean
    fp8_bitnet_scales = torch.stack(fp8_bitnet_scales)
    fp8_bitnet_scales = fp8_bitnet_scales.full_tensor()  # global absmean

    for i, p in enumerate(fp8_bitnet_params):
        p._local_tensor._precomputed_scale = fp8_bitnet_scales[i]


class _FP8BitNetTrainingLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: FP8BitNetTrainingLinearWeight,
        bias: Optional[Tensor] = None,
    ):
        # reshape input to 2D [M, K]
        batch_dims = input.shape[:-1]
        input_2d = input.view(-1, weight.shape[1]).to(torch.bfloat16)

        # Quantize activations row-wise to FP8 + get (M,1) FP32 scale
        a_fp8, scale_a = _quantize_rowwise_to_fp8(input_2d)

        # Quantize BitNet weights tensorwise to FP8 {-1,0,1} + scalar FP32 scale
        w_fp8, s_w32 = _quantize_tensorwise_bitnet_to_fp8(weight._data)

        # B for matmul is [K, N] = [in, out]
        B = _as_column_major(w_fp8.T)
        scale_b = s_w32.expand(1, B.shape[1]).contiguous()

        # Save tensors for backward
        ctx.save_for_backward(input_2d, w_fp8, s_w32)
        ctx.has_bias = bias is not None

        out2d = torch._scaled_mm(
            a_fp8.contiguous(),
            B,
            out_dtype=input_2d.dtype,
            scale_a=scale_a,
            scale_b=scale_b,
        )

        out = out2d.view(*batch_dims, weight.shape[0])
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_2d, w_fp8, s_w32 = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        batch_dims = grad_output.shape[:-1]
        grad_output_2d = grad_output.view(-1, w_fp8.shape[0]).to(input_2d.dtype)

        # 1) grad_input = grad_output @ W
        if ctx.needs_input_grad[0]:
            go_fp8, go_scale = _quantize_rowwise_to_fp8(grad_output_2d)
            B = _as_column_major(w_fp8)  # [out, in] column-major view
            scale_b = s_w32.expand(1, B.shape[1]).contiguous()
            grad_input_2d = torch._scaled_mm(
                go_fp8.contiguous(),
                B,
                out_dtype=grad_output_2d.dtype,
                scale_a=go_scale,
                scale_b=scale_b,
            )
            grad_input = grad_input_2d.view(*batch_dims, w_fp8.shape[1])

        # 2) grad_weight = grad_output.T @ input
        if ctx.needs_input_grad[1]:
            go_t = grad_output_2d.T
            go_t_fp8, go_t_scale = _quantize_rowwise_to_fp8(go_t)

            # Quantize input column-wise to FP8 for B
            input_fp8, input_col_scale = _quantize_colwise_to_fp8(input_2d)
            grad_weight = torch._scaled_mm(
                go_t_fp8.contiguous(),
                _as_column_major(input_fp8),
                out_dtype=input_2d.dtype,
                scale_a=go_t_scale,
                scale_b=input_col_scale,
            )

        if ctx.needs_input_grad[2] and ctx.has_bias:
            grad_bias = grad_output_2d.sum(0)

        return grad_input, grad_weight, grad_bias


class FP8BitNetTrainingConfig(AOBaseConfig):
    pass


# for bc
fp8_bitnet_training = FP8BitNetTrainingConfig


@register_quantize_module_handler(FP8BitNetTrainingConfig)
def _fp8_bitnet_training_transform(
    module: torch.nn.Module,
    config: FP8BitNetTrainingConfig,
) -> torch.nn.Module:
    import types

    new_weight = FP8BitNetTrainingLinearWeight(module.weight)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=True)
    module._get_name = types.MethodType(lambda self: "FP8BitNetTrainingLinear", module)
    return module


def _pack_f2_in_f8(x: Tensor):
    # perform packing: [xxxx xxaa, xxxx xxxbb, xxxx xxcc, xxxx xxdd] -> [aabb ccdd]
    # for each value, xxxx can be either all 0s or all 1s because these are signed numbers.
    # thus, we have to mask out the 2 least significant bits (right-most) before bit-shift.
    # e.g. 1111 1111 (value=-1) -> 0000 0011 -> 0011 0000

    x0 = (
        x[:, ::4] << 6
    )  # don't need to mask this number because we shift it to the left-most
    x1 = (x[:, 1::4] & 0b11) << 4
    x2 = (x[:, 2::4] & 0b11) << 2
    x3 = x[:, 3::4] & 0b11
    return x0 | x1 | x2 | x3


def _unpack_f2_in_f8(x: Tensor):
    # NOTE: this is signed integer, so left-shift then right-shift will perform sign extension correctly
    # e.g. aa10bbcc -> 10bbcc00 -> 11111110
    return torch.stack([x >> 6, x << 2 >> 6, x << 4 >> 6, x << 6 >> 6], dim=-1).view(
        x.shape[0], -1
    )


# currently this class mainly serves as a container for quantized FSDP2 all-gather,
# so only a minimal set of ops are implemented. this can be extended for inference.
class FP8BitNetPacked2bitLinearWeight(TorchAOBaseTensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data: Tensor, scale: Tensor):
        M, N = int_data.shape
        shape = (M, N * 4)
        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=scale.dtype,
            device=scale.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data: Tensor, scale: Tensor):
        assert int_data.dtype is torch.int8
        assert scale.shape == ()
        self.int_data = int_data
        self.scale = scale

    def __tensor_flatten__(self):
        return ["int_data", "scale"], []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            tensor_data_dict["int_data"], tensor_data_dict["scale"], *tensor_attributes
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.dequantize()})"

    def dequantize(self, out_dtype=None):
        out = _unpack_f2_in_f8(self.int_data) * self.scale
        if out_dtype is not None:
            out = out.to(out_dtype)
        return out


@FP8BitNetPacked2bitLinearWeight.implements(F.linear)
def _(func, types, args, kwargs):
    return _FP8BitNetPacked2bitLinear.apply(*args, **kwargs)


@FP8BitNetPacked2bitLinearWeight.implements(
    [
        aten.detach.default,
        aten.clone.default,
    ]
)
def _(func, types, args, kwargs):
    return FP8BitNetPacked2bitLinearWeight(
        func(args[0].int_data, *args[1:], **kwargs),
        func(args[0].scale, *args[1:], **kwargs),
    )


# this is a workaround to make it work with FSDP2.
# end-users should not call this op directly.
@FP8BitNetPacked2bitLinearWeight.implements(aten.as_strided.default)
def _(func, types, args, kwargs):
    return FP8BitNetPacked2bitLinearWeight(args[0].int_data, args[0].scale)


class _FP8BitNetPacked2bitLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: FP8BitNetPacked2bitLinearWeight,
        bias: Optional[Tensor] = None,
    ):
        """
        Forward pass using FP8 GEMM.
        Quantizes input row-wise and uses the pre-quantized 2-bit weight.
        """
        # Reshape input to 2D for matrix multiplication
        batch_dims = input.shape[:-1]
        input_2d = input.view(-1, weight.shape[1])

        weight_f2, tensor_scale = weight.int_data, weight.scale
        # Unpack 2-bit weights to int8 values in {-1,0,1}, then cast to FP8
        weight_fp8 = _unpack_f2_in_f8(weight_f2).to(torch.float8_e4m3fn)

        # Save tensors for backward
        ctx.save_for_backward(input_2d, weight_fp8, tensor_scale)
        ctx.batch_dims = batch_dims
        ctx.has_bias = bias is not None

        # Quantize activations row-wise to FP8 and construct FP32 scales
        a_fp8, scale_a = _quantize_rowwise_to_fp8(input_2d)
        B = _as_column_major(weight_fp8.T)
        scale_b = tensor_scale.to(torch.float32).expand(1, B.shape[1]).contiguous()
        output2d = torch._scaled_mm(
            a_fp8.contiguous(),
            B,
            out_dtype=input_2d.dtype,
            scale_a=scale_a,
            scale_b=scale_b,
        )

        # Reshape output to original batch dimensions
        output = output2d.view(*batch_dims, weight.shape[0])

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using FP8 GEMM if available; fall back otherwise."""
        input_2d, weight_fp8, tensor_scale = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Reshape grad_output to 2D for matrix multiplication
        grad_output_2d = grad_output.view(-1, weight_fp8.shape[0]).to(input_2d.dtype)

        # grad_input = grad_output @ W
        if ctx.needs_input_grad[0]:
            go_fp8, go_scale = _quantize_rowwise_to_fp8(grad_output_2d)
            B = _as_column_major(weight_fp8)
            scale_b = tensor_scale.to(torch.float32).expand(1, B.shape[1]).contiguous()
            grad_input_2d = torch._scaled_mm(
                go_fp8.contiguous(),
                B,
                out_dtype=grad_output_2d.dtype,
                scale_a=go_scale,
                scale_b=scale_b,
            )
            grad_input = grad_input_2d.view(*ctx.batch_dims, weight_fp8.shape[1])

        # grad_weight = grad_output.T @ input
        if ctx.needs_input_grad[1]:
            go_t = grad_output_2d.T
            go_t_fp8, go_t_scale = _quantize_rowwise_to_fp8(go_t)
            input_fp8, input_col_scale = _quantize_colwise_to_fp8(input_2d)
            grad_weight = torch._scaled_mm(
                go_t_fp8.contiguous(),
                _as_column_major(input_fp8),
                out_dtype=input_2d.dtype,
                scale_a=go_t_scale,
                scale_b=input_col_scale,
            )

        # grad_bias
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)

        return grad_input, grad_weight, grad_bias
