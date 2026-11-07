# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The routed experts' grouped GEMMs on DeepGEMM (flavor argument
``expert_gemm="deepgemm"``): its Hopper bf16 m-grouped GEMMs for the forward and
the input gradient, bitwise equal to torch._grouped_mm and 6-15% faster at 64k
tokens x top 8 x 256 experts in a one-GPU benchmark. The weight gradient stays
torch._grouped_mm: DeepGEMM's SM90 k-grouped bf16 GEMM needs the group sizes on
the host. Needs `GatherTokenDispatcher` with ``pad_multiple`` = `DEEPGEMM_ALIGN`
(every expert's rows start at a multiple of it; zero padding rows) and EP 1.

TODO: DeepGEMM is not a PyTorch library; opt_moe-only and opt-in.

Legend: M padded routed rows, E experts, I input features, O output features.
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.linear import GroupedLinear

DEEPGEMM_ALIGN = 128
"""DeepGEMM's M alignment for the contiguous grouped layout on SM90."""


def _deepgemm():
    import deep_gemm

    # Leave the configured SM carveout to communication, as torch's GEMMs do.
    carveout = torch._C._get_sm_carveout_experimental() or 0
    device = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count - carveout
    if deep_gemm.get_num_sms() != sms:
        deep_gemm.set_num_sms(sms)
    return deep_gemm


@torch.library.custom_op("opt_moe::deepgemm_grouped_nt", mutates_args=())
def _grouped_nt(
    a_MI: torch.Tensor, b_EOI: torch.Tensor, m_indices_M: torch.Tensor
) -> torch.Tensor:
    d_MO = a_MI.new_empty(a_MI.shape[0], b_EOI.shape[1])
    _deepgemm().m_grouped_bf16_gemm_nt_contiguous(a_MI, b_EOI, d_MO, m_indices_M)
    return d_MO


@_grouped_nt.register_fake
def _(a_MI, b_EOI, m_indices_M):
    return a_MI.new_empty(a_MI.shape[0], b_EOI.shape[1])


@torch.library.custom_op("opt_moe::deepgemm_grouped_nn", mutates_args=())
def _grouped_nn(
    a_MO: torch.Tensor, b_EOI: torch.Tensor, m_indices_M: torch.Tensor
) -> torch.Tensor:
    d_MI = a_MO.new_empty(a_MO.shape[0], b_EOI.shape[2])
    _deepgemm().m_grouped_bf16_gemm_nn_contiguous(a_MO, b_EOI, d_MI, m_indices_M)
    return d_MI


@_grouped_nn.register_fake
def _(a_MO, b_EOI, m_indices_M):
    return a_MO.new_empty(a_MO.shape[0], b_EOI.shape[2])


class _DeepGEMMGroupedMM(torch.autograd.Function):
    """``x_MI @ w_EOI[expert(row)].T`` on the padded expert-major layout whose
    group ends are ``offsets_E``; rows after the last end belong to no expert
    (DeepGEMM writes zeros there)."""

    @staticmethod
    def forward(ctx, x_MI, w_EOI, offsets_E):
        rows_M = torch.arange(x_MI.shape[0], device=x_MI.device, dtype=offsets_E.dtype)
        expert_M = torch.searchsorted(offsets_E, rows_M, right=True)
        m_indices_M = torch.where(expert_M < offsets_E.numel(), expert_M, -1).to(
            torch.int32
        )
        ctx.save_for_backward(x_MI, w_EOI, offsets_E, m_indices_M)
        return _grouped_nt(x_MI, w_EOI, m_indices_M)

    @staticmethod
    def backward(ctx, grad_MO):
        x_MI, w_EOI, offsets_E, m_indices_M = ctx.saved_tensors
        grad_MO = grad_MO.contiguous()
        grad_x_MI = _grouped_nn(grad_MO, w_EOI, m_indices_M)
        grad_w_EOI = torch._grouped_mm(grad_MO.t(), x_MI, offs=offsets_E)
        return grad_x_MI, grad_w_EOI, None


class DeepGEMMGroupedLinear(GroupedLinear):
    """GroupedLinear whose forward and input gradient run DeepGEMM's grouped GEMM
    (see the module docstring for the layout it needs)."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
        return _DeepGEMMGroupedMM.apply(
            input_RI.contiguous(), weight_EOI.bfloat16().contiguous(), offsets_E
        )
