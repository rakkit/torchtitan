# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

# from ..infra.expert_parallel import expert_parallel

from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

from torchtitan.models.inits import build_init_fn
from torchtitan.models.norms import build_norm

try:
    from grouped_gemm.ops import gmm as grouped_gemm
except ImportError:

    def mock_grouped_gemm(x, w, m_sizes):
        out = x.new_zeros(x.shape[0], w.shape[2])
        # G = m_sizes.shape[0]
        shift = 0
        for g, n_tokens in enumerate(m_sizes):
            input_tokens = x[shift : shift + n_tokens]
            out[shift : shift + n_tokens] = input_tokens @ w[g]
            shift += n_tokens
        return out

    grouped_gemm = mock_grouped_gemm


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network.
    See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim_in (int): Input dimension.
        dim_hidden (int): SwiGLU hidden dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        swiglu (bool): Whether to use gated linear unit. Default is True.
        activation (nn.Module): Activation function to use. Default is F.silu.
    """

    # when ep is enabled, these are set in parallelize.py
    ep_enable = False
    expert_per_rank = -1
    ep_size = -1

    def __init__(
        self,
        layer_id: int,
        *,
        dim_in: int,
        dim_hidden: int,
        num_experts: int = 1,
        activation: Callable = F.silu,
        moe_init_all_experts_same: bool = False,
        norm_everywhere: bool = False,
        norm_type: str | None = None,
        norm_eps: float | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_experts = num_experts
        self.expert_per_rank = num_experts

        self.w1 = nn.Parameter(torch.empty(num_experts, dim_in, dim_hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim_hidden, dim_in))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim_in, dim_hidden))

        self.act_fn = activation

        self.init_all_experts_same = moe_init_all_experts_same

        self.expert_per_rank = num_experts
        self.ep_size = 1

        if norm_everywhere:
            assert (
                norm_type is not None
            ), "`norm_type` needs to be passed when `norm_everywhere=True`"
            assert (
                norm_eps is not None
            ), "`norm_eps` needs to be passed when `norm_everywhere=True`"
            self.out_norm = build_norm(norm_type, dim=dim_hidden, eps=norm_eps)
        else:
            self.out_norm = nn.Identity()

    def __repr__(self):
        model_str = f"GroupedExperts(dim_in={self.dim_in}, hidden={self.dim_hidden},\n"
        # model_str += (
        #     f"\tnum_experts={self.num_experts}, local_experts={self.expert_per_rank}, "
        # )
        # model_str += f"ep_size={self.ep_size}, \n"
        model_str += f"\tup_proj={self.w1.shape}, \n"
        model_str += f"\tgate_proj={self.w3.shape}, \n"
        model_str += f"\tdown_proj={self.w2.shape}, \n"
        model_str += f"\tout_norm={self.out_norm}, \n"
        model_str += ")"
        return model_str

    def forward(
        self,
        x: torch.Tensor,
        m_sizes: torch.LongTensor,
    ) -> torch.Tensor:

        if not self.ep_enable:
            m_sizes = m_sizes.long()
            h = grouped_gemm(x, self.w1, m_sizes)
            h = self.act_fn(h) * grouped_gemm(x, self.w3, m_sizes)
            h = grouped_gemm(self.out_norm(h), self.w2, m_sizes)
            return h

        # ###### BELOW IS THE EP REGIME #######
        # TODO(JSC): For now, EP does not work with compile
        ALIGN_SIZE_M = 16
        # ALIGN_SIZE_M = 1

        if ALIGN_SIZE_M > 1:
            max_len = x.shape[0] + self.expert_per_rank * ALIGN_SIZE_M
        else:
            max_len = x.shape[0]

        with torch.no_grad():
            (permuted_indices, m_sizes, _,) = generate_permute_indices(  # offsets,
                m_sizes,
                self.expert_per_rank,
                self.ep_size,
                max_len,
                ALIGN_SIZE_M,
                use_cpu=True,
            )

        if ALIGN_SIZE_M > 1:
            x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
        input_shape = x.shape
        x = x[permuted_indices, :]
        m_sizes = m_sizes.long()

        out = grouped_gemm(x, self.w1.to_local(), m_sizes)
        out = self.act_fn(out) * grouped_gemm(x, self.w3.to_local(), m_sizes)
        out = grouped_gemm(self.out_norm(out), self.w2.to_local(), m_sizes)

        out_unpermuted = out.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = out
        out = out_unpermuted[:-1]
        return out

    def init_weights(
        self,
        init_std: float,
        residual_div: float,
        init_gate_as_residual: bool,
        init_fn_type: str,
    ):

        init_fn = build_init_fn(init_fn_type)
        gate_init_std = init_std / residual_div if init_gate_as_residual else init_std

        if self.init_all_experts_same:
            expert_init_fn = init_all_experts_same

        else:
            expert_init_fn = init_all_experts_different

        expert_init_fn(init_fn, self.w1.data, init_std)
        expert_init_fn(init_fn, self.w3.data, gate_init_std)
        expert_init_fn(init_fn, self.w2.data, init_std / residual_div)

        if not isinstance(self.out_norm, nn.Identity):
            self.out_norm.reset_parameters()


def init_all_experts_same(init_fn, w, init_std):
    """
    Notice that the weights are in the shape of [G, D_in, D_out]
    But we expected the weights to be [D_out, D_in] for `init_fn`
    """
    if isinstance(w, torch.distributed.tensor.DTensor):
        local_tensor = w.to_local()
    else:
        local_tensor = w

    init_fn(local_tensor[0].transpose(0, 1), mean=0.0, std=init_std)
    for e in range(1, local_tensor.shape[0]):
        local_tensor[e].data.copy_(local_tensor[0].data)

    if isinstance(w, torch.distributed.tensor.DTensor):
        w.to_local().copy_(local_tensor)
    else:
        w.copy_(local_tensor)


def init_all_experts_different(init_fn, w, init_std):
    if isinstance(w, torch.distributed.tensor.DTensor):
        local_tensor = w.to_local()
    else:
        local_tensor = w

    for e in range(local_tensor.shape[0]):
        rank = dist.get_rank()
        rand_offset = torch.randint(0, 10000, size=(), device="cpu").item()
        seed = rank * 50000 + e * 100 + rand_offset
        # for each rank, layer, expert, [w1, w2, w3], we need to set a different seed
        if w.device.type == "meta":
            rng = None
        else:
            rng = torch.Generator(device=w.device)
            rng.manual_seed(seed)

        init_fn(local_tensor[e].transpose(0, 1), mean=0.0, std=init_std, generator=rng)

    if isinstance(w, torch.distributed.tensor.DTensor):
        w.to_local().copy_(local_tensor)
    else:
        w.copy_(local_tensor)
