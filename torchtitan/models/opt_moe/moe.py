# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OPT MoE's MoE layer: upstream's ``MoE`` (its routers, routed and shared
experts, expert bias and auxiliary losses) plus the routing statistics the
DiSCO side logs and balances with (``optimizers/moe_balance.py``).

Also opt_moe's alternative EP-1 token dispatch (row gathers instead of the
scatter-add) and router top-k (one Triton pass per row), selected by the flavor
arguments ``token_dispatch`` / ``router_topk`` (defaults: upstream's).

Legend: T tokens, D model dim, E routed experts, K top-k, N = T * K routed rows,
M padded routed rows.
"""

from dataclasses import dataclass

import torch
import torch_remat as remat

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_mesh_group
from torchtitan.models.common.moe import (
    MoE,
    QuantileBalancedTopKRouter,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.ops.row_topk import row_topk


class OptMoE(MoE):
    """Upstream's MoE with opt_moe's routing statistics.

    ``padding_mask_T`` (True: not counted, upstream's padding mask) keeps
    tokens out of the router's counts and auxiliary losses, as upstream's. The
    experts still process every token (upstream's routed experts ignore the
    mask).

    The statistics, summed over the step's forwards (not on remat replay) and
    over the counted tokens: the router's ``tokens_per_expert_E`` (upstream's)
    and the gate entropy ``router_entropy`` of each token's top-k weights
    (before ``route_scale``); the balance hook divides the entropy by the
    counted tokens. ``tokens_per_expert_cumul`` is the hook's running count.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        bias_update_norm_factor: str = "sign"
        """The expert-bias update of ``optimizers/moe_balance.py``."""
        track_router_metrics: bool = True
        """Accumulate the gate entropy of the top-k weights."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.bias_update_norm_factor = config.bias_update_norm_factor
        self.track_router_metrics = config.track_router_metrics
        self.register_buffer(
            "router_entropy", torch.zeros(1, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "tokens_per_expert_cumul",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        device = buffer_device or self.router.tokens_per_expert_E.device
        with torch.device(device):
            self.router_entropy = torch.zeros(1, dtype=torch.float32)
            self.tokens_per_expert_cumul = torch.zeros(
                self.num_experts, dtype=torch.float32
            )

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upstream's ``MoE.forward`` plus the gate entropy."""
        (
            routed_x_TD,
            routed_padding_mask_T,
        ) = self._maybe_shard_routed_branch_inputs_across_tp(x_TD, padding_mask_T)
        topk_scores_TK, topk_expert_ids_TK, routing_map_TE = self.router(
            routed_x_TD,
            self.expert_bias_E,
            padding_mask_T=routed_padding_mask_T,
            aux_loss_denominator=aux_loss_denominator,
        )
        if self.track_router_metrics and self.training and not remat.is_recomputing():
            with torch.no_grad():
                # The renormalized top-k weights: a distribution before
                # route_scale, bounded by ln(K). Summed over the tokens the
                # router counts, as tokens_per_expert_E.
                w_TK = topk_scores_TK.detach() / self.router.route_scale
                entropy_T = -(w_TK * w_TK.clamp_min(1e-20).log()).sum(dim=-1)
                if routed_padding_mask_T is not None:
                    entropy_T = entropy_T.masked_fill(routed_padding_mask_T, 0.0)
                self.router_entropy.add_(entropy_T.sum())
        out_TD = self.routed_experts(
            routed_x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
            padding_mask_T=routed_padding_mask_T,
        )
        out_TD = self._maybe_zero_fill_routed_output_to_tp_partial(out_TD)
        if self.shared_experts is not None:
            shared_TD = self.shared_experts(x_TD)
            # Saved, as upstream's MoE: it saves nothing for backward.
            out_TD = remat.region(
                torch.add, self.remat_region_name("shared_add"), recompute=False
            )(out_TD, shared_TD)
        return self._maybe_all_reduce_moe_output_across_tp(out_TD)


def _ep_group() -> torch.distributed.ProcessGroup | None:
    with maybe_set_sparse_mesh():
        return spmd_mesh_group(MeshAxisName.EP)


@dataclass(frozen=True, kw_only=True)
class GatherDispatchMetadata:
    """From `GatherTokenDispatcher.dispatch` for its `combine`. ``slot_row_N``:
    each (token, k) slot's routed row; ``perm_N`` (unpadded layout only): each
    routed row's slot."""

    slot_row_N: torch.Tensor  # noqa: N815
    perm_N: torch.Tensor | None  # noqa: N815
    topk_scores_TK: torch.Tensor  # noqa: N815


class _DispatchRows(torch.autograd.Function):
    """``x_TD[perm_N // K]`` (unpadded): every token once per routed expert,
    expert-major. Backward: each token's K row gradients gathered and summed."""

    @staticmethod
    def forward(ctx, x_TD, perm_N, slot_row_N, top_k):
        ctx.save_for_backward(slot_row_N)
        ctx.top_k = top_k
        return x_TD[perm_N // top_k]

    @staticmethod
    def backward(ctx, grad_ND):
        (slot_row_N,) = ctx.saved_tensors
        grad_TKD = grad_ND[slot_row_N].view(-1, ctx.top_k, grad_ND.shape[-1])
        return grad_TKD.sum(dim=1), None, None, None


class _DispatchPaddedRows(torch.autograd.Function):
    """``[x_TD; 0][source_M]`` (padded): every token once per routed expert at its
    padded expert-major row, zero rows elsewhere. Backward as `_DispatchRows`."""

    @staticmethod
    def forward(ctx, x_TD, source_M, slot_row_N, top_k):
        ctx.save_for_backward(slot_row_N)
        ctx.top_k = top_k
        x_ext = torch.cat([x_TD, x_TD.new_zeros(1, x_TD.shape[-1])])
        return x_ext[source_M]

    @staticmethod
    def backward(ctx, grad_MD):
        (slot_row_N,) = ctx.saved_tensors
        grad_TKD = grad_MD[slot_row_N].view(-1, ctx.top_k, grad_MD.shape[-1])
        return grad_TKD.sum(dim=1), None, None, None


class _GatherRows(torch.autograd.Function):
    """``src_MD[index_N]`` for distinct rows. Backward: the gradient gathered by
    ``inverse_M`` when the gather is a permutation, else written back into zeros
    (an index_put without accumulate)."""

    @staticmethod
    def forward(ctx, src_MD, index_N, inverse_M):
        ctx.save_for_backward(index_N, inverse_M)
        ctx.num_rows = src_MD.shape[0]
        return src_MD[index_N]

    @staticmethod
    def backward(ctx, grad_ND):
        index_N, inverse_M = ctx.saved_tensors
        if inverse_M.numel():
            return grad_ND[inverse_M], None, None
        grad_MD = grad_ND.new_zeros(ctx.num_rows, grad_ND.shape[-1])
        grad_MD[index_N] = grad_ND
        return grad_MD, None, None


class GatherTokenDispatcher(AllToAllTokenDispatcher):
    """At EP 1, dispatch and combine as row gathers: the dispatch gathers each
    token once per routed expert in expert-major order, the combine gathers the
    expert outputs back to [T, K, D] and sums the K score-weighted rows in fp32.
    No backward is a sort-based index_put or a scatter_add. The combine rounds
    once per token instead of once per added expert output, so it is not
    bitwise equal to the scatter-add dispatcher (closer to the fp64 result).

    ``pad_multiple`` > 1: every expert's rows start at a multiple of it in a
    static ``T * K + E * (pad_multiple - 1)`` rows (rounded up), zero rows
    between (the layout DeepGEMM's grouped GEMM needs); the returned counts are
    the padded ones. EP > 1: the all-to-all dispatcher unchanged (unpadded)."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pad_multiple: int = 1

    def __init__(self, config: Config):
        super().__init__(config)
        self.pad_multiple = config.pad_multiple

    def dispatch(self, x_TD, topk_scores_TK, topk_expert_ids_TK, routing_map_TE):
        if _ep_group() is not None:
            return super().dispatch(
                x_TD, topk_scores_TK, topk_expert_ids_TK, routing_map_TE
            )
        counts_E = routing_map_TE.sum(dim=0)
        expert_N = topk_expert_ids_TK.reshape(-1)
        perm_N = torch.argsort(expert_N, stable=True)
        num_rows = perm_N.numel()
        arange_N = torch.arange(num_rows, device=perm_N.device)
        slot_row_N = torch.empty_like(perm_N)
        align = self.pad_multiple
        if align == 1:
            slot_row_N[perm_N] = arange_N
            routed_input = _DispatchRows.apply(x_TD, perm_N, slot_row_N, self.top_k)
            metadata = GatherDispatchMetadata(
                slot_row_N=slot_row_N, perm_N=perm_N, topk_scores_TK=topk_scores_TK
            )
            return routed_input, counts_E, metadata
        padded_counts_E = (counts_E + align - 1) // align * align
        pad_E = padded_counts_E - counts_E
        row_N = arange_N + (torch.cumsum(pad_E, 0) - pad_E)[expert_N[perm_N]]
        total = num_rows + self.num_experts * (align - 1)
        total = (total + align - 1) // align * align
        source_M = torch.full(
            (total,), x_TD.shape[0], dtype=perm_N.dtype, device=perm_N.device
        )
        source_M[row_N] = perm_N // self.top_k
        slot_row_N[perm_N] = row_N
        routed_input = _DispatchPaddedRows.apply(x_TD, source_M, slot_row_N, self.top_k)
        metadata = GatherDispatchMetadata(
            slot_row_N=slot_row_N, perm_N=None, topk_scores_TK=topk_scores_TK
        )
        return routed_input, padded_counts_E, metadata

    def combine(self, routed_output_RD, metadata, x_TD):
        if not isinstance(metadata, GatherDispatchMetadata):
            return super().combine(routed_output_RD, metadata, x_TD)
        # The combine reads the expert outputs with bare ops.
        remat.recompute_needs_tensor(routed_output_RD)
        inverse = (
            metadata.perm_N
            if metadata.perm_N is not None
            else metadata.slot_row_N.new_empty(0)
        )
        rows_TKD = _GatherRows.apply(routed_output_RD, metadata.slot_row_N, inverse)
        rows_TKD = rows_TKD.view(x_TD.shape[0], self.top_k, -1)
        scores_TK1 = metadata.topk_scores_TK.to(torch.float32).unsqueeze(-1)
        return (rows_TKD * scores_TK1).sum(dim=1).to(x_TD.dtype)


class RowTopKRouter(TokenChoiceTopKRouter):
    """`TokenChoiceTopKRouter` with the expert choice by `row_topk` (one Triton
    pass per row; equal scores: lowest expert id first)."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        pass

    def _select_experts(self, scores_TE, expert_bias_E=None, **router_kwargs):
        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        return row_topk(scores_for_choice_TE, self.top_k)[1]


class RowTopKQuantileBalancedRouter(QuantileBalancedTopKRouter):
    """`QuantileBalancedTopKRouter` with the top-(k + 1) by `row_topk`."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantileBalancedTopKRouter.Config):
        pass

    def _select_experts(
        self, scores_TE, expert_bias_E=None, padding_mask_T=None, **router_kwargs
    ):
        if expert_bias_E is None:
            raise ValueError("Quantile balancing requires an expert bias.")
        topk_plus_one_scores, topk_plus_one_expert_ids = row_topk(
            scores_TE + expert_bias_E, self.top_k + 1
        )
        if self.training and not remat.is_recomputing():
            self.quantile_balancer.observe(
                scores_TE,
                topk_plus_one_scores[:, self.top_k :],
                expert_bias_E,
                padding_mask_T,
            )
        return topk_plus_one_expert_ids[:, : self.top_k].contiguous()
