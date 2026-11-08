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
from torchtitan.ops.moe_rows import gather_sum, scatter_rows
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
        routed_experts_TK: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upstream's ``MoE.forward`` plus the gate entropy."""
        (
            routed_x_TD,
            routed_padding_mask_T,
        ) = self._maybe_shard_routed_branch_inputs_across_tp(x_TD, padding_mask_T)
        router_kwargs = (
            {}
            if routed_experts_TK is None
            else {
                "routed_experts_TK": self._maybe_shard_routed_experts_across_tp(
                    routed_experts_TK
                )
            }
        )
        topk_scores_TK, topk_expert_ids_TK, routing_map_TE = self.router(
            routed_x_TD,
            self.expert_bias_E,
            padding_mask_T=routed_padding_mask_T,
            aux_loss_denominator=aux_loss_denominator,
            **router_kwargs,
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
    """From `GatherTokenDispatcher.dispatch` for its `combine`: each (token, k)
    slot's routed row and the router's scores in slot order."""

    slot_row_N: torch.Tensor  # noqa: N815
    topk_scores_TK: torch.Tensor  # noqa: N815


class GatherTokenDispatcher(AllToAllTokenDispatcher):
    """At EP 1, dispatch and combine as row kernels (`torchtitan/ops/moe_rows.py`):
    the dispatch writes each token once per routed expert in expert-major order,
    the combine gathers each token's K expert outputs back and sums them
    score-weighted in fp32 (rounded once; upstream's scatter-add adds them into a
    bf16 accumulator one by one, so this is not bitwise equal to it). Every
    backward is a row gather or write (no sort-based index_put, no scatter-add).

    ``pad_multiple`` > 1: every expert's rows start at a multiple of it in a static
    ``T * K + E * (pad_multiple - 1)`` rows (rounded up), zero rows between (the
    layout DeepGEMM's grouped GEMM needs); the returned counts are the padded
    ones. EP > 1: the all-to-all dispatcher unchanged (unpadded)."""

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
        num_slots, align = perm_N.numel(), self.pad_multiple
        row_N = torch.arange(num_slots, device=perm_N.device)  # sorted slot -> row
        num_rows = num_slots
        if align > 1:
            padded_counts_E = (counts_E + align - 1) // align * align
            pad_E = padded_counts_E - counts_E
            row_N = row_N + (torch.cumsum(pad_E, 0) - pad_E)[expert_N[perm_N]]
            num_rows = num_slots + self.num_experts * (align - 1)
            num_rows = (num_rows + align - 1) // align * align
            counts_E = padded_counts_E
        slot_row_N = torch.empty_like(perm_N)
        slot_row_N[perm_N] = row_N
        routed_input_RD = scatter_rows(x_TD, slot_row_N, num_rows)
        metadata = GatherDispatchMetadata(
            slot_row_N=slot_row_N, topk_scores_TK=topk_scores_TK
        )
        return routed_input_RD, counts_E, metadata

    def combine(self, routed_output_RD, metadata, x_TD):
        if not isinstance(metadata, GatherDispatchMetadata):
            return super().combine(routed_output_RD, metadata, x_TD)
        # The combine saves the expert outputs for its backward (score gradients).
        remat.recompute_needs_tensor(routed_output_RD)
        out_TD = gather_sum(
            routed_output_RD, metadata.slot_row_N, metadata.topk_scores_TK, self.top_k
        )
        return out_TD.to(x_TD.dtype)


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
