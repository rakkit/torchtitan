# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OPT MoE's MoE layer: upstream's ``MoE`` (its routers, routed and shared
experts, expert bias and auxiliary losses) plus the routing statistics the
DiSCO side logs and balances with (``optimizers/moe_balance.py``).

Legend: T tokens, D model dim, E routed experts, K top-k.
"""

from dataclasses import dataclass

import torch
import torch_remat as remat

from torchtitan.models.common.moe import MoE


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
