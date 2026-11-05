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


class NormMoE(MoE):
    """Upstream's MoE with opt_moe's routing statistics.

    ``loss_mask`` (True: counted) keeps tokens out of the balance statistics:
    upstream's router counts and auxiliary losses take it as their padding mask
    (``~loss_mask``). Under SFT it also drops the prompt tokens, whose labels are
    ignored (``OPTMoEModel.preprocess_inputs``); the experts still process every
    token (upstream's routed experts ignore the padding mask).

    The statistics, per forward (not on remat replay): the router's
    ``tokens_per_expert_E`` (upstream's), the forward count ``acc_fwd_times`` and
    the summed gate entropy ``router_entropy`` of the top-k weights (before
    ``route_scale``); ``tokens_per_expert_cumul`` is the balance hook's running
    count.
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
            "acc_fwd_times", torch.zeros(1, dtype=torch.int64), persistent=False
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
            self.acc_fwd_times = torch.zeros(1, dtype=torch.int64)
            self.tokens_per_expert_cumul = torch.zeros(
                self.num_experts, dtype=torch.float32
            )

    def forward(
        self,
        x_TD: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        *,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upstream's ``MoE.forward`` with ``~loss_mask`` as the router's padding
        mask, plus the forward count and the gate entropy."""
        padding_mask_T = None if loss_mask is None else ~loss_mask.reshape(-1)
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
        if self.training and not remat.is_recomputing():
            with torch.no_grad():
                self.acc_fwd_times.add_(1)
                if self.track_router_metrics:
                    # The renormalized top-k weights: a distribution before
                    # route_scale, bounded by ln(K).
                    w_TK = topk_scores_TK.detach() / self.router.route_scale
                    self.router_entropy.add_(
                        -(w_TK * w_TK.clamp_min(1e-20).log()).sum(dim=-1).mean()
                    )
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
