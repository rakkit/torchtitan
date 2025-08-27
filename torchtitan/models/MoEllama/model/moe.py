# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.distributed.tensor
from torch import nn

from torchtitan.models.activations import build_activation
from torchtitan.models.inits import build_init_fn
from torchtitan.models.norms import build_norm

from .GroupedExperts import GroupedExperts


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation_type: str = "silu",
        norm_everywhere: bool = False,
        norm_type: Optional[str] = None,
        norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.act_fn = build_activation(activation_type)

        if norm_everywhere:
            assert (
                norm_type is not None
            ), "`norm_type` needs to be passed when `norm_everywhere=True`"
            assert (
                norm_eps is not None
            ), "`norm_eps` needs to be passed when `norm_everywhere=True`"
            self.out_norm = build_norm(
                norm_type,
                dim=hidden_dim,
                eps=norm_eps,
            )
        else:
            self.out_norm = nn.Identity()

    # @torch.compile(fullgraph=True)
    def forward(self, x):
        return self.w2(self.out_norm(self.act_fn(self.w1(x)) * self.w3(x)))

    def init_weights(
        self,
        init_std: float,
        residual_div: float,
        init_gate_as_residual: bool,
        init_fn_type: str,
        **kwargs,
    ):
        init_fn = build_init_fn(init_fn_type)
        init_fn(self.w1.weight, mean=0.0, std=init_std)
        init_fn(self.w2.weight, mean=0.0, std=init_std / residual_div)
        gate_init_std = init_std / residual_div if init_gate_as_residual else init_std
        init_fn(self.w3.weight, mean=0.0, std=gate_init_std)

        if not isinstance(self.out_norm, nn.Identity):
            self.out_norm.reset_parameters()


class TokenChoiceTopKRouter(nn.Module):
    force_gate_on_fp32: bool = False

    def __init__(
        self,
        hidden_size: int,
        experts=8,
        topk=2,
    ):
        super().__init__()

        self.topk = topk
        self.experts = experts
        self.gate = nn.Linear(hidden_size, experts, bias=False)
        # Expert embeddings

    def __repr__(self):
        return f"Gate(experts={self.experts}, topk={self.topk}"

    def init_weights(self, init_std: float, init_fn_type: str):
        # nn.init.xavier_uniform_(self.expert_embeddings)
        init_fn = build_init_fn(init_fn_type)
        init_fn(self.gate.weight, mean=0.0, std=init_std)

    # @torch.compile(fullgraph=True)
    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None, eps=1e-20
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        if self.force_gate_on_fp32:
            with torch.autocast(x.device, dtype=torch.float32):
                scores = self.gate(x)
        else:
            scores = self.gate(x)

        scores = torch.sigmoid(scores.to(torch.float32))

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.topk, dim=-1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.topk, dim=-1
            )

        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + eps)
        detached_top_scores = top_scores.detach()
        experts_entropy = (
            -(detached_top_scores * detached_top_scores.log()).sum(dim=-1).mean()
        )

        # group tokens together by expert indices from 0 to num_experts
        # and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.experts,
            min=0,
            max=self.experts,
        )

        return (
            top_scores,
            scores,
            selected_experts_indices,
            num_tokens_per_expert,
            experts_entropy,
        )


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size*seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class MoE(nn.Module):
    experts_parallel_enabled = False

    def __init__(
        self,
        layer_id: int,
        dim: int,
        multiple_of: int = 256,
        activation_type: str = "silu",
        n_shared_experts: int = 1,
        n_routed_experts: int = 8,
        activate_experts: int = 2,
        ffn_dim_multiplier: Optional[float] = None,
        match_dim_with_dense: bool = True,
        bias_update_speed: float = 0.001,  # Bias adjustment speed
        aux_loss_alpha: float = 0.001,  # Small weight for sequence-wise auxiliary loss
        bias_update_norm_factor: str = "sign",
        router_scaling_factor: float = 1.0,
        norm_everywhere: bool = False,
        norm_type: Optional[str] = None,
        norm_eps: Optional[float] = None,
    ):

        super().__init__()
        """
        match_dim_with_dense.

        Saying the dense model have the 'hidden_size' of 1024,
        Then "actual_dim * activate_experts = hidden_size" should be satisfied.
        """

        self.layer_id = layer_id

        if match_dim_with_dense:
            ratio = 1.0 / (activate_experts + n_shared_experts)
        else:
            ratio = 1.0

        hidden_dim = 2 * 4 * dim / 3
        if ffn_dim_multiplier is not None:
            hidden_dim = ffn_dim_multiplier * hidden_dim
        hidden_dim = int(hidden_dim * ratio)
        hidden_dim = hidden_dim - hidden_dim % multiple_of
        self.n_routed_experts = n_routed_experts
        self.topk = activate_experts
        self.n_shared_experts = n_shared_experts
        self.aux_loss_alpha = aux_loss_alpha  # Loss coefficient
        self.router_scaling_factor = router_scaling_factor
        self.bias_update_speed = bias_update_speed
        self.bias_update_norm_factor = bias_update_norm_factor

        # Use updated Gate with DeepSeekMoE-style routing and bias balancing
        self.router = TokenChoiceTopKRouter(
            hidden_size=dim, experts=n_routed_experts, topk=activate_experts
        )
        self.reorderer = TokenReorderer(
            num_experts=n_routed_experts, top_k=activate_experts
        )

        # Shared Experts (applies to all tokens)
        if n_shared_experts > 0:
            assert n_shared_experts == 1, "Only one shared expert is supported"
            """
            TODO(JSC):
            To make Muon work easier, we set the shared experts to either 0 or 1.
            So we dont use the GroupedExperts.
            """
            # self.shared_experts = GroupedExperts(
            #     dim_in=dim, dim_out=hidden_dim, num_experts=n_shared_experts
            # )
            self.shared_experts = FeedForward(
                dim,
                hidden_dim,
                activation_type=activation_type,
                norm_everywhere=norm_everywhere,
                norm_type=norm_type,
                norm_eps=norm_eps,
            )

        else:
            self.shared_experts = None

        # Routed Experts (only used when selected)
        assert n_routed_experts > 0, "n_routed_experts must be greater than 0"
        self.experts = GroupedExperts(
            layer_id,
            dim_in=dim,
            dim_hidden=hidden_dim,
            num_experts=n_routed_experts,
            activation_type=activation_type,
            norm_everywhere=norm_everywhere,
            norm_type=norm_type,
            norm_eps=norm_eps,
        )

        self.register_buffer(
            "expert_bias", torch.zeros(n_routed_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "tokens_per_expert", torch.zeros(n_routed_experts, dtype=torch.float32)
        )
        self.register_buffer("router_entropy", torch.zeros(1, dtype=torch.float32))
        self.acc_fwd_times = 0

    def init_weights(
        self,
        init_std: float,
        residual_div: float,
        init_gate_as_residual: bool,
        init_fn_type: str,
        router_init_fn_type: str,
    ):
        self.experts.init_weights(
            init_std,
            residual_div=residual_div,
            init_gate_as_residual=init_gate_as_residual,
            init_fn_type=init_fn_type,
        )
        if self.shared_experts is not None:
            self.shared_experts.init_weights(
                init_std,
                residual_div=residual_div,
                init_gate_as_residual=init_gate_as_residual,
                init_fn_type=init_fn_type,
            )
        self.router.init_weights(init_std, router_init_fn_type)

        self.expert_bias.zero_()
        self.tokens_per_expert.zero_()
        self.router_entropy.zero_()
        self.acc_fwd_times = 0

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bz, slen, dim = x.shape
        x = x.view(-1, dim)
        # TODO@JSC: check if we want to use FP32 remix

        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

        (
            top_scores,
            sigmoid_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            experts_entropy,
        ) = self.router(x, self.expert_bias)
        self.router_entropy.add_(experts_entropy)
        self.acc_fwd_times += 1
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        token_indices_experts_sorted = token_indices_experts_sorted.reshape(
            -1, 1
        ).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

        routed_output = self.experts(routed_input, num_tokens_per_expert)
        routed_output = (
            routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

        out = out.scatter_add(
            dim=0, index=token_indices_experts_sorted, src=routed_output
        )

        if self.training:
            aux_loss = MoE.sequence_wise_aux_loss(
                selected_experts_indices.long(),
                sigmoid_scores,
                bz,
                slen,
                self.n_routed_experts,
                self.topk,
                self.aux_loss_alpha,
            )
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        output = out.reshape(bz, slen, dim).to(x.dtype)
        return output, aux_loss

    @staticmethod
    @torch.compile(fullgraph=True)
    def sequence_wise_aux_loss(
        indices: torch.Tensor,  # Shape (B*S, K_val), type long
        scores: torch.Tensor,  # Shape (B*S, N_r_val), type float
        B: int,  # Batch size
        S: int,  # Sequence length
        total_experts: int,
        activate_experts: int,
        aux_loss_alpha: float,
        eps: float = 1e-15,
        force_flip: bool = False,  # If True, clamps f_i^(b) and P_i^(b) to [0,1]
    ) -> torch.Tensor:
        """
        Computes the Sequence-Wise Auxiliary Loss as described in DeepSeekMoE.

        Args:
            indices (torch.Tensor): Selected expert indices for each
                token, flattened across batch and sequence.
                Assumed to be of type long. Shape: (B*S, K_val).
            scores (torch.Tensor): Raw expert scores (before top-k
                selection and normalization) for each token and each
                expert. Shape: (B*S, N_r_val).
            B (int): Batch size.
            S (int): Sequence length.
            eps (float): Epsilon for numerical stability during score normalization.
            force_flip (bool): Whether to clamp intermediate f_i^(b) and P_i^(b) terms to [0,1].

        Returns:
            torch.Tensor: Computed sequence-wise auxiliary loss (scalar).

        """

        N_r = total_experts  # Total number of routed experts (N_r in formulas)
        K_val = activate_experts  # Number of experts selected per token (K in formulas)

        # Conditional returns based on your original code's logic
        # If topk == N_r, original code returned 0. This implies perfect load balancing by design.
        if K_val == N_r and N_r > 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # If not training, alpha is zero, or no experts are selected (K_val=0)
        if not (aux_loss_alpha > 0 and K_val > 0):
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # If batch or sequence length is zero, no tokens to process.
        if B == 0 or S == 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # Normalize scores: p_{t,i} (for token t, expert i)
        # scores shape: (B*S, N_r)
        norm_scores = scores / scores.sum(dim=-1, keepdim=True).clamp(
            min=eps
        )  # Shape: (B*S, N_r)

        # Reshape for sequence-wise processing
        # reshaped_indices shape: (B, S, K_val)
        # reshaped_norm_scores shape: (B, S, N_r)
        reshaped_indices = indices.view(B, S, K_val)
        reshaped_norm_scores = norm_scores.view(B, S, N_r)

        # 1. Compute f_i^(b) for all sequences b and experts i
        # f_i^(b) = (N_r / (S * K_val)) * sum_{s_idx, k_idx} 1{expert i is idx_{b,s_idx,k_idx}}

        # Create one-hot representation of selected expert indices
        # one_hot_indices[b, s_idx, k_idx, expert_i] = 1 if expert_i was the k_idx-th choice for token (b,s_idx)
        # Shape: (B, S, K_val, N_r)
        one_hot_indices = torch.nn.functional.one_hot(reshaped_indices, num_classes=N_r)

        # count_expert_selected_per_sequence[b, expert_i] = sum_{s_idx, k_idx} 1{...}
        # Sum over sequence tokens (S) and K_val choices
        count_expert_selected_per_sequence = one_hot_indices.sum(dim=[1, 2]).to(
            scores.dtype
        )  # Shape: (B, N_r)

        # Denominator S * K_val is non-zero due to earlier checks (S > 0, K_val > 0)
        f_i_b_all = (
            N_r / (S * K_val)
        ) * count_expert_selected_per_sequence  # Shape: (B, N_r)

        # 2. Compute P_i^(b) for all sequences b and experts i
        # P_i^(b) = (1/S) * sum_{s_idx, k_idx} p_{b,s_idx,expert_i} * 1{expert_i is idx_{b,s_idx,k_idx}}

        # Create a mask: mask_expert_selected[b,s_idx,k_idx,expert_i] is True if expert_i was selected
        # as k_idx-th choice for token (b,s_idx)
        arange_N_r = torch.arange(N_r, device=indices.device).view(
            1, 1, 1, N_r
        )  # Shape: (1,1,1,N_r) for broadcasting
        mask_expert_selected = (
            reshaped_indices.unsqueeze(-1) == arange_N_r
        )  # Shape: (B,S,K_val,N_r), boolean

        # Expand norm_scores for broadcasting: p_{b,s_idx,expert_i}
        reshaped_norm_scores_expanded = reshaped_norm_scores.unsqueeze(
            2
        )  # Shape: (B,S,1,N_r)

        # Element-wise product: p_{b,s_idx,expert_i} if expert_i was chosen, 0 otherwise
        # Product of float (norm_scores) and bool (mask) results in float.
        term_to_sum_for_P = (
            reshaped_norm_scores_expanded * mask_expert_selected
        )  # Shape: (B,S,K_val,N_r)

        # sum_val_P[b,expert_i] = sum_{s_idx,k_idx} (p_{b,s_idx,expert_i} * 1{...})
        # Sum over sequence tokens (S) and K_val choices
        sum_val_P = term_to_sum_for_P.sum(dim=[1, 2])  # Shape: (B, N_r)

        # Denominator S is non-zero due to earlier checks (S > 0)
        P_i_b_all = sum_val_P / S  # Shape: (B, N_r)

        if force_flip:
            f_i_b_all = torch.clamp(f_i_b_all, min=0, max=1)
            P_i_b_all = torch.clamp(P_i_b_all, min=0, max=1)

        # 3. Compute sum_i f_i^(b) * P_i^(b)
        loss_per_sequence = (f_i_b_all * P_i_b_all).sum(dim=1)

        # 4. Final auxiliary loss:
        aux_loss = loss_per_sequence.mean() * aux_loss_alpha

        return aux_loss
