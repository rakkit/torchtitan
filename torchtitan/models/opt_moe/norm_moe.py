# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OPT MoE, built on upstream's MoE primitives.

This used to be a full fork of an early upstream MoE -- its own experts,
router, reorderer and dispatch/combine. Upstream has since rebuilt that around
a token dispatcher, and picked up fixes the fork never got: DTensor-to-local
conversion for the expert weights, FP32 router gradients (not just FP32 forward),
expert counts derived from the routing map instead of a second ``torch.histc``,
and dispatchers for expert parallelism (all-to-all, DeepEP, HybridEP).

So the fork now subclasses upstream instead of duplicating it. What remains
here is only what upstream has no equivalent for:

  * ``norm_everywhere`` -- a norm between the gated activation and the down
    projection, inside the grouped expert GEMM.
  * a configurable activation (upstream hardcodes SiLU).
  * the sequence-wise / batch-wise auxiliary load-balance loss, whose gradient
    is injected at the layer via ``LoadBalanceLoss`` rather than returned.
    Upstream's MoE is aux-loss-free and steers balance purely through the
    expert bias.
  * per-expert initialization, so every expert draws different weights.
  * router entropy and max-violation metrics.
  * ``loss_mask``, so padding tokens do not count toward load balance.

Not supported: tensor and expert-tensor parallelism.

Expert parallelism routes through upstream's `make_token_dispatcher_config`
(`comm_backend` defaults to "standard" = all-to-all, which falls back to local
dispatch at EP=1). This file previously hardcoded `LocalTokenDispatcher`, which
does local dispatch ONLY, so EP>1 could not have worked however the mesh was
configured -- while this docstring claimed it needed "no code here".

EP is now live end to end: `model.py` sets `enable_ep=expert_parallel_degree > 1`,
`sharding.py` builds routed-expert and router EP plans from upstream's helpers, and
`parallelize.py` wires the EP and edp meshes. The only axis `sharding.py` refuses is
TENSOR parallelism.
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.config_utils import make_token_dispatcher_config
from torchtitan.models.common.linear import RouterGateLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import Identity
from torchtitan.protocols.module import Module

from .norm_ffn import FeedForward
from .utils.activations import build_activation
from .utils.inits import build_init_fn, make_param_init
from .utils.moe_utils import calc_gate_scaling_factor
from .utils.norms import build_norm_config


logger = logging.getLogger(__name__)


def make_seed_from_global(
    layer_id: int, slot: int, expert_id: int, total_experts: int
) -> int:
    # 3 slots per expert: w1, w2, w3
    return layer_id * (3 * total_experts) + slot * total_experts + expert_id


def init_all_experts_different(init_fn, w, init_std, slot, layer_id):
    # we should expect the experts to have same norms, rather than same weights
    assert layer_id is not None, "layer_id must be set "
    total_experts = w.shape[0]
    if isinstance(w, torch.distributed.tensor.DTensor):
        # we assume the DTensor is already sharded on dim 0
        local_tensor = w.to_local()
        shard_chunk = w.__create_chunk_list__()[0]
        offsets = shard_chunk.offsets[0]
    else:
        local_tensor = w
        offsets = 0

    for e in range(local_tensor.shape[0]):
        expert_id = e + offsets
        seed = make_seed_from_global(layer_id, slot, expert_id, total_experts)
        if local_tensor.device.type == "meta":
            rng = None
        else:
            rng = torch.Generator(device=local_tensor.device)
            rng.manual_seed(seed)

        init_fn(local_tensor[e], mean=0.0, std=init_std, generator=rng)

    # No copy back: `local_tensor` aliases `w`'s storage in both branches
    # above (`w.to_local()` shares it; the dense branch binds `w` itself), and
    # `init_fn` wrote through that alias in place. The copy was `w.copy_(w)`.


def saint_check_scaling_factor(
    num_routed_experts: int,
    activate_experts: int,
    scaling_factor: float | None = None,
    iter_times: int = 10_000,
) -> float:
    if scaling_factor is not None:
        return scaling_factor

    return calc_gate_scaling_factor(
        num_routed_experts,
        activate_experts,
        iter_times=iter_times,
    )


def sequence_wise_aux_loss(
    scores: torch.Tensor,  # Shape: (B*S, N) - Raw Sigmoid Affinities (s_{i,t})
    indices: torch.Tensor | None,  # Shape: (B*S, K) - Selected Expert Indices
    B: int,  # Batch size
    S: int,  # Sequence length (T in the paper)
    top_k: int,  # K_r
    loss_mask: torch.Tensor | None = None,  # Shape: (B*S,) True = real token
    positions: torch.Tensor | None = None,  # Shape: (T,), resets to 0 per doc
    doc_id: torch.Tensor | None = None,  # Shape: (T,), global document id
) -> torch.Tensor:
    """
    Computes Sequence-Wise Auxiliary Loss (DeepSeek-V3 Equations 17-20).

    Args:
        scores: The dense affinity scores (s_{i,t}) for routed experts.
                Should be the output of Sigmoid, shape (B*S, N).
        indices: The top-k selected expert indices. Shape (B*S, K).
    """
    # N_r: Total number of routed experts
    N = scores.size(-1)

    # DeepSeek-V3 Eq. 17-20 is a PER-SEQUENCE statistic. Token-flat batching
    # packs many documents into one stream, so scoring it as a single sequence
    # (B=1, S=T) silently turns this into a batch-wise loss under a
    # sequence-wise name, dropping exactly the within-document balancing
    # pressure this variant exists for. `positions` resets to 0 at each
    # document start (same convention as the attention document mask), so use
    # it to segment when it is available.
    if positions is not None:
        # Counts and per-document means accumulate in fp32. They are
        # integer-valued sums over up to T tokens, and bf16 saturates at 256 on
        # CUDA (CPU does not saturate, so a CPU-only check cannot catch this).
        # Measured with the accumulator back in scores.dtype, one document:
        # T=512 -> 2.06x wrong, T=1024 -> 4.00x, T=2048 -> 4.00x, T=4096 -> 4.00x. The router is
        # fp32 today, so this only costs a cast if someone changes that.
        acc_dtype = torch.promote_types(scores.dtype, torch.float32)
        # A new segment starts wherever `positions` is NOT the previous position
        # plus one. Keying on `positions == 0` alone is not enough:
        #
        #  - a CP shard can begin mid-document, so its first token has no 0; and
        #  - the default CP balancer is "headtail", which hands each rank two
        #    DISJOINT chunks concatenated. The join carries no `positions == 0`,
        #    so a reset-only rule merged the tail of one document with the
        #    middle of an unrelated one -- measured 1.32% error at cp=2.
        #
        # Token 0 always starts a segment, which also gives a shard that begins
        # mid-document a bucket of its own.
        #
        # LIMITATION: this makes the headtail merge conditional, not impossible.
        # The seam is missed exactly when `positions[tail_start] ==
        # positions[head_end] + 1`. Note rank cp-1 receives two globally
        # ADJACENT chunks, so a merge there is correct and must not be counted.
        # Measured:
        #   ragged packing, T=4096, doc lengths U(16,1200): 0.000% of
        #     (rank, step) pairs affected at cp=2, 0.083% at cp=4, 0.104% at
        #     cp=8; expected impact averaged over all pairs 0.00008%-0.00024%.
        #   equal-length documents, where the miss is DETERMINISTIC rather than
        #     rare: 0.088% at S=1024, 0.23% at S=4096, 2.6% at S=64.
        # Equal-length is not exotic (an unpacked loader, a fixed-length eval
        # batch, or eos_id=None all produce it), which is why this is worth
        # fixing rather than only documenting -- see `positions`/`doc_id` in
        # `models/common/decoder.py`.
        #
        # The attention mask is NOT affected: `Decoder.preprocess_inputs` builds
        # it from the GLOBAL `positions` before `prepare_context_parallel_input`
        # shards them, so only this loss has to rebuild boundaries from a
        # post-shard view.
        #
        # T buckets, because every token can start its own segment. STATIC
        # count: deriving n_docs via `.item()` would be a per-layer host sync
        # AND a data-dependent shape, which breaks `fullgraph=True`.
        new_seg = torch.ones_like(positions, dtype=torch.bool)
        if doc_id is not None:
            # `doc_id` is computed from the GLOBAL positions before CP sharding
            # (see OPTMoEModel.preprocess_inputs), so a change in it is a real
            # document boundary and the headtail seam cannot hide one. The
            # positions test is kept as well: it still catches a seam that
            # falls INSIDE one document, which doc_id alone would miss.
            did = doc_id.reshape(-1)
            new_seg[1:] = (did[1:] != did[:-1]) | (
                positions[1:] != (positions[:-1] + 1)
            )
        else:
            new_seg[1:] = positions[1:] != (positions[:-1] + 1)
        seg_id = torch.cumsum(new_seg.long(), dim=0) - 1
        T_all = seg_id.shape[0]
        n_buckets = T_all
        valid = (
            torch.ones(T_all, device=scores.device, dtype=acc_dtype)
            if loss_mask is None
            else loss_mask.reshape(-1).to(acc_dtype)
        )
        tok_per_doc = torch.zeros(n_buckets, device=scores.device, dtype=acc_dtype)
        tok_per_doc.scatter_add_(0, seg_id, valid)
        safe_tok = tok_per_doc.clamp_min(1.0)

        denom = scores.sum(dim=-1, keepdim=True) + 1e-20
        probs = (scores / denom).to(acc_dtype)  # s'_{i,t}
        # Eq 20 numerator, per document: psum[d, e] = sum_{t in d} s'_{e,t}
        psum = torch.zeros(n_buckets, N, device=scores.device, dtype=acc_dtype)
        psum.index_add_(0, seg_id, probs * valid.unsqueeze(-1))

        # Eq 17 contracts f_i against P_i. Both carry a 1/T_d, so
        #   sum_i f_i P_i = (N / (K * T_d^2)) * sum_e counts[d,e] * psum[d,e]
        # and the inner contraction can be rewritten
        #   sum_e counts[d,e] * psum[d,e] = sum_{t in d} sum_k psum[d, topk(t,k)]
        # because counts[d,e] only tallies how many of document d's tokens chose
        # expert e. That replaces the [T+1, N] count matrix with a [T, K] gather.
        # It matters: the count matrix was saved by autograd for the `f * P`
        # multiply and retained 40.8 MiB per MoE layer per microbatch at
        # T=40960/N=128 -- 1.28 GiB over 32 layers. A gather saves only indices.
        indices_l = indices.long()
        doc_rep = seg_id.unsqueeze(-1).expand_as(indices_l)
        gathered = psum.reshape(-1)[doc_rep.reshape(-1) * N + indices_l.reshape(-1)]
        per_tok = gathered.view_as(indices_l).sum(dim=-1) * valid
        numer = torch.zeros(n_buckets, device=scores.device, dtype=acc_dtype)
        numer.scatter_add_(0, seg_id, per_tok)
        per_doc = numer * (N / top_k) / (safe_tok * safe_tok)

        # TOKEN-WEIGHTED mean over documents, not an unweighted one.
        #
        # Each document is weighted by its VALID token count, so its influence
        # is proportional to what it actually contributes to the batch. Without
        # this, an unweighted mean gives a 1-token document the same say as a
        # 1200-token one, and under `pack_strategy="best_fit"` every pad token
        # is its own 1-token document. Measured on 3x1200 real + 496 pad tokens
        # with the router's real top-k, where 1.0 is perfect balance:
        #   unweighted, no mask     1.911   (91% inflated by padding)
        #   token-weighted, no mask 1.112   (11%)
        #   token-weighted + mask   1.00105 == real tokens only, exactly
        #
        # NOTE what this does NOT do: it does not make the statistic invariant
        # to document length. The per-document value is ~ 1 + c/T_d, because
        # the f/P correlation on a short document cannot be balanced away, only
        # flattened -- a 1-token document scores ~1.91 where a 512-token one
        # scores ~1.00. Weighting re-weights ACROSS ragged documents; it does
        # not remove that per-document bias. `loss_mask` is what removes padding
        # exactly, which is why SFT (`enable_token_mask_for_moe`) is the case
        # that lands on the target value.
        #
        # A fully padded document has weight 0 and drops out on its own, so no
        # separate non-empty mask is needed.
        #
        # This branch is taken unconditionally whenever `positions` is given:
        # `positions` IS the document structure, so segmenting by it is correct
        # for any B. Gating it on `n_real > 1` needed a `.item()` host sync and
        # made eager and compiled disagree for B > 1.
        return ((per_doc * tok_per_doc).sum() / tok_per_doc.sum().clamp_min(1.0)).to(
            scores.dtype
        )

    indices = indices.long()
    # 1. Reshape inputs to handle each sequence separately: (B, S, N)
    #    This ensures we calculate P_i and f_i per sequence (Eq 20 & 18).
    scores_per_seq = scores.view(B, S, N)
    indices_per_seq = indices.view(B, S, top_k)

    # 2. Eq 19: Normalize affinity scores s_{i,t} to get s'_{i,t}
    #    DeepSeek-V3 uses Sigmoid, so scores don't sum to 1.
    #    Eq 19 explicitly requires dividing by the sum of all affinities.
    #    denominator shape: (B, S, 1)
    denominator = scores_per_seq.sum(dim=-1, keepdim=True) + 1e-20
    probs_per_seq = scores_per_seq / denominator  # This is s'_{i,t}

    # 3. Eq 20: Calculate P_i (Average probability per expert for each sequence)
    #    P_i = (1/T) * sum_{t=1}^T (s'_{i,t})
    #    We average over the Sequence dimension (dim=1).
    #    P_i shape: (B, N)
    # `loss_mask` keeps padding / ignored positions out of both P_i and f_i.
    # With no mask this is exactly `probs_per_seq.mean(dim=1)` and
    # `valid_per_seq == S`, so the unmasked path is numerically unchanged.
    if loss_mask is None:
        mask_BS1 = None
        valid_per_seq = torch.full(
            (B, 1), float(S), device=scores.device, dtype=scores.dtype
        )
        P_i = probs_per_seq.mean(dim=1)
    else:
        mask_BS1 = loss_mask.reshape(B, S, 1).to(scores.dtype)
        valid_per_seq = mask_BS1.sum(dim=1).clamp_min(1.0)  # (B, 1)
        P_i = (probs_per_seq * mask_BS1).sum(dim=1) / valid_per_seq

    # 4. Eq 18: Calculate f_i (Fraction of tokens selecting expert i per sequence)
    #    f_i = (N / (K * T)) * count_i

    # Flatten the top-k dimension to count hits per sequence: (B, S*K)
    flat_indices_per_seq = indices_per_seq.view(B, -1)
    selection_counts = torch.zeros((B, N), device=scores.device, dtype=scores.dtype)
    if mask_BS1 is None:
        src = torch.ones_like(flat_indices_per_seq, dtype=scores.dtype)
    else:
        # each token contributes its mask value to all top_k of its slots
        src = mask_BS1.expand(B, S, top_k).reshape(B, -1).to(scores.dtype)
    selection_counts.scatter_add_(1, flat_indices_per_seq, src)

    # Calculate f_i per sequence; T is the number of *counted* tokens, which is
    # S when unmasked and the valid-token count otherwise.
    f_i = selection_counts * (N / top_k) / valid_per_seq

    # 5. Eq 17: Calculate Balance Loss
    loss_per_seq = (f_i * P_i).sum(dim=1)

    return loss_per_seq.mean()


def batch_wise_aux_loss(
    scores: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    top_k: int,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes Batch-Wise Auxiliary Loss.
    Args:
        scores: Dense probabilities (BS, N).
        num_tokens_per_expert: Token counts (N).
        top_k: Number of experts selected per token.
    """
    # Total number of routed experts (N)
    N = scores.size(1)

    # f_i, P_i and T must all be computed over the SAME token set. The caller
    # already masks the per-expert counts with `loss_mask`, so leaving P_i and T
    # unmasked made the three inconsistent: sum_i f_i < N, and P_i diluted by
    # pad rows. With no mask this reduces exactly to `scores.mean(dim=0)` and
    # `T = scores.size(0)`.
    if loss_mask is None:
        T = scores.size(0)
        P_i = scores.mean(dim=0)
    else:
        m = loss_mask.reshape(-1, 1).to(scores.dtype)
        T = m.sum().clamp_min(1.0)
        P_i = (scores * m).sum(dim=0) / T

    f_i = num_tokens_per_expert.to(scores.dtype) * (N / (top_k * T))

    loss = (f_i * P_i).sum()

    return loss


class NormGroupedExperts(Module):
    """Grouped experts with a mid-norm and a configurable activation.

    Upstream's ``forward`` is mirrored -- including the DTensor-to-local
    conversion, which expert parallelism needs, since EP feeds dynamic shapes
    that cannot be expressed as DTensors. The only changes are the activation
    and the norm applied between the gated activation and the down projection.

    NOT a subclass of ``common.moe.GroupedExperts``, deliberately. Subclassing
    would remove real duplication (``_grouped_mm`` is identical bar its
    docstring, and ``silu(gate) * up`` is bitwise equal to upstream's
    ``SwiGLU``), but ``Config.traverse`` matches by ``isinstance``, so deriving
    the Config would silently enrol these experts in the three quantization
    transforms that traverse ``GroupedExperts.Config``. Those swap in a
    quantized class chosen from ``type(config)._owner``, which knows nothing
    about ``mid_norm``. opt_moe never enables quantization today, so it would be
    a latent trap rather than a visible break -- which is the worse kind.
    Revisit if the Config hierarchy stops driving transform matching.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        norm_everywhere: bool = False
        norm_type: str = "np_rmsnorm"
        norm_eps: float = 1e-30
        norm_sharding_config: Any | None = None
        """Sharding plan for `mid_norm`, which this config builds internally.
        See GatedNormSWAttention.Config.norm_sharding_config."""
        activation_type: str = "silu"
        layer_id: int = 0
        # Per-expert initialization; see init_all_experts_different.
        w1_init_fn_type: str = "scaled_orthogonal"
        w2_init_fn_type: str = "scaled_orthogonal"
        w3_init_fn_type: str = "scaled_orthogonal"
        w1_init_std: float = 1.0
        w2_init_std: float = 1.0
        w3_init_std: float = 1.0
        residual_div: float = 1.0
        init_gate_as_residual: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.w1_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.act_fn = build_activation(config.activation_type)
        if config.norm_everywhere:
            self.mid_norm = build_norm_config(
                config.norm_type,
                config.hidden_dim,
                config.norm_eps,
                sharding_config=config.norm_sharding_config,
            ).build()
        else:
            self.mid_norm = Identity.Config().build()

    def forward(
        self, x_RD: torch.Tensor, num_tokens_per_expert_E: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            # EP feeds dynamic shapes that cannot be expressed as DTensors, so
            # drop to local tensors first. Mirrors upstream.
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD, w2_EDF, w3_EFD = self.w1_EFD, self.w2_EDF, self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                # if no EP, convert to V for grouped_mm, which would otherwise see
                # x:R, w1:V, offsets:P in local SPMD typechecking.
                # spmd.P is not currently allowed to mix with spmd.V.
                # Verbatim from upstream `models/common/moe.py:85-96`.
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        h_RF = self.act_fn(
            self._grouped_mm(A=x_RD.bfloat16(), weight_EOI=w1_EFD, offs=offsets_E)
        )
        h_RF = h_RF * self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=w3_EFD, offs=offsets_E
        )
        return self._grouped_mm(
            A=self.mid_norm(h_RF), weight_EOI=w2_EDF, offs=offsets_E
        ).type_as(x_RD)

    def _init_self_parameters(self) -> None:
        """Draw every expert's weights independently.

        A single initializer over the ``(E, ...)`` parameter would work, but
        seeding per (layer, slot, expert) keeps each expert's draw reproducible
        and independent of the expert count.
        """
        cfg = self.config
        w3_div = cfg.residual_div if cfg.init_gate_as_residual else 1.0
        for slot, (w, fn_type, std) in enumerate(
            (
                (self.w1_EFD, cfg.w1_init_fn_type, cfg.w1_init_std),
                (self.w2_EDF, cfg.w2_init_fn_type, cfg.w2_init_std / cfg.residual_div),
                (self.w3_EFD, cfg.w3_init_fn_type, cfg.w3_init_std / w3_div),
            )
        ):
            init_all_experts_different(
                build_init_fn(fn_type), w.data, std, slot=slot, layer_id=cfg.layer_id
            )

    def _grouped_mm(
        self, *, A: torch.Tensor, weight_EOI: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        """Grouped matmul of ``A @ weight_EOI.transpose(-2, -1)``.

        Vendored from ``models/common/moe.py``. ``weight_EOI`` is the stored
        ``(experts, out_features, in_features)`` orientation; the transpose to
        the grouped-GEMM right operand happens here. Overridable seam for
        low-precision variants (the MXFP8 converter swaps this for a scaled
        grouped GEMM), which is why the weight rather than its transpose is
        passed: a quantized representation is keyed off the stored orientation.
        """
        return torch._grouped_mm(A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs)


class RouterOutput(NamedTuple):
    """What NormRouter returns.

    Upstream returns a bare 3-tuple; the extra fields are the per-layer metrics
    this fork logs, which cannot be recovered downstream without redoing the
    router's work. ``experts_entropy`` and ``unbiased_expert_ids_TK`` are None
    when not requested.
    """

    topk_scores_TK: torch.Tensor  # noqa: N815
    topk_expert_ids_TK: torch.Tensor  # noqa: N815
    scores_TE: torch.Tensor  # noqa: N815
    experts_entropy: torch.Tensor | None
    unbiased_expert_ids_TK: torch.Tensor | None  # noqa: N815


class NormRouter(Module):
    """Token-choice top-K router, vendored from ``models/common/moe.py``.

    Upstream's ``TokenChoiceTopKRouter`` returns only
    ``(topk_scores_TK, topk_expert_ids_TK, scores_TE)``. The metrics this fork
    logs per layer -- routing entropy, masked per-expert token counts, and the
    *unbiased* top-k used by the complementary balance loss -- cannot be
    recovered from that return without recomputing work the router already did,
    so the class is copied here and extended rather than wrapped.

    Differences from upstream, all additive:

    * ``forward`` also returns ``experts_entropy`` and ``unbiased_expert_ids_TK``.
      (``counted_per_expert_E`` is NOT on ``RouterOutput``; it is computed in
      ``NormMoE.forward``, which is where the loss mask is available.)
    * Entropy is measured on the renormalised weights, before ``route_scale`` is
      applied -- scaled weights are not a distribution and their entropy is
      meaningless.
    * ``loss_mask`` marks padding tokens. It affects the *counted* statistics
      only; dispatch counts must include every routed token because they size
      the grouped GEMM.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: RouterGateLinear.Config
        num_expert_groups: int | None = None
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid"
        route_norm: bool = True
        """Upstream defaults this to False. It must stay True here:
        ``calc_gate_scaling_factor`` derives ``route_scale`` from weights that
        have already been normalised to sum to 1, so with it off the scale is
        calibrated against a distribution that never occurs."""
        route_scale: float = 1.0
        track_metrics: bool = True
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.track_metrics = config.track_metrics
        self._debug_force_load_balance = config._debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores_TE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment, shape ``(T, K)``."""
        num_tokens = scores_TE.shape[0]
        topk_expert_ids_TK = (
            torch.arange(
                num_tokens * self.top_k, device=scores_TE.device, dtype=torch.int64
            ).reshape(num_tokens, self.top_k)
            % self.num_experts
        )
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
        return topk_expert_ids_TK, topk_scores_TK

    def _get_node_limited_routing_scores(
        self, scores_for_choice_TE: torch.Tensor
    ) -> torch.Tensor:
        """Keep only ``num_limited_groups`` expert groups; -inf the rest."""
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice_TE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, group_idx, False)  # False = selected groups (keep)
        return scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

    def _select_experts(self, scores_TE: torch.Tensor) -> torch.Tensor:
        if self.num_expert_groups is not None:
            scores_TE = self._get_node_limited_routing_scores(scores_TE)
        return torch.topk(scores_TE, k=self.top_k, dim=-1, sorted=False).indices

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        # Accepted for interface compatibility but NOT used here: routing itself
        # is per-token and unaffected by padding. The mask is applied downstream,
        # where it matters -- expert counting and the aux load-balance loss in
        # NormMoE.forward.
        loss_mask: torch.Tensor | None = None,
        need_unbiased_ids: bool = False,
        **router_kwargs,
    ) -> "RouterOutput":
        scores_TE = self.gate(x_TD)  # RouterGateLinear returns FP32.

        if self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            scores_TE = F.softplus(scores_TE).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # The bias steers selection only; gate values come from the raw scores.
        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        topk_expert_ids_TK = self._select_experts(scores_for_choice_TE)
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)

        if self._debug_force_load_balance:
            (
                topk_expert_ids_TK,
                topk_scores_TK,
            ) = self._debug_force_load_balance_routing(scores_TE)

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator

        # Entropy is measured here, on the renormalised weights: after
        # route_scale they no longer form a distribution.
        experts_entropy = None
        if self.track_metrics:
            with torch.no_grad():
                detached = topk_scores_TK.detach()
                experts_entropy = (
                    -(detached * detached.clamp_min(1e-20).log()).sum(dim=-1).mean()
                )

        topk_scores_TK = topk_scores_TK * self.route_scale

        # DeepSeek-V3 Eq. 18 counts the top-k of the *raw* affinities: the bias
        # is a routing-time correction, and feeding it to the balance loss makes
        # the loss measure a distribution the bias has already flattened. With
        # no bias the selection above is already unbiased.
        unbiased_expert_ids_TK = None
        if need_unbiased_ids:
            unbiased_expert_ids_TK = (
                topk_expert_ids_TK
                if expert_bias_E is None
                else self._select_experts(scores_TE)
            )

        return RouterOutput(
            topk_scores_TK=topk_scores_TK,
            topk_expert_ids_TK=topk_expert_ids_TK,
            scores_TE=scores_TE,
            experts_entropy=experts_entropy,
            unbiased_expert_ids_TK=unbiased_expert_ids_TK,
        )


class LoadBalanceLoss(AuxLoss):
    """opt_moe's load-balance loss, on upstream's aux-loss injection path.

    Replaces the old ``(logits, load_balance_loss)`` tuple return. That tuple had
    to be threaded through every transformer block AND across pipeline stages
    (``model.forward``'s ``accumulated_load_balance_loss``); injecting the
    gradient at the layer keeps it local, so the model returns logits alone and
    the training loss is plain cross-entropy again.

    NOTE ON SCALE: ``sequence_wise_aux_loss`` / ``batch_wise_aux_loss`` return
    the UNSCALED loss; ``Config.coeff`` is the only coefficient, matching
    upstream's ``AuxLoss.Config``. ``inject`` applies
    ``coeff / global_valid_tokens``, and that per-step normalisation is NEW --
    it is what puts this term on the same footing as the main loss, which
    trainer.py computes as ``local_sum / global_valid_tokens``. The old
    ``MoEAuxLoss`` divided by neither, so a coefficient carried over from
    before this change is not equivalent and wants re-measuring.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """``AuxLoss.Config``'s ``coeff`` plus the formula to use.

        A distinct Config is required regardless: ``Config.build()`` constructs
        the class that owns it, so an ``AuxLoss.Config`` would build a bare
        ``AuxLoss`` with no ``forward``.
        """

        loss_type: Literal["sequence_wise", "batch_wise"] = "sequence_wise"
        """Which balance formula to use. Upstream declares one AuxLoss subclass
        per loss; we keep both DeepSeek variants behind one class."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.loss_type = config.loss_type

    def forward(self, raw_sum: torch.Tensor, *, carrier: torch.Tensor):
        """Inject ``raw_sum``'s gradient onto ``carrier`` and return it unchanged."""
        return self.inject(raw_sum, carrier=carrier)


class NormMoE(Module):
    """MoE with an auxiliary load-balance loss and routing metrics.

    ``forward`` returns just ``out_TD``. The load-balance gradient is injected
    at this layer by ``LoadBalanceLoss`` (upstream's AuxLoss path), so nothing
    is returned upward and no tuple crosses block or pipeline-stage boundaries.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        routed_experts: RoutedExperts.Config
        router: NormRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: Any | None = None
        aux_loss: LoadBalanceLoss.Config | None = None
        """Load-balance auxiliary loss, or None to disable it. Mirrors
        upstream's ``aux_loss: AuxLoss.Config | None``; ``coeff`` is the only
        coefficient."""
        bias_update_norm_factor: str = "sign"
        track_router_metrics: bool = True
        """Accumulate router entropy and max-violation. Costs a log() over the
        (T, K) scores each step, so it can be turned off."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.routed_experts = config.routed_experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )
        # Auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664).
        # tokens_per_expert_E accumulates in forward; expert_bias_E is updated
        # outside the model in an optimizer-step pre-hook so it composes with
        # gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(config.num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )
        # Upstream's aux-loss path: the gradient is injected at this layer, so
        # nothing has to be returned up through the blocks or across PP stages.
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None
        self.bias_update_norm_factor = config.bias_update_norm_factor
        self.track_router_metrics = config.track_router_metrics

        num_experts = config.num_experts
        self.register_buffer(
            "router_entropy", torch.zeros(1, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "acc_fwd_times", torch.zeros(1, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "tokens_per_expert_cumul",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        doc_id: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x_TD: Input ``(T, D)``.
            loss_mask: Optional ``(T,)`` mask of real (non-padding) tokens. It
                affects load-balance accounting only, never dispatch.

        Returns:
            ``out_TD``. Any load-balance gradient is injected in place via
            ``LoadBalanceLoss``; nothing is returned for the caller to sum.
        """
        need_lb_loss = self.training and self.aux_loss is not None
        routed = self.router(
            x_TD,
            self.expert_bias_E,
            loss_mask=loss_mask,
            need_unbiased_ids=need_lb_loss,
            **router_kwargs,
        )
        topk_scores_TK = routed.topk_scores_TK
        topk_expert_ids_TK = routed.topk_expert_ids_TK
        scores_TE = routed.scores_TE

        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1, topk_expert_ids_TK, True
        )
        # Dispatch counts must reflect every routed token, padding included --
        # they size the grouped GEMM. Only the balance statistics are masked.
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)
        if loss_mask is not None:
            counted_per_expert_E = (routing_map_TE & loss_mask.reshape(-1, 1)).sum(
                dim=0
            )
        else:
            counted_per_expert_E = num_local_tokens_per_expert_E

        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(counted_per_expert_E)
                # Count the forward pass unconditionally. This is a
                # forward-pass counter, not an entropy counter: the consumer
                # (optimizers/container.py) divides `tokens_per_expert_E` --
                # which accumulates on EVERY forward -- by it. Incrementing it
                # only when entropy was tracked meant `track_router_metrics=
                # False` left it at 0 and the consumer did `all_tokens // 0`.
                self.acc_fwd_times.add_(1)
                if routed.experts_entropy is not None:
                    self.router_entropy.add_(routed.experts_entropy)

        lb_raw = None
        if need_lb_loss:
            aux_ids_TK = routed.unbiased_expert_ids_TK
            if self.aux_loss.loss_type == "sequence_wise":
                # Token-flat batches are one packed stream, so the whole
                # microbatch is scored as a single sequence.
                lb_raw = sequence_wise_aux_loss(
                    scores_TE,
                    aux_ids_TK,
                    1,
                    x_TD.shape[0],
                    self.router.top_k,
                    loss_mask=loss_mask,
                    # segments the packed stream back into documents; falls
                    # back to whole-stream scoring when unavailable
                    positions=positions,
                    doc_id=doc_id,
                )
            elif self.aux_loss.loss_type == "batch_wise":
                aux_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
                    -1, aux_ids_TK, True
                )
                if loss_mask is not None:
                    aux_map_TE = aux_map_TE & loss_mask.reshape(-1, 1)
                lb_raw = batch_wise_aux_loss(
                    scores_TE,
                    aux_map_TE.sum(dim=0),
                    self.router.top_k,
                    loss_mask=loss_mask,
                )
            else:
                raise ValueError(f"Invalid loss_type: {self.aux_loss.loss_type}")
        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        if self.shared_experts is not None:
            out_TD = out_TD + self.shared_experts(x_TD)
        if lb_raw is not None and self.aux_loss is not None:
            # `inject` wants an unnormalised per-microbatch SUM and divides by
            # the step's global valid tokens; our loss fns return a mean. Weight
            # by THIS microbatch's valid tokens so that division becomes a
            # token-weighted mean over the accumulation window and DP. (Using
            # the global count instead cancels the division, and the aux
            # gradient then scales with gradient_accumulation_steps.)
            n_valid = (
                loss_mask.sum()
                if loss_mask is not None
                else torch.tensor(
                    x_TD.shape[0], device=x_TD.device, dtype=torch.float32
                )
            )
            # Identity forward: out_TD is unchanged, its backward now carries the
            # load-balance gradient, and the per-layer metric accumulates in
            # self.aux_loss.instance_acc.
            out_TD = self.aux_loss(lb_raw * n_valid.to(lb_raw.dtype), carrier=out_TD)
        return out_TD

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Reset the MoE counters, honouring an explicit ``buffer_device``.

        Zeroing in place is preferred over recreating the tensors the way
        upstream's ``MoE._init_self_buffers`` does: after ``to_empty()`` the
        buffers already sit on the target device with the right shapes, and
        recreating would replace a DTensor buffer with a plain tensor, which
        ``Module.init_states`` then has to undo by re-distributing
        (protocols/module.py).

        But ``buffer_device`` genuinely can differ from where the buffers are:
        ``training.enable_cpu_offload`` initialises on CPU and passes
        ``buffer_device=cuda`` (trainer.py: ``init_device="cpu"``,
        ``buffer_device=torch.device(device_type)``), which is the whole reason
        the parameter exists. So move first when it differs, then zero -- moving
        with ``.to()`` rather than rebuilding keeps a DTensor a DTensor.
        """
        target = torch.device(buffer_device) if buffer_device is not None else None

        def _reset(name: str) -> None:
            buf = getattr(self, name)
            if buf is None:
                return
            if target is not None:
                cur = buf.device
                # "cuda" and "cuda:0" name the same device but compare unequal,
                # so only compare the index when the caller supplied one.
                same = target.type == cur.type and (
                    target.index is None or target.index == cur.index
                )
                if not same:
                    buf = buf.to(target)
                    setattr(self, name, buf)
            buf.zero_()

        for _name in (
            "expert_bias_E",
            "tokens_per_expert_E",
            "router_entropy",
            "acc_fwd_times",
            "tokens_per_expert_cumul",
        ):
            _reset(_name)


def make_norm_moe_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    layer_id: int = 0,
    num_shared_experts: int = 1,
    scaling_factor: float | None = None,
    score_func: str = "sigmoid",
    load_balance_coeff: float | None = 1e-3,
    aux_loss: LoadBalanceLoss.Config | None = None,
    norm_everywhere: bool = False,
    norm_type: str = "np_rmsnorm",
    norm_eps: float = 1e-30,
    activation_type: str = "silu",
    residual_div: float = 1.0,
    init_gate_as_residual: bool = False,
    bias_update_norm_factor: str = "sign",
    moe_comm_backend: str = "standard",
    track_router_metrics: bool = True,
    w1_init_fn_type: str = "scaled_orthogonal",
    w2_init_fn_type: str = "scaled_orthogonal",
    w3_init_fn_type: str = "scaled_orthogonal",
    w1_init_std: float = 1.0,
    w2_init_std: float = 1.0,
    w3_init_std: float = 1.0,
    router_init_fn_type: str = "scion_normal_output",
    router_init_std: float = 1.0,
    _debug_force_load_balance: bool = False,
) -> "NormMoE.Config":
    """Build a fully-specified NormMoE.Config from OPT MoE's flat settings.

    Mirrors upstream's ``make_moe_config`` / ``make_router_config`` so the structure
    matches. The token dispatcher comes from upstream's
    ``make_token_dispatcher_config(comm_backend=...)`` -- "standard" is all-to-all and
    falls back to local dispatch at EP=1 -- rather than defaulting to the local one and
    being swapped afterwards; see the note further down this file.
    """
    route_scale = saint_check_scaling_factor(
        num_routed_experts=num_experts,
        activate_experts=top_k,
        scaling_factor=scaling_factor,
    )

    return NormMoE.Config(
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
        aux_loss=aux_loss,
        bias_update_norm_factor=bias_update_norm_factor,
        track_router_metrics=track_router_metrics,
        router=NormRouter.Config(
            num_experts=num_experts,
            # RouterGateLinear keeps the gate in FP32 through the backward pass
            # too, which the previous hand-rolled `.float()` did not.
            gate=RouterGateLinear.Config(
                in_features=dim,
                out_features=num_experts,
                param_init={
                    "weight": make_param_init(router_init_fn_type, router_init_std)
                },
            ),
            top_k=top_k,
            score_func=score_func,
            # `calc_gate_scaling_factor` derives `route_scale` from top-k weights
            # that have already been normalised to sum to 1 (`p = p / p.sum()`),
            # so `route_norm` MUST stay on: with it off, `route_scale` is
            # calibrated against a distribution that never occurs. Upstream's
            # default is False, which is why this is set explicitly.
            route_norm=True,
            route_scale=route_scale,
            track_metrics=track_router_metrics,
            _debug_force_load_balance=_debug_force_load_balance,
        ),
        routed_experts=RoutedExperts.Config(
            inner_experts=NormGroupedExperts.Config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                norm_everywhere=norm_everywhere,
                norm_type=norm_type,
                norm_eps=norm_eps,
                activation_type=activation_type,
                layer_id=layer_id,
                w1_init_fn_type=w1_init_fn_type,
                w2_init_fn_type=w2_init_fn_type,
                w3_init_fn_type=w3_init_fn_type,
                w1_init_std=w1_init_std,
                w2_init_std=w2_init_std,
                w3_init_std=w3_init_std,
                residual_div=residual_div,
                init_gate_as_residual=init_gate_as_residual,
            ),
            # Upstream's factory, not a hardcoded LocalTokenDispatcher: that
            # class only ever does local dispatch, so EP>1 could never work
            # however the mesh was configured -- despite the docstrings in this
            # file claiming EP "comes from upstream's token dispatcher and needs
            # no code here". `comm_backend="standard"` uses all-to-all and falls
            # back to local dispatch when EP=1 (ep_mesh is None at runtime), so
            # this is behaviour-preserving at the EP=1 we run today while
            # actually allowing EP>1, deepep and hybridep later.
            token_dispatcher=make_token_dispatcher_config(
                num_experts=num_experts,
                top_k=top_k,
                comm_backend=moe_comm_backend,
                # MODEL dim, not the expert FFN dim: this sizes the DeepEP /
                # HybridEP / MinimalAsyncEP comm buffers, whose rows are D-wide.
                # Upstream passes `dim` here too (config_utils.py, and its
                # docstring says "hidden_dim (model dim) sizes the buffer").
                # Dormant under comm_backend="standard", which ignores it.
                hidden_dim=dim,
            ),
        ),
        shared_experts=(
            FeedForward.Config(
                dim=dim,
                hidden_dim=hidden_dim * num_shared_experts,
                activation_type=activation_type,
                norm_everywhere=norm_everywhere,
                norm_type=norm_type,
                norm_eps=norm_eps,
                w1_init_fn_type=w1_init_fn_type,
                w2_init_fn_type=w2_init_fn_type,
                w3_init_fn_type=w3_init_fn_type,
                w1_init_std=w1_init_std,
                w2_init_std=w2_init_std,
                w3_init_std=w3_init_std,
                residual_div=residual_div,
                init_gate_as_residual=init_gate_as_residual,
            )
            if num_shared_experts > 0
            else None
        ),
    )


class MoE:
    """Flat authoring surface for ``NormMoE``.

    The flavor registry describes an MoE with flat scalars (``hidden_dim``,
    ``num_experts``, ``scaling_factor``, ...), while upstream's MoE.Config is
    structured (``router``, ``routed_experts``, a token dispatcher).
    ``OPTMoEModel.Config._expand_layers`` calls ``to_norm_moe_config`` to turn
    one into the other, so the per-layer config the model builds from is a real
    ``NormMoE.Config`` -- which is what lets upstream's
    ``update_ep_token_dispatcher_config`` find ``routed_experts`` and swap in an
    EP dispatcher.

    This exists only to avoid rewriting the registry in the same change; it is
    a translation layer, not a second source of truth.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        hidden_dim: int = 0
        num_experts: int = 8
        num_shared_experts: int = 1
        top_k: int = 1
        scaling_factor: float | None = None
        score_func: str = "sigmoid"
        load_balance_coeff: float | None = 1e-3
        aux_loss: LoadBalanceLoss.Config | None = None
        bias_update_norm_factor: str = "sign"
        track_router_metrics: bool = True
        moe_comm_backend: str = "standard"
        """EP communication backend: "standard" (all-to-all, falls back to local
        dispatch at EP=1), "deepep" or "hybridep". Upstream selects this per
        flavor rather than per run."""
        norm_everywhere: bool = False
        norm_type: str = "np_rmsnorm"
        norm_eps: float = 1e-30
        activation_type: str = "silu"
        # NOTE there is deliberately no `force_router_on_fp32`. 0.4.0 needed it to wrap
        # the router gate in autocast(float32); upstream's RouterGateLinear
        # (models/common/linear.py) now yields FP32 logits unconditionally -- CUDA bf16
        # operands go through torch.mm(..., out_dtype=torch.float32) (bf16 multiply,
        # FP32 accumulate), every other path promotes to fp32, and both backward GEMMs
        # are fp32. The knob has nothing left to switch off. It previously existed here
        # as a field that was declared, defaulted True and then silently dropped by
        # to_norm_moe_config, so setting it False was a no-op that read as supported.
        w1_init_fn_type: str = "scaled_orthogonal"
        w2_init_fn_type: str = "scaled_orthogonal"
        w3_init_fn_type: str = "scaled_orthogonal"
        w1_init_std: float = 1.0
        w2_init_std: float = 1.0
        w3_init_std: float = 1.0
        router_init_fn_type: str = "scion_normal_output"
        router_init_std: float = 1.0
        _debug_force_load_balance: bool = False
        # Stamped per layer by the model's config expansion.
        dim: int = 0
        layer_id: int = 0
        residual_div: float = 1.0
        init_gate_as_residual: bool = False

        # NOTE there are deliberately no `use_grouped_mm` or
        # `score_before_experts` fields either, for the same reason as
        # `force_router_on_fp32` above: both were declared, defaulted to the
        # only value that works, and then raised if you changed them. Upstream's
        # GroupedExperts is grouped-GEMM only and its token dispatcher applies
        # routing scores in combine(), so neither knob has anything to switch.
        # Nothing in the tree set either one.
        def to_norm_moe_config(self) -> "NormMoE.Config":
            return make_norm_moe_config(
                dim=self.dim,
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                layer_id=self.layer_id,
                num_shared_experts=self.num_shared_experts,
                scaling_factor=self.scaling_factor,
                score_func=self.score_func,
                load_balance_coeff=self.load_balance_coeff,
                aux_loss=self.aux_loss,
                norm_everywhere=self.norm_everywhere,
                norm_type=self.norm_type,
                norm_eps=self.norm_eps,
                activation_type=self.activation_type,
                residual_div=self.residual_div,
                init_gate_as_residual=self.init_gate_as_residual,
                track_router_metrics=self.track_router_metrics,
                bias_update_norm_factor=self.bias_update_norm_factor,
                moe_comm_backend=self.moe_comm_backend,
                w1_init_fn_type=self.w1_init_fn_type,
                w2_init_fn_type=self.w2_init_fn_type,
                w3_init_fn_type=self.w3_init_fn_type,
                w1_init_std=self.w1_init_std,
                w2_init_std=self.w2_init_std,
                w3_init_std=self.w3_init_std,
                router_init_fn_type=self.router_init_fn_type,
                router_init_std=self.router_init_std,
                _debug_force_load_balance=self._debug_force_load_balance,
            )
