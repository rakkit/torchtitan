# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OPT MoE load balancing: the expert-bias update and its metrics.

Registered as an optimizer step pre-hook by ``OPTMoEModel``. The bias update
runs every step; on the steps the container arms, the metrics ride one packed
D2H copy and are published per layer.
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import fsdp_shard_mesh
from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook

if TYPE_CHECKING:
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

__all__ = ["register_moe_load_balancing_hook", "publish_moe_metrics"]

logger = logging.getLogger(__name__)

MAXVIO_EMA_BETA = 0.995
MAXVIO_EPS = 1e-12


def publish_moe_metrics(data: tuple) -> None:
    """Per-layer router metrics from the packed host lists, onto each MoE
    module's `_log_expert_metrics` for `collect_moe_metrics` to read."""
    # Seeded BEFORE the try: if the unpack raises, the except handler
    # still needs the layer list to clear.
    moe_layers_info: list[dict] = data[0] if isinstance(data, tuple) and data else []
    # A metrics failure must not stop training: log it and publish nothing.
    try:
        (
            moe_layers_info,
            all_usages_cpu,
            all_biases_cpu,
            all_entropies_cpu,
            all_load_balance_losses_cpu,
            all_maxvio_batch_cpu,
            all_maxvio_ema_cpu,
            all_bias_mean_cpu,
            all_bias_std_cpu,
            num_experts,
            has_aux_loss,
        ) = data

        usage_offset = bias_offset = 0
        for i, info in enumerate(moe_layers_info):
            moe = info["module"]
            layer_id = info["layer_id"]

            layer_usages = all_usages_cpu[usage_offset : usage_offset + num_experts]
            layer_biases = all_biases_cpu[bias_offset : bias_offset + num_experts]
            metrics = {
                f"moe_entropy/L-{layer_id}": all_entropies_cpu[i],
                f"moe_maxvio_batch/L-{layer_id}": all_maxvio_batch_cpu[i],
                f"moe_maxvio_ema/L-{layer_id}": all_maxvio_ema_cpu[i],
                f"moe_bias_mean/L-{layer_id}": all_bias_mean_cpu[i],
                f"moe_bias_std/L-{layer_id}": all_bias_std_cpu[i],
            }
            if has_aux_loss:
                metrics[
                    f"moe_load_balance_loss/L-{layer_id}"
                ] = all_load_balance_losses_cpu[i]
            pre_usage = f"moe_ep_usage/L-{layer_id}_EP-"
            pre_bias = f"moe_bias/L-{layer_id}_EP-"
            metrics.update({f"{pre_usage}{j}": v for j, v in enumerate(layer_usages)})
            metrics.update({f"{pre_bias}{j}": v for j, v in enumerate(layer_biases)})
            moe._log_expert_metrics = metrics
            usage_offset += num_experts
            bias_offset += num_experts

        # Aggregated scalars across all MoE layers — attached to first layer
        num_moe_layers = len(moe_layers_info)
        moe_layers_info[0]["module"]._log_expert_metrics.update(
            {
                "moe_maxvio_batch/aggregate": sum(all_maxvio_batch_cpu)
                / num_moe_layers,
                "moe_maxvio_ema/aggregate": sum(all_maxvio_ema_cpu) / num_moe_layers,
            }
        )

    except Exception:
        logger.exception("MoE router metrics failed on one payload")
        # Clear, or collect_moe_metrics republishes the previous step's
        # values as current -- a flat chart segment rather than a gap.
        for info in moe_layers_info:
            info["module"]._log_expert_metrics = {}


def fused_hier_reduce_loss_stats(
    parallel_dims,
    all_tokens: torch.Tensor,
    all_entropies: torch.Tensor,
    all_load_balance_losses: torch.Tensor,
):
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    if loss_mesh is None:
        return

    # 1. Determine Topology
    fsdp_mesh = fsdp_shard_mesh(parallel_dims)
    dp_mesh = parallel_dims.get_optional_mesh("dp_replicate")

    # Hierarchical only under HSDP (both meshes present); the flat path is
    # mathematically equivalent -- same rank set, same normalisation -- so the
    # choice is a communication-cost one and needs no knob.
    use_hierarchical = (fsdp_mesh is not None) and (dp_mesh is not None)

    # 2. Fuse & Pack in fp32: measured-sufficient (zero sign flips across 18
    # simulated mesh/spread cells, max ~1.3e-6 relative at 1024 ranks). Keep the
    # `/acc_fwd_times` division BEFORE this reduce or the sums start rounding.
    acc = torch.float32
    t0 = all_tokens.reshape(-1).to(acc)
    t1 = all_entropies.reshape(-1).to(acc)
    t2 = all_load_balance_losses.reshape(-1).to(acc)

    buf = torch.cat([t0, t1, t2])

    # 3. Perform Reduction
    if use_hierarchical:
        # Hierarchical: FSDP (Intra-node) -> DP (Inter-node)
        # Using SUM for all, we will normalize averaging later
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=fsdp_mesh.get_group())
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
    else:
        # Fallback: Flat all-reduce on the global loss mesh
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=loss_mesh.get_group())

    # 4. Unpack & Normalize
    ws_loss = dist.get_world_size(group=loss_mesh.get_group())

    n0, n1, n2 = t0.numel(), t1.numel(), t2.numel()

    # Slicing views
    out_tokens = buf[0:n0].view_as(all_tokens)
    out_ent = buf[n0 : n0 + n1].view_as(all_entropies)
    out_lb = buf[n0 + n1 : n0 + n1 + n2].view_as(all_load_balance_losses)

    # 5. Copy back to inputs
    # Tokens: SUM (no division)
    all_tokens.copy_(out_tokens.to(all_tokens.dtype))

    # Entropy: AVG (divide SUM by world size) -- a per-forward mean per rank.
    all_entropies.copy_((out_ent / ws_loss).to(all_entropies.dtype))
    # Load-balance loss: SUM, matching upstream's `collect_aux_loss_metrics`.
    # Each rank holds its own token share, so the sum IS the global value.
    all_load_balance_losses.copy_(out_lb.to(all_load_balance_losses.dtype))


@torch.no_grad()
def _quantile_update_expert_bias(
    balancers: list,
    bias_params: list[torch.Tensor],
    parallel_dims: ParallelDims,
    loss_mesh,
) -> None:
    """Solve each layer's expert bias from its required-bias histogram.

    Replaces the DeepSeek-V3 sign nudge: instead of stepping the bias by
    +/-coeff on the sign of the load error, this reads off the bias quantile
    that would have given each expert exactly ``top_k / num_experts`` of the
    tokens. There is no learning rate, so it converges in one step rather than
    over many.
    """
    reduction_groups = []
    # With EP the router is token-sharded on the dense TP axis even when
    # model-wide sequence parallelism is off.
    if parallel_dims.ep_enabled and parallel_dims.tp > 1:
        reduction_groups.append(parallel_dims.get_dense_tp_mesh().get_group())
    if loss_mesh is not None:
        reduction_groups.append(loss_mesh.get_group())

    histograms = [b.required_bias_histogram_EB for b in balancers]
    if reduction_groups:
        stacked_LEB = torch.stack(histograms)
        for group in reduction_groups:
            torch.distributed.all_reduce(
                stacked_LEB, group=group, op=torch.distributed.ReduceOp.SUM
            )
        histograms = list(stacked_LEB.unbind())

    for histogram_EB, balancer, expert_bias_E in zip(
        histograms, balancers, bias_params, strict=True
    ):
        expert_bias_E.copy_(balancer.estimate_expert_bias(histogram_EB, expert_bias_E))
        balancer.required_bias_histogram_EB.zero_()


def register_moe_load_balancing_hook(
    optimizers: "DiSCOOptimizersContainer",
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> "DiSCOOptimizersContainer":
    def lmo_for_moe_bias(
        g,
        norm_factor="sign",
        epsilon=1e-32,
    ):
        if norm_factor in ["sign", "sign_zero_mean"]:
            return torch.sign(g)
        elif norm_factor in ["spectral", "spectral_zero_mean"]:
            is_flat = g.dim() == 1
            g = g.unsqueeze(0) if is_flat else g
            norms = torch.linalg.norm(g, ord=2, dim=1, keepdim=True)
            g = g / torch.clamp(norms, min=epsilon)
            g = g.squeeze(0) if is_flat else g
            return g
        elif norm_factor in ["rms", "rms_zero_mean"]:
            is_flat = g.dim() == 1
            g = g.unsqueeze(0) if is_flat else g
            rms = torch.sqrt(torch.mean(g.square(), dim=1, keepdim=True))
            g = g / torch.clamp(rms, min=epsilon)
            g = g.squeeze(0) if is_flat else g
            return g
        raise ValueError(
            f"unknown bias_update_norm_factor {norm_factor!r}. Returning None "
            "here would reach torch._foreach_add_ as a None element, and a "
            "near-miss like 'sign_zeromean' would silently drop zero-centring "
            "because the zero_mean check is endswith('zero_mean'). Valid: "
            "sign, spectral, rms, and their *_zero_mean variants."
        )

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        """
        Lets assume all MoE layers have same amount of experts.
        """

        loss_mesh = parallel_dims.get_optional_mesh("loss")

        # The `loss` mesh excludes tp, so loss-rank 0 is true on EVERY tp rank
        # and each would push a payload for the same layers. Require tp local
        # rank 0 too, as `components/metrics.py` does. Inert at tp=1.
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        is_dp_rank_0 = (loss_mesh is None or loss_mesh.get_local_rank() == 0) and (
            tp_mesh is None or tp_mesh.get_local_rank() == 0
        )
        # The metrics ride one D2H copy, so they are produced only on steps the
        # container will collect; every other step does the bias update alone.
        should_publish = optimizers.consume_moe_publish_arm() and is_dp_rank_0
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.

        moe_layers_info = []
        tok_buffers, ent_buffers, load_balance_loss_buffers = [], [], []
        ema_buffers = []
        acc_fwd_times_buffers = []
        num_experts = 0

        for part in model_parts:
            for block in part.layers.values():
                if not block.moe_enabled:
                    continue
                moe = block.moe
                # Assuming num_experts is the same for all, so we can just grab it once
                layer_num_experts = moe.tokens_per_expert_E.numel()
                if num_experts and layer_num_experts != num_experts:
                    raise ValueError(
                        "All MoE layers must have the same expert count: "
                        f"layer {block.layer_id} has {layer_num_experts}, an "
                        f"earlier layer has {num_experts}. The stats below are "
                        "packed with `.view(num_layers, num_experts)`, which "
                        "would silently mis-attribute them."
                    )
                num_experts = layer_num_experts
                moe_layers_info.append(
                    {
                        "module": moe,
                        "layer_id": block.layer_id,
                    }
                )
                tok_buffers.append(moe.tokens_per_expert_E)
                ema_buffers.append(moe.tokens_per_expert_cumul)
                ent_buffers.append(moe.router_entropy)
                acc_fwd_times_buffers.append(moe.acc_fwd_times)
                # Per-layer load-balance value: upstream's AuxLoss keeps a
                # per-instance accumulator, so the `moe_load_balance_loss/L-*`
                # breakdown survives. Zero scalar when the layer has none.
                aux = getattr(moe, "aux_loss", None)
                load_balance_loss_buffers.append(
                    aux.instance_acc.reshape(1)
                    if aux is not None
                    else torch.zeros(
                        1, dtype=torch.float32, device=moe.acc_fwd_times.device
                    )
                )
        # Early exit if no MoE layers were found
        if not moe_layers_info:
            return

        # Everything below is [num_layers, num_experts]; working 2-D directly
        # avoids a flatten plus `repeat_interleave`/`index_add_` round trip.
        num_layers = len(moe_layers_info)
        all_tokens = torch.stack(tok_buffers)  # [L, E]
        all_entropies = torch.cat(ent_buffers)  # [L]
        all_load_balance_losses = torch.cat(load_balance_loss_buffers)  # [L]

        # Each buffer accumulates once per FORWARD, so divide by each layer's
        # OWN counter (selective AC makes a shared one wrong). True division:
        # `//` would floor these float32 counts and collapse expert loads.
        sf = torch.cat(acc_fwd_times_buffers).to(all_tokens.dtype).clamp_min(1.0)
        all_tokens = all_tokens / sf.unsqueeze(1)
        all_entropies = all_entropies / sf
        # NOT all_load_balance_losses: AuxLoss.instance_acc already carries
        # `raw_sum / global_valid_tokens`, a whole-step, all-rank denominator.

        if loss_mesh is not None:
            fused_hier_reduce_loss_stats(
                parallel_dims, all_tokens, all_entropies, all_load_balance_losses
            )

        layer_sums = all_tokens.sum(dim=1, keepdim=True)  # [L, 1]
        layer_means = layer_sums / num_experts  # [L, 1]

        # Per-layer EMA: ema = beta * ema + (1 - beta) * step_counts. State, so
        # it advances every step; only its readout is gated below.
        torch._foreach_mul_(ema_buffers, MAXVIO_EMA_BETA)
        torch._foreach_add_(
            ema_buffers, list(all_tokens.unbind(0)), alpha=(1.0 - MAXVIO_EMA_BETA)
        )

        delta_2d = layer_means - all_tokens  # [L, E]

        # Vectorized bias update calculation (replaces the loop)
        with torch.no_grad():
            # Get norm factor and load_balance_coeff from the first MoE layer (assuming they are all the same)
            first_moe = moe_layers_info[0]["module"]
            norm_factor = first_moe.bias_update_norm_factor
            load_balance_coeff = first_moe.load_balance_coeff
            # Both are read from layer 0 and applied to every layer;
            # `load_balance_coeff` consistency is enforced at registration,
            # `bias_update_norm_factor` here.
            for info in moe_layers_info[1:]:
                if info["module"].bias_update_norm_factor != norm_factor:
                    raise ValueError(
                        "All MoE layers must share bias_update_norm_factor; "
                        f"layer {info['layer_id']} has "
                        f"{info['module'].bias_update_norm_factor!r}, layer 0 "
                        f"has {norm_factor!r}."
                    )

            bias_params = [info["module"].expert_bias_E for info in moe_layers_info]
            balancers = [
                getattr(info["module"].router, "quantile_balancer", None)
                for info in moe_layers_info
            ]
            if any(b is not None for b in balancers):
                if not all(b is not None for b in balancers):
                    raise ValueError(
                        "quantile_balance_bins must be set on every MoE layer "
                        "or none: the bias update is applied layer-wise from a "
                        "single stacked histogram."
                    )
                _quantile_update_expert_bias(
                    balancers, bias_params, parallel_dims, loss_mesh
                )
            else:
                updates_2d = lmo_for_moe_bias(delta_2d, norm_factor=norm_factor)

                if norm_factor.endswith("zero_mean"):
                    updates_2d = updates_2d - updates_2d.mean(dim=1, keepdim=True)

                # Collect all bias parameters and update them with a single
                # multi-tensor op
                torch._foreach_add_(
                    bias_params, list(updates_2d.unbind(0)), alpha=load_balance_coeff
                )

            # Reset router stats in bulk. NOT the aux-loss accumulators:
            # `register_aux_loss_zero_hook` owns those, and zeroing them here
            # would race it into silently reporting 0.
            torch._foreach_zero_(tok_buffers)
            torch._foreach_zero_(ent_buffers)
            torch._foreach_zero_(acc_fwd_times_buffers)

            if should_publish:
                # MaxVio_batch: worst-case overload in the current step window.
                # MaxVio_ema: the same over EMA-smoothed expert loads.
                means = layer_means.squeeze(1)  # [L]
                maxvio_batch = (all_tokens.max(dim=1).values - means) / (
                    means + MAXVIO_EPS
                )
                all_ema = torch.stack(ema_buffers)  # [L, E]
                ema_means = all_ema.mean(dim=1)
                maxvio_ema = (all_ema.max(dim=1).values - ema_means) / (
                    ema_means + MAXVIO_EPS
                )
                usage_2d = all_tokens / layer_sums.clamp_min(1.0)  # [L, E]

                # One packed D2H copy. Six separate `.cpu()` calls meant six
                # device syncs per logging step for a few KB of scalars.
                dt = usage_2d.dtype
                # Summaries are reduced HERE, on device, so they ride the same
                # single D2H copy. Doing them in the worker meant rebuilding two
                # CPU tensors per layer per step for three scalars.
                bias_2d = torch.stack(bias_params).to(dt)  # [L, E]
                packed = torch.cat(
                    [
                        usage_2d.reshape(-1),
                        bias_2d.reshape(-1),
                        all_entropies.to(dt),
                        all_load_balance_losses.to(dt),
                        maxvio_batch.to(dt),
                        maxvio_ema.to(dt),
                        bias_2d.mean(dim=1),
                        bias_2d.std(dim=1, unbiased=False),
                    ]
                ).cpu()
                n_le = num_layers * num_experts
                all_usages_cpu = packed[:n_le].tolist()
                all_biases_cpu = packed[n_le : 2 * n_le].tolist()
                off = 2 * n_le
                all_entropies_cpu = packed[off : off + num_layers].tolist()
                off += num_layers
                all_load_balance_losses_cpu = packed[off : off + num_layers].tolist()
                off += num_layers
                all_maxvio_batch_cpu = packed[off : off + num_layers].tolist()
                off += num_layers
                all_maxvio_ema_cpu = packed[off : off + num_layers].tolist()
                off += num_layers
                all_bias_mean_cpu = packed[off : off + num_layers].tolist()
                off += num_layers
                all_bias_std_cpu = packed[off : off + num_layers].tolist()
                payload = (
                    moe_layers_info,
                    all_usages_cpu,
                    all_biases_cpu,
                    all_entropies_cpu,
                    all_load_balance_losses_cpu,
                    all_maxvio_batch_cpu,
                    all_maxvio_ema_cpu,
                    all_bias_mean_cpu,
                    all_bias_std_cpu,
                    num_experts,
                    any(
                        getattr(x["module"], "aux_loss", None) is not None
                        for x in moe_layers_info
                    ),
                )
                publish_moe_metrics(payload)

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        # An MoE layer is not sufficient: with `load_balance_coeff is None`
        # the bias is None and the update would hand `torch._foreach_add_` a
        # None element. Mirrors upstream's predicate, consistency check included.
        moes = []
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if transformer_block.moe_enabled:
                    moes.append(transformer_block.moe)
        if not moes:
            return False
        enabled = moes[0].load_balance_coeff is not None
        for moe in moes[1:]:
            if (moe.load_balance_coeff is not None) != enabled:
                raise ValueError(
                    "MoE load_balance_coeff must be configured consistently "
                    "across all MoE layers. Either set it for every MoE layer "
                    "or leave it unset for all MoE layers."
                )
        return enabled

    if _should_register_moe_balancing_hook(model_parts):
        # This container reduces over the `loss` mesh only, which excludes tp,
        # so under EP+TP each rank would bias on its own shard's counts --
        # silently wrong. Refuse rather than train on it; inert at tp=1.
        if parallel_dims.ep_enabled and parallel_dims.tp > 1:
            raise NotImplementedError(
                "MoE load-balancing bias updates are not implemented for "
                f"EP + TP (ep={parallel_dims.ep}, tp={parallel_dims.tp}): the "
                "expert token counts are reduced over the 'loss' mesh only, "
                "which excludes the tp axis, so each rank would see just its "
                "own tp shard's counts. Run with tensor_parallel_degree=1, or "
                "add the dense-tp all_reduce that upstream performs."
            )
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )
        # Upstream pairs this hook with an aux-loss zero hook: it rolls each
        # `instance_acc` into the group registers `collect_aux_loss_metrics`
        # reduces, then clears them, so they cover exactly one step.
        register_aux_loss_zero_hook(optimizers, model_parts, parallel_dims)
