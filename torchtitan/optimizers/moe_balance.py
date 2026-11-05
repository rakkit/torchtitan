# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OPT MoE load balancing: the expert-bias update and its metrics.

Registered as an optimizer step pre-hook by ``OPTMoEModel`` for any MoE
model. Every step: the expert statistics are reduced, the expert bias (when the
layers have one) is updated and the counters are reset; on the steps the
container arms, the metrics ride one packed D2H copy and are published per
layer.
"""

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.utils import fsdp_shard_mesh
from torchtitan.models.common.moe import MoE

if TYPE_CHECKING:
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

__all__ = ["moe_layers", "publish_moe_metrics", "register_moe_load_balancing_hook"]

logger = logging.getLogger(__name__)

MAXVIO_EMA_BETA = 0.995
MAXVIO_EPS = 1e-12


def moe_layers(model_parts: list[nn.Module]) -> Iterator[tuple[int, MoE]]:
    """(layer id, MoE) of every block whose feed-forward is an MoE."""
    for part in model_parts:
        layers = getattr(part, "layers", None)
        if layers is None:
            continue
        for layer_key, block in layers.items():
            if isinstance(getattr(block, "feed_forward", None), MoE):
                yield int(layer_key), block.feed_forward


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
            has_bias,
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
            }
            if has_bias:
                metrics[f"moe_bias_mean/L-{layer_id}"] = all_bias_mean_cpu[i]
                metrics[f"moe_bias_std/L-{layer_id}"] = all_bias_std_cpu[i]
            if has_aux_loss:
                metrics[
                    f"moe_load_balance_loss/L-{layer_id}"
                ] = all_load_balance_losses_cpu[i]
            pre_usage = f"moe_ep_usage/L-{layer_id}_EP-"
            pre_bias = f"moe_bias/L-{layer_id}_EP-"
            metrics.update({f"{pre_usage}{j}": v for j, v in enumerate(layer_usages)})
            if has_bias:
                metrics.update(
                    {f"{pre_bias}{j}": v for j, v in enumerate(layer_biases)}
                )
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
    parallelism_context,
    all_tokens: torch.Tensor,
    all_entropies: torch.Tensor,
    all_load_balance_losses: torch.Tensor,
):
    loss_mesh = parallelism_context.get_optional_mesh("loss")
    if loss_mesh is None:
        return

    # 1. Determine Topology
    fsdp_mesh = fsdp_shard_mesh(parallelism_context)
    dp_mesh = parallelism_context.get_optional_mesh("dp_replicate")

    # Hierarchical only under HSDP (both meshes present); the flat path is
    # mathematically equivalent -- same rank set, same normalisation -- so the
    # choice is a communication-cost one and needs no knob.
    use_hierarchical = (fsdp_mesh is not None) and (dp_mesh is not None)

    # 2. Fuse & Pack in fp32: the expert counts are integers, exact up to 2^24
    # per expert and step; the entropy sum is a metric.
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
    n0, n1, n2 = t0.numel(), t1.numel(), t2.numel()

    # Slicing views
    out_tokens = buf[0:n0].view_as(all_tokens)
    out_ent = buf[n0 : n0 + n1].view_as(all_entropies)
    out_lb = buf[n0 + n1 : n0 + n1 + n2].view_as(all_load_balance_losses)

    # 5. Copy back to inputs
    # Tokens: SUM (no division)
    all_tokens.copy_(out_tokens.to(all_tokens.dtype))

    # Entropy: SUM (the caller divides by the summed counted tokens).
    all_entropies.copy_(out_ent.to(all_entropies.dtype))
    # Load-balance loss: SUM, matching upstream's `collect_aux_loss_metrics`.
    # Each rank holds its own token share, so the sum IS the global value.
    all_load_balance_losses.copy_(out_lb.to(all_load_balance_losses.dtype))


@torch.no_grad()
def _quantile_update_expert_bias(
    balancers: list,
    bias_params: list[torch.Tensor],
    parallelism_context: ParallelismContext,
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
    if parallelism_context.ep_enabled and parallelism_context.tp > 1:
        reduction_groups.append(parallelism_context.get_dense_tp_mesh().get_group())
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
    parallelism_context: ParallelismContext,
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
        parallelism_context: ParallelismContext,
    ):
        """The step's expert statistics (every step), the expert-bias update
        (when the layers have a bias) and, on armed steps, the metrics. All MoE
        layers have the same expert count."""

        loss_mesh = parallelism_context.get_optional_mesh("loss")

        # The `loss` mesh excludes tp, so loss-rank 0 is true on EVERY tp rank
        # and each would push a payload for the same layers. Require tp local
        # rank 0 too, as `components/metrics.py` does. Inert at tp=1.
        tp_mesh = parallelism_context.get_optional_mesh("tp")
        is_dp_rank_0 = (loss_mesh is None or loss_mesh.get_local_rank() == 0) and (
            tp_mesh is None or tp_mesh.get_local_rank() == 0
        )
        # The metrics ride one D2H copy, so they are produced only on steps the
        # container will collect; every other step only updates and resets.
        should_publish = optimizers.consume_moe_publish_arm() and is_dp_rank_0
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.

        moe_layers_info = []
        tok_buffers, ent_buffers, load_balance_loss_buffers = [], [], []
        ema_buffers = []
        top_ks = []
        num_experts = 0

        for layer_id, moe in moe_layers(model_parts):
            layer_num_experts = moe.router.tokens_per_expert_E.numel()
            if num_experts and layer_num_experts != num_experts:
                raise ValueError(
                    "All MoE layers must have the same expert count: "
                    f"layer {layer_id} has {layer_num_experts}, an "
                    f"earlier layer has {num_experts}. The stats below are "
                    "packed with `.view(num_layers, num_experts)`, which "
                    "would silently mis-attribute them."
                )
            num_experts = layer_num_experts
            moe_layers_info.append({"module": moe, "layer_id": layer_id})
            tok_buffers.append(moe.router.tokens_per_expert_E)
            ema_buffers.append(moe.tokens_per_expert_cumul)
            ent_buffers.append(moe.router_entropy)
            top_ks.append(moe.router.top_k)
            # Per-layer load-balance value: upstream's AuxLoss keeps a
            # per-instance accumulator, so the `moe_load_balance_loss/L-*`
            # breakdown survives. Zero scalar when the layer has none.
            aux = moe.router.aux_loss
            load_balance_loss_buffers.append(
                aux.instance_acc.reshape(1)
                if aux is not None
                else torch.zeros(
                    1, dtype=torch.float32, device=moe.router_entropy.device
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

        # The step's sums over every forward and counted token (upstream's
        # counts: no per-forward normalisation; the bias update and the ratios
        # below do not depend on the scale). AuxLoss.instance_acc already
        # carries `raw_sum / global_valid_tokens` for the whole step.
        if loss_mesh is not None:
            fused_hier_reduce_loss_stats(
                parallelism_context, all_tokens, all_entropies, all_load_balance_losses
            )

        layer_sums = all_tokens.sum(dim=1, keepdim=True)  # [L, 1]
        layer_means = layer_sums / num_experts  # [L, 1]
        # The mean gate entropy per counted token: each token is counted top_k
        # times in tokens_per_expert_E.
        counted_tokens = layer_sums.squeeze(1) / torch.tensor(
            top_ks, dtype=layer_sums.dtype, device=layer_sums.device
        )
        all_entropies = all_entropies / counted_tokens.clamp_min(1.0)

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
            # The bias is on for every layer or none (checked at registration).
            has_bias = bias_params[0] is not None
            balancers = [
                getattr(info["module"].router, "quantile_balancer", None)
                for info in moe_layers_info
            ]
            quantile = any(b is not None for b in balancers)
            if quantile and not all(b is not None for b in balancers):
                raise ValueError(
                    "quantile_balance_bins must be set on every MoE layer "
                    "or none: the bias update is applied layer-wise from a "
                    "single stacked histogram."
                )
            # No bias (load_balance_coeff None): statistics and metrics only.
            if has_bias and quantile:
                _quantile_update_expert_bias(
                    balancers, bias_params, parallelism_context, loss_mesh
                )
            elif has_bias:
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
                bias_2d = (
                    torch.stack(bias_params).to(dt)
                    if has_bias
                    else torch.zeros_like(usage_2d)
                )  # [L, E]
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
                        x["module"].router.aux_loss is not None for x in moe_layers_info
                    ),
                    has_bias,
                )
                publish_moe_metrics(payload)

    # The hook runs for any MoE model: the statistics and metrics whatever the
    # balancing (bias update, quantile solve, aux loss only, none); the bias
    # update only when the layers have a bias (load_balance_coeff set).
    moes = [moe for _, moe in moe_layers(model_parts)]
    if moes:
        enabled = moes[0].load_balance_coeff is not None
        if any((moe.load_balance_coeff is not None) != enabled for moe in moes):
            raise ValueError(
                "MoE load_balance_coeff must be configured consistently "
                "across all MoE layers. Either set it for every MoE layer "
                "or leave it unset for all MoE layers."
            )
        # This container reduces over the `loss` mesh only, which excludes tp,
        # so under EP+TP each rank would see its own shard's counts --
        # silently wrong. Refuse rather than train on it; inert at tp=1.
        if parallelism_context.ep_enabled and parallelism_context.tp > 1:
            raise NotImplementedError(
                "MoE load-balancing statistics are not implemented for "
                f"EP + TP (ep={parallelism_context.ep}, tp={parallelism_context.tp}): the "
                "expert token counts are reduced over the 'loss' mesh only, "
                "which excludes the tp axis, so each rank would see just its "
                "own tp shard's counts. Run with tensor_parallel_degree=1, or "
                "add the dense-tp all_reduce that upstream performs."
            )
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallelism_context=parallelism_context
            )
        )
