# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# adapted from
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/llama4/optimizer.py

import torch
import torch.nn as nn

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import build_optimizers, OptimizersContainer
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


def lmo_for_moe_bias(
    g,
    norm_factor="sign",
):
    if norm_factor == "sign":
        g = torch.sign(g)
    elif norm_factor == "spectral":
        g = g / torch.sqrt(g.pow(2).sum())
    return g


def need_rescale_stats(module):
    return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT


# for MoE auxiliary-loss-free load balancing
def _update_expert_bias(
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
):
    # we update the
    # 1. expert bias [load balancing, affect the training]
    # 2. expert usage count [for log only]
    # 3. router entropy [for log only]
    dp_cp_mesh = (
        parallel_dims.world_mesh["dp_cp"] if parallel_dims.dp_cp_enabled else None
    )
    # # TODO: Currently this sync is blocking (thus exposed) and happens on the
    # # default compute stream. Need to assess if this is OK performance-wise.
    # for model_part in model_parts:
    #     for transformer_block in model_part.layers.values():
    #         if transformer_block.moe_enabled:
    #             moe = transformer_block.feed_forward

    #             if dp_cp_mesh is not None:
    #                 # [All-reduce]
    #                 torch.distributed.all_reduce(
    #                     moe.tokens_per_expert, group=dp_cp_mesh.get_group()
    #                 )
    #                 torch.distributed.all_reduce(
    #                     moe.router_entropy,
    #                     group=dp_cp_mesh.get_group(),
    #                     op=torch.distributed.ReduceOp.AVG,
    #                 )

    #             moe._log_expert_metrics = {}

    #             with torch.no_grad():
    #                 # TODO(JSC): here we have options to use Row-wise norm rather than Sign
    #                 # according to the Scion paper.
    #                 delta = moe.tokens_per_expert.mean() - moe.tokens_per_expert
    #                 update = lmo_for_moe_bias(
    #                     delta, norm_factor=moe.bias_update_norm_factor
    #                 )
    #                 moe.expert_bias.add_(moe.bias_update_speed * update)

    #             layer_id = transformer_block.layer_id
    #             counts = moe.tokens_per_expert

    #             total = sum(counts) or 1.0
    #             base = f"moe_ep_usage/L-{layer_id}_EP-"
    #             # build and merge both “count” and “share” entries without explicit loops
    #             moe._log_expert_metrics.update(
    #                 {f"{base}{i}": cnt / total for i, cnt in enumerate(counts)}
    #             )

    #             base_name = f"moe_bias/L-{layer_id}_EP-"
    #             moe._log_expert_metrics.update(
    #                 {
    #                     f"{base_name}{i}": moe.expert_bias[i].mean()
    #                     for i in range(moe.expert_bias.shape[0])
    #                 }
    #             )

    #             moe._log_expert_metrics.update(
    #                 {f"moe_entropy/L-{layer_id}": moe.router_entropy.mean()}
    #             )

    #             moe.tokens_per_expert.zero_()
    #             moe.router_entropy.zero_()
    #             # update_norms = calculate_norm(update)
    #             # param_norms = calculate_norm(moe.expert_bias)

    # above is adapted from the upstream code
    # below is the optimized version that only uses 2 all_reduce calls
    tok_buf, ent_buf, sizes = [], [], []  # sizes = #experts per layer
    for part in model_parts:
        for block in part.layers.values():
            if not block.moe_enabled:
                continue
            moe = block.feed_forward

            tok_buf.append(moe.tokens_per_expert)  # vector (int32)
            ent_buf.append(moe.router_entropy)  # scalar (float32)
            sizes.append(tok_buf[-1].numel())

            if need_rescale_stats(moe) or need_rescale_stats(block):
                # because of the re-computation, these data might be added twice
                # dont have very good way to handle this, so this take this ad-hoc approach
                # remember to check both moe and block, incases of its OP-LEVEL checkpointing
                tok_buf[-1] /= 2
                ent_buf[-1] /= 2

    flat_tok = torch.cat(tok_buf, 0)  # int32
    flat_ent = torch.stack(ent_buf, 0)  # float32

    if dp_cp_mesh is not None:
        pg = dp_cp_mesh.get_group()
        torch.distributed.all_reduce(
            flat_tok, group=pg, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            flat_ent, group=pg, op=torch.distributed.ReduceOp.AVG
        )

    layer_ptr = 0  # index into sizes / flat_ent
    tok_ptr = 0  # offset into flat_tok
    with torch.no_grad():
        for part in model_parts:
            for block in part.layers.values():
                if not block.moe_enabled:
                    continue
                moe = block.feed_forward
                moe._log_expert_metrics = {}

                n_expert = sizes[layer_ptr]
                tokens = flat_tok[tok_ptr : tok_ptr + n_expert].float()
                entropy = flat_ent[layer_ptr]
                layer_ptr += 1
                tok_ptr += n_expert

                delta = tokens.mean() - tokens
                # TODO(JSC): here we have options to use Row-wise norm rather than Sign
                update = lmo_for_moe_bias(
                    delta, norm_factor=moe.bias_update_norm_factor
                )
                moe.expert_bias.add_(moe.bias_update_speed * update)

                total = tokens.sum().clamp(min=1.0)
                layer_id = block.layer_id
                moe._log_expert_metrics.update(
                    {
                        f"moe_ep_usage/L-{layer_id}_EP-{i}": c / total
                        for i, c in enumerate(tokens)
                    },
                    **{
                        f"moe_bias/L-{layer_id}_EP-{i}": moe.expert_bias[i].mean()
                        for i in range(moe.expert_bias.shape[0])
                    },
                    # **{f"moe_entropy/L-{layer_id}": entropy},
                    **{f"moe_entropy/moe_entropy_per_layer_{layer_id}": entropy},
                )

                moe.tokens_per_expert.zero_()
                moe.router_entropy.zero_()


def build_moe_optimizers(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    optimizers = build_optimizers(
        model_parts=model_parts,
        job_config=job_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )

    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: _update_expert_bias(
            model_parts, parallel_dims=parallel_dims
        )
    )

    return optimizers
