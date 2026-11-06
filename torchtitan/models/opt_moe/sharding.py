# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ``ShardingConfig`` constructors of OPT MoE's own components, used by
``flavors.py`` where each component config is built; everything else takes
upstream's (``decoder_sharding``, ``moe.sharding``). Tensor parallelism shards
the attention by KV group: wqkv (rows grouped per KV group), the gate, the k /
v convs and the sinks by head, wo by its input; the dense and shared-expert
FFNs column / row; the MoE as upstream's (tokens sequence-sharded over TP for
the router and the experts, EP >= TP)."""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    residual_activation_placement,
    token_id_placement,
)
from torchtitan.models.common.moe.sharding import moe_sharding_config

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def attention_sharding_config(*, enable_sp: bool, sinks: bool) -> ShardingConfig:
    """The attention boundary: ``x_TD`` in and out in the residual layout (the
    forward gathers ``x_TD`` once for wqkv and the gate); the sinks, one logit
    per head, sharded with the heads."""
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings=(
            {"sinks": dense_param_placement(tp=spmd.S(0))} if sinks else {}
        ),
        in_src_shardings={"x_TD": layout},
        out_src_shardings=layout,
    )


def conv_weight_sharding_config() -> ShardingConfig:
    """A k / v conv weight ``[G K, 1, W]``: depthwise, sharded on its channels
    (whole KV groups, as wqkv's k / v rows)."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))}
    )


def opt_moe_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Upstream's MoE boundary plus ``OptMoE``'s statistics buffers: the gate
    entropy, a partial sum over the tokens each rank routes (every data axis
    and TP, as the router's ``tokens_per_expert_E``), and the balance hook's
    running count of the reduced counts (the same on every rank)."""
    config = moe_sharding_config(enable_sp=enable_sp)
    config.state_shardings.update(
        router_entropy=SpmdType({DP: spmd.P, CP: spmd.P, TP: spmd.P}),
        tokens_per_expert_cumul=dense_param_placement(tp=spmd.R),
    )
    return config


def mtp_projection_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """The MTP ``eh_proj``: an unsharded weight over residual-layout activations
    (DeepSeek-V3's); as a norm weight: R under SP (each rank's tokens give a
    partial gradient), else I (every rank computes the full gradient)."""
    activation = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
    )


def mtp_block_sharding_config(*, enable_sp: bool) -> ShardingConfig | None:
    """The MTP block boundary: under SP the validity mask (token ids' layout)
    is sequence-sharded in the forward, as the hidden state it masks
    (DeepSeek-V3's)."""
    if not enable_sp:
        return None
    return ShardingConfig(
        in_src_shardings={"mtp_input_valid_mask_T": token_id_placement()}
    )
