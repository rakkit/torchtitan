# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ``ShardingConfig`` constructors of OPT MoE's own components, used by
``flavors.py`` where each component config is built. Tensor parallelism is not
supported (``build_model_config`` refuses it): the gate, the convs, the FFN /
shared-expert linears and the router gate are replicated over TP. Expert
parallelism takes upstream's routed-expert plans."""

import dataclasses

import spmd_types as spmd

from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe.sharding import routed_expert_linear_sharding_config


def dense_linear_sharding_config() -> ShardingConfig:
    """opt_moe's own dense linears and convs (the attention gate, the q / k / v
    convs, the FFN and shared-expert linears, the router gate): replicated over
    TP. Every parameter must be a DTensor before fully_shard() under spmd_types;
    ``bias`` is declared alongside ``weight`` (inert when absent)."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "bias": dense_param_placement(tp=spmd.I),
        }
    )


def routed_expert_weight_sharding_config(*, enable_ep: bool) -> ShardingConfig:
    """The routed experts' grouped linears: upstream's EP plan under EP, else
    replicated over TP."""
    if enable_ep:
        return routed_expert_linear_sharding_config(enable_ep=True)
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.I)})


def attention_sinks_state_sharding(base: ShardingConfig | None) -> ShardingConfig:
    """The attention's own plan with the sinks (one logit per head, a direct
    parameter of the attention) declared, replicated over TP."""
    sinks = {"sinks": dense_param_placement(tp=spmd.I)}
    if base is None:
        return ShardingConfig(state_shardings=sinks)
    return dataclasses.replace(base, state_shardings={**base.state_shardings, **sinks})
