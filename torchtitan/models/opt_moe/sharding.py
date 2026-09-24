# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative sharding for OPT MoE.

Upstream replaced the imperative ``parallelize_module``/``ExpertParallel``
plans with ``ShardingConfig`` objects attached to each sub-config, resolved at
``Module.parallelize`` time. Root modules (embedding, final norm, lm_head) and
the per-layer norms are stock upstream modules, so they use the shared helpers.

OPT MoE's attention (``GatedNormSWAttention``) and MoE (``norm_moe.MoE``) are
custom modules, not ``GQAttention``/``common.moe.MoE``, so the upstream
per-module TP sharding helpers do not apply to them, so TENSOR parallelism is
rejected explicitly rather than silently producing a wrong plan -- see the guard
in ``set_opt_moe_sharding_config``. EXPERT parallelism is supported: the routed
experts and router take upstream's EP plans below.
"""

import dataclasses
from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    norm_config,
    set_decoder_sharding_config,
    set_gqa_inner_attention_local_spmd,
)

from torchtitan.models.common.moe_sharding import (
    _routed_experts_sharding_configs,
    _router_sharding_config,
)
from torchtitan.models.opt_moe.utils.norms import ALL_NORM_PARAM_NAMES

from torchtitan.protocols.sharding import ShardingConfig


def _cover_all_norm_params(cfg: ShardingConfig) -> ShardingConfig:
    """Extend a norm plan so it declares every norm parameter name.

    Upstream's `norm_config` / `pre_lm_head_norm_config` declare only
    ``weight``, because upstream norms are RMSNorm-with-affine. opt_moe's
    ``norm_type`` also allows ``layernorm`` (weight + bias) and ``ss_rmsnorm``
    (ssnorm_scale), and `Module._distribute_states` raises on any parameter
    with no declared placement -- so `attention_norm`, `ffn_norm` and the root
    `norm` failed on every flavor under either of those types.

    Extra names are inert: the lookup is driven by the module's ACTUAL
    parameters, so a plan naming states a given norm does not own costs
    nothing. Activation shardings are preserved untouched. Done here rather
    than in `decoder_sharding.py` so the shared helper keeps upstream's shape.
    """
    if "weight" not in cfg.state_shardings:
        raise KeyError(
            "norm sharding plan declares no 'weight'; cannot infer the "
            f"placement for the other norm parameters. Got: "
            f"{sorted(cfg.state_shardings)}"
        )
    weight_placement = cfg.state_shardings["weight"]
    state = {name: weight_placement for name in ALL_NORM_PARAM_NAMES}
    state.update(cfg.state_shardings)
    return dataclasses.replace(cfg, state_shardings=state)


if TYPE_CHECKING:
    from torchtitan.models.opt_moe.model import OPTMoEModel


def set_opt_moe_sharding_config(
    config: "OPTMoEModel.Config",
    *,
    enable_sp: bool,
    enable_tp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on the OPT MoE sub-configs that support it.

    Args:
        config: The expanded model config (``_expand_layers`` must have run).
        enable_sp: Whether sequence parallelism is on (implies TP > 1).
        enable_tp: Whether tensor parallelism is on.
        enable_ep: Whether expert parallelism is on.
    """
    if enable_tp:
        raise NotImplementedError(
            "OPT MoE on the 0.5.0 base does not carry TENSOR parallel "
            "sharding. Upstream's set_gqa_attention_sharding targets "
            "GQAttention; GatedNormSWAttention's config is flat (it builds its "
            "own Linears) and needs its own TP plan first. Run with "
            "tensor_parallel_degree=1. EP, FSDP, HSDP and CP are unaffected."
        )

    set_decoder_sharding_config(config, enable_sp=enable_sp)
    # Root norm goes through the shared helper too, so widen it the same way.
    if getattr(config.norm, "sharding_config", None) is not None:
        config.norm.sharding_config = _cover_all_norm_params(
            config.norm.sharding_config
        )

    norm = _cover_all_norm_params(norm_config(enable_sp=enable_sp))
    # `bias` is declared alongside `weight` for the same reason the norm plan
    # names every norm parameter: a declared-but-absent state is inert (the
    # lookup is driven by the module's actual parameters), while an
    # undeclared-but-present one raises. Upstream's colwise/rowwise configs
    # declare bias for exactly this reason. No flavor sets bias=True on these
    # Linears today, so this is free insurance rather than a live fix.
    _replicated_linear = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "bias": dense_param_placement(tp=spmd.I),
        }
    )
    # Norms built INSIDE a module (qk_norm, v_norm, mid_norm, embeddings_norm)
    # are not config fields, so nothing here can reach them the way
    # `attention_norm`/`ffn_norm` are reached below. Every sibling model stamps
    # its own inner norms (qwen3/sharding.py, deepseek_v3/sharding.py,
    # muse_glimmer/sharding.py); opt_moe was the only one that did not, so with
    # a PARAMETRIC norm_type (`rmsnorm`, `layernorm`, `ss_rmsnorm`) those
    # weights stayed plain tensors and `fully_shard` rejected them under
    # spmd_types. Threaded in via `norm_sharding_config`, mirroring how the
    # flat Linear configs get `linear_sharding_config`.
    #
    # One plan serves all five norm types: it names the union of their
    # parameters, and `_distribute_states` looks up only the parameters that
    # actually exist. On `np_rmsnorm`/`np_layernorm` -- which own none -- it is
    # therefore a no-op, so this is safe to apply unconditionally.
    #
    # NOTE (deliberate): this declares state_shardings ONLY, no activation
    # shardings, unlike qwen3's qk_norm plan (`models/qwen3/sharding.py`) which
    # also sets in_src/in_dst/out_src/out_dst to `attention_activation_placement()`.
    # Copying qwen3 here would be WRONG: one plan is shared by norms with
    # DIFFERENT activation layouts -- qk_norm/v_norm see a per-head layout,
    # while `embeddings_norm` sees the residual stream. Stamping a head layout
    # on the residual-stream norm would mis-declare it. Getting activations
    # right needs a per-site plan, which is only required once SP or TP is
    # enabled; both are hard-refused above, so the omission is unreachable
    # today. Do not "fix" this by pattern-matching qwen3.
    #
    # Consistency note for the fast-path argument in `build_norm_config`: that
    # helper skips stamping parameter-free norms so they stay on
    # `Module.parallelize`'s no-config path. `attention_norm`/`ffn_norm` and the
    # root `norm` below are stamped UNCONDITIONALLY instead, because they come
    # from upstream's `norm_config()`/`pre_lm_head_norm_config()` helpers, which
    # own their own placements and are config fields rather than norms this
    # model builds internally. So ~17 parameter-free modules per 8-layer model
    # do leave the fast path. That costs a forward wrapper that traces away
    # under compile; it is not free, but it is upstream's shape, not ours.
    _replicated_norm = ShardingConfig(
        state_shardings={
            name: dense_param_placement(tp=spmd.R) for name in ALL_NORM_PARAM_NAMES
        }
    )
    config.norm_sharding_config = _replicated_norm

    for layer_cfg in config.layers:
        layer_cfg.attention_norm.sharding_config = norm
        layer_cfg.ffn_norm.sharding_config = norm
        # Localize q/k/v at the kernel boundary without changing placements.
        # Upstream #4322 moved the CP k/v all-gather OUT of this local_map and
        # into the CP attention implementations themselves
        # (KVAllGatherCPFlexInnerAttention / the Ulysses variants), so this now
        # only declares the local SPMD boundary. llama3 and qwen3 do the same
        # (`<model>/sharding.py`: set_gqa_attention_sharding then
        # set_gqa_inner_attention_local_spmd).
        #
        # It goes on the INNER attention, not on GatedNormSWAttention: the inner
        # module is where q/k/v cross the kernel boundary, and `Module.parallelize`
        # recurses into children BEFORE its `_sharding_config is None` early
        # return, so the parent needing no config of its own is fine.
        set_gqa_inner_attention_local_spmd(layer_cfg.attention.inner_attention)

        # Every parameter must be a DTensor before fully_shard() runs under
        # spmd_types -- "all parameters must be DTensors on the full SPMD mesh
        # ... Got plain tensor". With TP off every Linear takes the same
        # Invariant placement, so one config serves all of them. These configs
        # are flat (they build their Linears rather than exposing them as
        # fields), so the plan is threaded in via `linear_sharding_config`.
        layer_cfg.attention.linear_sharding_config = _replicated_linear
        layer_cfg.attention.norm_sharding_config = _replicated_norm
        if getattr(layer_cfg, "feed_forward", None) is not None:
            layer_cfg.feed_forward.linear_sharding_config = _replicated_linear
            layer_cfg.feed_forward.norm_sharding_config = _replicated_norm

        # MoE layers: fully_shard rejects the grouped expert weights outright
        # ("Got plain tensor for parameter 'w1_EFD'") because they are raw 3-D
        # nn.Parameters on NormGroupedExperts rather than Linear submodules, so
        # nothing else stamps them. The router gate and shared experts need the
        # same treatment. NOTE: the config goes on `inner_experts`, NOT on
        # NormMoE itself, which owns extra buffers (load_balance_loss,
        # router_entropy, ...) that `_distribute_states` would then demand
        # placements for.
        # KNOWN GAP (CP): `rope.cache` stays a PLAIN tensor while everything
        # around it becomes a DTensor. llama3/qwen3 get its placement from
        # `set_gqa_attention_sharding`, which cannot be used here (it asserts
        # GQAttention.Config; opt_moe's attention config is flat). Declaring
        # `ShardingConfig(state_shardings={"cache": ...})` on the rope config
        # was tried and is INERT -- the config attaches but the buffer is still
        # plain after both `parallelize` and `init_weights`. The cache is a full
        # [max_context_length, dim*2] lookup table indexed by CP-sharded
        # `positions` rather than sharded itself, so this may be benign; it is
        # unverified and needs a real multi-GPU CP run to settle.
        moe_cfg = getattr(layer_cfg, "moe", None)
        if moe_cfg is not None:
            inner = moe_cfg.routed_experts.inner_experts
            inner.sharding_config = ShardingConfig(
                state_shardings={
                    name: dense_param_placement(tp=spmd.I)
                    for name in ("w1_EFD", "w2_EDF", "w3_EFD")
                }
            )
            inner.norm_sharding_config = _replicated_norm
            moe_cfg.router.gate.sharding_config = _replicated_linear
            if getattr(moe_cfg, "shared_experts", None) is not None:
                moe_cfg.shared_experts.linear_sharding_config = _replicated_linear
                moe_cfg.shared_experts.norm_sharding_config = _replicated_norm

            if enable_ep:
                # EXPERT PARALLEL. Take the routed-expert plan from upstream's
                # own helpers rather than re-deriving it: `w1_EFD/w2_EDF/w3_EFD`
                # are the same parameter names upstream's GroupedExperts uses.
                #
                # Only the ROUTED-expert pieces come from upstream. Two parts of
                # `set_moe_sharding_config` are deliberately not used:
                #   * `shared_experts.w1/.w2/.w3` -- norm_moe's shared_experts
                #     config is FLAT (it builds its Linears internally), so it
                #     has no `.w1`; it goes through `linear_sharding_config`
                #     above instead. Calling upstream's helper raises
                #     AttributeError: 'Config' object has no attribute 'w1'.
                #   * `moe_cfg.sharding_config` -- NormMoE owns buffers upstream's
                #     MoE does not (load_balance_loss, router_entropy,
                #     tokens_per_expert_E, expert_bias_E, acc_fwd_times,
                #     tokens_per_expert_cumul). Declaring a plan on the wrapper
                #     makes `_distribute_states` demand a placement for each of
                #     them.
                routed_cfg, inner_cfg = _routed_experts_sharding_configs(
                    enable_ep=True,
                    enable_sp=enable_sp,
                )
                moe_cfg.routed_experts.sharding_config = routed_cfg
                inner.sharding_config = inner_cfg
                # The router MODULE gets the token redistribution plan; the gate
                # keeps the replicated-weight plan set above (upstream's
                # `_router_gate_sharding_config` at tp=1).
                moe_cfg.router.sharding_config = _router_sharding_config(
                    enable_ep=True, enable_sp=enable_sp
                )
