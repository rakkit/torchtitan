# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OPT MoE model flavors.

``_opt_moe_config`` writes the complete, sharded ``OPTMoEModel.Config`` (the
embedding, final norm and head; the blocks from ``build_opt_moe_layers``); the
defaults of every setting are in these two functions. A flavor is the settings
it changes; a recipe's ``build_model_config`` keywords replace the flavor's
setting of the same name (a name neither function takes raises)::

    build_model_config(
        "production-moe-336m", parallelism=parallelism, seq_len=4096,
        norm_eps=1e-30, wo_init_std=4.0,
    )

The recipe's keywords are kept in ``config.build_args``, so
``build_model_config(flavor, **build_args)`` rebuilds the same model for
another parallelism (evaluation, checkpoint conversion).
"""

import copy
import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Literal

from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.activation import Sigmoid, Softmax, SqrtSoftplus
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import (
    fused_gate_up_param_init,
    fused_grouped_gate_up_param_init,
    get_attention_config,
    make_gqa_config,
    make_shared_expert_down_proj_config,
)
from torchtitan.models.common.decoder_sharding import (
    feed_forward_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_input_colwise_sharding_config,
    residual_activation_placement,
    rowwise_sharding_config,
    stacked_colwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import (
    BatchWiseLoadBalanceLoss,
    MicrobatchWiseLoadBalanceLoss,
    QuantileBalancedTopKRouter,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.moe.sharding import (
    routed_expert_linear_sharding_config,
    routed_experts_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
    shared_experts_sharding_config,
)
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm
from torchtitan.models.common.rope import CosSinRoPE, RoPE
from .deepgemm_experts import DEEPGEMM_ALIGN, DeepGEMMGroupedLinear

from .gated_swa_attention import GatedSWAttention
from .model import layout_flags, OPTMoEModel, OPTMoEMTPBlock, OPTMoETransformerBlock
from .moe import (
    GatherTokenDispatcher,
    OptMoE,
    RowTopKQuantileBalancedRouter,
    RowTopKRouter,
)
from .param_init import (
    conv_init_fn,
    make_expert_param_init,
    make_param_init,
    setup_depth_init,
    setup_residual_scale,
)
from .sharding import (
    attention_sharding_config,
    conv_weight_sharding_config,
    mtp_block_sharding_config,
    mtp_projection_sharding_config,
    opt_moe_sharding_config,
)

__all__ = ["MODEL_FLAVORS", "build_model_config", "build_opt_moe_layers"]

_SCORE_FUNCS = {
    "sigmoid": Sigmoid.Config,
    "softmax": Softmax.Config,
    "sqrtsoftplus": SqrtSoftplus.Config,
}


def _norm(
    norm_type: Literal["rmsnorm", "np_rmsnorm"],
    dim: int,
    eps: float,
    *,
    sharding_config: ShardingConfig | None = None,
) -> RMSNorm.Config:
    """Upstream's RMSNorm: with learnable gains ("rmsnorm") or without ("np_rmsnorm")."""
    if norm_type not in ("rmsnorm", "np_rmsnorm"):
        raise ValueError(f"norm_type must be rmsnorm | np_rmsnorm, got {norm_type!r}")
    return RMSNorm.Config(
        normalized_shape=dim,
        eps=eps,
        elementwise_affine=norm_type == "rmsnorm",
        sharding_config=sharding_config,
    )


def _ffn(
    *,
    dim: int,
    hidden_dim: int,
    w1_init: Callable,
    w2_init: Callable,
    w3_init: Callable,
    shared_experts: bool,
    enable_sp: bool,
) -> DenseFeedForward.Config:
    """A SwiGLU FFN (upstream's ``DenseFeedForward``, gate and up in one
    ``[2, F, D]`` ``w13``) with one initializer per projection, sharded as
    upstream's ``make_ffn_config`` / ``make_shared_expert_ffn_config``
    (``shared_experts``: an MoE's shared experts, whose output the MoE reduces
    with the routed one)."""
    x_layout = residual_activation_placement(enable_sp=enable_sp)
    w13 = ColumnParallelLinear.Config(
        in_features=dim,
        out_features=hidden_dim,
        num_linears=2,
        param_init=fused_gate_up_param_init({"weight": w1_init}, {"weight": w3_init}),
        sharding_config=stacked_colwise_sharding_config(input_layout=x_layout),
    )
    if shared_experts:
        return DenseFeedForward.Config(
            w13=w13,
            w2=make_shared_expert_down_proj_config(
                in_features=hidden_dim,
                out_features=dim,
                param_init={"weight": w2_init},
                enable_sp=enable_sp,
            ),
            sharding_config=shared_experts_sharding_config(enable_sp=enable_sp),
        )
    return DenseFeedForward.Config(
        w13=w13,
        w2=RowParallelLinear.Config(
            in_features=hidden_dim,
            out_features=dim,
            param_init={"weight": w2_init},
            sharding_config=rowwise_sharding_config(output_layout=x_layout),
        ),
        sharding_config=feed_forward_sharding_config(enable_sp=enable_sp),
    )


def _aux_loss_config(aux_loss: dict[str, Any], num_experts: int):
    """Upstream's auxiliary load-balance loss from
    ``dict(loss_type="microbatch_wise" | "batch_wise", coeff=...)``."""
    args = dict(aux_loss)
    loss_type = args.pop("loss_type", None)
    if loss_type == "microbatch_wise":
        return MicrobatchWiseLoadBalanceLoss.Config(**args)
    if loss_type == "batch_wise":
        return BatchWiseLoadBalanceLoss.Config(**args, num_experts=num_experts)
    raise ValueError(
        f"aux_loss needs loss_type 'microbatch_wise' or 'batch_wise', got {loss_type!r}"
    )


def _layer_pattern(
    pattern: str | None, n_layers: int, on: str, off: str, *, default: bool
) -> list[bool]:
    """A per-layer pattern, one character per layer (e.g. "SSSF"): ``on`` /
    ``off`` per layer, ``default`` everywhere when None."""
    if pattern is None:
        return [default] * n_layers
    pattern = pattern.upper()
    if len(pattern) != n_layers or set(pattern) - {on, off}:
        raise ValueError(
            f"pattern {pattern!r} must be {n_layers} characters of {on!r} / {off!r}"
        )
    return [c == on for c in pattern]


def build_opt_moe_layers(
    *,
    n_layers: int,
    dim: int,
    norm_type: Literal["rmsnorm", "np_rmsnorm"],
    norm_eps: float,
    prenorm_gain_init: float = 1.0,
    residual_scale: str = "identity",
    depth_init: str = "total_depth_R",
    # attention
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    qk_norm: bool,
    attn_backend: Literal["flex", "flex_flash", "varlen"],
    rope: RoPE.Config,
    rope_of_swa: RoPE.Config | None = None,
    rope_pattern: str | None = None,
    sliding_window_size: int | None = None,
    swa_pattern: str | None = None,
    gated_attention_type: Literal["element-wise"] | None = None,
    conv_kernel_size: int = 0,
    attention_sinks: bool = False,
    wqkv_init_fn_type: str = "scaled_orthogonal",
    wqkv_init_std: float = 1.0,
    wo_init_fn_type: str = "scaled_orthogonal",
    wo_init_std: float = 1.0,
    w_gate_init_fn_type: str = "scaled_orthogonal",
    w_gate_init_std: float = 1.0,
    conv_init_fn_type: str = "delay",
    conv_init_gain: float = 4.0,
    sinks_init_fn_type: str = "zeros",
    sinks_init_std: float = 0.0,
    # feed-forward
    n_dense_layers: int,
    dense_hidden_dim: int | None = None,
    moe_hidden_dim: int | None = None,
    num_experts: int | None = None,
    num_shared_experts: int = 1,
    top_k: int | None = None,
    route_scale: float = 1.0,
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid",
    load_balance_coeff: float | None = 1e-3,
    bias_update_norm_factor: str = "sign",
    aux_loss: dict[str, Any] | None = None,
    quantile_balance_bins: int | None = None,
    track_router_metrics: bool = True,
    token_dispatch: Literal["scatter_add", "gather"] = "gather",
    expert_gemm: Literal["torch", "deepgemm"] = "torch",
    router_topk: Literal["torch", "triton"] = "triton",
    w1_init_fn_type: str = "scaled_orthogonal",
    w1_init_std: float = 1.0,
    w2_init_fn_type: str = "scaled_orthogonal",
    w2_init_std: float = 1.0,
    w3_init_fn_type: str = "scaled_orthogonal",
    w3_init_std: float = 1.0,
    init_gate_as_residual: bool = False,
    router_init_fn_type: str = "scion_normal_output",
    router_init_std: float = 1.0,
    num_mtp_layers: int = 0,
    enable_tp: bool,
    enable_sp: bool,
    enable_ep: bool,
    enable_cp: bool,
) -> list[OPTMoETransformerBlock.Config]:
    """The sharded per-layer ``OPTMoETransformerBlock`` configs: the
    ``n_layers`` decoder layers, then ``num_mtp_layers`` blocks for the MTP
    depths, each a mirror of the last layer (its attention, which must be full
    attention, its feed-forward / MoE and its inits, depth-init divisors
    included) with its own expert seeds (layer id ``n_layers + i``).

    Norms and residuals: ``norm_type`` / ``norm_eps`` for the q/k and block
    norms, ``prenorm_gain_init`` the block norms' starting gain;
    ``residual_scale`` / ``depth_init`` the residual scales and the per-layer
    init divisors of wo, the gate and the FFN / expert w2 (param_init.py).

    Attention: ``swa_pattern`` ("S" sliding-window / "F" full per layer; None:
    every layer sliding-window when ``sliding_window_size`` is set) and
    ``rope_pattern`` ("R" RoPE / "N" NoPE per layer; None: RoPE everywhere) pick
    each layer's; sliding-window layers take ``rope_of_swa`` when set.
    ``gated_attention_type`` "element-wise": the output times
    ``sigmoid(gate_proj(x))``; ``conv_kernel_size`` > 0: the Canon short conv
    ``x + conv(x)`` on k and v, each channel initialized by ``conv_init_fn_type`` ("delay",
    "random_opnorm", "torch_default") at ``conv_init_gain``;
    ``attention_sinks``: one learned logit per head in the softmax (the gate or
    sinks: one or neither).

    Feed-forward: layers ``< n_dense_layers`` a dense SwiGLU FFN of
    ``dense_hidden_dim``, the rest an ``OptMoE``: ``num_experts`` routed experts
    of ``moe_hidden_dim`` (each drawn with its own seed) and
    ``num_shared_experts`` shared ones as one FFN; upstream's top-k router with
    the top-k weights renormalized and scaled by ``route_scale``;
    ``load_balance_coeff`` the expert-bias update rate (None: no bias),
    ``quantile_balance_bins`` upstream's quantile-balanced router instead of the
    sign nudge, ``aux_loss`` upstream's auxiliary load-balance loss
    (``dict(loss_type="microbatch_wise" | "batch_wise", coeff=...)``). One
    w1 / w2 / w3 init for the dense FFN, the routed and the shared experts; w3
    takes the depth divisor too with ``init_gate_as_residual``. The token
    dispatcher: all-to-all under EP, local dispatch without (DeepEP / HybridEP:
    upstream's ``TokenDispatcherTransform`` on the recipe).

    ``enable_cp``: the attention runs under Context Parallel (all-gather K/V,
    flex or varlen).
    """
    if gated_attention_type not in (None, "element-wise"):
        raise ValueError(
            "gated_attention_type must be None (no gate) or 'element-wise', "
            f"got {gated_attention_type!r}"
        )
    use_swa = _layer_pattern(
        swa_pattern, n_layers, "S", "F", default=sliding_window_size is not None
    )
    if any(use_swa) and sliding_window_size is None:
        raise ValueError("sliding-window layers need sliding_window_size")
    use_rope = _layer_pattern(rope_pattern, n_layers, "R", "N", default=True)
    if n_dense_layers > 0 and dense_hidden_dim is None:
        raise ValueError(f"n_dense_layers={n_dense_layers} needs dense_hidden_dim")
    if n_dense_layers < n_layers and None in (moe_hidden_dim, num_experts, top_k):
        raise ValueError(
            f"layers {n_dense_layers}..{n_layers - 1} are MoE layers: they need "
            "moe_hidden_dim, num_experts and top_k"
        )
    if prenorm_gain_init != 1.0 and norm_type != "rmsnorm":
        raise ValueError(
            "prenorm_gain_init != 1 needs norm_type='rmsnorm' "
            f"(got {norm_type!r}): no gain to initialise"
        )
    block_scale, identity_scale = setup_residual_scale(residual_scale, n_layers)
    block_norm_sharding = norm_sharding_config(enable_sp=enable_sp)

    def block_norm() -> RMSNorm.Config:
        cfg = _norm(norm_type, dim, norm_eps, sharding_config=block_norm_sharding)
        if prenorm_gain_init != 1.0:
            cfg.param_init = {
                "weight": functools.partial(nn.init.constant_, val=prenorm_gain_init)
            }
        return cfg

    def conv(channels: int) -> Conv1d.Config:
        return Conv1d.Config(
            in_channels=channels,
            out_channels=channels,
            kernel_size=conv_kernel_size,
            groups=channels,
            bias=False,
            param_init={"weight": conv_init_fn(conv_init_fn_type, conv_init_gain)},
            sharding_config=conv_weight_sharding_config(),
        )

    if num_mtp_layers > 0 and use_swa[-1]:
        raise ValueError(
            "the MTP block mirrors the last layer, which must be full attention: "
            f"swa_pattern ends in S ({swa_pattern!r})"
        )
    layers = []
    for layer_id in range(n_layers + num_mtp_layers):
        # The layer whose settings this block takes: itself, or for an MTP
        # block the last decoder layer (the expert seeds stay layer_id's).
        mirror = min(layer_id, n_layers - 1)
        div_attn, div_ffn = setup_depth_init(depth_init, mirror, n_layers)
        w3_div = div_ffn if init_gate_as_residual else 1.0

        inner = get_attention_config(attn_backend)
        if use_swa[mirror]:
            if isinstance(inner, FlexInnerAttention.Config):
                inner = SlidingWindowFlexInnerAttention.Config(
                    block_size=inner.block_size,
                    kernel_options=inner.kernel_options,
                    sliding_window_size=sliding_window_size,
                )
            else:
                assert isinstance(inner, VarlenInnerAttention.Config)
                inner = SlidingWindowVarlenInnerAttention.Config(
                    **{
                        f.name: getattr(inner, f.name)
                        for f in dataclasses.fields(inner)
                    },
                    sliding_window_size=sliding_window_size,
                )
        layer_rope = (
            rope_of_swa if use_swa[mirror] and rope_of_swa is not None else rope
        )
        gqa = make_gqa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            wqkv_param_init={
                "weight": make_param_init(wqkv_init_fn_type, wqkv_init_std)
            },
            wo_param_init={
                "weight": make_param_init(wo_init_fn_type, wo_init_std, div_attn)
            },
            inner_attention=inner,
            rope=layer_rope if use_rope[mirror] else None,
            qk_norm=_norm(norm_type, head_dim, norm_eps) if qk_norm else None,
            enable_sp=enable_sp,
        )
        attention = {f.name: getattr(gqa, f.name) for f in dataclasses.fields(gqa)}
        # The attention gathers its input once for wqkv and the gate (TP): both
        # are plain Linears on the gathered input (a ColumnParallelLinear would
        # gather again: twice the tokens under SP, its backward all-reduce twice
        # without).
        wqkv = gqa.qkv_linear.wqkv
        attention["qkv_linear"] = dataclasses.replace(
            gqa.qkv_linear,
            wqkv=Linear.Config(
                **{
                    f.name: getattr(wqkv, f.name)
                    for f in dataclasses.fields(Linear.Config)
                    if f.name != "sharding_config"
                },
                sharding_config=replicated_input_colwise_sharding_config(),
            ),
        )
        attention["sharding_config"] = attention_sharding_config(
            enable_sp=enable_sp, sinks=attention_sinks
        )
        if gated_attention_type == "element-wise":
            attention["gate_proj"] = Linear.Config(
                in_features=dim,
                out_features=n_heads * head_dim,
                param_init={
                    "weight": make_param_init(
                        w_gate_init_fn_type, w_gate_init_std, div_attn
                    )
                },
                sharding_config=replicated_input_colwise_sharding_config(),
            )
        if conv_kernel_size > 0:
            attention["k_conv"] = conv(n_kv_heads * head_dim)
            attention["v_conv"] = conv(n_kv_heads * head_dim)
        if attention_sinks:
            attention["attention_sinks"] = True
            attention["sinks_after_region"] = not (enable_tp or enable_cp)
            attention["param_init"] = {
                "sinks": make_param_init(sinks_init_fn_type, sinks_init_std)
            }

        if mirror < n_dense_layers:
            feed_forward = _ffn(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_init=make_param_init(w1_init_fn_type, w1_init_std),
                w2_init=make_param_init(w2_init_fn_type, w2_init_std, div_ffn),
                w3_init=make_param_init(w3_init_fn_type, w3_init_std, w3_div),
                shared_experts=False,
                enable_sp=enable_sp,
            )
        else:
            router_args = dict(
                num_experts=num_experts,
                # fp32 output; "hi_mid_lo": close to fp32 gradients (upstream's
                # router setting).
                gate=HiMidLoLinear.Config(
                    backward_mode="hi_mid_lo",
                    in_features=dim,
                    out_features=num_experts,
                    param_init={
                        "weight": make_param_init(router_init_fn_type, router_init_std)
                    },
                    sharding_config=router_gate_sharding_config(),
                ),
                score_func=_SCORE_FUNCS[score_func](),
                top_k=top_k,
                # The top-k weights renormalized to sum 1 (upstream's default
                # is off).
                route_norm=True,
                route_scale=route_scale,
                aux_loss=(
                    None
                    if aux_loss is None
                    else _aux_loss_config(aux_loss, num_experts)
                ),
                sharding_config=router_sharding_config(),
            )
            expert_sharding = routed_expert_linear_sharding_config(enable_ep=enable_ep)
            if expert_gemm == "deepgemm" and (token_dispatch != "gather" or enable_ep):
                raise ValueError(
                    'expert_gemm="deepgemm" needs token_dispatch="gather" and EP 1 '
                    "(its padded expert-major layout)"
                )
            if expert_gemm == "deepgemm" and (dim % 64 or moe_hidden_dim % 64):
                raise ValueError(
                    'expert_gemm="deepgemm" needs dim and moe_hidden_dim multiples of '
                    f"64 (DeepGEMM's SM90 bf16 K granularity), got {dim} / {moe_hidden_dim}"
                )
            expert_linear = (
                DeepGEMMGroupedLinear if expert_gemm == "deepgemm" else GroupedLinear
            )
            dispatcher = (
                GatherTokenDispatcher.Config(
                    num_experts=num_experts,
                    top_k=top_k,
                    pad_multiple=DEEPGEMM_ALIGN if expert_gemm == "deepgemm" else 1,
                )
                if token_dispatch == "gather"
                else AllToAllTokenDispatcher.Config(
                    num_experts=num_experts, top_k=top_k
                )
            )
            plain_router, quantile_router = (
                (RowTopKRouter, RowTopKQuantileBalancedRouter)
                if router_topk == "triton"
                else (TokenChoiceTopKRouter, QuantileBalancedTopKRouter)
            )
            feed_forward = OptMoE.Config(
                num_experts=num_experts,
                load_balance_coeff=load_balance_coeff,
                bias_update_norm_factor=bias_update_norm_factor,
                track_router_metrics=track_router_metrics,
                router=(
                    quantile_router.Config(
                        **router_args, num_bins=quantile_balance_bins
                    )
                    if quantile_balance_bins is not None
                    else plain_router.Config(**router_args)
                ),
                routed_experts=RoutedExperts.Config(
                    w13=expert_linear.Config(
                        group_size=num_experts,
                        in_features=dim,
                        out_features=moe_hidden_dim,
                        num_linears=2,
                        param_init=fused_grouped_gate_up_param_init(
                            {
                                "w1_EFD": make_expert_param_init(
                                    w1_init_fn_type,
                                    w1_init_std,
                                    slot=0,
                                    layer_id=layer_id,
                                ),
                                "w3_EFD": make_expert_param_init(
                                    w3_init_fn_type,
                                    w3_init_std / w3_div,
                                    slot=2,
                                    layer_id=layer_id,
                                ),
                            }
                        ),
                        sharding_config=expert_sharding,
                    ),
                    w2=expert_linear.Config(
                        group_size=num_experts,
                        in_features=moe_hidden_dim,
                        out_features=dim,
                        param_init={
                            "weight": make_expert_param_init(
                                w2_init_fn_type,
                                w2_init_std / div_ffn,
                                slot=1,
                                layer_id=layer_id,
                            )
                        },
                        sharding_config=expert_sharding,
                    ),
                    # All-to-all under EP, local dispatch at EP 1 (no EP mesh).
                    token_dispatcher=dispatcher,
                    sharding_config=routed_experts_sharding_config(),
                ),
                shared_experts=(
                    _ffn(
                        dim=dim,
                        hidden_dim=moe_hidden_dim * num_shared_experts,
                        w1_init=make_param_init(w1_init_fn_type, w1_init_std),
                        w2_init=make_param_init(w2_init_fn_type, w2_init_std, div_ffn),
                        w3_init=make_param_init(w3_init_fn_type, w3_init_std, w3_div),
                        shared_experts=True,
                        enable_sp=enable_sp,
                    )
                    if num_shared_experts > 0
                    else None
                ),
                sharding_config=opt_moe_sharding_config(enable_sp=enable_sp),
            )

        layers.append(
            OPTMoETransformerBlock.Config(
                attention=GatedSWAttention.Config(**attention),
                attention_norm=block_norm(),
                ffn_norm=block_norm(),
                feed_forward=feed_forward,
                block_scale=block_scale,
                identity_scale=identity_scale,
            )
        )
    return layers


def _opt_moe_config(
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
    rope_length: int | None = None,
    dim: int,
    vocab_size: int = 201088,
    enable_weight_tying: bool = False,
    norm_type: Literal["rmsnorm", "np_rmsnorm"] = "rmsnorm",
    norm_eps: float = 1e-30,
    embedding_init_fn_type: str = "scion_normal_input",
    embedding_init_std: float = 1.0,
    lm_head_init_fn_type: str = "scion_normal_output",
    lm_head_init_std: float = 1.0,
    rope: dict[str, Any],
    rope_of_swa: dict[str, Any] | None = None,
    num_mtp_layers: int = 0,
    eh_proj_init_fn_type: str = "scaled_orthogonal",
    eh_proj_init_std: float = 1.0,
    **layer_settings: Any,
) -> OPTMoEModel.Config:
    """The complete, sharded model config: the embedding, the final norm, the
    head and ``build_opt_moe_layers(**layer_settings)``. ``rope`` /
    ``rope_of_swa``: ``CosSinRoPE.Config`` fields (partial RoPE when ``dim`` <
    head_dim), the cache ``max(seq_len, rope_length)`` long; ``norm_type`` /
    ``norm_eps``: every norm.

    ``num_mtp_layers`` (0 or 1): DeepSeek-V3's multi-token prediction, one
    ``OPTMoEMTPBlock`` predicting the token after next: a mirror of the last
    layer (``build_opt_moe_layers``: full attention, its own expert seeds) behind
    ``eh_proj(cat(enorm(e), hnorm(h)))`` (``eh_proj_init_*``: the fusion's
    init), then ``mtp_norm``. The run adds the MTP loss (``mtp.py``
    ``MTPCrossEntropyLoss``) and ``dataloader.num_mtp_layers``."""
    if num_mtp_layers not in (0, 1):
        raise ValueError(f"num_mtp_layers must be 0 or 1, got {num_mtp_layers}")
    flags = layout_flags(parallelism)
    cache_length = max(seq_len, rope_length or 0)
    blocks = build_opt_moe_layers(
        dim=dim,
        norm_type=norm_type,
        norm_eps=norm_eps,
        rope=CosSinRoPE.Config(**rope, max_context_length=cache_length),
        rope_of_swa=(
            None
            if rope_of_swa is None
            else CosSinRoPE.Config(**rope_of_swa, max_context_length=cache_length)
        ),
        **layer_settings,
        num_mtp_layers=num_mtp_layers,
        enable_tp=flags["tp"],
        enable_sp=flags["sp"],
        enable_ep=flags["ep"],
        enable_cp=flags["cp"],
    )
    n_layers = len(blocks) - num_mtp_layers
    layers = blocks[:n_layers]
    block_norm_sharding = norm_sharding_config(enable_sp=flags["sp"])
    mtp_layers = [
        OPTMoEMTPBlock.Config(
            **{
                f.name: getattr(block, f.name)
                for f in dataclasses.fields(block)
                if f.name != "sharding_config"
            },
            enorm=_norm(norm_type, dim, norm_eps, sharding_config=block_norm_sharding),
            hnorm=_norm(norm_type, dim, norm_eps, sharding_config=block_norm_sharding),
            eh_proj=Linear.Config(
                in_features=2 * dim,
                out_features=dim,
                param_init={
                    "weight": make_param_init(eh_proj_init_fn_type, eh_proj_init_std)
                },
                sharding_config=mtp_projection_sharding_config(enable_sp=flags["sp"]),
            ),
            mtp_norm=_norm(
                norm_type,
                dim,
                norm_eps,
                sharding_config=pre_lm_head_norm_sharding_config(enable_sp=flags["sp"]),
            ),
            sharding_config=mtp_block_sharding_config(enable_sp=flags["sp"]),
        )
        for block in blocks[n_layers:]
    ]
    return OPTMoEModel.Config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init={
                "weight": make_param_init(embedding_init_fn_type, embedding_init_std)
            },
            sharding_config=token_embedding_sharding_config(enable_sp=flags["sp"]),
        ),
        norm=_norm(
            norm_type,
            dim,
            norm_eps,
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=flags["sp"]),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init={
                "weight": make_param_init(lm_head_init_fn_type, lm_head_init_std)
            },
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        mtp_layers=mtp_layers,
        built_for=flags,
    )


# `mis-*`: SSSF hybrid attention (window 128), GQA with qk-norm, one shared
# expert, top-8 routing.
_MIS = dict(
    n_heads=32,
    n_kv_heads=4,
    head_dim=128,
    qk_norm=True,
    attn_backend="flex",
    sliding_window_size=128,
    rope=dict(dim=128, theta=10000.0),
    top_k=8,
)


def _mis_8b(**overrides: Any) -> OPTMoEModel.Config:
    return _opt_moe_config(
        **dict(
            _MIS,
            dim=2048,
            n_layers=24,
            swa_pattern="SSSF" * 6,
            rope_pattern="RRRN" * 6,
            n_dense_layers=0,
            moe_hidden_dim=640,
            num_experts=64,
        )
        | overrides
    )


def _mis_8b_proxy(**overrides: Any) -> OPTMoEModel.Config:
    return _opt_moe_config(
        **dict(
            _MIS,
            dim=512,
            n_layers=24,
            n_heads=8,
            n_kv_heads=1,
            swa_pattern="SSSF" * 6,
            rope_pattern="RRRN" * 6,
            n_dense_layers=0,
            moe_hidden_dim=160,
            num_experts=64,
        )
        | overrides
    )


def _mis_30b_a3b(**overrides: Any) -> OPTMoEModel.Config:
    return _opt_moe_config(
        **dict(
            _MIS,
            dim=2048,
            n_layers=40,
            swa_pattern="SSSF" * 10,
            n_dense_layers=1,
            dense_hidden_dim=8192,
            moe_hidden_dim=512,
            num_experts=256,
        )
        | overrides
    )


def _mis_100b_a5b(**overrides: Any) -> OPTMoEModel.Config:
    return _opt_moe_config(
        **dict(
            _MIS,
            dim=3072,
            n_layers=40,
            swa_pattern="SSSF" * 10,
            n_dense_layers=1,
            dense_hidden_dim=8192,
            moe_hidden_dim=1024,
            num_experts=256,
        )
        | overrides
    )


# `gauge-*`: the optimizer study's dense models (full attention, MHA, head_dim
# 128, qk-norm).
def _gauge(
    *, n_layers: int, dim: int, n_heads: int, dense_hidden_dim: int, **overrides: Any
) -> OPTMoEModel.Config:
    return _opt_moe_config(
        **dict(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            head_dim=128,
            qk_norm=True,
            attn_backend="flex",
            rope=dict(dim=128, theta=10000.0),
            n_dense_layers=n_layers,
            dense_hidden_dim=dense_hidden_dim,
        )
        | overrides
    )


# `production-*-336m`: the optimizer study's production recipe at 336M (the
# gauge-d16-w1024 shape): every hidden matrix scaled-orthogonal at std 4 (the x4
# in the init), embedding std 0.02.
_PRODUCTION_336M = dict(
    dim=1024,
    n_layers=16,
    n_heads=16,
    n_kv_heads=16,
    head_dim=128,
    qk_norm=True,
    attn_backend="flex",
    rope=dict(dim=128, theta=10000.0),
    embedding_init_std=0.02,
    wqkv_init_std=4.0,
    wo_init_std=4.0,
    w_gate_init_std=4.0,
    w1_init_std=4.0,
    w2_init_std=4.0,
    w3_init_std=4.0,
)


def _production_dense_336m(**overrides: Any) -> OPTMoEModel.Config:
    return _opt_moe_config(
        **dict(_PRODUCTION_336M, n_dense_layers=16, dense_hidden_dim=4096) | overrides
    )


# Every FFN an MoE: 64 routed experts of hidden 448 + 1 shared, top 8; the
# router gate scaled-orthogonal at std 4.
def _production_moe_336m(**overrides: Any) -> OPTMoEModel.Config:
    return _opt_moe_config(
        **dict(
            _PRODUCTION_336M,
            n_dense_layers=0,
            moe_hidden_dim=448,
            num_experts=64,
            top_k=8,
            router_init_fn_type="scaled_orthogonal",
            router_init_std=4.0,
        )
        | overrides
    )


# Each entry: the flavor's function and its default training context length.
MODEL_FLAVORS: dict[str, tuple[Callable[..., OPTMoEModel.Config], int]] = {
    "mis-8b": (_mis_8b, 4096),
    "mis-8b-proxy": (_mis_8b_proxy, 4096),
    "mis-8b-proxy-qbalance": (
        functools.partial(_mis_8b_proxy, quantile_balance_bins=1000),
        4096,
    ),
    "mis-30b-a3b": (_mis_30b_a3b, 4096),
    "mis-100b-a5b": (_mis_100b_a5b, 4096),
    "gauge-d8-w256": (
        functools.partial(
            _gauge, n_layers=8, dim=256, n_heads=4, dense_hidden_dim=1024
        ),
        4096,
    ),
    "gauge-d16-w256": (
        functools.partial(
            _gauge, n_layers=16, dim=256, n_heads=4, dense_hidden_dim=1024
        ),
        4096,
    ),
    "gauge-d16-w512": (
        functools.partial(
            _gauge, n_layers=16, dim=512, n_heads=8, dense_hidden_dim=2048
        ),
        4096,
    ),
    "gauge-d16-w1024": (
        functools.partial(
            _gauge, n_layers=16, dim=1024, n_heads=16, dense_hidden_dim=4096
        ),
        4096,
    ),
    "production-dense-336m": (_production_dense_336m, 4096),
    "production-moe-336m": (_production_moe_336m, 4096),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    rope_length: int | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
    cp_attention: Literal["all_gather_kv", "ulysses"] = "all_gather_kv",
    **overrides: Any,
) -> OPTMoEModel.Config:
    """The complete, sharded config of ``flavor`` for ``parallelism``, with the
    recipe's ``overrides`` (they replace the flavor's settings of the same
    name). ``seq_len``: the training context (default: the flavor's);
    ``rope_length``: a longer RoPE cache (validation sequences); ``converters``:
    model config converters applied to the result (e.g. TitanRL's fp32
    lm_head); ``cp_attention``: under CP, every rank's queries against the
    all-gathered K / V (``all_gather_kv``, with a load balancer for causal
    masks) or the whole sequence for ``heads / cp`` heads after an all-to-all
    (``ulysses``: needs the q and KV head counts divisible by CP, contiguous
    shards)."""
    if flavor not in MODEL_FLAVORS:
        raise ValueError(
            f"Unknown opt_moe flavor {flavor!r}; available: {', '.join(MODEL_FLAVORS)}"
        )
    get_config, default_seq_len = MODEL_FLAVORS[flavor]
    config = get_config(
        seq_len=seq_len or default_seq_len,
        parallelism=parallelism,
        rope_length=rope_length,
        **copy.deepcopy(overrides),
    )
    if parallelism.context_parallel_degree > 1:
        from torchtitan.config.transform.context_parallel import (
            ContextParallelTransform,
        )
        from torchtitan.models.common.attention import cp_attention as cp

        if cp_attention == "ulysses":
            if parallelism.tensor_parallel_degree > 1:
                # the heads are TP-sharded before Ulysses shards them over CP;
                # nothing tests the two together (the sinks slice per CP rank)
                raise ValueError("cp_attention 'ulysses' with TP is not supported")
            inner_attention_map = {
                FlexInnerAttention: cp.UlyssesFlexInnerAttention,
                SlidingWindowFlexInnerAttention: cp.UlyssesSlidingWindowFlexInnerAttention,
                VarlenInnerAttention: cp.UlyssesVarlenInnerAttention,
                SlidingWindowVarlenInnerAttention: cp.UlyssesSlidingWindowVarlenInnerAttention,
            }
        else:
            inner_attention_map = {
                FlexInnerAttention: cp.AllGatherKVFlexInnerAttention,
                SlidingWindowFlexInnerAttention: cp.AllGatherKVSlidingWindowFlexInnerAttention,
                VarlenInnerAttention: cp.AllGatherKVVarlenInnerAttention,
                SlidingWindowVarlenInnerAttention: cp.AllGatherKVSlidingWindowVarlenInnerAttention,
            }
        ContextParallelTransform(inner_attention_map=inner_attention_map).transform(
            config
        )
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    config.flavor = flavor
    config.build_args = copy.deepcopy(overrides)
    return config
