# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OPT MoE model flavors.

``build_model_config(flavor, *, parallelism, seq_len, **overrides)`` builds the
complete, sharded ``OPTMoEModel.Config``: ``_opt_moe_config`` loops over the
layers (SWA / RoPE patterns, depth-init divisors, dense-first layers, residual
scales) and builds each component with its sharding. A flavor is the builder's
arguments; a run passes the ones it changes::

    build_model_config(
        "gauge-d16-w512", parallelism=parallelism, seq_len=4096,
        depth_init="identity", swa_pattern="SSSF" * 4, sliding_window_size=128,
        attention=dict(gated_attention_type="element-wise", conv_kernel_size=4),
        moe=dict(hidden_dim=224, num_experts=64, top_k=8),
    )

``attention`` / ``feed_forward`` / ``moe`` / ``rope`` / ``rope_of_swa`` are
merged key by key into the flavor's (their keys are the keyword arguments of
``make_gated_swa_attention_config``, ``_feed_forward_config``,
``make_norm_moe_config`` and ``CosSinRoPE.Config``); every other argument
replaces the flavor's. All values are plain, so the merged arguments are kept
in ``config.build_args`` and ``build_model_config(flavor, **build_args)``
rebuilds the same model for another parallelism.
"""

import copy
import dataclasses
import functools
from typing import Any, Literal

import numpy as np

from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
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
)
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import (
    BatchWiseLoadBalanceLoss,
    MicrobatchWiseLoadBalanceLoss,
    QuantileBalancedTopKRouter,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.moe.sharding import (
    routed_experts_sharding_config,
    router_sharding_config,
)
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm
from torchtitan.models.common.rope import CosSinRoPE, RoPE

from .gated_swa_attention import GatedSWAttention
from .model import layout_flags, OPTMoEModel, OPTMoETransformerBlock
from .norm_moe import NormMoE
from .param_init import (
    conv_init_fn,
    make_expert_param_init,
    make_param_init,
    setup_depth_init,
    setup_residual_scale,
)
from .sharding import (
    attention_sinks_state_sharding,
    dense_linear_sharding_config,
    routed_expert_weight_sharding_config,
)

_SCORE_FUNCS = {
    "sigmoid": Sigmoid.Config,
    "softmax": Softmax.Config,
    "sqrtsoftplus": SqrtSoftplus.Config,
}

__all__ = [
    "MODEL_FLAVORS",
    "build_model_config",
    "make_gated_swa_attention_config",
    "make_norm_moe_config",
]


# The MoE gate scaling factors (https://arxiv.org/abs/2502.16982), pre-computed per
# (num_experts, top_k); calc_gate_scaling_factor estimates the others.
SCALING_FACTORS = {
    (4, 1): 1.0,
    (4, 2): 1.3975240382831564,
    (4, 4): 1.8746839497244312,
    (8, 1): 1.0,
    (8, 2): 1.4089312089643131,
    (8, 4): 1.9716372098310582,
    (8, 6): 2.3692189889782083,
    (8, 8): 2.6297724148600703,
    (16, 1): 1.0,
    (16, 2): 1.412131787337075,
    (16, 4): 1.990941161710805,
    (16, 6): 2.4283171262135257,
    (16, 8): 2.7879495517358537,
    (16, 16): 3.7047009687546923,
    (32, 1): 1.0,
    (32, 2): 1.4132875875185074,
    (32, 4): 1.9964881494834583,
    (32, 6): 2.4419467013356364,
    (32, 8): 2.8154055554895394,
    (32, 16): 3.943324881680555,
    (32, 32): 5.229939776738365,
    (64, 1): 1.0,
    (64, 2): 1.4137804787923913,
    (64, 4): 1.9984419912715523,
    (64, 6): 2.446338919929975,
    (64, 8): 2.8232884471713273,
    (64, 16): 3.981865423254339,
    (64, 32): 5.5779444133949685,
    (64, 64): 7.391360398907228,
    (128, 1): 1.0,
    (128, 2): 1.4139935847474265,
    (128, 4): 1.999243359037702,
    (128, 6): 2.448026924377305,
    (128, 8): 2.826111482320644,
    (128, 16): 3.992795051245798,
    (128, 32): 5.631330086465613,
    (128, 64): 7.8888558415100025,
    (128, 128): 10.447601374558701,
    (192, 1): 1.0,
    (192, 2): 1.4140596973689825,
    (192, 4): 1.9994875418633284,
    (192, 6): 2.448528209376721,
    (192, 8): 2.8269205350175617,
    (192, 16): 3.9954948027768986,
    (192, 32): 5.642154758774243,
    (192, 64): 7.9448577065669665,
    (192, 128): 11.03389500669084,
    (256, 1): 1.0,
    (256, 2): 1.4140933478999065,
    (256, 4): 1.9996002066706609,
    (256, 6): 2.4487450517795386,
    (256, 8): 2.827297937808976,
    (256, 16): 3.996711691215216,
    (256, 32): 5.646620595072715,
    (256, 64): 7.964034924039261,
    (256, 128): 11.157350249127516,
}


def calc_gate_scaling_factor(
    num_experts: int, topk: int, iter_times: int = 10000
) -> float:
    """The MoE gate scaling factor: E[1 / ||p||_2] over the top-k sigmoid scores
    of Gaussian logits, renormalized to sum 1 (the pre-computed table when it has
    the pair, else a seeded Monte Carlo estimate over ``iter_times`` draws)."""
    if num_experts < 1 or topk < 1:
        return 1.0
    if (num_experts, topk) in SCALING_FACTORS:
        return SCALING_FACTORS[(num_experts, topk)]
    # A fixed generator: every rank and every run gets the same route_scale.
    logits = np.random.default_rng(0).standard_normal((iter_times, num_experts))
    p = -np.sort(-1 / (1 + np.exp(-logits)), axis=-1)[:, :topk]
    p = p / p.sum(axis=-1, keepdims=True)
    return float(np.mean(1.0 / np.sqrt((p**2).sum(axis=-1))))


def _norm(
    norm_type: Literal["rmsnorm", "np_rmsnorm"], dim: int, eps: float
) -> RMSNorm.Config:
    """Upstream's RMSNorm: with learnable gains ("rmsnorm") or without ("np_rmsnorm")."""
    if norm_type not in ("rmsnorm", "np_rmsnorm"):
        raise ValueError(f"norm_type must be rmsnorm | np_rmsnorm, got {norm_type!r}")
    return RMSNorm.Config(
        normalized_shape=dim, eps=eps, elementwise_affine=norm_type == "rmsnorm"
    )


def make_gated_swa_attention_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    qk_norm: bool = False,
    attn_backend: Literal["flex", "flex_flash", "varlen"] = "flex",
    gated_attention_type: Literal["element-wise"] | None = None,
    conv_kernel_size: int = 0,
    conv_streams: Literal["qkv", "kv"] = "qkv",
    conv_init_fn_type: str = "delay",
    conv_init_gain: float = 4.0,
    attention_sinks: bool = False,
    sinks_init_fn_type: str = "zeros",
    sinks_init_std: float = 0.0,
    wqkv_init_fn_type: str = "scaled_orthogonal",
    wo_init_fn_type: str = "scaled_orthogonal",
    w_gate_init_fn_type: str = "scaled_orthogonal",
    wqkv_init_std: float = 1.0,
    wo_init_std: float = 1.0,
    w_gate_init_std: float = 1.0,
    sliding_window_size: int | None,
    rope: RoPE.Config | None,
    norm_type: str,
    norm_eps: float,
    residual_div: float,
    enable_sp: bool,
) -> GatedSWAttention.Config:
    """One layer's complete, sharded attention config: upstream's GQA part
    (``make_gqa_config``), then the gate, the convs and the sinks.

    ``conv_kernel_size`` 0: no conv; else the Canon short conv ``x + conv(x)`` of
    this width on q, k and v (``conv_streams`` "qkv") or k and v ("kv"), each
    channel initialized by ``conv_init_fn_type`` ("delay", "random_opnorm",
    "torch_default") at ``conv_init_gain``. ``sliding_window_size`` None: a
    full-attention layer; ``rope`` None: a NoPE layer; ``residual_div``: the
    depth-init divisor of wo and the gate.
    """
    if conv_kernel_size > 0 and conv_streams not in ("qkv", "kv"):
        raise ValueError(f"conv_streams must be qkv | kv, got {conv_streams!r}")
    inner = get_attention_config(attn_backend)
    if sliding_window_size is not None:
        if isinstance(inner, FlexInnerAttention.Config):
            inner = SlidingWindowFlexInnerAttention.Config(
                block_size=inner.block_size,
                kernel_options=inner.kernel_options,
                sliding_window_size=sliding_window_size,
            )
        else:
            assert isinstance(inner, VarlenInnerAttention.Config)
            inner = SlidingWindowVarlenInnerAttention.Config(
                **{f.name: getattr(inner, f.name) for f in dataclasses.fields(inner)},
                sliding_window_size=sliding_window_size,
            )
    n_kv = n_heads if n_kv_heads is None else n_kv_heads
    head = head_dim if head_dim is not None else dim // n_heads
    gqa = make_gqa_config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        wqkv_param_init={"weight": make_param_init(wqkv_init_fn_type, wqkv_init_std)},
        wo_param_init={
            "weight": make_param_init(wo_init_fn_type, wo_init_std, residual_div)
        },
        inner_attention=inner,
        rope=rope,
        qk_norm=_norm(norm_type, head, norm_eps) if qk_norm else None,
        enable_sp=enable_sp,
    )
    fields = {f.name: getattr(gqa, f.name) for f in dataclasses.fields(gqa)}
    if gated_attention_type == "element-wise":
        fields["gate_proj"] = Linear.Config(
            in_features=dim,
            out_features=n_heads * head,
            param_init={
                "weight": make_param_init(
                    w_gate_init_fn_type, w_gate_init_std, residual_div
                )
            },
            sharding_config=dense_linear_sharding_config(),
        )
    elif gated_attention_type is not None:
        raise ValueError(
            "gated_attention_type must be None (no gate) or 'element-wise', "
            f"got {gated_attention_type!r}"
        )
    if conv_kernel_size > 0:

        def conv(channels: int) -> Conv1d.Config:
            return Conv1d.Config(
                in_channels=channels,
                out_channels=channels,
                kernel_size=conv_kernel_size,
                groups=channels,
                bias=False,
                param_init={"weight": conv_init_fn(conv_init_fn_type, conv_init_gain)},
                sharding_config=dense_linear_sharding_config(),
            )

        if conv_streams == "qkv":
            fields["q_conv"] = conv(n_heads * head)
        fields["k_conv"] = conv(n_kv * head)
        fields["v_conv"] = conv(n_kv * head)
    if attention_sinks:
        fields["attention_sinks"] = True
        fields["param_init"] = {
            "sinks": make_param_init(sinks_init_fn_type, sinks_init_std)
        }
        fields["sharding_config"] = attention_sinks_state_sharding(gqa.sharding_config)
    return GatedSWAttention.Config(**fields)


def _feed_forward_config(
    *,
    dim: int,
    hidden_dim: int,
    w1_init_fn_type: str = "scaled_orthogonal",
    w2_init_fn_type: str = "scaled_orthogonal",
    w3_init_fn_type: str = "scaled_orthogonal",
    w1_init_std: float = 1.0,
    w2_init_std: float = 1.0,
    w3_init_std: float = 1.0,
    residual_div: float,
    init_gate_as_residual: bool,
) -> DenseFeedForward.Config:
    """A SwiGLU FFN (upstream's ``DenseFeedForward``, gate and up in one
    ``[2, F, D]`` ``w13``) with its own initializer per projection:
    ``residual_div`` divides w2 (and w3 with ``init_gate_as_residual``)."""
    w3_div = residual_div if init_gate_as_residual else 1.0
    linear_sharding = dense_linear_sharding_config()
    return DenseFeedForward.Config(
        w13=Linear.Config(
            in_features=dim,
            out_features=hidden_dim,
            num_linears=2,
            param_init=fused_gate_up_param_init(
                {"weight": make_param_init(w1_init_fn_type, w1_init_std)},
                {"weight": make_param_init(w3_init_fn_type, w3_init_std, w3_div)},
            ),
            sharding_config=linear_sharding,
        ),
        w2=Linear.Config(
            in_features=hidden_dim,
            out_features=dim,
            param_init={
                "weight": make_param_init(w2_init_fn_type, w2_init_std, residual_div)
            },
            sharding_config=linear_sharding,
        ),
    )


def make_norm_moe_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    num_shared_experts: int = 1,
    scaling_factor: float | None = None,
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid",
    load_balance_coeff: float | None = 1e-3,
    bias_update_norm_factor: str = "sign",
    aux_loss: dict[str, Any] | None = None,
    quantile_balance_bins: int | None = None,
    track_router_metrics: bool = True,
    w1_init_fn_type: str = "scaled_orthogonal",
    w2_init_fn_type: str = "scaled_orthogonal",
    w3_init_fn_type: str = "scaled_orthogonal",
    w1_init_std: float = 1.0,
    w2_init_std: float = 1.0,
    w3_init_std: float = 1.0,
    router_init_fn_type: str = "scion_normal_output",
    router_init_std: float = 1.0,
    layer_id: int = 0,
    residual_div: float = 1.0,
    init_gate_as_residual: bool = False,
    enable_ep: bool = False,
) -> NormMoE.Config:
    """One layer's complete, sharded MoE config: ``num_experts`` routed SwiGLU
    experts of ``hidden_dim`` (each drawn with its own seed), upstream's top-k
    router, ``num_shared_experts`` shared experts as one FFN.

    ``scaling_factor`` None: the router's ``route_scale`` from
    ``calc_gate_scaling_factor``; ``load_balance_coeff``: the expert-bias update
    rate (None: no bias); ``quantile_balance_bins``: upstream's quantile-balanced
    router (the bias solved from the required-bias histogram) instead of the
    sign nudge; ``aux_loss``: upstream's auxiliary load-balance loss,
    ``dict(loss_type="microbatch_wise" | "batch_wise", coeff=...)`` (None: none).
    The token dispatcher: all-to-all under EP, local dispatch without (DeepEP /
    HybridEP: upstream's ``TokenDispatcherTransform`` on the recipe).
    """
    route_scale = (
        scaling_factor
        if scaling_factor is not None
        else calc_gate_scaling_factor(num_experts, top_k, iter_times=10_000)
    )
    aux_loss_cfg = None
    if aux_loss is not None:
        aux_args = dict(aux_loss)
        loss_type = aux_args.pop("loss_type", None)
        if loss_type == "microbatch_wise":
            aux_loss_cfg = MicrobatchWiseLoadBalanceLoss.Config(**aux_args)
        elif loss_type == "batch_wise":
            aux_loss_cfg = BatchWiseLoadBalanceLoss.Config(
                **aux_args, num_experts=num_experts
            )
        else:
            raise ValueError(
                "aux_loss needs loss_type 'microbatch_wise' or 'batch_wise', "
                f"got {loss_type!r}"
            )
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
            sharding_config=dense_linear_sharding_config(),
        ),
        score_func=_SCORE_FUNCS[score_func](),
        top_k=top_k,
        # calc_gate_scaling_factor calibrates route_scale on top-k weights
        # normalized to sum 1, so route_norm stays on (upstream's default is off).
        route_norm=True,
        route_scale=route_scale,
        aux_loss=aux_loss_cfg,
        sharding_config=router_sharding_config() if enable_ep else None,
    )
    if quantile_balance_bins is not None:
        router = QuantileBalancedTopKRouter.Config(
            **router_args, num_bins=quantile_balance_bins
        )
    else:
        router = TokenChoiceTopKRouter.Config(**router_args)
    w3_div = residual_div if init_gate_as_residual else 1.0
    expert_weight_sharding = routed_expert_weight_sharding_config(enable_ep=enable_ep)
    return NormMoE.Config(
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
        bias_update_norm_factor=bias_update_norm_factor,
        track_router_metrics=track_router_metrics,
        router=router,
        routed_experts=RoutedExperts.Config(
            w13=GroupedLinear.Config(
                group_size=num_experts,
                in_features=dim,
                out_features=hidden_dim,
                num_linears=2,
                param_init=fused_grouped_gate_up_param_init(
                    {
                        "w1_EFD": make_expert_param_init(
                            w1_init_fn_type, w1_init_std, slot=0, layer_id=layer_id
                        ),
                        "w3_EFD": make_expert_param_init(
                            w3_init_fn_type,
                            w3_init_std / w3_div,
                            slot=2,
                            layer_id=layer_id,
                        ),
                    }
                ),
                sharding_config=expert_weight_sharding,
            ),
            w2=GroupedLinear.Config(
                group_size=num_experts,
                in_features=hidden_dim,
                out_features=dim,
                param_init={
                    "weight": make_expert_param_init(
                        w2_init_fn_type,
                        w2_init_std / residual_div,
                        slot=1,
                        layer_id=layer_id,
                    )
                },
                sharding_config=expert_weight_sharding,
            ),
            # All-to-all under EP, local dispatch at EP 1 (no EP mesh).
            token_dispatcher=AllToAllTokenDispatcher.Config(
                num_experts=num_experts, top_k=top_k
            ),
            sharding_config=routed_experts_sharding_config() if enable_ep else None,
        ),
        shared_experts=(
            _feed_forward_config(
                dim=dim,
                hidden_dim=hidden_dim * num_shared_experts,
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


def _opt_moe_config(
    *,
    parallelism: ParallelismConfig,
    seq_len: int,
    rope_length: int | None,
    dim: int,
    n_layers: int,
    attention: dict[str, Any],
    rope: dict[str, Any],
    feed_forward: dict[str, Any] | None = None,
    moe: dict[str, Any] | None = None,
    n_dense_layers: int = 0,
    vocab_size: int = 201088,
    sliding_window_size: int | None = None,
    swa_pattern: str | None = None,
    rope_pattern: str | None = None,
    rope_of_swa: dict[str, Any] | None = None,
    norm_type: str = "rmsnorm",
    norm_eps: float = 1e-30,
    prenorm_gain_init: float = 1.0,
    depth_init: str = "total_depth_R",
    residual_scale: str = "identity",
    init_gate_as_residual: bool = False,
    first_in_init_fn_type: str = "scion_normal_input",
    first_in_init_std: float = 1.0,
    final_out_init_fn_type: str = "scion_normal_output",
    final_out_init_std: float = 1.0,
    enable_weight_tying: bool = False,
    enable_token_mask_for_moe: bool = False,
) -> OPTMoEModel.Config:
    """The complete model config.

    Layers: the first ``n_dense_layers`` dense (``feed_forward``), the rest MoE
    (``moe``). ``swa_pattern`` ("S" sliding-window / "F" full per layer; None:
    every layer sliding-window when ``sliding_window_size`` is set) and
    ``rope_pattern`` ("R" RoPE / "N" NoPE per layer; None: RoPE everywhere) pick
    each layer's attention; sliding-window layers take ``rope_of_swa`` when set.
    ``rope``: ``CosSinRoPE.Config`` fields (partial RoPE when ``dim`` <
    head_dim), its cache ``max(seq_len, rope_length)`` long. ``norm_type`` /
    ``norm_eps``: every norm (q/k, block and final norms).
    ``depth_init`` / ``residual_scale``: the per-layer init divisors and the
    residual scales (param_init.py).
    """
    tp = parallelism.tensor_parallel_degree > 1
    cp = parallelism.context_parallel_degree > 1
    enable_sp = parallelism.enable_sequence_parallel and tp
    enable_ep = parallelism.expert_parallel_degree > 1
    if tp:
        raise NotImplementedError(
            "OPT MoE has no tensor-parallel plan (its gate, convs, sinks and "
            "NormMoE are replicated over TP): run with tensor_parallel_degree=1"
        )
    if cp and (
        attention.get("conv_kernel_size", 0) > 0 or attention.get("attention_sinks")
    ):
        raise NotImplementedError(
            "the attention short conv and sinks do not support CP"
        )
    if cp and attention.get("attn_backend", "flex") not in ("flex", "flex_flash"):
        raise ValueError("Context Parallel needs the flex attention backend")
    use_swa = _layer_pattern(
        swa_pattern, n_layers, "S", "F", default=sliding_window_size is not None
    )
    if any(use_swa) and sliding_window_size is None:
        raise ValueError("sliding-window layers need sliding_window_size")
    use_rope = _layer_pattern(rope_pattern, n_layers, "R", "N", default=True)
    if n_dense_layers > 0 and feed_forward is None:
        raise ValueError(f"n_dense_layers={n_dense_layers} needs feed_forward")
    if n_dense_layers < n_layers and moe is None:
        raise ValueError(
            f"layers {n_dense_layers}..{n_layers - 1} are MoE layers: they need moe"
        )
    cache_length = max(seq_len, rope_length or 0)
    rope_cfg = CosSinRoPE.Config(**rope, max_context_length=cache_length)
    rope_of_swa_cfg = (
        CosSinRoPE.Config(**rope_of_swa, max_context_length=cache_length)
        if rope_of_swa is not None
        else None
    )
    block_scale, identity_scale = setup_residual_scale(residual_scale, n_layers)
    block_norm_sharding = norm_sharding_config(enable_sp=enable_sp)

    def block_norm():
        cfg = _norm(norm_type, dim, norm_eps)
        if prenorm_gain_init != 1.0:
            if norm_type != "rmsnorm":
                raise ValueError(
                    "prenorm_gain_init != 1 needs norm_type='rmsnorm' "
                    f"(got {norm_type!r}): no gain to initialise"
                )
            cfg.param_init = {
                "weight": functools.partial(nn.init.constant_, val=prenorm_gain_init)
            }
        cfg.sharding_config = block_norm_sharding
        return cfg

    layers = []
    for layer_id in range(n_layers):
        residual_div_attn, residual_div_ffn = setup_depth_init(
            depth_init, layer_id, n_layers
        )
        layer_rope = (
            rope_of_swa_cfg
            if use_swa[layer_id] and rope_of_swa_cfg is not None
            else rope_cfg
        )
        attention_cfg = make_gated_swa_attention_config(
            **attention,
            dim=dim,
            sliding_window_size=sliding_window_size if use_swa[layer_id] else None,
            rope=layer_rope if use_rope[layer_id] else None,
            norm_type=norm_type,
            norm_eps=norm_eps,
            residual_div=residual_div_attn,
            enable_sp=enable_sp,
        )
        ffn = dict(
            dim=dim,
            residual_div=residual_div_ffn,
            init_gate_as_residual=init_gate_as_residual,
        )
        layers.append(
            OPTMoETransformerBlock.Config(
                attention=attention_cfg,
                attention_norm=block_norm(),
                ffn_norm=block_norm(),
                feed_forward=(
                    _feed_forward_config(**feed_forward, **ffn)
                    if layer_id < n_dense_layers
                    else None
                ),
                moe=(
                    make_norm_moe_config(
                        **moe, **ffn, layer_id=layer_id, enable_ep=enable_ep
                    )
                    if layer_id >= n_dense_layers
                    else None
                ),
                block_scale=block_scale,
                identity_scale=identity_scale,
            )
        )
    final_norm = _norm(norm_type, dim, norm_eps)
    final_norm.sharding_config = pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
    config = OPTMoEModel.Config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init={
                "weight": make_param_init(first_in_init_fn_type, first_in_init_std)
            },
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        layers=layers,
        norm=final_norm,
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init={
                "weight": make_param_init(final_out_init_fn_type, final_out_init_std)
            },
            sharding_config=lm_head_sharding_config(),
        ),
        enable_token_mask_for_moe=enable_token_mask_for_moe,
        built_for=layout_flags(parallelism),
    )
    if cp:  # the CP versions of the flex inner attentions (all-gather KV)
        from torchtitan.config.transform.context_parallel import (
            ContextParallelTransform,
        )
        from torchtitan.models.common.attention.cp_attention import (
            AllGatherKVFlexInnerAttention,
            AllGatherKVSlidingWindowFlexInnerAttention,
        )

        ContextParallelTransform(
            inner_attention_map={
                FlexInnerAttention: AllGatherKVFlexInnerAttention,
                SlidingWindowFlexInnerAttention: (
                    AllGatherKVSlidingWindowFlexInnerAttention
                ),
            }
        ).transform(config)
    return config


# Each entry: the builder's arguments (the rest take the builders' defaults) and
# the default training context length. `mis-*`: SSSF hybrid attention (window 128),
# qk-norm, one shared expert, top-8 routing, no MoE output factor. `gauge-*`: the
# optimizer study's dense models (full attention, MHA, FFN 4 x dim).
MODEL_FLAVORS: dict[str, tuple[dict[str, Any], int]] = {
    "mis-8b": (
        dict(
            n_layers=24,
            dim=2048,
            n_dense_layers=0,
            swa_pattern="SSSF" * 6,
            sliding_window_size=128,
            rope_pattern="RRRN" * 6,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                attn_backend="flex",
            ),
            feed_forward=dict(hidden_dim=5120),
            moe=dict(
                hidden_dim=640,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,
            ),
        ),
        4096,
    ),
    "mis-8b-proxy": (
        dict(
            n_layers=24,
            dim=512,
            n_dense_layers=0,
            swa_pattern="SSSF" * 6,
            sliding_window_size=128,
            rope_pattern="RRRN" * 6,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(
                n_heads=8,
                n_kv_heads=1,
                head_dim=128,
                qk_norm=True,
                attn_backend="flex",
            ),
            feed_forward=dict(hidden_dim=1280),
            moe=dict(
                hidden_dim=160,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,
            ),
        ),
        4096,
    ),
    # mis-8b-proxy with the expert bias solved from the required-bias histogram
    # instead of the DeepSeek sign nudge.
    "mis-8b-proxy-qbalance": (
        dict(
            n_layers=24,
            dim=512,
            n_dense_layers=0,
            swa_pattern="SSSF" * 6,
            sliding_window_size=128,
            rope_pattern="RRRN" * 6,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(
                n_heads=8,
                n_kv_heads=1,
                head_dim=128,
                qk_norm=True,
                attn_backend="flex",
            ),
            feed_forward=dict(hidden_dim=1280),
            moe=dict(
                hidden_dim=160,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,
                quantile_balance_bins=1000,
            ),
        ),
        4096,
    ),
    "mis-30b-a3b": (
        dict(
            n_layers=40,
            dim=2048,
            n_dense_layers=1,
            swa_pattern="SSSF" * 10,
            sliding_window_size=128,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                attn_backend="flex",
            ),
            feed_forward=dict(hidden_dim=8192),
            moe=dict(
                hidden_dim=512,
                num_experts=256,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,
            ),
        ),
        4096,
    ),
    "mis-100b-a5b": (
        dict(
            n_layers=40,
            dim=3072,
            n_dense_layers=1,
            swa_pattern="SSSF" * 10,
            sliding_window_size=128,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                attn_backend="flex",
            ),
            feed_forward=dict(hidden_dim=8192),
            moe=dict(
                hidden_dim=1024,
                num_experts=256,
                num_shared_experts=1,
                top_k=8,
                scaling_factor=1.0,
            ),
        ),
        4096,
    ),
    "gauge-d8-w256": (
        dict(
            n_layers=8,
            dim=256,
            n_dense_layers=8,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(n_heads=4, n_kv_heads=4, head_dim=128, qk_norm=True),
            feed_forward=dict(hidden_dim=1024),
        ),
        4096,
    ),
    "gauge-d16-w256": (
        dict(
            n_layers=16,
            dim=256,
            n_dense_layers=16,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(n_heads=4, n_kv_heads=4, head_dim=128, qk_norm=True),
            feed_forward=dict(hidden_dim=1024),
        ),
        4096,
    ),
    "gauge-d16-w512": (
        dict(
            n_layers=16,
            dim=512,
            n_dense_layers=16,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(n_heads=8, n_kv_heads=8, head_dim=128, qk_norm=True),
            feed_forward=dict(hidden_dim=2048),
        ),
        4096,
    ),
    "gauge-d16-w1024": (
        dict(
            n_layers=16,
            dim=1024,
            n_dense_layers=16,
            rope=dict(dim=128, theta=10000.0),
            attention=dict(n_heads=16, n_kv_heads=16, head_dim=128, qk_norm=True),
            feed_forward=dict(hidden_dim=4096),
        ),
        4096,
    ),
}

# The builder arguments merged key by key (the rest replace the flavor's).
_MERGED_ARGS = ("attention", "feed_forward", "moe", "rope", "rope_of_swa")


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    rope_length: int | None = None,
    **overrides: Any,
) -> OPTMoEModel.Config:
    """The complete, sharded config of ``flavor`` with the run's ``overrides``
    (``_opt_moe_config``'s arguments; see the module docstring), for
    ``parallelism``. ``seq_len``: the training context (default: the flavor's);
    ``rope_length``: a longer RoPE cache (validation sequences)."""
    if flavor not in MODEL_FLAVORS:
        raise ValueError(
            f"Unknown opt_moe flavor {flavor!r}; available: {', '.join(MODEL_FLAVORS)}"
        )
    flavor_args, default_seq_len = MODEL_FLAVORS[flavor]
    args = copy.deepcopy(flavor_args)
    for name, value in copy.deepcopy(overrides).items():
        if name in _MERGED_ARGS and value is not None and args.get(name) is not None:
            args[name] = {**args[name], **value}
        else:
            args[name] = value
    config = _opt_moe_config(
        **args,
        parallelism=parallelism,
        seq_len=seq_len or default_seq_len,
        rope_length=rope_length,
    )
    config.flavor = flavor
    config.build_args = args
    return config
