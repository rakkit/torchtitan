# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The routed experts on SonicMoE's kernels (flavor argument
``expert_gemm="sonicmoe"``): the router's selected experts and scores go to
SonicMoE's external-routing path (its ``moe_general_routing_inputs``): one
grouped GEMM with the SwiGLU fused in its epilogue for the up projection, one
for the down projection, and a gather-sum that applies the (fp32) scores.
Nothing is dispatched or padded: SonicMoE gathers each expert's tokens inside
its GEMMs. The router, the score weighting, the statistics and the shared
expert stay opt_moe's; SonicMoE's own router is not used. Needs EP 1 and no TP.

SonicMoE (PyPI ``sonic-moe``) is not a PyTorch library and not in the
environment: ``SONICMOE_SITE`` names the folder that holds the ``sonicmoe``
package, and it is imported on the first use of this backend only.

The expert weights are the same parameters as the other backends' (``w13``
[E, 2, F, D] and ``w2`` [E, D, F]); SonicMoE's kernels take them as zero-copy
permuted views (``w13`` as [E, 2F, D] with its concatenated gate / up layout).

Compile: the model compiles whole blocks (``compile_blocks``), SonicMoE asserts
against ``torch.compile`` and its Python glue is not traceable. So its kernels
sit behind the custom ops below (opaque to the compiler, as DeepGEMM's), whose
Python bodies run eagerly at runtime; the glue in them is SonicMoE's
(functional/__init__.py ``moe_general_routing_inputs`` and its autograd
functions) split into three ops: the routing metadata, the up projection and
the down projection with the combine. The op split lets the rematerialisation
replay the metadata and the up projection only (the down projection's backward
reads the up projection's pre-activation, not its output).

Memory: the up projection's pre-activation ``h`` [N, 2F] (N = T * K) and the
routing metadata are what the backward needs: ``h`` is 1 GiB per layer at 64k
tokens, top 8, F 512. ``SonicMoERoutedExperts.Config.recompute``: True: the
experts are one rematerialisation region that is rerun in the backward
(nothing held); False: the region is saved and autograd keeps ``h``.

Legend: T tokens, D model dim, E routed experts, K top-k, N = T * K routed
(token, expert) pairs, F expert hidden dim.
"""

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch_remat as remat

from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.moe import RoutedExperts

SONICMOE_ALIGN = 8
"""Features per row are multiples of this (16-byte rows of bf16 for TMA)."""

_ENV = "SONICMOE_SITE"
_sonic_ns: SimpleNamespace | None = None


def _sonic() -> SimpleNamespace:
    """SonicMoE's low-level kernels, imported on first use from ``$SONICMOE_SITE``."""
    global _sonic_ns
    if _sonic_ns is not None:
        return _sonic_ns
    site = os.environ.get(_ENV)
    if not site:
        raise RuntimeError(
            f'expert_gemm="sonicmoe" needs the sonicmoe package: set {_ENV} to the '
            "folder that holds it (the PyPI wheel sonic-moe unpacked; it is not "
            "installed in the environment)"
        )
    if site not in sys.path:
        sys.path.append(site)
    try:
        from sonicmoe.enums import ActivationType
        from sonicmoe.functional.backward import (
            _down_projection_backward_act,
            _down_projection_backward_weight,
            _token_broadcast_backward,
            _up_projection_backward_act,
            _up_projection_backward_weight,
        )
        from sonicmoe.functional.forward import (
            _down_projection_forward,
            _router_forward,
            _up_projection_forward,
        )
        from sonicmoe.functional.triton_kernels import (
            general_routing_router_metadata_triton,
        )
    except ImportError as e:
        raise RuntimeError(
            f"{_ENV}={site!r} does not hold an importable sonicmoe package "
            f"(sonic-moe 0.1.2.post1 with quack-kernels and nvidia-cutlass-dsl): {e}"
        ) from e
    _sonic_ns = SimpleNamespace(
        activation=ActivationType.SWIGLU.value,
        metadata=general_routing_router_metadata_triton,
        up_forward=_up_projection_forward,
        down_forward=_down_projection_forward,
        router_forward=_router_forward,
        down_backward_act=_down_projection_backward_act,
        down_backward_weight=_down_projection_backward_weight,
        up_backward_act=_up_projection_backward_act,
        up_backward_weight=_up_projection_backward_weight,
        token_broadcast_backward=_token_broadcast_backward,
    )
    return _sonic_ns


# --------------------------------------------------------------------------------------
# The ops. All int32 index tensors are SonicMoE's routing metadata for N pairs sorted by
# expert: offset_E1 the expert row boundaries, gather_N the token of each grouped row,
# scatter_N its (token, k) pair, reverse_N the grouped row of each pair, per_token_T1
# the pairs-per-token boundaries.


@torch.library.custom_op("opt_moe::sonic_metadata", mutates_args=())
def _metadata(
    expert_TK: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SonicMoE's metadata of the routing ``expert_TK`` [T, K] (token-major, so
    the token ids are ascending as its kernels need)."""
    num_tokens, top_k = expert_TK.shape
    n, device = num_tokens * top_k, expert_TK.device
    i32 = dict(dtype=torch.int32, device=device)
    token_N = torch.arange(num_tokens, **i32).repeat_interleave(top_k)
    expert_N = expert_TK.reshape(-1).to(torch.int32)
    frequency_E = torch.empty(num_experts, **i32)
    offset_E1 = torch.empty(num_experts + 1, **i32)
    gather_N = torch.empty(n, **i32)
    scatter_N = torch.empty(n, **i32)
    reverse_N = torch.empty(n, **i32)
    per_token_T1 = torch.empty(num_tokens + 1, **i32)
    _sonic().metadata(
        token_N,
        expert_N,
        num_tokens,
        num_experts,
        frequency_E,
        offset_E1,
        gather_N,
        scatter_N,
        reverse_N,
        per_token_T1,
    )
    return offset_E1, gather_N, scatter_N, reverse_N, per_token_T1


@_metadata.register_fake
def _(expert_TK, num_experts):
    num_tokens, top_k = expert_TK.shape
    n = num_tokens * top_k
    new = lambda size: expert_TK.new_empty(size, dtype=torch.int32)
    return (new(num_experts + 1), new(n), new(n), new(n), new(num_tokens + 1))


@torch.library.custom_op("opt_moe::sonic_up", mutates_args=())
def _up(
    x_TD: torch.Tensor,
    w13_EOI: torch.Tensor,
    offset_E1: torch.Tensor,
    gather_N: torch.Tensor,
    reverse_N: torch.Tensor,
    per_token_T1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The up projection ``x @ w13[expert].T`` of the rows ``gather_N`` in expert-major
    order, and its SwiGLU: the post-activation ``a_NF`` and the pre-activation ``h_N2F``
    (``reverse_N`` / ``per_token_T1`` are its backward's, here unused)."""
    del reverse_N, per_token_T1
    n, f = gather_N.numel(), w13_EOI.shape[1] // 2
    a_NF = x_TD.new_empty(n, f)
    h_N2F = x_TD.new_empty(n, 2 * f)
    _sonic().up_forward(
        x=x_TD,
        w1=w13_EOI.permute(1, 2, 0),
        h=h_N2F,
        a=a_NF,
        b1=None,
        expert_frequency_offset=offset_E1,
        x_gather_idx=gather_N,
        activation_type=_sonic().activation,
        is_inference_mode_enabled=False,
        concat_layout=True,
    )
    return a_NF, h_N2F


@_up.register_fake
def _(x_TD, w13_EOI, offset_E1, gather_N, reverse_N, per_token_T1):
    n, f = gather_N.numel(), w13_EOI.shape[1] // 2
    return x_TD.new_empty(n, f), x_TD.new_empty(n, 2 * f)


@torch.library.custom_op("opt_moe::sonic_up_backward", mutates_args=())
def _up_backward(
    dh_N2F: torch.Tensor,
    x_TD: torch.Tensor,
    w13_EOI: torch.Tensor,
    offset_E1: torch.Tensor,
    gather_N: torch.Tensor,
    reverse_N: torch.Tensor,
    per_token_T1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The input gradient [T, D] and the weight gradient (``w13_EOI``'s shape)."""
    sonic = _sonic()
    num_experts, o, d = w13_EOI.shape
    dx_expanded_ND = dh_N2F.new_empty(gather_N.numel(), d)
    dw13_EOI = torch.empty_like(w13_EOI)
    dh_N2F = dh_N2F.contiguous()
    sonic.up_backward_act(
        w1=w13_EOI.permute(1, 2, 0),
        dx_expanded=dx_expanded_ND,
        dh=dh_N2F,
        db1=None,
        expert_frequency_offset=offset_E1,
        is_glu_activation=True,
        concat_layout=True,
    )
    sonic.up_backward_weight(
        x=x_TD,
        dw1=dw13_EOI.permute(1, 2, 0),
        dh=dh_N2F,
        expert_frequency_offset=offset_E1,
        x_gather_idx=gather_N,
        is_glu_activation=True,
        concat_layout=True,
    )
    dx_TD = dh_N2F.new_empty(x_TD.shape)
    sonic.token_broadcast_backward(
        dx_reduced=dx_TD,
        dx_expanded=dx_expanded_ND,
        s_reverse_scatter_idx=reverse_N,
        num_activated_expert_per_token_offset=per_token_T1,
        varlen_K_max=num_experts,
        H=d,
        is_varlen_K=True,
    )
    return dx_TD, dw13_EOI


@_up_backward.register_fake
def _(dh_N2F, x_TD, w13_EOI, offset_E1, gather_N, reverse_N, per_token_T1):
    return dh_N2F.new_empty(x_TD.shape), w13_EOI.new_empty(w13_EOI.shape)


@torch.library.custom_op("opt_moe::sonic_down", mutates_args=())
def _down(
    a_NF: torch.Tensor,
    h_N2F: torch.Tensor,
    w2_EDI: torch.Tensor,
    scores_N: torch.Tensor,
    offset_E1: torch.Tensor,
    gather_N: torch.Tensor,
    scatter_N: torch.Tensor,
    reverse_N: torch.Tensor,
    per_token_T1: torch.Tensor,
) -> torch.Tensor:
    """The down projection of ``a_NF`` and the combine
    ``out[t] = sum over t's pairs of score * row`` (fp32 sum, rounded once)."""
    del h_N2F, gather_N, scatter_N  # its backward's
    sonic = _sonic()
    num_experts, d, _ = w2_EDI.shape
    y_ND = a_NF.new_empty(a_NF.shape[0], d)
    sonic.down_forward(
        w2=w2_EDI.permute(1, 2, 0),
        a=a_NF,
        y=y_ND,
        b2=None,
        expert_frequency_offset=offset_E1,
    )
    out_TD = a_NF.new_empty(per_token_T1.numel() - 1, d)
    sonic.router_forward(
        y=y_ND,
        o=out_TD,
        topk_scores=scores_N,
        s_reverse_scatter_idx=reverse_N,
        num_activated_expert_per_token_offset=per_token_T1,
        varlen_K_max=num_experts,
        H=d,
        is_varlen_K=True,
    )
    return out_TD


@_down.register_fake
def _(
    a_NF,
    h_N2F,
    w2_EDI,
    scores_N,
    offset_E1,
    gather_N,
    scatter_N,
    reverse_N,
    per_token_T1,
):
    return a_NF.new_empty(per_token_T1.numel() - 1, w2_EDI.shape[1])


@torch.library.custom_op("opt_moe::sonic_down_backward", mutates_args=())
def _down_backward(
    dout_TD: torch.Tensor,
    h_N2F: torch.Tensor,
    w2_EDI: torch.Tensor,
    scores_N: torch.Tensor,
    offset_E1: torch.Tensor,
    gather_N: torch.Tensor,
    scatter_N: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gradients of ``h_N2F``, ``w2_EDI`` and the scores [N]; recomputes the
    post-activation from ``h_N2F`` (SonicMoE's ``_DownProjection.backward``)."""
    sonic = _sonic()
    dout_TD = dout_TD.contiguous()
    dh_N2F = torch.empty_like(h_N2F)
    dw2_EDI = torch.empty_like(w2_EDI)
    ds_N = torch.empty_like(scores_N)
    a_prime_NF = h_N2F.new_empty(gather_N.numel(), w2_EDI.shape[2])
    sonic.down_backward_act(
        dout=dout_TD,  # codespell:ignore dout
        h=h_N2F,
        w2=w2_EDI.permute(1, 2, 0),
        dh=dh_N2F,
        ds=ds_N,
        b2=None,
        db2=None,
        a_prime=a_prime_NF,
        topk_scores=scores_N,
        expert_frequency_offset=offset_E1,
        x_gather_idx=gather_N,
        s_scatter_idx=scatter_N,
        activation_type=sonic.activation,
    )
    sonic.down_backward_weight(
        dout=dout_TD,  # codespell:ignore dout
        a_prime=a_prime_NF,
        dw2=dw2_EDI.permute(1, 2, 0),
        expert_frequency_offset=offset_E1,
        x_gather_idx=gather_N,
    )
    return dh_N2F, dw2_EDI, ds_N


@_down_backward.register_fake
def _(dout_TD, h_N2F, w2_EDI, scores_N, offset_E1, gather_N, scatter_N):
    return (
        h_N2F.new_empty(h_N2F.shape),
        w2_EDI.new_empty(w2_EDI.shape),
        scores_N.new_empty(scores_N.shape),
    )


def _up_setup(ctx, inputs, output):
    x_TD, w13_EOI, offset_E1, gather_N, reverse_N, per_token_T1 = inputs
    ctx.save_for_backward(x_TD, w13_EOI, offset_E1, gather_N, reverse_N, per_token_T1)
    # The post-activation gets no gradient (SonicMoE: non-differentiable): skip
    # materialising a zero [N, F] for it.
    ctx.set_materialize_grads(False)


def _up_autograd(ctx, grad_a, grad_h_N2F):
    x_TD, w13_EOI, offset_E1, gather_N, reverse_N, per_token_T1 = ctx.saved_tensors
    dx_TD, dw13_EOI = _up_backward(
        grad_h_N2F, x_TD, w13_EOI, offset_E1, gather_N, reverse_N, per_token_T1
    )
    return dx_TD, dw13_EOI, None, None, None, None


_up.register_autograd(_up_autograd, setup_context=_up_setup)


def _down_setup(ctx, inputs, output):
    (
        _,
        h_N2F,
        w2_EDI,
        scores_N,
        offset_E1,
        gather_N,
        scatter_N,
        _,
        _,
    ) = inputs
    ctx.save_for_backward(h_N2F, w2_EDI, scores_N, offset_E1, gather_N, scatter_N)


def _down_autograd(ctx, grad_out_TD):
    h_N2F, w2_EDI, scores_N, offset_E1, gather_N, scatter_N = ctx.saved_tensors
    dh_N2F, dw2_EDI, ds_N = _down_backward(
        grad_out_TD, h_N2F, w2_EDI, scores_N, offset_E1, gather_N, scatter_N
    )
    return None, dh_N2F, dw2_EDI, ds_N, None, None, None, None, None


_down.register_autograd(_down_autograd, setup_context=_down_setup)


def sonic_experts(
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    w13_EOI: torch.Tensor,
    w2_EDI: torch.Tensor,
) -> torch.Tensor:
    """``sum over the token's K experts of score * (swiglu(x @ w13.T) @ w2.T)`` as
    SonicMoE's three ops; ``w13_EOI`` [E, 2F, D] (gate rows, then up rows) and
    ``w2_EDI`` [E, D, F] bf16 contiguous, ``x_TD`` bf16, ``topk_scores_TK`` fp32.
    Gradients: x, both weights and the scores."""
    offset_E1, gather_N, scatter_N, reverse_N, per_token_T1 = _metadata(
        topk_expert_ids_TK, w13_EOI.shape[0]
    )
    a_NF, h_N2F = _up(x_TD, w13_EOI, offset_E1, gather_N, reverse_N, per_token_T1)
    return _down(
        a_NF,
        h_N2F,
        w2_EDI,
        topk_scores_TK.reshape(-1),
        offset_E1,
        gather_N,
        scatter_N,
        reverse_N,
        per_token_T1,
    )


def check_sonicmoe_dims(dim: int, moe_hidden_dim: int) -> None:
    """The build-time check of the dims SonicMoE's TMA-based kernels take."""
    if dim % SONICMOE_ALIGN or moe_hidden_dim % SONICMOE_ALIGN:
        raise ValueError(
            'expert_gemm="sonicmoe" needs dim and moe_hidden_dim multiples of '
            f"{SONICMOE_ALIGN} (16-byte rows of bf16), got {dim} / {moe_hidden_dim}"
        )


class SonicMoERoutedExperts(RoutedExperts):
    """`RoutedExperts` whose dispatch, grouped GEMMs, SwiGLU and combine are
    SonicMoE's (see the module docstring). Same parameters and submodules
    (``w13``, ``w2``, ``token_dispatcher``, ...) as `RoutedExperts`; the
    dispatcher and ``activation_fn`` are built but not called.

    The experts are the one rematerialisation region
    ``<fqn>.sonicmoe``. ``recompute`` (the model decides per layer, see
    ``build_opt_moe_layers`` ``sonic_state`` / ``sonic_store_layers``) selects
    it, not the activation_remat patterns: True: the region is rerun in the
    backward and nothing of it is held; False: it is saved (autograd keeps the
    pre-activation and the routing metadata). Outside a ``remat.checkpoint``
    the flag has no effect (everything is held)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        recompute: bool = True

        def __post_init__(self) -> None:
            RoutedExperts.Config.__post_init__(self)  # slots: no zero-arg super()
            if not isinstance(self.activation_fn, SwiGLU.Config):
                raise ValueError(
                    "SonicMoERoutedExperts fuses SwiGLU into its GEMM: activation_fn "
                    f"must be SwiGLU.Config, got {type(self.activation_fn).__name__}"
                )
            if self.output_postprocess is not None:
                raise ValueError("SonicMoERoutedExperts has no output_postprocess")
            check_sonicmoe_dims(self.w13.in_features, self.w13.out_features)

    def __init__(self, config: Config):
        super().__init__(config)
        self.recompute = config.recompute
        _sonic()  # fail at build time when SONICMOE_SITE is unset or not importable

    def _experts(self, x_TD, topk_scores_TK, topk_expert_ids_TK, w13_EOI, w2_EDI):
        """The region's body: bf16 operands as the other backends take them."""
        return sonic_experts(
            x_TD.bfloat16().contiguous(),
            topk_scores_TK.float().contiguous(),
            topk_expert_ids_TK,
            w13_EOI.bfloat16().contiguous(),
            w2_EDI.bfloat16().contiguous(),
        ).to(x_TD.dtype)

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        routing_map_TE: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`RoutedExperts.forward` on SonicMoE; ``routing_map_TE`` and
        ``padding_mask_T`` are unused (as there, padding rows are computed like
        real tokens)."""
        del routing_map_TE, padding_mask_T
        # Read each weight once (a data parallel parametrization all-gathers on
        # every read); [E, 2, F, D] -> [E, 2F, D] is a view.
        w13_EOI = self.w13.weight.flatten(1, -2)
        w2_EDI = self.w2.weight
        return remat.region(
            self._experts,
            self.remat_region_name("sonicmoe"),
            recompute=self.recompute,
        )(x_TD, topk_scores_TK, topk_expert_ids_TK, w13_EOI, w2_EDI)
