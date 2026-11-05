# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses as _dc
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn

from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.local_compile import local_compile, LocalCompileConfig

from torchtitan.models.common.attention import (
    AttentionMetadata,
    AttentionMetadataMap,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import Identity
from torchtitan.models.common.param_layout import (
    by_rank,
    named,
    ParamLayout,
    stacked_units,
)
from torchtitan.models.common.rope import RoPE
from .gated_norm_swattention import AttentionSpec, build_attention_config
from .hf_assests import setup_hf
from .norm_moe import NormMoE
from .state_dict_adapter import OPTMoEStateDictAdapter
from .utils.inits import (
    make_param_init,
    parse_depth_init,
    setup_depth_init,
    setup_residual_scale,
)
from .utils.norms import build_norm_config


logger = logging.getLogger(__name__)


def _parse_layer_pattern(
    pattern: "str | list[bool] | list[str] | None",
    n_layers: int,
    true_char: str,
    false_char: str,
    default_true: bool = False,
) -> "list[bool]":
    """Parse a per-layer pattern into a flat list of booleans.

    Args:
        pattern: A string (one char per layer, e.g. ``'SSSF'`` or ``'RRRN'``),
                 a ``list[bool]``, a list-wrapped string (e.g. ``['SSSF']``),
                 or ``None``.
        n_layers: Expected number of layers; length is validated.
        true_char: Character that maps to ``True`` (case-insensitive).
        false_char: Character that maps to ``False`` (case-insensitive).
        default_true: Value returned for every layer when ``pattern`` is ``None``.
    """
    if pattern is None:
        return [default_true] * n_layers
    if isinstance(pattern, list):
        # Some config frontends pass single string values as one-item lists.
        # Accept both ['SSSF'] and ['S','S','S','F'] in addition to list[bool].
        if len(pattern) == 1 and isinstance(pattern[0], str):
            pattern = pattern[0]
        elif pattern and all(isinstance(x, str) and len(x) == 1 for x in pattern):
            pattern = "".join(pattern)
        else:
            if len(pattern) != n_layers:
                raise ValueError(
                    f"Pattern list length {len(pattern)} != n_layers {n_layers}"
                )
            if not all(isinstance(x, bool) for x in pattern):
                raise ValueError(
                    "Pattern list must be list[bool], ['PATTERN'], or list of single-character strings."
                )
            return list(pattern)
    # String path
    allowed = {true_char.upper(), false_char.upper()}
    pattern_up = pattern.upper()
    if len(pattern_up) != n_layers:
        raise ValueError(
            f"Pattern string length {len(pattern_up)} != n_layers {n_layers}"
        )
    invalid = set(pattern_up) - allowed
    if invalid:
        raise ValueError(
            f"Invalid characters {invalid!r} in pattern. Expected only {allowed!r}."
        )
    return [c == true_char.upper() for c in pattern_up]


def _layout_flags(parallelism: ParallelismConfig) -> dict[str, bool]:
    """What the model config's sharding depends on (data parallel does not)."""
    tp = parallelism.tensor_parallel_degree > 1
    return {
        "tp": tp,
        "cp": parallelism.context_parallel_degree > 1,
        "ep": parallelism.expert_parallel_degree > 1,
        "sp": parallelism.enable_sequence_parallel and tp,
    }


def _active_weights(model: nn.Module, params) -> dict:
    """Each parameter's active share: 1, or a NormMoE's top_k / num_experts for
    its routed experts (upstream's `_get_active_parameter_weights` for NormMoE)."""
    from fractions import Fraction

    w = {id(p): Fraction(1) for p in params}
    for m in model.modules():
        if isinstance(m, NormMoE):
            ratio = Fraction(m.router.top_k, m.router.num_experts)
            for p in m.routed_experts.parameters():
                w[id(p)] = ratio
    return w


class OPTMoETransformerBlock(TransformerBlock):
    """OPT MoE TransformerBlock.

    Token-flat throughout: ``x`` is ``[T, D]``. Returns ``output``
    so the model can accumulate per-layer MoE load-balance losses without
    threading a running tensor through every layer signature.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # ``TransformerBlock.Config`` requires these; OPT MoE builds its norms
        # from ``norm_type``/``norm_eps`` instead, so they stay unset.
        attention_norm: Any = None
        ffn_norm: Any = None
        # The template holds both; each expanded layer keeps one (`n_dense_layers`
        # first layers dense). The MoE is built as `self.moe` (FQNs `moe.*`).
        feed_forward: Any = None
        moe: Any = None

        n_dense_layers: int = 0
        init_gate_as_residual: bool = False
        depth_init: bool | str = "total_depth"
        residual_scale: str = "identity"
        norm_eps: float = 1e-30
        norm_type: str = "rmsnorm"
        prenorm_gain_init: float = 1.0
        """Constant init of the attention_norm / ffn_norm gains (``norm_type="rmsnorm"``
        only). 1.0 leaves the norms on their stock ``reset_parameters`` (ones)."""

        # Stamped per layer by OPTMoEModel.Config._expand_layers(). Upstream
        # builds each layer with a bare build(), so these can no longer be
        # build() kwargs.
        dim: int = 0
        layer_id: int = 0
        n_layers: int = 1

    def __init__(self, config: Config):
        super().__init__()
        dim = config.dim
        assert dim > 0, (
            "OPTMoETransformerBlock.Config.dim must be stamped by "
            "OPTMoEModel.Config._expand_layers() before build()."
        )
        self.layer_id = config.layer_id
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        # Per-layer attention mode (the model hands each layer the metadata of
        # its inner attention's type: full or sliding-window flex, or varlen).
        inner = config.attention.inner_attention
        self.use_swa: bool = isinstance(
            inner,
            (
                SlidingWindowFlexInnerAttention.Config,
                SlidingWindowVarlenInnerAttention.Config,
            ),
        )
        # Per-layer debug metadata surfaced in __repr__/print(model).
        self._repr_use_rope: bool = config.attention.rope is not None
        self._repr_swa_window_size: int = (
            inner.sliding_window_size if self.use_swa else -1
        )
        self._repr_rope_theta: float = (
            float(config.attention.rope.theta)
            if config.attention.rope is not None
            else -1.0
        )

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        # x = identity_scale * x + block_scale * block(x)
        self.block_scale, self.identity_scale = setup_residual_scale(
            config.residual_scale, config.n_layers
        )
        # Skip each multiply INDEPENDENTLY when its scale is 1.0.
        #
        # `1.0 * x` is what breaks AC + per-block compile: it makes `x` feed two
        # branches and the resulting gradient-accumulation add becomes a graph
        # output the min-cut partitioner rejects. Measured in pure torch, with
        # checkpoint_wrapper + per-block compile:
        #     a=1.0 b=1.0     -> FAIL        a=0.5 b=1.0 -> OK
        #     a=1.0 b=0.5     -> FAIL        a=0.5 b=0.5 -> OK
        # i.e. it fails iff identity_scale == 1.0. A NON-unit identity scale is
        # fine, so `depth_scale_R`/`depth_scale_N` were never affected.
        #
        # Requiring BOTH scales to be 1.0 would miss `complete_p_R`/`complete_p_N`,
        # which have identity_scale == 1.0 with a non-unit block_scale -- they
        # would still take the multiply path and still fail. Hence per-term.
        self._skip_identity_mul = self.identity_scale == 1.0
        self._skip_block_mul = self.block_scale == 1.0

    def extra_repr(self) -> str:
        return (
            f"layer_id={self.layer_id}, "
            f"rope_theta={self._repr_rope_theta}, "
            f"use_rope={self._repr_use_rope}, "
            f"qk_rope_dim={self.attention.qk_rope_dim}, "
            f"swa_window_size={self._repr_swa_window_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: AttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        doc_id: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            x: Input tensor ``[T, D]``.
            attention_masks: Mask(s) for this layer; a dict keyed ``full``/``swa``
                when layers mix backends, or a single mask.
            positions: Optional position indices ``[T]``.
            loss_mask: Optional token loss mask for the MoE load-balance loss.
            padding_mask: Upstream's per-token padding flag (#4594), accepted
                because ``Decoder.forward`` now passes it to every layer. NOT
                used here: opt_moe derives its own ``loss_mask`` from
                ``labels != IGNORE_INDEX`` in ``preprocess_inputs``, which is
                not the same signal. Under SFT a prompt token has
                ``IGNORE_INDEX`` labels but is not padding, so ours excludes
                prompt tokens from the load-balance statistics while upstream's
                would include them. Switching to upstream's mask is a numerics
                change, not a rename.

        Returns:
            ``output``. The load-balance gradient is injected inside the MoE
            (see norm_moe.LoadBalanceLoss), not returned.
        """
        # Unit scales are skipped, not multiplied. `residual_scale="identity"`
        # gives block_scale == identity_scale == 1.0, so `1.0 * x + 1.0 * f(x)`
        # is mathematically `x + f(x)` -- but the explicit multiplies make `x`
        # feed two branches, and the resulting gradient-accumulation node breaks
        # the min-cut partitioner under activation checkpointing + per-block
        # compile: "AssertionError: Node add_N was invalid, but is output"
        # (torch 2.15.0.dev20260906). Reproduced in pure torch: a*x + b*f(x)
        # under checkpoint_wrapper + per-block compile fails, x + f(x) does not.
        # Taking the plain path when the scales are unit costs nothing and
        # restores AC. Non-unit scales still hit the torch bug -- see the
        # migration doc.
        _attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(_attn_out)
        h, h_normed = self._attention_residual_and_ffn_norm(x, _attn_out)

        if self.moe_enabled:
            mlp_output = self.moe(
                h_normed,
                loss_mask,
                positions=positions,
                doc_id=doc_id,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            mlp_output = self.feed_forward(h_normed)

        _lhs = h if self._skip_identity_mul else self.identity_scale * h
        if not self._skip_block_mul:
            # The scale reads the feed-forward output with a bare op.
            remat.recompute_needs_tensor(mlp_output)
            mlp_output = self.block_scale * mlp_output
        # Trailing add, always saved: it saves nothing for backward, so replay
        # skips it and its inputs need no persisting (as upstream's blocks).
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(_lhs, mlp_output)

    # Limit 16: one graph per norm type and layer width, more under validation shapes.
    @local_compile("opt_moe_residual_norm", batch_invariant=False, recompile_limit=16)
    def _attention_residual_and_ffn_norm(
        self, x: torch.Tensor, attn_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention residual (scaled when the scales are not unit) and the
        feed-forward norm of its result, fused: ``(h, ffn_norm(h))``."""
        lhs = x if self._skip_identity_mul else self.identity_scale * x
        rhs = attn_out if self._skip_block_mul else self.block_scale * attn_out
        h = lhs + rhs
        return h, self.ffn_norm(h)


class OPTMoEModel(Decoder):
    """OPT MoE Transformer model with attention and feed-forward layers."""

    state_dict_adapter_cls = OPTMoEStateDictAdapter

    # Copies config.json / modeling_*.py / chat template into the checkpoint
    # dir so exports load with trust_remote_code. Was ModelSpec.hf_assets_setup_fn.
    hf_assets_setup_fn: ClassVar[Any] = staticmethod(
        setup_hf.copy_and_overwrite_model_config
    )

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context) -> None:
        from torchtitan.optimizers.moe_balance import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    def param_layouts(self) -> dict:
        """What every parameter is, keyed by FQN (see
        `models/common/param_layout.py`): modules declare what only they know
        (expert batches, fused projections, the router gate), the model the two
        vocabulary matrices, and the rest is stated by rank."""
        layouts = {}
        for module_name, module in self.named_modules():
            declare = getattr(module, "param_layouts", None)
            if module is self or declare is None:
                continue
            for param_name, layout in declare().items():
                layouts[
                    f"{module_name}.{param_name}" if module_name else param_name
                ] = layout
        for module in self.modules():
            if isinstance(module, RoutedExperts):  # upstream's: declared here
                w13 = module.w13.weight
                slots = stacked_units(("w1", "w3"), w13.shape[-2], w13.shape[-1])
                layouts[named(self, w13)] = ParamLayout("expert_batch", units=slots)
                layouts[named(self, module.w2.weight)] = ParamLayout("expert_batch")
        for module, kind in (
            (self.tok_embeddings, "embedding"),
            (self.lm_head, "unembedding"),
        ):
            if module is not None:  # PP stages hold one or neither
                layouts[named(self, module.weight)] = ParamLayout(kind)
        for name, p in self.named_parameters():
            if name not in layouts:
                layouts[name] = by_rank(p.shape)
        return layouts

    def parallelize(self, *, parallelism: ParallelismConfig, **kwargs):
        """Upstream's lifecycle, after checking that the model config was built
        for this parallelism (tp / cp / ep / sp)."""
        got = _layout_flags(parallelism)
        if self.config.built_for != got:
            raise ValueError(
                f"the model config was built for {self.config.built_for}, the run "
                f"has {got}: build it with build_model_config for the run's "
                "parallelism (a command-line override of these comes too late)"
            )
        return super().parallelize(parallelism=parallelism, **kwargs)

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        precision: PrecisionConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import (
            apply_fsdp_to_decoder,
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
        apply_fsdp_to_decoder(
            self,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
            reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
            pp_enabled=parallelism_context.pp_enabled,
            cpu_offload=parallelism.fsdp_enable_cpu_offloading,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallelism_context.ep,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
            # DiSCO runs each expert's rule on the rank's LOCAL expert rows
            # with no gather, so a rank must own whole expert matrices.
            expert_shard_dim=0,
        )

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # Were ModelSpec.name/.flavor; kept here so they survive into the
        # dumped job_config, which convert_to_hf and the evaluator read back.
        name: str = "opt_moe"
        flavor: str = ""
        max_context_length: int = 8192
        dim: int = 2048
        n_layers: int = 24
        vocab_size: int = 201088
        layer: TransformerBlock.Config
        # Upstream's regions the model passes through plus opt_moe's own
        # elementwise code (0.5.4 compiled whole blocks instead). Also defined,
        # off until measured on the production dims: "opt_moe_attention_qkv"
        # (conv -> q/k norm -> RoPE in one region) and "opt_moe_residual_norm".
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "fused_binary_activation",
                    "fp32_to_bf16_split",
                    "rope",
                    "opt_moe_short_conv",
                    "opt_moe_sink_rescale",
                    "opt_moe_attention_output",
                ]
            )
        )

        # Upstream's Decoder.Config requires these outright. OPT MoE describes a
        # model with one `layer` template plus per-layer patterns, so they are
        # derived in _expand_layers() instead of being spelled out per flavor.
        layers: list = field(default_factory=list)
        lm_head: Any = None
        tok_embeddings: Any = None
        norm: Any = None

        rope: RoPE.Config
        norm_eps: float = 1e-30
        norm_type: str = "rmsnorm"
        norm_sharding_config: Any | None = None
        """Sharding plan for `embeddings_norm`, which this config builds
        internally."""

        first_in_init_fn_type: str = "scion_normal_input"
        first_in_init_std: float = 1.0

        final_out_init_fn_type: str = "scion_normal_output"
        final_out_init_std: float = 1.0

        use_embeddings_norm: bool = False

        # Set by finalize (build_model_config); consumed by
        # preprocess_inputs to build the MoE load-balance token mask.
        enable_token_mask_for_moe: bool = False

        rope_of_swa: RoPE.Config | None = None
        # RoPE cache for SWA layers (typically a lower theta for local context).
        # The global ``rope`` config applies to full-attention layers.
        # None → all layers share the global ``rope`` cache.
        # Ignored for NoPE SWA layers.

        rope_pattern: str | list[bool] | None = None
        # Per-layer RoPE vs NoPE selection.
        # String: one char per layer — 'R' = RoPE, 'N' = NoPE.  E.g. ``"RRRN"``.
        # list[bool]: True = RoPE, False = NoPE.
        # None: RoPE on every layer. A layer's RoPE is ``rope`` (full-attention
        # layers) or ``rope_of_swa`` (SWA layers, when set); partial RoPE when its
        # ``dim`` is below head_dim.

        built_for: dict[str, bool] | None = None
        """Set by `finalize`: the parallelism (tp / cp / ep / sp on or off) the
        per-layer configs were built for; `parallelize` refuses another one
        (a CLI override of these after the build would otherwise be ignored)."""

        swa_pattern: str | list[bool] | None = None
        # Per-layer sliding-window vs full-attention selection.
        # String: one char per layer — 'S' = SWA, 'F' = Full attention.
        # None keeps the uniform behaviour from ``layer.attention``.
        # SWA layers are automatically assigned attn_backend="flex".

        def _expand_layers(self, *, enable_sp: bool) -> None:
            """Expand the ``layer`` template into upstream's per-layer config list.

            Upstream builds each block from a fully-specified entry in
            ``Decoder.Config.layers`` via a bare ``build()``, so everything the
            old code passed as a build kwarg or an ``init_weights`` argument --
            dim, layer_id, n_layers, the depth-init divisors, the RoPE cache and
            the inner-attention backend -- has to be stamped onto a per-layer
            copy here.
            """
            n_layers = self.n_layers
            base_attn = self.layer.attention
            assert isinstance(base_attn, AttentionSpec)

            use_rope = _parse_layer_pattern(
                self.rope_pattern, n_layers, "R", "N", default_true=True
            )
            use_swa = _parse_layer_pattern(
                self.swa_pattern,
                n_layers,
                "S",
                "F",
                default_true=base_attn.sliding_window_size > 0,
            )
            swa_window = base_attn.sliding_window_size
            depth_init = parse_depth_init(self.layer.depth_init)

            def pre_norm():
                cfg = build_norm_config(
                    self.layer.norm_type, self.dim, self.layer.norm_eps
                )
                if self.layer.prenorm_gain_init != 1.0:
                    if self.layer.norm_type != "rmsnorm":
                        raise ValueError(
                            "prenorm_gain_init != 1 needs norm_type='rmsnorm' "
                            f"(got {self.layer.norm_type!r}): no gain to initialise"
                        )
                    cfg.param_init = {
                        "weight": functools.partial(
                            nn.init.constant_, val=self.layer.prenorm_gain_init
                        )
                    }
                return cfg

            layers: list = []
            for layer_id in range(n_layers):
                residual_div_attn, residual_div_ffn = setup_depth_init(
                    depth_init, layer_id, n_layers
                )
                # SWA layers use the local RoPE cache when one is configured.
                layer_rope = (
                    self.rope_of_swa
                    if (use_swa[layer_id] and self.rope_of_swa is not None)
                    else self.rope
                )
                attn_cfg = build_attention_config(
                    # SWA requires FlexInnerAttention; non-SWA keeps the
                    # configured backend.
                    _dc.replace(base_attn, attn_backend="flex")
                    if use_swa[layer_id]
                    else base_attn,
                    dim=self.dim,
                    # None on a NoPE layer (rope_pattern "N"): upstream's
                    # GQAttention builds no RoPE then.
                    rope=layer_rope if use_rope[layer_id] else None,
                    sliding_window_size=swa_window if use_swa[layer_id] else -1,
                    residual_div=residual_div_attn,
                    enable_sp=enable_sp,
                )

                is_dense = layer_id < self.layer.n_dense_layers
                moe_cfg = None
                ff_cfg = None
                if is_dense:
                    assert self.layer.feed_forward is not None, (
                        f"layer {layer_id} is dense (n_dense_layers="
                        f"{self.layer.n_dense_layers}) but no feed_forward config is set"
                    )
                    ff_cfg = _dc.replace(
                        self.layer.feed_forward,
                        dim=self.dim,
                        residual_div=residual_div_ffn,
                        init_gate_as_residual=self.layer.init_gate_as_residual,
                    )
                else:
                    assert (
                        self.layer.moe is not None
                    ), f"layer {layer_id} is an MoE layer but no moe config is set"
                    # Flat authoring config -> upstream-shaped NormMoE.Config,
                    # so update_ep_token_dispatcher_config can find
                    # routed_experts and swap in an EP dispatcher.
                    moe_cfg = _dc.replace(
                        self.layer.moe,
                        dim=self.dim,
                        layer_id=layer_id,
                        residual_div=residual_div_ffn,
                        init_gate_as_residual=self.layer.init_gate_as_residual,
                    ).to_norm_moe_config()

                layers.append(
                    _dc.replace(
                        self.layer,
                        attention=attn_cfg,
                        moe=moe_cfg,
                        feed_forward=ff_cfg,
                        attention_norm=pre_norm(),
                        ffn_norm=pre_norm(),
                        dim=self.dim,
                        layer_id=layer_id,
                        n_layers=n_layers,
                    )
                )
            self.layers = layers
            self.norm = build_norm_config(self.norm_type, self.dim, self.norm_eps)

            # Decoder.__init__ builds these directly; OPT MoE overrides `norm`
            # with its own build_norm, but tok_embeddings/lm_head are stock.
            self.tok_embeddings = Embedding.Config(
                num_embeddings=self.vocab_size,
                embedding_dim=self.dim,
                param_init={
                    "weight": make_param_init(
                        self.first_in_init_fn_type, self.first_in_init_std
                    )
                },
            )
            self.lm_head = Linear.Config(
                in_features=self.dim,
                out_features=self.vocab_size,
                param_init={
                    "weight": make_param_init(
                        self.final_out_init_fn_type, self.final_out_init_std
                    )
                },
            )

        def finalize(
            self,
            *,
            parallelism: ParallelismConfig,
            max_context_length: int,
            rope_length: int | None = None,
            enable_token_mask_for_moe: bool | None = None,
        ) -> None:
            """Expand this flat spec into the complete per-layer config, sharded
            for `parallelism` (upstream builds model configs complete; nothing
            changes them later). `rope_length`: the RoPE caches' length when
            longer than the training context (validation sequences index them).
            Call through `build_model_config`."""
            if self.layers:
                raise ValueError("model config already finalized")
            if max_context_length > self.rope.max_context_length:
                logger.warning(
                    f"Context length {max_context_length} exceeds original "
                    f"maximum {self.rope.max_context_length}."
                )
            self.max_context_length = max_context_length
            rope_length = max(max_context_length, rope_length or 0)
            self.rope = _dc.replace(self.rope, max_context_length=rope_length)
            if self.rope_of_swa is not None:
                self.rope_of_swa = _dc.replace(
                    self.rope_of_swa, max_context_length=rope_length
                )
            if enable_token_mask_for_moe is not None:  # None: the spec's own value
                self.enable_token_mask_for_moe = enable_token_mask_for_moe

            base_attn = self.layer.attention
            assert isinstance(base_attn, AttentionSpec)
            use_swa = _parse_layer_pattern(self.swa_pattern, self.n_layers, "S", "F")
            uses_swa = any(use_swa) or (
                self.swa_pattern is None and base_attn.sliding_window_size > 0
            )
            if uses_swa and base_attn.sliding_window_size <= 0:
                raise ValueError(
                    "SWA is enabled but layer.attention.sliding_window_size "
                    "is not set (must be > 0)."
                )
            cp = parallelism.context_parallel_degree > 1
            tp = parallelism.tensor_parallel_degree > 1
            self.built_for = _layout_flags(parallelism)
            if (base_attn.conv_kernel_size > 0 or base_attn.attention_sinks) and (
                tp or cp
            ):
                raise NotImplementedError(
                    "the attention short conv and sinks support neither TP nor CP"
                )
            if cp and base_attn.attn_backend not in ("flex", "flex_flash"):
                raise ValueError(
                    "Context Parallel needs the flex attention backend (got "
                    f"attn_backend={base_attn.attn_backend!r})"
                )
            self._expand_layers(enable_sp=parallelism.enable_sequence_parallel and tp)

            from torchtitan.models.opt_moe.sharding import set_opt_moe_sharding_config

            # enable_sequence_parallel defaults to True regardless of TP: gate on
            # the degree, or a plain FSDP run would look like TP to the plan.
            set_opt_moe_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel and tp,
                enable_tp=tp,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )
            if cp:  # the CP versions of the flex inner attentions (all-gather KV)
                from torchtitan.config.transform.context_parallel import (
                    ContextParallelTransform,
                )
                from torchtitan.models.common.attention import (
                    FlexInnerAttention,
                    SlidingWindowFlexInnerAttention,
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
                ).transform(self)

        def get_parameter_counts(self, model: nn.Module) -> tuple[int, int]:
            """Total and active parameters, the routed experts of each NormMoE at
            its top_k / num_experts (upstream's count knows only its own MoE)."""
            params = list(model.parameters())
            w = _active_weights(model, params)
            active = sum(p.numel() * w[id(p)] for p in params)
            assert active.denominator == 1, active
            return sum(p.numel() for p in params), active.numerator

        def build_flops_estimator(
            self,
            model: nn.Module,
            *,
            seq_len: int,
            excluded_submodules: tuple = (),
        ):
            """Upstream's estimate (6 x active parameters, lookup-only embeddings
            excluded, plus each layer's attention at its own window), with the
            NormMoE expert weighting."""
            from fractions import Fraction

            params = list(model.parameters())
            w = _active_weights(model, params)
            head = getattr(model, "lm_head", None)
            head_ids = (
                {id(p) for p in head.parameters()} if isinstance(head, nn.Module) else set()
            )
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    for p in m.parameters(recurse=False):
                        if id(p) not in head_ids:
                            w[id(p)] = Fraction(0)
            for m in excluded_submodules:
                if m is not None:
                    for p in m.parameters():
                        w[id(p)] = Fraction(0)
            active = sum(p.numel() * w[id(p)] for p in params)
            assert active.denominator == 1, active
            attention = sum(  # the expanded layers (not the `layer` template)
                layer.attention.flops_per_token(seq_len) for layer in self.layers
            )
            per_token = 6 * active.numerator + attention
            return lambda input_dict: per_token * input_dict["input"].numel()

    def __init__(self, config: Config):
        super().__init__(config)
        if config.use_embeddings_norm:
            self.embeddings_norm = build_norm_config(
                config.norm_type,
                config.dim,
                config.norm_eps,
                sharding_config=config.norm_sharding_config,
            ).build()
        else:
            self.embeddings_norm = Identity.Config().build()
        # Set to [] by the optimizer container on logging steps (log_residual_stats);
        # _trunk then appends each forward's residual statistics. Not a buffer.
        self.residual_capture: list[torch.Tensor] | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        loss_mask: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            tokens: Input token indices ``[T]``, or hidden states when this is
                not the first pipeline stage.
            positions: Position indices ``[T]``.
            attention_masks: Per-layer masks (dict) or a single mask.
            loss_mask: Token mask for the MoE load-balance loss.
            padding_mask: Accepted and ignored; see
                ``OPTMoETransformerBlock.forward`` for why it is not
                interchangeable with ``loss_mask``.

        Returns:
            ``output``.
        """
        h = self._trunk(
            tokens, positions, attention_metadata, loss_mask, aux_loss_denominators
        )
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h

    def _trunk(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        loss_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Everything up to and including the final norm: the head's input,
        which ``forward`` returns as is under the chunked loss."""
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        # PP nulls any root child not in [tok_embeddings, layers.*, norm,
        # lm_head] (distributed/pipeline_parallel.py::_split_module), and
        # `embeddings_norm` is not in that list, so under PP every stage
        # gets None here and an unguarded call raises. Guarding keeps PP
        # from crashing; note a stage-0 `use_embeddings_norm=True` model
        # would then lose the norm, so PP + use_embeddings_norm still
        # needs it folded into tok_embeddings. Not done: PP is unused.
        if self.embeddings_norm is not None:
            h = self.embeddings_norm(h)

        # The load-balance loss is no longer returned: NormMoE injects its
        # gradient at the layer (see LoadBalanceLoss), so nothing has to be
        # threaded through the blocks or carried across pipeline stages.
        # The residual statistics are taken here, in the eager loop between the
        # compiled blocks, so the blocks and their graphs are the same either way.
        rows = [] if self.residual_capture is not None else None
        # Upstream's aux-loss normaliser: the step-global routed-token count (one
        # per block; every block routes the same tokens), as Decoder.forward.
        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        # Document ids for the MoE's sequence-wise statistics: a document starts
        # where `positions` does not continue the previous token's.
        doc_id = None
        if positions is not None:
            flat = positions.reshape(-1)
            starts = torch.ones_like(flat, dtype=torch.bool)
            starts[1:] = flat[1:] != (flat[:-1] + 1)
            doc_id = torch.cumsum(starts, dim=0).to(torch.int32).reshape(positions.shape)
        for layer in self.layers.values():
            meta = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention))
            )
            out = layer(
                h,
                meta,
                positions,
                loss_mask,
                doc_id,
                aux_loss_denominator=aux_loss_denominator,
            )
            if rows is not None:
                with torch.no_grad():
                    hf = h.float()
                    rows.append(
                        torch.stack(
                            [hf.square().mean(), (out.float() - hf).square().mean()]
                        )
                    )
            h = out
        if rows is not None:
            with torch.no_grad():
                # [ms(input), ms(write) per block..., ms(final norm input)]
                self.residual_capture.append(
                    torch.cat(
                        [torch.stack(rows).flatten(), h.float().square().mean()[None]]
                    )
                )

        return self.norm(h) if self.norm is not None else h

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        """Upstream's (masks, CP shards, SPMD types), then the MoE token mask,
        from the CP-sharded labels so its token layout matches the hidden
        states the router sees."""
        inputs, labels, model_kwargs, loss_kwargs = super().preprocess_inputs(
            input_dict,
            parallelism_context=parallelism_context,
            parallelism=parallelism,
            max_num_documents=max_num_documents,
            global_routing_token_counts=global_routing_token_counts,
            **kwargs,
        )
        if self.config.enable_token_mask_for_moe:
            from torchtitan.components.loss import IGNORE_INDEX

            model_kwargs["loss_mask"] = labels != IGNORE_INDEX
        return inputs, labels, model_kwargs, loss_kwargs
