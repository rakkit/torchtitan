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

import torch
from torch import nn

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.transform import convert_config_type
from torchtitan.distributed.parallel_dims import ParallelDims

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
    ScaledDotProductInnerAttention,
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
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from .gated_norm_swattention import GatedNormSWAttention
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

        # Per-layer attention-mode flags (derived from the per-layer attention config).
        assert isinstance(config.attention, GatedNormSWAttention.Config)
        self.use_swa: bool = config.attention.sliding_window_size > 0
        self.attn_backend: str = config.attention.attn_backend

        # Pre-compute the mask dict key used in forward() so we avoid string
        # comparisons and conditional logic on every training step.
        #   "swa" → FlexInnerAttention with sliding-window mask
        #   "full"→ FlexInnerAttention with full-causal mask
        # SDPA is not reachable through `build_inner_attention` any more (it
        # delegates to upstream's `get_attention_config`, which raises for
        # language models), but a hand-built `inner_attention=
        # ScaledDotProductInnerAttention.Config()` could still land here. Refuse it
        # rather than silently returning a None mask key, which is what made
        # attention span document boundaries in the first place.
        self._mask_key: str | None
        # "flex_flash" is also a flex backend: get_attention_config maps it to a
        # FlexInnerAttention.Config with FLASH kernel options, so it takes a mask.
        # Check the RESOLVED inner attention, not just the backend string: a
        # hand-built `inner_attention=ScaledDotProductInnerAttention.Config()` keeps
        # whatever `attn_backend` says while actually running SDPA, and SDPA
        # cannot take a mask at all (it only has a boolean is_causal), which is
        # exactly what made attention span document boundaries.
        _inner = getattr(config.attention, "inner_attention", None)
        _is_sdpa = isinstance(_inner, ScaledDotProductInnerAttention.Config)
        if self.attn_backend not in ("flex", "flex_flash", "varlen") or _is_sdpa:
            raise ValueError(
                f"attn_backend={self.attn_backend!r} / "
                f"inner_attention={type(_inner).__name__} cannot carry a "
                "document mask; token-flat batches need per-document positions. "
                "Use 'flex', 'flex_flash' or 'varlen'."
            )
        elif self.use_swa:
            self._mask_key = "swa"
        else:
            self._mask_key = "full"

        # Per-layer debug metadata surfaced in __repr__/print(model).
        self._repr_use_rope: bool = config.attention.rope is not None
        self._repr_swa_window_size: int = (
            config.attention.sliding_window_size if self.use_swa else -1
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
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        doc_id: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
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
        # _mask_key is pre-computed at init; no string comparisons or tensor ops
        # here. It is never None -- __init__ raises for the only backend that
        # produced None (sdpa) -- so there is no maskless path left.
        if isinstance(attention_masks, dict):
            layer_mask = attention_masks[self._mask_key]
        else:
            layer_mask = attention_masks  # single BlockMask (backward compat)

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
        _attn_out = self.attention(self.attention_norm(x), layer_mask, positions)
        _lhs = x if self._skip_identity_mul else self.identity_scale * x
        _rhs = _attn_out if self._skip_block_mul else self.block_scale * _attn_out
        h = _lhs + _rhs

        if self.moe_enabled:
            mlp_output = self.moe(
                self.ffn_norm(h), loss_mask, positions=positions, doc_id=doc_id
            )
        else:
            mlp_output = self.feed_forward(self.ffn_norm(h))

        _lhs = h if self._skip_identity_mul else self.identity_scale * h
        _rhs = mlp_output if self._skip_block_mul else self.block_scale * mlp_output
        return _lhs + _rhs


class OPTMoEModel(Decoder):
    """OPT MoE Transformer model with attention and feed-forward layers."""

    state_dict_adapter_cls = OPTMoEStateDictAdapter

    # Copies config.json / modeling_*.py / chat template into the checkpoint
    # dir so exports load with trust_remote_code. Was ModelSpec.hf_assets_setup_fn.
    hf_assets_setup_fn: ClassVar[Any] = staticmethod(
        setup_hf.copy_and_overwrite_model_config
    )

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallel_dims) -> None:
        from torchtitan.optimizers.moe_balance import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)

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

    def _apply_fsdp(
        self,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import (
            apply_fsdp_to_decoder,
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
        apply_fsdp_to_decoder(
            self,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
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
        internally. See GatedNormSWAttention.Config.norm_sharding_config."""

        first_in_init_fn_type: str = "scion_normal_input"
        first_in_init_std: float = 1.0

        final_out_init_fn_type: str = "scion_normal_output"
        final_out_init_std: float = 1.0

        use_embeddings_norm: bool = False

        # Mirrored from training config in update_from_config; consumed by
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

        swa_pattern: str | list[bool] | None = None
        # Per-layer sliding-window vs full-attention selection.
        # String: one char per layer — 'S' = SWA, 'F' = Full attention.
        # None keeps the uniform behaviour from ``layer.attention``.
        # SWA layers are automatically assigned attn_backend="flex".

        def _expand_layers(self) -> None:
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
            assert isinstance(base_attn, GatedNormSWAttention.Config)

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
                attn_backend = "flex" if use_swa[layer_id] else base_attn.attn_backend
                attn_cfg = _dc.replace(
                    base_attn,
                    dim=self.dim,
                    sliding_window_size=swa_window if use_swa[layer_id] else -1,
                    # SWA requires FlexInnerAttention; non-SWA keeps the configured backend.
                    attn_backend=attn_backend,
                    # None on a NoPE layer (rope_pattern "N"): upstream's
                    # GQAttention builds no RoPE then.
                    rope=layer_rope if use_rope[layer_id] else None,
                    residual_div=residual_div_attn,
                )
                attn_cfg.inner_attention = attn_cfg.build_inner_attention()

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

        def apply_parallelism_transforms(self, *, parallelism) -> None:
            """Rewrite the model config for the selected parallelism.

            Called from ``Trainer.Config.__post_init__`` BEFORE
            ``validate_model_training_config``, which is where upstream enforces
            "cp > 1 requires a CPInnerAttention". ``update_from_config`` is too
            late -- it runs after validation has already rejected the config.
            """
            if parallelism.context_parallel_degree <= 1:
                return

            # Upstream #4322 moved the k/v all-gather out of a local_map in
            # Module.forward() and into the attention implementation itself, so
            # CP is selected by swapping the inner attention. This mirrors
            # ContextParallelTransform; it runs here rather than in a recipe
            # because our context_parallel_degree comes from the CLI.
            from torchtitan.models.common.cp_attention import (
                KVAllGatherCPFlexInnerAttention,
            )

            base_attn = self.layer.attention
            assert isinstance(base_attn, GatedNormSWAttention.Config)
            if base_attn.attn_backend not in ("flex", "flex_flash"):
                raise ValueError(
                    "Context Parallel requires the flex attention backend "
                    f"(got attn_backend={base_attn.attn_backend!r}). The CP "
                    "kernels are flex-only: sdpa cannot express the "
                    "per-document mask, and there is no CP varlen flex "
                    "variant for this model."
                )
            # Traverse, as ContextParallelTransform does: the flavor registry
            # may already have expanded the layer list, in which case mutating
            # the `self.layer` TEMPLATE alone converts 1 of 25 configs and
            # leaves the real layers on the non-CP kernel. When the list is NOT
            # yet expanded, only the template exists here and _expand_layers()
            # copies the converted template into it -- correct either way.
            for _, attn_cfg, _, _ in self.traverse(BaseAttention.Config):
                attn_cfg.inner_attention = convert_config_type(
                    attn_cfg.inner_attention, KVAllGatherCPFlexInnerAttention
                )

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            max_context_length = config.training.max_context_length
            if max_context_length > self.rope.max_context_length:
                logger.warning(
                    f"Context length {max_context_length} exceeds original "
                    f"maximum {self.rope.max_context_length}."
                )

            # Sync rope length (both global and SWA-local caches). The caches also
            # cover the validation sequences: validator.fixed_seq_len may be longer than
            # the training context, and positions index the cache (K1).
            self.max_context_length = max_context_length
            rope_length = max(
                max_context_length,
                getattr(getattr(config, "validator", None), "fixed_seq_len", None) or 0,
            )
            self.rope = _dc.replace(self.rope, max_context_length=rope_length)
            if self.rope_of_swa is not None:
                self.rope_of_swa = _dc.replace(
                    self.rope_of_swa, max_context_length=rope_length
                )

            # Direct: TrainingConfig is a slots dataclass that always has the
            # field, so a rename must raise rather than silently disable the
            # SFT MoE token mask.
            self.enable_token_mask_for_moe = config.training.enable_token_mask_for_moe

            base_attn = self.layer.attention
            assert isinstance(base_attn, GatedNormSWAttention.Config)
            use_swa = _parse_layer_pattern(self.swa_pattern, self.n_layers, "S", "F")
            uses_swa = any(use_swa) or (
                self.swa_pattern is None and base_attn.sliding_window_size > 0
            )
            if uses_swa and base_attn.sliding_window_size <= 0:
                raise ValueError(
                    "SWA is enabled but layer.attention.sliding_window_size "
                    "is not set (must be > 0)."
                )
            if uses_swa and base_attn.attn_backend == "varlen":
                raise ValueError("SWA is not supported with varlen attention.")
            if (base_attn.conv_kernel_size > 0 or base_attn.attention_sinks) and (
                parallelism.tensor_parallel_degree > 1
                or parallelism.context_parallel_degree > 1
            ):
                raise NotImplementedError(
                    "the attention short conv and sinks support neither TP nor CP"
                )

            # Expand before delegating: Decoder.Config.update_from_config walks
            # self.layers for TP/EP validation and token-dispatcher setup, so the
            # per-layer list has to exist by then.
            self._expand_layers()

            Decoder.Config.update_from_config(self, config=config, **kwargs)

            from torchtitan.models.opt_moe.sharding import set_opt_moe_sharding_config

            # enable_sequence_parallel defaults to True regardless of TP, so
            # gate on the actual degree -- otherwise a plain FSDP run would look
            # like a TP run to the sharding plan.
            set_opt_moe_sharding_config(
                self,
                enable_sp=(
                    parallelism.enable_sequence_parallel
                    and parallelism.tensor_parallel_degree > 1
                ),
                enable_tp=parallelism.tensor_parallel_degree > 1,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # NormMoE is not upstream's MoE; without moe_types its routed
            # experts count at full weight and inflate active params.
            nparams, active_nparams = get_nparams_and_active_nparams(
                model, moe_types=(NormMoE,)
            )
            base_attn = self.layer.attention
            assert isinstance(base_attn, GatedNormSWAttention.Config)
            # Per layer, not n_layers * template: with `swa_pattern` the windowed
            # layers cost far less than full attention, and costing them all as
            # full inflates the attention term (measured 2.91x on a swa_512
            # flavor) and therefore every reported MFU/TFLOPs number. Matches
            # gpt_oss, the other SWA model. Note opt_moe encodes "no window" as
            # -1 while the helper expects None; passing -1 raw would silently
            # produce min(seq_len, -1).
            attention_op_flops = 0
            for layer in self.layers:
                layer_attn = layer.attention
                window = getattr(layer_attn, "sliding_window_size", -1)
                layer_head_dim = (
                    layer_attn.head_dim
                    if layer_attn.head_dim is not None
                    else self.dim // layer_attn.n_heads
                )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=layer_attn.n_heads,
                    qk_head_dim=layer_head_dim,
                    v_head_dim=layer_head_dim,
                    seq_len=seq_len,
                    sliding_window_size=window if window > 0 else None,
                )
            return nparams, 6 * active_nparams + attention_op_flops

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
        attention_masks: AttentionMasksType | None = None,
        loss_mask: torch.Tensor | None = None,
        doc_id: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
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
        h = self._trunk(tokens, positions, attention_masks, loss_mask, doc_id)
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h

    def _trunk(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        loss_mask: torch.Tensor | None = None,
        doc_id: torch.Tensor | None = None,
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
        for layer in self.layers.values():
            out = layer(h, attention_masks, positions, loss_mask, doc_id)
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

    @torch.no_grad()
    def get_attention_masks(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        # Added by upstream #4156, and the base's preprocess_inputs passes all
        # three positionally-by-keyword. Accepted and forwarded rather than
        # dropped, for the same reason as preprocess_inputs above.
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> "AttentionMasksType | None":
        """Return attention masks appropriate for the mix of layer backends.

        Returns:
            ``{"full": BlockMask}``                   -- flex layers, no SWA.
            ``{"full": BlockMask, "swa": BlockMask}`` -- mixed flex full + SWA.
            Delegates to ``super()`` for varlen.
        """
        has_swa = any(
            getattr(layer, "use_swa", False) for layer in self.layers.values()
        )
        has_varlen = any(
            getattr(layer, "attn_backend", None) == "varlen"
            for layer in self.layers.values()
        )

        if has_varlen and has_swa:
            raise ValueError("SWA is not supported with varlen attention.")
        if has_varlen:
            return super().get_attention_masks(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

        # Every layer is flex here: the block refuses any backend but flex,
        # flex_flash and varlen. Document boundaries come from `positions`
        # resetting to 0, so masks are always per document (block causal).
        attn_config = self.config.first_attention
        assert attn_config is not None
        base_attn = self.config.layer.attention
        assert isinstance(base_attn, GatedNormSWAttention.Config)
        # Upstream's helper is the reference for causal + packed-document
        # masking; use it rather than re-deriving the mask mods here.
        mask_mods = [
            get_causal_mask_mod(),
            get_efficient_causal_mask_mod_for_packed_document(positions),
        ]
        full_mask = self._create_flex_attention_mask_for_document(
            positions, attn_config
        )
        if not has_swa:
            return {"full": full_mask}

        # No upstream helper composes a sliding window with the document mask,
        # so the SWA variant is assembled from the same mods plus the window.
        swa_mask = self._create_flex_attention_mask(
            positions,
            attn_config,
            mask_mods + [get_sliding_window_mask_mod(base_attn.sliding_window_size)],
        )
        return {"full": full_mask, "swa": swa_mask}

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims,
        parallelism,
        # Added by upstream #4156 (cudagraph for varlen pretrain). Accepted and
        # forwarded rather than dropped: the base uses them to size the
        # document-aware packing path, and omitting them from this override made
        # the trainer raise `unexpected keyword argument 'max_num_documents'`.
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ):
        """Build masks/CP shards, then add the MoE token mask.

        The MoE loss mask is derived from the CP-sharded labels so its token
        layout matches the hidden states the router sees; building it before CP
        would leave it full-length and break indexing inside the router.
        """
        # Derive the document id from the GLOBAL positions, BEFORE
        # `super().preprocess_inputs` CP-shards them. `doc_id` is registered in
        # `decoder_input_sharding()`, so it is sharded with the same load
        # balancer as `positions` and returns inside `extra_kwargs`.
        #
        # Why: under the default "headtail" balancer a CP rank receives two
        # NON-adjacent global chunks concatenated, and the join carries no
        # `positions == 0`. Segmenting on positions alone therefore merges the
        # tail of one document with the middle of another whenever the two
        # happen to line up -- ~0.1% of rank/step pairs for ragged packing, but
        # DETERMINISTIC for equal-length documents. An id computed before
        # sharding cannot be fooled by the seam.
        positions = input_dict.get("positions", None)
        if positions is not None:
            flat = positions.reshape(-1)
            starts = torch.ones_like(flat, dtype=torch.bool)
            starts[1:] = flat[1:] != (flat[:-1] + 1)
            input_dict = dict(input_dict)
            # Cast AFTER the cumsum: `torch.cumsum` promotes every integer
            # dtype to int64, so casting the input was dead code and the
            # tensor went through `cp_shard` at 8 B/token instead of 4.
            input_dict["doc_id"] = (
                torch.cumsum(starts, dim=0).to(torch.int32).reshape(positions.shape)
            )

        inputs, labels, extra_kwargs = super().preprocess_inputs(
            input_dict,
            parallel_dims=parallel_dims,
            parallelism=parallelism,
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
        )
        if self.config.enable_token_mask_for_moe:
            from torchtitan.components.loss import IGNORE_INDEX

            extra_kwargs["loss_mask"] = labels != IGNORE_INDEX
        return inputs, labels, extra_kwargs
