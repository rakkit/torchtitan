# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.local_compile import LocalCompileConfig

from torchtitan.models.common.attention import AttentionMetadata, AttentionMetadataMap
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.moe import RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.param_layout import (
    by_rank,
    named,
    ParamLayout,
    stacked_units,
)
from .norm_moe import NormMoE
from .state_dict_adapter import OPTMoEStateDictAdapter


def layout_flags(parallelism: ParallelismConfig) -> dict[str, bool]:
    """What the model config's sharding depends on (data parallel does not)."""
    tp = parallelism.tensor_parallel_degree > 1
    return {
        "tp": tp,
        "cp": parallelism.context_parallel_degree > 1,
        "ep": parallelism.expert_parallel_degree > 1,
        "sp": parallelism.enable_sequence_parallel and tp,
    }


class OPTMoETransformerBlock(TransformerBlock):
    """OPT MoE TransformerBlock: pre-norm attention and a dense or MoE
    feed-forward, ``x = identity_scale * x + block_scale * block(x)`` for each.

    Token-flat throughout: ``x`` is ``[T, D]``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        feed_forward: DenseFeedForward.Config | None = None
        moe: NormMoE.Config | None = None
        """The layer's feed-forward: dense or MoE, exactly one of the two. The
        MoE is built as ``self.moe`` (FQNs ``moe.*``)."""
        block_scale: float = 1.0
        identity_scale: float = 1.0
        """The residual scales: ``x = identity_scale * x + block_scale * f(x)``
        for the attention and the feed-forward (1.0: the plain residual)."""

        def __post_init__(self) -> None:
            if (self.feed_forward is None) == (self.moe is None):
                raise ValueError("a block has exactly one of feed_forward / moe")

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            self.moe = config.moe.build()
        else:
            self.feed_forward = config.feed_forward.build()
        self.block_scale = config.block_scale
        self.identity_scale = config.identity_scale
        # A unit scale is not multiplied (x * 1.0 == x). Block side: two [T, D]
        # kernels less per block. Identity side also a workaround:
        # TODO: inductor still rejects `1.0 * x + b * f(x)` under activation
        # checkpointing + torch.compile around the block ("Node add_N was
        # invalid, but is output", torch 2.16.0.dev20261030; any b, only an
        # identity scale of exactly 1.0).
        self._skip_identity_mul = self.identity_scale == 1.0
        self._skip_block_mul = self.block_scale == 1.0

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: AttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: ``[T, D]``; ``loss_mask``: the tokens the MoE counts in its
        balance statistics (``OPTMoEModel.preprocess_inputs``); ``padding_mask``:
        upstream's, accepted and ignored (``loss_mask`` is the stricter mask)."""
        _attn_out = self.attention(
            self.attention_norm(x), attention_metadata, positions
        )
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(_attn_out)
        lhs = x if self._skip_identity_mul else self.identity_scale * x
        rhs = _attn_out if self._skip_block_mul else self.block_scale * _attn_out
        h = lhs + rhs
        h_normed = self.ffn_norm(h)

        if self.moe_enabled:
            mlp_output = self.moe(
                h_normed, loss_mask, aux_loss_denominator=aux_loss_denominator
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


class OPTMoEModel(Decoder):
    """OPT MoE Transformer model with attention and feed-forward layers."""

    state_dict_adapter_cls = OPTMoEStateDictAdapter

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.optimizers.moe_balance import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    def param_layouts(self) -> dict:
        """What every parameter is, keyed by FQN (see
        `models/common/param_layout.py`): modules declare what only they know
        (expert batches, fused projections, the router gate), the model the two
        vocabulary matrices, and the rest is stated by rank."""
        layouts = {}
        for module_name, module in self.named_modules():
            # Upstream's modules, declared here.
            if isinstance(module, DenseFeedForward):
                w13 = module.w13.weight  # [2, F, D]: w1 and w3 stacked
                units = stacked_units(("w1", "w3"), w13.shape[-2], w13.shape[-1])
                layouts[f"{module_name}.w13.weight"] = ParamLayout(
                    "matrix", units=units
                )
                continue
            if isinstance(module, TokenChoiceTopKRouter):  # the gate's rows: experts
                layouts[f"{module_name}.gate.weight"] = ParamLayout("router")
                continue
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
        got = layout_flags(parallelism)
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
        # name / flavor / build_args: how the config was built, in the dumped
        # job_config.json, so evaluation and HF conversion rebuild the run's
        # exact model (build_model_config(flavor, **build_args)).
        name: str = "opt_moe"
        flavor: str = ""
        build_args: dict[str, Any] = field(default_factory=dict)
        """The flavor's arguments merged with the run's overrides (plain values)."""
        enable_token_mask_for_moe: bool = False
        """Keep tokens whose label is ignored out of the MoE balance statistics
        (``preprocess_inputs`` builds the mask)."""
        built_for: dict[str, bool] | None = None
        """The parallelism (tp / cp / ep / sp on or off) the per-layer configs
        were built for; ``parallelize`` refuses another one (a command-line
        override of these after the build would otherwise be ignored)."""
        # Upstream's regions the model passes through plus opt_moe's own
        # elementwise code (fused attention-prep / residual-norm regions: only
        # once measured on the production dims).
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

    def __init__(self, config: Config):
        super().__init__(config)
        # Set to [] by the optimizer container on logging steps (log_residual_stats);
        # forward then appends its residual statistics. Not a buffer.
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
            attention_metadata: The inner attentions' metadata, by type.
            loss_mask: Tokens counted in the MoE balance statistics (True:
                counted); see ``preprocess_inputs``.
            padding_mask: Accepted and ignored: ``loss_mask`` is the stricter
                mask (labels ignored, prompt tokens of SFT included).
        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        # Upstream's aux-loss normaliser: the step-global routed-token count (one
        # per block; every block routes the same tokens), as Decoder.forward.
        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        # Residual statistics (log_residual_stats): taken in this eager loop, so
        # the blocks and their compiled graphs are the same either way.
        rows = [] if self.residual_capture is not None else None
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
        h = self.norm(h) if self.norm is not None else h
        # _skip_lm_head: the chunked loss applies lm_head itself (upstream's).
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h

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
