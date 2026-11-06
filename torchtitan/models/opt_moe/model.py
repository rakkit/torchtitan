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

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.base.model import FlopsEstimator
from torchtitan.models.base.module import Module, ModuleList

from torchtitan.models.common.attention import (
    AttentionMetadata,
    AttentionMetadataMap,
    create_varlen_metadata_for_document,
)
from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_layout import (
    by_rank,
    named,
    ParamLayout,
    stacked_units,
)
from .gated_swa_attention import ShortConvDocuments
from .mtp import roll_mtp_sequence
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
    feed-forward (``feed_forward``, by its config type, as upstream's blocks),
    ``x = identity_scale * x + block_scale * block(x)`` for each.

    Token-flat throughout: ``x`` is ``[T, D]``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        block_scale: float = 1.0
        identity_scale: float = 1.0
        """The residual scales: ``x = identity_scale * x + block_scale * f(x)``
        for the attention and the feed-forward (1.0: the plain residual)."""

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
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
        *,
        conv_metadata: LinearAttentionMetadata | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: ``[T, D]``; ``conv_metadata``: the attention short conv's
        document offsets; ``padding_mask`` (True: not counted): upstream's
        padding mask, the tokens the MoE leaves out of its balance statistics."""
        _attn_out = self.attention(
            self.attention_norm(x),
            attention_metadata,
            positions,
            conv_metadata=conv_metadata,
        )
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(_attn_out)
        lhs = x if self._skip_identity_mul else self.identity_scale * x
        rhs = _attn_out if self._skip_block_mul else self.block_scale * _attn_out
        h = lhs + rhs
        h_normed = self.ffn_norm(h)

        if isinstance(self.feed_forward, MoE):
            mlp_output = self.feed_forward(
                h_normed,
                padding_mask_T=padding_mask,
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


class OPTMoEMTPBlock(OPTMoETransformerBlock):
    """One multi-token-prediction depth (DeepSeek-V3's MTP module): the fused
    input ``eh_proj(cat(enorm(e), hnorm(h)))`` of the next token's embedding
    ``e`` and the previous depth's hidden state ``h``, then one OPT MoE block,
    then ``mtp_norm`` (the shared lm_head follows). ``h`` is zeroed where the
    rolled token crosses a document or padding, and those tokens leave the MoE
    statistics (``padding_mask``)."""

    @dataclass(kw_only=True, slots=True)
    class Config(OPTMoETransformerBlock.Config):
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        eh_proj: Linear.Config
        mtp_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()
        self.mtp_norm = config.mtp_norm.build()

    def forward(  # pyrefly: ignore [bad-override]
        self,
        mtp_input_embed_TD: torch.Tensor,
        prev_hidden_TD: torch.Tensor,
        mtp_input_valid_mask_T: torch.Tensor,
        attention_metadata: AttentionMetadata | None,
        positions: torch.Tensor | None,
        *,
        conv_metadata: LinearAttentionMetadata | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mtp_padding_mask_T = ~mtp_input_valid_mask_T
        if padding_mask is not None:
            mtp_padding_mask_T = mtp_padding_mask_T | padding_mask
        prev_hidden_TD = prev_hidden_TD * mtp_input_valid_mask_T.unsqueeze(-1).to(
            prev_hidden_TD.dtype
        )
        h_TD = self.eh_proj(
            torch.cat(
                [self.enorm(mtp_input_embed_TD), self.hnorm(prev_hidden_TD)], dim=-1
            )
        )
        h_TD = super().forward(
            h_TD,
            attention_metadata,
            positions,
            conv_metadata=conv_metadata,
            padding_mask=mtp_padding_mask_T,
            aux_loss_denominator=aux_loss_denominator,
        )
        return self.mtp_norm(h_TD)


class OPTMoEModel(Decoder):
    """OPT MoE Transformer model with attention and feed-forward layers."""

    state_dict_adapter_cls = OPTMoEStateDictAdapter

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.optimizers.moe_balance import register_moe_load_balancing_hook

        # The balance hook reads each aux loss's per-step value before upstream's
        # zero hook rolls it into the aux-loss metrics and clears it.
        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

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
        try:
            return super().parallelize(parallelism=parallelism, **kwargs)
        finally:
            for key in self._listed_mtp_keys:
                del self.layers[key]
            self._listed_mtp_keys = []

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        """Upstream's sharding of every module; then, for the steps of
        ``parallelize`` that walk ``layers`` (activation remat, FSDP wrapping
        and prefetch), the MTP blocks are listed there after the decoder
        blocks, as decoder blocks, until ``parallelize`` returns."""
        super()._parallelize(parallelism_context)
        for i, block in enumerate(self.mtp_layers or ()):
            key = str(len(self.config.layers) + i)
            self.layers[key] = block
            self._listed_mtp_keys.append(key)

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
        """The recipe's overrides of the flavor's settings (plain values)."""
        built_for: dict[str, bool] | None = None
        """The parallelism (tp / cp / ep / sp on or off) the per-layer configs
        were built for; ``parallelize`` refuses another one (a command-line
        override of these after the build would otherwise be ignored)."""
        mtp_layers: list[OPTMoEMTPBlock.Config] = field(default_factory=list)
        """The multi-token-prediction depths (``OPTMoEMTPBlock``; empty: no
        MTP). With MTP the model takes and returns one tensor per objective
        (the main prediction, then one per depth)."""
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
                    "opt_moe_sink_rescale",
                    "opt_moe_attention_output",
                ]
            )
        )

        def build_flops_estimator(
            self,
            model: Module,
            *,
            seq_len: int,
            excluded_submodules: tuple[Module | None, ...] = (),
        ) -> FlopsEstimator:
            """Upstream's decoder estimate (it counts the MTP blocks' parameters
            and attention) plus the lm_head pass of every MTP depth."""
            estimate = Decoder.Config.build_flops_estimator(
                self, model, seq_len=seq_len, excluded_submodules=excluded_submodules
            )
            if not self.mtp_layers:
                return estimate
            head_params = sum(p.numel() for p in model.lm_head.parameters())
            head_flops_per_token = 6 * len(self.mtp_layers) * head_params
            return lambda input_dict: (
                estimate(input_dict)
                + head_flops_per_token * input_dict["input"].numel()
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.mtp_layers: ModuleList | None = None
        self._listed_mtp_keys: list[str] = []
        if config.mtp_layers:
            self.mtp_layers = ModuleList()
            for layer_config in config.mtp_layers:
                self.mtp_layers.append(layer_config.build())
        # Set to [] by the optimizer container on logging steps (log_residual_stats);
        # forward then appends its residual statistics. Not a buffer.
        self.residual_capture: list[torch.Tensor] | None = None
        # The attention short conv's width (0: no conv; every layer has the same).
        self.conv_kernel_size = next(
            (
                layer.attention.k_conv.kernel_size
                for layer in config.layers
                if layer.attention.k_conv is not None
            ),
            0,
        )

    def _get_attention_metadata(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
    ) -> AttentionMetadataMap:
        """Upstream's inner-attention metadata, plus the short conv's document
        offsets (the ``ShortConvDocuments`` entry) when the attention has a conv."""
        attention_metadata = super()._get_attention_metadata(
            positions,
            padding_mask=padding_mask,
            max_num_documents=max_num_documents,
        )
        if self.conv_kernel_size > 0:
            attention_metadata[ShortConvDocuments] = LinearAttentionMetadata(
                varlen=create_varlen_metadata_for_document(
                    positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=self.config.max_context_length,
                ),
                num_conv_history_tokens=self.conv_kernel_size - 1,
            )
        return attention_metadata

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
        """Upstream's ``Decoder.preprocess_inputs``; with MTP, first each depth's
        inputs and labels: the tokens and labels rolled left by the depth
        (``roll_mtp_sequence``: never across a document or padding), CP-sharded
        with the batch, then returned as one tuple per objective."""
        if self.mtp_layers is None:
            return super().preprocess_inputs(
                input_dict,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
                max_num_documents=max_num_documents,
                global_routing_token_counts=global_routing_token_counts,
                **kwargs,
            )
        del kwargs
        loss_kwargs = input_dict.pop("loss_kwargs", {})
        positions = input_dict.get("positions")
        padding_mask = input_dict.get("padding_mask")
        if positions is None:
            raise ValueError("MTP needs the positions (document boundaries)")
        depths = range(1, len(self.mtp_layers) + 1)
        input_shardings = decoder_input_sharding()
        for depth in depths:
            tokens, valid = roll_mtp_sequence(
                input_dict["input"],
                shift=depth,
                fill_value=0,
                positions_T=positions,
                padding_mask_T=padding_mask,
            )
            labels, _ = roll_mtp_sequence(
                input_dict["labels"],
                shift=depth,
                fill_value=IGNORE_INDEX,
                positions_T=positions,
                padding_mask_T=padding_mask,
            )
            for name, value, like in (
                (f"mtp_input_{depth}", tokens, "input"),
                (f"mtp_labels_{depth}", labels, "labels"),
                (f"mtp_valid_{depth}", valid, "input"),
            ):
                input_dict[name] = value
                input_shardings[name] = input_shardings[like]
        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)
        input_dict["attention_metadata"] = self._get_attention_metadata(
            positions, padding_mask=padding_mask, max_num_documents=max_num_documents
        )
        if parallelism_context.cp_enabled:
            input_dict = self._cp_shard(
                input_dict,
                input_shardings=input_shardings,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
        input_dict = annotate_input_spmd_types(
            parallelism_context, input_dict, input_shardings
        )
        loss_kwargs = {name: input_dict.pop(name) for name in loss_kwargs}
        inputs = (
            input_dict.pop("input"),
            *(input_dict.pop(f"mtp_input_{d}") for d in depths),
        )
        labels = (
            input_dict.pop("labels"),
            *(input_dict.pop(f"mtp_labels_{d}") for d in depths),
        )
        input_dict["mtp_input_valid_masks"] = tuple(
            input_dict.pop(f"mtp_valid_{d}") for d in depths
        )
        # one routed-token count per objective (the collator's, MTP depths too)
        input_dict["aux_loss_denominators"] = global_routing_token_counts
        return inputs, labels, input_dict, loss_kwargs

    def forward(
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        """Upstream's ``Decoder.forward`` plus the residual statistics and,
        with MTP, the MTP depths after the main stack.

        Args:
            tokens: Input token indices ``[T]``, or hidden states when this is
                not the first pipeline stage; with MTP one tensor per objective
                (the main tokens, then each depth's rolled tokens).
            positions: Position indices ``[T]``.
            attention_metadata: The inner attentions' metadata, by type.
            mtp_input_valid_masks: Per MTP depth, where its rolled token is real.
            padding_mask: Upstream's padding mask (True: padding), the tokens
                the MoE leaves out of its balance statistics.

        Returns:
            The logits (hidden states with ``_skip_lm_head``); with MTP a tuple,
            the main prediction then one per depth (in eval mode the main
            prediction only).
        """
        if self.mtp_layers is None:
            mtp_tokens: list[torch.Tensor] = []
        else:
            if not isinstance(tokens, tuple) or len(tokens) != len(self.mtp_layers) + 1:
                raise ValueError("MTP: one token tensor per objective")
            if mtp_input_valid_masks is None or len(mtp_input_valid_masks) != len(
                self.mtp_layers
            ):
                raise ValueError("MTP: one validity mask per MTP depth")
            tokens, *mtp_tokens = tokens
        # In eval mode (validation, offline evaluation: they score the main
        # prediction) the MTP depths do not run.
        run_mtp = self.mtp_layers is not None and self.training
        if run_mtp:
            # One embedding call for the main and the MTP tokens: FSDP gathers
            # the embedding once.
            embedded = self.tok_embeddings(torch.cat([tokens, *mtp_tokens]))
            h, *mtp_embeds = embedded.split(tokens.shape[0])
        else:
            mtp_embeds = []
            # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
            h = (
                self.tok_embeddings(tokens)
                if self.tok_embeddings is not None
                else tokens
            )
        # Upstream's aux-loss normaliser: the step-global routed-token count (one
        # per block; every block routes the same tokens), as Decoder.forward.
        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        # Residual statistics (log_residual_stats): taken in this eager loop, so
        # the blocks and their compiled graphs are the same either way.
        rows = [] if self.residual_capture is not None else None
        conv_metadata = (
            None
            if attention_metadata is None
            else attention_metadata.get(ShortConvDocuments)
        )
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
                conv_metadata=conv_metadata,
                padding_mask=padding_mask,
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
        prev_hidden = h  # the MTP depths read the pre-norm hidden state
        h = self.norm(h) if self.norm is not None else h
        outputs = [h]
        for depth, (layer, depth_embed, valid) in enumerate(
            zip(
                self.mtp_layers if run_mtp else (),
                mtp_embeds,
                mtp_input_valid_masks if run_mtp else (),
                strict=True,
            ),
            1,
        ):
            with spmd.no_typecheck():
                depth_denominator = (
                    None
                    if aux_loss_denominators is None
                    else aux_loss_denominators[depth]
                )
            prev_hidden = layer(
                depth_embed,
                prev_hidden,
                valid,
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention)),
                positions,
                conv_metadata=conv_metadata,
                padding_mask=padding_mask,
                aux_loss_denominator=depth_denominator,
            )
            outputs.append(prev_hidden)
        # _skip_lm_head: the chunked loss applies lm_head itself (upstream's).
        if not self._skip_lm_head and self.lm_head is not None:
            outputs = [self.lm_head(o) for o in outputs]
        return outputs[0] if self.mtp_layers is None else tuple(outputs)
