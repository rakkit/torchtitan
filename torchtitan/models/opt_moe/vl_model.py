# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    spmd_local_context,
)
from torchtitan.models.base.model import FlopsEstimator
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import AttentionMetadataMap
from torchtitan.models.common.decoder_sharding import (
    decoder_input_sharding,
    token_id_placement,
)
from torchtitan.models.common.multimodal import (
    add_zero_vision_dependency,
    build_dummy_vision_inputs,
    build_vision_bank_indices,
    gather_vision_embeds,
)
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.qwen3_5.vision_encoder import Qwen35VisionEncoder
from .model import _annotate_conv_metadata, _BLOCK_RECOMPILE_LIMIT, OPTMoEModel


class OPTMoEVLModel(OPTMoEModel):
    """``OPTMoEModel`` with a vision encoder (images only; FSDP only).

    The image features are gathered into the image placeholder positions of the
    token embeddings (no host sync); the decoder then runs unchanged. A text-only
    microbatch runs the encoder on one dummy patch grid so every rank takes part
    in the encoder's FSDP collectives. No CP, PP or MTP.

    Shape suffixes: T = packed tokens, D = model dimension, P = packed patches,
    F = patch features, V = vision tokens.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OPTMoEModel.Config):
        vision_encoder: Qwen35VisionEncoder.Config

        def build_flops_estimator(
            self,
            model: Module,
            *,
            seq_len: int,
            excluded_submodules: tuple[Module | None, ...] = (),
        ) -> FlopsEstimator:
            decoder_estimate = OPTMoEModel.Config.build_flops_estimator(
                self,
                model,
                seq_len=seq_len,
                excluded_submodules=(model.vision_encoder, *excluded_submodules),
            )
            encoder_estimate = self.vision_encoder.build_flops_estimator(
                model.vision_encoder
            )
            return lambda input_dict: decoder_estimate(input_dict) + encoder_estimate(
                input_dict
            )

        def __post_init__(self) -> None:
            OPTMoEModel.Config.__post_init__(self)
            if self.mtp_layers:
                raise ValueError("OPTMoEVLModel does not support MTP")

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision_encoder = config.vision_encoder.build()

    def parallelize(self, *, activation_remat: ActivationRematConfig | None, **kwargs):
        """The text model's lifecycle, then the encoder's remat and compile. The
        run's ``activation_remat`` patterns name the encoder's regions too
        ("attn.*" / "mlp.*" are relative to a block); ``save_all_layers`` names
        decoder blocks, so it is not applied to the encoder."""
        model = super().parallelize(activation_remat=activation_remat, **kwargs)
        if activation_remat is not None:
            dataclasses.replace(activation_remat, save_all_layers=[]).apply(
                self.vision_encoder
            )
        if self.config.compile_blocks:
            # As the decoder blocks (`_apply_fsdp`). The patch count differs every
            # step: the second shape recompiles with a dynamic token dimension.
            for block in self.vision_encoder.layers.values():
                for module in block.modules():
                    if hasattr(module, "inplace_wgrad_accum"):
                        module.inplace_wgrad_accum = False
                block.compile(
                    fullgraph=True,
                    recompile_limit=_BLOCK_RECOMPILE_LIMIT,
                    # Inductor times the fused kernels' block sizes: its heuristics
                    # make LayerNorm at width 1152 slower than eager.
                    options={"coordinate_descent_tuning": True},
                )
        return model

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        precision: PrecisionConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import (
            apply_fsdp_to_multimodal_encoder,
            resolve_fsdp_mesh,
        )

        # The encoder is CP-invariant: FSDP over dp_shard only, one unit.
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(
            parallelism_context, shard_on_cp_axis=False
        )
        apply_fsdp_to_multimodal_encoder(
            self.vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
            reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallelism_context.pp_enabled,
            cpu_offload=parallelism.fsdp_enable_cpu_offloading,
            dp_mesh_dims=dp_mesh_dims,
        )
        OPTMoEModel._apply_fsdp(
            self,
            parallelism_context=parallelism_context,
            precision=precision,
            parallelism=parallelism,
        )

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
        """``Decoder.preprocess_inputs`` with the vision inputs' layouts and, for
        an image microbatch, each token's row in the packed image features."""
        del kwargs
        if parallelism_context.cp_enabled or parallelism_context.pp_enabled:
            raise ValueError("OPTMoEVLModel supports neither CP nor PP")
        loss_kwargs = input_dict.pop("loss_kwargs", {})
        positions = input_dict.get("positions")
        padding_mask = input_dict.get("padding_mask")
        if positions is not None:
            input_dict["attention_metadata"] = self._get_attention_metadata(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
            )
        if input_dict.get("pixel_values_videos") is not None:
            raise ValueError("OPTMoEVLModel takes images only")
        input_dict.pop("pixel_values_videos", None)
        input_dict.pop("grid_thw_videos", None)
        input_shardings = {**decoder_input_sharding(), **multimodal_input_sharding()}
        if input_dict.get("pixel_values") is not None:
            input_dict["image_vision_bank_indices_T"] = build_vision_bank_indices(
                input_dict["input"],
                placeholder_id=input_dict["special_tokens"]["image_id"],
            )
            input_shardings["image_vision_bank_indices_T"] = token_id_placement()
        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)
        input_dict = annotate_input_spmd_types(
            parallelism_context, input_dict, input_shardings
        )
        _annotate_conv_metadata(input_dict.get("attention_metadata"))
        loss_kwargs = {name: input_dict.pop(name) for name in loss_kwargs}
        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        input_dict["aux_loss_denominators"] = global_routing_token_counts
        return inputs, labels, input_dict, loss_kwargs

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        image_vision_bank_indices_T: torch.Tensor | None = None,
    ):
        del special_tokens  # the placeholder rows are in image_vision_bank_indices_T
        encoder = self.vision_encoder
        with spmd_local_context("dp"):
            h_TD = self.tok_embeddings(tokens)
            if pixel_values is None:
                pixel_values_PF, grid_thw = build_dummy_vision_inputs(
                    patch_dim=encoder.patch_embed.in_features,
                    grid_thw=(
                        1,
                        encoder.spatial_merge_size,
                        encoder.spatial_merge_size,
                    ),
                    device=h_TD.device,
                )
                vision_VD = encoder(
                    pixel_values_PF.to(encoder.patch_embed.weight.dtype),
                    grid_thw=grid_thw,
                )
                h_TD = add_zero_vision_dependency(h_TD, vision_VD)
            else:
                vision_VD = encoder(
                    pixel_values.to(encoder.patch_embed.weight.dtype),
                    grid_thw=grid_thw,
                )
                h_TD = gather_vision_embeds(
                    h_TD,
                    vision_bank_VD=vision_VD,
                    vision_bank_indices_T=image_vision_bank_indices_T,
                )
        return super().forward(
            tokens,
            positions,
            attention_metadata,
            padding_mask=padding_mask,
            aux_loss_denominators=aux_loss_denominators,
            routed_experts=routed_experts,
            input_embeds=h_TD,
        )
