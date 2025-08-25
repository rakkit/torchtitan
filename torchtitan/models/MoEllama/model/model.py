# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed.tensor
from torch import nn

from torchtitan.models.inits import build_init_fn
from torchtitan.models.inputs import MoEInputs, MoEInputsDict
from torchtitan.models.llama3.model.model import (
    # apply_rotary_emb,
    Attention,
    precompute_freqs_cis,
    # repeat_kv,
)
from torchtitan.models.norms import build_norm
from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.tools.logging import logger

from .args import MoEModelArgs
from .moe import FeedForward, MoE
from .moe_utils import calc_gate_scaling_factor


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    attention_cls = Attention

    def __init__(self, layer_id: int, model_args: MoEModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = self.attention_cls(model_args)
        # if self.attention_cls is Attention:
        #     self.attention.forward = torch.compile(
        #         self.attention.forward, fullgraph=True
        #     )
        # TODO(JSC): Need ablation, feels like this does not really matter
        if model_args.moe_router_scaling_factor is None:
            router_scaling_factor = calc_gate_scaling_factor(
                model_args.n_routed_experts,
                model_args.activate_experts,
                iter_times=10_000,
            )
            if layer_id == 0:
                logger.info(
                    f"Auto-computed router_scaling_factor: {router_scaling_factor}"
                )
        else:
            router_scaling_factor = model_args.moe_router_scaling_factor
            if layer_id == 0:
                logger.info(
                    f"Using manually set router_scaling_factor: {router_scaling_factor}"
                )

        self.moe_enabled = layer_id >= model_args.n_dense_layers

        if self.moe_enabled:
            self.feed_forward = MoE(
                layer_id,
                dim=model_args.dim,
                multiple_of=model_args.multiple_of,
                ffn_dim_multiplier=model_args.ffn_dim_multiplier,
                activation_type=model_args.activation_type,
                n_shared_experts=model_args.n_shared_experts,
                n_routed_experts=model_args.n_routed_experts,
                activate_experts=model_args.activate_experts,
                use_bias_for_routing=model_args.moe_router_use_bias_for_routing,
                bias_update_speed=model_args.moe_router_bias_update_speed,
                aux_loss_alpha=model_args.moe_aux_loss_alpha,
                bias_update_norm_factor=model_args.moe_router_bias_update_norm_factor,
                match_dim_with_dense=True,
                router_scaling_factor=router_scaling_factor,
                norm_everywhere=model_args.norm_everywhere,
                norm_type=model_args.norm_type,
                norm_eps=model_args.norm_eps,
            )
        else:
            hidden_dim = 2 * 4 * model_args.dim / 3
            if model_args.ffn_dim_multiplier is not None:
                hidden_dim = model_args.ffn_dim_multiplier * hidden_dim
            hidden_dim = int(hidden_dim - hidden_dim % model_args.multiple_of)

            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=hidden_dim,
                activation_type=model_args.activation_type,
            )

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.weight_init_fn_type = model_args.intermediate_init_fn_type
        self.weight_init_std = (
            model_args.intermediate_init_std
            * model_args.dim**model_args.intermediate_exp
        )
        self.router_init_fn_type = model_args.router_init_fn_type
        self.init_gate_as_residual = model_args.init_gate_as_residual

        # x  = identity_scale * x + block_scale * block(x)
        if model_args.depth_init == "identity":
            self.residual_div = 1.0
        elif model_args.depth_init == "depth":
            self.residual_div = (2 * (layer_id + 1)) ** 0.5
        elif model_args.depth_init == "total_depth":
            self.residual_div = (2 * model_args.n_layers) ** 0.5
        else:
            raise ValueError(f"Invalid depth_init: {model_args.depth_init}")

        if model_args.residual_scale == "depth_scale":
            total_depth = 2 * model_args.n_layers
            self.block_scale = 1 / total_depth
            self.identity_scale = (total_depth - 1) / total_depth
        elif model_args.residual_scale == "complete_p":
            total_depth = 2 * model_args.n_layers
            self.block_scale = 1 / total_depth
            self.identity_scale = 1.0
        elif model_args.residual_scale == "identity":
            self.block_scale = 1.0
            self.identity_scale = 1.0
        else:
            raise ValueError(f"Invalid residual_scale: {model_args.residual_scale}")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """

        h = self.identity_scale * x + self.block_scale * self.attention(
            self.attention_norm(x), freqs_cis
        )

        if self.moe_enabled:
            mlp_output, moe_aux_loss = self.feed_forward(self.ffn_norm(h))
        else:
            mlp_output = self.feed_forward(self.ffn_norm(h))
            moe_aux_loss = torch.tensor(0.0, device=x.device, pin_memory=True)

        out = self.identity_scale * h + self.block_scale * mlp_output

        return out, moe_aux_loss

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(
            self.weight_init_std,
            residual_div=self.residual_div,
            init_fn_type=self.weight_init_fn_type,
        )
        self.feed_forward.init_weights(
            self.weight_init_std,
            residual_div=self.residual_div,
            init_gate_as_residual=self.init_gate_as_residual,
            init_fn_type=self.weight_init_fn_type,
            router_init_fn_type=self.router_init_fn_type,
        )

    def init_kv_cache(self, max_batch_size: int, max_seq_length: int):
        self.attention.init_kv_cache(max_batch_size, max_seq_length)


class Transformer(nn.Module, ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (Linear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    transformer_block_cls = TransformerBlock

    def __init__(self, model_args: MoEModelArgs):
        if model_args.num_mtp_modules > 0:
            raise ValueError("currently, MTP is not supported with MoE")
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.pad_id = model_args.pad_id

        logger.info(
            f"model_args.dim = {model_args.dim} | model_args.vocab_size = {model_args.vocab_size}"
        )

        self.tok_embeddings = nn.Embedding(
            model_args.vocab_size,
            model_args.dim,
            padding_idx=self.pad_id if self.pad_id >= 0 else None,
        )

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = self.transformer_block_cls(
                layer_id, model_args
            )
        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        first_in_init_fn = build_init_fn(self.model_args.first_in_init_fn_type)
        first_in_std = (
            self.model_args.first_in_init_std
            * self.model_args.vocab_size**self.model_args.first_in_exp
        )
        if self.tok_embeddings is not None:
            if self.model_args.first_in_init_fn_type == "scion_normal":
                # catch cases when axis=1 is sharded
                assert self.tok_embeddings.weight.size(1) == self.model_args.dim, (
                    f"Input embedding last dim does not match model dim. "
                    f"Got shape: {self.tok_embeddings.weight.shape}"
                )
            first_in_init_fn(
                self.tok_embeddings.weight,
                mean=0.0,
                std=first_in_std,
            )
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_init_fn = build_init_fn(self.model_args.final_out_init_fn_type)
        final_out_std = (
            self.model_args.final_out_init_std
            * self.model_args.dim**self.model_args.final_out_exp
        )
        cutoff_factor = 3
        if self.output is not None:
            extra_kwargs = {}
            if self.model_args.final_out_init_fn_type == "trunc_normal":
                extra_kwargs["a"] = -cutoff_factor * final_out_std
                extra_kwargs["b"] = cutoff_factor * final_out_std
            if self.model_args.final_out_init_fn_type == "scion_normal":
                # catch cases when axis=1 is sharded
                assert self.output.weight.size(1) == self.model_args.dim, (
                    f"Output last dim does not match model dim. "
                    f"Got shape: {self.output.weight.shape}"
                )
            final_out_init_fn(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                **extra_kwargs,
            )
        if self.model_args.num_mtp_modules > 0:
            for layer in self.mtp_layers.values():
                if layer is not None:
                    layer.init_weights()

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_kv_cache(self, max_batch_size: int, max_seq_length: int):
        for layer in self.layers.values():
            if layer is not None:
                layer.init_kv_cache(max_batch_size, max_seq_length)

    def forward(
        self,
        inputs: MoEInputs,
        input_batch: torch.Tensor | None = None,
    ) -> MoEInputsDict:
        """
        Perform a forward pass through the Transformer model.

        Args:
            inputs (MoEInputs): Single tensor or dictionary containing the
                following keys and values:
                - tokens_list (Union[list[torch.Tensor | None],
                  torch.Tensor]): Input token indices if pipeline parallelism is not enabled.
                  If pipeline parallelism is enabled, this will be the input token indices
                  for the ranks on the first pipeline stage. This will be the activation of the
                  previous pipeline stage if the current rank is not on the first stage.
                - aux_loss (torch.Tensor): Sequence-wise auxiliary balance loss.
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            MoEInputsDict: Dictionary containing the following keys and values:
                - tokens_list (list[torch.Tensor]): Output logits after applying
                  the Transformer model.
                - aux_loss (torch.Tensor): Sequence-wise auxiliary balance loss.


        """
        if not isinstance(inputs, dict):
            inputs = {"tokens_list": inputs}
        tokens = inputs["tokens_list"]
        total_moe_aux_loss = inputs.get("aux_loss", 0.0)
        if isinstance(tokens, list):
            tokens = tokens[0]

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h, moe_aux_loss = layer(h, self.freqs_cis)
            total_moe_aux_loss += moe_aux_loss

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return {
            "tokens_list": [output],
            "aux_loss": total_moe_aux_loss,
        }
