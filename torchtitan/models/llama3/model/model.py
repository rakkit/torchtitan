# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.attention import build_attention
from torchtitan.models.inputs import MTPInputs, MTPInputsDict
from torchtitan.models.norms import build_norm
from torchtitan.protocols.train_spec import ModelProtocol

from .args import TransformerModelArgs


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

        self.qk_norm = model_args.qk_norm or model_args.norm_everywhere
        self.norm_everywhere = model_args.norm_everywhere
        if self.qk_norm:
            self.q_norm = build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
            )
            self.k_norm = build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
            )
        if self.norm_everywhere:
            self.v_norm = build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
            )
            self.o_norm = build_norm(
                model_args.norm_type,
                dim=model_args.dim,
                eps=model_args.norm_eps,
            )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        if self.qk_norm:
            for norm in (self.q_norm, self.k_norm):
                norm.reset_parameters()
        if self.norm_everywhere:
            for norm in (self.v_norm, self.o_norm):
                norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply optional QK normalization
        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        if self.norm_everywhere:
            xv = self.v_norm(xv)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        if self.norm_everywhere:
            output = self.o_norm(output)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.
        norm_everywhere (bool): Whether to normalize the gating output.
        norm_type (Optional[str]): Normalization function to use. Only
            relevant and required, if `norm_everywhere=True`.
        norm_eps (str): Numerical stability epsilon for normalization
            layers. Only relevant and required, if
            `norm_everywhere=True`.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_everywhere: bool = False,
        norm_type: str | None = None,
        norm_eps: float | None = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        if norm_everywhere:
            assert (
                norm_type is not None
            ), "`norm_type` needs to be passed when `norm_everywhere=True`"
            assert (
                norm_eps is not None
            ), "`norm_eps` needs to be passed when `norm_everywhere=True`"
            self.out_norm = build_norm(
                norm_type,
                dim=hidden_dim,
                eps=norm_eps,
            )
        else:
            self.out_norm = nn.Identity()

    def forward(self, x):
        return self.w2(self.out_norm(F.silu(self.w1(x)) * self.w3(x)))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        if not isinstance(self.out_norm, nn.Identity):
            self.out_norm.reset_parameters()


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
    feed_forward_cls = FeedForward

    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = self.attention_cls(model_args)
        self.feed_forward = self.feed_forward_cls(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            norm_type=model_args.norm_type,
            norm_everywhere=model_args.norm_everywhere,
            norm_eps=model_args.norm_eps,
        )
        self.attention_norm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
        )
        self.ffn_norm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

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
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class MTPModule(nn.Module):
    def __init__(
        self,
        layer_id: int,
        model_args: TransformerModelArgs,
        parent_transformer: nn.Module,
    ):
        super().__init__()
        self.model_args = model_args

        # TODO handle these for pipelining
        self.tok_embeddings = parent_transformer.tok_embeddings
        self.norm = parent_transformer.norm
        self.output = parent_transformer.output

        self.in_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.mtp_proj = nn.Linear(2 * model_args.dim, model_args.dim)
        self.block = parent_transformer.transformer_block_cls(layer_id, model_args)

    def init_weights(self):
        self.in_norm.reset_parameters()
        # Re-use block's init std.
        self.mtp_proj.init_weights(self.block.weight_init_std)
        self.block.init_weights()

    def forward(
        self,
        tokens: torch.Tensor,
        prev_embed: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = -1,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            prev_embed (torch.Tensor): Output token embeddings of previous
                Transformer layer (after output norm, before unembedding).
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.
            torch.Tensor: Output token embeddings after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens)

        h = self.in_norm(h)

        h = torch.cat([h, prev_embed], dim=-1)
        h = self.mtp_proj(h)
        h = self.block(h, freqs_cis, start_pos=start_pos)

        h = self.norm(h)
        output = self.output(h)
        prev_embed = h
        return output, prev_embed


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

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = self.transformer_block_cls(
                layer_id, model_args
            )
        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # Optionally add MTP modules.
        if model_args.num_mtp_modules > 0:
            self.mtp_layers = torch.nn.ModuleDict()
            for mtp_layer_id in range(model_args.num_mtp_modules):
                layer_id = mtp_layer_id + model_args.n_layers
                self.mtp_layers[str(mtp_layer_id)] = MTPModule(
                    layer_id,
                    model_args,
                    self,
                )

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
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
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

    def forward(
        self,
        inputs: MTPInputs,
        input_batch: torch.Tensor | None = None,
    ) -> MTPInputsDict:
        """
        Perform a forward pass through the Transformer model.

        Args:
            inputs (MTPInputs): Single tensor or dictionary containing the
                following keys and values:
                - tokens_list (Union[list[torch.Tensor | None], torch.Tensor]):
                  Input token indices if pipeline parallelism is not enabled.
                  If pipeline parallelism is enabled, this will be the input token indices
                  for the ranks on the first pipeline stage. This will be the activation of the
                  previous pipeline stage if the current rank is not on the first stage.
                - prev_embed (torch.Tensor | None): Output token embeddings
                  of previous Transformer layer (after output norm, before
                  unembedding).
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            MTPInputsDict: Dictionary containing the following keys and
                values:
                - tokens_list (list[torch.Tensor | None]): Output logits
                  after applying the Transformer model for each output token.
                - prev_embed (torch.Tensor | None): Output token embeddings
                  of previous Transformer layer (after output norm, before
                  unembedding).

        """
        if not isinstance(inputs, dict):
            inputs = {"tokens_list": inputs}
        tokens_list = inputs["tokens_list"]
        prev_embed = inputs.get("prev_embed", None)
        if not isinstance(tokens_list, list):
            tokens = tokens_list
            tokens_list = [None] * (1 + self.model_args.num_mtp_modules)
            tokens_list[0] = tokens
        else:
            tokens = tokens_list[0]

        if input_batch is None:
            input_batch = tokens

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = (
            self.tok_embeddings(tokens[:, : self.model_args.max_seq_len])
            if self.tok_embeddings
            else tokens
        )

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h

        output = self.output(h) if self.output else h
        tokens_list[0] = output

        # Calculate multi-token predictions.

        if self.model_args.num_mtp_modules > 0:
            # Check if output norm is in this stage. If yes, assign the
            # hidden embedding.
            if self.norm and prev_embed is None:
                prev_embed = h

            for (mtp_layer_id, mtp_layer) in self.mtp_layers.items():
                mtp_layer_id = int(mtp_layer_id)
                token_offset = mtp_layer_id + 1
                output, prev_embed = mtp_layer(
                    input_batch[
                        :, token_offset : token_offset + self.model_args.max_seq_len
                    ],
                    prev_embed,
                    freqs_cis,
                    start_pos=start_pos,
                )
                tokens_list[mtp_layer_id + 1] = output

        return {
            "tokens_list": tokens_list,
            "prev_embed": prev_embed,
        }
