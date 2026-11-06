# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The HF model of opt_moe (shipped with an HF checkpoint, loaded with
``trust_remote_code``; no torchtitan import). Two forwards over the same
weights:

- ``packed_forward``: torchtitan's packed token rows (``position_ids`` restart
  at 0 per document) with torchtitan's ops in torchtitan's order and dtypes
  (FlexAttention, attn_gym's conv, bf16 grouped GEMMs, a deterministic
  scatter-add): torchtitan's forward bitwise, MTP depths included.
- ``forward`` (``generate``): ``[B, T]`` rows with left padding, a
  ``DynamicCache`` from ``layer_types`` (a ``hybrid`` layer keeps K / V and
  each conv stream's last inputs), ``masking_utils`` masks, eager fp32
  attention with the sinks as one more softmax logit (gpt_oss), a torch conv
  (Inkling). Other kernels, so other rounding; serving speed is vLLM's.

Tensor suffixes: ``B`` rows, ``T`` tokens, ``S`` cached + new tokens, ``D``
model dim, ``H`` heads, ``G`` KV heads, ``K`` head dim, ``C`` conv channels,
``E`` experts, ``F`` FFN dim, ``N`` routed (token, expert) pairs."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (
    AuxRequest,
    create_block_mask,
    flex_attention,
)
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutput,
    CausalLMOutputWithPast,
)

from .configuration_opt_moe import OptMoeConfig

# torchtitan's FlexInnerAttention compile (models/common/attention/attention.py).
_FLEX_OPTIONS = {
    "wrap_inductor_compiled_regions": True,
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": False,
}
_compiled_flex = torch.compile(flex_attention, fullgraph=True, options=_FLEX_OPTIONS)
_compiled_create_block_mask = torch.compile(create_block_mask)


def _block_mask(positions_T: torch.Tensor, window: int | None):
    """torchtitan's FlexAttention mask of a packed row: causal within each
    document, and within ``window`` tokens on a sliding-window layer."""
    document_T = torch.cumsum(positions_T == 0, dim=0)

    def mask_mod(b, h, q_idx, kv_idx):
        allowed = (document_T[q_idx] == document_T[kv_idx]) & (kv_idx <= q_idx)
        return allowed if window is None else allowed & (q_idx - kv_idx < window)

    seq_len = positions_T.shape[0]
    return _compiled_create_block_mask(
        mask_mod,
        1,
        None,
        seq_len,
        seq_len,
        device=positions_T.device,
        BLOCK_SIZE=128,
        separate_full_blocks=True,
    )


def _document_offsets(positions_T: torch.Tensor) -> torch.Tensor:
    """``cu_seqlens`` of the packed documents (int32, ``[num_docs + 1]``)."""
    starts = torch.nonzero(positions_T == 0).flatten()
    end = torch.tensor([positions_T.shape[0]], device=positions_T.device)
    return torch.cat([starts, end]).to(torch.int32)


def _canon_conv(x_TC: torch.Tensor, conv: nn.Conv1d, cu_seqlens: torch.Tensor):
    """``x + conv(x)``: attn_gym's depthwise causal conv per document."""
    from attn_gym.linear.short_conv import causal_conv1d

    out_1TC = causal_conv1d(
        x_TC.unsqueeze(0),
        conv.weight.squeeze(1).to(x_TC.dtype),
        cu_seqlens=cu_seqlens,
    )
    return out_1TC.squeeze(0) + x_TC


def _cached_canon_conv(x_BTC, conv: nn.Conv1d, rows: "_Rows", layer_idx, state_idx):
    """``x + conv(x)`` continuing the cache's conv state: the depthwise causal
    conv in fp32 over the cached and the new inputs (Inkling's short conv),
    rounded to x's dtype before the residual add as attn_gym's conv. Padding
    enters as zeros: a left-padded row sees the zero history of a row alone."""
    t, w = x_BTC.shape[1], conv.kernel_size[0]
    inputs_BTC = x_BTC
    if rows.conv_mask is not None:
        inputs_BTC = x_BTC * rows.conv_mask[:, :, None].to(x_BTC.dtype)
    inputs_BCS = inputs_BTC.transpose(1, 2)
    if rows.cache is not None:
        inputs_BCS = rows.cache.update_conv_state(
            inputs_BCS, layer_idx, state_idx=state_idx, conv_kernel_size=w
        )
    s = inputs_BCS.shape[-1]
    out_BCS = F.conv1d(
        inputs_BCS.float(),
        conv.weight.float(),
        padding=w - 1,
        groups=conv.weight.shape[0],
    )[..., :s]
    return out_BCS[..., -t:].transpose(1, 2).to(x_BTC.dtype) + x_BTC


def _rms_norm(config: OptMoeConfig, dim: int) -> nn.RMSNorm:
    return nn.RMSNorm(
        dim, eps=config.rms_norm_eps, elementwise_affine=config.norm_has_weight
    )


@dataclass
class _Rows:
    """What a layer needs besides its input. ``packed``: torchtitan's packed
    row (``masks`` FlexAttention block masks, ``cu_seqlens`` the documents for
    the conv); else ``[B, T]`` rows continuing ``cache`` (``masks`` additive
    ``[B, 1, T, S]`` masks, ``conv_mask`` 0 on padding). ``masks`` is keyed by
    the sliding window (None: full attention)."""

    packed: bool
    positions: torch.Tensor
    masks: dict
    cu_seqlens: torch.Tensor | None = None
    conv_mask: torch.Tensor | None = None
    cache: Cache | None = None


class OptMoeRotary(nn.Module):
    """torchtitan's CosSinRoPE (rotate-half) over the first ``dim`` dims: the
    angles ``position x inv_freq`` in fp32 on the input's device, the values of
    torchtitan's cos / sin cache."""

    def __init__(self, dim: int, theta: float):
        super().__init__()
        self.dim, self.theta = dim, theta

    def forward(self, q_THK, k_THK, positions_T):
        """``q`` / ``k`` ``[..., T, heads, K]``, ``positions`` ``[..., T]``."""
        d = self.dim
        exponents = torch.arange(0, d, 2, device=q_THK.device)[: d // 2].float() / d
        freqs_TC = positions_T.float().unsqueeze(-1) * (1.0 / self.theta**exponents)
        angles_T1C = torch.cat([freqs_TC, freqs_TC], dim=-1).unsqueeze(-2)
        cos, sin = angles_T1C.cos(), angles_T1C.sin()

        def rotate(x_THK):
            x = x_THK[..., :d].float()
            x1, x2 = x.chunk(2, dim=-1)
            out = ((x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)).type_as(x_THK)
            return out if d == x_THK.shape[-1] else torch.cat([out, x_THK[..., d:]], -1)

        return rotate(q_THK), rotate(k_THK)


class OptMoeAttention(nn.Module):
    def __init__(self, config: OptMoeConfig, layer_idx: int):
        super().__init__()
        d, h, kv, k = (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.layer_idx = layer_idx
        self.n_heads, self.n_kv_heads, self.head_dim = h, kv, k
        self.q_proj = nn.Linear(d, h * k, bias=False)
        self.k_proj = nn.Linear(d, kv * k, bias=False)
        self.v_proj = nn.Linear(d, kv * k, bias=False)
        self.o_proj = nn.Linear(h * k, d, bias=False)
        self.q_norm = _rms_norm(config, k) if config.qk_norm else None
        self.k_norm = _rms_norm(config, k) if config.qk_norm else None
        self.gate_proj = (
            nn.Linear(d, h * k, bias=False)
            if config.attention_gate == "element-wise"
            else None
        )
        w = config.conv_kernel_size
        conv_q = w > 0 and config.conv_streams == "qkv"

        def conv(channels):
            return nn.Conv1d(channels, channels, w, groups=channels, bias=False)

        self.q_conv = conv(h * k) if conv_q else None
        self.k_conv = conv(kv * k) if w > 0 else None
        self.v_conv = conv(kv * k) if w > 0 else None
        self.sinks = nn.Parameter(torch.zeros(h)) if config.attention_sinks else None
        rope = config.rope_layers[layer_idx]
        self.rope = None if rope is None else OptMoeRotary(rope["dim"], rope["theta"])
        self.window = (
            config.sliding_window
            if config.layer_types[layer_idx] in ("sliding_attention", "hybrid_sliding")
            else None
        )

    def _fused_qkv_weight(self) -> torch.Tensor:
        """torchtitan's ``wqkv``: per KV group, its q heads, then k, then v."""
        kv, k = self.n_kv_heads, self.head_dim
        d = self.q_proj.weight.shape[1]
        return torch.cat(
            [
                self.q_proj.weight.view(kv, -1, d),
                self.k_proj.weight.view(kv, k, d),
                self.v_proj.weight.view(kv, k, d),
            ],
            dim=1,
        ).reshape(-1, d)

    def forward(self, x_TD, rows):
        """``x`` ``[T, D]`` (packed rows) or ``[B, T, D]``."""
        lead = x_TD.shape[:-1]
        qkv = F.linear(x_TD, self._fused_qkv_weight())
        qkv = qkv.view(*lead, -1, self.n_heads // self.n_kv_heads + 2, self.head_dim)
        xq, xk, xv = torch.split(qkv, [self.n_heads // self.n_kv_heads, 1, 1], dim=-2)
        xq = xq.reshape(*lead, -1, self.head_dim).contiguous()
        xk = xk.reshape(*lead, -1, self.head_dim).contiguous()
        xv = xv.reshape(*lead, -1, self.head_dim).contiguous()
        gate = self.gate_proj(x_TD) if self.gate_proj is not None else None
        if self.k_conv is not None:
            # the cache's conv states: q 0, k 1, v 2 (or k 0, v 1)
            k_state = 0 if self.q_conv is None else 1
            if self.q_conv is not None:
                xq = self._conv(xq, self.q_conv, rows, 0)
            xk = self._conv(xk, self.k_conv, rows, k_state)
            xv = self._conv(xv, self.v_conv, rows, k_state + 1)
        if self.q_norm is not None:
            xq, xk = self.q_norm(xq), self.k_norm(xk)
        if self.rope is not None:
            xq, xk = self.rope(xq, xk, rows.positions)
        if rows.packed:
            out_THK = self._flex(xq, xk, xv, rows.masks[self.window])
        else:
            out_THK = self._eager(xq, xk, xv, rows)
        out_TF = out_THK.contiguous().reshape(*lead, -1)
        if gate is not None:
            out_TF = (out_TF.float() * torch.sigmoid(gate.float())).to(out_TF.dtype)
        return self.o_proj(out_TF)

    def _conv(self, x_THK, conv, rows, state_idx):
        x_TC = x_THK.flatten(-2)
        if rows.packed:
            out_TC = _canon_conv(x_TC, conv, rows.cu_seqlens)
        else:
            out_TC = _cached_canon_conv(x_TC, conv, rows, self.layer_idx, state_idx)
        return out_TC.view_as(x_THK)

    def _flex(self, xq_THK, xk_TGK, xv_TGK, block_mask):
        """torchtitan's FlexAttention on one packed row; the sinks rescale the
        output by ``sigmoid(lse - sink)``."""
        out, aux = _compiled_flex(
            xq_THK.transpose(0, 1).unsqueeze(0),
            xk_TGK.transpose(0, 1).unsqueeze(0),
            xv_TGK.transpose(0, 1).unsqueeze(0),
            block_mask=block_mask,
            scale=self.head_dim**-0.5,
            enable_gqa=self.n_heads != self.n_kv_heads,
            return_aux=AuxRequest(lse=self.sinks is not None),
            kernel_options={},
        )
        out_THK = out.squeeze(0).transpose(0, 1)
        if self.sinks is not None:
            lse_TH = aux.lse.squeeze(0).transpose(0, 1)
            scale = torch.sigmoid(lse_TH - self.sinks.view(1, -1)).unsqueeze(-1)
            out_THK = out_THK * scale.to(out_THK.dtype)
        return out_THK

    def _eager(self, xq_BTHK, xk_BTGK, xv_BTGK, rows):
        """Eager attention in fp32 over the cached and the new K / V, the sinks
        one more logit of the softmax (gpt_oss; the same as torchtitan's
        ``sigmoid(lse - sink)`` rescale)."""
        q_BHTK = xq_BTHK.transpose(1, 2)
        k_BGSK, v_BGSK = xk_BTGK.transpose(1, 2), xv_BTGK.transpose(1, 2)
        if rows.cache is not None:
            k_BGSK, v_BGSK = rows.cache.update(k_BGSK, v_BGSK, self.layer_idx)
        # q head i reads KV head i // (heads per KV head): torchtitan's layout
        groups = self.n_heads // self.n_kv_heads
        k_BHSK = k_BGSK.repeat_interleave(groups, dim=1).float()
        v_BHSK = v_BGSK.repeat_interleave(groups, dim=1).float()
        scores = torch.matmul(q_BHTK.float(), k_BHSK.transpose(-2, -1))
        scores = scores * self.head_dim**-0.5
        mask = rows.masks[self.window]
        if mask is not None:
            scores = scores + mask[..., : k_BHSK.shape[-2]]
        if self.sinks is not None:
            b, h, t, _ = scores.shape
            sinks = self.sinks.float().view(1, h, 1, 1).expand(b, h, t, 1)
            scores = torch.cat([scores, sinks], dim=-1)
        probs = F.softmax(scores, dim=-1)
        if self.sinks is not None:
            probs = probs[..., :-1]
        return torch.matmul(probs, v_BHSK).transpose(1, 2).to(xq_BTHK.dtype)


class OptMoeMLP(nn.Module):
    """SwiGLU; gate and up as one GEMM (torchtitan's ``w13``)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x_TD):
        w13 = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        gate_up = F.linear(x_TD, w13).unflatten(-1, (2, -1))
        gate, up = gate_up.unbind(-2)
        return self.down_proj(F.silu(gate) * up)


class OptMoeRouter(nn.Module):
    def __init__(self, config: OptMoeConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.e_score_correction_bias = (
            nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32))
            if config.router_bias
            else None
        )


class OptMoeExperts(nn.Module):
    def __init__(self, config: OptMoeConfig):
        super().__init__()
        e, d, f = (
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * f, d))
        self.down_proj = nn.Parameter(torch.empty(e, d, f))


class OptMoeSparseMoe(nn.Module):
    def __init__(self, config: OptMoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.norm_topk_epsilon = config.norm_topk_epsilon
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = OptMoeRouter(config)
        self.experts = OptMoeExperts(config)
        self.shared_experts = (
            OptMoeMLP(config.hidden_size, config.shared_expert_intermediate_size)
            if config.shared_expert_intermediate_size
            else None
        )

    def _scores(self, gate_TE: torch.Tensor) -> torch.Tensor:
        if self.scoring_func == "sigmoid":
            return torch.sigmoid(gate_TE)
        if self.scoring_func == "softmax":
            return F.softmax(gate_TE, dim=-1)
        if self.scoring_func == "sqrtsoftplus":
            return torch.sqrt(F.softplus(gate_TE))
        raise ValueError(f"unknown scoring_func {self.scoring_func!r}")

    def forward(self, x):
        """``x`` ``[..., D]``: every token routed on its own."""
        x_TD = x.reshape(-1, x.shape[-1])
        w = self.gate.weight
        if x_TD.is_cuda and x_TD.dtype == w.dtype == torch.bfloat16:
            gate_TE = torch.mm(x_TD, w.T, out_dtype=torch.float32)
        else:
            gate_TE = torch.mm(x_TD.float(), w.float().T)
        scores_TE = self._scores(gate_TE)
        bias = self.gate.e_score_correction_bias
        choice_TE = scores_TE if bias is None else scores_TE + bias
        ids_TK = torch.topk(choice_TE, k=self.top_k, dim=-1, sorted=False).indices
        topk_TK = scores_TE.gather(dim=-1, index=ids_TK)
        if self.norm_topk_prob:
            topk_TK = topk_TK / (
                topk_TK.sum(dim=-1, keepdim=True) + self.norm_topk_epsilon
            )
        topk_TK = topk_TK * self.routed_scaling_factor

        counts_E = torch.bincount(ids_TK.view(-1), minlength=w.shape[0])
        order_N = torch.argsort(ids_TK.view(-1), stable=True)
        scores_N = topk_TK.view(-1)[order_N]
        tokens_N = order_N // self.top_k
        routed_ND = x_TD[tokens_N]
        offsets_E = torch.cumsum(counts_E, dim=0, dtype=torch.int32)
        gate_up = torch._grouped_mm(
            routed_ND.bfloat16(),
            self.experts.gate_up_proj.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).unflatten(-1, (2, -1))
        gate, up = gate_up.unbind(-2)
        out_ND = torch._grouped_mm(
            F.silu(gate) * up,
            self.experts.down_proj.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(routed_ND)
        out_ND = (out_ND * scores_N.reshape(-1, 1).to(torch.float32)).to(out_ND.dtype)
        prev = torch.are_deterministic_algorithms_enabled()
        prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            out_TD = torch.zeros_like(x_TD).scatter_add(
                0, tokens_N.reshape(-1, 1).expand(-1, x_TD.shape[-1]), out_ND
            )
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
        if self.shared_experts is not None:
            out_TD = out_TD + self.shared_experts(x_TD)
        return out_TD.view_as(x)


class OptMoeDecoderLayer(nn.Module):
    def __init__(self, config: OptMoeConfig, layer_idx: int):
        super().__init__()
        d = config.hidden_size
        self.self_attn = OptMoeAttention(config, layer_idx)
        self.input_layernorm = _rms_norm(config, d)
        self.post_attention_layernorm = _rms_norm(config, d)
        self.mlp = (
            OptMoeMLP(d, config.intermediate_size)
            if layer_idx < config.first_k_dense_replace
            else OptMoeSparseMoe(config)
        )
        self.block_scale = config.residual_block_scales[layer_idx]
        self.identity_scale = config.residual_identity_scales[layer_idx]

    def _residual(self, x, f):
        lhs = x if self.identity_scale == 1.0 else self.identity_scale * x
        rhs = f if self.block_scale == 1.0 else self.block_scale * f
        return lhs + rhs

    def forward(self, x_TD, rows):
        attn = self.self_attn(self.input_layernorm(x_TD), rows)
        h = self._residual(x_TD, attn)
        return self._residual(h, self.mlp(self.post_attention_layernorm(h)))


class OptMoeMTPLayer(OptMoeDecoderLayer):
    """One MTP depth (DeepSeek-V3): ``shared_head.norm(layer(eh_proj(cat(
    enorm(e), hnorm(h)))))``, the decoder layer a copy of the last one."""

    def __init__(self, config: OptMoeConfig, layer_idx: int):
        super().__init__(config, config.num_hidden_layers - 1)
        d = config.hidden_size
        self.enorm, self.hnorm = _rms_norm(config, d), _rms_norm(config, d)
        self.eh_proj = nn.Linear(2 * d, d, bias=False)
        self.shared_head = nn.Module()
        self.shared_head.norm = _rms_norm(config, d)

    def forward(self, embed_TD, prev_TD, valid_T, rows):
        prev_TD = prev_TD * valid_T.unsqueeze(-1).to(prev_TD.dtype)
        h = self.eh_proj(torch.cat([self.enorm(embed_TD), self.hnorm(prev_TD)], dim=-1))
        return self.shared_head.norm(super().forward(h, rows))


class OptMoePreTrainedModel(PreTrainedModel):
    config_class = OptMoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["OptMoeDecoderLayer", "OptMoeMTPLayer"]
    # eager attention only (the sinks join its softmax); a hybrid DynamicCache
    # (K / V + conv states) does not compile
    _can_compile_fullgraph = False

    def _init_weights(self, module: nn.Module) -> None:
        """Fills fresh tensors only: the weights come from a checkpoint."""
        for p in module.parameters(recurse=False):
            nn.init.normal_(p, std=0.02)


class OptMoeModel(OptMoePreTrainedModel):
    def __init__(self, config: OptMoeConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        n = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [OptMoeDecoderLayer(config, i) for i in range(n)]
            + [
                OptMoeMTPLayer(config, n + i)
                for i in range(config.num_nextn_predict_layers)
            ]
        )
        self.norm = _rms_norm(config, config.hidden_size)
        self.post_init()

    def packed_forward(
        self, input_ids_T, positions_T, mtp_input_ids=None, mtp_valid=None
    ):
        """``input_ids_T`` / ``positions_T``: one packed token row, bitwise
        torchtitan's forward. Returns the final hidden states, and with
        ``mtp_input_ids`` (each depth's rolled tokens) / ``mtp_valid`` the MTP
        depths' hidden states."""
        n = self.config.num_hidden_layers
        windows = {layer.self_attn.window for layer in self.layers[:n]}
        rows = _Rows(
            packed=True,
            positions=positions_T,
            masks={w: _block_mask(positions_T, w) for w in windows},
            cu_seqlens=(
                _document_offsets(positions_T) if self.config.conv_kernel_size else None
            ),
        )
        h = self.embed_tokens(input_ids_T)
        for layer in self.layers[:n]:
            h = layer(h, rows)
        prev, outputs = h, [self.norm(h)]
        for layer, ids, valid in zip(
            self.layers[n:], mtp_input_ids or (), mtp_valid or (), strict=False
        ):
            prev = layer(self.embed_tokens(ids), prev, valid, rows)
            outputs.append(prev)
        return outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """``[B, T]`` rows (left padding: ``attention_mask`` 0), continuing
        ``past_key_values``; the main layers only (no MTP)."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past = 0 if past_key_values is None else past_key_values.get_seq_length()
            position_ids = torch.arange(
                past, past + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        n = self.config.num_hidden_layers
        windows = {layer.self_attn.window for layer in self.layers[:n]}
        masks = {
            w: (create_causal_mask if w is None else create_sliding_window_causal_mask)(
                **mask_kwargs
            )
            for w in windows
        }
        rows = _Rows(
            packed=False,
            positions=position_ids,
            masks=masks,
            conv_mask=(
                create_recurrent_attention_mask(**mask_kwargs)
                if self.config.conv_kernel_size
                else None
            ),
            cache=past_key_values,
        )
        h = inputs_embeds
        for layer in self.layers[:n]:
            h = layer(h, rows)
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(h), past_key_values=past_key_values
        )


class OptMoeForCausalLM(OptMoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: OptMoeConfig):
        super().__init__(config)
        self.model = OptMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """The transformers forward (``generate``): logits ``[B, T, V]`` of the
        last ``logits_to_keep`` positions (0: all), the cache, and with
        ``labels`` the shifted next-token loss."""
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        keep = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(out.last_hidden_state[:, keep, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=out.past_key_values
        )

    def packed_forward(self, input_ids, position_ids=None):
        """torchtitan's forward, bitwise: ``input_ids`` ``[B, T]`` packed rows,
        flattened into one token stream (``position_ids`` restart at 0 per
        document; default: one document per row); the main logits."""
        b, t = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(t, device=input_ids.device).expand(b, t)
        hidden = self.model.packed_forward(
            input_ids.reshape(-1), position_ids.reshape(-1)
        )[0]
        return CausalLMOutput(logits=self.lm_head(hidden).view(b, t, -1))


__all__ = ["OptMoeForCausalLM", "OptMoeModel", "OptMoePreTrainedModel"]
