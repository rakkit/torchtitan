# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe in vLLM: a native model for an opt_moe HF checkpoint folder (its
``config.json`` is torchtitan's ``opt_moe/hf/configuration_opt_moe.py``
``OptMoeConfig``, loaded with trust_remote_code; ``register`` in
``__init__.py`` adds this model to vLLM's registry, the package's
``vllm.general_plugins`` entry point). No torchtitan import.

Each decoder layer holds two caches: the attention's KV cache (``Attention``
with the sinks, the layer's sliding window) and the q / k / v short conv's
state, the last ``conv_kernel_size`` raw inputs per sequence, kept as one more
sliding-window KV cache (window = block = the conv width) as vLLM's Inkling
model keeps its conv state (``OptMoeConvState``; vLLM's Inkling sconv backend
and ``fused_sconv`` kernel: insert + conv in one launch, positions addressed,
the same path for prefill / decode / speculative tokens, CUDA-graph safe). The ops follow the HF model's
transformers forward: q / k / v -> x + conv(x) -> qk-norm -> RoPE (none on
NoPE layers, partial when ``dim`` < head_dim) -> attention -> the element-wise
gate sigmoid(gate_proj(x)) -> o_proj; RMSNorm blocks with the per-layer
residual scales; the first ``first_k_dense_replace`` layers a SwiGLU FFN, the
rest a sigmoid-routed MoE with the score-correction bias for the choice and a
shared expert. The MTP depths are not served (``num_nextn_predict_layers`` is
ignored here).

Tensor suffixes: ``T`` tokens, ``D`` model dim, ``C`` conv channels (q, k, v of
this TP rank), ``W`` conv width, ``H`` / ``G`` this rank's query / KV heads,
``K`` head dim, ``E`` experts, ``F`` expert FFN dim."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import get_current_vllm_config, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling.nvidia.ops import fused_sconv
from vllm.models.inkling.nvidia.sconv_swa_attn import (
    InklingSconvBackend,
    InklingSconvMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec


def _sliding(config, layer_idx: int) -> bool:
    return config.layer_types[layer_idx] in ("sliding_attention", "hybrid_sliding")


class OptMoeConvState(nn.Module, AttentionLayerBase):
    """A layer's conv state: one sliding-window KV cache (window = block = the
    conv width) holding, per KV head of this rank, its q heads' / k / v raw
    inputs packed head-major into one width (padded to a power of two, as
    Inkling's, so every layer's page is the same size). The convs read and
    write it out of band (``OptMoeShortConv``); vLLM's Inkling sconv backend
    builds the block table / slot metadata."""

    def __init__(self, config, *, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        tp = get_tensor_model_parallel_world_size()
        if tp > config.num_key_value_heads:
            raise NotImplementedError("vLLM opt_moe: TP above the KV head count")
        k = config.head_dim
        q_width = config.num_attention_heads // config.num_key_value_heads * k
        self.num_kv_heads = config.num_key_value_heads // tp
        self.head_size = 1 << (q_width + 2 * k - 1).bit_length()
        self.sliding_window = self.block_size = config.conv_kernel_size
        # per KV head: (offset, width) of q (its group's heads), k, v
        self.stream_ranges = ((0, q_width), (q_width, k), (q_width + k, k))
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        if self._dtype != torch.bfloat16:
            raise NotImplementedError("vLLM opt_moe: the conv state is bf16 only")
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def forward(self):  # the convs use this layer's cache out of band
        raise NotImplementedError

    def get_attn_backend(self):
        return InklingSconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,  # the streams packed into head_size
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )


class OptMoeShortConv(nn.Module):
    """``x + conv(x)`` for one stream (q, k or v) of this rank: the depthwise
    causal conv over each sequence's last ``W`` inputs, read from / written to
    the layer's ``OptMoeConvState``."""

    def __init__(self, channels: int, width: int, owner: OptMoeConvState, stream: int):
        super().__init__()
        self.owner, self.stream = owner, stream
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = nn.Parameter(torch.empty(channels, 1, width), requires_grad=False)
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if loaded.shape[0] != param.shape[0]:  # this rank's heads' channels
            n = param.shape[0]
            loaded = loaded.narrow(0, self.tp_rank * n, n)
        param.data.copy_(loaded)

    def forward(self, x_TC: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):  # memory profiling: no state
            return x_TC
        m = metadata.get(self.owner.prefix)
        if m is None or self.owner.kv_cache.numel() == 0:
            return x_TC
        assert isinstance(m, InklingSconvMetadata)
        offset, width = self.owner.stream_ranges[self.stream]
        return fused_sconv(
            x_TC.contiguous(),
            self.weight.squeeze(1),
            self.owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            self.owner.block_size,
            activation=None,
            use_residual=True,
        )


class OptMoeAttention(nn.Module):
    def __init__(self, config, layer_idx: int, *, cache_config, quant_config, prefix):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        d, k = config.hidden_size, config.head_dim
        self.n_heads = config.num_attention_heads // tp
        self.n_kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim = k
        self.q_size, self.kv_size = self.n_heads * k, self.n_kv_heads * k
        self.qkv_proj = QKVParallelLinear(
            d,
            k,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * k,
            d,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.conv_state = None
        if config.conv_kernel_size:
            if config.conv_streams != "qkv":
                raise NotImplementedError("vLLM opt_moe: conv_streams kv")
            w = config.conv_kernel_size
            self.conv_state = OptMoeConvState(config, prefix=f"{prefix}.conv_state")
            self.q_conv = OptMoeShortConv(self.q_size, w, self.conv_state, 0)
            self.k_conv = OptMoeShortConv(self.kv_size, w, self.conv_state, 1)
            self.v_conv = OptMoeShortConv(self.kv_size, w, self.conv_state, 2)
        affine, eps = config.norm_has_weight, config.rms_norm_eps
        self.q_norm = RMSNorm(k, eps=eps, has_weight=affine) if config.qk_norm else None
        self.k_norm = RMSNorm(k, eps=eps, has_weight=affine) if config.qk_norm else None
        self.gate_proj = (
            ColumnParallelLinear(
                d,
                config.num_attention_heads * k,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )
            if config.attention_gate == "element-wise"
            else None
        )
        rope = config.rope_layers[layer_idx]
        self.rotary_emb = (
            None
            if rope is None
            else get_rope(
                k,
                max_position=config.max_position_embeddings,
                is_neox_style=True,  # torchtitan's rotate-half
                rope_parameters={"rope_theta": rope["theta"], "rope_dim": rope["dim"]},
                dtype=torch.float32,
            )
        )
        # the model dtype (vLLM sets the default dtype while building): the
        # FlashAttention backends take the sinks ("S aux") in bf16 only
        self.sinks = (
            nn.Parameter(torch.empty(self.n_heads), requires_grad=False)
            if config.attention_sinks
            else None
        )
        self.attn = Attention(
            self.n_heads,
            k,
            k**-0.5,
            num_kv_heads=self.n_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=(
                config.sliding_window if _sliding(config, layer_idx) else None
            ),
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

    def forward(self, positions: torch.Tensor, x_TD: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x_TD)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.conv_state is not None:
            q = self.q_conv(q, positions)
            k = self.k_conv(k, positions)
            v = self.v_conv(v, positions)
        if self.q_norm is not None:
            t = q.shape[0]
            q = self.q_norm(q.reshape(t, -1, self.head_dim)).reshape(t, -1)
            k = self.k_norm(k.reshape(t, -1, self.head_dim)).reshape(t, -1)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)
        if self.gate_proj is not None:
            gate, _ = self.gate_proj(x_TD)
            out = (out.float() * torch.sigmoid(gate.float())).to(out.dtype)
        out, _ = self.o_proj(out)
        return out


class OptMoeMLP(nn.Module):
    def __init__(self, d: int, f: int, *, quant_config, prefix, reduce_results=True):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            d,
            [f, f],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            f,
            d,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act = SiluAndMul()

    def forward(self, x_TD):
        h, _ = self.gate_up_proj(x_TD)
        out, _ = self.down_proj(self.act(h))
        return out


class OptMoeSparseMoe(nn.Module):
    """Sigmoid scores; the choice by score + e_score_correction_bias; the
    chosen scores renormalized and scaled (vLLM's grouped top-k with one
    group: the plain top-k with the bias); the shared expert added."""

    def __init__(self, config, *, quant_config, prefix):
        super().__init__()
        if config.scoring_func != "sigmoid":
            raise NotImplementedError(
                f"vLLM opt_moe: scoring_func {config.scoring_func}"
            )
        d, e = config.hidden_size, config.n_routed_experts
        self.gate = nn.Linear(d, e, bias=False, dtype=torch.float32)
        self.gate.e_score_correction_bias = (
            nn.Parameter(torch.zeros(e, dtype=torch.float32))
            if config.router_bias
            else None
        )
        self.shared_experts = (
            OptMoeMLP(
                d,
                config.shared_expert_intermediate_size,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
            if config.shared_expert_intermediate_size
            else None
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=e,
            top_k=config.num_experts_per_tok,
            hidden_size=d,
            intermediate_size=config.moe_intermediate_size,
            renormalize=bool(config.norm_topk_prob),
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_scaling_factor=config.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            router_logits_dtype=torch.float32,
        )

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        logits_TE = self.gate(x_TD.to(torch.float32))
        return self.experts(hidden_states=x_TD, router_logits=logits_TE)


class OptMoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, *, cache_config, quant_config, prefix):
        super().__init__()
        d, eps, affine = config.hidden_size, config.rms_norm_eps, config.norm_has_weight
        self.self_attn = OptMoeAttention(
            config,
            layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(d, eps=eps, has_weight=affine)
        self.post_attention_layernorm = RMSNorm(d, eps=eps, has_weight=affine)
        self.mlp = (
            OptMoeMLP(
                d,
                config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            if layer_idx < config.first_k_dense_replace
            else OptMoeSparseMoe(
                config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        )
        self.block_scale = config.residual_block_scales[layer_idx]
        self.identity_scale = config.residual_identity_scales[layer_idx]

    def _residual(self, x, f):
        return self.identity_scale * x + self.block_scale * f

    def forward(self, positions, x_TD):
        h = self._residual(x_TD, self.self_attn(positions, self.input_layernorm(x_TD)))
        return self._residual(h, self.mlp(self.post_attention_layernorm(h)))


class OptMoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config, quant_config = vllm_config.cache_config, vllm_config.quant_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                OptMoeDecoderLayer(
                    config,
                    i,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=config.norm_has_weight,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None):
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            h = layer(positions, h)
        return self.norm(h)


class OptMoeForCausalLM(nn.Module):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = OptMoeModel(vllm_config=vllm_config, prefix=f"{prefix}model")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Our HF names (``hf/modeling_opt_moe.py``) onto this model: q / k / v
        into qkv_proj, gate / up into gate_up_proj, the 3-D expert stacks one
        expert at a time into FusedMoE; the q / k / v convs (same names) take
        their rank's channels in their own loader; the MTP layers are skipped."""
        params = dict(self.named_parameters())
        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        n_layers = self.config.num_hidden_layers
        loaded = set()
        for name, w in weights:
            parts = name.split(".")
            if name.startswith("model.layers.") and int(parts[2]) >= n_layers:
                continue  # MTP depth
            for shard, src in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
                if f".self_attn.{src}." in name:
                    key = name.replace(src, "qkv_proj")
                    params[key].weight_loader(params[key], w, shard)
                    loaded.add(key)
                    break
            else:
                if (
                    ".mlp.experts.gate_up_proj" in name
                    or ".mlp.experts.down_proj" in name
                ):
                    loaded |= self._load_experts(params, name, w)
                    continue
                for shard, src in ((0, "gate_proj"), (1, "up_proj")):
                    if f".mlp.{src}." in name or f".shared_experts.{src}." in name:
                        key = name.replace(src, "gate_up_proj")
                        params[key].weight_loader(params[key], w, shard)
                        loaded.add(key)
                        break
                else:
                    if name.endswith(".self_attn.sinks"):
                        w = w.chunk(tp)[rank]
                    param = params[name]
                    getattr(param, "weight_loader", default_weight_loader)(param, w)
                    loaded.add(name)
        return loaded

    def _load_experts(self, params, name: str, w_EXY: torch.Tensor) -> set[str]:
        prefix = name.rsplit(".mlp.experts.", 1)[0] + ".mlp.experts."
        if name.endswith("gate_up_proj"):
            f = w_EXY.shape[1] // 2
            targets = (
                ("w13_weight", "w1", w_EXY[:, :f]),
                ("w13_weight", "w3", w_EXY[:, f:]),
            )
        else:
            targets = (("w2_weight", "w2", w_EXY),)
        out = set()
        for param_name, shard, w in targets:
            # FusedMoE keeps them in a submodule (``routed_experts`` in vLLM 1.0)
            (key,) = [
                k for k in params if k.startswith(prefix) and k.endswith(param_name)
            ]
            param = params[key]
            for expert in range(w.shape[0]):
                param.weight_loader(
                    param, w[expert], key, shard_id=shard, expert_id=expert
                )
            out.add(key)
        return out
