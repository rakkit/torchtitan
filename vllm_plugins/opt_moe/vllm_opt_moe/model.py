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
with the sinks, the layer's sliding window) and the k / v short conv's
state, the last ``conv_kernel_size`` raw k / v per sequence, kept as one more
sliding-window KV cache (window = block = the conv width) as vLLM's Inkling
model keeps its conv state (``OptMoeConvState``; vLLM's Inkling sconv backend
for its block table and slots). The q / k / v preparation is one launch per
layer (``prologue.py``: positions addressed, the same path for prefill / decode /
speculative tokens, CUDA-graph safe). The ops follow the HF model's
transformers forward: q / k / v -> x + conv(x) on k / v -> qk-norm -> RoPE (none on
NoPE layers, partial when ``dim`` < head_dim) -> attention -> the element-wise
gate sigmoid(gate_proj(x)) -> o_proj; RMSNorm blocks with the per-layer
residual scales; the first ``first_k_dense_replace`` layers a SwiGLU FFN, the
rest a sigmoid-routed MoE with the score-correction bias for the choice and a
shared expert. The MTP depths are not served (``num_nextn_predict_layers`` is
ignored here).

Tensor suffixes: ``T`` tokens, ``D`` model dim, ``C`` conv channels (k, v of
this TP rank), ``W`` conv width, ``H`` / ``G`` this rank's query / KV heads,
``K`` head dim, ``E`` experts, ``F`` expert FFN dim."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
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
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
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
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling.nvidia.sconv_swa_attn import (
    InklingSconvBackend,
    InklingSconvMetadata,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec

from .prologue import attention_prologue


def _sliding(config, layer_idx: int) -> bool:
    return config.layer_types[layer_idx] in ("sliding_attention", "hybrid_sliding")


class OptMoeConvState(nn.Module, AttentionLayerBase):
    """A layer's conv state: one sliding-window KV cache (window = the conv
    width) holding the last raw k and v of each sequence, as ``2 G`` heads of
    ``head_dim`` (this rank's k heads, then its v heads: the channel order of
    the qkv projection's ``[k | v]`` slice). The conv reads and writes it out
    of band (``opt_moe_attention_prologue``); vLLM's Inkling sconv backend builds the
    block table / slot metadata."""

    def __init__(self, config, *, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        tp = get_tensor_model_parallel_world_size()
        if tp > config.num_key_value_heads:
            raise NotImplementedError("vLLM opt_moe: TP above the KV head count")
        self.num_kv_heads = config.num_key_value_heads // tp
        self.head_size = config.head_dim
        self.sliding_window = self.block_size = config.conv_kernel_size
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
            num_kv_heads=2 * self.num_kv_heads,  # k heads, then v heads
            head_size=self.head_size,
            head_size_v=0,  # one stream per head
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )


def opt_moe_attention_prologue(
    qkv_TC: torch.Tensor,
    conv_weight_CW: torch.Tensor,
    q_norm_weight: torch.Tensor | None,
    k_norm_weight: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    positions: torch.Tensor,
    layer_name: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    qk_norm: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The layer's attention prologue (``prologue.attention_prologue``: q / k
    / v from the projection, the k / v conv on the state, qk-norm, RoPE) as a
    custom op, so torch.compile sees one opaque call; the conv state and its
    metadata come from the forward context, as vLLM's short_conv op takes
    them."""
    context = get_forward_context()
    metadata = context.attn_metadata
    owner = context.no_compile_layers[layer_name]
    T, dev = qkv_TC.shape[0], qkv_TC.device
    if metadata is None:  # vLLM's profiling / dummy runs: no state, no conv
        none = torch.full((T,), -1, dtype=torch.int64, device=dev)
        zeros = torch.zeros((T,), dtype=torch.int32, device=dev)
        state = torch.empty(
            (1, 2 * num_kv_heads, 1, head_dim), dtype=qkv_TC.dtype, device=dev
        )
        table = torch.zeros((1, 1), dtype=torch.int32, device=dev)
        meta = (table, zeros, none, zeros)
    else:
        m = metadata.get(layer_name) if isinstance(metadata, dict) else None
        if not isinstance(m, InklingSconvMetadata) or owner.kv_cache.numel() == 0:
            # never run without the conv (the model would be wrong, not slow)
            raise RuntimeError(
                f"vLLM opt_moe: no conv state for {layer_name} in this forward "
                f"(metadata {type(m).__name__}, cache bound: "
                f"{owner.kv_cache.numel() > 0})"
            )
        state = owner.kv_cache
        meta = (m.block_table, m.seq_idx, m.slot_mapping, m.query_start)
    block_table, seq_idx, slot_mapping, query_start = meta
    return attention_prologue(
        qkv_TC,
        conv_weight_CW,
        q_norm_weight,
        k_norm_weight,
        qk_norm,
        eps,
        cos_sin_cache,
        rope_dim,
        state,
        positions,
        block_table,
        seq_idx,
        slot_mapping,
        query_start,
        num_heads,
        num_kv_heads,
        head_dim,
    )


def _opt_moe_attention_prologue_fake(
    qkv_TC: torch.Tensor,
    conv_weight_CW: torch.Tensor,
    q_norm_weight: torch.Tensor | None,
    k_norm_weight: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    positions: torch.Tensor,
    layer_name: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    qk_norm: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = qkv_TC.shape[0]
    q = qkv_TC.new_empty((T, num_heads * head_dim))
    return (
        q,
        qkv_TC.new_empty((T, num_kv_heads * head_dim)),
        qkv_TC.new_empty((T, num_kv_heads * head_dim)),
    )


direct_register_custom_op(
    op_name="opt_moe_attention_prologue",
    op_func=opt_moe_attention_prologue,
    fake_impl=_opt_moe_attention_prologue_fake,
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
            self.conv_state = OptMoeConvState(config, prefix=f"{prefix}.conv_state")
            # this rank's k conv rows, then its v conv rows (the HF k_conv /
            # v_conv weights): one parameter, no per-forward concatenation
            self.conv_weight = nn.Parameter(
                torch.empty(2 * self.kv_size, config.conv_kernel_size),
                requires_grad=False,
            )
            set_weight_attrs(self.conv_weight, {"weight_loader": self._load_conv})
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

    def _load_conv(self, param: nn.Parameter, loaded: torch.Tensor, shard: int):
        """An HF ``k_conv`` (shard 0) / ``v_conv`` (1) weight ``[G K, 1, W]``:
        this rank's heads' rows into its half of ``conv_weight``."""
        rows = loaded.squeeze(1)
        n = self.kv_size
        if rows.shape[0] != n:
            rows = rows.narrow(0, get_tensor_model_parallel_rank() * n, n)
        param.data[shard * n : (shard + 1) * n].copy_(rows)

    def forward(self, positions: torch.Tensor, x_TD: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x_TD)
        if self.conv_state is not None:
            # one launch: the k / v conv on its state, qk-norm, RoPE
            norm = self.q_norm
            q, k, v = torch.ops.vllm.opt_moe_attention_prologue(
                qkv,
                self.conv_weight,
                norm.weight if norm is not None and norm.has_weight else None,
                self.k_norm.weight if norm is not None and norm.has_weight else None,
                None if self.rotary_emb is None else self.rotary_emb.cos_sin_cache,
                positions,
                self.conv_state.prefix,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                0 if self.rotary_emb is None else self.rotary_emb.rotary_dim,
                norm is not None,
                1e-6 if norm is None else norm.variance_epsilon,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
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
        # bf16 weight x bf16 input, fp32 logits (cuBLAS / CuTe tiers): the
        # router's math in training, without an fp32 GEMM on CUDA cores
        self.gate = GateLinear(d, e, out_dtype=torch.float32, prefix=f"{prefix}.gate")
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
        logits_TE, _ = self.gate(x_TD)
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


@support_torch_compile
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

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            h = layer(positions, h)
        # pre-norm: the MTP draft reads it (torchtitan's MTP does); compute_logits norms
        return h


class OptMoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

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
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """The main model's tensors (the MTP depths, HF layers ``L ..``, are the
        draft's: ``OptMoeMTP``)."""
        n, tied = self.config.num_hidden_layers, self.config.tie_word_embeddings
        return _load_weights(
            self,
            weights,
            keep=lambda name: _layer_of(name) < n
            and not (tied and name == "lm_head.weight"),
        )


class OptMoeMTPLayer(OptMoeDecoderLayer):
    """One MTP depth: ``shared_head.norm(layer(eh_proj(cat(enorm(e), hnorm(h)))))``
    with the decoder layer built as the last one (torchtitan's OPTMoEMTPBlock,
    the HF model's OptMoeMTPLayer)."""

    def __init__(self, config, *, cache_config, quant_config, prefix):
        super().__init__(
            config,
            config.num_hidden_layers - 1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        d, eps, affine = config.hidden_size, config.rms_norm_eps, config.norm_has_weight
        self.enorm = RMSNorm(d, eps=eps, has_weight=affine)
        self.hnorm = RMSNorm(d, eps=eps, has_weight=affine)
        self.eh_proj = nn.Linear(2 * d, d, bias=False)
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(d, eps=eps, has_weight=affine)

    def forward(self, positions, embed_TD, prev_TD):  # pyrefly: ignore [bad-override]
        h = self.eh_proj(torch.cat([self.enorm(embed_TD), self.hnorm(prev_TD)], dim=-1))
        return self.shared_head.norm(super().forward(positions, h))


class OptMoeMTP(nn.Module):
    """The MTP depths as vLLM's speculative draft (``"method": "mtp"``, as for
    DeepSeek / Qwen). Draft step ``k`` runs depth ``k % depths`` on the target's
    pre-norm hidden state (step 0) or the previous step's output and returns
    its normed output: the lm_head input and the next step's hidden state, as
    torchtitan chains its depths. The embedding and lm_head are the target's
    (vLLM shares them with an MTP draft); the checkpoint's MTP layers are HF
    layers ``L ..``."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        n = config.num_hidden_layers
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict(
            {
                str(n + i): OptMoeMTPLayer(
                    config,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{maybe_prefix(prefix, 'model')}.layers.{n + i}",
                )
                for i in range(config.num_nextn_predict_layers)
            }
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx: int = 0,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        n, depths = self.config.num_hidden_layers, self.config.num_nextn_predict_layers
        depth = self.model.layers[str(n + spec_step_idx % depths)]
        return depth(positions, inputs_embeds, hidden_states)

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        n = self.config.num_hidden_layers
        return _load_weights(self, weights, keep=lambda name: _layer_of(name) >= n)


def _layer_of(name: str) -> int:
    """The HF layer of a checkpoint tensor (-1: embedding, final norm, lm_head)."""
    return int(name.split(".")[2]) if name.startswith("model.layers.") else -1


def _load_weights(model: nn.Module, weights, *, keep) -> set[str]:
    """Our HF names (``hf/modeling_opt_moe.py``) onto ``model``'s parameters,
    the checkpoint tensors ``keep`` picks: q / k / v into qkv_proj, gate / up
    into gate_up_proj, the 3-D expert stacks one expert at a time into
    FusedMoE; the k / v convs into the attention's conv_weight (their rank's
    channels)."""
    params = dict(model.named_parameters())
    tp, rank = (
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
    )
    loaded = set()
    for name, w in weights:
        if not keep(name):
            continue
        if name.endswith((".self_attn.k_conv.weight", ".self_attn.v_conv.weight")):
            key = name.rsplit(".", 2)[0] + ".conv_weight"
            params[key].weight_loader(params[key], w, int(".v_conv." in name))
            loaded.add(key)
            continue
        for shard, src in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
            if f".self_attn.{src}." in name:
                key = name.replace(src, "qkv_proj")
                params[key].weight_loader(params[key], w, shard)
                loaded.add(key)
                break
        else:
            if ".mlp.experts.gate_up_proj" in name or ".mlp.experts.down_proj" in name:
                loaded |= _load_experts(params, name, w)
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


def _load_experts(params, name: str, w_EXY: torch.Tensor) -> set[str]:
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
        (key,) = [k for k in params if k.startswith(prefix) and k.endswith(param_name)]
        param = params[key]
        for expert in range(w.shape[0]):
            param.weight_loader(param, w[expert], key, shard_id=shard, expert_id=expert)
        out.add(key)
    return out
