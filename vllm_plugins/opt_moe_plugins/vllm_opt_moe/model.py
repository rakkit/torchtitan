# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-only OptMoE model implemented with native vLLM layers.

Supports:
- Dense-only, MoE-only, or hybrid (first N layers dense, rest MoE)
- Per-layer RoPE pattern ("RRRN" etc.) — NoPE layers skip rotary embeddings
- Per-layer SWA pattern ("SSSF" etc.) — SWA layers use sliding-window attention
- Independent SWA-specific RoPE theta (rope_theta_swa + rope_parameters_swa)
- Gated attention: head-wise or element-wise sigmoid gating
- gate_only and before/after attention mid-norm placement
- QK norm + norm-everywhere (parameter-free mid RMSNorm in FFN/experts)
- Mid-norm MoE (norm inside expert after activation, before down-proj)
- Shared experts (always active, non-routed)
- Fused MoE path (SharedFusedMoE) with NormEverywhere support; exact fallback
- Tensor parallelism, pipeline parallelism, LoRA
"""

import os
from collections.abc import Iterable
from itertools import islice
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from torch import nn

try:
    from vllm.attention.layer import Attention
except ImportError:
    from vllm.model_executor.layers.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, get_current_vllm_config, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size

try:
    from vllm.model_executor.layers.fused_moe import SharedFusedMoE

    _FUSED_MOE_RETURNS_FINAL_OUTPUT = False
except ImportError:
    from vllm.model_executor.layers.fused_moe import FusedMoE as _VllmFusedMoE

    class SharedFusedMoE(_VllmFusedMoE):
        """Compatibility wrapper for vLLM versions that folded SharedFusedMoE into FusedMoE."""

        def __init__(self, *args, reduce_results: bool | None = None, **kwargs):
            super().__init__(*args, **kwargs)

    _FUSED_MOE_RETURNS_FINAL_OUTPUT = True
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    PPMissingLayer,
)
from vllm.sequence import IntermediateTensors

from .norm_everywhere_fused_moe import NormEverywhereSharedFusedMoE


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _weightless_rms_norm(hidden_size: int, eps: float) -> RMSNorm:
    return RMSNorm(hidden_size, eps=eps, has_weight=False)


def _compute_residual_scales(residual_scale: str, n_layers: int) -> tuple[float, float]:
    """Return (block_scale, identity_scale) matching native setup_residual_scale."""
    if residual_scale == "depth_scale":
        total_depth = 2 * n_layers
        return 1.0 / total_depth, (total_depth - 1) / total_depth
    return 1.0, 1.0  # "identity"


def _get_attention_norm_everywhere(config: Any) -> bool:
    return bool(
        getattr(
            config,
            "attention_norm_everywhere",
            getattr(config, "norm_everywhere", False),
        )
    )


def _get_ffn_norm_everywhere(config: Any) -> bool:
    return bool(
        getattr(
            config,
            "ffn_norm_everywhere",
            getattr(config, "norm_everywhere", False),
        )
    )


def _get_moe_norm_everywhere(config: Any) -> bool:
    return bool(
        getattr(
            config,
            "moe_norm_everywhere",
            getattr(
                config,
                "ffn_norm_everywhere",
                getattr(config, "norm_everywhere", False),
            ),
        )
    )


def _get_partial_rotary_factor(config: Any) -> float | None:
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    if partial_rotary_factor is not None:
        return float(partial_rotary_factor)

    qk_rope_dim = getattr(config, "qk_rope_dim", None)
    head_dim = getattr(config, "head_dim", None)
    if qk_rope_dim is None or head_dim is None:
        return None
    return float(qk_rope_dim) / float(head_dim)


def _get_rope_parameters(config: Any) -> dict[str, Any]:
    """Parse RoPE config from HF config into vLLM get_rope() kwargs."""
    rope_parameters: dict[str, Any] = {}
    raw_rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(raw_rope_parameters, dict):
        if raw_rope_parameters and all(
            isinstance(v, dict) for v in raw_rope_parameters.values()
        ):
            if "" in raw_rope_parameters:
                rope_parameters.update(dict(raw_rope_parameters[""]))
            elif len(raw_rope_parameters) == 1:
                rope_parameters.update(dict(next(iter(raw_rope_parameters.values()))))
        else:
            rope_parameters.update(dict(raw_rope_parameters))

    if not rope_parameters:
        rope_parameters = {"rope_type": "default"}

    if "type" in rope_parameters and "rope_type" not in rope_parameters:
        rope_parameters["rope_type"] = rope_parameters["type"]
    if "rope_type" not in rope_parameters:
        rope_parameters["rope_type"] = "default"

    rope_type = rope_parameters.get("rope_type", "default")
    if (
        rope_type == "yarn"
        and "attention_factor" in rope_parameters
        and "attn_factor" not in rope_parameters
    ):
        rope_parameters["attn_factor"] = rope_parameters["attention_factor"]
        rope_parameters.setdefault("apply_yarn_scaling", False)
    if "attn_factor" in rope_parameters and "attention_factor" not in rope_parameters:
        rope_parameters["attention_factor"] = rope_parameters["attn_factor"]

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None and "rope_theta" not in rope_parameters:
        rope_parameters["rope_theta"] = rope_theta

    partial_rotary_factor = _get_partial_rotary_factor(config)
    if (
        partial_rotary_factor is not None
        and "partial_rotary_factor" not in rope_parameters
    ):
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

    return rope_parameters


def _get_swa_rope_parameters(config: Any) -> dict[str, Any]:
    """Parse SWA-specific RoPE parameters (rope_theta_swa + rope_parameters_swa)."""
    rope_parameters: dict[str, Any] = {"rope_type": "default"}

    rope_parameters_swa = getattr(config, "rope_parameters_swa", None)
    if isinstance(rope_parameters_swa, dict):
        rope_parameters.update(dict(rope_parameters_swa))
        if "type" in rope_parameters and "rope_type" not in rope_parameters:
            rope_parameters["rope_type"] = rope_parameters["type"]

    rope_theta_swa = getattr(config, "rope_theta_swa", None)
    if rope_theta_swa is not None:
        rope_parameters["rope_theta"] = float(rope_theta_swa)

    partial_rotary_factor = _get_partial_rotary_factor(config)
    if (
        partial_rotary_factor is not None
        and "partial_rotary_factor" not in rope_parameters
    ):
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

    return rope_parameters


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_gated_attention_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if normalized == "head-wise":
            return "head-wise"
        if normalized == "element-wise":
            return "element-wise"
    return cast(str | None, value)


def _normalize_mid_norm_position(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"after", "before"}:
        raise ValueError(
            f"mid_norm_position must be either 'after' or 'before', got {value!r}"
        )
    return normalized


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def _parse_pattern(
    pattern: "str | list | None",
    n_layers: int,
    true_char: str,
    false_char: str,
    default: bool = True,
) -> "list[bool]":
    """Convert a layer pattern string ("RRRN", "SSSF") to a list of booleans."""
    if pattern is None:
        return [default] * n_layers
    if isinstance(pattern, (list, tuple)):
        # Keep compatibility with native parser:
        #   ['SSSF'] and ['S', 'S', 'S', 'F'] are valid string forms.
        if len(pattern) == 1 and isinstance(pattern[0], str):
            pattern = pattern[0]
        elif pattern and all(isinstance(v, str) and len(v) == 1 for v in pattern):
            pattern = "".join(pattern)
        else:
            if len(pattern) != n_layers:
                raise ValueError(
                    f"Pattern list length {len(pattern)} != n_layers {n_layers}"
                )
            if not all(isinstance(v, bool) for v in pattern):
                raise ValueError(
                    "Pattern list must be list[bool], ['PATTERN'], or list of single-character strings."
                )
            return list(pattern)
    if len(pattern) != n_layers:
        raise ValueError(f"Pattern string length {len(pattern)} != n_layers {n_layers}")
    result = []
    for c in pattern:
        if c.upper() == true_char.upper():
            result.append(True)
        elif c.upper() == false_char.upper():
            result.append(False)
        else:
            raise ValueError(
                f"Unknown character '{c}' in pattern '{pattern}'. "
                f"Expected '{true_char}' or '{false_char}'."
            )
    return result


# ---------------------------------------------------------------------------
# Dense FFN (used for dense layers and shared experts)
# ---------------------------------------------------------------------------


class OptMoEMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        rms_norm_eps: float,
        norm_everywhere: bool,
        bias: bool,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )

        if norm_everywhere:
            self.mid_norm = _weightless_rms_norm(intermediate_size, eps=rms_norm_eps)
            self._mlp_forward = self._forward_with_mid_norm
        else:
            self.mid_norm = nn.Identity()
            self._mlp_forward = self._forward_no_norm

    def _forward_no_norm(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden, _ = self.down_proj(hidden)
        return hidden

    def _forward_with_mid_norm(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden = self.mid_norm(hidden)
        hidden, _ = self.down_proj(hidden)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp_forward(x)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class OptMoEAttention(nn.Module):
    """Multi-head attention supporting NoPE, RoPE, SWA, and gated attention."""

    _NORM_MODE_NONE = "no_norm"
    _NORM_MODE_QK = "qk_norm"
    _NORM_MODE_O = "o_norm"
    _NORM_MODE_QKO = "qko_norm"
    _NORM_MODE_QKVO = "qkvo_norm"

    def __init__(
        self,
        config: Any,
        use_rope: bool,
        use_swa: bool,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.gate_only = bool(getattr(config, "gate_only", False))
        self.mid_norm_position = _normalize_mid_norm_position(
            getattr(config, "mid_norm_position", "after")
        )
        self.head_wise_mid_norm = bool(getattr(config, "head_wise_mid_norm", False))

        attention_bias = getattr(config, "attention_bias", False)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # --- Norms ---
        use_norm_everywhere = _get_attention_norm_everywhere(config)
        use_qk_norm = bool(getattr(config, "qk_norm", False))
        use_mid_norm = bool(getattr(config, "mid_norm", False))
        if use_norm_everywhere:
            self.norm_mode = self._NORM_MODE_QKVO
        elif use_qk_norm and use_mid_norm:
            self.norm_mode = self._NORM_MODE_QKO
        elif use_qk_norm:
            self.norm_mode = self._NORM_MODE_QK
        elif use_mid_norm:
            self.norm_mode = self._NORM_MODE_O
        else:
            self.norm_mode = self._NORM_MODE_NONE

        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.q_norm: nn.Module = nn.Identity()
        self.k_norm: nn.Module = nn.Identity()
        self.v_norm: nn.Module = nn.Identity()
        self.mid_norm: nn.Module = nn.Identity()

        # Compile-only path can leverage vLLM's fused_qk_norm_rope pass.
        self.enable_qk_norm_rope_fusion = False
        try:
            vllm_config = get_current_vllm_config()
            pass_config = getattr(
                getattr(vllm_config, "compilation_config", None),
                "pass_config",
                None,
            )
            self.enable_qk_norm_rope_fusion = bool(
                getattr(pass_config, "enable_qk_norm_rope_fusion", False)
            )
        except Exception:
            self.enable_qk_norm_rope_fusion = False

        if self.norm_mode == self._NORM_MODE_QK:
            self.q_norm = _weightless_rms_norm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = _weightless_rms_norm(self.head_dim, eps=rms_norm_eps)
            self._attn_forward = self._forward_qk_norm
        elif self.norm_mode == self._NORM_MODE_O:
            if not self.gate_only:
                self.mid_norm = self._build_mid_norm(rms_norm_eps)
            self._attn_forward = self._forward_o_norm
        elif self.norm_mode == self._NORM_MODE_QKO:
            self.q_norm = _weightless_rms_norm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = _weightless_rms_norm(self.head_dim, eps=rms_norm_eps)
            if not self.gate_only:
                self.mid_norm = self._build_mid_norm(rms_norm_eps)
            self._attn_forward = self._forward_qko_norm
        elif self.norm_mode == self._NORM_MODE_QKVO:
            self.q_norm = _weightless_rms_norm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = _weightless_rms_norm(self.head_dim, eps=rms_norm_eps)
            self.v_norm = _weightless_rms_norm(self.head_dim, eps=rms_norm_eps)
            if not self.gate_only:
                self.mid_norm = self._build_mid_norm(rms_norm_eps)
            self._attn_forward = self._forward_qkvo_norm
        else:
            self._attn_forward = self._forward_no_norm

        # --- RoPE ---
        # use_rope: whether this layer applies positional encoding at all
        # use_swa_rope: whether this SWA layer has its own independent theta
        self.use_rope = use_rope
        rope_theta_swa = getattr(config, "rope_theta_swa", None)
        self.use_swa_rope = use_swa and use_rope and rope_theta_swa is not None

        if use_rope:
            max_position = getattr(config, "max_position_embeddings", 8192)
            if self.use_swa_rope:
                rope_params = _get_swa_rope_parameters(config)
            else:
                rope_params = _get_rope_parameters(config)
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=max_position,
                rope_parameters=rope_params,
                is_neox_style=True,
            )
        # NoPE layers: no rotary_emb allocated at all

        # --- Sliding window attention ---
        sliding_window_size = int(getattr(config, "sliding_window_size", -1))
        self.sliding_window = sliding_window_size if use_swa else -1

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=self.sliding_window
            if self.sliding_window > 0
            else None,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # --- Gated attention ---
        self.gated_attention_type = _normalize_gated_attention_type(
            getattr(config, "gated_attention_type", None)
        )
        if self.gated_attention_type == "head-wise":
            # One gate scalar per head (per TP rank: num_heads scalars)
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=None,  # gate is non-quantized
                prefix=f"{prefix}.gate_proj",
            )
        elif self.gated_attention_type == "element-wise":
            # One gate value per output element
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.gate_proj",
            )
        # else: no gate_proj

    # ------------------------------------------------------------------
    # Internal attention forward variants
    # ------------------------------------------------------------------

    def _apply_mid_norm_before(self, attn_output: torch.Tensor) -> torch.Tensor:
        tokens = attn_output.shape[0]
        attn_output = attn_output.view(tokens, self.num_heads, self.head_dim)
        return self.mid_norm(attn_output).view(tokens, -1)

    def _apply_mid_norm_after(self, attn_output: torch.Tensor) -> torch.Tensor:
        if self.head_wise_mid_norm:
            tokens = attn_output.shape[0]
            attn_output = attn_output.view(tokens, self.num_heads, self.head_dim)
            return self.mid_norm(attn_output).view(tokens, -1)
        return self.mid_norm(attn_output)

    def _build_mid_norm(self, rms_norm_eps: float) -> nn.Module:
        if self.mid_norm_position == "after" and not self.head_wise_mid_norm:
            return _weightless_rms_norm(self.q_size, eps=rms_norm_eps)
        return _weightless_rms_norm(self.head_dim, eps=rms_norm_eps)

    def _forward_no_norm(
        self,
        positions: torch.Tensor,
        qkv: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_qk_norm(
        self,
        positions: torch.Tensor,
        qkv: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Per-head QK norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_o_norm(
        self,
        positions: torch.Tensor,
        qkv: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if not self.gate_only:
            if self.mid_norm_position == "before":
                attn_output = self._apply_mid_norm_before(attn_output)
            else:
                attn_output = self._apply_mid_norm_after(attn_output)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_qko_norm(
        self,
        positions: torch.Tensor,
        qkv: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if not self.gate_only:
            if self.mid_norm_position == "before":
                attn_output = self._apply_mid_norm_before(attn_output)
            else:
                attn_output = self._apply_mid_norm_after(attn_output)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_qkvo_norm(
        self,
        positions: torch.Tensor,
        qkv: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        v_by_head = v.view(*v.shape[:-1], v.shape[-1] // self.head_dim, self.head_dim)
        v = self.v_norm(v_by_head).view(v.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if not self.gate_only:
            if self.mid_norm_position == "before":
                attn_output = self._apply_mid_norm_before(attn_output)
            else:
                attn_output = self._apply_mid_norm_after(attn_output)
        output, _ = self.o_proj(attn_output)
        return output

    def _apply_gate(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply head-wise or element-wise sigmoid gating."""
        gate_logits, _ = self.gate_proj(hidden_states)
        orig_dtype = attn_output.dtype
        gate = torch.sigmoid(gate_logits.float())
        if self.gated_attention_type == "head-wise":
            # gate: [tokens, num_heads_local] → broadcast over head_dim
            tokens = attn_output.shape[0]
            attn_output = attn_output.view(tokens, self.num_heads, self.head_dim)
            attn_output = (attn_output.float() * gate.unsqueeze(-1)).to(orig_dtype)
            attn_output = attn_output.view(tokens, -1)
        else:
            # element-wise: gate has same shape as attn_output
            attn_output = (attn_output.float() * gate).to(orig_dtype)
        return attn_output

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        # Apply the selected norm+rope+attn path
        if self.gated_attention_type is not None:
            # We need hidden_states for the gate projection, so we must
            # intercept before o_proj. Split out the gate here.
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Apply norms
            if self.norm_mode in (
                self._NORM_MODE_QK,
                self._NORM_MODE_QKO,
                self._NORM_MODE_QKVO,
            ):
                q_by_head = q.view(
                    *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
                )
                q = self.q_norm(q_by_head).view(q.shape)
                k_by_head = k.view(
                    *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
                )
                k = self.k_norm(k_by_head).view(k.shape)
            if self.norm_mode == self._NORM_MODE_QKVO:
                v_by_head = v.view(
                    *v.shape[:-1], v.shape[-1] // self.head_dim, self.head_dim
                )
                v = self.v_norm(v_by_head).view(v.shape)

            # RoPE
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)

            attn_output = self.attn(q, k, v)

            apply_mid_norm = (
                self.norm_mode
                in (
                    self._NORM_MODE_O,
                    self._NORM_MODE_QKO,
                    self._NORM_MODE_QKVO,
                )
                and not self.gate_only
            )
            if apply_mid_norm and self.mid_norm_position == "before":
                attn_output = self._apply_mid_norm_before(attn_output)

            attn_output = self._apply_gate(attn_output, hidden_states)

            if apply_mid_norm and self.mid_norm_position == "after":
                attn_output = self._apply_mid_norm_after(attn_output)

            output, _ = self.o_proj(attn_output)
            return output

        return self._attn_forward(positions, qkv)


# ---------------------------------------------------------------------------
# MoE Router
# ---------------------------------------------------------------------------


class OptMoERouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits, _ = self.gate(hidden_states)
        return logits


# ---------------------------------------------------------------------------
# MoE layer
# ---------------------------------------------------------------------------


class OptMoEMoE(nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts = config.n_total_experts
        self.top_k = config.n_active_experts
        self.route_scale = float(config.moe_scaling_factor)
        self.use_fused_moe = False
        self.use_fused_topk_bias = not _env_bool(
            "VLLM_OPT_MOE_DISABLE_FUSED_TOPK_BIAS", False
        )
        disable_fused_moe = _env_bool("VLLM_OPT_MOE_DISABLE_FUSED", False)
        use_norm_everywhere = _get_moe_norm_everywhere(config)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.router = OptMoERouter(
            hidden_size=config.hidden_size,
            num_experts=self.num_experts,
            prefix=f"{prefix}.router",
        )

        self.expert_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        shared_intermediate_size = int(config.moe_intermediate_size) * max(
            1, int(getattr(config, "n_shared_experts", 1))
        )

        self.shared_experts = OptMoEMLP(
            hidden_size=config.hidden_size,
            intermediate_size=shared_intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=rms_norm_eps,
            norm_everywhere=use_norm_everywhere,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

        def build_exact_experts() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    OptMoEMLP(
                        hidden_size=config.hidden_size,
                        intermediate_size=config.moe_intermediate_size,
                        hidden_act=config.hidden_act,
                        rms_norm_eps=rms_norm_eps,
                        norm_everywhere=use_norm_everywhere,
                        bias=False,
                        quant_config=quant_config,
                        reduce_results=False,
                        prefix=f"{prefix}.experts.{expert_idx}",
                    )
                    for expert_idx in range(self.num_experts)
                ]
            )

        def build_base_fused_experts() -> SharedFusedMoE:
            return SharedFusedMoE(
                shared_experts=self.shared_experts,
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                reduce_results=False,
                renormalize=True,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                scoring_func="sigmoid",
                routed_scaling_factor=self.route_scale,
                e_score_correction_bias=self.expert_bias,
                enable_eplb=enable_eplb,
                num_redundant_experts=num_redundant_experts,
            )

        self.experts: SharedFusedMoE | nn.ModuleList
        if disable_fused_moe:
            self.experts = build_exact_experts()
        elif not use_norm_everywhere:
            self.experts = build_base_fused_experts()
            self.use_fused_moe = True
        elif torch.cuda.is_available():
            try:
                maybe_norm_fused = NormEverywhereSharedFusedMoE(
                    shared_experts=self.shared_experts,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    reduce_results=False,
                    renormalize=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.experts",
                    scoring_func="sigmoid",
                    routed_scaling_factor=self.route_scale,
                    e_score_correction_bias=self.expert_bias,
                    enable_eplb=enable_eplb,
                    num_redundant_experts=num_redundant_experts,
                    rms_norm_eps=rms_norm_eps,
                )
                supports_norm_everywhere = bool(
                    getattr(maybe_norm_fused, "supports_norm_everywhere", False)
                )
                if supports_norm_everywhere:
                    self.experts = maybe_norm_fused
                    self.use_fused_moe = True
                else:
                    self.experts = build_exact_experts()
            except Exception:
                self.experts = build_exact_experts()
        else:
            self.experts = build_exact_experts()

    def _route_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(x).to(torch.float32)
        if x.is_cuda and self.use_fused_topk_bias:
            try:
                top_scores, selected_experts = fused_topk_bias(
                    hidden_states=x,
                    gating_output=router_logits,
                    e_score_correction_bias=self.expert_bias.data,
                    topk=self.top_k,
                    renormalize=True,
                    scoring_func="sigmoid",
                )
                if self.route_scale != 1.0:
                    top_scores = top_scores * self.route_scale
                return selected_experts.to(torch.long), top_scores
            except Exception:
                pass

        scores = torch.sigmoid(router_logits)
        _, selected_experts = torch.topk(scores + self.expert_bias, k=self.top_k, dim=1)
        top_scores = scores.gather(dim=1, index=selected_experts)
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return selected_experts, top_scores

    @torch.no_grad()
    def _exact_moe_infer_compile_safe(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        experts = cast(nn.ModuleList, self.experts)
        for expert_idx in range(self.num_experts):
            mask = selected_experts.eq(expert_idx)
            token_weights = (mask.to(top_scores.dtype) * top_scores).sum(dim=1)
            expert_outputs = experts[expert_idx](x).to(output.dtype)
            output = output + expert_outputs * token_weights.unsqueeze(-1).to(
                output.dtype
            )
        return output

    @torch.no_grad()
    def _exact_moe_infer(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        if _is_torch_compiling():
            return self._exact_moe_infer_compile_safe(x, top_scores, selected_experts)

        num_tokens, hidden_dim = x.shape
        top_k = selected_experts.size(1)

        selected_flat = selected_experts.view(-1)
        sort_idx = torch.argsort(selected_flat, stable=True)
        selected_sorted = selected_flat[sort_idx]
        tokens_per_expert = torch.bincount(selected_sorted, minlength=self.num_experts)

        token_indices = sort_idx // top_k
        token_indices_2d = token_indices.unsqueeze(-1).expand(-1, hidden_dim)
        expert_inputs = torch.gather(x, dim=0, index=token_indices_2d)

        expert_outputs = torch.empty_like(expert_inputs)
        experts = cast(nn.ModuleList, self.experts)
        start = 0
        for expert_idx, n_tokens in enumerate(tokens_per_expert.tolist()):
            if n_tokens == 0:
                continue
            end = start + n_tokens
            expert_outputs[start:end] = experts[expert_idx](
                expert_inputs[start:end]
            ).to(expert_inputs.dtype)
            start = end

        gate_flat_sorted = top_scores.view(-1)[sort_idx]
        weighted_outputs = expert_outputs * gate_flat_sorted.unsqueeze(-1)

        output = torch.zeros_like(x)
        output = output.scatter_add(
            dim=0,
            index=token_indices_2d,
            src=weighted_outputs.to(output.dtype),
        )
        assert output.shape[0] == num_tokens
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        x = hidden_states.reshape(-1, hidden_dim)

        if self.use_fused_moe:
            router_logits = self.router(x).to(torch.float32)
            fused_experts = cast(SharedFusedMoE, self.experts)
            fused_output = fused_experts(
                hidden_states=x,
                router_logits=router_logits,
            )

            if _FUSED_MOE_RETURNS_FINAL_OUTPUT:
                return fused_output.reshape(orig_shape)

            shared_output, routed_output = fused_output
            final_hidden_states = routed_output
            if shared_output is not None:
                final_hidden_states = final_hidden_states + shared_output

            if self.tp_size > 1:
                final_hidden_states = (
                    fused_experts.maybe_all_reduce_tensor_model_parallel(
                        final_hidden_states
                    )
                )
            return final_hidden_states.reshape(orig_shape)

        selected_experts, top_scores = self._route_tokens(x)
        routed_output = self._exact_moe_infer(x, top_scores, selected_experts)
        shared_output = self.shared_experts(x)
        final_hidden_states = routed_output + shared_output
        if self.tp_size > 1:
            final_hidden_states = final_hidden_states.reshape(-1, hidden_dim)
            from vllm.distributed import tensor_model_parallel_all_reduce

            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class OptMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        n_layers = config.num_hidden_layers
        base_use_rope = bool(getattr(config, "use_rope", True))
        base_use_swa = int(getattr(config, "sliding_window_size", -1)) > 0

        # Determine per-layer positional encoding and attention type
        use_rope = _parse_pattern(
            getattr(config, "rope_pattern", None),
            n_layers,
            "R",
            "N",
            default=base_use_rope,
        )[self.layer_idx]
        use_swa = _parse_pattern(
            getattr(config, "swa_pattern", None),
            n_layers,
            "S",
            "F",
            default=base_use_swa,
        )[self.layer_idx]

        self.self_attn = OptMoEAttention(
            config=config,
            use_rope=use_rope,
            use_swa=use_swa,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if self.layer_idx < config.n_dense_layers:
            self.mlp = OptMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
                norm_everywhere=_get_ffn_norm_everywhere(config),
                bias=bool(getattr(config, "mlp_bias", False)),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = OptMoEMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                enable_eplb=enable_eplb,
                num_redundant_experts=num_redundant_experts,
            )

        self.input_layernorm = _weightless_rms_norm(
            config.hidden_size, eps=float(getattr(config, "rms_norm_eps", 1e-6))
        )
        self.post_attention_layernorm = _weightless_rms_norm(
            config.hidden_size, eps=float(getattr(config, "rms_norm_eps", 1e-6))
        )

        residual_scale = getattr(config, "residual_scale", "identity")
        self.block_scale, self.identity_scale = _compute_residual_scales(
            residual_scale, config.num_hidden_layers
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm with explicit residual — matches HF reference ordering.
        if residual is None:
            hidden_input = hidden_states
        else:
            hidden_input = hidden_states + residual

        hidden_states = self.input_layernorm(hidden_input)
        attn_output = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_after_attn = (
            self.identity_scale * hidden_input + self.block_scale * attn_output
        )

        mlp_input = self.post_attention_layernorm(hidden_after_attn)
        mlp_output = self.mlp(mlp_input)
        hidden_out = (
            self.identity_scale * hidden_after_attn + self.block_scale * mlp_output
        )

        return hidden_out, None


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class OptMoEModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.config = config
        self.quant_config = quant_config
        self.force_fp32_embed_lm_head = _env_bool(
            "VLLM_OPT_MOE_FP32_EMBED_LM_HEAD", False
        )

        self.enable_eplb = bool(getattr(parallel_config, "enable_eplb", False))
        eplb_config = getattr(parallel_config, "eplb_config", None)
        self.num_redundant_experts = int(
            getattr(eplb_config, "num_redundant_experts", 0)
        )

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OptMoEDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.num_redundant_experts,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = _weightless_rms_norm(
                config.hidden_size,
                eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            if self.force_fp32_embed_lm_head and hidden_states.dtype == torch.float32:
                for param in self.parameters():
                    if param.dtype != torch.float32 and param.is_floating_point():
                        hidden_states = hidden_states.to(param.dtype)
                        break
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        if residual is not None:
            hidden_states = hidden_states + residual
            residual = None

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        has_fused_moe = False
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, OptMoEDecoderLayer):
                continue
            if isinstance(layer.mlp, OptMoEMoE) and layer.mlp.use_fused_moe:
                has_fused_moe = True
                break

        if not has_fused_moe:
            return []

        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_total_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    # Training-only buffers present in HF checkpoints that have no inference
    # counterpart in vLLM (see state_dict_adapter.py non-converted list).
    _SKIP_WEIGHT_PREFIXES = (
        "load_balance_loss",
        "tokens_per_expert",
        "router_entropy",
        "acc_fwd_times",
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # MLP gate/up → gate_up_proj.
            # Note: self_attn.gate_proj (gated-attention) won't match because
            # "self_attn.gate_up_proj" doesn't exist in params_dict → skipped.
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            # Skip RoPE cached buffers
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Skip training-only buffers from HF checkpoints
            if any(skip in name for skip in self._SKIP_WEIGHT_PREFIXES):
                continue

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                if scale_name not in params_dict:
                    continue
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            if "scale" in name:
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is not None:
                    name = remapped

            # Try stacked-param mapping (qkv, gate_up)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                mapped_name = name.replace(weight_name, param_name)
                if (
                    mapped_name.endswith(".bias") or mapped_name.endswith("_bias")
                ) and mapped_name not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped_name, self):
                    continue
                if mapped_name not in params_dict:
                    continue

                param = params_dict[mapped_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                # Try fused expert mapping
                is_expert_weight = False
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    mapped_name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(mapped_name, self):
                        continue
                    if mapped_name not in params_dict:
                        continue

                    param = params_dict[mapped_name]
                    weight_loader = cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = mapped_name
                        break
                else:
                    if is_expert_weight:
                        continue

                    if (name.endswith(".bias") or name.endswith("_bias")) and (
                        name not in params_dict
                    ):
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


# ---------------------------------------------------------------------------
# Top-level causal LM class
# ---------------------------------------------------------------------------


class OptMoEForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.force_fp32_embed_lm_head = _env_bool(
            "VLLM_OPT_MOE_FP32_EMBED_LM_HEAD", False
        )

        self.model = OptMoEModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self.force_fp32_embed_lm_head and hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.to(torch.float32)
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head."]
                if getattr(self.config, "tie_word_embeddings", False)
                else None
            ),
        )
        loaded = loader.load_weights(weights)

        # Validate critical packed-weight paths
        n_dense = getattr(self.config, "n_dense_layers", 0)
        n_total = self.config.num_hidden_layers
        has_moe = n_dense < n_total

        required_markers = ["model.layers.0.self_attn.qkv_proj"]
        if n_dense > 0:
            required_markers.append("model.layers.0.mlp.gate_up_proj")
        if has_moe:
            required_markers.append(
                f"model.layers.{n_dense}.mlp.shared_experts.gate_up_proj"
            )

        missing_markers = [
            m for m in required_markers if not any(m in name for name in loaded)
        ]
        if missing_markers:
            raise RuntimeError(
                "Critical model parameters were not loaded. "
                "Packed-weight mapping may be broken.\n"
                f"Missing markers:\n" + "\n".join(missing_markers)
            )

        expected_params = {name for name, _ in self.named_parameters()}
        missing_params = sorted(expected_params - loaded)
        if missing_params:
            preview = "\n".join(missing_params[:64])
            raise RuntimeError(
                "Some model parameters were left unloaded.\n"
                f"missing_count={len(missing_params)}\n"
                f"first_missing:\n{preview}"
            )

        if self.force_fp32_embed_lm_head:
            if get_pp_group().is_first_rank and hasattr(
                self.model.embed_tokens, "weight"
            ):
                self.model.embed_tokens.weight.data = (
                    self.model.embed_tokens.weight.data.to(torch.float32)
                )
            if get_pp_group().is_last_rank and hasattr(self.lm_head, "weight"):
                self.lm_head.weight.data = self.lm_head.weight.data.to(torch.float32)

        return loaded


__all__ = ["OptMoEForCausalLM"]
