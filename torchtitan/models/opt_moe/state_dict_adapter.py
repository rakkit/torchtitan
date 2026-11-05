# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native opt_moe <-> HF (`hf_assests/modeling_opt_moe.py`) state dicts.

`wqkv` and the dense / shared `w13` are split into HF's projections; routed
experts stay 3D (`w13` `[E, 2, F, D]` <-> `gate_up_proj` `[E, 2F, D]`)."""

import re
from typing import Any

from torchtitan.models.base.state_dict_adapter import StateDictAdapter

# Native key (fused projections split) -> HF key; `{}` is the layer index.
_TO_HF = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "lm_head.weight": "lm_head.weight",
    "layers.{}.attention.qkv_linear.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.qkv_linear.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.qkv_linear.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention.gate_proj.weight": "model.layers.{}.self_attn.gate_proj.weight",
    "layers.{}.attention.q_conv.weight": "model.layers.{}.self_attn.q_conv.weight",
    "layers.{}.attention.k_conv.weight": "model.layers.{}.self_attn.k_conv.weight",
    "layers.{}.attention.v_conv.weight": "model.layers.{}.self_attn.v_conv.weight",
    "layers.{}.attention.sinks": "model.layers.{}.self_attn.sinks",
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "layers.{}.moe.router.gate.weight": "model.layers.{}.mlp.gate.weight",
    "layers.{}.moe.expert_bias_E": "model.layers.{}.mlp.gate.e_score_correction_bias",
    "layers.{}.moe.routed_experts.w13.weight": "model.layers.{}.mlp.experts.gate_up_proj",
    "layers.{}.moe.routed_experts.w2.weight": "model.layers.{}.mlp.experts.down_proj",
    "layers.{}.moe.shared_experts.w1.weight": "model.layers.{}.mlp.shared_experts.gate_proj.weight",
    "layers.{}.moe.shared_experts.w3.weight": "model.layers.{}.mlp.shared_experts.up_proj.weight",
    "layers.{}.moe.shared_experts.w2.weight": "model.layers.{}.mlp.shared_experts.down_proj.weight",
}
_FROM_HF = {v: k for k, v in _TO_HF.items()}
_ROUTED_W13 = "moe.routed_experts.w13.weight"


def _rename(state_dict: dict[str, Any], table: dict[str, str]) -> dict[str, Any]:
    out, unmapped = {}, []
    for key, value in state_dict.items():
        m = re.match(r"(model\.)?layers\.(\d+)\.", key)
        abstract = key.replace(f"layers.{m.group(2)}.", "layers.{}.", 1) if m else key
        target = table.get(abstract)
        if target is None:
            unmapped.append(key)
            continue
        out[target.format(m.group(2)) if m else target] = value
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} tensor(s) have no opt_moe <-> HF mapping: "
            f"{sorted(unmapped)[:8]}"
        )
    return out


class OPTMoEStateDictAdapter(StateDictAdapter):
    # HF tensors the model computes with in fp32 whatever the export dtype.
    keep_in_fp32 = ("e_score_correction_bias",)

    def _qkv_args(self) -> dict[str, int]:
        attn = self.model_config.layer.attention
        n_kv = attn.n_kv_heads or attn.n_heads
        return {
            "head_dim": attn.head_dim or attn.dim // attn.n_heads,
            "heads_per_kv": attn.n_heads // n_kv,
        }

    @staticmethod
    def _prefixes(keys, suffix: str) -> list[str]:
        return sorted({k[: -len(suffix)] for k in keys if k.endswith(suffix)})

    def _unfuse(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """The fused projections as the tensors the HF model holds."""
        sd = dict(state_dict)
        for prefix in self._prefixes(sd, "wqkv.weight"):
            self._split_qkv_linear(sd, prefix=prefix, **self._qkv_args())
        for key in [k for k in sd if k.endswith(_ROUTED_W13)]:
            sd[key] = sd[key].flatten(1, 2)
        for prefix in self._prefixes(sd, "w13.weight"):
            if not prefix.endswith("routed_experts."):
                self._split_stacked_linear(
                    sd,
                    fused_key=f"{prefix}w13.weight",
                    logical_keys=(f"{prefix}w1.weight", f"{prefix}w3.weight"),
                    dim=0,
                )
        return sd

    def _fuse(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """The inverse of `_unfuse`."""
        sd = dict(state_dict)
        for prefix in self._prefixes(sd, "wq.weight"):
            self._merge_qkv_linear(sd, prefix=prefix, **self._qkv_args())
        for prefix in self._prefixes(sd, "w1.weight"):
            self._stack_logical_linears(
                sd,
                fused_key=f"{prefix}w13.weight",
                logical_keys=(f"{prefix}w1.weight", f"{prefix}w3.weight"),
                dim=0,
            )
        for key in [k for k in sd if k.endswith(_ROUTED_W13)]:
            sd[key] = sd[key].unflatten(1, (2, -1))
        return sd

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return _rename(self._unfuse(state_dict), _TO_HF)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        return self._fuse(_rename(hf_state_dict, _FROM_HF))
