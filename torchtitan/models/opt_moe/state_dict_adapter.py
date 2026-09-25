# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter

from torchtitan.protocols.model import BaseModel

logger = logging.getLogger(__name__)


class OPTMoEStateDictAdapter(MoEStateDictAdapter):
    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        # self.model_config and self.hf_assets_path already set by MoEStateDictAdapter

        # NOTE: these native FQNs must track the model. The fused projections
        # (`wqkv`, `w13`) are mapped through their logical names: `to_hf`
        # splits them first (`_unfuse`), `from_hf` packs them last (`_fuse`).
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.gate_proj.weight": "layers.{}.attention.gate_proj.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.routed_experts.w1_EFD",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.routed_experts.w3_EFD",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.routed_experts.w2.weight",
            "model.layers.{}.mlp.router.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.expert_bias": "layers.{}.moe.expert_bias_E",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
        }

    @staticmethod
    def _is_unsupported_norm_key(key: str) -> bool:
        return bool(
            re.match(
                r"layers\.\d+\.attention\.(q_norm|k_norm|v_norm|mid_norm)\.",
                key,
            )
        )

    @staticmethod
    def _prefixes(keys, suffix: str) -> list[str]:
        return sorted({k[: -len(suffix)] for k in keys if k.endswith(suffix)})

    def _qkv_args(self) -> dict[str, int]:
        attn = self.model_config.layer.attention
        n_kv = attn.n_kv_heads or attn.n_heads
        return {
            "head_dim": attn.head_dim or attn.dim // attn.n_heads,
            "heads_per_kv": attn.n_heads // n_kv,
        }

    def _unfuse(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """The fused projections as the logical weights the HF map names."""
        sd = dict(state_dict)
        for prefix in self._prefixes(sd, "wqkv.weight"):
            self._split_qkv_linear(sd, prefix=prefix, **self._qkv_args())
        for prefix in self._prefixes(sd, "w13.weight"):
            experts = prefix.endswith("routed_experts.")
            self._split_stacked_linear(
                sd,
                fused_key=f"{prefix}w13.weight",
                logical_keys=(
                    (f"{prefix}w1_EFD", f"{prefix}w3_EFD")
                    if experts
                    else (f"{prefix}w1.weight", f"{prefix}w3.weight")
                ),
                dim=1 if experts else 0,
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
        for prefix in self._prefixes(sd, "w1_EFD"):
            self._stack_logical_linears(
                sd,
                fused_key=f"{prefix}w13.weight",
                logical_keys=(f"{prefix}w1_EFD", f"{prefix}w3_EFD"),
                dim=1,
            )
        return sd

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = self._unfuse(state_dict)
        # Both native (apply_rotary_emb_cos_sin / rotate_half) and HF use the
        # "consecutive halves" RoPE convention, so wq/wk weights are copied verbatim.
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}
        # Collected, then fatal. A tensor with no mapping used to be dropped
        # behind a warning while the export reported success -- that is how the
        # 0.5.0 rename shipped a checkpoint with 10 of 21 tensors. Deliberate
        # skips are explicit elsewhere (`tokens_per_expert` by name below,
        # learned norms via `_is_unsupported_norm_key`), so reaching here means
        # the map and the model have genuinely diverged.
        unmapped: list[str] = []

        for key, value in state_dict.items():
            if "load_balance_loss" in key:
                continue
            if "tokens_per_expert" in key:
                continue
            if "router_entropy" in key:
                continue
            if "acc_fwd_times" in key:
                continue

            if "layers" in key:
                if self._is_unsupported_norm_key(key):
                    raise ValueError(
                        "HF opt_moe export does not support learned norm tensors for "
                        f"'{key}'. The HF template currently mirrors the parameter-free "
                        "native norm path, so exporting this checkpoint would silently "
                        "drop weights."
                    )
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)

                if abstract_key not in to_hf_map:
                    unmapped.append(key)
                    continue

                new_key = to_hf_map[abstract_key]

                if "moe.routed_experts." in key:
                    # Store the GroupedExperts Weight metadata for from_hf()
                    if isinstance(value, DTensor):
                        self.grouped_expert_weight_placements[
                            abstract_key
                        ] = value.placements
                        self.grouped_expert_weight_shape[abstract_key] = value.shape

                        # Split GroupedExperts weight to local individual expert weights
                        local_expert_fqn = self._get_local_experts_weights(
                            new_key,
                            abstract_key,
                            layer_num,
                            value,
                        )
                        hf_state_dict.update(local_expert_fqn)

                    else:
                        # keep this path for offline conversion
                        split_values = self._split_experts_weights(
                            value, self.model_config.layer.moe.num_experts
                        )

                        for expert_num in range(
                            self.model_config.layer.moe.num_experts
                        ):
                            expert_new_key = new_key.format(layer_num, expert_num)
                            hf_state_dict[expert_new_key] = split_values[
                                expert_num
                            ].squeeze(0)
                else:
                    if new_key is None:
                        continue
                    new_key = new_key.format(layer_num)
                    hf_state_dict[new_key] = value
            else:
                if key not in to_hf_map:
                    unmapped.append(key)
                    continue
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        if unmapped:
            raise ValueError(
                f"{len(unmapped)} tensor(s) have no HF mapping, so exporting "
                "would silently produce a short checkpoint: "
                f"{sorted(unmapped)[:10]}"
                f"{' ...' if len(unmapped) > 10 else ''}. Add them to "
                "`from_hf_map`, or to `_is_unsupported_norm_key` if they are "
                "deliberately not exportable."
            )
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}

        num_experts = (
            self.model_config.layer.moe.num_experts
            if self.model_config.layer.moe is not None
            else 0
        )

        for key, value in hf_state_dict.items():
            if "layers" in key:
                # --- MoE experts (two indices: layer + expert) ---
                if ".mlp.experts." in key:
                    abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                    nums = re.findall(r"\d+", key)
                    if len(nums) < 2:
                        logger.warning(
                            "Found MoE expert key without expert index: %s", key
                        )
                        continue

                    layer_num, expert_num = nums[0], nums[1]
                    new_key_template = self.from_hf_map.get(abstract_key, None)
                    if new_key_template is None:
                        continue

                    if layer_num not in expert_weights_by_layer:
                        expert_weights_by_layer[layer_num] = {}
                    if new_key_template not in expert_weights_by_layer[layer_num]:
                        expert_weights_by_layer[layer_num][new_key_template] = {}
                    expert_weights_by_layer[layer_num][new_key_template][
                        int(expert_num)
                    ] = value

                    # Online mode: local_experts_indices was populated during to_hf().
                    if new_key_template in self.local_experts_indices:
                        stacked_value = self._concatenate_expert_weights_dtensor(
                            expert_weights_by_layer,
                            new_key_template,
                            layer_num,
                        )
                    else:
                        # Offline conversion path.
                        stacked_value = self._concatenate_expert_weights(
                            expert_weights_by_layer,
                            new_key_template,
                            layer_num,
                            num_experts,
                        )

                    if stacked_value is not None:
                        new_key = new_key_template.format(layer_num)
                        state_dict[new_key] = stacked_value
                    continue

                # --- Non-expert layer parameters (attention, FFN, router, shared_experts, etc.) ---
                nums = re.findall(r"\d+", key)
                if not nums:
                    continue
                layer_num = nums[0]
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                new_key_template = self.from_hf_map.get(abstract_key, None)
                if new_key_template is None:
                    continue
                state_dict[new_key_template.format(layer_num)] = value
            else:
                new_key = self.from_hf_map.get(key, None)
                if new_key is None:
                    continue
                state_dict[new_key] = value

        return self._fuse(state_dict)
