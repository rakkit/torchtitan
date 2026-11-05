# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility checks between OPTMoE state_dict conversion and vLLM loading expectations."""

import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
TORCHTITAN_SRC = REPO_ROOT / "resources" / "torchtitan"
if str(TORCHTITAN_SRC) not in sys.path:
    sys.path.insert(0, str(TORCHTITAN_SRC))

from torchtitan.models.opt_moe.state_dict_adapter import OPTMoEStateDictAdapter


class TestOptMoEStateDictAdapterCompat(unittest.TestCase):
    def _adapter(self, num_experts: int = 2) -> OPTMoEStateDictAdapter:
        model_cfg = types.SimpleNamespace(
            layer=types.SimpleNamespace(
                moe=types.SimpleNamespace(num_experts=num_experts),
            )
        )
        return OPTMoEStateDictAdapter(model_cfg, hf_assets_path=None)

    def test_to_hf_and_from_hf_roundtrip_for_moe_and_shared_weights(self):
        adapter = self._adapter(num_experts=2)
        native_sd = {
            "tok_embeddings.weight": torch.randn(8, 4),
            "output.weight": torch.randn(8, 4),
            "layers.0.attention.wq.weight": torch.randn(4, 4),
            "layers.0.attention.wk.weight": torch.randn(4, 4),
            "layers.0.attention.wv.weight": torch.randn(4, 4),
            "layers.0.attention.wo.weight": torch.randn(4, 4),
            "layers.0.attention.gate_proj.weight": torch.randn(2, 4),
            "layers.0.feed_forward.w1.weight": torch.randn(6, 4),
            "layers.0.feed_forward.w2.weight": torch.randn(4, 6),
            "layers.0.feed_forward.w3.weight": torch.randn(6, 4),
            "layers.0.attention_norm.weight": torch.randn(4),
            "layers.0.ffn_norm.weight": torch.randn(4),
            "layers.1.moe.router.gate.weight": torch.randn(2, 4),
            "layers.1.moe.expert_bias": torch.randn(2),
            "layers.1.moe.shared_experts.w1.weight": torch.randn(6, 4),
            "layers.1.moe.shared_experts.w2.weight": torch.randn(4, 6),
            "layers.1.moe.shared_experts.w3.weight": torch.randn(6, 4),
            "layers.1.moe.experts.w1": torch.randn(2, 6, 4),
            "layers.1.moe.experts.w2": torch.randn(2, 4, 6),
            "layers.1.moe.experts.w3": torch.randn(2, 6, 4),
            "layers.1.moe.load_balance_loss": torch.tensor([0.0]),
            "layers.1.moe.tokens_per_expert": torch.tensor([1, 2]),
            "layers.1.moe.router_entropy": torch.tensor([0.0]),
            "layers.1.moe.acc_fwd_times": torch.tensor([1]),
        }

        hf_sd = adapter.to_hf(native_sd)

        self.assertIn("model.layers.1.mlp.experts.0.gate_proj.weight", hf_sd)
        self.assertIn("model.layers.1.mlp.experts.1.up_proj.weight", hf_sd)
        self.assertIn("model.layers.1.mlp.shared_experts.down_proj.weight", hf_sd)
        self.assertIn("model.layers.1.mlp.router.gate.weight", hf_sd)
        self.assertIn("model.layers.1.mlp.expert_bias", hf_sd)

        for skipped in (
            "load_balance_loss",
            "tokens_per_expert",
            "router_entropy",
            "acc_fwd_times",
        ):
            self.assertFalse(any(skipped in k for k in hf_sd))

        roundtrip_native = adapter.from_hf(hf_sd)

        self.assertIn("layers.1.moe.experts.w1", roundtrip_native)
        self.assertIn("layers.1.moe.experts.w2", roundtrip_native)
        self.assertIn("layers.1.moe.experts.w3", roundtrip_native)

        self.assertTrue(
            torch.equal(
                roundtrip_native["layers.1.moe.experts.w1"],
                native_sd["layers.1.moe.experts.w1"],
            )
        )
        self.assertTrue(
            torch.equal(
                roundtrip_native["layers.1.moe.shared_experts.w2.weight"],
                native_sd["layers.1.moe.shared_experts.w2.weight"],
            )
        )

    def test_adapter_outputs_match_vllm_packed_module_expectations(self):
        adapter = self._adapter(num_experts=2)
        hf_keys = set(adapter.from_hf_map.keys())

        self.assertIn("model.layers.{}.self_attn.q_proj.weight", hf_keys)
        self.assertIn("model.layers.{}.self_attn.k_proj.weight", hf_keys)
        self.assertIn("model.layers.{}.self_attn.v_proj.weight", hf_keys)
        self.assertIn("model.layers.{}.mlp.gate_proj.weight", hf_keys)
        self.assertIn("model.layers.{}.mlp.up_proj.weight", hf_keys)

        plugin_source = (
            REPO_ROOT / "opt_moe_plugins" / "vllm_opt_moe" / "model.py"
        ).read_text()
        self.assertIn('"qkv_proj": ["q_proj", "k_proj", "v_proj"]', plugin_source)
        self.assertIn('"gate_up_proj": ["gate_proj", "up_proj"]', plugin_source)

    def test_training_only_skip_prefixes_are_covered(self):
        training_only = {
            "load_balance_loss",
            "tokens_per_expert",
            "router_entropy",
            "acc_fwd_times",
        }
        plugin_source = (
            REPO_ROOT / "opt_moe_plugins" / "vllm_opt_moe" / "model.py"
        ).read_text()
        for prefix in training_only:
            self.assertIn(f'"{prefix}"', plugin_source)

    def test_from_hf_partial_experts_skips_incomplete_group(self):
        adapter = self._adapter(num_experts=4)
        partial_hf = {
            "model.layers.1.mlp.experts.0.gate_proj.weight": torch.randn(6, 4),
            "model.layers.1.mlp.experts.1.gate_proj.weight": torch.randn(6, 4),
            "model.layers.1.mlp.router.gate.weight": torch.randn(4, 4),
        }

        native = adapter.from_hf(partial_hf)
        self.assertIn("layers.1.moe.router.gate.weight", native)
        self.assertNotIn("layers.1.moe.experts.w1", native)


if __name__ == "__main__":
    unittest.main(verbosity=2)
