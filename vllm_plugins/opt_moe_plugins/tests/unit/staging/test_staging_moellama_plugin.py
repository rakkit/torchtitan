# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the staging_moellama vLLM plugin."""

import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from vllm import ModelRegistry
from vllm.config import CompilationMode

from opt_moe_plugins.tests.unit.shared_stubs import (
    DummyEmbedding,
    DummyExpertMLP,
    DummyFusedMoE,
    DummyLayer,
    DummyNormEverywhereFusedMoE,
    DummyPPGroup,
    DummyRaisingNormEverywhereFusedMoE,
    DummyRouter,
)

from opt_moe_plugins.vllm_staging_moellama import (
    model as staging_model_mod,
    register as staging_register,
)


class TestStagingMoEllamaOutOfTree(unittest.TestCase):
    def _hf_config(self, *, norm_everywhere: bool) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            n_shared_experts=1,
            n_active_experts=2,
            n_total_experts=4,
            moe_scaling_factor=1.5,
            n_dense_layers=1,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_act="silu",
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            head_dim=4,
            qk_norm=True,
            norm_everywhere=norm_everywhere,
            rope_scaling={"rope_type": "default"},
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )

    def _vllm_config(self, *, norm_everywhere: bool) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=self._hf_config(norm_everywhere=norm_everywhere)
            ),
            cache_config=None,
            quant_config=None,
            parallel_config=types.SimpleNamespace(
                enable_eplb=False,
                eplb_config=types.SimpleNamespace(num_redundant_experts=0),
            ),
            compilation_config=types.SimpleNamespace(mode=CompilationMode.NONE),
        )

    def test_plugin_registers_architecture(self):
        staging_register()
        self.assertIn(
            "StagingMoEllamaForCausalLM",
            ModelRegistry.get_supported_archs(),
        )

    def test_model_init_accepts_prefix_keyword_factory(self):
        def fake_make_layers(num_hidden_layers, layer_fn, prefix):
            layer = layer_fn(prefix=f"{prefix}.0")
            return 0, 1, nn.ModuleList([layer])

        with (
            patch.object(
                staging_model_mod, "make_layers", side_effect=fake_make_layers
            ),
            patch.object(staging_model_mod, "MoEllamaDecoderLayer", DummyLayer),
            patch.object(
                staging_model_mod, "get_pp_group", return_value=DummyPPGroup()
            ),
            patch.object(staging_model_mod, "VocabParallelEmbedding", DummyEmbedding),
            patch.object(
                staging_model_mod,
                "_weightless_rms_norm",
                side_effect=lambda *args, **kwargs: nn.Identity(),
            ),
        ):
            model = staging_model_mod.StagingMoEllamaModel(
                vllm_config=self._vllm_config(norm_everywhere=True),
                prefix="model",
            )
        self.assertEqual(model.start_layer, 0)
        self.assertEqual(model.end_layer, 1)

    def test_moe_exact_path_with_norm_everywhere(self):
        hf_config = self._hf_config(norm_everywhere=True)
        with (
            patch.object(
                staging_model_mod,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "opt_moe_plugins.vllm_staging_moellama.model.torch.cuda.is_available",
                return_value=False,
            ),
            patch.object(staging_model_mod, "MoEllamaRouter", DummyRouter),
            patch.object(staging_model_mod, "MoEllamaMLP", DummyExpertMLP),
        ):
            moe = staging_model_mod.MoEllamaMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        self.assertFalse(moe.use_fused_moe)
        self.assertIsInstance(moe.experts, nn.ModuleList)
        self.assertEqual(len(moe.experts), hf_config.n_total_experts)
        self.assertTrue(
            all(not isinstance(expert.mid_norm, nn.Identity) for expert in moe.experts)
        )

        x = torch.randn(7, hf_config.hidden_size)
        y = moe(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_moe_fused_path_without_norm_everywhere(self):
        hf_config = self._hf_config(norm_everywhere=False)

        with (
            patch.dict("os.environ", {"VLLM_STAGING_MOE_DISABLE_FUSED": "0"}),
            patch.object(
                staging_model_mod,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "opt_moe_plugins.vllm_staging_moellama.model.torch.cuda.is_available",
                return_value=True,
            ),
            patch.object(staging_model_mod, "MoEllamaRouter", DummyRouter),
            patch.object(staging_model_mod, "MoEllamaMLP", DummyExpertMLP),
            patch.object(staging_model_mod, "SharedFusedMoE", DummyFusedMoE),
        ):
            moe = staging_model_mod.MoEllamaMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        self.assertTrue(moe.use_fused_moe)
        self.assertIsInstance(moe.experts, DummyFusedMoE)

    def test_moe_fused_path_with_norm_everywhere_on_cuda(self):
        hf_config = self._hf_config(norm_everywhere=True)

        with (
            patch.dict("os.environ", {"VLLM_STAGING_MOE_DISABLE_FUSED": "0"}),
            patch.object(
                staging_model_mod,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "opt_moe_plugins.vllm_staging_moellama.model.torch.cuda.is_available",
                return_value=True,
            ),
            patch.object(staging_model_mod, "MoEllamaRouter", DummyRouter),
            patch.object(staging_model_mod, "MoEllamaMLP", DummyExpertMLP),
            patch.object(
                staging_model_mod,
                "NormEverywhereSharedFusedMoE",
                DummyNormEverywhereFusedMoE,
            ),
        ):
            moe = staging_model_mod.MoEllamaMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        self.assertTrue(moe.use_fused_moe)
        self.assertIsInstance(moe.experts, DummyNormEverywhereFusedMoE)

    def test_moe_norm_everywhere_fused_init_failure_falls_back_to_exact(self):
        hf_config = self._hf_config(norm_everywhere=True)

        with (
            patch.dict("os.environ", {"VLLM_STAGING_MOE_DISABLE_FUSED": "0"}),
            patch.object(
                staging_model_mod,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "opt_moe_plugins.vllm_staging_moellama.model.torch.cuda.is_available",
                return_value=True,
            ),
            patch.object(staging_model_mod, "MoEllamaRouter", DummyRouter),
            patch.object(staging_model_mod, "MoEllamaMLP", DummyExpertMLP),
            patch.object(
                staging_model_mod,
                "NormEverywhereSharedFusedMoE",
                DummyRaisingNormEverywhereFusedMoE,
            ),
        ):
            moe = staging_model_mod.MoEllamaMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        self.assertFalse(moe.use_fused_moe)
        self.assertIsInstance(moe.experts, nn.ModuleList)
        self.assertEqual(len(moe.experts), hf_config.n_total_experts)

    def test_moe_exact_path_accepts_noncontiguous_hidden_states(self):
        hf_config = self._hf_config(norm_everywhere=True)
        with (
            patch.object(
                staging_model_mod,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "opt_moe_plugins.vllm_staging_moellama.model.torch.cuda.is_available",
                return_value=False,
            ),
            patch.object(staging_model_mod, "MoEllamaRouter", DummyRouter),
            patch.object(staging_model_mod, "MoEllamaMLP", DummyExpertMLP),
        ):
            moe = staging_model_mod.MoEllamaMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        x = torch.randn(2, 4, hf_config.hidden_size).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        y = moe(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_moe_exact_path_compile_safe_branch(self):
        hf_config = self._hf_config(norm_everywhere=True)
        with (
            patch.object(
                staging_model_mod,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "opt_moe_plugins.vllm_staging_moellama.model.torch.cuda.is_available",
                return_value=False,
            ),
            patch.object(staging_model_mod, "_is_torch_compiling", return_value=True),
            patch.object(staging_model_mod, "MoEllamaRouter", DummyRouter),
            patch.object(staging_model_mod, "MoEllamaMLP", DummyExpertMLP),
        ):
            moe = staging_model_mod.MoEllamaMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        x = torch.randn(9, hf_config.hidden_size)
        y = moe(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main(verbosity=2)
