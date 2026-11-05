# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the opt_moe vLLM plugin."""

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
    OptMoEDummyRouter,
)

from opt_moe_plugins.vllm_opt_moe import (
    model as opt_moe_mod,
    register as opt_moe_register,
)


class TestOptMoEPlugin(unittest.TestCase):
    def _hf_config(
        self,
        *,
        norm_everywhere: bool = False,
        qk_norm: bool = False,
        mid_norm: bool = False,
        mid_norm_position: str = "after",
        head_wise_mid_norm: bool = False,
        n_dense_layers: int = 1,
        n_shared_experts: int = 1,
        use_rope: bool = True,
        rope_pattern: "str | None" = None,
        swa_pattern: "str | None" = None,
        rope_theta_swa: "float | None" = None,
        gated_attention_type: "str | None" = None,
    ) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            n_shared_experts=n_shared_experts,
            n_active_experts=2,
            n_total_experts=4,
            moe_scaling_factor=1.5,
            n_dense_layers=n_dense_layers,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_act="silu",
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            head_dim=4,
            qk_norm=qk_norm,
            mid_norm=mid_norm,
            norm_everywhere=norm_everywhere,
            mid_norm_position=mid_norm_position,
            head_wise_mid_norm=head_wise_mid_norm,
            rope_scaling={"rope_type": "default"},
            rope_theta=10000.0,
            rope_theta_swa=rope_theta_swa,
            rope_scaling_swa=None,
            use_rope=use_rope,
            tie_word_embeddings=False,
            rope_pattern=rope_pattern,
            swa_pattern=swa_pattern,
            sliding_window_size=4 if swa_pattern is not None else -1,
            gated_attention_type=gated_attention_type,
        )

    def _vllm_config(self, hf_config) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            model_config=types.SimpleNamespace(hf_config=hf_config),
            cache_config=None,
            quant_config=None,
            parallel_config=types.SimpleNamespace(
                enable_eplb=False,
                eplb_config=types.SimpleNamespace(num_redundant_experts=0),
            ),
            compilation_config=types.SimpleNamespace(mode=CompilationMode.NONE),
        )

    def test_plugin_registers_architecture(self):
        opt_moe_register()
        self.assertIn(
            "OptMoEForCausalLM",
            ModelRegistry.get_supported_archs(),
        )

    def test_parse_pattern_none_returns_defaults(self):
        result = opt_moe_mod._parse_pattern(None, 4, "R", "N", default=True)
        self.assertEqual(result, [True, True, True, True])

    def test_parse_pattern_string_rope(self):
        result = opt_moe_mod._parse_pattern("RRNR", 4, "R", "N", default=True)
        self.assertEqual(result, [True, True, False, True])

    def test_parse_pattern_string_swa(self):
        result = opt_moe_mod._parse_pattern("SSFF", 4, "S", "F", default=False)
        self.assertEqual(result, [True, True, False, False])

    def test_parse_pattern_invalid_char_raises(self):
        with self.assertRaises(ValueError):
            opt_moe_mod._parse_pattern("RRXN", 4, "R", "N")

    def test_parse_pattern_invalid_length_string_raises(self):
        with self.assertRaises(ValueError):
            opt_moe_mod._parse_pattern("RRN", 4, "R", "N")

    def test_parse_pattern_invalid_length_list_raises(self):
        with self.assertRaises(ValueError):
            opt_moe_mod._parse_pattern([True, False], 4, "R", "N")

    def test_opt_moe_exact_path_with_norm_everywhere(self):
        hf_config = self._hf_config(norm_everywhere=True)
        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch(
                "opt_moe_plugins.vllm_opt_moe.model.torch.cuda.is_available",
                return_value=False,
            ),
            patch.object(opt_moe_mod, "OptMoERouter", OptMoEDummyRouter),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyExpertMLP),
        ):
            moe = opt_moe_mod.OptMoEMoE(
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

    def test_opt_moe_shared_expert_uses_n_shared_experts_multiplier(self):
        hf_config = self._hf_config(norm_everywhere=True, n_shared_experts=3)
        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch(
                "opt_moe_plugins.vllm_opt_moe.model.torch.cuda.is_available",
                return_value=False,
            ),
            patch.object(opt_moe_mod, "OptMoERouter", OptMoEDummyRouter),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyExpertMLP),
        ):
            moe = opt_moe_mod.OptMoEMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        self.assertEqual(
            moe.shared_experts.intermediate_size,
            hf_config.moe_intermediate_size * hf_config.n_shared_experts,
        )
        self.assertEqual(
            moe.experts[0].intermediate_size, hf_config.moe_intermediate_size
        )

    def test_opt_moe_fused_path_without_norm_everywhere(self):
        hf_config = self._hf_config(norm_everywhere=False)

        with (
            patch.dict("os.environ", {"VLLM_OPT_MOE_DISABLE_FUSED": "0"}),
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch(
                "opt_moe_plugins.vllm_opt_moe.model.torch.cuda.is_available",
                return_value=True,
            ),
            patch.object(opt_moe_mod, "OptMoERouter", OptMoEDummyRouter),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyExpertMLP),
            patch.object(opt_moe_mod, "SharedFusedMoE", DummyFusedMoE),
        ):
            moe = opt_moe_mod.OptMoEMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        self.assertTrue(moe.use_fused_moe)
        self.assertIsInstance(moe.experts, DummyFusedMoE)

    def test_opt_moe_fused_norm_everywhere_cuda(self):
        hf_config = self._hf_config(norm_everywhere=True)

        with (
            patch.dict("os.environ", {"VLLM_OPT_MOE_DISABLE_FUSED": "0"}),
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch(
                "opt_moe_plugins.vllm_opt_moe.model.torch.cuda.is_available",
                return_value=True,
            ),
            patch.object(opt_moe_mod, "OptMoERouter", OptMoEDummyRouter),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyExpertMLP),
            patch.object(
                opt_moe_mod, "NormEverywhereSharedFusedMoE", DummyNormEverywhereFusedMoE
            ),
        ):
            moe = opt_moe_mod.OptMoEMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        self.assertTrue(moe.use_fused_moe)
        self.assertIsInstance(moe.experts, DummyNormEverywhereFusedMoE)

    def test_opt_moe_fused_norm_everywhere_fallback_on_init_error(self):
        hf_config = self._hf_config(norm_everywhere=True)

        with (
            patch.dict("os.environ", {"VLLM_OPT_MOE_DISABLE_FUSED": "0"}),
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch(
                "opt_moe_plugins.vllm_opt_moe.model.torch.cuda.is_available",
                return_value=True,
            ),
            patch.object(opt_moe_mod, "OptMoERouter", OptMoEDummyRouter),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyExpertMLP),
            patch.object(
                opt_moe_mod,
                "NormEverywhereSharedFusedMoE",
                DummyRaisingNormEverywhereFusedMoE,
            ),
        ):
            moe = opt_moe_mod.OptMoEMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
                enable_eplb=False,
                num_redundant_experts=0,
            )

        self.assertFalse(moe.use_fused_moe)
        self.assertIsInstance(moe.experts, nn.ModuleList)

    def test_opt_moe_exact_path_noncontiguous(self):
        hf_config = self._hf_config(norm_everywhere=True)
        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch(
                "opt_moe_plugins.vllm_opt_moe.model.torch.cuda.is_available",
                return_value=False,
            ),
            patch.object(opt_moe_mod, "OptMoERouter", OptMoEDummyRouter),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyExpertMLP),
        ):
            moe = opt_moe_mod.OptMoEMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
            )
        x = torch.randn(2, 4, hf_config.hidden_size).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        y = moe(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_opt_moe_exact_path_compile_safe(self):
        hf_config = self._hf_config(norm_everywhere=False)
        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch(
                "opt_moe_plugins.vllm_opt_moe.model.torch.cuda.is_available",
                return_value=False,
            ),
            patch.object(opt_moe_mod, "_is_torch_compiling", return_value=True),
            patch.object(opt_moe_mod, "OptMoERouter", OptMoEDummyRouter),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyExpertMLP),
        ):
            moe = opt_moe_mod.OptMoEMoE(
                config=hf_config,
                quant_config=None,
                prefix="model.layers.1.mlp",
            )
        x = torch.randn(9, hf_config.hidden_size)
        y = moe(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_nope_layer_has_no_rotary_emb(self):
        hf_config = self._hf_config(rope_pattern="NNNNN")

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
        ):
            attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=False,
                use_swa=False,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )

        self.assertFalse(attn.use_rope)
        self.assertFalse(hasattr(attn, "rotary_emb"))

    def test_rope_layer_has_rotary_emb(self):
        hf_config = self._hf_config()

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "get_rope", return_value=nn.Identity()),
            patch.object(opt_moe_mod, "Attention", DummyLayer),
        ):
            attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=True,
                use_swa=False,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )

        self.assertTrue(attn.use_rope)
        self.assertFalse(attn.use_swa_rope)
        self.assertTrue(hasattr(attn, "rotary_emb"))

    def test_swa_rope_layer_uses_swa_rope_params(self):
        hf_config = self._hf_config(
            swa_pattern="SSFF",
            rope_theta_swa=500000.0,
        )
        collected_params = {}

        def mock_get_rope(head_dim, max_position, rope_parameters, **kwargs):
            collected_params.update(rope_parameters)
            return nn.Identity()

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "get_rope", side_effect=mock_get_rope),
            patch.object(opt_moe_mod, "Attention", DummyLayer),
        ):
            attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=True,
                use_swa=True,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )

        self.assertTrue(attn.use_swa_rope)
        self.assertIn("rope_theta", collected_params)
        self.assertAlmostEqual(collected_params["rope_theta"], 500000.0)

    def test_sliding_window_set_for_swa_layer(self):
        hf_config = self._hf_config(swa_pattern="SSFF")

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "get_rope", return_value=nn.Identity()),
            patch.object(opt_moe_mod, "Attention", DummyLayer),
        ):
            swa_attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=True,
                use_swa=True,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )
            full_attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=True,
                use_swa=False,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.2.self_attn",
            )

        self.assertGreater(swa_attn.sliding_window, 0)
        self.assertLessEqual(full_attn.sliding_window, 0)

    def test_head_wise_gate_proj_exists(self):
        hf_config = self._hf_config(gated_attention_type="head-wise")

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "get_rope", return_value=nn.Identity()),
            patch.object(opt_moe_mod, "Attention", DummyLayer),
            patch.object(opt_moe_mod, "ColumnParallelLinear", nn.Identity),
        ):
            attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=True,
                use_swa=False,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )

        self.assertEqual(attn.gated_attention_type, "head-wise")
        self.assertTrue(hasattr(attn, "gate_proj"))

    def test_element_wise_gate_proj_exists(self):
        hf_config = self._hf_config(gated_attention_type="element-wise")

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "get_rope", return_value=nn.Identity()),
            patch.object(opt_moe_mod, "Attention", DummyLayer),
            patch.object(opt_moe_mod, "ColumnParallelLinear", nn.Identity),
        ):
            attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=True,
                use_swa=False,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )

        self.assertEqual(attn.gated_attention_type, "element-wise")
        self.assertTrue(hasattr(attn, "gate_proj"))

    def test_no_gate_proj_when_gated_attention_type_is_none(self):
        hf_config = self._hf_config(gated_attention_type=None)

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "get_rope", return_value=nn.Identity()),
            patch.object(opt_moe_mod, "Attention", DummyLayer),
        ):
            attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=True,
                use_swa=False,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )

        self.assertIsNone(attn.gated_attention_type)
        self.assertFalse(hasattr(attn, "gate_proj"))

    def test_string_none_disables_gated_attention(self):
        hf_config = self._hf_config(gated_attention_type="none")

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "get_rope", return_value=nn.Identity()),
            patch.object(opt_moe_mod, "Attention", DummyLayer),
        ):
            attn = opt_moe_mod.OptMoEAttention(
                config=hf_config,
                use_rope=True,
                use_swa=False,
                cache_config=None,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )

        self.assertIsNone(attn.gated_attention_type)
        self.assertFalse(hasattr(attn, "gate_proj"))

    def test_decoder_layer_parses_rope_pattern(self):
        hf_config = self._hf_config(rope_pattern="RRNN")

        created_attrs = {}

        class CapturingAttention(nn.Module):
            def __init__(self, config, use_rope, use_swa, **kwargs):
                super().__init__()
                created_attrs["use_rope_layer0"] = use_rope

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "OptMoEAttention", CapturingAttention),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyLayer),
            patch.object(opt_moe_mod, "extract_layer_index", return_value=2),
            patch.object(
                opt_moe_mod, "_weightless_rms_norm", return_value=nn.Identity()
            ),
        ):
            opt_moe_mod.OptMoEDecoderLayer(
                config=hf_config,
                prefix="model.layers.2",
            )

        self.assertFalse(created_attrs.get("use_rope_layer0"))

    def test_decoder_layer_swa_defaults_to_sliding_window_when_pattern_absent(self):
        hf_config = self._hf_config(swa_pattern=None)
        hf_config.sliding_window_size = 8
        created_attrs = {}

        class CapturingAttention(nn.Module):
            def __init__(self, config, use_rope, use_swa, **kwargs):
                super().__init__()
                created_attrs["use_swa"] = use_swa

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "OptMoEAttention", CapturingAttention),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyLayer),
            patch.object(opt_moe_mod, "extract_layer_index", return_value=1),
            patch.object(
                opt_moe_mod, "_weightless_rms_norm", return_value=nn.Identity()
            ),
        ):
            opt_moe_mod.OptMoEDecoderLayer(
                config=hf_config,
                prefix="model.layers.1",
            )

        self.assertTrue(created_attrs["use_swa"])

    def test_decoder_layer_rope_defaults_to_use_rope_when_pattern_absent(self):
        hf_config = self._hf_config(rope_pattern=None, use_rope=False)
        created_attrs = {}

        class CapturingAttention(nn.Module):
            def __init__(self, config, use_rope, use_swa, **kwargs):
                super().__init__()
                created_attrs["use_rope"] = use_rope

        with (
            patch.object(
                opt_moe_mod, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(opt_moe_mod, "OptMoEAttention", CapturingAttention),
            patch.object(opt_moe_mod, "OptMoEMLP", DummyLayer),
            patch.object(opt_moe_mod, "extract_layer_index", return_value=0),
            patch.object(
                opt_moe_mod, "_weightless_rms_norm", return_value=nn.Identity()
            ),
        ):
            opt_moe_mod.OptMoEDecoderLayer(
                config=hf_config,
                prefix="model.layers.0",
            )

        self.assertFalse(created_attrs["use_rope"])

    def test_opt_moe_model_init_smoke(self):
        hf_config = self._hf_config()

        def fake_make_layers(num_hidden_layers, layer_fn, prefix):
            layer = layer_fn(prefix=f"{prefix}.0")
            return 0, 1, nn.ModuleList([layer])

        with (
            patch.object(opt_moe_mod, "make_layers", side_effect=fake_make_layers),
            patch.object(opt_moe_mod, "OptMoEDecoderLayer", DummyLayer),
            patch.object(opt_moe_mod, "get_pp_group", return_value=DummyPPGroup()),
            patch.object(opt_moe_mod, "VocabParallelEmbedding", DummyEmbedding),
            patch.object(
                opt_moe_mod,
                "_weightless_rms_norm",
                side_effect=lambda *args, **kwargs: nn.Identity(),
            ),
        ):
            model = opt_moe_mod.OptMoEModel(
                vllm_config=self._vllm_config(hf_config),
                prefix="model",
            )

        self.assertEqual(model.start_layer, 0)
        self.assertEqual(model.end_layer, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
