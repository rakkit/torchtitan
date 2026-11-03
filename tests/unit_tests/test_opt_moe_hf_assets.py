# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe's HF export (`models/opt_moe/hf_assests`): the config it resolves,
the tensors it writes against the HF model's own parameters, a load through
`from_pretrained`, and -- on GPU -- a bf16 prefill bitwise equal to native."""

import dataclasses
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.opt_moe import model_registry
from torchtitan.models.opt_moe.gated_norm_swattention import GatedNormSWAttention
from torchtitan.models.opt_moe.hf_assests.modeling_opt_moe import OptMoEForCausalLM
from torchtitan.models.opt_moe.hf_assests.setup_hf import (
    copy_and_overwrite_model_config,
    hf_config,
)
from torchtitan.models.opt_moe.model import OPTMoEModel, OPTMoETransformerBlock
from torchtitan.models.opt_moe.norm_ffn import FeedForward
from torchtitan.models.opt_moe.norm_moe import MoE
from torchtitan.models.opt_moe.state_dict_adapter import OPTMoEStateDictAdapter
from torchtitan.models.opt_moe.utils.inits import setup_residual_scale

T = 256


def _native(*, moe=True, norms=True, gate="head-wise", gate_only=False, conv=0, sinks=False, **model):
    """A 4-layer opt_moe: NoPE layer 2, sliding layers 0, 1, 3 with their own
    theta, partial RoPE, a dense first layer."""
    cfg = OPTMoEModel.Config(
        dim=64,
        n_layers=4,
        vocab_size=128,
        rope_pattern="RRNR",
        swa_pattern="SSFS",
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=1 if moe else 4,
            residual_scale="depth_scale_N",
            attention=GatedNormSWAttention.Config(
                n_heads=4,
                n_kv_heads=2,
                head_dim=16,
                qk_norm=True,
                norm_everywhere=norms,
                gated_attention_type=gate,
                gate_only=gate_only,
                qk_rope_dim=8,
                sliding_window_size=32,
                conv_kernel_size=conv,
                conv_init_fn_type="delay" if conv else None,
                conv_init_gain=1.0 if conv else None,
                conv_streams="qkv" if conv else None,
                attention_sinks=sinks,
            ),
            feed_forward=FeedForward.Config(hidden_dim=96, norm_everywhere=norms),
            moe=(
                MoE.Config(
                    hidden_dim=32,
                    num_experts=8,
                    num_shared_experts=1,
                    top_k=2,
                    scaling_factor=2.5,
                    norm_everywhere=norms,
                )
                if moe
                else None
            ),
        ),
        rope=CosSinRoPE.Config(dim=8, max_context_length=T, theta=10000.0),
        rope_of_swa=CosSinRoPE.Config(dim=8, max_context_length=T, theta=5000.0),
        **model,
    )
    return _expanded(cfg)


def _expanded(cfg):
    cfg.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=T),
        )
    )
    return cfg


def _config_only(cfg):
    return SimpleNamespace(config=cfg)


def _pair(cfg, device, attn_implementation="sdpa"):
    """The native model (seeded init, a nonzero balancing bias) and the HF
    model holding its converted weights."""
    with torch.device("meta"):
        native = cfg.build()
    native.to_empty(device=device)
    torch.manual_seed(0)
    with torch.no_grad():
        native.init_states(buffer_device=device)
        for name, buf in native.named_buffers():
            if name.endswith("expert_bias_E"):
                buf.copy_(torch.randn_like(buf) * 1e-3)
    hf = OptMoEForCausalLM._from_config(
        hf_config(native), attn_implementation=attn_implementation
    ).to(device)
    sd = OPTMoEStateDictAdapter(cfg, None).to_hf(dict(native.state_dict()))
    hf.load_state_dict(sd, strict=True)
    return native.eval(), hf.eval()


class TestHFConfig(unittest.TestCase):
    def test_resolves_what_the_native_model_computes(self):
        cfg = _native()
        c = hf_config(_config_only(cfg))
        self.assertEqual(c.use_rope_layers, [True, True, False, True])
        self.assertEqual(
            c.layer_types,
            ["sliding_attention"] * 2 + ["full_attention", "sliding_attention"],
        )
        self.assertEqual(c.sliding_window, 32)
        self.assertEqual(c.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(c.rope_parameters_sliding["rope_theta"], 5000.0)
        self.assertEqual((c.head_dim, c.rotary_dim), (16, 8))
        self.assertTrue(c.qk_norm and c.v_norm and c.attention_mid_norm)
        self.assertEqual(c.gated_attention, "head-wise")
        self.assertEqual((c.first_k_dense_replace, c.n_shared_experts), (1, 1))
        self.assertEqual((c.routed_scaling_factor, c.moe_mid_norm), (2.5, True))
        block, identity = setup_residual_scale("depth_scale_N", 4)
        self.assertEqual(
            (c.residual_block_scale, c.residual_identity_scale), (block, identity)
        )
        with tempfile.TemporaryDirectory() as d:
            c.save_pretrained(d)
            self.assertEqual(type(c).from_pretrained(d).to_dict(), c.to_dict())

    def test_conv_layers_cache_conv_states(self):
        c = hf_config(_config_only(_native(conv=4, sinks=True)))
        self.assertEqual(c.layer_types, ["hybrid_sliding"] * 2 + ["hybrid", "hybrid_sliding"])
        self.assertEqual((c.conv_kernel_size, c.attention_sinks), (4, True))
        self.assertEqual(c.number_of_conv_states, 3)

    def test_gate_only_has_no_mid_norm(self):
        c = hf_config(_config_only(_native(gate_only=True)))
        self.assertFalse(c.attention_mid_norm)
        self.assertTrue(c.v_norm)

    def test_refuses_what_the_template_cannot_compute(self):
        learned = _native()
        learned.layer.attention.norm_type = "rmsnorm"
        with self.assertRaisesRegex(ValueError, "parameter-free RMSNorm"):
            hf_config(_config_only(learned))
        yarn = _native()
        yarn.rope = dataclasses.replace(yarn.rope, scaling="yarn")
        with self.assertRaisesRegex(ValueError, "unscaled CosSinRoPE"):
            hf_config(_config_only(yarn))


class TestHFStateDict(unittest.TestCase):
    CASES = (
        "tiny-moe",
        "tiny-dense",
        "tiny-conv-sinks",
        "mis-8b-proxy",
        "M1-dense-600M-full_rope-gate",
    )

    def _cfg(self, name):
        if name == "tiny-conv-sinks":
            return _native(conv=4, sinks=True)
        if name.startswith("tiny"):
            return _native(moe=name == "tiny-moe")
        return _expanded(model_registry(name))

    def test_every_exported_tensor_is_an_hf_parameter(self):
        for name in self.CASES:
            with self.subTest(name):
                cfg = self._cfg(name)
                with torch.device("meta"):
                    native = cfg.build()
                    hf = OptMoEForCausalLM(hf_config(native))
                exported = OPTMoEStateDictAdapter(cfg, None).to_hf(native.state_dict())
                want = {k: tuple(v.shape) for k, v in hf.state_dict().items()}
                got = {k: tuple(v.shape) for k, v in exported.items()}
                self.assertEqual(got, want)

    def test_round_trip_is_exact(self):
        for name in ("tiny-moe", "tiny-dense", "tiny-conv-sinks"):
            with self.subTest(name):
                cfg = self._cfg(name)
                with torch.device("meta"):
                    native = cfg.build()
                torch.manual_seed(0)
                sd = {k: torch.randn(v.shape) for k, v in native.state_dict().items()}
                adapter = OPTMoEStateDictAdapter(cfg, None)
                back = adapter.from_hf(adapter.to_hf(sd))
                self.assertEqual(set(back), set(sd))
                for k in sd:
                    self.assertTrue(torch.equal(back[k], sd[k]), k)


class TestHFExport(unittest.TestCase):
    """The convert script's weights, then `from_pretrained` in bf16."""

    def test_export_loads_in_bf16_with_the_bias_in_fp32(self):
        from transformers import AutoModelForCausalLM

        scripts = os.path.join(
            os.path.dirname(__file__), "../../scripts/checkpoint_conversion"
        )
        sys.path.insert(0, os.path.abspath(scripts))
        from convert_to_hf import _export_hf_weights
        from pathlib import Path

        cfg = _native()
        native, _ = _pair(cfg, torch.device("cpu"))
        adapter = OPTMoEStateDictAdapter(cfg, None)
        with tempfile.TemporaryDirectory() as d:
            copy_and_overwrite_model_config(native, None, d)
            _export_hf_weights(dict(native.state_dict()), adapter, torch.bfloat16, Path(d))
            hf, info = AutoModelForCausalLM.from_pretrained(
                d, trust_remote_code=True, dtype=torch.bfloat16, output_loading_info=True
            )
        self.assertEqual(type(hf).__name__, "OptMoEForCausalLM")
        self.assertFalse(any(info.values()), info)
        self.assertEqual(hf.config._experts_implementation, "eager")
        sd = adapter.to_hf(dict(native.state_dict()))
        biases = [k for k in sd if k.endswith(adapter.keep_in_fp32)]
        self.assertEqual(len(biases), cfg.n_layers - 1)
        for k, v in hf.state_dict().items():
            want = torch.float32 if k in biases else torch.bfloat16
            self.assertEqual(v.dtype, want, k)
            self.assertTrue(torch.equal(v, sd[k].to(want)), k)


@unittest.skipUnless(torch.cuda.is_available(), "flex attention and grouped GEMMs need CUDA")
class TestHFForwardParity(unittest.TestCase):
    """bf16 parameters (FSDP's mixed precision: buffers stay fp32) and the
    native compiled flex kernel: every logit bitwise."""

    def test_bf16_prefill_is_bitwise(self):
        cases = {
            "moe, norms, head-wise gate": _native(),
            "moe, no norms, element-wise gate": _native(norms=False, gate="element-wise"),
            "dense, gate only": _native(moe=False, gate_only=True),
            "moe, short conv, sinks": _native(conv=4, sinks=True),
        }
        for label, cfg in cases.items():
            with self.subTest(label):
                native, hf = _pair(cfg, torch.device("cuda"), "torchtitan_flex")
                for p in [*native.parameters(), *hf.parameters()]:
                    p.data = p.data.to(torch.bfloat16)
                torch.manual_seed(1)
                tokens = torch.randint(0, cfg.vocab_size, (T,), device="cuda")
                positions = torch.arange(T, device="cuda")
                with torch.no_grad():
                    want = native(
                        tokens,
                        positions=positions,
                        attention_masks=native.get_attention_masks(positions),
                    )
                    got = hf(input_ids=tokens[None]).logits[0]
                self.assertTrue(
                    torch.equal(got, want),
                    f"max |d| {(got.float() - want.float()).abs().max().item():.3e}",
                )

    def test_decoding_with_the_conv_state_matches_one_forward(self):
        """Prefill, then one token at a time through the cache (conv states and
        KV): the logits of a single forward over the whole sequence."""
        cfg = _native(conv=4, sinks=True)
        _, hf = _pair(cfg, torch.device("cuda"), "eager")
        torch.manual_seed(2)
        tokens = torch.randint(0, cfg.vocab_size, (1, 48), device="cuda")
        with torch.no_grad():
            want = hf(input_ids=tokens).logits
            out = hf(input_ids=tokens[:, :40], use_cache=True)
            got, cache = [out.logits], out.past_key_values
            for t in range(40, 48):
                out = hf(input_ids=tokens[:, t : t + 1], past_key_values=cache, use_cache=True)
                got.append(out.logits)
        torch.testing.assert_close(torch.cat(got, dim=1), want, rtol=1e-4, atol=1e-4)

    def test_transformers_expert_backends_stay_selectable(self):
        cfg = _native()
        native, hf = _pair(cfg, torch.device("cuda"))
        hf.set_experts_implementation("grouped_mm")
        for p in hf.parameters():
            p.data = p.data.to(torch.bfloat16)
        tokens = torch.randint(0, cfg.vocab_size, (1, T), device="cuda")
        with torch.no_grad():
            logits = hf(input_ids=tokens).logits
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
