# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.opt_moe.gated_norm_swattention import GatedNormSWAttention
from torchtitan.models.opt_moe.hf_assests.configuration_opt_moe import OptMoEConfig
from torchtitan.models.opt_moe.hf_assests.modeling_opt_moe import OptMoEAttention
from torchtitan.models.opt_moe.hf_assests.setup_hf import overwrite_config
from torchtitan.models.opt_moe.model import OPTMoEModel, OPTMoETransformerBlock
from torchtitan.models.opt_moe.norm_ffn import FeedForward
from torchtitan.models.opt_moe.norm_moe import MoE
from torchtitan.models.opt_moe.state_dict_adapter import OPTMoEStateDictAdapter


def _build_native_model_config(
    *,
    gate_only: bool = False,
    mid_norm_position: str = "after",
    head_wise_mid_norm: bool = False,
    qk_norm: bool = False,
    mid_norm: bool = False,
    norm_everywhere: bool = True,
    qk_rope_dim: int | None = None,
    head_dim: int = 4,
) -> OPTMoEModel.Config:
    dim = 2 * head_dim
    return OPTMoEModel.Config(
        dim=dim,
        n_layers=1,
        vocab_size=32,
        norm_eps=1e-30,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=1,
            norm_eps=1e-30,
            attention=GatedNormSWAttention.Config(
                dim=dim,
                n_heads=2,
                n_kv_heads=2,
                head_dim=head_dim,
                qk_norm=qk_norm,
                mid_norm=mid_norm,
                norm_everywhere=norm_everywhere,
                gated_attention_type="head-wise",
                gate_only=gate_only,
                mid_norm_position=mid_norm_position,
                head_wise_mid_norm=head_wise_mid_norm,
                use_rope=False,
                qk_rope_dim=qk_rope_dim,
                attn_backend="flex",
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=16,
                norm_everywhere=False,
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=qk_rope_dim or head_dim,
            max_context_length=8,
            theta=10000.0,
        ),
    )


def _build_expanded(model_config):
    """Build a model the way the trainer does: expand the config first."""
    from types import SimpleNamespace

    from torchtitan.config import ParallelismConfig, TrainingConfig

    model_config.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=8),
        )
    )
    return model_config.build()


def _build_hf_attention(
    *,
    gate_only: bool = False,
    mid_norm_position: str = "after",
    head_wise_mid_norm: bool = False,
    qk_norm: bool = False,
    mid_norm: bool = False,
    norm_everywhere: bool = True,
    qk_rope_dim: int | None = None,
    head_dim: int = 4,
) -> OptMoEAttention:
    config = OptMoEConfig(
        hidden_size=2 * head_dim,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=8,
        qk_norm=qk_norm,
        mid_norm=mid_norm,
        norm_everywhere=norm_everywhere,
        gated_attention_type="head-wise",
        gate_only=gate_only,
        mid_norm_position=mid_norm_position,
        head_wise_mid_norm=head_wise_mid_norm,
        use_rope=False,
        qk_rope_dim=qk_rope_dim,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    attention = OptMoEAttention(config=config, layer_idx=0)
    attention.use_rope = False
    return attention


def _build_native_moe_model_config() -> OPTMoEModel.Config:
    return OPTMoEModel.Config(
        dim=12,
        n_layers=4,
        vocab_size=64,
        norm_eps=1e-5,
        rope_pattern="RRNR",
        swa_pattern="SSFS",
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=0,
            norm_eps=1e-5,
            residual_scale="depth_scale",
            attention=GatedNormSWAttention.Config(
                n_heads=3,
                n_kv_heads=1,
                head_dim=4,
                qk_norm=True,
                norm_everywhere=True,
                gated_attention_type="head-wise",
                gate_only=True,
                mid_norm_position="before",
                use_rope=True,
                qk_rope_dim=2,
                sliding_window_size=16,
                attn_backend="flex",
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=18,
                norm_everywhere=False,
            ),
            moe=MoE.Config(
                hidden_dim=24,
                num_experts=16,
                num_shared_experts=2,
                top_k=4,
                scaling_factor=1.0,
                norm_everywhere=True,
            ),
        ),
        rope=CosSinRoPE.Config(
            dim=2,
            max_context_length=32,
            theta=10000.0,
        ),
        rope_of_swa=CosSinRoPE.Config(
            dim=2,
            max_context_length=32,
            theta=5000.0,
        ),
    )


class _ConfigOnlyModel:
    __slots__ = ("config",)

    def __init__(self, config):
        self.config = config


def _copy_attention_weights(
    native_attention: GatedNormSWAttention,
    hf_attention: OptMoEAttention,
    *,
    head_dim: int,
) -> None:
    with torch.no_grad():
        q, k, v = (
            hf_attention.q_proj.weight,
            hf_attention.k_proj.weight,
            hf_attention.v_proj.weight,
        )
        # wqkv is [n_kv, heads_per_kv + 2, head_dim, D]: the q heads, k, v.
        packed = native_attention.qkv_linear.wqkv.weight.view(
            k.shape[0] // head_dim, -1, head_dim, q.shape[1]
        )
        q.copy_(packed[:, :-2].reshape_as(q))
        k.copy_(packed[:, -2].reshape_as(k))
        v.copy_(packed[:, -1].reshape_as(v))
        hf_attention.o_proj.weight.copy_(native_attention.wo.weight)
        hf_attention.gate_proj.weight.copy_(native_attention.gate_proj.weight)


class TestOptMoEHFAssets(unittest.TestCase):
    def test_overwrite_config_preserves_opt_moe_attention_flags(self):
        model_config = _build_native_model_config(
            gate_only=True,
            mid_norm_position="before",
            qk_rope_dim=2,
        )
        model = _build_expanded(model_config)

        exported = overwrite_config(model)

        self.assertTrue(exported["gate_only"])
        self.assertEqual(exported["mid_norm_position"], "before")
        self.assertEqual(exported["qk_rope_dim"], 2)
        self.assertEqual(exported["partial_rotary_factor"], 0.5)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "flex attention needs torch.compile, unsupported on this CPU platform; "
        "sdpa is no longer accepted for language models, so there is no CPU "
        "path for this comparison",
    )
    def test_hf_attention_matches_native_for_new_attention_modes(self):
        for (
            gate_only,
            mid_norm_position,
            head_wise_mid_norm,
            qk_norm,
            mid_norm,
            norm_everywhere,
        ) in (
            (True, "after", False, False, False, True),
            (False, "before", False, False, False, True),
            (False, "after", True, False, False, True),
            # mid_norm-only: no qk_norm, no norm_everywhere -> v_norm stays off.
            (False, "after", False, False, True, False),
            # qk_norm + mid_norm without norm_everywhere -> v_norm still off.
            (False, "after", False, True, True, False),
        ):
            with self.subTest(
                gate_only=gate_only,
                mid_norm_position=mid_norm_position,
                head_wise_mid_norm=head_wise_mid_norm,
                qk_norm=qk_norm,
                mid_norm=mid_norm,
                norm_everywhere=norm_everywhere,
            ):
                torch.manual_seed(0)
                native_attention = _build_native_model_config(
                    head_dim=16,
                    gate_only=gate_only,
                    mid_norm_position=mid_norm_position,
                    head_wise_mid_norm=head_wise_mid_norm,
                    qk_norm=qk_norm,
                    mid_norm=mid_norm,
                    norm_everywhere=norm_everywhere,
                ).layer.attention.build()
                hf_attention = _build_hf_attention(
                    head_dim=16,
                    gate_only=gate_only,
                    mid_norm_position=mid_norm_position,
                    head_wise_mid_norm=head_wise_mid_norm,
                    qk_norm=qk_norm,
                    mid_norm=mid_norm,
                    norm_everywhere=norm_everywhere,
                )

                if not norm_everywhere:
                    self.assertIsInstance(native_attention.v_norm, torch.nn.Identity)
                    self.assertIsInstance(hf_attention.v_norm, torch.nn.Identity)

                _copy_attention_weights(
                    native_attention, hf_attention, head_dim=16
                )
                native_attention.to("cuda")
                hf_attention.to("cuda")

                device = "cuda"
                num_tokens = 4
                hidden_states = torch.randn(
                    num_tokens, 32, dtype=torch.float32, device=device
                )
                block_mask = create_block_mask(
                    # All-True: HF is called with attention_mask=None, so the
                    # two must see the same (unmasked) attention.
                    lambda b, h, q_idx, kv_idx: q_idx >= 0,
                    B=1,
                    H=None,
                    Q_LEN=num_tokens,
                    KV_LEN=num_tokens,
                    device=device,
                )

                # FlexAttention has no CPU backward; this test only compares
                # forward values.
                with torch.no_grad():
                    native_output = native_attention(
                        hidden_states,
                        attention_masks=block_mask,
                        positions=torch.arange(num_tokens),
                    )
                    # Native is token-flat [T, D] since 0.5.0; HF is still
                    # batched [B, T, D].
                    hf_output, _ = hf_attention(
                        hidden_states=hidden_states.unsqueeze(0),
                        position_embeddings=None,
                        attention_mask=None,
                    )

                torch.testing.assert_close(
                    native_output, hf_output.squeeze(0), rtol=1e-4, atol=1e-4
                )

    def test_state_dict_adapter_rejects_unsupported_norm_tensors(self):
        adapter = OPTMoEStateDictAdapter(_build_native_model_config(), None)

        with self.assertRaisesRegex(ValueError, "learned norm tensors"):
            adapter.to_hf({"layers.0.attention.mid_norm.weight": torch.ones(4)})

    def test_overwrite_config_uses_runtime_model_config_without_module_inspection(self):
        model_config = _build_native_moe_model_config()

        exported = overwrite_config(_ConfigOnlyModel(model_config))

        self.assertEqual(exported["hidden_size"], 12)
        self.assertEqual(exported["num_hidden_layers"], 4)
        self.assertEqual(exported["num_attention_heads"], 3)
        self.assertEqual(exported["num_key_value_heads"], 1)
        self.assertEqual(exported["head_dim"], 4)
        self.assertEqual(exported["qk_rope_dim"], 2)
        self.assertEqual(exported["partial_rotary_factor"], 0.5)
        self.assertEqual(exported["intermediate_size"], 18)
        self.assertEqual(exported["moe_intermediate_size"], 24)
        self.assertEqual(exported["moe_scaling_factor"], 1.0)
        self.assertEqual(exported["n_active_experts"], 4)
        self.assertEqual(exported["n_total_experts"], 16)
        self.assertEqual(exported["n_shared_experts"], 2)
        self.assertEqual(exported["rope_pattern"], "RRNR")
        self.assertEqual(exported["swa_pattern"], "SSFS")
        self.assertEqual(exported["rope_theta_swa"], 5000.0)
        self.assertEqual(exported["hidden_act"], "silu")
        self.assertEqual(exported["residual_scale"], "depth_scale")
