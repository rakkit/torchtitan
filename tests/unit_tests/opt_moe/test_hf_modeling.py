# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The HF model (hf/modeling_opt_moe.py), loaded from a torchtitan state dict
through the adapter, reproduces torchtitan's forward bitwise: the main and the
MTP logits of a model with every feature (conv on q / k / v, sinks, gate, a
sliding-window, a full and a NoPE layer, a dense and two MoE layers with a
shared expert and the router bias, one MTP depth), packed documents, fp32 and
bf16. torchtitan runs without its local_compile regions (they fuse the same
element-wise ops, so they round differently)."""

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config
from torchtitan.models.opt_moe.hf.configuration_opt_moe import OptMoeConfig
from torchtitan.models.opt_moe.hf.modeling_opt_moe import OptMoeForCausalLM
from torchtitan.models.opt_moe.mtp import roll_mtp_sequence
from torchtitan.models.opt_moe.state_dict_adapter import (
    hf_config,
    OPTMoEStateDictAdapter,
)

DOCS = [70, 1, 185]
VOCAB = 512


def _torchtitan_model():
    cfg = build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=256,
        n_layers=3,
        swa_pattern="SSF",
        rope_pattern="RRN",
        n_dense_layers=1,
        dense_hidden_dim=256,
        conv_kernel_size=4,
        attention_sinks=True,
        sinks_init_fn_type="trunc_normal",
        sinks_init_std=0.5,
        gated_attention_type="element-wise",
        num_mtp_layers=1,
        vocab_size=VOCAB,
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.add_(0.2 * torch.randn_like(p))
            if "_conv." in name:
                p.add_(0.3 * torch.randn_like(p))
        for module in model.modules():
            bias = getattr(module, "expert_bias_E", None)
            if bias is not None:
                bias.copy_(0.05 * torch.randn_like(bias))
    return cfg, model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex / conv need CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hf_forward_is_torchtitan_forward_bitwise(dtype):
    # Both models' compiled FlexAttention start from the same state: with
    # torch.compile's default dynamic=None, one that has seen other shapes
    # (earlier tests) runs a dynamic-shape kernel, rounding differently.
    torch._dynamo.reset()
    cfg, tt = _torchtitan_model()
    hf = OptMoeForCausalLM(OptMoeConfig(**hf_config(cfg))).cuda()
    hf.load_state_dict(OPTMoEStateDictAdapter(cfg, None).to_hf(tt.state_dict()))
    # The parameters in `dtype`, as FSDP's param_dtype: buffers (torchtitan's
    # RoPE cache, the expert bias) stay fp32, and so does the HF router bias
    # (the converter keeps it fp32).
    for model in (tt, hf):
        for name, p in model.named_parameters():
            if not name.endswith("e_score_correction_bias"):
                p.data = p.data.to(dtype)

    positions = torch.cat([torch.arange(n) for n in DOCS]).cuda()
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB, (positions.numel(),), device="cuda")
    rolled, valid = roll_mtp_sequence(
        tokens, shift=1, fill_value=0, positions_T=positions
    )
    with torch.no_grad():
        want_main, want_mtp = tt(
            (tokens, rolled),
            positions=positions,
            attention_metadata=tt._get_attention_metadata(positions),
            mtp_input_valid_masks=(valid,),
        )
        hidden = hf.model(
            tokens, positions, mtp_input_ids=(rolled,), mtp_valid=(valid,)
        )
        got_main, got_mtp = (hf.lm_head(h) for h in hidden)
        out = hf(tokens.view(1, -1), position_ids=positions.view(1, -1))
    assert torch.equal(got_main, want_main)
    assert torch.equal(got_mtp, want_mtp)
    assert torch.equal(out.logits.view_as(want_main), want_main)


def test_a_dcp_checkpoint_converts_to_an_hf_folder_that_loads(tmp_path):
    """convert_opt_moe_to_hf: a run's DCP checkpoint (its model rebuilt from
    job_config.json) becomes an HF folder that AutoModelForCausalLM loads with
    trust_remote_code, every weight unchanged (fp32 export)."""
    import json

    import torch.distributed.checkpoint as dcp

    from scripts.checkpoint_conversion.convert_opt_moe_to_hf import (
        convert_opt_moe_to_hf,
    )
    from torchtitan_recipes.models.opt_moe import opt_moe_mis_8b
    from transformers import AutoModelForCausalLM

    run = opt_moe_mis_8b()
    run.model = build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=256,
        n_layers=2,
        swa_pattern="SF",
        rope_pattern=None,
        conv_kernel_size=4,
        attention_sinks=True,
        num_mtp_layers=1,
        vocab_size=VOCAB,
    )
    run.hf_assets_path = str(tmp_path / "assets")  # no tokenizer files to copy
    (tmp_path / "job_config.json").write_text(json.dumps(run.to_dict()))
    model = run.model.build()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p))
    dcp.save(model.state_dict(), checkpoint_id=str(tmp_path / "step-1"))

    convert_opt_moe_to_hf(
        tmp_path / "step-1", tmp_path / "job_config.json", tmp_path / "hf", "float32"
    )
    hf = AutoModelForCausalLM.from_pretrained(
        tmp_path / "hf", trust_remote_code=True, dtype=torch.float32
    )
    want = OPTMoEStateDictAdapter(run.model, None).to_hf(model.state_dict())
    got = hf.state_dict()
    assert got.keys() == want.keys()
    for key, value in want.items():
        assert torch.equal(got[key], value), key
