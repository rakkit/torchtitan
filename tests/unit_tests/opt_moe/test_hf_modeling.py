# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The HF model (hf/modeling_opt_moe.py), loaded from a torchtitan state dict
through the adapter, on a model with every feature (conv on q / k / v, sinks,
gate, a sliding-window, a full and a NoPE layer, a dense and two MoE layers with
a shared expert and the router bias, one MTP depth):

- ``packed_forward`` reproduces torchtitan's forward bitwise: the main and the
  MTP logits, packed documents, fp32 and bf16 (torchtitan without its
  local_compile regions: they fuse the same element-wise ops, so they round
  differently);
- the transformers ``forward`` (``generate``): a cached decode past the sliding
  window gives the full forward's logits, a left-padded batch gives each row's
  own greedy tokens, and one document's logits are ``packed_forward``'s up to
  the kernels' rounding."""

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


# sinks in place of the gate (one or neither)
_SINKS = dict(
    gated_attention_type=None,
    attention_sinks=True,
    sinks_init_fn_type="trunc_normal",
    sinks_init_std=0.5,
)


def _torchtitan_model(**overrides):
    every_feature = dict(
        n_layers=3,
        swa_pattern="SSF",
        rope_pattern="RRN",
        n_dense_layers=1,
        dense_hidden_dim=256,
        conv_kernel_size=4,
        gated_attention_type="element-wise",
        num_mtp_layers=1,
        vocab_size=VOCAB,
        # two KV groups: one would make every per-group layout trivially right
        n_kv_heads=2,
    )
    cfg = build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=256,
        **every_feature | overrides,
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
@pytest.mark.parametrize(
    "variant",
    [
        {},
        _SINKS,
        {"gated_attention_type": None},
        {"rope_of_swa": {"dim": 64, "theta": 10000.0}},
    ],
    ids=["every-feature-gate", "sinks", "neither", "partial-rope-on-swa"],
)
def test_hf_forward_is_torchtitan_forward_bitwise(dtype, variant):
    # Both models' FlexAttention compile from the same state: what compiled
    # before (other shapes, kernels autotuned under other load) can give one
    # of them a different kernel, which rounds differently.
    torch._dynamo.reset()
    cfg, tt = _torchtitan_model(**variant)
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
        hidden = hf.model.packed_forward(
            tokens, positions, mtp_input_ids=(rolled,), mtp_valid=(valid,)
        )
        got_main, got_mtp = (hf.lm_head(h) for h in hidden)
        out = hf.packed_forward(tokens.view(1, -1), position_ids=positions.view(1, -1))
    assert torch.equal(got_main, want_main)
    assert torch.equal(got_mtp, want_mtp)
    assert torch.equal(out.logits.view_as(want_main), want_main)


def _hf_model(dtype=torch.float32, **overrides):
    """The HF model of ``_torchtitan_model(**overrides)``'s weights, in eval."""
    cfg, tt = _torchtitan_model(**overrides)
    hf = OptMoeForCausalLM(OptMoeConfig(**hf_config(cfg))).cuda()
    hf.load_state_dict(OPTMoEStateDictAdapter(cfg, None).to_hf(tt.state_dict()))
    for name, p in hf.named_parameters():
        if not name.endswith("e_score_correction_bias"):
            p.data = p.data.to(dtype)
    return hf.eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped GEMMs need CUDA")
@pytest.mark.parametrize(
    "arch",
    [
        {"sliding_window_size": 16},
        {"sliding_window_size": 16, **_SINKS},
        {"swa_pattern": "FFF", "conv_kernel_size": 0},
    ],
    ids=["swa16-conv-gate", "swa16-conv-sinks", "full-attention-no-conv"],
)
def test_cached_decode_gives_the_full_forward(arch):
    """A 21-token prefill, then 19 one-token steps through the DynamicCache
    (its K / V, the sliding-window layers past their 16-token window and the
    conv states; or full attention only, plain KV layers): every position's
    logits are the full forward's (fp32, the same math over other shapes)."""
    hf = _hf_model(**arch)
    torch.manual_seed(2)
    tokens = torch.randint(0, VOCAB, (1, 40), device="cuda")
    with torch.no_grad():
        full = hf(tokens, use_cache=False).logits
        out = hf(tokens[:, :21], use_cache=True)
        cache, steps = out.past_key_values, [out.logits]
        for i in range(21, 40):
            out = hf(tokens[:, i : i + 1], past_key_values=cache, use_cache=True)
            steps.append(out.logits)
    assert cache.get_seq_length() == 40
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped GEMMs need CUDA")
@pytest.mark.parametrize("arch", [{}, _SINKS], ids=["gate", "sinks"])
def test_the_eager_attention_in_query_chunks_is_the_whole(arch, monkeypatch):
    """Long contexts run the eager attention in query chunks: one query per
    chunk (the window and causal masks, the sinks sliced per chunk) gives the
    one-chunk forward's logits."""
    from torchtitan.models.opt_moe.hf import modeling_opt_moe

    hf = _hf_model(sliding_window_size=16, **arch)
    torch.manual_seed(2)
    ids = torch.randint(0, VOCAB, (2, 37), device="cuda")
    with torch.no_grad():
        whole = hf(ids, use_cache=False).logits
        monkeypatch.setattr(modeling_opt_moe, "_EAGER_SCORES_PER_CHUNK", 1)
        chunked = hf(ids, use_cache=False).logits
    torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped GEMMs need CUDA")
def test_a_left_padded_batch_generates_each_rows_tokens():
    """``generate`` on two left-padded prompts (13 and 29 tokens, the window 16
    exceeded) gives each prompt's greedy tokens and logits alone: the padding is
    masked in attention and enters the conv as the zero history a row starts
    from."""
    hf = _hf_model(sliding_window_size=16)
    hf.generation_config.pad_token_id = 0
    torch.manual_seed(3)
    prompts = [torch.randint(1, VOCAB, (n,), device="cuda") for n in (13, 29)]
    greedy = {"max_new_tokens": 8, "do_sample": False}
    greedy |= {"return_dict_in_generate": True, "output_logits": True}
    alone = []
    with torch.no_grad():
        for p in prompts:
            alone.append(hf.generate(p.view(1, -1), **greedy))
        width = max(p.numel() for p in prompts)
        ids = torch.zeros(len(prompts), width, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for row, p in enumerate(prompts):
            ids[row, width - p.numel() :] = p
            mask[row, width - p.numel() :] = 1
        batch = hf.generate(ids, attention_mask=mask, **greedy)
    for row, (p, own) in enumerate(zip(prompts, alone, strict=True)):
        assert torch.equal(batch.sequences[row, width:], own.sequences[0, p.numel() :])
        for got, want in zip(batch.logits, own.logits, strict=True):
            torch.testing.assert_close(got[row], want[0], atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex / conv need CUDA")
def test_the_transformers_forward_is_the_packed_forward():
    """One document: the transformers forward (eager attention, torch conv)
    gives ``packed_forward``'s logits (FlexAttention, attn_gym's conv) up to
    the kernels' rounding (fp32)."""
    torch._dynamo.reset()
    hf = _hf_model()
    torch.manual_seed(4)
    tokens = torch.randint(0, VOCAB, (1, 200), device="cuda")
    with torch.no_grad():
        packed = hf.packed_forward(tokens).logits
        got = hf(tokens, use_cache=False).logits
    torch.testing.assert_close(got, packed, atol=1e-4, rtol=1e-4)


def _pretrain_run(tmp_path):
    """A small opt_moe run as conversion sees it: its job_config.json and a
    pretrain run's assets (a tokenizer, no chat template, no generation
    config); the model built with random weights."""
    import json
    import shutil
    from pathlib import Path

    from torchtitan_recipes.models.opt_moe import opt_moe_mis_8b

    assets = tmp_path / "assets"
    shutil.copytree(Path(__file__).parents[2] / "assets" / "tokenizer", assets)
    tokenizer_config = json.loads((assets / "tokenizer_config.json").read_text())
    del tokenizer_config["chat_template"]
    (assets / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

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
    run.hf_assets_path = str(assets)
    (tmp_path / "job_config.json").write_text(json.dumps(run.to_dict()))
    model = run.model.build()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p))
    return run, model


def _hf_weights(folder):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        folder, trust_remote_code=True, dtype=torch.float32
    ).state_dict()


def test_a_dcp_checkpoint_converts_to_an_hf_folder_that_loads(tmp_path, monkeypatch):
    """convert_opt_moe_to_hf: a run's DCP checkpoint (its model rebuilt from
    job_config.json) becomes a complete HF folder in several weight files (the
    file size limit shrunk to 256 KiB here): AutoModelForCausalLM loads it
    with trust_remote_code, every weight unchanged (fp32 export), and
    AutoTokenizer loads it with a chat template. The run's assets are a pretrain
    run's (a tokenizer, no chat template, no generation config): the folder gets
    the placeholder template and a generation config with the bos / eos ids."""
    import json
    from pathlib import Path

    import torch.distributed.checkpoint as dcp

    from scripts.checkpoint_conversion import convert_opt_moe_to_hf as conversion
    from scripts.checkpoint_conversion.convert_opt_moe_to_hf import (
        convert_opt_moe_to_hf,
    )
    from torchtitan.models.opt_moe import hf as hf_files
    from transformers import AutoTokenizer, GenerationConfig

    run, model = _pretrain_run(tmp_path)
    dcp.save(model.state_dict(), checkpoint_id=str(tmp_path / "step-1"))

    monkeypatch.setattr(conversion, "MAX_FILE_BYTES", 256 * 1024)
    convert_opt_moe_to_hf(
        tmp_path / "step-1",
        tmp_path / "job_config.json",
        tmp_path / "hf",
        "float32",
        read_threads=4,
    )
    out = tmp_path / "hf"
    files = sorted(f.name for f in out.glob("*.safetensors"))
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert len(files) > 1 and sorted(set(index["weight_map"].values())) == files
    # the HF files only: no staged copy of the tensors left next to them
    assert not (out / "sharded").exists()
    placeholder = Path(hf_files.__file__).parent / "chat_template.jinja"
    assert (out / "chat_template.jinja").read_text() == placeholder.read_text()
    tokenizer = AutoTokenizer.from_pretrained(out)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    # The placeholder joins the turns' text with a blank line.
    assert tokenizer.apply_chat_template(messages, tokenize=False) == "hi\n\nyo"
    config = json.loads((out / "config.json").read_text())
    generation = GenerationConfig.from_pretrained(out)
    assert (config["bos_token_id"], config["eos_token_id"]) == (2002, 2003)
    assert (generation.bos_token_id, generation.eos_token_id) == (2002, 2003)
    want = OPTMoEStateDictAdapter(run.model, None).to_hf(model.state_dict())
    got = _hf_weights(out)
    assert got.keys() == want.keys()
    for key, value in want.items():
        assert torch.equal(got[key], value), key


def test_a_bf16_export_keeps_the_router_bias_exact(tmp_path):
    """The router bias picks the top-k experts and training keeps it in fp32:
    a bf16 export must carry the exact fp32 values (HF and vLLM read them in
    fp32), not bf16-rounded ones that change expert choices."""
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import load_file

    from scripts.checkpoint_conversion.convert_opt_moe_to_hf import (
        convert_opt_moe_to_hf,
    )

    run, model = _pretrain_run(tmp_path)
    with torch.no_grad():
        biases = [b for n, b in model.named_buffers() if n.endswith("expert_bias_E")]
        assert biases
        for bias in biases:
            bias.copy_(0.05 * torch.randn_like(bias))
    dcp.save(model.state_dict(), checkpoint_id=str(tmp_path / "step-1"))
    convert_opt_moe_to_hf(
        tmp_path / "step-1",
        tmp_path / "job_config.json",
        tmp_path / "hf",
        "bfloat16",
        read_threads=4,
    )
    got = {}
    for f in (tmp_path / "hf").glob("*.safetensors"):
        got |= load_file(f)
    want = OPTMoEStateDictAdapter(run.model, None).to_hf(model.state_dict())
    keys = [k for k in want if k.endswith("e_score_correction_bias")]
    assert keys
    for key in keys:
        assert got[key].dtype == torch.float32, key
        torch.testing.assert_close(got[key], want[key], rtol=0, atol=0, msg=key)


def test_an_ema_copy_converts_in_place_of_the_trained_weights(tmp_path):
    """convert_opt_moe_ema_to_hf: a checkpoint laid out as CheckpointManager
    writes it (the model at the top level, two EMA copies under "ema") exports
    the chosen copy's averages for every parameter and the trained buffers (the
    router bias, not averaged); with two copies and no ema_key it refuses."""
    import pytest
    import torch.distributed.checkpoint as dcp

    from scripts.checkpoint_conversion.convert_opt_moe_ema_to_hf import (
        convert_opt_moe_ema_to_hf,
    )
    from torchtitan.components.checkpointer import ModelWrapper
    from torchtitan.components.optim import EMA

    run, model = _pretrain_run(tmp_path)
    ema = EMA.Config(decays=[0.9, 0.99]).build(model_parts=[model])
    averages = {}
    for opt in ema._param_optimizers:
        for group in opt.param_groups:
            for name, tensor in zip(group["param_names"], group["params"]):
                copies = opt.state[tensor]["ema_params"]
                copies["decay_0p9"].copy_(3 * tensor.detach())
                copies["decay_0p99"].copy_(-2 * tensor.detach())
                averages[name] = copies["decay_0p99"].clone()
    assert averages.keys() == dict(model.named_parameters()).keys()
    dcp.save(
        {"ema": ema, **ModelWrapper(model).state_dict()},
        checkpoint_id=str(tmp_path / "step-1"),
    )

    args = (tmp_path / "step-1", tmp_path / "job_config.json")
    with pytest.raises(ValueError, match="pass --ema_key"):
        convert_opt_moe_ema_to_hf(*args, tmp_path / "hf", "float32")
    convert_opt_moe_ema_to_hf(
        *args, tmp_path / "hf", "float32", read_threads=4, ema_key="decay_0p99"
    )
    want = OPTMoEStateDictAdapter(run.model, None).to_hf(model.state_dict() | averages)
    got = _hf_weights(tmp_path / "hf")
    assert got.keys() == want.keys()
    for key, value in want.items():
        assert torch.equal(got[key], value), key
