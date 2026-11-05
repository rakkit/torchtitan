# opt_moe HF export

`configuration_opt_moe.py` and `modeling_opt_moe.py` are the HF model; every
export copies them next to its weights (`trust_remote_code`). `setup_hf.py`
turns the native config into `OptMoEConfig`. The weights are converted by
`../state_dict_adapter.py`. Written against transformers 5.17.

## Export and load

```bash
python scripts/checkpoint_conversion/convert_to_hf.py <dcp_step_dir> <out_dir> \
    --job_config <job_config.json> --model_flavor <flavor> --hf_assets_path <tokenizer_dir>
# EMA weights: convert_ema_to_hf.py, same arguments
```

```python
model = AutoModelForCausalLM.from_pretrained(
    out_dir, trust_remote_code=True, dtype=torch.bfloat16,
    attn_implementation="torchtitan_flex",  # or "sdpa", "flash_attention_2", ...
)
```

Both converters write bf16 by default (`--export_dtype`). That loses nothing
for inference: the trainer computes from bf16 copies of every weight.

## What matches the trainer

The model runs torchtitan's forward op for op.

With `attn_implementation="torchtitan_flex"` (CUDA), attention is the compiled
flex kernel torchtitan uses, and a bf16 prefill equals the native forward
**bitwise**. This is tested on MoE and dense flavors with every gating, norm
and RoPE variant.

With the stock backends (sdpa, flash, eager), only the attention kernel
differs. The differences are last-bit ones, but MoE routing can amplify them
on tokens whose top-k scores nearly tie.

| op | precision |
|---|---|
| weights, activations, attention, expert GEMMs | bf16 (fp32 accumulation in the kernels) |
| router gate | bf16 x bf16 -> fp32 output |
| router scores, balancing bias, top-k, renormalise, scale | fp32 |
| routing-weight multiply of the expert outputs | fp32, then bf16 for the sum |
| RoPE cos/sin and the q/k rotation | fp32 |
| attention output gate | fp32 |

The balancing bias (`mlp.gate.e_score_correction_bias`) stays fp32 in every
checkpoint and after a bf16 load, as it does in training.

## Checkpoint layout

Attention and dense / shared MLPs use the usual `q/k/v/o_proj` and
`gate/up/down_proj`. Routed experts are stacked, as in transformers' current
MoE models, and are exactly torchtitan's tensors:

| HF | shape | native |
|---|---|---|
| `mlp.experts.gate_up_proj` | `[E, 2F, D]` (gate rows, then up rows) | `feed_forward.routed_experts.w13` `[E, 2, F, D]` |
| `mlp.experts.down_proj` | `[E, D, F]` | `feed_forward.routed_experts.w2` |
| `mlp.gate.weight` | `[E, D]` | `feed_forward.router.gate.weight` |

Expert backends: unset means this model's forward. `experts_implementation=`
`"grouped_mm"`, `"batched_mm"`, ... select transformers' own; they are
equivalent up to rounding.

## Attention short conv and sinks

Both are optional on the native attention and exported as-is:
- **Short conv** (`conv_kernel_size > 0`) is KDA's: a depthwise causal conv on
  q, k and v, then SiLU, in `self_attn.{q,k,v}_conv.weight`. It is gated by
  `position_ids`, so packed documents and left padding never leak. Layers are
  `"hybrid"` / `"hybrid_sliding"`, whose cache also carries the last `K - 1`
  inputs of q, k and v, so generation decodes token by token.
- **Sinks** (`attention_sinks`) are gpt-oss's per-head logit
  (`self_attn.sinks`). They need `attn_implementation` `"torchtitan_flex"` or
  `"eager"`; other backends raise.

## Not exported

`setup_hf.hf_config` raises on anything the template does not compute:
- learned (non parameter-free) norms;
- non-SwiGLU MLPs;
- scaled RoPE (yarn, llama3).

The vLLM plugin (`vllm_plugins/opt_moe_plugins`) still reads per-expert keys;
it is not yet updated for the stacked experts.
