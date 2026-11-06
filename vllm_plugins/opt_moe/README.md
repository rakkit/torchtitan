# vllm-opt-moe

A vLLM plugin that serves torchtitan's `opt_moe` checkpoints in their HF format,
without torchtitan.

```bash
pip install ./vllm_plugins/opt_moe
vllm serve <an opt_moe HF folder> --trust-remote-code --dtype bfloat16
```

The HF folder (written by torchtitan's
`scripts/checkpoint_conversion/convert_opt_moe_to_hf.py`) carries its own config
class (`configuration_opt_moe.py`, loaded with `--trust-remote-code`), the
weights, tokenizer files, a chat template and a generation config. The package's
`vllm.general_plugins` entry point registers `OptMoeForCausalLM`, the
architecture the folder's `config.json` names.

## The model (`vllm_opt_moe/model.py`)

Per decoder layer, two caches:

```
x -> RMSNorm -> qkv_proj -> [q|k|v] -> x + conv(x)   <- conv state: the last W - 1 raw
                                                        q / k / v of each sequence
                            -> qk-norm -> RoPE (none on NoPE layers, partial RoPE)
                            -> Attention              <- KV cache (sinks, per-layer
                                                        sliding window)
                            -> * sigmoid(gate_proj(x)) -> o_proj -> residual
  -> RMSNorm -> SwiGLU FFN (first layers) or sigmoid MoE (+ score-correction
     bias for the choice, renormalized top-k, scaling, shared expert) -> residual
```

The KV cache holds k and v after the conv, qk-norm and RoPE, and q is not
cached, so the raw inputs the next step's conv needs are kept separately: as one
more sliding-window KV cache per layer (window = block = the conv width), the
three streams packed per KV head, read and written by vLLM's Inkling
`fused_sconv` kernel (insert + conv in one launch, the same path for prefill,
decode and speculative tokens, CUDA-graph safe), as vLLM's Inkling model keeps
its conv state. Weights in bf16; `--quantization fp8` quantizes the linears and
experts online (the router and the convs stay as they are). The model runs
under vLLM's `torch.compile` (each conv is the custom op
`vllm::opt_moe_short_conv`, which reads its state from the forward context) with
piecewise CUDA graphs for prefill and full ones for decoding.

Tensor parallelism on nodes without NVSwitch (e.g. 4 x GH200) needs
`VLLM_ALLREDUCE_USE_FLASHINFER=0`: FlashInfer's all-reduce falls back to its
trtllm backend there, and the first forward hangs.

## Not yet supported

- `conv_streams: kv` (the conv on k and v only), other `scoring_func`s than
  sigmoid;
- tensor parallelism above the KV head count;
- the MTP depths (ignored, not refused: not loaded; speculative decoding with
  them is a later step).

Untested: partial RoPE (`rope.dim < head_dim`). `max_position_embeddings` is the
training context (`seq_len`) and caps vLLM's `max_model_len`.

## Checks

Against the HF model's transformers forward in fp32, on a small converted model
with every feature (q / k / v conv, sinks, gate, a sliding window of 16, a full
and a NoPE layer, a dense and MoE layers with the router bias and a shared
expert), bf16, eager, TP 1 (11-06): teacher-forced logprobs of 41 / 64 / 100
tokens and of 40 decoded tokens past the window, |vLLM - HF fp32| mean 0.0002 /
0.0003, max <= 0.0012, equal to the HF bf16 model's own (the bf16 floor); with
the conv broken to the identity 0.006 - 0.0075 (the check sees the conv). With
2 KV heads (11-06), prefill / decode mean: TP 1 0.0002 / 0.0004, with CUDA
graphs 0.0003 / 0.0004, TP 2 (custom all-reduce, CUDA graphs) 0.0003 / 0.0003,
all at the floor; FP8 0.0009 / 0.0007 (max 0.0028).

## Speed of the conv

FP8 (online, from the bf16 folder), TP 1, one GH200, CUDA graphs, a 41.4B /
3.05B-active shape (40 layers, d 2048, 32 q / 4 kv heads, 256 experts top-6 of
width 640, SWA 512 at 5:1) with vs without the q / k / v conv, random weights,
1,024-token prompts, 256 decoded tokens, 3 fresh processes each (11-06):

| batch | decode ms / step, conv / no conv |
|---|---|
| 1 | 8.05 / 7.55 (+6.6%) |
| 16 | 13.89 / 13.09 (+6.1%) |
| 64 | 20.44 / 19.59 (+4.3%) |
| 256 | 30.25 / 29.25 (+3.4%) |

Prefill not measured yet: that run's prefill timings were prefix-cache hits
(the timed calls repeated the warm-up's prompts).
