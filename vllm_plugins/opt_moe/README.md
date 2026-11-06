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
                                     on k and v         k / v of each sequence
                            -> qk-norm -> RoPE (none on NoPE layers, partial RoPE)
                            -> Attention              <- KV cache (sinks, per-layer
                                                        sliding window)
                            -> * sigmoid(gate_proj(x)) -> o_proj -> residual
  -> RMSNorm -> SwiGLU FFN (first layers) or sigmoid MoE (+ score-correction
     bias for the choice, renormalized top-k, scaling, shared expert) -> residual
```

The KV cache holds k and v after the conv, qk-norm and RoPE, and q is not
cached, so the raw k / v the next step's conv needs are kept separately: as one
more sliding-window KV cache per layer (window = the conv width; vLLM's Inkling
sconv backend builds its block table and slots), the k heads then the v heads
as its heads. Everything between the qkv projection and the attention is one
launch per layer (`vllm_opt_moe/prologue.py`, after vLLM's Inkling prologue):
q -> RMSNorm -> RoPE; k / v read in place from the projection, into the conv
state, x + conv(x) over this forward's rows or the state, k -> RMSNorm -> RoPE
(partial on `rope.dim < head_dim`, none on NoPE layers); the same path for
prefill, decode and speculative tokens. Weights in bf16; FP8 serving is vLLM's
online block-wise FP8, `--quantization fp8_per_block` (128 x 128 weight blocks,
1 x 128 activation groups, the linears and experts quantized at load; the router
and the convs stay as they are; every quantized dim a multiple of 128). The model runs under vLLM's `torch.compile` (the prologue is the custom op
`vllm::opt_moe_attention_prologue`, which reads the conv state from the forward
context) with piecewise CUDA graphs for prefill and full ones for decoding.

MoE kernel configs: vLLM ships tuned fused-MoE configs for many GPUs but none
for GH200, so its Triton MoE runs a default config there. `register()` points
`VLLM_TUNED_CONFIG_FOLDER` at `vllm_opt_moe/moe_configs/` (unless already set;
vLLM reads it before its own folder). For now it holds H100's bf16 config for
256 experts of width 512 and H100's block-FP8 config for 256 experts of width
384 (`E=256,N=384,...,block_shape=[128,128]`), the H100 configs matching our
shapes. TODO: tune block FP8 and bf16 on GH200 for the final expert shape
(vLLM's benchmarks/kernels/benchmark_moe.py) once the architecture is final.
vLLM's DeepGEMM block-FP8 MoE does not import in this environment (its bundled
`deep_gemm` lacks `_C`), so the block-FP8 MoE runs vLLM's Triton kernel.

Tensor parallelism on nodes without NVSwitch (e.g. 4 x GH200) needs
`VLLM_ALLREDUCE_USE_FLASHINFER=0`: FlashInfer's all-reduce falls back to its
trtllm backend there, and the first forward hangs.

## Speculative decoding with the MTP depths

A folder with MTP depths (`num_nextn_predict_layers`, HF layers `L ..`) drafts
with them, as DeepSeek / Qwen do:

```bash
vllm serve <folder> --trust-remote-code --dtype bfloat16 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

The draft (`OptMoeMTP`, architecture `OptMoeMTPModel`) is torchtitan's MTP
block: `shared_head.norm(layer(eh_proj(cat(enorm(e), hnorm(h)))))` on the
target's pre-norm hidden state (the target applies its final norm in
`compute_logits`), the decoder layer a copy of the last one, the embedding and
lm_head shared with the target; more speculative tokens than depths reuse the
depths. TODO: vLLM maps a target's `model_type` to its MTP draft only inside
`SpeculativeConfig.hf_config_override`, with no entry for out-of-tree models;
the plugin wraps that static method for `opt_moe`.

Check (11-06; `check_vllm_mtp.py` in the user's configs): the every-feature
4-layer model with one MTP depth, trained on a pattern whose token `t + 2` is
token `t` + 1 (the draft needs the target's hidden state), bf16, eager: 16
greedy continuations of 48 tokens with the draft equal the ones without,
384 / 384 drafts accepted; with the draft's hidden state zeroed 11 / 741
(0.015).

## Not yet supported

- other `scoring_func`s than sigmoid;
- tensor parallelism above the KV head count.

`max_position_embeddings` is the training context (`seq_len`) and caps vLLM's
`max_model_len`.

## Checks

Against the HF model's transformers forward in fp32 (`check_vllm_model.py` in
the user's configs), on a small converted model: 4 layers, 2 KV heads, the k / v
conv, a sliding window of 16, a full and a NoPE layer, a dense and MoE layers
with the router bias and a shared expert, with the element-wise gate, sinks or
neither;
teacher-forced logprobs of 41 / 64 / 100 tokens and of 40 decoded tokens past
the window, |vLLM - HF fp32| against the HF bf16 model's own (the bf16 floor:
mean 0.0002-0.0003, max 0.0006-0.0013). With the fused prologue (11-06), bf16:

| model | prefill mean / max | decode mean / max |
|---|---|---|
| gate, eager | 0.0002 / 0.0014 | 0.0003 / 0.0011 |
| gate, CUDA graphs | 0.0002-0.0003 / 0.0014 | 0.0003 / 0.0011 |
| sinks, CUDA graphs | 0.0002 / 0.0011 | 0.0003 / 0.0009 |
| neither gate nor sinks, eager / CUDA graphs | 0.0002-0.0003 / 0.0011 | 0.0003 / 0.0009 |
| gate + partial RoPE (64 of 128 dims on the SWA layers), CUDA graphs | 0.0002-0.0003 / 0.0014 | 0.0003 / 0.0013 |
| gate, block FP8 (`fp8_per_block`, experts 256 wide), eager | 0.0008-0.0011 / 0.0040 | 0.0009 / 0.0025 |
| gate, block FP8, CUDA graphs | 0.0009-0.0010 / 0.0036 | 0.0009 / 0.0025 |
| gate, the conv broken (zero conv weights) | 0.0091-0.0119 / 0.0411 | 0.0109 / 0.0320 |

Preemption (`check_vllm_preempt.py`): 24 sequences of 150 tokens on a
400-block pool, 23 preemptions (vLLM recomputes the evicted requests, their
conv state rebuilt from the prompt): decode mean 0.0003 / max 0.0022 eager,
0.0003 / 0.0019 with CUDA graphs, worst sequence 0.0004, at the floor.

The kernel alone against a PyTorch reference (a prefill then 6 decode steps
through the paged state, rope_dim 128 / 64 / 0): within 0.09 bf16 ulp; with the
state wiped between steps 126-223 ulps. Before the k / v-only change (the q / k
/ v conv, 11-06): TP 2 (custom all-reduce, CUDA graphs) at the floor.

## Speed

To be measured: block FP8 (TP 1) and bf16 (TP 2) prefill / decode on one GH200
at a 41.4B / 3.05B-active shape, with the k / v conv (fused prologue and not),
the earlier q / k / v conv, and no conv (11-06 bench job).
