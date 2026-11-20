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
x -> RMSNorm -> qkv_proj -> [q|gate|k|v] -> x + conv(x)   <- conv state: the last W - 1 raw
                                          on k and v         k / v of each sequence
                            -> qk-norm -> RoPE (none on NoPE layers, partial RoPE)
                            -> Attention              <- KV cache (sinks, per-layer
                                                        sliding window)
                            -> * sigmoid(gate) -> o_proj -> residual
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
vLLM reads it before its own folder). It holds configs tuned on GH200 for 256
experts, hidden 2048, top 8: width 512 (TP 1) and 256 (TP 2 shards), bf16 and
block FP8, at 1-16,384 tokens (each token count timed over a pruned config
space under CUDA graphs; bf16 outputs bitwise equal to the default config's,
FP8 within 1.5e-4). Per MoE layer vs the default config: TP 1 bf16 1.00-1.13x,
FP8 1.00-1.18x; TP 2 bf16 1.01-1.15x, FP8 1.00-1.22x. vLLM looks the file up
by the device name: JUPITER's compute nodes report `NVIDIA_GH200_120GB`, its
login nodes `NVIDIA_GH200_480GB`, so each file is there under both names.
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

## Images (`vllm_opt_moe/vl.py`)

A VLM folder (torchtitan's `convert_opt_moe_vl_to_hf.py`: `config.json` names
`OptMoeForConditionalGeneration`, `vision_config` carries SigLIP 2's config and
`spatial_merge_size`, `tile_size`, `max_tiles` and the token ids) is served by
the second class `register()` adds, in the same package and entry point:

```bash
vllm serve <VL folder> --trust-remote-code --dtype bfloat16
vllm serve <VL folder> --trust-remote-code --dtype bfloat16 \
    --limit-mm-per-prompt '{"image": 0}'      # text only: the tower is not built
```

`OptMoeForConditionalGeneration` = vLLM's `SiglipVisionTransformer` (its
parameter names are the folder's `visual.*`, q / k / v stacked at load; the
tower stays bf16 under `--quantization`) + `OptMoeVisionMerger` (`merger.*`: the
HF reference's connector, LayerNorm, 2 x 2 merge in block order, biased MLP with
tanh GELU) + `OptMoeForCausalLM` as `language_model`, unchanged (the text names
`model.*` / `lm_head.*` map onto it with a `WeightsMapper`). The tower and the
merger are built under vLLM's `_mark_tower_model`, so with
`--limit-mm-per-prompt '{"image": 0}'` they are not allocated or loaded and the
engine runs in text-only mode (token ids into the text model).
`--compilation-config '{"compile_mm_encoder": true}'` compiles the tower as a
whole (`OptMoeVisionTower`, vLLM's tower under `support_torch_compile`, the
number of tiles dynamic); off by default.

Serving speed (11-21, one GH200, bf16, the 8B VLM, in-process engine,
`max_num_batched_tokens` 16,384; 1-tile prompts 277 tokens, 13-tile prompts
3,349 tokens):

| 13-tile prompts | batch 1 | batch 16 | batch 64 |
|---|---|---|---|
| tower, ms per tile (eager / `compile_mm_encoder`) | 2.64 / 2.54 | 2.49 / 2.47 | 2.51 / 2.46 |
| prefill, requests / s: images vs the same length in text | 13.0 vs 27.3 | 14.2 vs 48.0 | 14.2 vs 48.9 |
| decode tokens / s: image vs text requests | 474 vs 479 | 4,750 vs 4,475 | 12,215 vs 12,173 |

A lone 1-tile image takes 16.1 ms in the tower (10.2 ms compiled): one tile is
launch-bound. An image costs its tower time once in prefill (46% of a 13-tile
prompt's prefill at batch 64); decode is unchanged. Text served by this class
runs within 3% of the text-only engine (image limit 0). FlashAttention is the
tower's fastest backend (TORCH_SDPA up to 4% slower; FLASHINFER needs varlen
inputs the tower does not pass); `max_num_batched_tokens` 4,096 caps the tower
at 16 tiles per call and lowers image throughput by up to 44% (batched 1-tile
requests). `fp8_per_block` raises decode by 15-31% at batch 16-64 (the tower
stays bf16).

There is no HF processor: `OptMoeMultiModalProcessor` tiles the image on the
frontend's CPU exactly as the reference's `tile_images` (InternVL's dynamic
tiling: the grid of at most `max_tiles` 512-px squares closest to the aspect
ratio, bicubic antialiased resize, a thumbnail when there is more than one
square; `max_tiles // number of images in the prompt` per image) and hands the
uint8 tiles over; the model normalizes to [-1, 1] on the GPU. 256 LLM tokens per
tile (32 x 32 patches, 2 x 2 merge). The prompt holds ONE
`<|image_pad|>` per image, `<|vision_start|><|image_pad|><|vision_end|>` (what
vLLM's chat API inserts for an image part, `get_placeholder_str`); the processor
expands it to the tiles' tokens. The HF reference's `image_prompt` already
expanded text must not be sent. All tiles of the images scheduled in a step go
through the tower as one batch of 1024-patch sequences (no varlen); the
connector's output replaces the pads' embeddings (vLLM's encoder cache keeps it
across chunked-prefill steps).

Checks (11-20, one shared login GPU, bf16, the 8B VLM trial's export; scripts
`vllm_check.py`, `hf_check.py`, `tiling_equal.py` in the user's configs): the
vLLM engine in one process, hooks on the model for the image embeddings and the
logits of every sampled token, against the export's own HF
`OptMoeForConditionalGeneration` (transformers, bf16, sdpa), teacher-forced on
vLLM's tokens. Real DataComp-VLM samples: 1 tile, 3 tiles (2 x 1 + thumbnail),
13 tiles (4 x 3 + thumbnail), 2 images (7 + 3 tiles: the budget split), a 13-tile
image behind 2,200 text tokens (prompt 5,500: the image straddles the 4,096-token
prefill chunk), and 2 text prompts, each sent twice (the second a prefix-cache
hit: all but the last 5-16 tokens cached, the tower does not run again).

| check | result |
|---|---|
| tiles, plugin vs HF `tile_images` | 65 (image size, budget) cases bit-equal after the normalization, equal token counts; the grid rule (vLLM's InternVL helpers) equals HF's `tile_grid` on 120,000 random (size, budget) pairs |
| prompt ids, tiles | the expanded ids equal the HF `image_prompt` ones; the uint8 tiles normalized equal HF's float tiles bit for bit |
| image embeddings, vLLM vs HF bf16 | median abs diff 0.023-0.031, max 1.7-2.3; the noise floor, HF bf16 vs HF fp32 (tower and merger): median 0.031-0.036, max 2.0-2.8 |
| logits, per generated token max abs diff over the vocabulary | median 0.20-0.88 (text-only prompts 0.44-0.84), max 0.5-3.2; argmax agreement 0.97-1.0 (the misses are exact bf16 ties) |
| greedy, up to 64 tokens | identical to HF to the end for 3 of 7 prompts (8-64 tokens), the others diverge at token 63, 44, 5 and 0, each at a bf16 near tie (HF top-2 gap 0.000-0.188) |
| image straddling a prefill chunk | `max_num_batched_tokens` 4,096 vs 8,192: tokens, 64 logit rows and embeddings bit-equal; prefix caching on or off: the first request bit-equal, a request repeated without the cache bit-equal to its first |
| `--limit-mm-per-prompt '{"image": 0}'` | tower and merger not built (0 parameters; 17,058,569,920 vs 17,946,998,432 bytes, the 888,428,512 of the tower and merger); text tokens and logits bit-equal to the full load |
| tower compiled (`compile_mm_encoder`) | one compile of the tower for 1, 10 and 13 tiles; embeddings vs HF fp32: median 0.025-0.028 (bf16 floor 0.031-0.036), vs HF bf16 0.033-0.041, max 2.3-2.9; vs the eager tower median 0.033-0.039; logits vs HF median 0.35-0.78; without the flag bit-equal to the plain tower |
| torch.compile + CUDA graphs (as served) | same prompts (5 of the 7), the text model compiled, piecewise graphs for prefill and full ones for decode, the tower eager: logits against HF per-token max abs diff median 0.41-1.0, max 1.0-3.1, argmax agreement 0.95-1.0; against the eager run (rows up to the first differing token) median 0.31-1.1 |

The text classes (`OptMoeForCausalLM`, MTP) are unchanged by the VL class.
`--quantization fp8_per_block` loads and serves image and text requests (the LLM
part only; the tower and merger stay bf16), its accuracy against HF was not
checked. Not checked: tensor parallelism, speculative decoding, vLLM's OpenAI
server / chat API path (`get_placeholder_str`), image inputs other than PIL.

## Not yet supported

- other `scoring_func`s than sigmoid.

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

One GH200 node, vLLM 1.0.0.dev20261030, CUDA graphs, random-token prompts
(prefix caching off), 257 generated tokens; median of 3 fresh processes
(`bench_vllm_decode.py` in the user's configs). Block FP8 =
`--quantization fp8_per_block` at TP 1; bf16 at TP 2. Decode in tokens / s
over the whole engine, prefill in prompt tokens / s.

The conv's cost (41.4B total / 3.05B active: 40 layers, 256 experts top 6 of
width 640, gate):

| attention prologue | FP8 decode B 1 / B 256 (1k prompt) | FP8 prefill B 1 (8k) | bf16 decode B 1 / B 256 | bf16 prefill B 1 (8k) |
|---|---|---|---|---|
| no conv | 245 / 10,264 | 65,497 | 264 / 11,820 | 75,827 |
| k / v conv, fused prologue (this plugin) | 243 / 10,187 | 62,689 | 261 / 11,720 | 73,277 |
| k / v conv, unfused | 233 / 10,130 | 64,034 | 248 / 11,666 | 73,703 |
| q / k / v conv (removed) | 214 / 9,934 | 59,884 | 232 / 11,512 | 64,849 |

The k / v conv with the fused prologue costs ~1% of decode and ~3-4% of long
prefill.

The element-wise gate's rows ride in `qkv_proj` (one GEMM; per TP rank
`[q | gate | k | v]`). Gate vs sinks (44 layers, q 32, 256 experts top 8 of
width 512, k / v conv, block FP8 TP 1, 3 alternated runs): decode B 1 238 vs 247
(-3%), B 16 1,801 vs 1,824 (-1%), B 256 10,898 vs 11,094 (-2%), 32k prompts B 1
229 vs 235; prefill 70k vs 75k tok/s at 1k (-7%). With a separate gate GEMM
decode B 1 was 218.

Two candidate shapes (k / v conv, gate): 40 layers, 256 experts top 8 of
width 512 (l40-k8-f512) is faster than 48 layers of width 384 (l48-k8-f384) in
every cell: block FP8 decode B 1 243 vs 208 (+17%), B 256 11,368 vs 10,715
(+6%), 32k-token prompts B 16 1,478 vs 1,342 (+10%); prefill +8 to +21%; bf16
TP 2 decode B 256 13,620 vs 13,184 (+3%).

DeepGEMM (block FP8, l40-k8-f512, TP 1): for the linears
(`VLLM_USE_DEEP_GEMM=1 VLLM_MOE_USE_DEEP_GEMM=0`) prefill +5 to +7%, decode
B 16 +7 to +9%, B 1 -2%; for the MoE as well (vLLM's defaults once DeepGEMM
imports) prefill +18 to +22% but decode -14 to -24%: serve with
`VLLM_MOE_USE_DEEP_GEMM=0`.
