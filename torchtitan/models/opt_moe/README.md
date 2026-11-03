# opt_moe: options added by the optimizer study

The optimizer study (branch `optimizer-study`) added the model options below
to opt_moe, on top of `llm-0.5.3`. Every one is off, or at the value that
leaves the model unchanged, by default. The optimizer side (the `amphione`
rule and the rest) is in `torchtitan/optimizers/readme.md`.

"Ablation" marks an option the study used only to test an idea; the study's
Amphione recipe (dense, qk-norm on) uses none of them.

## Attention (`GatedNormSWAttention.Config`)

**Short convolution.** `conv_kernel_size > 0` (already on `llm-0.5.3`) puts a
depthwise causal conv of that width after the projection. It masks across
document boundaries. With the conv on, three choices are required (no
defaults) and must be absent without it:

| field | values | meaning |
|---|---|---|
| `conv_format` | `kda`, `canon` | `kda` (KDA / Gated DeltaNet): `silu(conv(x))` replaces x. `canon` (Canon / Inkling): `conv(x) + x`, no activation |
| `conv_streams` | `qkv`, `kv` | a conv on q, k and v, or on k and v only (q passes through and has no `q_conv`) |
| `conv_init_fn_type`, `conv_init_gain` | `delay`, `random_opnorm`, `torch_default`; gain > 0 | `delay`: one tap of +-gain per channel (position and sign uniform), so every channel's operator norm is gain. `random_opnorm`: a random filter scaled to that operator norm. `torch_default`: gain x nn.Conv1d's default draw |

The study's conv setting: `conv_kernel_size: 4`, `canon`, `qkv`, `delay` at
gain 1, the conv weights in their own optimizer group (`match:
(^|\.)attention\.(q|k|v)_conv\.weight$`) on the hidden rule with `row-l2` as
the direction (`rule: amphione`, `compute: row-l2`, `row_direction: true`;
see "Row computes and `row_direction`" in `torchtitan/optimizers/readme.md`),
the same lr, and the row lengths x4 at step 1 (`radius: 4`); `amphione` has
no step-1 rescale, so its counterpart is `conv_init_gain: 4` (not run).
`random_opnorm`, `torch_default` and `kv` are ablations.

**Sinks.** `attention_sinks: true` (already on `llm-0.5.3`) adds one learned
logit per head to the softmax denominator: the output is
`out * sigmoid(lse - s)` (`utils/attention_sinks.py`, a copy of gpt-oss's
rescale so opt_moe imports no gpt-oss code). `sink_init` (added) sets their
start: None = trunc_normal with std 0.02 (gpt-oss's std without its depth
scaling), 0.0 = zeros; it needs `attention_sinks`. `sink_init` is an
ablation.

**Gates** (`gated_attention_type`: `head-wise`, `element-wise`) and the
**sliding window** (`sliding_window_size`, `swa_pattern`) were already on
`llm-0.5.3`.

**Row multipliers** (ablation). `row_multipliers`: learned per-row output
multipliers, init 1, `W_eff = Diag(m) W` (arXiv 2601.04890), on any of `q`,
`k`, `v` (those rows of wqkv) and `o` (wo). `scalar_multipliers: true` makes
each one a single scalar per projection instead of one per row.

## Feed-forward (`FeedForward.Config`)

- `row_multipliers` on `w1` (gate), `w3` (up), `w2` (the residual write) and
  `scalar_multipliers`, as in the attention (ablation).
- `swish_beta: true`: one learned scalar beta per layer (shape [1], init 1)
  on the gate channel, Swish beta (Ramachandran et al.) (ablation).
- `models/common/activation.py` adds `ReGLU` (`relu(gate) * up`), whose gate
  scale is an exact gauge (ablation).

## Block and model

- `post_norm_type` (default `"none"`): a sandwich norm, one norm of that type
  on each branch output before the residual add (`post_attention_norm`,
  `post_ffn_norm`). Ablation; the HF export refuses it.
- `prenorm_gain_init` (default 1.0): a constant init of the attention_norm /
  ffn_norm gains (`norm_type: rmsnorm` only). Ablation.
- `residual_scale: complete_p_base<K>`: CompleteP's residual, alpha 1, base
  depth K (arXiv 2505.01618 Table 1): `h = x + (K / L) f(x)`. Ablation.
- Model sizes `gauge-d16-w1024`, `gauge-d16-w1024-vnorm`, `gauge-d32-w1024`,
  `gauge-d16-w2048` (dense, qk-norm, head_dim 128): the study's gauge models.
- Residual statistics: with the optimizer's `log_residual_stats`, the model's
  eager layer loop records the RMS of every block's input and residual write
  and of the final norm's input; off, nothing runs.
- The RoPE caches cover `validator.fixed_seq_len` when it is longer than the
  training context.

## HF export

`hf_assests/setup_hf.py` exports only the `kda` conv on q, k and v, and no
sandwich norm; any other setting raises. The study's `canon` conv does not
export yet.

## What the study measured (13.4B tokens, batch 128, run learning rates)

At 336M and 671M, against the plain model at the same lr: the element-wise
gate -0.021 to -0.025, sinks -0.006 to -0.007, the conv a further -0.004 to
-0.008, a sliding window alone +0.0058 / +0.0037 (worse).
At 84M the conv gained 0.017 (3.3118 against 3.3292). The sandwich norm and
the multipliers did not help (Part 2 of the study, 84M).

## Pruning candidates (nothing removed yet)

For a clean production branch. None of these is used by the study's Amphione
recipe (dense, qk-norm on). Removing an option also removes its tests and
any HF-export branch for it; the study's configs that set it keep running
on their frozen trees and tags.

**Added by the study; most can go:**

| option | where |
|---|---|
| `row_multipliers`, `scalar_multipliers` | attention, feed-forward |
| `swish_beta` | feed-forward |
| `post_norm_type`, `post_attention_norm`, `post_ffn_norm` (sandwich norm) | block |
| `prenorm_gain_init` | block |
| `residual_scale: complete_p_base<K>` (CompleteP) | `utils/inits.py` |
| `ReGLU` | `models/common/activation.py` |
| `gauge-d16-w1024`, `gauge-d16-w1024-vnorm`, `gauge-d32-w1024`, `gauge-d16-w2048` | model sizes |
| `sink_init` | attention |
| `conv_init_fn_type` `random_opnorm` / `torch_default`, `conv_streams` `kv` | attention |

**Already on `llm-0.5.3`; some can go** (check the other production configs
first): `v_norm`, `mid_norm` with `mid_norm_position` and
`head_wise_mid_norm`, `norm_everywhere`, `gate_only`, `init_gate_as_residual`,
`residual_div`, `use_embeddings_norm`, `qk_rope_dim`, the `depth_init`
variants.

**Keep:** the conv with both formats (`kda` until the HF export takes
`canon`), the gates, the sinks and the sliding window (measured above).
