# opt_moe

A token-flat decoder (`x` is `[T, D]`, document boundaries travel in
`positions`) built on upstream's modules: `GQAttention` with the fused
`QKVLinear`, `FeedForward` (`w13` `[2, F, D]`), `RoutedExperts` (`w13`
`[E, 2, F, D]`, one grouped GEMM). Everything below is a config field of
`OPTMoEModel.Config`, its `layer` template (`OPTMoETransformerBlock.Config`)
or the template's `attention`, `feed_forward` and `moe`. The optimizer side
(`amphione` and the other DiSCO rules) is `torchtitan/optimizers/readme.md`.

## Attention (`GatedNormSWAttention.Config`)

| field | default | meaning |
|---|---|---|
| `qk_norm` | false | a norm of `norm_type` on each q and k head, before RoPE |
| `norm_type`, `norm_eps` | `rmsnorm` (learnable gains), 1e-30 | the qk-norm; any type of `utils/norms.py` |
| `gated_attention_type` | None | None: no gate; `element-wise`: `gate_proj` `[n_heads * head_dim, dim]`, the attention output multiplied by `sigmoid(gate_proj(x))` (fp32) before `wo`. Any other value is refused at build |
| `attention_sinks` | false | one learned logit per head in the softmax denominator (gpt-oss): `out * sigmoid(lse - s)`, `utils/attention_sinks.py` |
| `sinks_init_fn_type`, `sinks_init_std` | `zeros`, 0.0 | the sinks' init, any type of the init helper (`utils/inits.py`); `zeros` ignores the std |
| `conv_kernel_size` | 0 | the Canon short conv: `x + conv(x)`, a depthwise causal conv of this width on the streams below, after the projection; it restarts at every packed document. 0: no conv, and the conv fields below do nothing |
| `conv_streams` | `qkv` | `qkv` (q, k and v) or `kv` (q passes through and has no `q_conv`) |
| `conv_init_fn_type`, `conv_init_gain` | `delay`, 4.0 | `delay` (one tap of +-gain per channel), `random_opnorm` (a random filter with operator norm gain) or `torch_default` (gain x the `nn.Conv1d` draw); 4.0 is the recipe's x4 in the init (never run: the study ran 1.0 with its optimizer's step-1 x4) |
| `sliding_window_size` | -1 | the window of the sliding-window layers (`swa_pattern` below) |
| `attn_backend` | `flex` | `flex`, `flex_flash` or `varlen`; masks are always per document (block causal) |
| `w{q,k,v,o,_gate}_init_fn_type`, `..._init_std` | `scaled_orthogonal`, 1.0 | the projections' init; q, k and v must match (one fused `wqkv`) |

The model chooses the attention per layer: `swa_pattern` (`S` sliding window,
`F` full, one character per layer, e.g. `"SSSF" * 4`) and `rope_pattern` (`R`
RoPE, `N` NoPE; None: RoPE everywhere). A RoPE layer takes `rope` (full layers)
or `rope_of_swa` (sliding-window layers, when set); RoPE is partial when that
config's `dim` is below `head_dim` (the first `dim` head dims rotate, the rest
pass through), so full and sliding-window layers may differ in theta and in
rotated width.

## Feed-forward and MoE

- `FeedForward.Config`: `hidden_dim`, `activation_fn` (SwiGLU), the `w1`,
  `w2`, `w3` init types and stds.
- `MoE.Config`: `num_experts`, `num_shared_experts`, `top_k`, `hidden_dim`
  (each expert's), `scaling_factor` (None computes it from experts and top-k),
  `score_func`; balancing by the expert bias (`load_balance_coeff`,
  `bias_update_norm_factor`), by quantile balancing (`quantile_balance_bins`,
  sigmoid scores only) or an auxiliary loss (`aux_loss`); the router's and the
  experts' init types and stds; `moe_comm_backend` (`standard`, `deepep`,
  `hybridep`).
- `layer.n_dense_layers`: the first layers keep the dense `feed_forward`.

## Block and model

| field | default | meaning |
|---|---|---|
| `norm_type`, `norm_eps` (model and `layer`) | `rmsnorm`, 1e-30 | the pre-norms and the final norm |
| `layer.depth_init` | `total_depth` | depth-scaled init of the output projections: `total_depth_{R,N}`, `relative_depth_{R,N}`, or `identity` (none; the optimizer study's recipe) |
| `layer.residual_scale` | `identity` | `x = identity_scale x + block_scale f(x)`: `depth_scale_{R,N}`, `complete_p_{R,N}`, `complete_p_base<K>` (CompleteP, arXiv 2505.01618) |
| `layer.init_gate_as_residual` | false | the FFN's up projection takes the output projections' depth divisor |
| `layer.prenorm_gain_init` | 1.0 | a constant init of the attention_norm / ffn_norm gains (`rmsnorm` only) |
| `first_in_init_*`, `final_out_init_*` | `scion_normal_input` / `scion_normal_output`, 1.0 | the embedding's and the head's init |
| `use_embeddings_norm` | false | a norm after the embedding |
| `enable_token_mask_for_moe` | false | padding (and prompt) tokens, whose labels are ignored, do not count toward MoE balance (SFT with first-fit packing) |

Residual statistics: with the optimizer's `log_residual_stats`, the eager
layer loop records the RMS of every block's input and residual write and of
the final norm's input; off, nothing runs.

## Flavors

`mis-8b`, `mis-8b-proxy`, `mis-8b-proxy-qbalance` (quantile balancing, 1000
bins), `mis-30b-a3b`, `mis-100b-a5b`, and the gauge models `gauge-d8-w256`,
`gauge-d16-w256`, `gauge-d16-w512` and `gauge-d16-w1024` (the optimizer
study's 84M and 336M models), used by tests and the study's configs.
`moe_template_config()` starts from `mis-8b`. A run config edits a flat spec
and builds the complete model config for its parallelism last:

```python
m = model_registry("gauge-d16-w512")   # the flat spec: `layer` template + patterns
m.layer.attention.conv_kernel_size = 4  # ... every setting written out
config.model = build_model_config(m, parallelism=config.parallelism, seq_len=4096)
```

`build_model_config` expands the template per layer (patterns, depth init,
inits), sets each component's sharding for that parallelism and the CP inner
attentions when CP is on; nothing changes the model config afterwards
(upstream's rule).

## Recipes

The optimizer study's production settings, written out as full configs:
`reproduce_cfgs/configs/0.5.0-optimizer-v2/0.5.0-optimizer-v2-final/`
`final-3-production-336m.py` (dense) and `final-moe-recipe-repro-336m.py`
(MoE), with the gate, sink and conv blocks commented in them. In short:
qk-norm, `rmsnorm` gains, no depth scaling, every hidden matrix
`scaled_orthogonal` at std 4, embedding `scion_normal_input` std 0.02, head
`scion_normal_output` std 1; MoE: 64 routed experts + 1 shared, top 8,
`scaling_factor` 1.0, sigmoid scores, the bias update (sign, 0.001), router
`scion_normal_output` std 4.

## HF export

`hf_assests/setup_hf.py` exports parameter-free norms only and no conv (it
refuses both); the export is rewritten for this model in a later step of the
`llm-0.5.4` plan.

## What the optimizer study measured (13.4B tokens, batch 128, run learning rates)

At 336M and 671M, against the plain model at the same lr: the element-wise
gate -0.021 to -0.025, sinks -0.006 to -0.007, the conv a further -0.004 to
-0.008, a sliding window alone +0.0058 / +0.0037 (worse). At 84M the conv
gained 0.017 (3.3118 against 3.3292).
