# opt_moe

A token-flat decoder (`x` is `[T, D]`, document boundaries travel in
`positions`) built on upstream's modules: `GQAttention` with the fused
`QKVLinear`, `DenseFeedForward` (`w13` `[2, F, D]`), `RoutedExperts` (`w13`
`[E, 2, F, D]`, one grouped GEMM). The files follow upstream's model layout:
`gated_swa_attention.py` (`GatedSWAttention`), `moe.py` (`OptMoE`: upstream's
`MoE` plus the routing statistics) and `model.py` (block and model) hold the
modules and their Configs; `flavors.py` builds the complete, sharded config;
`sharding.py` holds the sharding constructors it uses. The optimizer side
(`amphione` and the other DiSCO rules) is `torchtitan/optimizers/readme.md`.

Parallelism: FSDP / HSDP, EP (upstream's dispatchers) and CP (all-gather K/V,
flex or varlen, with the conv, the sinks and the gate; the conv's first
`W - 1` taps of each rank's fragments come from the ranks that hold those
tokens; `tests/unit_tests/opt_moe/test_context_parallel.py`). TP is refused at
build.

## Building a model

`build_model_config(flavor, *, parallelism, seq_len, rope_length, **overrides)`
returns the complete config for that parallelism (nothing changes it
afterwards). In `flavors.py`, `_opt_moe_config` writes the model config (the
embedding, final norm and head; the blocks from `build_opt_moe_layers`) and
holds, with `build_opt_moe_layers`, the default of every setting below; a
flavor is a function passing the settings it changes (`MODEL_FLAVORS`: name ->
(function, default context length), as upstream's models). A recipe's keywords
replace the flavor's setting of the same name; an unknown name raises:

```python
from torchtitan.models.opt_moe import build_model_config

config.model = build_model_config(
    "production-moe-336m", parallelism=config.parallelism, seq_len=4096,
    norm_eps=1e-30, wo_init_std=4.0,
)
```

The recipe's keywords are kept in `config.build_args` (plain values, in the
dumped `job_config.json`): offline evaluation rebuilds the run's model with
`build_model_config(flavor, **build_args)`. The recipe
`torchtitan_recipes/models/opt_moe.py` (`opt_moe_mis_8b()`) is the production
MoE recipe on `mis-8b`.

## Attention

| argument | default | meaning |
|---|---|---|
| `n_heads`, `n_kv_heads`, `head_dim` | - | GQA shape |
| `qk_norm` | - | a norm of the model's `norm_type` on each q and k head, before RoPE |
| `gated_attention_type` | None | None: no gate; `element-wise`: `gate_proj` `[n_heads * head_dim, dim]`, the attention output multiplied by `sigmoid(gate_proj(x))` (fp32) before `wo`. Any other value is refused |
| `attention_sinks` | false | one learned logit per head in the softmax denominator (gpt-oss): `out * sigmoid(lse - s)` |
| `sinks_init_fn_type`, `sinks_init_std` | `zeros`, 0.0 | the sinks' init, any init type of `param_init.py`; `zeros` ignores the std |
| `conv_kernel_size` | 0 | the Canon short conv: `x + conv(x)`, a depthwise causal conv of this width on the streams below, after the projection (attn_gym's CUDA `causal_conv1d`); it restarts at every packed document. 0: no conv, and the conv arguments below do nothing |
| `conv_streams` | `qkv` | `qkv` (q, k and v) or `kv` (q passes through and has no `q_conv`) |
| `conv_init_fn_type`, `conv_init_gain` | `delay`, 4.0 | `delay` (one tap of +-gain per channel), `random_opnorm` (a random filter with operator norm gain) or `torch_default` (gain x the `nn.Conv1d` draw); 4.0 is the recipe's x4 in the init (never run: the study ran 1.0 with its optimizer's step-1 x4) |
| `attn_backend` | - | `flex`, `flex_flash` or `varlen` (upstream's configs, sliding-window variants included); masks are always per document (block causal) |
| `w{qkv,o,_gate}_init_fn_type`, `..._init_std` | `scaled_orthogonal`, 1.0 | the projections' init (q, k and v: one fused `wqkv`) |

The model chooses the attention per layer: `swa_pattern` (`S` sliding window,
`F` full, one character per layer, e.g. `"SSSF" * 4`; None: every layer
sliding-window when `sliding_window_size` is set) with `sliding_window_size`,
and `rope_pattern` (`R` RoPE, `N` NoPE; None: RoPE everywhere). A RoPE layer
takes `rope` (full layers) or `rope_of_swa` (sliding-window layers, when set),
each a dict of `CosSinRoPE.Config` fields (the cache is `seq_len` long, or
`rope_length` when longer); RoPE is partial when that `dim` is below `head_dim`
(the first `dim` head dims rotate, the rest pass through).

## Feed-forward and MoE

- `n_dense_layers`: the first layers are a dense SwiGLU FFN of
  `dense_hidden_dim`, the rest an `OptMoE` (on upstream's routers):
  `num_experts`, `num_shared_experts` (1), `top_k`, `moe_hidden_dim` (each
  expert's), `route_scale` (1.0: the factor on the renormalized top-k
  weights), `score_func` (`sigmoid`);
  balancing by the expert bias (`load_balance_coeff`,
  `bias_update_norm_factor`), by upstream's quantile-balanced router
  (`quantile_balance_bins`, sigmoid scores only) or
  upstream's auxiliary loss
  (`aux_loss=dict(loss_type="microbatch_wise" | "batch_wise", coeff=...)`);
  `router_init_fn_type` / `router_init_std` (`scion_normal_output`, 1.0).
  Tokens go all-to-all under EP (DeepEP / HybridEP: upstream's
  `TokenDispatcherTransform` on the recipe). Upstream's padding mask keeps
  padding tokens out of the balance statistics; the experts still process
  every token.
- `w{1,2,3}_init_fn_type`, `..._init_std` (`scaled_orthogonal`, 1.0): one
  init for the dense FFN, the routed and the shared experts.

## Block and model

| argument | default | meaning |
|---|---|---|
| `norm_type`, `norm_eps` | `rmsnorm`, 1e-30 | every norm (q/k, the pre-norms, the final norm): upstream's `RMSNorm` with learnable gains (`rmsnorm`) or without (`np_rmsnorm`) |
| `depth_init` | `total_depth_R` | depth-scaled init of the output projections: `total_depth_{R,N}`, `relative_depth_{R,N}`, or `identity` (none; the optimizer study's recipe) |
| `residual_scale` | `identity` | `x = identity_scale x + block_scale f(x)`: `depth_scale_{R,N}`, `complete_p_{R,N}`, `complete_p_base<K>` (CompleteP, arXiv 2505.01618) |
| `init_gate_as_residual` | false | the FFN's up projection takes the output projections' depth divisor |
| `prenorm_gain_init` | 1.0 | a constant init of the attention_norm / ffn_norm gains (`rmsnorm` only) |
| `embedding_init_*`, `lm_head_init_*` | `scion_normal_input` / `scion_normal_output`, 1.0 | the embedding's and the head's init |
| `vocab_size` | 201088 | |
| `num_mtp_layers` | 0 | 0 or 1: multi-token prediction (DeepSeek-V3's MTP, `mtp.py` / `OPTMoEMTPBlock`): `mtp_norm(block(eh_proj(cat(enorm(e_{t+1}), hnorm(h_t)))))` predicts token t + 2 through the shared head; the block mirrors the last layer (its full attention, feed-forward / MoE and inits; its own expert seeds); the last layer must be full attention. The run sets `dataloader.num_mtp_layers` and the loss `ChunkedLossWrapper(MTPCrossEntropyLoss(mtp_scale=...))` (0.3: DeepSeek-V3's first phase); the logs add `mtp/main_loss_mean`, `mtp/depth_1_loss_mean`; in eval mode the model skips the MTP depth (validation and offline evaluation score the main prediction). The gains group must match `enorm`, `hnorm`, `mtp_norm` |
| `eh_proj_init_fn_type`, `eh_proj_init_std` | `scaled_orthogonal`, 1.0 | the MTP fusion's init |

Residual statistics: with the optimizer's `log_residual_stats`, the eager
layer loop records the RMS of every block's input and residual write and of
the final norm's input; off, nothing runs.

## Flavors

`mis-8b`, `mis-8b-proxy`, `mis-8b-proxy-qbalance` (quantile balancing, 1000
bins), `mis-30b-a3b`, `mis-100b-a5b`, and the gauge models `gauge-d8-w256`,
`gauge-d16-w256`, `gauge-d16-w512` and `gauge-d16-w1024` (the optimizer
study's 84M and 336M models), used by tests and the study's configs, and
`production-dense-336m` / `production-moe-336m`: the study's production recipe
at 336M (the gauge-d16-w1024 shape; dense, and every FFN an MoE of 64 + 1
experts, top 8).

## Recipes

The optimizer study's production runs: the flavors `production-dense-336m` /
`production-moe-336m` (the model) and the configs
`reproduce_cfgs/configs/0.5.5-debug/production-dense-336m.py` /
`production-moe-336m.py` (the rest), with the gate, sink and conv blocks
commented in them as overrides of the flavor.
In short: qk-norm, `rmsnorm` gains, depth init `total_depth_R`, every hidden
matrix `scaled_orthogonal` at std 4, embedding `scion_normal_input` std 0.02,
head `scion_normal_output` std 1; MoE: 64 routed experts + 1 shared, top 8,
`route_scale` 1.0, sigmoid scores, the bias update (sign, 0.001), router
`scaled_orthogonal` std 4.

## HF export

`state_dict_adapter.py` maps every native tensor to its HF name and back
(fused projections split; the MTP depth is HF layer `L` with DeepSeek-V3's
`enorm` / `hnorm` / `eh_proj` / `shared_head.norm`), and `hf_config` writes the
HF config of a model config (`hf/configuration_opt_moe.py`, `model_type`
`opt_moe`, loaded with `trust_remote_code`). The HF modeling file is not
written yet.

## What the optimizer study measured (13.4B tokens, batch 128, run learning rates)

At 336M and 671M, against the plain model at the same lr: the element-wise
gate -0.021 to -0.025, sinks -0.006 to -0.007, the conv a further -0.004 to
-0.008, a sliding window alone +0.0058 / +0.0037 (worse). At 84M the conv
gained 0.017 (3.3118 against 3.3292).
