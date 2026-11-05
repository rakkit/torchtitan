# opt_moe_plugins

Out-of-tree vLLM model plugins for:

| Architecture | Class | Plugin module |
|---|---|---|
| StagingMoEllama | `StagingMoEllamaForCausalLM` | `vllm_staging_moellama` |
| OptMoE | `OptMoEForCausalLM` | `vllm_opt_moe` |

## Install

```bash
pip install -e ./opt_moe_plugins
```

Both plugins are registered as vLLM entry points via `vllm.general_plugins` in `setup.py`.

---

## OptMoE features

The `vllm_opt_moe` plugin extends the staging plugin with:

| Feature | Config field | Example |
|---|---|---|
| Per-layer NoPE/RoPE | `rope_pattern` | `"RRRN"` — last layer has no RoPE |
| Per-layer SWA | `swa_pattern` | `"SSFF"` — first 2 layers use sliding-window attention |
| Independent SWA theta | `rope_theta_swa` | `500000.0` |
| Independent SWA rope_parameters | `rope_parameters_swa` | `{...}` |
| SWA window size | `sliding_window_size` | `4096` |
| Head-wise gated attention | `gated_attention_type` | `"head-wise"` |
| Element-wise gated attention | `gated_attention_type` | `"element-wise"` |
| Gate-only attention | `gate_only` | `true` |
| Attention mid-norm before/after gate | `mid_norm_position` | `"before"` |
| QK norm | `qk_norm` | `true` |
| Mid-norm in FFN/experts | `norm_everywhere` | `true` |
| Shared (non-routed) experts | `n_shared_experts` | `1` |
| Expert bias routing | `expert_bias` buffer | (loaded from checkpoint) |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_OPT_MOE_DISABLE_FUSED` | `0` | Force exact (non-fused) MoE path |
| `VLLM_OPT_MOE_DISABLE_FUSED_TOPK_BIAS` | `0` | Disable GPU-fused top-k with bias routing |
| `VLLM_OPT_MOE_FP32_EMBED_LM_HEAD` | `0` | Cast embedding and LM head to fp32 |

---

## Unit tests (no GPU)

```bash
bash opt_moe_plugins/unit_test.sh
# or directly:
python -m pytest opt_moe_plugins/tests/unit -v
```

## Generation correctness test (GPU required)

Compares vLLM custom backend vs HF reference (or vLLM-transformers backend).

```bash
# StagingMoEllama (default)
VLLM_TEST_MODEL_PATH=./my-staging-model python opt_moe_plugins/test_generation_correctness.py

# OptMoE
VLLM_TEST_MODEL_ARCH=OptMoEForCausalLM \
VLLM_TEST_MODEL_PATH=./my-opt-moe-model \
python opt_moe_plugins/test_generation_correctness.py
```

Key env vars for the correctness test:

| Variable | Default | Description |
|---|---|---|
| `VLLM_TEST_MODEL_ARCH` | `StagingMoEllamaForCausalLM` | Architecture to test |
| `VLLM_TEST_MODEL_PATH` | `./test-sft` | Path to HF checkpoint |
| `VLLM_TEST_PROMPTS` | (3 defaults) | `\|\|\|`-separated prompts |
| `VLLM_TEST_MAX_NEW_TOKENS` | `16` | Generation length |
| `VLLM_TEST_TP_SIZE` | `1` | Tensor-parallel size |
| `VLLM_TEST_DTYPE` | `bfloat16` | Computation dtype |
| `VLLM_TEST_REFERENCE_BACKEND` | `vllm_transformers` | `vllm_transformers` or `hf` |
| `VLLM_TEST_REQUIRE_EXACT` | `0` | Require byte-exact output match |
| `VLLM_TEST_HARD_LOGPROBS` | `0` | Strict per-token logprob delta check |

---

## Package structure

```
opt_moe_plugins/
├── setup.py                           # Registers both entry points
├── test_unit.py                       # Backward-compatible wrapper for legacy path
├── test_generation_correctness.py     # Correctness test (GPU)
├── unit_test.sh
├── tests/unit/
│   ├── shared_stubs.py                # Shared dummy modules for unit tests
│   ├── staging/
│   │   └── test_staging_moellama_plugin.py
│   └── opt_moe/
│       ├── test_opt_moe_plugin.py
│       └── test_state_dict_adapter_compat.py
├── vllm_staging_moellama/             # BC: staging_moellama plugin (self-contained)
│   ├── __init__.py
│   ├── model.py
│   └── norm_everywhere_fused_moe.py
└── vllm_opt_moe/                      # New: opt_moe plugin (fully independent)
    ├── __init__.py
    ├── model.py
    └── norm_everywhere_fused_moe.py   # Independent copy (no cross-imports)
```
