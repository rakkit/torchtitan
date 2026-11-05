# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generation-correctness test for opt_moe_plugins.

Supports two models, selected via VLLM_TEST_MODEL_ARCH:
  - StagingMoEllamaForCausalLM  (default)
  - OptMoEForCausalLM

Environment variables:
  VLLM_TEST_MODEL_ARCH             Model architecture to test (see above)
  VLLM_TEST_MODEL_PATH             Path to model checkpoint (default: ./test-sft)
  VLLM_TEST_PROMPTS                '|||'-separated list of prompts
  VLLM_TEST_MAX_NEW_TOKENS         Number of tokens to generate (default: 16)
  VLLM_TEST_NUM_LOGPROBS           Top-K logprobs to compare (default: 20)
  VLLM_TEST_TP_SIZE                Tensor-parallel size (default: 1)
  VLLM_TEST_DTYPE                  bfloat16 | float16 | float32 (default: bfloat16)
  VLLM_TEST_SEED                   Random seed (default: 0)
  VLLM_TEST_GPU_MEMORY_UTILIZATION GPU memory fraction (default: 0.9)
  VLLM_TEST_ENFORCE_EAGER          Disable CUDA graphs (default: 1)
  VLLM_TEST_DISABLE_CUDAGRAPH      Disable CUDA graph compilation (default: 1)
  VLLM_TEST_REQUIRE_EXACT          Require byte-exact output match (default: 0)
  VLLM_TEST_HARD_LOGPROBS          Strict logprob delta check (default: same as REQUIRE_EXACT)
  VLLM_TEST_SAMPLED_TOKEN_LOGPROB_ATOL   Logprob tolerance (default: 0.05)
  VLLM_TEST_REFERENCE_BACKEND      vllm_transformers | hf (default: vllm_transformers)
  VLLM_TEST_DISABLE_TF32           Disable TF32 matmul/cudnn (default: 1)
  VLLM_TEST_DETERMINISTIC_ALGORITHMS  torch.use_deterministic_algorithms (default: 0)
  VLLM_TEST_FP32_EMBED_LM_HEAD     Cast embed/lm_head to fp32 (default: 0)
  VLLM_TEST_FIX_MISTRAL_REGEX      Fix Mistral tokenizer regex (default: 1)
  VLLM_TEST_DISABLE_V1_MULTIPROCESSING  Reduce ZMQ noise (default: 1)
  VLLM_TEST_DISABLE_VLLM_PLUGINS   Isolate plugin registration (default: 1)
  VLLM_TEST_DISABLE_PREFIX_CACHING Disable prefix cache (default: 1)
"""

import contextlib
import gc
import os
import unittest
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from vllm import LLM, ModelRegistry, SamplingParams


# ---------------------------------------------------------------------------
# Model selection — must happen before any model-specific import
# ---------------------------------------------------------------------------

_ARCH_STAGING = "StagingMoEllamaForCausalLM"
_ARCH_OPT_MOE = "OptMoEForCausalLM"
_SUPPORTED_ARCHS = {_ARCH_STAGING, _ARCH_OPT_MOE}

_SELECTED_ARCH = os.environ.get("VLLM_TEST_MODEL_ARCH", _ARCH_STAGING).strip()
if _SELECTED_ARCH not in _SUPPORTED_ARCHS:
    raise ValueError(
        f"VLLM_TEST_MODEL_ARCH={_SELECTED_ARCH!r} is not supported. "
        f"Choose one of: {sorted(_SUPPORTED_ARCHS)}"
    )

if _SELECTED_ARCH == _ARCH_STAGING:
    from opt_moe_plugins.vllm_staging_moellama import MODEL_ARCH, register as _register

    _PLUGIN_MODULE_PATH = "opt_moe_plugins.vllm_staging_moellama.model"
else:
    from opt_moe_plugins.vllm_opt_moe import MODEL_ARCH, register as _register

    _PLUGIN_MODULE_PATH = "opt_moe_plugins.vllm_opt_moe.model"


warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"Unclosed context <zmq\.Context",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"builtin type swigvarlink has no __module__ attribute",
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TokensText = tuple[list[int], str]
TokensTextLogprobs = tuple[list[int], str, list[dict[int, float]] | None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _format_token_id_preview(token_ids: list[int], max_items: int = 64) -> str:
    if len(token_ids) <= max_items:
        return str(token_ids)
    preview = token_ids[:max_items]
    return f"{preview} ... (total={len(token_ids)})"


@dataclass
class TestConfig:
    model_path: str
    prompts: list[str]
    max_new_tokens: int
    num_logprobs: int
    dtype: str
    seed: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    enforce_eager: bool
    disable_cudagraph: bool
    fix_mistral_regex: bool
    require_exact: bool
    hard_logprobs: bool
    sampled_token_logprob_atol: float
    eos_token_ids: list[int] | None
    pad_token_id: int | None
    disable_v1_multiprocessing: bool
    disable_vllm_plugins: bool
    disable_prefix_caching: bool
    reference_backend: str
    disable_tf32: bool
    deterministic_algorithms: bool
    fp32_embed_lm_head: bool


def _load_generation_ids(model_path: str) -> tuple[list[int] | None, int | None]:
    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path)
    except Exception:
        return None, None

    eos = gen_cfg.eos_token_id
    if eos is None:
        eos_ids = None
    elif isinstance(eos, int):
        eos_ids = [int(eos)]
    else:
        eos_ids = [int(x) for x in eos]

    pad = gen_cfg.pad_token_id
    pad_id = int(pad) if pad is not None else None
    return eos_ids, pad_id


def _load_config_from_env() -> TestConfig:
    model_path = os.environ.get("VLLM_TEST_MODEL_PATH", "./test-sft")
    prompt_str = os.environ.get(
        "VLLM_TEST_PROMPTS",
        "Hello, my name is|||The capital of France is|||The future of AI is",
    )
    prompts = [p for p in prompt_str.split("|||") if p]
    if not prompts:
        raise ValueError("VLLM_TEST_PROMPTS must contain at least one prompt.")

    eos_token_ids, pad_token_id = _load_generation_ids(model_path)
    require_exact = _env_bool("VLLM_TEST_REQUIRE_EXACT", False)
    hard_logprobs = _env_bool("VLLM_TEST_HARD_LOGPROBS", require_exact)

    sampled_token_logprob_atol = float(
        os.environ.get("VLLM_TEST_SAMPLED_TOKEN_LOGPROB_ATOL", "0.05")
    )
    max_new_tokens = int(os.environ.get("VLLM_TEST_MAX_NEW_TOKENS", "16"))
    num_logprobs = int(os.environ.get("VLLM_TEST_NUM_LOGPROBS", "20"))

    if max_new_tokens <= 0:
        raise ValueError("VLLM_TEST_MAX_NEW_TOKENS must be > 0.")
    if num_logprobs <= 0:
        raise ValueError("VLLM_TEST_NUM_LOGPROBS must be > 0.")
    if sampled_token_logprob_atol <= 0:
        raise ValueError("VLLM_TEST_SAMPLED_TOKEN_LOGPROB_ATOL must be > 0.")

    reference_backend = (
        os.environ.get("VLLM_TEST_REFERENCE_BACKEND", "vllm_transformers")
        .strip()
        .lower()
    )
    valid_reference_backends = {"vllm_transformers", "hf"}
    if reference_backend not in valid_reference_backends:
        raise ValueError(
            "VLLM_TEST_REFERENCE_BACKEND must be one of "
            f"{sorted(valid_reference_backends)}, got {reference_backend!r}."
        )

    return TestConfig(
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        num_logprobs=num_logprobs,
        dtype=os.environ.get("VLLM_TEST_DTYPE", "bfloat16"),
        seed=int(os.environ.get("VLLM_TEST_SEED", "0")),
        tensor_parallel_size=int(os.environ.get("VLLM_TEST_TP_SIZE", "1")),
        gpu_memory_utilization=float(
            os.environ.get("VLLM_TEST_GPU_MEMORY_UTILIZATION", "0.9")
        ),
        enforce_eager=_env_bool("VLLM_TEST_ENFORCE_EAGER", True),
        disable_cudagraph=_env_bool("VLLM_TEST_DISABLE_CUDAGRAPH", True),
        fix_mistral_regex=_env_bool("VLLM_TEST_FIX_MISTRAL_REGEX", True),
        require_exact=require_exact,
        hard_logprobs=hard_logprobs,
        sampled_token_logprob_atol=sampled_token_logprob_atol,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
        disable_v1_multiprocessing=_env_bool(
            "VLLM_TEST_DISABLE_V1_MULTIPROCESSING", True
        ),
        disable_vllm_plugins=_env_bool("VLLM_TEST_DISABLE_VLLM_PLUGINS", True),
        disable_prefix_caching=_env_bool("VLLM_TEST_DISABLE_PREFIX_CACHING", True),
        reference_backend=reference_backend,
        disable_tf32=_env_bool("VLLM_TEST_DISABLE_TF32", True),
        deterministic_algorithms=_env_bool("VLLM_TEST_DETERMINISTIC_ALGORITHMS", False),
        fp32_embed_lm_head=_env_bool("VLLM_TEST_FP32_EMBED_LM_HEAD", False),
    )


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported dtype {dtype_name!r}; use one of {sorted(mapping)}"
        )
    return mapping[dtype_name]


def _load_hf_tokenizer(config: TestConfig):
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if config.fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True
    try:
        return AutoTokenizer.from_pretrained(config.model_path, **tokenizer_kwargs)
    except TypeError as err:
        if "fix_mistral_regex" in str(err):
            tokenizer_kwargs.pop("fix_mistral_regex", None)
            return AutoTokenizer.from_pretrained(config.model_path, **tokenizer_kwargs)
        raise


def _encode_prompts(tokenizer, prompts: list[str]) -> list[list[int]]:
    return [
        list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in prompts
    ]


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _effective_max_model_len(
    config: TestConfig, prompt_token_ids: list[list[int]]
) -> int:
    prompt_max_len = max((len(ids) for ids in prompt_token_ids), default=0)
    return max(1, prompt_max_len + config.max_new_tokens)


def _effective_num_logprobs(config: TestConfig, tokenizer) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        try:
            vocab_size = int(len(tokenizer))
        except Exception:
            vocab_size = config.num_logprobs
    return max(1, min(config.num_logprobs, vocab_size))


def _resolve_special_token_ids(
    config: TestConfig,
    tokenizer,
) -> tuple[list[int] | None, int | None]:
    eos_ids: list[int] = []
    if config.eos_token_ids is not None:
        eos_ids.extend(int(tok) for tok in config.eos_token_ids)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        eos_ids.append(int(tokenizer_eos))
    resolved_eos = sorted(set(eos_ids)) if eos_ids else None

    pad_id = config.pad_token_id
    tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None and tokenizer_pad is not None:
        pad_id = int(tokenizer_pad)
    return resolved_eos, pad_id


def _assert_shared_setup(config: TestConfig, prompt_token_ids: list[list[int]]) -> None:
    mismatches: list[str] = []
    if not prompt_token_ids:
        mismatches.append("no prompts were encoded")
    for idx, token_ids in enumerate(prompt_token_ids):
        if not token_ids:
            mismatches.append(f"prompt {idx} encoded to an empty token list")
    try:
        _dtype_from_name(config.dtype)
    except ValueError as err:
        mismatches.append(str(err))
    if mismatches:
        raise AssertionError("Invalid shared HF/vLLM setup:\n" + "\n".join(mismatches))


def _print_shared_setup(config: TestConfig, prompt_token_ids: list[list[int]]) -> None:
    prompt_lens = [len(ids) for ids in prompt_token_ids]
    print(
        f"Model arch: {_SELECTED_ARCH} | "
        f"dtype={config.dtype}, seed={config.seed}, "
        f"max_new_tokens={config.max_new_tokens}, "
        f"num_logprobs={config.num_logprobs}, "
        f"prompt_lens={prompt_lens}, "
        f"reference_backend={config.reference_backend}"
    )


def _configure_torch_runtime(config: TestConfig) -> None:
    if config.disable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    if config.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def _build_vllm_sampling_params(
    config: TestConfig,
    num_logprobs: int,
) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=config.max_new_tokens,
        logprobs=num_logprobs,
        seed=config.seed,
        detokenize=False,
        stop_token_ids=config.eos_token_ids,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


def _convert_vllm_logprobs(logprobs: Any) -> list[dict[int, float]] | None:
    if logprobs is None:
        return None
    converted: list[dict[int, float]] = []
    for elem in logprobs:
        converted_elem: dict[int, float] = {}
        for token_id, item in elem.items():
            if hasattr(item, "logprob"):
                converted_elem[int(token_id)] = float(item.logprob)
            else:
                converted_elem[int(token_id)] = float(item)
        converted.append(converted_elem)
    return converted


def _is_gpu_utilization_startup_error(err: Exception) -> bool:
    msg = str(err)
    return "desired GPU memory utilization" in msg and "Free memory on device" in msg


def _gpu_memory_utilization_candidates(requested: float) -> list[float]:
    candidates: list[float] = []

    def _add(val: float) -> None:
        clipped = max(0.05, min(0.99, float(val)))
        if all(abs(clipped - x) > 1e-6 for x in candidates):
            candidates.append(clipped)

    _add(requested)
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_ratio = free_bytes / max(1, total_bytes)
            _add(free_ratio * 0.9)
            _add(free_ratio * 0.8)
        except Exception:
            pass

    for val in (0.5, 0.35, 0.25, 0.2, 0.15, 0.1):
        _add(val)
    return candidates


def _hard_cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _shutdown_llm(llm: LLM) -> None:
    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is not None:
        engine_core = getattr(llm_engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()


def _canonicalize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        converted = {str(k): _canonicalize_config_value(v) for k, v in value.items()}
        if "type" in converted and "rope_type" not in converted:
            converted["rope_type"] = converted["type"]
        if "attention_factor" in converted and "attn_factor" not in converted:
            converted["attn_factor"] = converted["attention_factor"]
        if "attn_factor" in converted and "attention_factor" not in converted:
            converted["attention_factor"] = converted["attn_factor"]
        return {k: converted[k] for k in sorted(converted)}
    if isinstance(value, list):
        return [_canonicalize_config_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_config_value(v) for v in value)
    if isinstance(value, float):
        return float(value)
    return value


def _resolve_rope_config(config_obj: Any) -> Any:
    rope_merged: dict[str, Any] = {}
    rope_parameters = getattr(config_obj, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        if rope_parameters and all(
            isinstance(v, dict) for v in rope_parameters.values()
        ):
            if "" in rope_parameters:
                rope_merged.update(dict(rope_parameters[""]))
            elif len(rope_parameters) == 1:
                rope_merged.update(dict(next(iter(rope_parameters.values()))))
            else:
                return _canonicalize_config_value(rope_parameters)
        else:
            rope_merged.update(dict(rope_parameters))
    rope_theta = getattr(config_obj, "rope_theta", None)
    if rope_theta is not None and "rope_theta" not in rope_merged:
        rope_merged["rope_theta"] = rope_theta
    if not rope_merged:
        return None
    return _canonicalize_config_value(rope_merged)


def _assert_vllm_runtime_matches_kwargs(llm: LLM, llm_kwargs: dict[str, Any]) -> None:
    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is None:
        return
    vllm_config = getattr(llm_engine, "vllm_config", None)
    if vllm_config is None:
        return

    mismatches: list[str] = []
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is not None:
        expected_max_model_len = llm_kwargs.get("max_model_len")
        actual_max_model_len = getattr(model_config, "max_model_len", None)
        if (
            isinstance(expected_max_model_len, int)
            and isinstance(actual_max_model_len, int)
            and actual_max_model_len != expected_max_model_len
        ):
            mismatches.append(
                f"max_model_len: expected={expected_max_model_len!r}, actual={actual_max_model_len!r}"
            )

    if mismatches:
        raise RuntimeError(
            "vLLM runtime options differ from requested setup.\n"
            + "\n".join(mismatches)
        )


def _assert_vllm_tokenizer_parity(llm: LLM, hf_tokenizer: Any) -> None:
    try:
        vllm_tokenizer = llm.get_tokenizer()
    except Exception:
        return
    mismatches: list[str] = []
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        hf_val = getattr(hf_tokenizer, field, None)
        vllm_val = getattr(vllm_tokenizer, field, None)
        if hf_val != vllm_val:
            mismatches.append(f"{field}: hf={hf_val!r}, vllm={vllm_val!r}")
    if mismatches:
        raise RuntimeError(
            "HF/vLLM tokenizer special-token config mismatch.\n" + "\n".join(mismatches)
        )


def _assert_vllm_loaded_expected_hf_config(llm: LLM, model_path: str) -> None:
    try:
        expected_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return

    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is None:
        return
    model_config = getattr(llm_engine, "model_config", None)
    actual_cfg = getattr(model_config, "hf_config", None)
    if actual_cfg is None:
        return

    fields = [
        "model_type",
        "hidden_size",
        "head_dim",
        "max_position_embeddings",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "attention_bias",
        "attention_dropout",
        "mlp_bias",
        "hidden_act",
        "rms_norm_eps",
        "rope_theta",
        "n_dense_layers",
        "moe_intermediate_size",
        "moe_scaling_factor",
        "n_total_experts",
        "n_active_experts",
        "n_shared_experts",
        "norm_everywhere",
        "qk_norm",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        # opt_moe-specific optional fields
        "gated_attention_type",
        "gate_only",
        "mid_norm_position",
        "use_rope",
        "sliding_window_size",
        "rope_pattern",
        "swa_pattern",
        "rope_theta_swa",
        "qk_rope_dim",
        "partial_rotary_factor",
    ]
    mismatches: list[str] = []
    for field in fields:
        expected_val = getattr(expected_cfg, field, None)
        actual_val = getattr(actual_cfg, field, None)
        if expected_val != actual_val:
            mismatches.append(
                f"{field}: expected={expected_val!r}, actual={actual_val!r}"
            )

    expected_rope = _resolve_rope_config(expected_cfg)
    actual_rope = _resolve_rope_config(actual_cfg)
    if expected_rope != actual_rope:
        mismatches.append(
            f"rope_config: expected={expected_rope!r}, actual={actual_rope!r}"
        )

    if expected_rope is not None:
        print(f"Resolved rope config: {expected_rope}")

    if mismatches:
        raise RuntimeError(
            "vLLM loaded a config that differs from HF reference.\n"
            + "\n".join(mismatches)
        )


@contextlib.contextmanager
def _temp_env(name: str, value: str):
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def run_vllm_greedy(
    config: TestConfig,
    prompt_token_ids: list[list[int]],
    hf_tokenizer,
    *,
    model_impl: str,
    check_custom_registration: bool,
    enforce_eager_override: bool | None = None,
    disable_cudagraph_override: bool | None = None,
    disable_prefix_caching_override: bool | None = None,
    gpu_memory_utilization_override: float | None = None,
) -> list[TokensTextLogprobs]:
    if check_custom_registration:
        _register()
    num_logprobs = _effective_num_logprobs(config, hf_tokenizer)

    enforce_eager = (
        config.enforce_eager
        if enforce_eager_override is None
        else enforce_eager_override
    )
    disable_cudagraph = (
        config.disable_cudagraph
        if disable_cudagraph_override is None
        else disable_cudagraph_override
    )
    disable_prefix_caching = (
        config.disable_prefix_caching
        if disable_prefix_caching_override is None
        else disable_prefix_caching_override
    )
    requested_gpu_utilization = (
        config.gpu_memory_utilization
        if gpu_memory_utilization_override is None
        else float(gpu_memory_utilization_override)
    )

    llm_kwargs: dict[str, Any] = {
        "model": config.model_path,
        "tokenizer": config.model_path,
        "tokenizer_mode": "hf",
        "generation_config": "vllm",
        "trust_remote_code": True,
        "dtype": config.dtype,
        "seed": config.seed,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": requested_gpu_utilization,
        "max_model_len": _effective_max_model_len(config, prompt_token_ids),
        "max_logprobs": max(20, num_logprobs),
        "model_impl": model_impl,
        "enable_prefix_caching": (not disable_prefix_caching),
        "enforce_eager": enforce_eager,
        "skip_tokenizer_init": False,
    }
    if config.disable_v1_multiprocessing:
        llm_kwargs["distributed_executor_backend"] = "uni"
    if disable_cudagraph:
        llm_kwargs["compilation_config"] = {"cudagraph_mode": "NONE"}

    maybe_disable_mp = (
        _temp_env("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if config.disable_v1_multiprocessing
        else contextlib.nullcontext()
    )
    maybe_disable_plugins = (
        _temp_env("VLLM_PLUGINS", "")
        if config.disable_vllm_plugins
        else contextlib.nullcontext()
    )

    with maybe_disable_mp, maybe_disable_plugins:
        if check_custom_registration:
            _register()
            registered = ModelRegistry.models.get(MODEL_ARCH)
            module_name = getattr(registered, "module_name", "")
            if _PLUGIN_MODULE_PATH not in str(module_name):
                raise RuntimeError(
                    f"Stale {MODEL_ARCH} registration detected. "
                    f"Expected module path containing '{_PLUGIN_MODULE_PATH}', "
                    f"got: {module_name!r}."
                )

        llm: LLM | None = None
        tried_utils: list[float] = []
        last_mem_err: Exception | None = None
        for gpu_util in _gpu_memory_utilization_candidates(requested_gpu_utilization):
            llm_kwargs["gpu_memory_utilization"] = gpu_util
            tried_utils.append(gpu_util)
            try:
                llm = LLM(**llm_kwargs)
                if abs(gpu_util - requested_gpu_utilization) > 1e-6:
                    print(
                        f"Adjusted gpu_memory_utilization: "
                        f"requested={requested_gpu_utilization}, used={gpu_util}."
                    )
                break
            except ValueError as err:
                if not _is_gpu_utilization_startup_error(err):
                    raise
                last_mem_err = err
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if llm is None:
            tried_str = ", ".join(f"{x:.3f}" for x in tried_utils)
            raise RuntimeError(
                "Unable to start vLLM due to GPU memory reservation checks. "
                f"Tried gpu_memory_utilization values: [{tried_str}].\n"
                f"Last error: {last_mem_err}"
            )

        try:
            _assert_vllm_runtime_matches_kwargs(llm, llm_kwargs)
            _assert_vllm_loaded_expected_hf_config(llm, config.model_path)
            _assert_vllm_tokenizer_parity(llm, hf_tokenizer)

            sampling_params = _build_vllm_sampling_params(config, num_logprobs)
            prompts = [
                {"prompt_token_ids": token_ids} for token_ids in prompt_token_ids
            ]
            req_outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            outputs: list[TokensTextLogprobs] = []
            for expected_prompt_ids, req_output in zip(
                prompt_token_ids, req_outputs, strict=True
            ):
                used_prompt_ids = getattr(req_output, "prompt_token_ids", None)
                if used_prompt_ids is None:
                    raise AssertionError(
                        "vLLM RequestOutput.prompt_token_ids is None; cannot verify prompt parity."
                    )
                if list(used_prompt_ids) != list(expected_prompt_ids):
                    raise AssertionError(
                        "vLLM prompt token IDs diverged from expected input IDs.\n"
                        f"expected={_format_token_id_preview(list(expected_prompt_ids))}\n"
                        f"used={_format_token_id_preview(list(used_prompt_ids))}"
                    )
                if not req_output.outputs:
                    raise AssertionError("vLLM returned empty outputs list.")
                sample = req_output.outputs[0]
                if len(sample.token_ids) == 0:
                    raise AssertionError("vLLM generated zero new tokens.")

                token_ids = list(sample.token_ids)
                outputs.append(
                    (
                        token_ids,
                        _decode_token_ids(hf_tokenizer, token_ids),
                        _convert_vllm_logprobs(sample.logprobs),
                    )
                )
            return outputs
        finally:
            _shutdown_llm(llm)
            del llm
            _hard_cuda_cleanup()


def run_hf_greedy(
    config: TestConfig,
    prompt_token_ids: list[list[int]],
    tokenizer,
) -> list[TokensTextLogprobs]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(config.dtype)
    num_logprobs = _effective_num_logprobs(config, tokenizer)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": (dtype if device.type == "cuda" else torch.float32),
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = dtype if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)

    model.to(device)
    model.eval()
    eos_set = set(config.eos_token_ids or [])

    outputs: list[TokensTextLogprobs] = []
    with torch.inference_mode():
        for prompt_ids in prompt_token_ids:
            generated_ids: list[int] = []
            output_logprobs: list[dict[int, float]] = []
            running_ids = list(prompt_ids)

            for _ in range(config.max_new_tokens):
                input_ids = torch.tensor([running_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                step_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                step_scores = step_out.logits[0, -1, :]
                step_logprobs = torch.log_softmax(
                    step_scores, dim=-1, dtype=torch.float32
                )
                topk = min(num_logprobs, step_logprobs.shape[0])
                top_values, top_indices = torch.topk(step_logprobs, k=topk)
                lp_dict = {
                    int(tok): float(lp)
                    for tok, lp in zip(top_indices.tolist(), top_values.tolist())
                }
                next_token_id = int(torch.argmax(step_logprobs).item())
                if next_token_id not in lp_dict:
                    lp_dict[next_token_id] = float(step_logprobs[next_token_id].item())
                output_logprobs.append(lp_dict)
                generated_ids.append(next_token_id)
                running_ids.append(next_token_id)
                if eos_set and next_token_id in eos_set:
                    break

            outputs.append(
                (
                    generated_ids,
                    _decode_token_ids(tokenizer, generated_ids),
                    output_logprobs,
                )
            )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def run_reference_greedy(
    config: TestConfig,
    prompt_token_ids: list[list[int]],
    tokenizer,
    *,
    gpu_memory_utilization_override: float | None = None,
) -> tuple[str, list[TokensTextLogprobs]]:
    if config.reference_backend == "vllm_transformers":
        return (
            "vllm_transformers",
            run_vllm_greedy(
                config,
                prompt_token_ids,
                tokenizer,
                model_impl="transformers",
                check_custom_registration=False,
                enforce_eager_override=True,
                disable_cudagraph_override=True,
                disable_prefix_caching_override=True,
                gpu_memory_utilization_override=gpu_memory_utilization_override,
            ),
        )
    if config.reference_backend == "hf":
        return ("hf", run_hf_greedy(config, prompt_token_ids, tokenizer))
    raise ValueError(f"Unsupported reference backend: {config.reference_backend!r}")


def check_logprobs_close(
    *,
    outputs_0_lst: list[TokensTextLogprobs],
    outputs_1_lst: list[TokensTextLogprobs],
    name_0: str,
    name_1: str,
    hard_mode: bool,
    sampled_token_logprob_atol: float,
) -> None:
    assert len(outputs_0_lst) == len(outputs_1_lst)
    for prompt_idx, (outputs_0, outputs_1) in enumerate(
        zip(outputs_0_lst, outputs_1_lst)
    ):
        output_ids_0, output_str_0, logprobs_0 = outputs_0
        output_ids_1, output_str_1, logprobs_1 = outputs_1

        if logprobs_0 is None:
            logprobs_0 = [dict() for _ in output_ids_0]
        if logprobs_1 is None:
            logprobs_1 = [dict() for _ in output_ids_1]

        if hard_mode:
            fail_msg = (
                f"Test{prompt_idx}:"
                f"\n{name_0}_ids:\t{_format_token_id_preview(output_ids_0)}"
                f"\n{name_1}_ids:\t{_format_token_id_preview(output_ids_1)}"
            )
            assert len(output_ids_0) == len(output_ids_1), fail_msg

        for idx, (output_id_0, output_id_1) in enumerate(
            zip(output_ids_0, output_ids_1)
        ):
            fail_msg = (
                f"Test{prompt_idx}:"
                f"\nMatched tokens:\t{_format_token_id_preview(output_ids_0[:idx])}"
                f"\n{name_0}:\t{output_str_0!r}\t{logprobs_0[idx]}"
                f"\n{name_1}:\t{output_str_1!r}\t{logprobs_1[idx]}"
            )
            assert idx < len(logprobs_0), fail_msg
            assert idx < len(logprobs_1), fail_msg
            assert output_id_0 in logprobs_1[idx], fail_msg
            assert output_id_1 in logprobs_0[idx], fail_msg

            if hard_mode:
                assert output_id_0 == output_id_1, fail_msg
                sampled_lp_0 = float(logprobs_0[idx][output_id_0])
                sampled_lp_1 = float(logprobs_1[idx][output_id_0])
                lp_delta = abs(sampled_lp_0 - sampled_lp_1)
                assert lp_delta <= sampled_token_logprob_atol, (
                    f"{fail_msg}\n"
                    f"sampled_token_id={output_id_0}, "
                    f"{name_0}_logprob={sampled_lp_0:.6f}, "
                    f"{name_1}_logprob={sampled_lp_1:.6f}, "
                    f"abs_delta={lp_delta:.6f}, "
                    f"atol={sampled_token_logprob_atol:.6f}"
                )

            if output_id_0 != output_id_1:
                break


# ---------------------------------------------------------------------------
# Test class (model-agnostic — works for both staging_moellama and opt_moe)
# ---------------------------------------------------------------------------


class TestModelGenerativeCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "CUDA is required for this model-size correctness test."
            )

        cls.config = _load_config_from_env()
        _configure_torch_runtime(cls.config)
        cls.tokenizer = _load_hf_tokenizer(cls.config)
        cls.config.eos_token_ids, cls.config.pad_token_id = _resolve_special_token_ids(
            cls.config,
            cls.tokenizer,
        )
        cls.prompt_token_ids = _encode_prompts(cls.tokenizer, cls.config.prompts)
        _assert_shared_setup(cls.config, cls.prompt_token_ids)
        _print_shared_setup(cls.config, cls.prompt_token_ids)

        force_matched_engine_flags = cls.config.reference_backend == "vllm_transformers"
        paired_gpu_util: float | None = None
        if force_matched_engine_flags:
            paired_gpu_util = min(cls.config.gpu_memory_utilization, 0.5)
            print(
                "Forcing matched vLLM engine flags for custom/transformers parity: "
                "enable_prefix_caching=False, enforce_eager=True, cudagraph_mode=NONE."
            )

        cls.vllm_outputs = run_vllm_greedy(
            cls.config,
            cls.prompt_token_ids,
            cls.tokenizer,
            model_impl="vllm",
            check_custom_registration=True,
            enforce_eager_override=(True if force_matched_engine_flags else None),
            disable_cudagraph_override=(True if force_matched_engine_flags else None),
            disable_prefix_caching_override=(
                True if force_matched_engine_flags else None
            ),
            gpu_memory_utilization_override=paired_gpu_util,
        )
        _hard_cuda_cleanup()

        cls.reference_name, cls.reference_outputs = run_reference_greedy(
            cls.config,
            cls.prompt_token_ids,
            cls.tokenizer,
            gpu_memory_utilization_override=paired_gpu_util,
        )
        _hard_cuda_cleanup()

    def test_logprobs_similarity(self):
        """vLLM custom model logprobs are close to reference."""
        check_logprobs_close(
            outputs_0_lst=self.reference_outputs,
            outputs_1_lst=self.vllm_outputs,
            name_0=self.reference_name,
            name_1="vllm_custom",
            hard_mode=self.config.hard_logprobs,
            sampled_token_logprob_atol=self.config.sampled_token_logprob_atol,
        )

    def test_exact_outputs(self):
        """vLLM custom model produces byte-exact output (informative, gated by VLLM_TEST_REQUIRE_EXACT)."""
        if not self.config.require_exact:
            self.skipTest(
                "Exact-output matching is disabled. "
                "Set VLLM_TEST_REQUIRE_EXACT=1 to enable."
            )
        for prompt_idx, (ref_out, vllm_out) in enumerate(
            zip(self.reference_outputs, self.vllm_outputs)
        ):
            ref_ids, ref_str, _ = ref_out
            vllm_ids, vllm_str, _ = vllm_out
            fail_msg = (
                f"Prompt {prompt_idx}:"
                f"\n{self.reference_name}:\t{ref_str!r}"
                f"\nvllm_custom:\t{vllm_str!r}"
                f"\n{self.reference_name}_ids:\t{ref_ids}"
                f"\nvllm_custom_ids:\t{vllm_ids}"
            )
            self.assertEqual(ref_str, vllm_str, fail_msg)
            self.assertEqual(ref_ids, vllm_ids, fail_msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
