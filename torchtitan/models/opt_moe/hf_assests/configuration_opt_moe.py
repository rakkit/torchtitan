# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import math
from numbers import Real

from transformers.configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)


def _warning_once(message):
    if hasattr(logger, "warning_once"):
        logger.warning_once(message)
    else:
        logger.warning(message)


def _normalize_gated_attention_type(value):
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return value


def _normalize_mid_norm_position(value):
    normalized = value.strip().lower()
    if normalized not in {"after", "before"}:
        raise ValueError(
            "mid_norm_position must be either 'after' or 'before', " f"got {value!r}"
        )
    return normalized


class OptMoEConfig(PretrainedConfig):
    model_type = "OptMoE"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=11008,
        n_shared_experts=1,
        n_active_experts=8,
        n_total_experts=64,
        moe_scaling_factor=2.8232,
        n_dense_layers=0,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-30,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        qk_norm=False,
        mid_norm=False,
        norm_everywhere=False,
        attention_norm_everywhere=None,
        ffn_norm_everywhere=None,
        moe_norm_everywhere=None,
        gated_attention_type=None,
        gate_only=False,
        mid_norm_position="after",
        head_wise_mid_norm=False,
        use_rope=True,
        sliding_window_size=-1,
        qk_rope_dim=None,
        partial_rotary_factor=None,
        rope_pattern=None,
        swa_pattern=None,
        rope_theta_swa=None,
        rope_parameters_swa=None,
        residual_scale="identity",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.n_active_experts = n_active_experts
        self.n_total_experts = n_total_experts
        self.moe_scaling_factor = moe_scaling_factor
        self.n_dense_layers = n_dense_layers
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = (
            head_dim
            if head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )
        self.norm_everywhere = norm_everywhere
        # None means inherit the global flag.
        self.attention_norm_everywhere = (
            norm_everywhere
            if attention_norm_everywhere is None
            else bool(attention_norm_everywhere)
        )
        self.ffn_norm_everywhere = (
            norm_everywhere
            if ffn_norm_everywhere is None
            else bool(ffn_norm_everywhere)
        )
        self.moe_norm_everywhere = (
            self.ffn_norm_everywhere
            if moe_norm_everywhere is None
            else bool(moe_norm_everywhere)
        )
        self.qk_norm = qk_norm
        self.mid_norm = bool(mid_norm)
        self.gated_attention_type = _normalize_gated_attention_type(
            gated_attention_type
        )
        self.gate_only = bool(gate_only)
        self.mid_norm_position = _normalize_mid_norm_position(mid_norm_position)
        self.head_wise_mid_norm = bool(head_wise_mid_norm)
        self.use_rope = use_rope
        self.sliding_window_size = sliding_window_size

        if qk_rope_dim is None:
            if partial_rotary_factor is None:
                qk_rope_dim = self.head_dim
            else:
                qk_rope_dim = int(self.head_dim * partial_rotary_factor)
                if not math.isclose(
                    qk_rope_dim / self.head_dim,
                    partial_rotary_factor,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        "partial_rotary_factor must map to an integer qk_rope_dim. "
                        f"Got head_dim={self.head_dim}, "
                        f"partial_rotary_factor={partial_rotary_factor}."
                    )
        self.qk_rope_dim = int(qk_rope_dim)
        if not (0 < self.qk_rope_dim <= self.head_dim):
            raise ValueError(
                f"qk_rope_dim must be in (0, head_dim], got {self.qk_rope_dim} "
                f"for head_dim={self.head_dim}."
            )

        derived_partial_rotary_factor = self.qk_rope_dim / self.head_dim
        if partial_rotary_factor is not None and not math.isclose(
            partial_rotary_factor,
            derived_partial_rotary_factor,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "qk_rope_dim and partial_rotary_factor disagree. "
                f"Got qk_rope_dim={self.qk_rope_dim}, head_dim={self.head_dim}, "
                f"partial_rotary_factor={partial_rotary_factor}."
            )
        self.partial_rotary_factor = derived_partial_rotary_factor

        if residual_scale not in ("identity", "depth_scale"):
            raise ValueError(
                f"residual_scale must be 'identity' or 'depth_scale', got {residual_scale!r}"
            )
        self.residual_scale = residual_scale

        self.rope_pattern = rope_pattern
        self.swa_pattern = swa_pattern
        self.rope_theta = rope_theta
        self.rope_theta_swa = rope_theta_swa
        self.rope_parameters = rope_parameters
        self.rope_parameters_swa = rope_parameters_swa
        self.standardize_rope_params()
        self.validate_rope()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def standardize_rope_params(self):
        parent_impl = getattr(PretrainedConfig, "standardize_rope_params", None)
        if parent_impl is not None:
            return parent_impl(self)

        rope_theta = getattr(self, "rope_theta", None)
        partial_rotary_factor = getattr(self, "partial_rotary_factor", None)
        rope_parameters = dict(getattr(self, "rope_parameters", None) or {})
        layer_types = getattr(self, "layer_types", None)

        if not (rope_parameters or rope_theta is not None):
            _warning_once(
                "`standardize_rope_params` was called but no RoPE parameters were found."
            )
            return

        if (
            layer_types is None
            or rope_parameters == {}
            or not set(rope_parameters.keys()).issubset(set(layer_types))
        ):
            rope_parameters.setdefault(
                "rope_type", rope_parameters.get("type", "default")
            )
            if rope_theta is not None:
                rope_parameters.setdefault("rope_theta", rope_theta)
            if partial_rotary_factor is not None:
                rope_parameters["partial_rotary_factor"] = partial_rotary_factor

            if rope_parameters["rope_type"] in {"llama3", "yarn", "longrope"}:
                original_max_position_embeddings = getattr(
                    self,
                    "original_max_position_embeddings",
                    self.max_position_embeddings,
                )
                rope_parameters.setdefault(
                    "original_max_position_embeddings",
                    original_max_position_embeddings,
                )
        else:
            for layer_type in set(layer_types):
                layer_rope_parameters = dict(rope_parameters.get(layer_type) or {})
                layer_rope_parameters.setdefault(
                    "rope_type", layer_rope_parameters.get("type", "default")
                )
                if rope_theta is not None:
                    layer_rope_parameters.setdefault("rope_theta", rope_theta)
                if partial_rotary_factor is not None:
                    layer_rope_parameters[
                        "partial_rotary_factor"
                    ] = partial_rotary_factor
                if layer_rope_parameters["rope_type"] in {"llama3", "yarn", "longrope"}:
                    layer_rope_parameters.setdefault(
                        "original_max_position_embeddings",
                        self.max_position_embeddings,
                    )
                rope_parameters[layer_type] = layer_rope_parameters

        self.rope_parameters = rope_parameters

    def validate_rope(self, ignore_keys=None):
        parent_impl = getattr(PretrainedConfig, "validate_rope", None)
        if parent_impl is not None:
            if "ignore_keys" in inspect.signature(parent_impl).parameters:
                return parent_impl(self, ignore_keys=ignore_keys)

            if ignore_keys is None:
                return parent_impl(self)

            previous_ignore_keys = getattr(self, "ignore_keys_at_rope_validation", None)
            self.ignore_keys_at_rope_validation = set(previous_ignore_keys or ()) | set(
                ignore_keys
            )
            try:
                return parent_impl(self)
            finally:
                if previous_ignore_keys is None:
                    self.ignore_keys_at_rope_validation = set()
                else:
                    self.ignore_keys_at_rope_validation = previous_ignore_keys

        rope_parameters_dict = getattr(self, "rope_parameters", None)
        if rope_parameters_dict is None:
            return
        if not isinstance(rope_parameters_dict, dict):
            raise ValueError(
                f"`rope_parameters` must be a dictionary but got {rope_parameters_dict!r}"
            )

        if getattr(self, "layer_types", None) is not None and set(
            rope_parameters_dict.keys()
        ).issubset(set(self.layer_types)):
            rope_parameter_sets = rope_parameters_dict.values()
        else:
            rope_parameter_sets = (rope_parameters_dict,)

        for rope_parameters in rope_parameter_sets:
            rope_type = rope_parameters.get(
                "rope_type", rope_parameters.get("type", "default")
            )
            rope_parameters["rope_type"] = rope_type
            validation_fn = getattr(
                self, f"_validate_{rope_type}_rope_parameters", None
            )
            if validation_fn is None:
                _warning_once(
                    "Missing RoPE validation function for "
                    f"`rope_type`={rope_type!r}; skipping validation."
                )
                continue
            validation_fn(rope_parameters, ignore_keys=ignore_keys)

    @staticmethod
    def _check_received_keys(
        rope_type, received_keys, required_keys, optional_keys=None, ignore_keys=None
    ):
        received_keys = set(received_keys)
        required_keys = set(required_keys)
        optional_keys = set(optional_keys or ())

        if "type" in received_keys:
            received_keys.discard("type")
            required_keys.add("rope_type")

        optional_keys.add("partial_rotary_factor")

        if ignore_keys is not None:
            received_keys -= set(ignore_keys)

        missing_keys = required_keys - received_keys
        if missing_keys:
            raise KeyError(
                "Missing required keys in `rope_parameters` for "
                f"`rope_type`={rope_type!r}: {missing_keys}"
            )

        unused_keys = received_keys - required_keys - optional_keys
        if unused_keys:
            _warning_once(
                "Unrecognized keys in `rope_parameters` for "
                f"`rope_type`={rope_type!r}: {unused_keys}"
            )

    def _validate_default_rope_parameters(self, rope_parameters, ignore_keys=None):
        self._check_received_keys(
            rope_parameters["rope_type"],
            rope_parameters.keys(),
            {"rope_type", "rope_theta"},
            ignore_keys=ignore_keys,
        )

    def _validate_linear_rope_parameters(self, rope_parameters, ignore_keys=None):
        self._check_received_keys(
            rope_parameters["rope_type"],
            rope_parameters.keys(),
            {"rope_type", "factor", "rope_theta"},
            ignore_keys=ignore_keys,
        )
        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, Real) or factor < 1.0:
            _warning_once(
                "`rope_parameters['factor']` must be a real number >= 1, "
                f"got {factor!r}"
            )

    def _validate_dynamic_rope_parameters(self, rope_parameters, ignore_keys=None):
        self._check_received_keys(
            rope_parameters["rope_type"],
            rope_parameters.keys(),
            {"rope_type", "factor"},
            ignore_keys=ignore_keys,
        )
        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, Real) or factor < 1.0:
            _warning_once(
                "`rope_parameters['factor']` must be a real number >= 1, "
                f"got {factor!r}"
            )

    def _validate_yarn_rope_parameters(self, rope_parameters, ignore_keys=None):
        self._check_received_keys(
            rope_parameters["rope_type"],
            rope_parameters.keys(),
            {
                "rope_type",
                "factor",
                "rope_theta",
                "original_max_position_embeddings",
            },
            {
                "attention_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
                "truncate",
            },
            ignore_keys=ignore_keys,
        )
        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, Real) or factor < 1.0:
            _warning_once(
                "`rope_parameters['factor']` must be a real number >= 1, "
                f"got {factor!r}"
            )

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (
            not isinstance(attention_factor, Real) or attention_factor < 0.0
        ):
            _warning_once(
                "`rope_parameters['attention_factor']` must be a real number > 0, "
                f"got {attention_factor!r}"
            )

        beta_fast = rope_parameters.get("beta_fast")
        if beta_fast is not None and not isinstance(beta_fast, Real):
            _warning_once(
                "`rope_parameters['beta_fast']` must be a real number, "
                f"got {beta_fast!r}"
            )

        beta_slow = rope_parameters.get("beta_slow")
        if beta_slow is not None and not isinstance(beta_slow, Real):
            _warning_once(
                "`rope_parameters['beta_slow']` must be a real number, "
                f"got {beta_slow!r}"
            )

        if (beta_fast or 32) < (beta_slow or 1):
            _warning_once(
                "`rope_parameters['beta_fast']` must be >= `beta_slow`; "
                f"got beta_fast={beta_fast!r}, beta_slow={beta_slow!r}"
            )

    def _validate_longrope_rope_parameters(self, rope_parameters, ignore_keys=None):
        self._check_received_keys(
            rope_parameters["rope_type"],
            rope_parameters.keys(),
            {
                "rope_type",
                "short_factor",
                "long_factor",
                "rope_theta",
                "original_max_position_embeddings",
            },
            {"attention_factor", "factor"},
            ignore_keys=ignore_keys,
        )

        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(
            self, "head_dim", self.hidden_size // self.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

        short_factor = rope_parameters.get("short_factor")
        if not (
            isinstance(short_factor, list)
            and all(isinstance(x, Real) for x in short_factor)
        ):
            _warning_once(
                "`rope_parameters['short_factor']` must be a list of real numbers, "
                f"got {short_factor!r}"
            )
        elif len(short_factor) != dim // 2:
            _warning_once(
                "`rope_parameters['short_factor']` must have length "
                f"{dim // 2}, got {len(short_factor)}"
            )

        long_factor = rope_parameters.get("long_factor")
        if not (
            isinstance(long_factor, list)
            and all(isinstance(x, Real) for x in long_factor)
        ):
            _warning_once(
                "`rope_parameters['long_factor']` must be a list of real numbers, "
                f"got {long_factor!r}"
            )
        elif len(long_factor) != dim // 2:
            _warning_once(
                "`rope_parameters['long_factor']` must have length "
                f"{dim // 2}, got {len(long_factor)}"
            )

        factor = rope_parameters.get("factor")
        if factor is not None and (not isinstance(factor, Real) or factor < 1.0):
            _warning_once(
                "`rope_parameters['factor']` must be a real number >= 1, "
                f"got {factor!r}"
            )

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (
            not isinstance(attention_factor, Real) or attention_factor < 0.0
        ):
            _warning_once(
                "`rope_parameters['attention_factor']` must be a real number > 0, "
                f"got {attention_factor!r}"
            )

    def _validate_llama3_rope_parameters(self, rope_parameters, ignore_keys=None):
        self._check_received_keys(
            rope_parameters["rope_type"],
            rope_parameters.keys(),
            {
                "rope_type",
                "factor",
                "original_max_position_embeddings",
                "low_freq_factor",
                "high_freq_factor",
                "rope_theta",
            },
            ignore_keys=ignore_keys,
        )

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, Real) or factor < 1.0:
            _warning_once(
                "`rope_parameters['factor']` must be a real number >= 1, "
                f"got {factor!r}"
            )

        low_freq_factor = rope_parameters["low_freq_factor"]
        if low_freq_factor is None or not isinstance(low_freq_factor, Real):
            _warning_once(
                "`rope_parameters['low_freq_factor']` must be a real number, "
                f"got {low_freq_factor!r}"
            )

        high_freq_factor = rope_parameters["high_freq_factor"]
        if high_freq_factor is None or not isinstance(high_freq_factor, Real):
            _warning_once(
                "`rope_parameters['high_freq_factor']` must be a real number, "
                f"got {high_freq_factor!r}"
            )
        elif high_freq_factor <= low_freq_factor:
            _warning_once(
                "`rope_parameters['high_freq_factor']` must be greater than "
                f"`low_freq_factor`; got high={high_freq_factor!r}, low={low_freq_factor!r}"
            )

        original_max_position_embeddings = rope_parameters[
            "original_max_position_embeddings"
        ]
        if original_max_position_embeddings is None or not isinstance(
            original_max_position_embeddings, int
        ):
            _warning_once(
                "`rope_parameters['original_max_position_embeddings']` must be "
                f"an integer, got {original_max_position_embeddings!r}"
            )
        elif original_max_position_embeddings >= self.max_position_embeddings:
            _warning_once(
                "`rope_parameters['original_max_position_embeddings']` must be less "
                "than `max_position_embeddings`; got "
                f"{original_max_position_embeddings} vs {self.max_position_embeddings}"
            )


__all__ = ["OptMoEConfig"]
