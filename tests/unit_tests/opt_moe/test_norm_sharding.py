# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Norms built INSIDE a module must still receive a sharding plan.

`attention_norm`, `ffn_norm` and the root `norm` are config fields, so
`sharding.py` reaches them directly; the attention's `qk_norm` takes its plan
where `build_attention_config` builds it. `embeddings_norm` is built inside the
model, so its plan is threaded in via `norm_sharding_config`. With a PARAMETRIC
norm type those
weights would otherwise stay plain tensors and `fully_shard` rejects them under
`spmd_backend="spmd_types"`.

Run: python tests/unit_tests/opt_moe/test_norm_sharding.py
"""

import dataclasses

import spmd_types as spmd
import torch

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.opt_moe.gated_norm_swattention import (
    AttentionSpec,
    build_attention_config,
)
from torchtitan.models.opt_moe.model import OPTMoEModel
from torchtitan.models.opt_moe.utils.norms import (
    ALL_NORM_PARAM_NAMES,
    build_norm_config,
    norm_has_parameters,
    NORM_PARAM_NAMES,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.base.sharding import ShardingConfig

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:52} {'PASS' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def _is_norm(mod) -> bool:
    """True for the norm modules whose parameters the plan must cover."""
    return type(mod).__name__ in ("RMSNorm", "LayerNorm", "SingleScaleRMSNorm")


def plan():
    return ShardingConfig(
        state_shardings={
            n: dense_param_placement(tp=spmd.R) for n in ALL_NORM_PARAM_NAMES
        }
    )


def main():
    print("norm parameter registry")
    # The registry must match what the modules actually build, or the plan
    # names states that do not exist (inert) or misses ones that do (breaks).
    for norm_type, expected in NORM_PARAM_NAMES.items():
        mod = build_norm_config(norm_type, 32, 1e-6).build()
        actual = tuple(n for n, _ in mod.named_parameters(recurse=False))
        check(
            f"{norm_type}: declared == actual parameters",
            set(actual) == set(expected),
            f"{actual} vs {expected}",
        )
        check(
            f"{norm_type}: norm_has_parameters agrees",
            norm_has_parameters(norm_type) == bool(actual),
        )

    print("parameter-free norms are never stamped")
    for norm_type in NORM_PARAM_NAMES:
        cfg = build_norm_config(norm_type, 32, 1e-6, sharding_config=plan())
        attached = cfg.sharding_config is not None
        check(
            f"{norm_type}: stamped == has parameters",
            attached == norm_has_parameters(norm_type),
            f"attached={attached}",
        )

    print("the model config declares the threading field (embeddings_norm)")
    names = [f.name for f in dataclasses.fields(OPTMoEModel.Config)]
    check(
        "OPTMoEModel.Config declares norm_sharding_config",
        "norm_sharding_config" in names,
    )

    print("every internal attention norm receives a plan for all its parameters")
    for norm_type in ("np_rmsnorm", "rmsnorm", "layernorm", "ss_rmsnorm"):
        cfg = build_attention_config(
            AttentionSpec(n_heads=4, n_kv_heads=4, norm_type=norm_type, qk_norm=True),
            dim=64,
            rope=CosSinRoPE.Config(dim=16, max_context_length=128),
            sliding_window_size=-1,
            residual_div=1.0,
            enable_sp=False,
        )
        with torch.device("meta"):
            attn = cfg.build()
        missing, covered = [], 0
        for mod_name, mod in attn.named_modules():
            if not mod_name.endswith(("q_norm", "k_norm")):
                continue
            sc = getattr(mod, "_sharding_config", None)
            for pname, _ in mod.named_parameters(recurse=False):
                if sc is None or pname not in sc.state_shardings:
                    missing.append(f"{mod_name}.{pname}")
                else:
                    covered += 1
        expected = len(NORM_PARAM_NAMES[norm_type]) * 2  # q, k
        check(
            f"{norm_type}: all internal norm params covered",
            not missing and covered == expected,
            f"covered={covered} expected={expected} missing={missing}",
        )

    # WHOLE-MODEL walk. The per-attention check above cannot see
    # `embeddings_norm` (model.py) -- dropping the threading there left the
    # suite green. Build the real model and assert EVERY parameter of EVERY norm has
    # a declared placement, which is the property the sharding actually needs.
    print("whole-model walk: every norm parameter has a declared placement")
    import copy

    from torchtitan.models.opt_moe import moe_opt_moe_configs

    for flavor, norm_type, embeddings_norm in (
        ("gauge-d16-w512", "rmsnorm", True),  # exercises embeddings_norm
        ("mis-8b-proxy", "rmsnorm", False),  # an MoE model
    ):
        if flavor not in moe_opt_moe_configs:
            check(f"{flavor}: present in registry", False, "flavor missing")
            continue
        spec = copy.deepcopy(moe_opt_moe_configs[flavor])
        spec.n_layers = 2
        spec.swa_pattern = (
            spec.rope_pattern
        ) = None  # the flavor's patterns are per layer
        spec.norm_type = norm_type
        spec.layer.attention.norm_type = norm_type
        spec.layer.attention.qk_norm = True
        spec.use_embeddings_norm = embeddings_norm
        # `finalize` populates tok_embeddings / layers and sets the sharding.
        spec.finalize(parallelism=ParallelismConfig(), max_context_length=256)
        with torch.device("meta"):
            model = spec.build()
        uncovered = []
        for mod_name, mod in model.named_modules():
            params = list(mod.named_parameters(recurse=False))
            if not params or not _is_norm(mod):
                continue
            sc = getattr(mod, "_sharding_config", None)
            for pname, _ in params:
                if sc is None or pname not in sc.state_shardings:
                    uncovered.append(f"{mod_name}.{pname}")
        check(
            f"{flavor}: all norm params have a placement",
            not uncovered,
            f"uncovered={uncovered[:4]}",
        )

    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()


def test_a_model_built_for_one_layout_refuses_another():
    """build_model_config shards for a parallelism; overriding EP / CP / TP / SP
    afterwards (a command-line override) would otherwise be ignored by the model."""
    import pytest

    from torchtitan.models.opt_moe import build_model_config, model_registry

    cfg = build_model_config(
        model_registry("gauge-d8-w256"), parallelism=ParallelismConfig(), seq_len=256
    )
    with torch.device("meta"):
        model = cfg.build()
    with pytest.raises(ValueError, match="built for"):
        model.parallelize(parallelism=ParallelismConfig(expert_parallel_degree=2))
