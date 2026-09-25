# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`to_hf` must not silently drop tensors, and `from_hf` must invert it.

This is a regression test for a real, silent bug. The 0.5.0 port ("Rebuild OPT MoE
attention on upstream's GQAttention") moved q/k/v under `qkv_linear`, renamed
`output` -> `lm_head`, and moved the experts under `routed_experts.inner_experts`
with shape-suffixed names (`w1_EFD` etc.) -- but `from_hf_map` still held the 0.4.0
names. Every renamed tensor therefore missed the map and `to_hf` dropped it,
announcing 21 tensors in / 10 out behind a root-logger warning that a normal run
never surfaces. An exported checkpoint would have been missing attention and expert
weights entirely.

The check is against the ACTUAL model's parameter names, built on meta, not against a
hand-written list -- a hand-written list would drift with the model exactly the way
the map did.

Run: python tests/unit_tests/opt_moe/test_state_dict_adapter_roundtrip.py
"""

import copy
import re
import sys

import torch

from torchtitan.models.opt_moe import moe_opt_moe_configs
from torchtitan.models.opt_moe.config_registry import moe_template_config
from torchtitan.models.opt_moe.state_dict_adapter import OPTMoEStateDictAdapter

FLAVOR = "bsc-1B-7B-opt-g-proxy-8layers"

# Buffers/statistics that are deliberately not exported. Anything NOT matching these
# must survive to_hf, or it is a silent weight loss.
NON_EXPORTED = (
    "load_balance_loss",
    "tokens_per_expert",
    "router_entropy",
    "acc_fwd_times",
    "expert_bias",
    "freqs_cis",
    "rope",
)


def main() -> int:
    spec = copy.deepcopy(moe_opt_moe_configs[FLAVOR])
    spec.n_layers = 2
    job = moe_template_config()
    job.model = spec
    spec.update_from_config(config=job)
    with torch.device("meta"):
        model = spec.build()

    native = dict(model.state_dict())
    adapter = OPTMoEStateDictAdapter(spec, hf_assets_path=None)

    exportable = {
        k: v for k, v in native.items() if not any(p in k for p in NON_EXPORTED)
    }
    print(
        f"  model state_dict: {len(native)} tensors "
        f"({len(exportable)} expected to export)"
    )

    hf = adapter.to_hf(exportable)
    print(f"  to_hf: {len(exportable)} in -> {len(hf)} out")

    failures = []

    # 1. Nothing silently dropped. Experts fan out (one native stacked tensor becomes
    #    one HF tensor per expert), so out >= in and every input must be accounted for.
    if len(hf) < len(exportable):
        failures.append(f"to_hf DROPPED tensors: {len(exportable)} in, {len(hf)} out")

    # 2. Every native key must be reachable through the map. This is the check that
    #    would have caught the original bug.
    # from_hf_map has None values for HF keys with no native counterpart; drop them
    # so the inverse map is keyed only by real native names.
    to_hf_map = {v: k for k, v in adapter.from_hf_map.items() if v is not None}
    unmapped = []
    for key in adapter._unfuse(exportable):
        abstract = re.sub(r"(\d+)", "{}", key, count=1) if "layers" in key else key
        if abstract not in to_hf_map:
            unmapped.append(key)
    if unmapped:
        failures.append(
            f"{len(unmapped)} native keys have NO entry in from_hf_map: "
            f"{sorted(unmapped)[:6]}"
        )

    # 3. Round trip: from_hf(to_hf(x)) must recover exactly the native key set.
    back = adapter.from_hf(hf)
    missing = set(exportable) - set(back)
    extra = set(back) - set(exportable)
    if missing:
        failures.append(f"round trip LOST {len(missing)} keys: {sorted(missing)[:6]}")
    if extra:
        failures.append(f"round trip INVENTED {len(extra)} keys: {sorted(extra)[:6]}")

    # 4. Shapes preserved through the round trip.
    bad_shape = [
        k
        for k in (set(exportable) & set(back))
        if tuple(exportable[k].shape) != tuple(back[k].shape)
    ]
    if bad_shape:
        failures.append(f"{len(bad_shape)} keys changed shape: {bad_shape[:6]}")

    # 5. The specific 0.5.0 renames must be in the map. Compare against the ABSTRACT
    #    ("{}"-templated) form the map actually stores, not a concrete layer index.
    for abstract_name in (
        "layers.{}.attention.qkv_linear.wq.weight",  # was layers.{}.attention.wq.weight
        "lm_head.weight",  # was output.weight
        "layers.{}.moe.routed_experts.w1_EFD",  # the gate half of the fused w13
        "layers.{}.moe.expert_bias_E",  # was layers.{}.moe.expert_bias
    ):
        if abstract_name not in to_hf_map:
            failures.append(f"0.5.0 rename missing from from_hf_map: {abstract_name}")

    # 6. Values survive the round trip: the fused wqkv / w13 are split into
    #    HF's projections and packed back, so a wrong interleave shows here.
    torch.manual_seed(0)
    real = {k: torch.randn(v.shape) for k, v in exportable.items()}
    real_back = adapter.from_hf(adapter.to_hf(real))
    changed = [k for k in real if not torch.equal(real[k], real_back[k])]
    if changed:
        failures.append(f"round trip CHANGED values of {sorted(changed)[:6]}")

    print(
        f"  round trip: {len(hf)} HF -> {len(back)} native "
        f"(missing={len(missing)}, extra={len(extra)})"
    )

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print("\nstate dict adapter round trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
