# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Project the per-logging-step metric cost at production scale.

Method, deliberately NOT a microbenchmark extrapolated to a different regime:

  1. Build the real flavor on the META device (no memory, no allocation) and
     enumerate its actual parameters -- so the inventory is the model's own,
     not a reconstruction of it.
  2. Route every parameter through DiSCO's own rules (`get_param_type` plus
     the scalar/embed interception in `_build_param_lists`) and shard it the
     way each family does, to get the per-rank matrix inventory.
  3. MEASURE the cost of each distinct shape on this GPU, with warmup and CUDA
     events, then sum over the inventory.

What this can and cannot tell you: it bounds the per-rank *compute*. It says
nothing about collective latency, network contention, or the full-weight
all-to-all at 1024+ GPUs. Treat the totals as a projection of one component,
never as a predicted step time.

Usage:  python analyze_logging_cost.py [flavor ...]
"""
import math
import os
import sys
from collections import Counter, defaultdict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from torchtitan.models.opt_moe import moe_opt_moe_configs  # noqa: E402
from torchtitan.optimizers.norm_helper import (  # noqa: E402
    calculate_norm_batched,
    get_norms_to_log,
)
from torchtitan.optimizers.radial_helper import (  # noqa: E402
    calculate_radial_metrics,
    new_radial_state,
)

DEV = "cuda"
NORMS = get_norms_to_log("all")  # the resolved list, not the "all" alias


def classify(name, shape):
    """DiSCO's routing, mirrored (disco.get_param_type + _build_param_lists)."""
    numel = math.prod(shape)
    if numel == 1:
        return "scalar"
    # embed group == identity backend + embed/unembed norm_factor. In the real
    # configs that is the token embedding, the lm_head, and the MoE routers
    # (confirmed against a live run: embed_params == 2 + n_moe_layers).
    root = name.split(".")[0]
    if root in ("tok_embeddings", "output") or "router.gate" in name:
        return "embed"
    if len(shape) == 3:
        return "expert"  # fsdp_enabled is always true in these configs
    return "fsdp"


def inventory(flavor, dp_shard):
    with torch.device("meta"):
        model = moe_opt_moe_configs[flavor].build()
    fams = defaultdict(list)
    for n, p in model.named_parameters():
        fams[classify(n, tuple(p.shape))].append(tuple(p.shape))

    per_rank = Counter()
    # experts: dim 0 is sharded over the fsdp mesh, ep_per_rank each
    for shp in fams["expert"]:
        n_glob = shp[0]
        ep = math.ceil(n_glob / dp_shard)
        per_rank[(shp[1], shp[2])] += ep
    # fsdp: whole matrices, round-robin param_idx % world_size
    n_fsdp = len(fams["fsdp"])
    for i, shp in enumerate(sorted(fams["fsdp"])):
        if i % dp_shard == 0:  # rank 0's share; every rank owns ~n/world_size
            per_rank[shp] += 1
    # embed: pinned to shard-local rank 0 (see _stores_replicated_norms)
    for shp in fams["embed"]:
        per_rank[shp] += 1
    return fams, per_rank, n_fsdp


def timed(fn, warmup=25, trials=5, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(trials):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    out.sort()
    return out[len(out) // 2], out[0], out[-1]


def bench_shape(m, n, batch):
    """ms for norm+radial over `batch` matrices of [m,n], batched as disco does."""
    W = torch.randn(batch, m, n, device=DEV)
    Wa = W + 0.01 * torch.randn_like(W)
    st = new_radial_state(W.device, shape=(batch,))
    res = {}
    res["norm"] = timed(lambda: calculate_norm_batched(W, NORMS, want_spectrum=False))[
        0
    ]
    res["radial_batched"] = timed(
        lambda: calculate_radial_metrics(W, Wa, st, batch_ndim=1)
    )[0]
    st1 = new_radial_state(W.device)
    res["radial_loop"] = timed(
        lambda: [calculate_radial_metrics(W[i], Wa[i], st1) for i in range(batch)]
    )[0]
    return res


def main(flavors, shards=(64, 128)):
    print(
        f"device: {torch.cuda.get_device_name(0)}   norms_to_log=all ({len(NORMS)} norms)\n"
    )
    for flavor in flavors:
        for dp_shard in shards:
            fams, per_rank, n_fsdp = inventory(flavor, dp_shard)
            tot = sum(per_rank.values())
            print("=" * 74)
            print(f"{flavor}   dp_shard={dp_shard}")
            print(
                f"  params: expert={len(fams['expert'])} fsdp={n_fsdp} "
                f"embed={len(fams['embed'])} scalar={len(fams['scalar'])}"
            )
            print(f"  per-rank matrices to measure: {tot}")
            grand = defaultdict(float)
            for (m, n), cnt in sorted(per_rank.items(), key=lambda kv: -kv[1]):
                b = min(cnt, 64)  # measure a representative batch, scale linearly
                try:
                    r = bench_shape(m, n, b)
                except torch.OutOfMemoryError:
                    print(f"    [{m}x{n}] x{cnt}: OOM at batch {b}, skipped")
                    continue
                scale = cnt / b
                for k, v in r.items():
                    grand[k] += v * scale
                print(
                    f"    [{m:>6} x {n:<6}] x{cnt:<4} "
                    f"norm {r['norm']*scale:8.1f} ms | "
                    f"radial batched {r['radial_batched']*scale:7.1f} ms  "
                    f"loop {r['radial_loop']*scale:8.1f} ms"
                )
                del r
                torch.cuda.empty_cache()
            b_tot = grand["norm"] + grand["radial_batched"]
            l_tot = grand["norm"] + grand["radial_loop"]
            print(f"  ---- per-rank logging compute (projection) ----")
            print(f"    norm                 {grand['norm']:9.1f} ms")
            print(f"    radial  (batched)    {grand['radial_batched']:9.1f} ms")
            print(f"    radial  (per-expert) {grand['radial_loop']:9.1f} ms")
            print(f"    TOTAL   batched      {b_tot:9.1f} ms")
            print(
                f"    TOTAL   before       {l_tot:9.1f} ms"
                f"   -> speedup {l_tot / max(b_tot, 1e-9):.2f}x"
            )
            print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["mis-8b", "qwen30b-a3b"])
