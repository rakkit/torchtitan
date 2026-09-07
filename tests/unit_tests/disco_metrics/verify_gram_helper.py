# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gram_helper: the sync-free rewrites must be numerically identical.

`_dominance_ratio` and `_offdiag_stats` were rewritten to avoid boolean-mask
indexing, which runs a data-dependent `nonzero()` and therefore forces a
device->host sync on a path that is already dispatch-bound. The rewrites must
match the originals exactly, especially in the two regimes those helpers exist
to handle: all-zero input, and strong diagonal dominance (where the file's own
comments warn about catastrophic cancellation).

Reference implementations below are the pre-change code, kept verbatim.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import torchtitan.optimizers.gram_helper as G  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def ref_dominance_ratio(diagonal, off_mean):
    eps_rel = torch.finfo(diagonal.dtype).eps
    denominator = torch.maximum(off_mean, G._relative_floor(diagonal, eps_rel))
    result = torch.zeros_like(diagonal)
    nonzero = diagonal > 0
    result[nonzero] = diagonal[nonzero] / denominator[nonzero]
    return result


def ref_offdiag_stats(C, m):
    off_mask = ~torch.eye(m, dtype=torch.bool, device=C.device)
    x = C[off_mask]
    ax = x.abs()
    return ax.mean(), x.square().mean().sqrt(), ax.max()


def rel(a, b):
    return (a - b).abs().max().item() / max(1e-30, a.abs().max().item())


print("== _offdiag_stats and _dominance_ratio match the pre-change code ==")
torch.manual_seed(0)
CASES = ("random", "all-zero", "diag-dominant", "single-row", "negative")
for m in (2, 8, 64, 768):
    for case in CASES:
        C = torch.randn(m, m, device=dev)
        C = C + C.T
        if case == "all-zero":
            C = C * 0
        elif case == "diag-dominant":
            C.fill_diagonal_(1e6)
        elif case == "single-row":
            C = C * 0
            C[0, :] = 3.0
        elif case == "negative":
            C = -C.abs()
        a, b = ref_offdiag_stats(C, m), G._offdiag_stats(C, m)
        d = max(rel(x, y) for x, y in zip(a, b))
        check(f"_offdiag_stats m={m:<4} {case}", d < 1e-6, f"rel {d:.2e}")

        diag = torch.diagonal(C).abs().clone()
        for off in (
            torch.rand(m, device=dev),
            torch.zeros(m, device=dev),
            torch.full((m,), 1e30, device=dev),
        ):
            p, q = ref_dominance_ratio(diag, off), G._dominance_ratio(diag, off)
            check(
                f"_dominance_ratio m={m:<4} {case}",
                rel(p, q) < 1e-6 and torch.isfinite(q).all(),
                f"rel {rel(p, q):.2e}",
            )

print("\n== calculate_gram_metrics: key sets and finiteness by level ==")
W = torch.randn(768, 2048, device=dev)
V = torch.randn_like(W)
Wa = W + 0.01 * torch.randn_like(W)
for lvl in (0, 1, 2, 3):
    out = G.calculate_gram_metrics(W, V, Wa, level=lvl)
    exp_s, exp_v = set(G.gram_scalar_names(lvl)), set(G.gram_vector_names(lvl))
    got_s = {k for k, v in out.items() if v.numel() == 1}
    got_v = {k for k, v in out.items() if v.numel() > 1}
    ok = (
        got_s == exp_s
        and got_v == exp_v
        and all(torch.isfinite(v).all() for v in out.values())
    )
    check(f"level {lvl}: {len(got_s)} scalars + {len(got_v)} vectors, finite", ok)

print("\n== degenerate inputs must not produce NaN/Inf ==")
for label, (a, b, c) in {
    "all zero": (torch.zeros(64, 128, device=dev),) * 3,
    "zero update": (W[:64, :128], V[:64, :128], W[:64, :128].clone()),
    "zero V_raw": (W[:64, :128], torch.zeros(64, 128, device=dev), Wa[:64, :128]),
}.items():
    out = G.calculate_gram_metrics(a, b, c, level=1)
    check(f"{label}", all(torch.isfinite(v).all() for v in out.values()))

print()
if fails:
    print(f"FAILED {len(fails)}: {sorted(set(fails))}")
    sys.exit(1)
print("ALL GRAM HELPER CHECKS PASSED")
