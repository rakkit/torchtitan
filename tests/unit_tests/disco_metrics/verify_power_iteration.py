# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Contract + acceptance tests for optimizers/power_iteration.py.

Section 1 (CONTRACT) must pass at all times, stub or not: shapes, dtypes,
unit-norm vectors, determinism, batched-equals-per-item, degenerate slices.
disco.py and radial_helper.py depend on exactly these properties.

Section 2 (ACCURACY) is the acceptance test for a real implementation. It is
skipped while `power_iteration.IS_STUB` is True and runs automatically once
that flag is flipped, so whoever replaces the stub gets an immediate verdict.
"""
import sys

sys.path.insert(0, "resources/torchtitan")
import torch
from torchtitan.optimizers import power_iteration as pi
from torchtitan.optimizers.power_iteration import spectral_norm, top_singular_pair

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
fails = []
print(f"device: {dev}   IS_STUB: {pi.IS_STUB}")


def check(name, cond, detail=""):
    print(
        ("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else "")
    )
    if not cond:
        fails.append(name)


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)


print("\n== 1. CONTRACT: shapes / dtypes / norms ==")
for shape in [(64, 48), (48, 64), (1, 8), (8, 1), (5, 32, 16), (3, 4, 20, 10)]:
    W = torch.randn(*shape, device=dev)
    s, u, v = top_singular_pair(W)
    m, n = shape[-2], shape[-1]
    b = shape[:-2]
    ok_shape = s.shape == b and u.shape == b + (m,) and v.shape == b + (n,)
    check(
        f"shapes {shape}",
        ok_shape,
        f"{tuple(s.shape)} {tuple(u.shape)} {tuple(v.shape)}",
    )
    check(f"fp32 out {shape}", s.dtype == u.dtype == v.dtype == torch.float32)
    check(
        f"unit vectors {shape}",
        torch.allclose(u.norm(dim=-1), torch.ones_like(s), atol=1e-5)
        and torch.allclose(v.norm(dim=-1), torch.ones_like(s), atol=1e-5),
    )
    check(f"sigma non-negative {shape}", bool((s >= 0).all()))
    check(f"device preserved {shape}", s.device.type == W.device.type)

print("\n== 1. CONTRACT: batched == per-item ==")
Wb = torch.randn(7, 96, 64, device=dev)
sb, _, _ = top_singular_pair(Wb)
check(
    "batched sigma == per-item sigma",
    all(rel(sb[i], top_singular_pair(Wb[i])[0]) < 1e-6 for i in range(7)),
)

print("\n== 1. CONTRACT: degenerate slices ==")
z, uz, vz = top_singular_pair(torch.zeros(16, 12, device=dev))
check("zero matrix -> sigma 0", float(z) == 0.0)
check(
    "zero matrix -> unit vectors",
    rel(uz.norm(), 1.0) < 1e-5 and rel(vz.norm(), 1.0) < 1e-5,
)
nf, _, _ = top_singular_pair(torch.full((8, 8), float("nan"), device=dev))
check("non-finite -> sigma 0 (not NaN)", float(nf) == 0.0)
Wm = torch.randn(3, 10, 10, device=dev)
Wm[1] = 0
sm, _, _ = top_singular_pair(Wm)
check("mixed batch: zero slice isolated", float(sm[1]) == 0.0 and float(sm[0]) > 0)

print("\n== 1. CONTRACT: warm-start robustness ==")
W = torch.randn(64, 48, device=dev)
base = float(top_singular_pair(W)[0])
for label, v0 in [
    ("zeros", torch.zeros(48, device=dev)),
    ("nan", torch.full((48,), float("nan"), device=dev)),
    ("wrong shape", torch.randn(999, device=dev)),
]:
    try:
        s, _, _ = top_singular_pair(W, v0=v0)
        check(f"bad v0 ({label}) tolerated, finite", torch.isfinite(s).all())
    except Exception as e:
        check(f"bad v0 ({label}) tolerated", False, f"{type(e).__name__}: {e}")

print("\n== 1. CONTRACT: determinism / RNG isolation ==")
Wd = torch.randn(50, 50, device=dev)
torch.manual_seed(12345)
sA, _, vA = top_singular_pair(Wd)
torch.manual_seed(999)
sB, _, vB = top_singular_pair(Wd)
check("independent of global RNG state", float(sA) == float(sB) and torch.equal(vA, vB))
check(
    "spectral_norm agrees with top_singular_pair", float(spectral_norm(Wd)) == float(sA)
)

print("\n== 1. CONTRACT: rejects non-matrix input ==")
try:
    top_singular_pair(torch.randn(8, device=dev))
    check("1-D raises", False)
except ValueError:
    check("1-D raises ValueError", True)

if pi.IS_STUB:
    print("\n== 2. ACCURACY: SKIPPED (IS_STUB=True) ==")
    print("     These run automatically once power_iteration.IS_STUB is False.")
else:
    print("\n== 2. ACCURACY vs a float64 SVD ==")
    # The reference must be float64. `top_singular_pair` computes in float64
    # internally and returns float32, so comparing against an *fp32* SVD
    # measures the reference's error, not the implementation's.
    for shape in [(64, 64), (128, 32), (32, 128), (256, 257), (512, 512), (768, 2048)]:
        W = torch.randn(*shape, device=dev)
        s_got, u, v = top_singular_pair(W)
        ref = torch.linalg.svdvals(W.double())[0]
        check(
            f"sigma {shape} vs fp64 rel<1e-5",
            rel(s_got, ref) < 1e-5,
            f"rel={rel(s_got, ref):.2e}",
        )
        resid = float(
            (W.double() @ v.double() - float(s_got) * u.double()).norm() / ref
        )
        check(f"W v1 == sigma u1 {shape}", resid < 1e-5, f"resid={resid:.2e}")
        check(
            f"unit vectors {shape}",
            rel(u.norm(), 1.0) < 1e-5 and rel(v.norm(), 1.0) < 1e-5,
        )
        del W
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("\n== 2. no iteration state: result is independent of `v0` ==")
    # There is no warm start any more. Passing any v0 -- good, stale, or
    # nonsense -- must give bit-identical output, which is the property that
    # makes the removed warm-start cache safe to drop.
    W = torch.randn(512, 384, device=dev)
    base = top_singular_pair(W)
    for label, v0 in [
        ("None", None),
        ("zeros", torch.zeros(384, device=dev)),
        ("random", torch.randn(384, device=dev)),
        ("nan", torch.full((384,), float("nan"), device=dev)),
    ]:
        got = top_singular_pair(W, v0=v0)
        check(
            f"identical with v0={label}",
            float(got[0]) == float(base[0]) and torch.equal(got[2], base[2]),
        )

    print("\n== 2. exact on a perturbed weight (the real per-step case) ==")
    # What actually happens between logging events: W moves a little. There is
    # no accumulated state, so accuracy must not depend on the history.
    W = torch.randn(384, 256, device=dev)
    errs = []
    for _ in range(8):
        W = W + 3e-3 * torch.randn_like(W)
        s_got, _, _ = top_singular_pair(W)
        errs.append(rel(s_got, torch.linalg.svdvals(W.double())[0]))
    print("     errors:", " ".join(f"{e:.1e}" for e in errs))
    check(
        "every step within 1e-5 of fp64 truth", max(errs) < 1e-5, f"max={max(errs):.2e}"
    )
    check(
        "no drift (last no worse than 10x the best)",
        errs[-1] < 10 * min(errs) or errs[-1] < 1e-5,
    )

    print("\n== 2. ill-conditioned and rank-deficient ==")
    for label, mk in [
        (
            "kappa 1e6",
            lambda: (
                torch.linalg.qr(torch.randn(384, 384, device=dev))[0]
                * torch.logspace(0, -6, 384, device=dev)
            )
            @ torch.linalg.qr(torch.randn(512, 384, device=dev))[0].T,
        ),
        (
            "rank-deficient",
            lambda: torch.randn(384, 64, device=dev) @ torch.randn(64, 512, device=dev),
        ),
    ]:
        W = mk()
        s_got, _, _ = top_singular_pair(W)
        ref = torch.linalg.svdvals(W.double())[0]
        check(
            f"sigma_max exact, {label}",
            rel(s_got, ref) < 1e-5,
            f"rel={rel(s_got, ref):.2e}",
        )
        del W
        if dev == "cuda":
            torch.cuda.empty_cache()

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print(
    "ALL POWER-ITERATION CONTRACT CHECKS PASSED"
    + ("  (accuracy section skipped: stub)" if pi.IS_STUB else "")
)
