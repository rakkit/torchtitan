# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verification for optimizers/radial_helper.py (plan Verification steps 4-5)."""
import sys

sys.path.insert(0, "resources/torchtitan")
import torch
from torchtitan.optimizers.radial_helper import (
    calculate_radial_metrics,
    new_radial_state,
    RADIAL_METRIC_NAMES,
    SpectralInputs,
)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
fails = []
print(f"device: {dev}  |  {len(RADIAL_METRIC_NAMES)} metrics")


def check(name, cond, detail=""):
    print(
        ("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else "")
    )
    if not cond:
        fails.append(name)


def rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(b), 1e-30)


# ---- reference implementations, transcribed from the spec sketch ----
def ref_rms_to_rms(W, U):
    Uw, Sw, Vhw = torch.linalg.svd(W, full_matrices=False)
    u1, v1 = Uw[:, 0], Vhw[0, :]
    sigma_u = torch.linalg.matrix_norm(U, ord=2)
    return u1 @ (U @ v1) / sigma_u


def ref_rms_to_inf(W, U):
    rw = torch.linalg.vector_norm(W, ord=2, dim=1)
    ru = torch.linalg.vector_norm(U, ord=2, dim=1)
    max_w, i = torch.max(rw, dim=0)
    return torch.dot(W[i, :], U[i, :]) / (max_w * ru.max())


def ref_l1_to_rms(W, U):
    cw = torch.linalg.vector_norm(W, ord=2, dim=0)
    cu = torch.linalg.vector_norm(U, ord=2, dim=0)
    max_w, j = torch.max(cw, dim=0)
    return torch.dot(W[:, j], U[:, j]) / (max_w * cu.max())


def ref_aus(W, U, mode):
    if mode == "fro":
        a, b = W.norm(), U.norm()
        return (W / a - U / b).norm()
    dim = 1 if mode == "row" else 0
    a = torch.linalg.vector_norm(W, ord=2, dim=dim).max()
    b = torch.linalg.vector_norm(U, ord=2, dim=dim).max()
    D = W / a - U / b
    return torch.linalg.vector_norm(D, ord=2, dim=dim).max()


def run(W, U, spectral=None, transpose=False):
    st = new_radial_state(W.device, torch.float32)
    return calculate_radial_metrics(
        W, W + U, st, spectral=spectral, transpose=transpose
    )


def exact_spectral(W, U):
    Uw, Sw, Vhw = torch.linalg.svd(W.float(), full_matrices=False)
    sig_w = Sw[0]
    sig_u = torch.linalg.matrix_norm(U.float(), ord=2)
    Wn = W.float() / sig_w
    Un = U.float() / sig_u
    return SpectralInputs(
        sigma_before=sig_w,
        u1_before=Uw[:, 0],
        v1_before=Vhw[0, :],
        sigma_after=torch.linalg.matrix_norm((W + U).float(), ord=2),
        sigma_update=sig_u,
        aus_sigma=torch.linalg.matrix_norm(Wn - Un, ord=2),
    )


print("\n== radiality vs reference (exact spectral inputs) ==")
for shape in [(64, 48), (48, 64), (128, 128), (33, 17)]:
    W = torch.randn(*shape, device=dev)
    U = 0.01 * torch.randn(*shape, device=dev)
    m = run(W, U, exact_spectral(W, U))
    checks = [
        ("rms_to_rms", m["radiality_rms_to_rms"], ref_rms_to_rms(W, U)),
        ("rms_to_inf", m["radiality_rms_to_inf"], ref_rms_to_inf(W, U)),
        ("l1_to_rms", m["radiality_l1_to_rms"], ref_l1_to_rms(W, U)),
    ]
    bad = [f"{n}:{rel(g, r):.1e}" for n, g, r in checks if rel(g, r) > 1e-4]
    check(f"radiality {shape}", not bad, " ".join(bad))

print("\n== aus norms vs reference ==")
for shape in [(64, 48), (48, 64), (100, 100)]:
    W = torch.randn(*shape, device=dev)
    U = 0.05 * torch.randn(*shape, device=dev)
    m = run(W, U, exact_spectral(W, U))
    checks = [
        ("fro", m["aus_frobenius"], ref_aus(W, U, "fro")),
        ("row", m["aus_rms_to_inf"], ref_aus(W, U, "row")),
        ("col", m["aus_l1_to_rms"], ref_aus(W, U, "col")),
    ]
    bad = [f"{n}:{rel(g, r):.1e}" for n, g, r in checks if rel(g, r) > 1e-4]
    check(f"aus {shape}", not bad, " ".join(bad))
    # the documented identity aus_frobenius == sqrt(2 - 2*radial_cosine)
    ident = torch.sqrt((2.0 - 2.0 * m["radial_cosine"]).clamp_min(0))
    check(
        f"aus_frobenius==sqrt(2-2cos) {shape}",
        rel(m["aus_frobenius"], ident) < 1e-5,
        f"{float(m['aus_frobenius']):.6f} vs {float(ident):.6f}",
    )

print("\n== spectral scalars ==")
W = torch.randn(80, 60, device=dev)
U = 0.02 * torch.randn(80, 60, device=dev)
sp = exact_spectral(W, U)
m = run(W, U, sp)
check("spectral_radius", rel(m["spectral_radius"], sp.sigma_before) < 1e-6)
check("spectral_radius_next", rel(m["spectral_radius_next"], sp.sigma_after) < 1e-6)
check(
    "spectral_growth",
    rel(m["spectral_growth"], sp.sigma_after / sp.sigma_before) < 1e-6,
)
check(
    "spectral_relative_step",
    rel(m["spectral_relative_step"], sp.sigma_update / sp.sigma_before) < 1e-6,
)
check("aus_rms_to_rms", rel(m["aus_rms_to_rms"], sp.aus_sigma) < 1e-6)

print("\n== transpose flips the row/col geometry ==")
W = torch.randn(64, 40, device=dev)
U = 0.02 * torch.randn(64, 40, device=dev)
mF, mT = run(W, U, transpose=False), run(W, U, transpose=True)
check(
    "transpose swaps rms_to_inf <-> l1_to_rms",
    rel(mT["radiality_rms_to_inf"], mF["radiality_l1_to_rms"]) < 1e-5
    and rel(mT["radiality_l1_to_rms"], mF["radiality_rms_to_inf"]) < 1e-5,
)
check(
    "transpose leaves aus_frobenius alone",
    rel(mT["aus_frobenius"], mF["aus_frobenius"]) < 1e-6,
)

print("\n== FIXED ARITY (disco.py buffer sizing depends on this) ==")
cases = {
    "2-D + spectral": (
        torch.randn(32, 24, device=dev),
        0.01 * torch.randn(32, 24, device=dev),
        True,
    ),
    "2-D no spectral": (
        torch.randn(32, 24, device=dev),
        0.01 * torch.randn(32, 24, device=dev),
        False,
    ),
    "1-D + spectral": (
        torch.randn(48, device=dev),
        0.01 * torch.randn(48, device=dev),
        False,
    ),
    "1-D no spectral": (
        torch.randn(48, device=dev),
        0.01 * torch.randn(48, device=dev),
        False,
    ),
    "zero W": (
        torch.zeros(16, 12, device=dev),
        0.01 * torch.randn(16, 12, device=dev),
        False,
    ),
    "zero U": (torch.randn(16, 12, device=dev), torch.zeros(16, 12, device=dev), False),
    "both zero": (
        torch.zeros(16, 12, device=dev),
        torch.zeros(16, 12, device=dev),
        False,
    ),
    "3-D": (
        torch.randn(4, 8, 6, device=dev),
        0.01 * torch.randn(4, 8, 6, device=dev),
        False,
    ),
    "bf16": (
        torch.randn(20, 20, device=dev).bfloat16(),
        (0.01 * torch.randn(20, 20, device=dev)).bfloat16(),
        False,
    ),
}
for name, (W, U, with_sp) in cases.items():
    sp = exact_spectral(W, U) if with_sp else None
    m = run(W, U, sp)
    ok_keys = list(m.keys()) == RADIAL_METRIC_NAMES
    all_0d = all(m[k].ndim == 0 for k in m)
    finite = all(torch.isfinite(m[k]).all() for k in m)
    check(
        f"arity/finite: {name}",
        ok_keys and all_0d and finite,
        ("keys" if not ok_keys else "")
        + (" ndim" if not all_0d else "")
        + (
            " nonfinite:" + str([k for k in m if not torch.isfinite(m[k]).all()])
            if not finite
            else ""
        ),
    )

print("\n== accumulator semantics unchanged by the extension ==")
W = torch.randn(24, 18, device=dev)
st = new_radial_state(W.device, torch.float32)
prev = None
for i in range(4):
    U = 0.01 * torch.randn(24, 18, device=dev)
    m = calculate_radial_metrics(W, W + U, st)
    if i == 0:
        check("first call reports zero history", float(m["raw_A2"]) == 0.0)
    else:
        check(f"raw_A2 accumulates (call {i})", float(m["raw_A2"]) > float(prev))
    prev = m["raw_A2"]
    W = W + U

print("\n== degenerate: sentinel is exactly 0, not NaN ==")
W = torch.randn(16, 12, device=dev)
m = run(W, torch.zeros_like(W), exact_spectral(W, torch.zeros_like(W) + 1e-30))
for k in [
    "radiality_rms_to_rms",
    "radiality_rms_to_inf",
    "radiality_l1_to_rms",
    "aus_frobenius",
    "aus_rms_to_inf",
    "aus_l1_to_rms",
    "aus_rms_to_rms",
]:
    check(f"zero-update sentinel {k}", float(m[k]) == 0.0, f"{float(m[k])}")

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL RADIAL_HELPER CHECKS PASSED")
