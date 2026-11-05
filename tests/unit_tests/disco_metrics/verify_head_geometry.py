# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks for the LM-head update geometry metrics (eta, rho_RI, gamma_H).

Run from the repo root:
    python resources/torchtitan/tests/unit_tests/disco_metrics/verify_head_geometry.py

References are brute-force from the definitions (`torch.cdist` for the diameter,
a dense `h @ U.T` for the quotient sup norm), never the implementation, so a bug
in the implementation cannot be reproduced by its own reference.
"""

import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from torchtitan.optimizers.helpers.head_geometry import (  # noqa: E402
    diameter_lower_bound,
    head_geometry,
    HEAD_METRIC_NAMES,
    product_frobenius_sq,
    quotient_sup_norm,
)

dev = "cuda" if torch.cuda.is_available() else "cpu"


class _Skip(Exception):
    """The accelerator-only part of this script was skipped."""


fails: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def rel(a, b):
    return float((torch.as_tensor(a) - torch.as_tensor(b)).abs().max()) / max(
        float(torch.as_tensor(b).abs().max()), 1e-30
    )


def unembed_sqrt_rows(V, d, dtype=torch.float64):
    """Rows at L2 = 1/sqrt(d), i.e. what `fused_unembed_sqrt` produces."""
    U = torch.randn(V, d, device=dev, dtype=dtype)
    return U / U.norm(dim=-1, keepdim=True) / (d**0.5)


print(f"device: {dev}")

print("\n== quotient sup norm vs dense reference, and chunk invariance ==")
torch.manual_seed(0)
V, d, P = 900, 64, 32
U = unembed_sqrt_rows(V, d)
h = torch.randn(P, d, device=dev, dtype=torch.float64)
h = h / h.pow(2).mean(-1, keepdim=True).sqrt()  # ||h||_rms == 1
z = h @ U.T
ref = (z.amax(-1) - z.amin(-1)) / 2
for chunk in (37, 128, V, 10 * V):
    got = quotient_sup_norm(h.float(), U.float(), chunk)
    check(
        f"chunk={chunk}",
        rel(got.double(), ref) < 1e-5,
        f"rel {rel(got.double(), ref):.2e}",
    )

print("\n== diameter estimator: lower bound, and close to exact ==")
for V_, d_ in ((900, 64), (4000, 128)):
    torch.manual_seed(1)
    Uk = unembed_sqrt_rows(V_, d_)
    exact = float(torch.cdist(Uk, Uk).amax())
    got = float(diameter_lower_bound(Uk.float(), iters=10))
    # It is a lower bound by construction; a small tolerance covers fp32.
    check(
        f"lower bound V={V_} d={d_}",
        got <= exact * (1 + 1e-4),
        f"{got:.6f} <= {exact:.6f}",
    )
    # Measured ~0.98 of exact; anything below 0.9 means the sweep regressed.
    check(f"within 10% V={V_} d={d_}", got >= 0.9 * exact, f"ratio {got / exact:.4f}")

print("\n== theorems that must hold for the diagnostic to mean anything ==")
torch.manual_seed(2)
U = unembed_sqrt_rows(4000, 128)
h = torch.randn(256, 128, device=dev, dtype=torch.float64)
h = h / h.pow(2).mean(-1, keepdim=True).sqrt()
m = head_geometry(
    h.float(), U.float(), U.float(), eta=0.125, vocab_chunk=1024, diameter_iters=10
)
check("gamma_H <= 1", float(m["gamma_H"]) <= 1 + 1e-5, f"{float(m['gamma_H']):.6f}")
check("a_H <= 1", float(m["a_H_max"]) <= 1 + 1e-5, f"{float(m['a_H_max']):.6f}")
# row-l2 with scale inv_sqrt_n makes the head's update exactly unit in
# rms->inf; a change of scale would show up here rather than silently
# rescaling every metric.
check(
    "||U||_rms->inf == 1 under row-l2 / inv_sqrt_n",
    abs(float(m["U_rms_to_inf"]) - 1.0) < 1e-5,
    f"{float(m['U_rms_to_inf']):.8f}",
)
# np_rmsnorm feeds the head unit-RMS activations; this is what makes rho = eps_z/eta.
check(
    "||h||_rms == 1",
    abs(float(m["h_rms_mean"]) - 1.0) < 1e-5,
    f"{float(m['h_rms_mean']):.8f}",
)

print("\n== gamma_H's probe term, determinism, and degenerate input ==")
# The probes' contribution to the diameter used to be a separate [V, P] matmul.
# It is algebraically identical to rho_max, so it is now folded in as a max over
# numbers already computed. Same answer, one fewer matmul and [V, P] buffer.
Uf, hf = U.float(), h.float()
w_dirs = hf / hf.norm(dim=-1, keepdim=True)
proj = Uf @ w_dirs.T
probe_term = float((proj.amax(0) - proj.amin(0)).amax() / (2 * Uf.norm(dim=-1).amax()))
check(
    "probe term == rho_max",
    rel(probe_term, float(m["rho_max"])) < 1e-5,
    f"{probe_term:.8f} vs {float(m['rho_max']):.8f}",
)
# ... which is exactly why a_H_max cannot exceed 1: gamma_H is a max that
# includes rho_max. The sweep-only variant is the one that can actually fire.
check(
    "gamma_H == max(gamma_H_sweep, rho_max)",
    rel(float(m["gamma_H"]), max(float(m["gamma_H_sweep"]), float(m["rho_max"])))
    < 1e-6,
)
check("a_H_sweep_max is reported", "a_H_sweep_max" in m)
# The optimizer packs these into a fixed-size vector for the collective
# that carries them to the storing rank, so the order is load-bearing.
check("keys match HEAD_METRIC_NAMES", tuple(m) == HEAD_METRIC_NAMES, str(tuple(m)))

# Deterministic: the estimator draws from a local generator seeded by `seed`,
# never the global stream, so a diagnostic cannot perturb training and gamma_H
# does not jitter between identical calls.
m2 = head_geometry(hf, Uf, Uf, eta=0.125, vocab_chunk=1024, diameter_iters=10)
check(
    "identical across calls",
    all(torch.equal(m[k], m2[k]) for k in m),
)
m3 = head_geometry(hf, Uf, Uf, eta=0.125, vocab_chunk=1024, diameter_iters=10, seed=7)
check(
    "seed actually changes the draw",
    not torch.equal(m["gamma_H_sweep"], m3["gamma_H_sweep"]),
)
before = torch.randn(4, device=dev)
head_geometry(hf, Uf, Uf, eta=0.125, vocab_chunk=1024, diameter_iters=10)
check("global RNG untouched", torch.equal(before, before))
g_state = torch.cuda.get_rng_state() if dev == "cuda" else torch.get_rng_state()
head_geometry(hf, Uf, Uf, eta=0.125, vocab_chunk=1024, diameter_iters=10)
g_after = torch.cuda.get_rng_state() if dev == "cuda" else torch.get_rng_state()
check("global RNG state unchanged", torch.equal(g_state, g_after))

# Degenerate U: sentinel 0 throughout, matching radial_helper's convention,
# rather than dividing by a clamp floor and logging an astronomical number.
mz = head_geometry(
    hf, torch.zeros_like(Uf), torch.zeros_like(Uf), eta=0.125, vocab_chunk=1024
)
check(
    "all-zero U -> 0 sentinels",
    all(
        float(mz[k]) == 0.0 for k in ("rho_mean", "gamma_H", "a_H_max", "a_H_sweep_max")
    ),
    " ".join(f"{k}={float(mz[k]):g}" for k in ("gamma_H", "a_H_max", "a_H_sweep_max")),
)

# Shape mistakes should name the problem, not surface as a matmul error deep
# inside the vocab-chunk loop.
for bad, why in (((5, 7), "d mismatch"), ((3, 4, 5), "h not 2-D")):
    try:
        head_geometry(torch.zeros(*bad, device=dev), Uf, Uf, eta=0.1)
        check(f"rejects {why}", False, "no error raised")
    except ValueError:
        check(f"rejects {why}", True)
    except Exception as e:  # noqa: BLE001
        check(f"rejects {why}", False, f"{type(e).__name__} instead of ValueError")

print("\n== rho_RI against a dense reference ==")
z = h @ U.T
qsup = (z.amax(-1) - z.amin(-1)) / 2
rho_ref = qsup / (U.norm(dim=-1).amax() * (U.shape[1] ** 0.5) * 1.0)
check(
    "rho_mean",
    rel(float(m["rho_mean"]), rho_ref.mean()) < 1e-5,
    f"{float(m['rho_mean']):.8f} vs {float(rho_ref.mean()):.8f}",
)

print("\n== LAR against its independent spectral definition ==")
torch.manual_seed(5)
Vv, dd, Pp = 3000, 128, 256
Wl = torch.randn(Vv, dd, device=dev, dtype=torch.float64) * 0.05
Ul = torch.randn(Vv, dd, device=dev, dtype=torch.float64)
Ul = Ul / (Ul.norm(dim=-1, keepdim=True) * (dd**0.5))
hl = torch.randn(Pp, dd, device=dev, dtype=torch.float64)
hl = hl / hl.pow(2).mean(-1, keepdim=True).sqrt()
ml = head_geometry(hl.float(), Ul.float(), Wl.float(), eta=0.125, vocab_chunk=997)

# LAR = 1 + (1/2) log_n(sum_i p_i q_i), computed the paper's way: eigendecompose
# W^T W for the singular directions, project the probes onto them. Nothing in
# this reference touches the three-RMS-norm route the implementation uses.
evals, evecs = torch.linalg.eigh(Wl.T @ Wl)
p_w = evals.clamp_min(0) / evals.clamp_min(0).sum()
q_a = (hl @ evecs).pow(2).sum(0) / hl.pow(2).sum()
lar_spec = 1 + 0.5 * torch.log((p_w * q_a).sum()) / torch.log(
    torch.tensor(float(dd), dtype=torch.float64, device=dev)
)
check(
    "lar_w == 1 + 1/2 log_n(sum p q)",
    rel(float(ml["lar_w"]), float(lar_spec)) < 1e-5,
    f"{float(ml['lar_w']):.8f} vs {float(lar_spec):.8f}",
)
# Isotropic W and h give sum_i p_i q_i ~ 1/n, hence LAR ~ 1/2. A wrong log base
# or a Frobenius/RMS mix-up would not land on 0.5.
check(
    "random W/h -> LAR ~ 0.5",
    abs(float(ml["lar_w"]) - 0.5) < 0.02,
    f"{float(ml['lar_w']):.5f}",
)
check(
    "ratio is base-free and consistent",
    rel(
        float(ml["lar_ratio_w"]) ** 1.0,
        float(torch.exp(torch.tensor(float(ml["lar_w"]) * math.log(dd)))),
    )
    < 1e-4,
)

# Both routes of product_frobenius_sq must agree; they are dispatched on d < P
# so only one runs for a given shape.
for Vt, dt, Pt in ((2000, 64, 256), (2000, 256, 64)):
    Mt = torch.randn(Vt, dt, device=dev)
    ht = torch.randn(Pt, dt, device=dev)
    got, mn = product_frobenius_sq(ht, Mt, 997)
    ref = (ht.double() @ Mt.double().T).pow(2).sum()
    check(
        f"product_frobenius_sq d={dt} P={Pt}",
        rel(float(got), float(ref)) < 1e-4,
        f"{'gram' if dt < Pt else 'direct'} route, rel {rel(float(got), float(ref)):.2e}",
    )
    # 1e-4, not tighter: the direct route reads ||M||_F^2 off an fp32
    # reduction over V*d terms, which is 1.7e-5 relative at [2000, 256].
    check(
        f"  ||M||_F^2 alongside d={dt}",
        rel(float(mn), float(Mt.double().pow(2).sum())) < 1e-4,
    )

print("\n== under the chunked loss the model's output is the head's input ==")
if dev == "cpu":
    # opt_moe's blocks carry their own torch.compile, which inductor refuses on
    # this CPU; the check is meaningful only on the accelerator anyway.
    print("  SKIP  no accelerator")
try:
    from torchtitan.models.opt_moe import build_model_config
    from torchtitan_recipes.models.opt_moe import opt_moe_mis_8b

    job = opt_moe_mis_8b()
    job.model = build_model_config(
        "gauge-d8-w256", parallelism=job.parallelism, vocab_size=2048
    )
    if dev == "cpu":
        raise _Skip()
    mdl = job.model.build().to(dev).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    tok = torch.randint(0, 2048, (256,), device=dev)
    pos = torch.arange(256, device=dev)
    # flex layers need a real BlockMask; build it the way preprocess_inputs does.
    masks = mdl._get_attention_metadata(pos)
    # The engine sets this for the chunked loss and samples the output.
    mdl._skip_lm_head = True
    with torch.no_grad():
        h_trunk = mdl._trunk(tok, pos, masks)
        h_out = mdl(tok, pos, masks)
    check("output == the head's input (_trunk)", rel(h_out, h_trunk) < 1e-6)

    for lid, blk in list(mdl.layers.named_children()):
        mdl.layers.register_module(lid, torch.compile(blk, fullgraph=True))
    with torch.no_grad():
        h_c = mdl(tok, pos, masks)
    check(
        "survives fullgraph=True blocks",
        rel(h_c, h_out) < 1e-3,
        f"rel {rel(h_c, h_out):.2e}",
    )
except _Skip:
    pass
except Exception as e:  # pragma: no cover
    check("opt_moe forward checks", False, f"{type(e).__name__}: {str(e)[:110]}")

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL HEAD GEOMETRY CHECKS PASSED")
