# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batched (`batch_ndim=1`) radial metrics: equivalence in both directions.

Two independent claims, because the batching refactor could break either:

  1. The unbatched path is STILL bit-identical to `_radial_helper_pre_batch_
     reference.py`, a verbatim snapshot taken just before the refactor. A perf
     change must not move values that are already being logged.
  2. The batched path equals looping the unbatched one, per expert, including
     the running accumulators across several steps.

`angle_from_cos` is exempted from (2) at fp32-exact tolerance, and only that
one: it is `arccos(cos)` at cos ~= 0.99995, where the derivative is ~1e2, so a
last-digit difference in the reduction order of `dot_ww` (summing [E,m,n] at
once vs [m,n] per expert) is amplified ~1000x. That ill-conditioning is exactly
why the primary `angle` uses atan2 -- and `angle` IS bit-identical here, which
is the check that would actually catch a geometry bug.
"""
import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from torchtitan.optimizers.radial_helper import (  # noqa: E402
    calculate_radial_metrics,
    new_radial_state,
    RADIAL_METRIC_NAMES,
    SpectralInputs,
)

_spec = importlib.util.spec_from_file_location(
    "radial_ref", os.path.join(_HERE, "_radial_helper_pre_batch_reference.py")
)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def mkspec(W, U, batched):
    """SpectralInputs built from real SVDs, the way disco.py supplies them."""
    Ws = W if batched else W.unsqueeze(0)
    Us = U if batched else U.unsqueeze(0)
    uw, sw, vw = torch.linalg.svd(Ws.float(), full_matrices=False)
    su = torch.linalg.svdvals(Us.float())
    nw = Ws / sw[:, :1, None].clamp_min(1e-30)
    nu = Us / su[:, :1, None].clamp_min(1e-30)
    aus = torch.linalg.svdvals((nw - nu).float())[:, 0]

    def sq(t):
        return t if batched else t.squeeze(0)

    return SpectralInputs(
        sigma_before=sq(sw[:, 0]),
        u1_before=sq(uw[:, :, 0]),
        v1_before=sq(vw[:, 0, :]),
        sigma_after=sq(sw[:, 0] * 1.01),
        sigma_update=sq(su[:, 0]),
        aus_sigma=sq(aus),
    )


print("== 1. unbatched path bit-identical to the pre-batch snapshot ==")
for shape in [(64, 128), (128, 64), (32, 32), (1, 16), (7, 7), (16,)]:
    for T in (False, True):
        W = torch.randn(*shape, device=dev)
        Wa = W + 0.01 * torch.randn_like(W)
        spec = mkspec(W, Wa - W, False) if len(shape) == 2 else None
        s_new, s_ref = new_radial_state(W.device), ref.new_radial_state(W.device)
        for _ in range(3):
            a = calculate_radial_metrics(W, Wa, s_new, spectral=spec, transpose=T)
            b = ref.calculate_radial_metrics(
                W,
                Wa,
                s_ref,
                spectral=None if spec is None else ref.SpectralInputs(*spec),
                transpose=T,
            )
        bad = [k for k in RADIAL_METRIC_NAMES if not torch.equal(a[k], b[k])]
        st = [k for k in s_new if not torch.equal(s_new[k], s_ref[k])]
        check(f"shape={shape} transpose={T}", not bad and not st, str(bad + st)[:70])

print("\n== 2. batched == looping the unbatched path ==")
# angle_from_cos: see module docstring. Everything else must hold at fp32.
LOOSE = {"angle_from_cos": 5e-3}
TOL = 5e-6
for (E, m, n) in [(2, 64, 128), (4, 128, 64), (1, 32, 32), (8, 48, 48), (3, 16, 64)]:
    for T in (False, True):
        Wb = torch.randn(E, m, n, device=dev)
        Wa = Wb + 0.01 * torch.randn_like(Wb)
        spec_b = mkspec(Wb, Wa - Wb, True)
        sb = new_radial_state(Wb.device, shape=(E,))
        ss = [new_radial_state(Wb.device) for _ in range(E)]
        for _ in range(3):
            got = calculate_radial_metrics(
                Wb, Wa, sb, spectral=spec_b, transpose=T, batch_ndim=1
            )
            per = [
                calculate_radial_metrics(
                    Wb[e],
                    Wa[e],
                    ss[e],
                    spectral=SpectralInputs(
                        *[None if x is None else x[e] for x in spec_b]
                    ),
                    transpose=T,
                )
                for e in range(E)
            ]
        worst, wk, bad = 0.0, "", []
        for k in RADIAL_METRIC_NAMES:
            stacked = torch.stack([p[k] for p in per])
            if got[k].shape != (E,):
                bad.append(f"{k}:shape{tuple(got[k].shape)}")
                continue
            rel = (got[k] - stacked).abs().max().item() / max(
                1e-12, stacked.abs().max().item()
            )
            if rel > LOOSE.get(k, TOL):
                bad.append(f"{k}:{rel:.1e}")
            if k not in LOOSE and rel > worst:
                worst, wk = rel, k
        acc = [
            k
            for k in sb
            if (sb[k] - torch.stack([s[k] for s in ss])).abs().max().item() > 1e-5
        ]
        check(
            f"E={E} {m}x{n} T={T}",
            not bad and not acc,
            f"worst {worst:.1e} on {wk}" + (f" BAD={bad + acc}" if bad or acc else ""),
        )
        # The check that would actually catch a geometry bug: the two
        # independent formulas for the same angle must still agree with each
        # other inside the batched path, as closely as they do inside the
        # looped one. (Not bit-exactness against the loop -- the batched
        # reduction sums [E,m,n] in one go and legitimately differs in the
        # last fp32 digit; `angle`'s agreement at TOL is asserted above.)
        agree_b = (got["angle"] - got["angle_from_cos"]).abs().max().item()
        agree_l = max((p["angle"] - p["angle_from_cos"]).abs().item() for p in per)
        check(
            f"  angle vs angle_from_cos agree E={E} {m}x{n} T={T}",
            agree_b <= max(10 * agree_l, 1e-4),
            f"batched {agree_b:.1e} vs looped {agree_l:.1e}",
        )

print("\n== 3. fixed arity and finiteness, incl. degenerate slices ==")
cases = [
    (torch.zeros(4, 8, device=dev), 0, ()),
    (torch.zeros(3, 4, 8, device=dev), 1, (3,)),
    (torch.randn(16, device=dev), 0, ()),
    (torch.zeros(2, 5, 5, device=dev), 1, (2,)),
]
# a batch where only SOME experts are degenerate
mixed = torch.randn(3, 8, 8, device=dev)
mixed[1] = 0.0
cases.append((mixed, 1, (3,)))
for W, bn, sh in cases:
    st = new_radial_state(W.device, shape=sh)
    out = calculate_radial_metrics(W, W.clone(), st, batch_ndim=bn)
    ok = set(out) == set(RADIAL_METRIC_NAMES) and all(
        torch.isfinite(v).all() for v in out.values()
    )
    shapes_ok = all(tuple(v.shape) == sh for v in out.values())
    check(
        f"arity/finite/shape ndim={W.ndim} batch_ndim={bn}",
        ok and shapes_ok,
        f"{len(out)} keys",
    )

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL BATCHED RADIAL CHECKS PASSED")
