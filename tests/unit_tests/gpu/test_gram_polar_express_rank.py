# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gram_polar_express (amphione's production orthogonaliser: quack kernels,
fp16 work dtype, 6 steps, safety 1.05) on rank-deficient units: an expert that
saw few tokens, a low-rank momentum at step 1.

Measured 11-05 (GH200), rank r of the unit, the energy share of the output's
singular values outside the input's range ("null share"):

    fp16 kernel (production)         0.082 / r  (r = 1, 8, 64; any shape)
    the same iteration in fp32       < 1e-7
    bf16 polar_express / NS5 (Muon)  0.85 / 0.46 at r = 1

Every half-precision polar iteration lifts its rounding noise in the null
space (it is built to lift small singular values towards 1); the fp16 Gram
kernel lifts less of it than the bf16 iterations Muon uses. Nothing blows up:
the output is finite and the input's directions keep singular values near 1.
These bounds guard that behaviour."""

import pytest
import torch

pytest.importorskip("quack")
if not torch.cuda.is_available():
    pytest.skip("quack's kernels need CUDA", allow_module_level=True)

from torchtitan.optimizers.disco.presets import EPS
from torchtitan.optimizers.disco.rules import rule_of
from torchtitan.optimizers.kernels.gram_newton_schulz import gram_polar_express

STEPS, SAFETY = 6, 1.05  # the recipe's; EPS: what the rule passes (presets.EPS)
# amphione's direction as the production recipe builds it
DIRECTION = rule_of(
    {
        "rule": "amphione",
        "args": {
            "compute": "gram_polar_express",
            "steps": STEPS,
            "polar_express_safety": SAFETY,
        },
    }
).compute.direction


def _low_rank(b, m, n, r):
    g = torch.Generator(device="cuda").manual_seed(r)
    a = torch.randn(b, m, r, generator=g, device="cuda", dtype=torch.float64)
    c = torch.randn(b, r, n, generator=g, device="cuda", dtype=torch.float64)
    return (a @ c / r**0.5).float()


def _spectrum(G, **kw):
    """The production call (the rule's direction) without `kw`; the kernel
    with `kw` (fp32 iterations) for the comparison."""
    if kw:
        O = gram_polar_express(G, steps=STEPS, eps=EPS, safety=SAFETY, **kw)
    else:
        O = DIRECTION.run_whole(G).update
    assert torch.isfinite(O).all()
    return torch.linalg.svdvals(O.double())


@pytest.mark.parametrize("shape", [(2, 512, 256), (2, 256, 512), (1, 768, 768)])
@pytest.mark.parametrize("rank", [1, 8, 64])
def test_rank_deficient_units_stay_bounded(shape, rank):
    b, m, n = shape
    G = _low_rank(b, m, n, rank)
    s = _spectrum(G)  # the production kernel
    signal, null = s[..., :rank], s[..., rank:]
    share = (null.square().sum(-1) / s.square().sum(-1)).max().item()
    assert 0.95 <= signal.min().item() and signal.max().item() <= 1.25, signal
    assert share <= 0.12 / rank, f"null share {share:.2e} (measured 0.082 / r)"
    s32 = _spectrum(G, use_kernels=False, work_dtype=torch.float32)
    share32 = (s32[..., rank:].square().sum(-1) / s32.square().sum(-1)).max().item()
    assert share32 < 1e-6  # the leak is the half precision's, not the iteration's
