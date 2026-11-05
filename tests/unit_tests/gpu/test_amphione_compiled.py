# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Amphione's compiled path on CUDA (no column scales: the production recipe)
against its eager body, the same equations with the reductions in another
order. Two steps from an empty state at the recipe's momentum (0.95, Nesterov
on the direction), frozen row scales on one unit: the averages (M, m_g) agree
to fp32 rounding; the weights to the rounding of the turn and of a sign step
whose average sits at zero (a flip moves a row scale by 2 lr)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the compiled path is CUDA's", allow_module_level=True)

from torchtitan.optimizers.disco.rules import rule_of, StepCtx


@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("scale_nesterov", [False, True])
@pytest.mark.parametrize("shape", [(3, 256, 128), (2, 128, 512)])
def test_the_compiled_path_is_the_eager_one(shape, scale_nesterov, nesterov):
    b, m, n = shape
    cm = rule_of(
        {
            "rule": "amphione",
            "args": {
                "compute": "svd",
                "frozen_scale_units": ["u"],
                "scale_nesterov": scale_nesterov,
            },
        }
    ).compute
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    W = torch.randn(b, m, n, device=dev, generator=gen) * 4 / n**0.5
    W[1, 3] = 0.0  # a zero row starts at sqrt(0.33)
    Gs = [1e-2 * torch.randn(b, m, n, device=dev, generator=gen) for _ in range(2)]
    frozen = torch.tensor([True] + [False] * (b - 1), device=dev)
    out = []
    for run in (cm._run_eager, cm._run_compiled):
        w, s = W.clone(), torch.zeros(b, *cm.state_shape(m, n), device=dev)
        for t, G in enumerate(Gs, start=1):
            ctx = StepCtx(0.02, 0.0, t, momentum=0.95, nesterov=nesterov, frozen=frozen)
            res = run(G, w, s, ctx)
            w, s = res.weight, res.state
        out.append((w, s))
    (we, se), (wc, sc) = out
    M_e, M_c = se[..., :n], sc[..., :n]
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
    assert rel(M_c, M_e) < 1e-6 and rel(sc[..., n + 1], se[..., n + 1]) < 1e-6
    start = W[0].norm(dim=-1)  # the frozen unit's scales stay at its row norms
    torch.testing.assert_close(sc[0, :, n], start, rtol=1e-6, atol=0)
    torch.testing.assert_close(se[0, :, n], start, rtol=1e-6, atol=0)
    assert rel(wc, we) < 1e-3


def test_run_takes_the_compiled_path_on_cuda(monkeypatch):
    """`run` dispatches every CUDA call without column scales to the compiled
    path, an fp32 or a bf16 gradient stack (bf16 transport) alike, and computes
    in fp32 there; with column scales it stays eager."""
    from torchtitan.optimizers.disco import presets

    used = []
    real = presets._amphione_compiled
    monkeypatch.setattr(
        presets, "_amphione_compiled", lambda name: used.append(name) or real(name)
    )
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    W = torch.randn(2, 128, 64, device=dev, generator=gen) / 8
    G = 1e-2 * torch.randn(2, 128, 64, device=dev, generator=gen)
    ctx = StepCtx(0.02, 0.0, 1, momentum=0.95, nesterov=True)
    cm = rule_of({"rule": "amphione", "args": {"compute": "svd"}}).compute
    out = {}
    for dt in (torch.float32, torch.bfloat16):
        S = torch.zeros(2, *cm.state_shape(128, 64), device=dev)
        used.clear()
        res = cm.run(G.to(dt), W, S, ctx)
        assert used == ["prepare", "finish"]
        assert res.weight.dtype == W.dtype and res.state.dtype == S.dtype
        out[dt] = res.weight
    eager = cm._run_eager(G.to(torch.bfloat16), W, torch.zeros_like(S), ctx).weight
    torch.testing.assert_close(out[torch.bfloat16], eager, rtol=1e-3, atol=1e-5)
    cols = rule_of(
        {"rule": "amphione", "args": {"compute": "svd", "column_scales": True}}
    ).compute
    used.clear()
    cols.run(
        G,
        W,
        torch.zeros(2, *cols.state_shape(128, 64), device=dev),
        ctx._replace(colmask=torch.ones(2, dtype=torch.bool, device=dev)),
    )
    assert used == []
