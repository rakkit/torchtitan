# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""optimizer.log_gram_input: sigma_1 / ||X||_F of the exact tensor each
gram_polar_express call receives, per DiSCO unit, on spectral logging steps;
off by default and training unchanged either way. The backend is replaced by a
spy (an SVD polar, CPU-safe) that keeps every input it receives."""

import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout, RunList, UnitSpec
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.kernels import zeropower


@pytest.fixture(scope="module")
def one_rank():
    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _fresh_compiled_rules():
    """The spy backend gets traced into the cached compiled rule functions;
    drop them around each test so no other test reuses a spy's graph."""
    from torchtitan.optimizers.disco import rules as _rules

    def clear():
        _rules._COMPILED.clear()
        torch._dynamo.reset()

    clear()
    yield
    clear()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


RULES = {
    "muon": dict(
        rule="muon",
        args={"compute": "gram_polar_express", "steps": 6},
        group=dict(momentum=0.9, nesterov=True),
    ),
    "hyperball": dict(
        rule="hyperball",
        args={"compute": "gram_polar_express", "steps": 6},
        group=dict(momentum=0.95, nesterov=True),
    ),
    "amphione": dict(
        rule="amphione",
        args=dict(compute="gram_polar_express", steps=6),
        group=dict(momentum=0.0, nesterov=False, weight_decay=0.0),
    ),
}
UNITS = (
    UnitSpec("a", RunList(((0, 8),)), (8, 8)),
    UnitSpec("b", RunList(((8, 8),)), (8, 8)),
)


def _svd_polar(x, steps=5, eps=1e-7, safety=1.01):
    u, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    return (u @ vh).to(x.dtype)


def _run(case, log, spectral_steps=(1, 2, 3), steps=3, spy=None):
    gen = torch.Generator().manual_seed(5)
    W = torch.randn(16, 8, generator=gen)
    grads = [0.1 * torch.randn(16, 8, generator=gen) for _ in range(steps)]
    w = nn.Parameter(W.clone())
    r = RULES[case]
    opt = DiSCO(
        [{"params": [w], "param_names": ["w"], **r["group"]}],
        parallel_dims=_NoParallel(),
        lr=0.02,
        rule=r["rule"],
        args=dict(r["args"]),
        param_layouts={"w": ParamLayout("matrix", units=UNITS)},
    )
    opt.log_gram_input = log
    logs = []
    for t, G in enumerate(grads, start=1):
        sp = t in spectral_steps
        opt.calculate_norm_at_next_step(
            norms_to_log=["frobenius_norm"],
            gram_level=0,
            track_spectrum=False,
            spectral=sp,
        )
        if spy is not None:
            spy.clear()
        w.grad = G.clone()
        opt.step()
        logs.append(
            (
                dict(opt.get_norms_at_current_step()),
                list(spy) if spy is not None else [],
            )
        )
    return w.detach().clone(), logs


def _s1f(x):
    x = x.double()
    return float(torch.linalg.matrix_norm(x, ord=2) / x.norm())


@pytest.mark.parametrize("case", list(RULES))
def test_each_units_gram_input_is_logged_on_spectral_steps_only(
    one_rank, case, monkeypatch
):
    from torchtitan.optimizers.disco import gram_input_stats as gis

    monkeypatch.setattr(
        gis, "POWER_ITERS", 2000
    )  # converged on these small flat units: compare to the SVD
    seen = []

    def spy(x, steps=5, eps=1e-7, safety=1.01):
        seen.append(x.detach().clone())
        return _svd_polar(x, steps, eps, safety)

    monkeypatch.setitem(zeropower.zeropower_backends, "gram_polar_express", spy)
    w_on, on = _run(case, True, spectral_steps=(1, 3), spy=seen)
    w_off, off = _run(case, False, spectral_steps=(1, 3))
    assert torch.equal(w_on, w_off)  # training unchanged
    assert not any(k.startswith("gram_input") for got, _ in off for k in got)
    for t, (got, inputs) in enumerate(on, start=1):
        keys = {k for k in got if k.startswith("gram_input")}
        if t == 2:
            assert not keys, t  # not a spectral step
            continue
        # every input the backend received, per unit (stacks of the two units, in call order)
        per_unit = [[], []]
        for x in inputs:
            x = x if x.ndim == 3 else x[None]
            if x.shape[0] == 2:
                for j in range(2):
                    per_unit[j].append(_s1f(x[j]))
        unit_keys = sorted(
            k.split("/", 1)[1] for k in keys if k.startswith("gram_input_s1f/")
        )
        assert len(unit_keys) == 2, unit_keys
        for j, uk in enumerate(unit_keys):
            assert got[f"gram_input_s1f/{uk}"] == pytest.approx(
                per_unit[j][0], rel=1e-5
            )
            assert got[f"gram_input_s1f_max/{uk}"] == pytest.approx(
                max(per_unit[j]), rel=1e-5
            )
        n_band = sum(0.94 <= max(v) <= 0.955 for v in per_unit)
        assert (
            got["gram_input_band/count"] == n_band and got["gram_input_band/units"] == 2
        )


def test_the_band_count(one_rank, monkeypatch):
    """A unit built to sigma_1 / F = 0.95 is counted; the other is not."""
    from torchtitan.optimizers.disco import gram_input_stats as gis

    a = torch.zeros(8, 8)
    a[0, 0], a[1, 1] = 0.95, (1 - 0.95**2) ** 0.5
    b = torch.eye(8)
    gis.arm()
    with gis.units(["u.a", "u.b"]):
        from torchtitan.optimizers.disco.presets import Orthogonalise

        gis.record(Orthogonalise("gram_polar_express", 6), torch.stack([a, b]))
    rec = gis.disarm()
    assert rec["u.a"][0].item() == pytest.approx(0.95, rel=1e-6)
    assert rec["u.b"][0].item() == pytest.approx(8**-0.5, rel=1e-6)
    assert not gis.is_armed()
    gis.record(Orthogonalise("gram_polar_express", 6), a[None])  # disarmed: nothing
    assert gis.disarm() == {}


def test_the_default_power_iteration_is_exact_where_sigma_1_dominates():
    """In and near the band sigma_1 dominates: 30 iterations match the SVD."""
    from torchtitan.optimizers.disco import gram_input_stats as gis

    gen = torch.Generator().manual_seed(1)
    x = torch.randn(6, 256, 64, generator=gen)
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    s[:, 0] = s[:, 1:].norm(dim=-1) * torch.tensor([3.0, 3.2, 3.4, 4.0, 1.0, 0.5])[
        :, None
    ].squeeze(-1)
    x = u @ torch.diag_embed(s) @ vh
    want = (
        torch.linalg.matrix_norm(x.double(), ord=2) / x.double().norm(dim=(-2, -1))
    ).float()
    got = gis.s1f(x)
    for i in range(4):  # sigma_1 / F 0.95 .. 0.97
        assert got[i].item() == pytest.approx(want[i].item(), rel=1e-5)
    assert (got <= want * (1 + 1e-5)).all()  # a lower bound everywhere
