# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""sso's per-unit-type step factors (`unit_eta`): the angle-matched spectral cell's per-type lrs inside fused
parameters (wqkv's wq / wk / wv, w13's w1 / w3), which one group lr cannot give."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
from torchtitan.models.common.param_layout import ParamLayout, qkv_units, stacked_units
from torchtitan.optimizers.disco.optimizer import DiSCO, unit_rows_local
from torchtitan.optimizers.disco.presets import SpectralSphere
from torchtitan.optimizers.disco.rules import rule_of


@pytest.fixture(autouse=True, scope="module")
def _process_group():
    if not dist.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


D, H, HD, F = 8, 2, 4, 16
NAMES = [
    "layers.0.attention.qkv_linear.wqkv.weight",
    "layers.0.attention.wo.weight",
    "layers.0.feed_forward.w13.weight",
    "layers.0.feed_forward.w2.weight",
]
LAYOUTS = {
    NAMES[0]: ParamLayout("matrix", units=qkv_units(H, H, HD, D)),
    NAMES[1]: ParamLayout("matrix"),
    NAMES[2]: ParamLayout("matrix", units=stacked_units(("w1", "w3"), F, D)),
    NAMES[3]: ParamLayout("matrix"),
}
FACTORS = {
    "attention.qkv_linear.wq": 0.9,
    "attention.qkv_linear.wk": 1.0,
    "attention.qkv_linear.wv": 1.2,
    "attention.wo": 1.0,
    "feed_forward.w1": 0.7,
    "feed_forward.w3": 1.3,
    "feed_forward.w2": 1.1,
}


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = [(3 * H * HD, D), (D, H * HD), (2, F, D), (D, F)]
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.3) for s in shapes]


def _opt(params, unit_eta="absent", **args):
    a = {"compute": "svd", "radius": 1.0, **args}
    if unit_eta != "absent":
        a["unit_eta"] = unit_eta
    return DiSCO(
        [{"params": params, "param_names": NAMES, "rule": "sso", "args": a}],
        parallel_dims=_NoParallel(),
        lr=0.02,
        momentum=1.0,
        nesterov=False,
        param_layouts=LAYOUTS,
    )


def _grads(seed):
    g = torch.Generator().manual_seed(100 + seed)
    return [
        torch.randn(*p, generator=g)
        for p in ((3 * H * HD, D), (D, H * HD), (2, F, D), (D, F))
    ]


def _step(opt, params, seed, record):
    for p, g in zip(params, _grads(seed)):
        p.grad = g.clone()
    orig = opt._apply

    def apply(u, *, sharded):
        record.update(
            {
                e.i: u[e.i].detach().clone()
                for es in [opt.entries]
                for e in es
                if e.i in u
            }
        )
        record["coef"] = {k: v.detach().clone() for k, v in opt._coef.items()}
        return orig(u, sharded=sharded)

    opt._apply = apply
    opt.step()
    opt._apply = orig


def _unit_type(e, un):
    key = e.clean.rsplit(".", 1)[0] + "." + un.name if un.name else e.clean
    return key.split("layers.0.", 1)[1]


def test_factors_scale_each_unit_of_the_fused_stacks_exactly():
    """Per unit, the applied update is the factor-free update times the unit type's factor (bitwise, one fp32
    multiply); alpha (the retraction) and the rule state are bitwise unchanged."""
    p0, pf = _params(), _params()
    o0, of = _opt(p0), _opt(pf, unit_eta=FACTORS)
    r0, rf = {}, {}
    _step(o0, p0, 0, r0)
    _step(of, pf, 0, rf)
    seen = set()
    for e0, ef in zip(o0.entries, of.entries):
        u0, uf = e0.rows2d(r0[e0.i]), ef.rows2d(rf[ef.i])
        for un in ef.units:
            t = _unit_type(ef, un)
            seen.add(t)
            a, b = unit_rows_local(u0, un.rows), unit_rows_local(uf, un.rows)
            f = torch.tensor(FACTORS[t], dtype=a.dtype)
            assert torch.equal(b, a * f), t
            assert torch.allclose(b / a, torch.full_like(a, FACTORS[t]), rtol=1e-6), t
        assert torch.equal(r0["coef"][e0.i], rf["coef"][ef.i])
        assert torch.equal(
            o0.state[e0.param]["rule_state"], of.state[ef.param]["rule_state"]
        )
    assert seen == set(FACTORS)


def test_an_empty_map_is_a_bitwise_no_op():
    pa, pb = _params(), _params()
    oa, ob = _opt(pa), _opt(pb, unit_eta={})
    for s in range(3):
        _step(oa, pa, s, {})
        _step(ob, pb, s, {})
    for a, b in zip(pa, pb):
        assert torch.equal(a, b)


def test_a_missing_or_unknown_key_raises():
    missing = {k: v for k, v in FACTORS.items() if k != "attention.qkv_linear.wv"}
    with pytest.raises(ValueError, match="no unit_eta factor"):
        _opt(_params(), unit_eta=missing)
    with pytest.raises(ValueError, match="matches no unit"):
        _opt(_params(), unit_eta={**FACTORS, "attention.qkv_linear.wz": 1.0})
    with pytest.raises(ValueError, match="unit_eta"):
        _opt(_params(), unit_eta={**FACTORS, "attention.wo": 0.0})


def test_unit_eta_needs_a_step_linear_in_eta(monkeypatch):
    """The hook fires only for a compute that declares its step linear in eta with an eta-free alpha (sso);
    any other compute carrying factors raises at build, and the presets without the arg reject it."""
    monkeypatch.setattr(SpectralSphere, "eta_linear", False)
    with pytest.raises(ValueError, match="linear in eta"):
        _opt(_params(), unit_eta=FACTORS)
    monkeypatch.setattr(SpectralSphere, "eta_linear", True)
    monkeypatch.setattr(
        SpectralSphere, "output", "weight"
    )  # a returned weight cannot be scaled
    with pytest.raises(ValueError, match="linear in eta"):
        _opt(_params(), unit_eta=FACTORS)
    for name in ("aus", "hyperball", "muon"):
        with pytest.raises(ValueError, match="unknown args"):
            rule_of({"rule": name, "args": {"compute": "svd", "unit_eta": FACTORS}})


def test_each_units_factor_is_logged_once_at_build(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="torchtitan.optimizers.disco.optimizer"):
        opt = _opt(_params(), unit_eta=FACTORS)
        _step(opt, opt.param_groups[0]["params"], 0, {})
    lines = [r.getMessage() for r in caplog.records if "unit_eta" in r.getMessage()]
    assert len(lines) == 1 and "7 units" in lines[0]
    for t, f in FACTORS.items():
        assert f"layers.0.{t} x{f!r}" in lines[0]
