# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Column gains on chosen unit types only: `col_units` of angular_muown col_scale (ps2r gate route A2).

From the definition: with col_scale and col_units, a unit whose type is in col_units is the all-matrix
col_scale rule (W = Diag(g) U Diag(c), c learned by g's rule); every other unit keeps c = 1 and its column
moments, i.e. it is the rule WITHOUT col_scale (rows only). Both references are the same class run
without the option: one with col_units empty (all units learn c), one with col_scale False. DiSCO maps the
type names (`feed_forward.w1`, `attention.qkv_linear.wq`, ...) to the stack's mask, as frozen_units does."""

import copy
import os
import socket
import sys  # noqa: F401
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout, qkv_units, stacked_units
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import AngularMuownMap, EtaSchedule
from torchtitan.optimizers.disco.rules import Compute, Rule, rule_of, StepCtx

LR, MAG_LR, M_START = 0.05, 0.3, 4.0
ETA = EtaSchedule("inverse_poly", start=1, c=0.5)
Q, K, V, W1, W3, W2, WO = (
    "attention.qkv_linear.wq",
    "attention.qkv_linear.wk",
    "attention.qkv_linear.wv",
    "feed_forward.w1",
    "feed_forward.w3",
    "feed_forward.w2",
    "attention.wo",
)


@dataclass(frozen=True)
class _Polar(Compute):
    k: float = 0.5

    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return self.k * (u @ vh)


def _map(col_units=(), col_scale=True, rule="sign", tied=True, k=0.0625):
    """OneRate's column form (n4): sign length rule, gain_lr tied, column rate k x the turn."""
    extra = dict(gain_lr="tied", col_gain_lr_mult=k) if (tied and col_scale) else {}
    return AngularMuownMap(
        Rule((), _Polar(0.5), scale="one"),
        warmup=0,
        c=0.0,
        eta=ETA,
        sphere="row",
        tangent=True,
        native_norm=False,
        radius=M_START,
        magnitude_lr=1.0,
        magnitude_rule=rule,
        col_scale=col_scale,
        col_units=col_units,
        **extra,
    )


def _stack(m=12, n=8, units=3, steps=7, seed=1):
    gen = torch.Generator().manual_seed(seed)
    w = torch.stack(
        [
            torch.linalg.qr(torch.randn(m, m, generator=gen, dtype=torch.float64))[0][
                :, :n
            ]
            * torch.logspace(-0.3, 0.3, m, dtype=torch.float64)[:, None]
            for _ in range(units)
        ]
    )
    return w, [
        0.1 * torch.randn(units, m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]


def _run(cm, w, grads, colmask=None):
    state = w.new_zeros(w.shape[0], *cm.state_shape(*w.shape[1:]))
    out = []
    for t, G in enumerate(grads, start=1):
        res = cm.run(G, w, state, StepCtx(LR, 0.0, t, colmask=colmask))
        w, state = res.weight, res.state
        out.append((w, state))
    return out


@pytest.mark.parametrize("rule", ["sign", "adam"])
@pytest.mark.parametrize(
    "mask",
    [
        (True, False, False),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_masked_in_units_are_the_all_matrix_rule_and_masked_out_units_are_the_rows_only_rule(
    rule, mask
):
    w, grads = _stack()
    cm = _map(col_units=(W1,), rule=rule)
    got = _run(cm, w, grads, torch.tensor(mask))
    all_cols = _run(_map(rule=rule), w, grads)
    # the rows-only class has the same turn / length rates (gain_lr tied is a rows rule too)
    no_cols = _run(
        AngularMuownMap(
            Rule((), _Polar(0.5), scale="one"),
            warmup=0,
            c=0.0,
            eta=ETA,
            sphere="row",
            tangent=True,
            native_norm=False,
            radius=M_START,
            magnitude_lr=1.0,
            magnitude_rule=rule,
            gain_lr="tied",
        ),
        w,
        grads,
    )
    n = w.shape[-1]
    for t, ((wg, sg), (wa, sa), (wn, sn)) in enumerate(
        zip(got, all_cols, no_cols), start=1
    ):
        for j, on in enumerate(mask):
            if on:
                assert torch.equal(wg[j], wa[j]) and torch.equal(sg[j], sa[j]), (
                    t,
                    j,
                    "columns on",
                )
            else:
                assert torch.equal(wg[j], wn[j]), (
                    t,
                    j,
                    "columns off: the rows-only weights",
                )
                m = w.shape[1]
                assert torch.equal(sg[j, :m], sn[j]), (t, j, "rows state unchanged")
                assert torch.equal(sg[j, m, :n], torch.ones(n, dtype=sg.dtype)), (
                    t,
                    j,
                    "c stays 1",
                )
                assert torch.count_nonzero(sg[j, m + 1 :]) == 0, (
                    t,
                    j,
                    "column moments stay 0",
                )


def test_the_columns_of_a_masked_in_unit_move_and_the_masked_out_ones_do_not():
    w, grads = _stack(steps=5)
    got = _run(_map(col_units=(W1,)), w, grads, torch.tensor([True, False, True]))
    m, n = w.shape[1], w.shape[2]
    c = got[-1][1][:, m, :n]
    assert float((c[0] - 1).abs().max()) > 1e-3 and float((c[2] - 1).abs().max()) > 1e-3
    assert torch.equal(c[1], torch.ones(n, dtype=c.dtype))


def test_a_mask_is_required_and_the_settings_are_checked():
    w, grads = _stack(steps=1)
    with pytest.raises(ValueError, match="col_units needs the stack's mask"):
        _run(_map(col_units=(W1,)), w, grads, None)
    with pytest.raises(ValueError, match="col_units needs the stack's mask"):
        _run(_map(col_units=(W1,)), w, grads, torch.tensor([True, False]))
    with pytest.raises(
        ValueError, match="col_units picks the units with column gains: need col_scale"
    ):
        _map(col_units=(W1,), col_scale=False)
    with pytest.raises(ValueError, match="col_units: distinct unit types"):
        _map(col_units=(W1, W1))
    with pytest.raises(ValueError, match="col_units: distinct unit types"):
        _map(col_units=("",))


def test_the_preset_builds_it():
    cm = rule_of(
        {
            "rule": "angular_muown",
            "args": {
                "compute": "svd",
                "scale": "one",
                "warmup": 0,
                "c": 0.0,
                "native_norm": False,
                "tangent": True,
                "radius": M_START,
                "magnitude_lr": 1.0,
                "col_scale": True,
                "col_units": [W3, W2],
            },
        }
    ).compute
    assert cm.col_units == (W3, W2) and cm.state_shape(12, 8) == (15, 11)


# ---------------------------------------------------------------- in DiSCO: fused parameters
@pytest.fixture(scope="module")
def one_rank():
    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


D, H, KV, HD, FF = 8, 2, 2, 4, 16
NAMES = [
    "layers.0.attention.qkv_linear.wqkv.weight",
    "layers.0.attention.wo.weight",
    "layers.0.feed_forward.w13.weight",
    "layers.0.feed_forward.w2.weight",
]
LAYOUTS = {
    NAMES[0]: ParamLayout("matrix", units=qkv_units(H, KV, HD, D)),
    NAMES[1]: ParamLayout("matrix"),
    NAMES[2]: ParamLayout("matrix", units=stacked_units(("w1", "w3"), FF, D)),
    NAMES[3]: ParamLayout("matrix"),
}
SHAPES = [((H + 2 * KV) * HD, D), (D, H * HD), (2, FF, D), (D, FF)]
_ARGS = {
    "compute": "svd",
    "scale": "one",
    "warmup": 0,
    "c": 0.0,
    "native_norm": False,
    "radius": 4.0,
    "sphere": "row",
    "tangent": True,
    "magnitude_rule": "sign",
    "gain_lr": "tied",
    "magnitude_lr": 1.0,
    "col_scale": True,
    "col_gain_lr_mult": 0.0625,
    "eta": {"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0},
}
UNITS = {"wq": Q, "wk": K, "wv": V, "w1": W1, "w3": W3}


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.randn(*s, generator=g) * 0.3) for s in SHAPES]


def _disco(params, col_units=None):
    args = dict(_ARGS)
    if col_units is not None:
        args["col_units"] = list(col_units)
    return DiSCO(
        [
            {
                "params": params,
                "param_names": NAMES,
                "momentum": 1.0,
                "nesterov": False,
                "weight_decay": 0.0,
            }
        ],
        parallel_dims=_NoParallel(),
        lr=LR,
        rule="angular_muown",
        args=args,
        param_layouts=LAYOUTS,
    )


def _steps(opt, params, seeds):
    for s in seeds:
        g = torch.Generator().manual_seed(100 + s)
        for p, shape in zip(params, SHAPES):
            p.grad = 0.1 * torch.randn(*shape, generator=g)
        opt.step()


def _learned(opt, p):
    """Per unit position of parameter p: max |c - 1| of its column state."""
    e = next(x for x in opt.entries if x.param is p)
    rs = opt.state[p]["rule_state"]
    m = rs.shape[1] - 3
    return {
        u.name: float((rs[u.pos, m, : p.shape[-1]] - 1).abs().max()) for u in e.units
    }


@pytest.mark.parametrize(
    "sel,learned",
    [
        ((W1,), {("w13", "w1")}),
        ((W3, W2), {("w13", "w3"), ("w2", "")}),
        ((Q, K, V, WO), {("wqkv", "wq"), ("wqkv", "wk"), ("wqkv", "wv"), ("wo", "")}),
    ],
)
def test_disco_learns_columns_on_exactly_the_named_unit_types(one_rank, sel, learned):
    pa = _params()
    a = _disco(pa, sel)
    _steps(a, pa, range(5))
    got = set()
    for i, key in ((0, "wqkv"), (1, "wo"), (2, "w13"), (3, "w2")):
        for name, dev in _learned(a, pa[i]).items():
            if dev > 1e-6:
                got.add((key, name if key in ("wqkv", "w13") else ""))
    assert got == learned, got
    # the all-matrix run learns every unit's columns
    pb = _params()
    b = _disco(pb)
    _steps(b, pb, range(5))
    assert all(v > 1e-6 for i in range(4) for v in _learned(b, pb[i]).values())


def test_a_selection_equals_the_all_matrix_run_on_its_units_only(one_rank):
    """The masked-in units' weights after 5 steps are those of the unmasked all-matrix run (each unit's step is
    independent of the others); the masked-out units' weights differ from it."""
    pa, pb = _params(), _params()
    a, b = _disco(pa, (W1,)), _disco(pb)
    _steps(a, pa, range(5))
    _steps(b, pb, range(5))

    def rows(opt, p, name):
        e = next(x for x in opt.entries if x.param is p)
        u = next(u for u in e.units if u.name == name)
        W = p.detach().reshape(-1, p.shape[-1])
        return torch.cat([W[a_ : a_ + n] for a_, n in u.rows.runs])

    assert torch.equal(rows(a, pa[2], "w1"), rows(b, pb[2], "w1"))
    assert float((rows(a, pa[2], "w3") - rows(b, pb[2], "w3")).abs().max()) > 1e-5
    assert float((pa[0].detach() - pb[0].detach()).abs().max()) > 1e-5


def test_disco_refuses_a_type_that_matches_no_unit(one_rank):
    with pytest.raises(ValueError, match="col_units .* match no unit"):
        _disco(_params(), ("feed_forward.w9",))


def test_a_resumed_run_with_col_units_is_the_continuous_one(one_rank):
    pa = _params()
    a = _disco(pa, (W3, W2))
    _steps(a, pa, range(3))
    saved = [p.detach().clone() for p in pa]
    sd = copy.deepcopy(a.state_dict())
    _steps(a, pa, range(3, 5))
    pb = [nn.Parameter(s.clone()) for s in saved]
    b = _disco(pb, (W3, W2))
    b.load_state_dict(sd)
    _steps(b, pb, range(3, 5))
    assert all(torch.equal(x.detach(), y.detach()) for x, y in zip(pa, pb))
