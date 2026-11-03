# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`amphione` (AmphioneMap, the final recipe) and `onerate` (the study's
legacy form on angular_muown, kept for ablations).

1. `onerate` with the stored recipe runs' args is the rule those runs used.
2. `amphione` steps bitwise as `onerate` (weights and state, with and without
   column gains), on the unit and inside DiSCO on fused parameters.
3. `amphione`'s step is the one its docstring writes down (a reference from
   those equations, exact polar in place of Polar Express)."""

import dataclasses
import os
import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout, qkv_units, stacked_units
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import (
    AMPHIONE_COLUMN_UNITS,
    AmphioneMap,
    EtaSchedule,
)
from torchtitan.optimizers.disco.rules import Compute, Rule, rule_of, StepCtx

TABLE = {"shape": "table", "points": [[0, 1.0], [4, 0.5], [8, 0.0]]}

# job_config optimizer.groups.default.args of the stored recipe run
# v2ps2-f-row-row-off-w512-32x-53.7B-bs.256-...-gains.sign0.5-seed.1-stats (eta table shortened)
RECIPE = dict(
    compute="gram_polar_express",
    steps=6,
    scale="one",
    warmup=0,
    c=0.0,
    eta=TABLE,
    native_norm=False,
    sphere="row",
    magnitude_shape="row",
    tangent=False,
    radius=4.0,
    magnitude_lr=1.0,
    magnitude_rule="sign",
    polar_express_safety=1.05,  # the stored configs spelled it `safety`
    gain_lr="tied",
)
# the stored column run v2ps2-f-row-row-off-w1024-2x-13.4B-bs.128-...-col.on-...-colk.0.0625-cu.in-...
COLUMNS = dict(
    col_scale=True, col_gain_lr_mult=0.0625, col_units=list(AMPHIONE_COLUMN_UNITS)
)
STUDY = {
    "eta": TABLE,
    "radius": 4.0,
    "polar_express_safety": 1.05,
}  # onerate's radius default is 4
STUDY_AMPHIONE = {"eta": TABLE, "polar_express_safety": 1.05, "gain_beta": 0.9}
# the stepping tests: no quack kernel on CPU (Gram Polar Express), so both rules
# take the exact SVD polar. amphione has no step-1 rescale: onerate at radius 1.
STEP = {"eta": TABLE, "radius": 1.0, "compute": "svd"}
STEP_AMPHIONE = {"eta": TABLE, "gain_beta": 0.9, "compute": "svd"}
# amphione's default gain_beta (None) is beta: onerate's lengths average at 0.95 too
STEP_DEFAULT = {**STEP, "betas": (0.95, 0.95)}
STEP_AMPHIONE_DEFAULT = {"eta": TABLE, "compute": "svd"}


def _pair(columns, onerate_args, amphione_args):
    cols = COLUMNS if columns else {}
    return (
        rule_of({"rule": "onerate", "args": {**onerate_args, **cols}}),
        rule_of({"rule": "amphione", "args": {**amphione_args, "columns": columns}}),
    )


def _rules(columns):
    study = rule_of(
        {"rule": "angular_muown", "args": {**RECIPE, **(COLUMNS if columns else {})}}
    )
    return (study, *_pair(columns, STUDY, STUDY_AMPHIONE))


@pytest.mark.parametrize("columns", [False, True])
def test_onerate_is_the_study_recipe(columns):
    study, onerate, _ = _rules(columns)
    assert onerate == study


@pytest.mark.parametrize("columns", [False, True])
def test_amphione_has_the_study_direction(columns):
    study, _, amphione = _rules(columns)
    assert amphione.compute.direction == study.compute.direction


def test_amphione_defaults():
    cm = rule_of({"rule": "amphione"}).compute
    assert isinstance(cm, AmphioneMap)
    assert (cm.eta, cm.columns, cm.col_units) == (None, False, ())
    assert (cm.beta, cm.nesterov, cm.gain_beta) == (0.95, True, None)
    assert (cm.column_units, cm.column_lr_mult) == (AMPHIONE_COLUMN_UNITS, 0.0625)
    assert cm.direction.compute.backend == "gram_polar_express"
    assert rule_of(
        {"rule": "amphione", "args": {"polar_express_safety": 1.01}}
    ) == rule_of({"rule": "amphione"})
    assert rule_of(
        {"rule": "amphione", "args": {"columns": True}}
    ).compute.col_units == (AMPHIONE_COLUMN_UNITS)


@pytest.mark.parametrize("compute", ["polar_express", "newtonschulz5", "svd"])
def test_amphione_takes_any_whole_unit_orthogonaliser(compute):
    onerate = rule_of({"rule": "onerate", "args": {"compute": compute}})
    amphione = rule_of({"rule": "amphione", "args": {"compute": compute}})
    assert amphione.compute.direction == onerate.compute.direction


def test_amphione_refuses_a_gram_compute_and_a_row_direction_without_a_row_compute():
    with pytest.raises(ValueError, match="Gram compute"):
        rule_of({"rule": "amphione", "args": {"compute": "gram_polar"}})
    with pytest.raises(ValueError, match="row_direction needs a row compute"):
        rule_of({"rule": "amphione", "args": {"row_direction": True}})


# ------------------------------------------------------- row computes: row_direction
# the stored conv group's args (q, k, v conv weights) of
# v2ps2-f-row-row-off-w1024-2x-13.4B-bs.128-...-cv.dlncq-cvg.1-gains.sign0.5-seed.1-stats
CONV_RECIPE = dict(
    compute="row-l2",
    steps=6,
    scale="one",
    warmup=0,
    c=0.0,
    native_norm=False,
    sphere="row",
    magnitude_shape="row",
    tangent=False,
    radius=4.0,
    magnitude_lr=1.0,
    magnitude_rule="sign",
    gain_lr="tied",
    row_direction=True,
)


def test_onerate_is_the_study_conv_rule():
    assert rule_of({"rule": "onerate", "args": dict(CONV_RECIPE)}) == rule_of(
        {"rule": "angular_muown", "args": dict(CONV_RECIPE)}
    )


@pytest.mark.parametrize("scale", ["one", "sqrt_n", "inv_sqrt_n"])
def test_a_row_compute_without_row_direction_is_the_plain_row_rule(scale):
    """An embedding or head group that inherits `rule: amphione` and sets its own
    row compute gets that row rule, as under hyperball / elr / angular_muown."""
    args = {"compute": "row-l2", "scale": scale}
    amphione = rule_of({"rule": "amphione", "args": args})
    assert amphione.kind == "rows"
    assert not isinstance(amphione.compute, AmphioneMap)
    assert amphione == rule_of({"rule": "onerate", "args": args})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_row_direction_steps_bitwise_as_the_study_conv_rule(dtype):
    """row-l2 as the direction (row_direction) on [C, 4] conv-like units, against
    onerate with the stored conv group's args at radius 1 (amphione has no step-1
    rescale) and the same row-length average."""
    onerate = rule_of({"rule": "onerate", "args": {**CONV_RECIPE, "radius": 1.0}})
    amphione = rule_of(
        {
            "rule": "amphione",
            "args": {"compute": "row-l2", "row_direction": True, "gain_beta": 0.9},
        }
    )
    assert isinstance(amphione.compute, AmphioneMap)
    w, grads = _case(units=3, m=16, n=4, dtype=dtype)
    for (wa, sa), (wb, sb) in zip(
        _run(amphione.compute, w, grads), _run(onerate.compute, w, grads)
    ):
        assert torch.equal(wa, wb) and torch.equal(sa, sb)


def _case(units=3, m=16, n=8, steps=9, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(units, m, n, generator=gen, dtype=dtype) / n**0.5
    w[1, 3] = 0.0  # a zero row starts at sqrt(0.33)
    grads = [
        0.1 * torch.randn(units, m, n, generator=gen, dtype=dtype) for _ in range(steps)
    ]
    return w, grads


def _run(cm, w, grads, colmask=None, lr=0.03):
    state = w.new_zeros(w.shape[0], *cm.state_shape(*w.shape[1:]))
    out = []
    for t, G in enumerate(grads, start=1):
        res = cm.run(G, w, state, StepCtx(lr, 0.0, t, colmask=colmask))
        w, state = res.weight, res.state
        out.append((w, state))
    return out


@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "onerate_args,amphione_args",
    [(STEP, STEP_AMPHIONE), (STEP_DEFAULT, STEP_AMPHIONE_DEFAULT)],
    ids=["gain_beta 0.9", "gain_beta default"],
)
def test_amphione_steps_bitwise_as_onerate(columns, dtype, onerate_args, amphione_args):
    onerate, amphione = _pair(columns, onerate_args, amphione_args)
    w, grads = _case(dtype=dtype)
    mask = torch.tensor([True, False, True]) if columns else None
    for (wa, sa), (wb, sb) in zip(
        _run(amphione.compute, w, grads, mask), _run(onerate.compute, w, grads, mask)
    ):
        assert torch.equal(wa, wb) and torch.equal(sa, sb)


def test_amphione_columns_need_the_mask():
    _, amphione = _pair(True, STEP, STEP_AMPHIONE)
    w, grads = _case()
    with pytest.raises(ValueError, match="mask"):
        _run(amphione.compute, w, grads[:1])


# ------------------------------------------------------- the step from its equations
@dataclasses.dataclass(frozen=True)
class _Polar(Compute):
    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return u @ vh


LR, BETA = 0.05, 0.95  # the lengths average at beta (gain_beta default)
ETA = EtaSchedule("sqrt_decay", start=0, total=12)


def _reference(w, grads, columns):
    """The docstring's step, unit by unit."""
    out = []
    for W0, Gs in zip(w, zip(*grads)):
        g = W0.norm(dim=-1)
        U = W0 / g[:, None]
        c = torch.ones(W0.shape[-1], dtype=W0.dtype)
        M = torch.zeros_like(W0)
        mg, mc = torch.zeros_like(g), torch.zeros_like(c)
        unit = []
        for t, G in enumerate(Gs, start=1):
            eta_t = LR * ETA(t)
            if columns:
                gamma_c = (G * (g[:, None] * U)).sum(dim=0)
                G = G * c[None, :]
            gamma = (G * U).sum(dim=-1)
            D = g[:, None] * G
            M = BETA * M + D
            u, _, vh = torch.linalg.svd(D + BETA * M, full_matrices=False)
            O = u @ vh
            O = O * U.norm() / O.norm()
            V = U - eta_t * O
            U = V / V.norm(dim=-1, keepdim=True)
            mg = BETA * mg + (1 - BETA) * gamma
            g = g - eta_t * torch.sign(mg)
            if columns:
                mc = BETA * mc + (1 - BETA) * gamma_c
                c = c - eta_t / 16 * torch.sign(mc)
            unit.append(g[:, None] * U * c[None, :])
        out.append(unit)
    return [torch.stack(ws) for ws in zip(*out)]


@pytest.mark.parametrize("columns", [False, True])
def test_amphione_step_is_its_equations(columns):
    rule = rule_of(
        {
            "rule": "amphione",
            "args": {
                "eta": {"shape": "sqrt_decay", "start": 0, "total": 12},
                "columns": columns,
            },
        }
    )
    cm = dataclasses.replace(rule.compute, direction=Rule((), _Polar(), scale="one"))
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(3, 12, 8, generator=gen, dtype=torch.float64) / 8**0.5
    grads = [
        0.1 * torch.randn(3, 12, 8, generator=gen, dtype=torch.float64)
        for _ in range(6)
    ]
    mask = torch.ones(3, dtype=torch.bool) if columns else None
    for (got, _), want in zip(
        _run(cm, w, grads, mask, lr=LR), _reference(w, grads, columns)
    ):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)


# ------------------------------------------------------- in DiSCO: fused parameters
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


def _disco(rule, args, seed=0):
    g = torch.Generator().manual_seed(seed)
    params = [nn.Parameter(torch.randn(*s, generator=g) * 0.3) for s in SHAPES]
    opt = DiSCO(
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
        lr=0.03,
        rule=rule,
        args=args,
        param_layouts=LAYOUTS,
    )
    return opt, params


@pytest.mark.parametrize("columns", [False, True])
def test_disco_amphione_is_onerate(one_rank, columns):
    cols = COLUMNS if columns else {}
    a, pa = _disco("amphione", {**STEP_AMPHIONE, "columns": columns})
    b, pb = _disco("onerate", {**STEP, **cols})
    for s in range(6):
        g = torch.Generator().manual_seed(100 + s)
        grads = [0.1 * torch.randn(*shape, generator=g) for shape in SHAPES]
        for ps in (pa, pb):
            for p, gr in zip(ps, grads):
                p.grad = gr.clone()
        a.step()
        b.step()
        for x, y in zip(pa, pb):
            assert torch.equal(x, y)
            assert torch.equal(a.state[x]["rule_state"], b.state[y]["rule_state"])
        assert a.eta_used == b.eta_used
    if columns:  # the column-gain logging reads amphione's state as onerate's
        assert a.col_gain_metrics(a.col_gain_snapshot()) == b.col_gain_metrics(
            b.col_gain_snapshot()
        )


# ------------------------------------------- frozen row lengths (the MoE router gate)
FROZEN = (AMPHIONE_COLUMN_UNITS[1], AMPHIONE_COLUMN_UNITS[3])  # wk in wqkv, w1 in w13


def _run_frozen(cm, w, grads, frozen, lr=0.03):
    state = w.new_zeros(w.shape[0], *cm.state_shape(*w.shape[1:]))
    out = []
    for t, G in enumerate(grads, start=1):
        res = cm.run(G, w, state, StepCtx(lr, 0.0, t, frozen=frozen))
        w, state = res.weight, res.state
        out.append((w, state))
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_amphione_frozen_lengths_step_bitwise_as_onerate(dtype):
    onerate = rule_of({"rule": "onerate", "args": {**STEP, "frozen_units": ["u"]}})
    amphione = rule_of(
        {"rule": "amphione", "args": {**STEP_AMPHIONE, "frozen_length_units": ["u"]}}
    )
    w, grads = _case(dtype=dtype)
    mask = torch.tensor([True, False, True])
    out = _run_frozen(amphione.compute, w, grads, mask)
    for (wa, sa), (wb, sb) in zip(out, _run_frozen(onerate.compute, w, grads, mask)):
        assert torch.equal(wa, wb) and torch.equal(sa, sb)
    n = w.shape[-1]
    start = out[0][1][..., n]  # g after step 1: the frozen units' start lengths
    assert torch.equal(start[0], w[0].norm(dim=-1)) and torch.equal(
        start[2], w[2].norm(dim=-1)
    )
    for wa, sa in out:
        assert torch.equal(sa[0, :, n], start[0]) and torch.equal(sa[2, :, n], start[2])
    assert not torch.equal(out[-1][1][1, :, n], start[1])  # the other unit learns g
    assert not torch.equal(out[-1][0][0], w[0])  # the frozen unit's rows still turn


@pytest.mark.parametrize(
    "args,match",
    [
        ({"frozen_length_units": ["u", "u"]}, "distinct unit types"),
        ({"frozen_length_units": [""]}, "distinct unit types"),
        ({"frozen_length_units": ["u"], "columns": True}, "with columns"),
    ],
)
def test_amphione_frozen_lengths_refusals(args, match):
    with pytest.raises(ValueError, match=match):
        rule_of({"rule": "amphione", "args": {**STEP_AMPHIONE, **args}})


def test_amphione_frozen_lengths_need_the_mask():
    amphione = rule_of(
        {"rule": "amphione", "args": {**STEP_AMPHIONE, "frozen_length_units": ["u"]}}
    )
    w, grads = _case()
    with pytest.raises(ValueError, match="needs the stack's mask"):
        _run_frozen(amphione.compute, w, grads[:1], None)


def test_amphione_frozen_lengths_are_the_groups_own():
    from torchtitan.optimizers.disco.rules import RULES

    assert "frozen_length_units" not in RULES["amphione"].inherits
    assert RULES["amphione"].args["frozen_length_units"] == ()


def test_disco_amphione_frozen_lengths_is_onerate(one_rank):
    a, pa = _disco("amphione", {**STEP_AMPHIONE, "frozen_length_units": list(FROZEN)})
    b, pb = _disco("onerate", {**STEP, "frozen_units": list(FROZEN)})
    c, pc = _disco("amphione", STEP_AMPHIONE)  # nothing frozen
    for s in range(6):
        g = torch.Generator().manual_seed(100 + s)
        grads = [0.1 * torch.randn(*shape, generator=g) for shape in SHAPES]
        for ps in (pa, pb, pc):
            for p, gr in zip(ps, grads):
                p.grad = gr.clone()
        a.step()
        b.step()
        c.step()
        for x, y in zip(pa, pb):
            assert torch.equal(x, y)
            assert torch.equal(a.state[x]["rule_state"], b.state[y]["rule_state"])
        assert a.eta_used == b.eta_used
    # the mask reached the rule: the fused wqkv and w13 differ from the unfrozen run, wo and w2 do not
    assert [torch.equal(x, z) for x, z in zip(pa, pc)] == [False, True, False, True]
