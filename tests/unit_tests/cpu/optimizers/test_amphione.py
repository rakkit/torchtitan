# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`amphione` (AmphioneMap, the final recipe).

1. Its step is the one its docstring writes down (a reference from those
   equations, exact polar in place of Polar Express; and the row-l2 direction
   of the conv group).
2. Inside DiSCO on fused parameters (q / k / v of one wqkv, w1 / w3 of one w13)
   it is its compute run on each unit whole, with the masks from their
   definitions (column scales, frozen row lengths).
3. What it declares and refuses."""

import dataclasses
import os
import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout, qkv_units, stacked_units
from torchtitan.optimizers.disco.optimizer import _unit_type, DiSCO
from torchtitan.optimizers.disco.presets import (
    AMPHIONE_COLUMN_SCALE_UNITS,
    AmphioneMap,
    Orthogonalise,
    RowL2,
)
from torchtitan.optimizers.disco.rules import Compute, Rule, rule_of, StepCtx

TABLE = {"shape": "table", "points": [[0, 1.0], [4, 0.5], [8, 0.0]]}
# the stepping tests: no quack kernel on CPU (Gram Polar Express): the exact SVD polar
STEP_AMPHIONE = {"compute": "svd"}
LR, BETA = 0.05, 0.95  # the group's momentum: amphione's direction and its scales


def test_amphione_has_the_recipe_direction():
    """The production recipe: Gram Polar Express, 6 steps, safety 1.05, no scale."""
    rule = rule_of({"rule": "amphione", "args": {"polar_express_safety": 1.05}})
    assert rule.compute.direction == Rule(
        (), Orthogonalise("gram_polar_express", 6, polar_express_safety=1.05)
    )


def test_amphione_defaults():
    cm = rule_of({"rule": "amphione"}).compute
    assert isinstance(cm, AmphioneMap)
    assert (cm.column_scales, cm.col_units) == (False, ())
    assert cm.scale_nesterov is False
    assert (cm.column_scale_units, cm.column_scale_lr_mult) == (
        AMPHIONE_COLUMN_SCALE_UNITS,
        0.0625,
    )
    assert cm.direction.compute.backend == "gram_polar_express"
    assert rule_of(
        {"rule": "amphione", "args": {"polar_express_safety": 1.01}}
    ) == rule_of({"rule": "amphione"})
    assert rule_of(
        {"rule": "amphione", "args": {"column_scales": True}}
    ).compute.col_units == (AMPHIONE_COLUMN_SCALE_UNITS)


@pytest.mark.parametrize("arg", ["beta", "nesterov", "gain_beta"])
def test_amphione_takes_the_groups_momentum_not_its_own(arg):
    """One momentum per group: amphione reads the group's `momentum` and
    `nesterov` from the step context; it has no momentum args of its own."""
    with pytest.raises(ValueError, match="unknown args"):
        rule_of({"rule": "amphione", "args": {arg: 0.9}})


def test_amphione_has_no_eta_schedule():
    """The step size is the trainer's lr: an eta schedule is refused."""
    assert not hasattr(rule_of({"rule": "amphione"}).compute, "eta")
    with pytest.raises(ValueError, match="eta"):
        rule_of({"rule": "amphione", "args": {"eta": TABLE}})


@pytest.mark.parametrize("compute", ["polar_express", "newtonschulz5", "svd"])
def test_amphione_takes_any_whole_unit_orthogonaliser(compute):
    amphione = rule_of({"rule": "amphione", "args": {"compute": compute}})
    assert amphione.compute.direction == Rule((), Orthogonalise(compute, 6))


def test_amphione_refuses_a_gram_compute_and_row_direction():
    with pytest.raises(ValueError, match="Gram compute"):
        rule_of({"rule": "amphione", "args": {"compute": "gram_polar"}})
    with pytest.raises(
        ValueError, match="unknown args"
    ):  # no flag: a row compute is the row direction
        rule_of(
            {"rule": "amphione", "args": {"compute": "row-l2", "row_direction": True}}
        )


def _case(units=3, m=16, n=8, steps=9, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(units, m, n, generator=gen, dtype=dtype) / n**0.5
    w[1, 3] = 0.0  # a zero row starts at sqrt(0.33)
    grads = [
        0.1 * torch.randn(units, m, n, generator=gen, dtype=dtype) for _ in range(steps)
    ]
    return w, grads


def _run(cm, w, grads, colmask=None, lr=0.03, beta=BETA, nesterov=True):
    state = w.new_zeros(w.shape[0], *cm.state_shape(*w.shape[1:]))
    out = []
    for t, G in enumerate(grads, start=1):
        ctx = StepCtx(lr, 0.0, t, momentum=beta, nesterov=nesterov, colmask=colmask)
        res = cm.run(G, w, state, ctx)
        w, state = res.weight, res.state
        out.append((w, state))
    return out


def test_amphione_columns_need_the_mask():
    amphione = rule_of(
        {"rule": "amphione", "args": {**STEP_AMPHIONE, "column_scales": True}}
    )
    w, grads = _case()
    with pytest.raises(ValueError, match="mask"):
        _run(amphione.compute, w, grads[:1])


# ------------------------------------------------------- the step from its equations
@dataclasses.dataclass(frozen=True)
class _Polar(Compute):
    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return u @ vh


def _reference(
    w,
    grads,
    columns,
    row_direction=False,
    beta=BETA,
    nesterov=True,
    scale_nesterov=False,
    sum_form=False,
):
    """The docstring's step, unit by unit, at the group's momentum `beta`:
    M <- beta M + (1 - beta) D, O = Orth(beta M + (1 - beta) D) with `nesterov`
    (else Orth(M)); the row and column scales average at beta, with
    `scale_nesterov` their sign steps take beta m + (1 - beta) gamma.
    `row_direction`: O = rowl2(.) of the same input. `sum_form`: the direction
    as the code before wave 2 ran it, M <- beta M + D, O = Orth(D + beta M)."""
    out = []
    for W0, Gs in zip(w, zip(*grads)):
        g = W0.norm(dim=-1)
        U = W0 / g[:, None]
        c = torch.ones(W0.shape[-1], dtype=W0.dtype)
        M = torch.zeros_like(W0)
        mg, mc = torch.zeros_like(g), torch.zeros_like(c)
        unit = []
        for t, G in enumerate(Gs, start=1):
            eta_t = LR  # the trainer's lr at this step
            if columns:
                gamma_c = (G * (g[:, None] * U)).sum(dim=0)
                G = G * c[None, :]
            gamma = (G * U).sum(dim=-1)
            D = g[:, None] * G
            if sum_form:
                M = beta * M + D
                X = D + beta * M if nesterov else M
            else:
                M = beta * M + (1 - beta) * D
                X = beta * M + (1 - beta) * D if nesterov else M
            if row_direction:
                O = X / X.norm(dim=-1, keepdim=True)
            else:
                u, _, vh = torch.linalg.svd(X, full_matrices=False)
                O = u @ vh
            O = O * U.norm() / O.norm()
            V = U - eta_t * O
            U = V / V.norm(dim=-1, keepdim=True)
            mg = beta * mg + (1 - beta) * gamma
            sg = beta * mg + (1 - beta) * gamma if scale_nesterov else mg
            g = g - eta_t * torch.sign(sg)
            if columns:
                mc = beta * mc + (1 - beta) * gamma_c
                sc = beta * mc + (1 - beta) * gamma_c if scale_nesterov else mc
                c = c - eta_t / 16 * torch.sign(sc)
            unit.append(g[:, None] * U * c[None, :])
        out.append(unit)
    return [torch.stack(ws) for ws in zip(*out)]


def _f64_case(seed=1, units=3, m=12, n=8, steps=6):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(units, m, n, generator=gen, dtype=torch.float64) / n**0.5
    grads = [
        0.1 * torch.randn(units, m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]
    return w, grads


@pytest.mark.parametrize(
    "beta,nesterov,scale_nesterov",
    [(0.95, True, False), (0.9, False, False), (0.95, True, True), (0.0, True, False)],
)
@pytest.mark.parametrize("columns", [False, True])
def test_amphione_step_is_its_equations(columns, beta, nesterov, scale_nesterov):
    rule = rule_of(
        {
            "rule": "amphione",
            "args": {"column_scales": columns, "scale_nesterov": scale_nesterov},
        }
    )
    cm = dataclasses.replace(rule.compute, direction=Rule((), _Polar(), scale="one"))
    w, grads = _f64_case()
    mask = torch.ones(3, dtype=torch.bool) if columns else None
    want = _reference(
        w,
        grads,
        columns,
        beta=beta,
        nesterov=nesterov,
        scale_nesterov=scale_nesterov,
    )
    got = _run(cm, w, grads, mask, lr=LR, beta=beta, nesterov=nesterov)
    for (gw, _), ww in zip(got, want):
        torch.testing.assert_close(gw, ww, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("columns", [False, True])
def test_the_averaged_direction_takes_the_sum_forms_step(columns, nesterov):
    """The direction's input in the averaged form is (1 - beta) x the one of the
    sum form the code ran before wave 2; the polar factor and the rescale to
    ||U||_F drop that factor, so every step is the same (to rounding)."""
    w, grads = _f64_case(seed=4, steps=8)
    a = _reference(w, grads, columns, nesterov=nesterov)
    b = _reference(w, grads, columns, nesterov=nesterov, sum_form=True)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-13)


def test_amphione_with_row_l2_is_its_equations():
    """The conv group's rule: amphione with each row of D + beta M normalised as
    the direction (RowL2 runs in fp32, so to ~1e-6)."""
    cm = rule_of({"rule": "amphione", "args": {"compute": "row-l2"}}).compute
    assert isinstance(cm.direction.compute, RowL2)
    gen = torch.Generator().manual_seed(2)
    w = torch.randn(3, 12, 4, generator=gen, dtype=torch.float64) / 2.0
    grads = [
        0.1 * torch.randn(3, 12, 4, generator=gen, dtype=torch.float64)
        for _ in range(6)
    ]
    for (got, _), want in zip(
        _run(cm, w, grads, lr=LR), _reference(w, grads, False, row_direction=True)
    ):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


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
                "momentum": BETA,  # the recipe's: amphione reads it, DiSCO averages nothing
                "nesterov": True,
                "weight_decay": 0.0,
            }
        ],
        parallelism_context=_NoParallel(),
        lr=0.03,
        rule=rule,
        args=args,
        param_layouts=LAYOUTS,
    )
    return opt, params


def _units_step(opt, weights, grads, state, t, lr=0.03):
    """One step of amphione's compute on every unit DiSCO declared, each whole,
    the masks from their definitions; returns the new weights."""
    new = []
    for e, W, G in zip(opt.entries, weights, grads):
        cm = e.rule.compute
        W2, G2 = W.reshape(-1, e.view[1]), G.reshape(-1, e.view[1])
        out = W2.clone()
        for k, u in enumerate(e.units):
            rows = [(a, a + n) for a, n in u.rows.runs]
            Wu = torch.cat([W2[a:b] for a, b in rows])
            Gu = torch.cat([G2[a:b] for a, b in rows])
            ut = _unit_type(e, u)
            grp = opt.param_groups[e.group_idx]
            ctx = StepCtx(
                lr,
                0.0,
                t,
                momentum=float(grp["momentum"]),
                nesterov=bool(grp["nesterov"]),
                frozen=torch.tensor([ut in cm.frozen_units])
                if cm.frozen_units
                else None,
                colmask=torch.tensor([ut in cm.col_units]) if cm.col_units else None,
            )
            st = state.get((e.i, k))
            if st is None:
                st = torch.zeros(1, *cm.state_shape(*u.shape))
            res = cm.run(Gu[None], Wu[None], st, ctx)
            state[(e.i, k)] = res.state
            off = 0
            for a, b in rows:
                out[a:b] = res.weight[0][off : off + b - a]
                off += b - a
        new.append(out.reshape(W.shape))
    return new


def _check_disco_against_units(opt, params, steps=6):
    weights = [p.detach().clone() for p in params]
    state = {}
    for s in range(steps):
        g = torch.Generator().manual_seed(100 + s)
        grads = [0.1 * torch.randn(*shape, generator=g) for shape in SHAPES]
        for p, gr in zip(params, grads):
            p.grad = gr.clone()
        opt.step()
        weights = _units_step(opt, weights, grads, state, s + 1)
        for p, want in zip(params, weights):
            torch.testing.assert_close(p.detach(), want, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("columns", [False, True])
def test_disco_amphione_is_its_compute_unit_by_unit(one_rank, columns):
    opt, params = _disco("amphione", {**STEP_AMPHIONE, "column_scales": columns})
    _check_disco_against_units(opt, params)


def test_disco_writes_amphione_next_weight_exactly(one_rank):
    """Under fp32 transport the holders copy the rule's W_next: after each step w2
    (one unit, alone in its batch) is bitwise the compute's new weight. The step
    form w - (w - W_next) is only within a rounding of it."""
    opt, params = _disco("amphione", dict(STEP_AMPHIONE))
    weights = [p.detach().clone() for p in params]
    state = {}
    for s in range(3):
        g = torch.Generator().manual_seed(100 + s)
        grads = [0.1 * torch.randn(*shape, generator=g) for shape in SHAPES]
        for p, gr in zip(params, grads):
            p.grad = gr.clone()
        opt.step()
        weights = _units_step(opt, weights, grads, state, s + 1)
        assert torch.equal(params[3].detach(), weights[3]), s


# ------------------------------------------- frozen row lengths (the MoE router gate)
FROZEN = (
    AMPHIONE_COLUMN_SCALE_UNITS[1],
    AMPHIONE_COLUMN_SCALE_UNITS[3],
)  # wk in wqkv, w1 in w13


def _run_frozen(cm, w, grads, frozen, lr=0.03):
    state = w.new_zeros(w.shape[0], *cm.state_shape(*w.shape[1:]))
    out = []
    for t, G in enumerate(grads, start=1):
        ctx = StepCtx(lr, 0.0, t, momentum=BETA, nesterov=True, frozen=frozen)
        res = cm.run(G, w, state, ctx)
        w, state = res.weight, res.state
        out.append((w, state))
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_amphione_frozen_lengths_keep_the_start_and_turn_the_rows(dtype):
    amphione = rule_of(
        {"rule": "amphione", "args": {**STEP_AMPHIONE, "frozen_scale_units": ["u"]}}
    )
    w, grads = _case(dtype=dtype)
    mask = torch.tensor([True, False, True])
    out = _run_frozen(amphione.compute, w, grads, mask)
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
        ({"frozen_scale_units": ["u", "u"]}, "distinct unit types"),
        ({"frozen_scale_units": [""]}, "distinct unit types"),
        ({"frozen_scale_units": ["u"], "column_scales": True}, "with column_scales"),
    ],
)
def test_amphione_frozen_lengths_refusals(args, match):
    with pytest.raises(ValueError, match=match):
        rule_of({"rule": "amphione", "args": {**STEP_AMPHIONE, **args}})


def test_amphione_frozen_lengths_need_the_mask():
    amphione = rule_of(
        {"rule": "amphione", "args": {**STEP_AMPHIONE, "frozen_scale_units": ["u"]}}
    )
    w, grads = _case()
    with pytest.raises(ValueError, match="needs the stack's mask"):
        _run_frozen(amphione.compute, w, grads[:1], None)


def test_amphione_frozen_lengths_are_the_groups_own():
    from torchtitan.optimizers.disco.rules import RULES

    assert "frozen_scale_units" not in RULES["amphione"].inherits
    assert RULES["amphione"].args["frozen_scale_units"] == ()


def test_disco_amphione_frozen_lengths_reach_the_rule(one_rank):
    a, pa = _disco("amphione", {**STEP_AMPHIONE, "frozen_scale_units": list(FROZEN)})
    _check_disco_against_units(a, pa)
    c, pc = _disco(
        "amphione", STEP_AMPHIONE
    )  # nothing frozen, the same start and gradients
    for s in range(6):
        g = torch.Generator().manual_seed(100 + s)
        grads = [0.1 * torch.randn(*shape, generator=g) for shape in SHAPES]
        for p, gr in zip(pc, grads):
            p.grad = gr.clone()
        c.step()
    # the mask reached the rule: the fused wqkv and w13 differ from the unfrozen run, wo and w2 do not
    assert [torch.equal(x, z) for x, z in zip(pa, pc)] == [False, True, False, True]


def test_a_same_rule_group_does_not_take_the_defaults_own_args(one_rank):
    """`own` args (frozen_scale_units) are never taken from the default
    group, also by a group that runs the default's rule: an amphione conv group
    under an MoE default must not inherit the router's frozen lengths."""
    g = torch.Generator().manual_seed(0)
    params = [nn.Parameter(torch.randn(*s, generator=g) * 0.3) for s in SHAPES]
    plain = {"momentum": 0.0, "nesterov": False, "weight_decay": 0.0}
    opt = DiSCO(
        [
            {"params": params[:3], "param_names": NAMES[:3], **plain},
            {
                "params": params[3:],
                "param_names": NAMES[3:],
                **plain,
                "rule": "amphione",
                "args": {"compute": "svd"},
            },
        ],
        parallelism_context=_NoParallel(),
        lr=0.03,
        rule="amphione",
        args={
            **STEP_AMPHIONE,
            "frozen_scale_units": [FROZEN[0]],
            "scale_nesterov": True,
        },
        param_layouts=LAYOUTS,
    )
    own, same = opt.param_groups[0]["args"], opt.param_groups[1]["args"]
    assert own["frozen_scale_units"] == [FROZEN[0]]
    assert "frozen_scale_units" not in same
    assert same["compute"] == "svd" and same["scale_nesterov"] is True  # own; inherited


def test_the_checkpointers_state_init_never_steps_disco(one_rank):
    """Amphione starts its row scales at the row norms on step t = 1 only (later
    steps skip the norms). The checkpointer's `init_optim_state` takes a
    zero-lr step when a parameter has no state yet, which would make the real
    first step t = 2: DiSCO must hold state for every parameter from
    construction, so that step never happens."""
    from torchtitan.components.optim.utils import init_optim_state

    opt, params = _disco("amphione", {"compute": "svd"})
    assert all(opt.state.get(p) for p in params)
    init_optim_state(opt)
    assert all(int(g["t"]) == 0 for g in opt.param_groups)
