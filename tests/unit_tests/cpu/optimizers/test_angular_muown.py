# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""angular_muown against AngularMuown's Algorithm 1 (arXiv 2606.23637) as its
reference code steps one matrix (fhueb/angular-muown, angular_muown.py,
`_step_param`), in float64 with an exact polar factor."""

import math
import os
import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import AngularMuownMap, EtaSchedule
from torchtitan.optimizers.disco.rules import rule_of, StepCtx
from torchtitan.optimizers.helpers.norm_helper import get_norms_to_log

EPS = 1e-7
KW = dict(lr=0.05, warmup=2, c=0.5)  # kappa moves within six steps


def _polar(X):
    """The orthogonaliser both sides use (DiSCO's svd backend, which is within
    ~5e-7 of the exact polar factor): the comparison is of everything else."""
    return _SVD.run_whole(X[None]).update[0]


_SVD = rule_of({"rule": "muon", "args": {"compute": "svd", "scale": "one"}})


def _reference(
    W,
    grads,
    *,
    lr,
    beta=0.95,
    nesterov=True,
    betas=(0.9, 0.95),
    eps=1e-8,
    warmup=100,
    c=1e-3,
    p=1.0,
    magnitude_lr=1.0,
    eta=None,
    magnitude_rule="adam",
    mutate=None,
):
    """The reference's step on one matrix, step after step; its lr follows a
    WSD warmup, here lr min(1, t / warmup). `eta(t)`, when given, multiplies
    U's turn only: lr_t kappa eta(t). `mutate` makes one plausible mistake, to show
    the comparison would see it."""
    m, n = W.shape
    norm = W.norm(dim=1, keepdim=True)
    g = torch.where(norm <= EPS, torch.full_like(norm, 0.33**0.5), norm)
    m_u, m_g, v_g = torch.zeros_like(W), torch.zeros_like(g), torch.zeros_like(g)
    out = []
    for t, G in enumerate(grads, start=1):
        lr_t = lr * min(1.0, t / warmup) if warmup else lr
        if mutate == "g_is_row_norm" and t > 1:
            g = W.norm(dim=1, keepdim=True)
        u = W / torch.copysign(g.abs().clamp_min(EPS), g)
        grad_g = (G * u).sum(dim=1, keepdim=True)
        grad_u = G - u * grad_g if mutate == "no_diag_g" else g * (G - u * grad_g)
        src = G if mutate == "momentum_on_raw_grad" else grad_u
        m_u = beta * m_u + src
        update = src + beta * m_u if nesterov else m_u.clone()
        angular_step = t if mutate == "kappa_off_by_one" else t - 1
        kappa = (1.0 + c * max(0, angular_step - warmup)) ** -p
        step = lr_t * kappa * (1.0 if eta is None else eta(t))
        if mutate == "eta_replaces_lr":
            step = eta(t) * kappa
        u_step = u - step * math.sqrt(max(1.0, m / n)) * _polar(update)
        if mutate == "adam_on_momentum":
            grad_g = (m_u * u).sum(dim=1, keepdim=True)
        m_g = betas[0] * m_g + (1 - betas[0]) * grad_g
        v_g = betas[1] * v_g + (1 - betas[1]) * grad_g**2
        adam = (m_g / (1 - betas[0] ** t)) / ((v_g / (1 - betas[1] ** t)).sqrt() + eps)
        if magnitude_rule == "sign":  # the l-inf steepest step on the averaged gradient
            adam = torch.sign(grad_g if mutate == "sign_of_raw_gradient" else m_g)
        g = (
            g
            - (lr_t * eta(t) if mutate == "turn_on_magnitudes" else lr_t)
            * magnitude_lr
            * adam
        )
        g = torch.copysign(g.abs().clamp_min(EPS), g)
        W = g * (u_step / u_step.norm(dim=1, keepdim=True).clamp_min(EPS))
        out.append(W)
    return out


def _rule(**args):
    return rule_of({"rule": "angular_muown", "args": {"compute": "svd", **args}})


def _run(W, grads, *, lr, **args):
    """The rule on the same matrix: weights and state after every step."""
    rule = _rule(**args)
    m, n = W.shape
    w, state, ws, states = (
        W[None],
        W.new_zeros(1, *rule.compute.state_shape(m, n)),
        [],
        [],
    )
    for t, G in enumerate(grads, start=1):
        res = rule.run_whole(G[None], w, StepCtx(lr, 0.0, t), state)
        w, state = res.weight, res.state
        ws.append(w[0])
        states.append(state[0])
    return ws, states


def _data(m, n, seed, steps=6, *, spread=False):
    """An orthogonal-ish W (rows of unequal norm with `spread`) and gradients."""
    gen = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(
        torch.randn(max(m, n), max(m, n), generator=gen, dtype=torch.float64)
    )[0][:m, :n]
    W = W * (math.sqrt(m / n) if m > n else 1.0)
    if spread:
        W = W * torch.logspace(-0.5, 0.5, m, dtype=torch.float64)[:, None]
    grads = [
        0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]
    return W, grads


def _max_diff(xs, ys):
    return max(float((x - y).abs().max()) for x, y in zip(xs, ys))


def test_what_it_declares():
    r = _rule()
    assert isinstance(r.compute, AngularMuownMap)
    assert (r.kind, r.needs_weight, r.has_state, r.output) == (
        "unit",
        True,
        True,
        "weight",
    )
    assert r.compute.owns_momentum and not r.compute.keeps_radius
    assert r.compute.state_shape(12, 8) == (12, 11)
    assert r.compute.direction.scale == "image_spectral"
    assert _rule(compute="row-l2", scale="sqrt_n").kind == "rows"  # an embedding group


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize("nesterov", [True, False])
def test_it_is_the_reference_step_after_step(m, n, nesterov):
    W, grads = _data(m, n, seed=m + n, spread=True)
    ref = _reference(W, grads, nesterov=nesterov, **KW)
    got, _ = _run(W, grads, nesterov=nesterov, **KW)
    assert _max_diff(ref, got) < 1e-10


@pytest.mark.parametrize(
    "mutate",
    ["momentum_on_raw_grad", "adam_on_momentum", "no_diag_g", "kappa_off_by_one"],
)
def test_the_comparison_sees_each_plausible_mistake(mutate):
    """Each mistake moves W far past the 1e-10 the rule is held to."""
    W, grads = _data(12, 8, seed=1, spread=True)
    assert (
        _max_diff(_reference(W, grads, **KW), _reference(W, grads, mutate=mutate, **KW))
        > 1e-4
    )


def test_magnitudes_keep_their_sign_through_zero():
    """A magnitude driven through zero flips its row; recomputing g as the
    row norm would flip U instead, under its momentum."""
    W, grads = _data(8, 8, seed=2, steps=8)
    kw = dict(lr=1.0, warmup=2, c=0.5)
    ref = _reference(W, grads, **kw)
    got, states = _run(W, grads, **kw)
    assert bool((states[-1][:, 8] < 0).any())  # some g crossed zero
    assert _max_diff(ref, got) < 1e-10
    assert _max_diff(ref, _reference(W, grads, mutate="g_is_row_norm", **kw)) > 1e-2


@pytest.mark.parametrize("magnitude_lr", [1.0, 0.0])
def test_row_norms_are_the_magnitudes(magnitude_lr):
    W, grads = _data(12, 8, seed=3, spread=True)
    got, states = _run(W, grads, magnitude_lr=magnitude_lr, **KW)
    for w, s in zip(got, states):
        assert torch.allclose(w.norm(dim=1), s[:, 8].abs(), rtol=1e-12)
    if magnitude_lr == 0.0:  # the frozen control: row norms stay at their start
        assert torch.allclose(got[-1].norm(dim=1), W.norm(dim=1), rtol=1e-12)
    else:
        assert not torch.allclose(got[-1].norm(dim=1), W.norm(dim=1), rtol=1e-3)


def test_a_zero_row_starts_at_the_reference_scale():
    W, grads = _data(8, 8, seed=4, steps=1)
    W[3] = 0.0
    got, _ = _run(W, grads, magnitude_lr=0.0, **KW)
    assert got[0][3].norm() == pytest.approx(0.33**0.5, rel=1e-12)
    assert _max_diff(_reference(W, grads, magnitude_lr=0.0, **KW), got) < 1e-10


SCHED = dict(shape="sqrt_decay", start=2, total=6)  # an angular schedule over 6 steps


@pytest.mark.parametrize("nesterov", [True, False])
def test_an_eta_schedule_multiplies_the_turn_only(nesterov):
    """`eta` multiplies U's turn, lr_t kappa eta(t); the magnitudes keep lr_t."""
    W, grads = _data(12, 8, seed=5, spread=True)
    eta = EtaSchedule(**SCHED)
    ref = _reference(W, grads, nesterov=nesterov, eta=eta, **KW)
    got, _ = _run(W, grads, nesterov=nesterov, eta=SCHED, **KW)
    assert _max_diff(ref, got) < 1e-10
    assert _max_diff(ref, _reference(W, grads, nesterov=nesterov, **KW)) > 1e-4


@pytest.mark.parametrize("mutate", ["eta_replaces_lr", "turn_on_magnitudes"])
def test_the_eta_schedule_comparison_sees_each_mistake(mutate):
    W, grads = _data(12, 8, seed=5, spread=True)
    eta = EtaSchedule(**SCHED)
    assert (
        _max_diff(
            _reference(W, grads, eta=eta, **KW),
            _reference(W, grads, eta=eta, mutate=mutate, **KW),
        )
        > 1e-4
    )


def test_without_an_eta_schedule_the_step_is_unchanged():
    assert _rule().compute.eta is None
    W, grads = _data(12, 8, seed=6, spread=True)
    assert _max_diff(_reference(W, grads, **KW), _run(W, grads, **KW)[0]) < 1e-10


def _reference_unit(
    W,
    grads,
    *,
    lr,
    beta=0.95,
    nesterov=True,
    betas=(0.9, 0.95),
    eps=1e-8,
    warmup=100,
    c=1e-3,
    p=1.0,
    magnitude_lr=1.0,
    mutate=None,
):
    """One learned magnitude per matrix, from the equations: W = a Diag(shape) U
    with the fixed row shape = g0 / rms(g0). U turns as in Algorithm 1 with
    g = a shape; a moves by Adam on dL/da = sum_i shape_i <G_i, U_i> at
    lr_t magnitude_lr, the rate the per-row magnitudes move at."""
    m, n = W.shape
    g0 = W.norm(dim=1, keepdim=True)
    a = g0.square().mean().sqrt()
    shape, u = g0 / a, W / g0
    m_u, m_a, v_a = torch.zeros_like(W), torch.zeros_like(a), torch.zeros_like(a)
    out = []
    for t, G in enumerate(grads, start=1):
        lr_t = lr * min(1.0, t / warmup) if warmup else lr
        g = a * shape
        grad_g = (G * u).sum(dim=1, keepdim=True)
        grad_u = g * (G - u * grad_g)
        m_u = beta * m_u + grad_u
        update = grad_u + beta * m_u if nesterov else m_u.clone()
        kappa = (1.0 + c * max(0, t - 1 - warmup)) ** -p
        u_step = u - lr_t * kappa * math.sqrt(max(1.0, m / n)) * _polar(update)
        u = u_step / u_step.norm(dim=1, keepdim=True)
        grad_a = (
            grad_g.sum() if mutate == "unweighted_gradient" else (shape * grad_g).sum()
        )
        m_a = betas[0] * m_a + (1 - betas[0]) * grad_a
        v_a = betas[1] * v_a + (1 - betas[1]) * grad_a**2
        adam = (m_a / (1 - betas[0] ** t)) / ((v_a / (1 - betas[1] ** t)).sqrt() + eps)
        step = lr_t * magnitude_lr * adam
        a = a * (1.0 - step) if mutate == "relative_step" else a - step
        out.append(a * shape * u)
    return out


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize("nesterov", [True, False])
def test_one_magnitude_per_unit_is_the_tied_reference(m, n, nesterov):
    W, grads = _data(m, n, seed=m + n + 1, spread=True)
    ref = _reference_unit(W, grads, nesterov=nesterov, **KW)
    got, _ = _run(W, grads, nesterov=nesterov, magnitude_shape="unit", **KW)
    assert _max_diff(ref, got) < 1e-10
    assert (
        _max_diff(ref, _reference(W, grads, nesterov=nesterov, **KW)) > 1e-4
    )  # not per row


@pytest.mark.parametrize("mutate", ["unweighted_gradient", "relative_step"])
def test_the_per_unit_comparison_sees_each_mistake(mutate):
    W, grads = _data(12, 8, seed=9, spread=True)
    assert (
        _max_diff(
            _reference_unit(W, grads, **KW),
            _reference_unit(W, grads, mutate=mutate, **KW),
        )
        > 1e-4
    )


def test_one_magnitude_per_unit_keeps_the_row_ratios():
    W, grads = _data(12, 8, seed=10, spread=True)
    got, states = _run(W, grads, magnitude_shape="unit", **KW)
    assert _rule(magnitude_shape="unit").compute.state_shape(12, 8) == (12, 12)
    ratio = got[-1].norm(dim=1) / W.norm(dim=1)
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-12)
    assert abs(float(ratio[0]) - 1.0) > 1e-3  # the magnitude moved
    for st in states:  # the state keeps the starting norms and one s per unit
        assert torch.equal(st[:, 8], W.norm(dim=1))
        assert torch.equal(st[:, 11], st[0, 11].expand(12))
    per_row = _run(W, grads, **KW)[0][-1].norm(dim=1) / W.norm(dim=1)
    assert not torch.allclose(per_row, per_row[0].expand_as(per_row), rtol=1e-3)


def test_frozen_per_unit_is_frozen_per_row_bitwise():
    W, grads = _data(12, 8, seed=11, spread=True)
    unit, _ = _run(W, grads, magnitude_shape="unit", magnitude_lr=0.0, **KW)
    row, _ = _run(W, grads, magnitude_lr=0.0, **KW)
    assert all(torch.equal(a, b) for a, b in zip(unit, row))


def test_each_unit_of_a_stack_has_its_own_magnitude():
    (Wa, ga), (Wb, gb) = _data(12, 8, seed=12, spread=True), _data(12, 8, seed=13)
    rule = _rule(magnitude_shape="unit", **{k: v for k, v in KW.items() if k != "lr"})
    w, state = torch.stack([Wa, Wb]), Wa.new_zeros(2, 12, 12)
    for t, (A, B) in enumerate(zip(ga, gb), start=1):
        res = rule.run_whole(torch.stack([A, B]), w, StepCtx(KW["lr"], 0.0, t), state)
        w, state = res.weight, res.state
    for k, (W, grads) in enumerate(((Wa, ga), (Wb, gb))):
        alone = _run(W, grads, magnitude_shape="unit", **KW)[0][-1]
        assert float((w[k] - alone).abs().max()) < 1e-12
    assert float((state[0, 0, 11] - state[1, 0, 11]).abs()) > 1e-4


def test_a_lost_per_unit_state_restarts_from_the_row_norms():
    W, grads = _data(12, 8, seed=6, steps=1, spread=True)
    res = _rule(magnitude_shape="unit", magnitude_lr=0.0).run_whole(
        grads[0][None], W[None], StepCtx(0.05, 0.0, 3001), W.new_zeros(1, 12, 12)
    )
    assert torch.allclose(res.weight[0].norm(dim=1), W.norm(dim=1), rtol=1e-12)
    assert torch.equal(res.state[0, :, 11], torch.ones(12, dtype=W.dtype))


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize("nesterov", [True, False])
def test_sign_magnitudes_are_the_reference_step_after_step(m, n, nesterov):
    W, grads = _data(m, n, seed=m + n + 2, spread=True)
    ref = _reference(W, grads, nesterov=nesterov, magnitude_rule="sign", **KW)
    got, _ = _run(W, grads, nesterov=nesterov, magnitude_rule="sign", **KW)
    assert _max_diff(ref, got) < 1e-10
    assert (
        _max_diff(ref, _reference(W, grads, nesterov=nesterov, **KW)) > 1e-4
    )  # not Adam


def test_the_sign_comparison_sees_the_missing_average():
    W, grads = _data(12, 8, seed=14, spread=True)
    ref = _reference(W, grads, magnitude_rule="sign", **KW)
    bad = _reference(
        W, grads, magnitude_rule="sign", mutate="sign_of_raw_gradient", **KW
    )
    assert _max_diff(ref, bad) > 1e-4


def test_sign_moves_every_magnitude_by_exactly_the_step():
    """|g+ - g| = lr_t magnitude_lr for every row: the l-inf ball is filled."""
    W, grads = _data(12, 8, seed=15, spread=True)
    _, states = _run(W, grads, magnitude_rule="sign", magnitude_lr=0.5, **KW)
    g = [W.norm(dim=1)] + [st[:, 8] for st in states]
    for t in range(1, len(g)):
        lr_t = KW["lr"] * min(1.0, t / KW["warmup"])
        torch.testing.assert_close(
            (g[t] - g[t - 1]).abs(), torch.full_like(g[t], 0.5 * lr_t)
        )


def test_sign_magnitudes_per_unit_move_the_rms_norm_by_the_step():
    W, grads = _data(12, 8, seed=16, spread=True)
    got, _ = _run(W, grads, magnitude_shape="unit", magnitude_rule="sign", **KW)
    rms = [W.norm(dim=1).square().mean().sqrt()] + [
        x.norm(dim=1).square().mean().sqrt() for x in got
    ]
    for t in range(1, len(rms)):
        lr_t = KW["lr"] * min(1.0, t / KW["warmup"])
        assert abs(abs(float(rms[t] - rms[t - 1])) - lr_t) < 1e-12


def test_bad_settings_are_refused():
    for bad in (
        dict(beta=1.0),
        dict(betas=(0.9, 1.0)),
        dict(warmup=-1),
        dict(magnitude_lr=-1.0),
        dict(eta="lr"),
        dict(magnitude_shape="matrix"),
        dict(magnitude_rule="lion"),
    ):
        with pytest.raises(ValueError, match="angular_muown"):
            _rule(**bad)


# ------------------------------------------------------------------ in DiSCO
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


_RAW = dict(momentum=1.0, nesterov=False, weight_decay=0.0)


def _disco(w, args=None, **group):
    return DiSCO(
        [{"params": [w], "param_names": ["w"], **{**_RAW, **group}}],
        parallel_dims=_NoParallel(),
        lr=KW["lr"],
        rule="angular_muown",
        args={"compute": "svd", "warmup": KW["warmup"], "c": KW["c"], **(args or {})},
        param_layouts={"w": ParamLayout("matrix")},
    )


@pytest.mark.parametrize(
    "bad", [dict(momentum=0.1), dict(nesterov=True), dict(weight_decay=0.01)]
)
def test_the_group_must_pass_the_raw_gradient(one_rank, bad):
    with pytest.raises(ValueError, match="raw gradient"):
        _disco(nn.Parameter(torch.randn(12, 8)), **bad)


def test_a_disco_step_is_the_rule(one_rank):
    """The raw gradient reaches the rule, its state carries over, and the
    weight it returns is the one applied."""
    W, grads = _data(12, 8, seed=5, steps=4, spread=True)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w)
    for G in grads:
        w.grad = G.float().clone()
        opt.step()
    ref = _reference(W, grads, **KW)[-1]
    assert float((w.detach().double() - ref).abs().max()) < 1e-5


def test_a_disco_step_with_one_magnitude_per_unit_is_the_rule(one_rank):
    """DiSCO allocates the wider state and carries s and g0 between steps."""
    W, grads = _data(12, 8, seed=8, steps=4, spread=True)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w, args={"magnitude_shape": "unit"})
    for G in grads:
        w.grad = G.float().clone()
        opt.step()
    ref = _reference_unit(W, grads, **KW)[-1]
    assert float((w.detach().double() - ref).abs().max()) < 1e-5
    assert float((ref - _reference(W, grads, **KW)[-1]).abs().max()) > 1e-3


def test_a_disco_step_with_sign_magnitudes_is_the_rule(one_rank):
    W, grads = _data(12, 8, seed=17, steps=4, spread=True)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w, args={"magnitude_rule": "sign"})
    for G in grads:
        w.grad = G.float().clone()
        opt.step()
    ref = _reference(W, grads, magnitude_rule="sign", **KW)[-1]
    assert float((w.detach().double() - ref).abs().max()) < 1e-5
    assert float((ref - _reference(W, grads, **KW)[-1]).abs().max()) > 1e-3


def test_a_lost_state_restarts_from_the_row_norms():
    """A zero state after step 1 (never a real one: |g| >= 1e-7) must not
    collapse the rows: g restarts at the row norms, as on an empty state."""
    W, grads = _data(12, 8, seed=6, steps=1, spread=True)
    res = _rule(magnitude_lr=0.0).run_whole(
        grads[0][None], W[None], StepCtx(0.05, 0.0, 3001), W.new_zeros(1, 12, 11)
    )
    assert torch.allclose(res.weight[0].norm(dim=1), W.norm(dim=1), rtol=1e-12)


def test_a_logging_step_runs_and_logs_finite_values(one_rank):
    W, grads = _data(12, 8, seed=7, steps=2)
    w = nn.Parameter(W.float().clone())
    opt = _disco(w)
    for G in grads:
        opt.calculate_norm_at_next_step(
            norms_to_log=get_norms_to_log("all"), gram_level=0, track_spectrum=False
        )
        w.grad = G.float().clone()
        opt.step()
        logged = opt.get_norms_at_current_step()
    assert logged and all(math.isfinite(float(v)) for v in logged.values())


def test_units_of_different_shapes_are_refused(one_rank):
    """One state tensor per parameter: a fused q/k/v whose q is larger than its
    k and v (grouped-query attention) cannot keep per-row state."""
    from torchtitan.models.common.param_layout import RunList, UnitSpec

    units = (
        UnitSpec("q", RunList.contiguous(0, 8), (8, 6)),
        UnitSpec("k", RunList.contiguous(8, 4), (4, 6)),
        UnitSpec("v", RunList.contiguous(12, 4), (4, 6)),
    )
    w = nn.Parameter(torch.randn(16, 6))
    with pytest.raises(ValueError, match="state"):
        DiSCO(
            [{"params": [w], "param_names": ["w"], **_RAW}],
            parallel_dims=_NoParallel(),
            lr=0.05,
            rule="angular_muown",
            args={"compute": "svd"},
            param_layouts={"w": ParamLayout("matrix", units=units)},
        )
