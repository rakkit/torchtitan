# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rule "lie_relative" (V3a, theory-radius-norm.md 6.6, reduced form) against its definition, in
float64 with an exact polar factor. Per unit (m >= n; a wide unit runs on W^T, G^T, M^T):

    M <- beta M + G,  Mn = G + beta M (Nesterov)
    Q = Orth(W),  P = sym(Q^T W),  C = Orth(Mn P),  S = Q^T C,  eta = lr_t eta(t)
    W <- W - eta C P + 0.5 eta^2 C S P
    T = C^T M,  M <- M + eta Q T + 0.5 eta^2 Q S^T T            (cotangent transport)

and against the naive m x m form Y = Orth(Mn W^T), W <- (I - eta Y + 0.5 eta^2 Y^2) W."""

import copy
import os
import socket
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import EtaSchedule, LieRelativeMap
from torchtitan.optimizers.disco.rules import Compute, Rule, rule_of, StepCtx

LR = 0.05
ETA = EtaSchedule("inverse_poly", start=1, c=0.5)


@dataclass(frozen=True)
class _Polar(Compute):
    def core(self, x):
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        return u @ vh


def _orth(X):
    u, _, vh = torch.linalg.svd(X, full_matrices=False)
    return u @ vh


def _data(m, n, seed, steps=6):
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(m, n, generator=gen, dtype=torch.float64) / max(m, n) ** 0.5
    return W, [
        0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]


def _max_diff(xs, ys):
    return max(float((x - y).abs().max()) for x, y in zip(xs, ys))


def _reference(W, grads, *, lr=LR, beta=0.95, nesterov=True, eta=ETA, mutate=None):
    wide = W.shape[0] < W.shape[1]
    W = W.T if wide else W
    M, out = torch.zeros_like(W), []
    for t, G in enumerate(grads, start=1):
        G = G.T if wide else G
        M = beta * M + G
        Mn = G + beta * M if nesterov else M
        Q = _orth(W)
        P = Q.T @ W
        P = 0.5 * (P + P.T)
        C = _orth(Mn if mutate == "orth_without_P" else Mn @ P)
        S = Q.T @ C
        e = lr * eta(t)
        k = 0.0 if mutate == "no_second_order" else 0.5
        W = W - e * C @ P + k * e * e * C @ S @ P
        T = C.T @ M
        if mutate != "no_transport":
            M = M + e * Q @ T + k * e * e * Q @ S.T @ T
        out.append((W.T if wide else W, M))
    return out


def _naive(W, grads, *, lr=LR, beta=0.95, eta=ETA):
    """The m x m form: Y = Orth(Mn W^T), A = I - eta Y + 0.5 eta^2 Y^2, W <- A W,
    M <- (I + eta Y^T + 0.5 eta^2 Y^T Y^T) M. Mn W^T has rank n: Y is its partial isometry
    (the top-n singular pairs; a full SVD would invent the null directions)."""
    M, out = torch.zeros_like(W), []
    eye = torch.eye(W.shape[0], dtype=W.dtype)
    for t, G in enumerate(grads, start=1):
        M = beta * M + G
        Mn = G + beta * M
        u, _, vh = torch.linalg.svd(Mn @ W.T)
        Y = u[:, : W.shape[1]] @ vh[: W.shape[1]]
        e = lr * eta(t)
        W = (eye - e * Y + 0.5 * e * e * Y @ Y) @ W
        M = (eye + e * Y.T + 0.5 * e * e * Y.T @ Y.T) @ M
        out.append(W)
    return out


def _map(**kw):
    return LieRelativeMap(Rule((), _Polar()), **{"eta": ETA, **kw})


def _run(cm, W, grads, lr=LR):
    w, state, out = W[None], W.new_zeros(1, *cm.state_shape(*W.shape)), []
    for t, G in enumerate(grads, start=1):
        res = cm.run(G[None], w, state, StepCtx(lr, 0.0, t))
        w, state = res.weight, res.state
        out.append((w[0], state[0]))
    return out


# ---------------------------------------------------------------- the definition
@pytest.mark.parametrize("m,n", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize("nesterov", [True, False])
def test_lie_relative_is_its_definition_step_after_step(m, n, nesterov):
    W, grads = _data(m, n, seed=m + 3 * n)
    ref = _reference(W, grads, nesterov=nesterov)
    got = _run(_map(nesterov=nesterov), W, grads)
    assert _max_diff([w for w, _ in ref], [w for w, _ in got]) < 1e-12
    assert _max_diff([s for _, s in ref], [s for _, s in got]) < 1e-12


@pytest.mark.parametrize(
    "mutate", ["orth_without_P", "no_second_order", "no_transport"]
)
def test_the_comparison_sees_each_plausible_mistake(mutate):
    W, grads = _data(12, 8, seed=1)
    ref, bad = _reference(W, grads), _reference(W, grads, mutate=mutate)
    assert _max_diff([w for w, _ in ref], [w for w, _ in bad]) > 1e-6


def test_the_reduced_form_is_the_naive_m_by_m_form():
    """Orth(Mn W^T) = Orth(Mn P) Q^T for a full-rank tall W: the weights agree step after step."""
    W, grads = _data(12, 8, seed=2)
    got = [w for w, _ in _run(_map(), W, grads)]
    assert _max_diff(got, _naive(W, grads)) < 1e-10


def test_the_relative_operator_step_is_eta():
    """dW P^-1 = -eta C + 0.5 eta^2 C S with ||C||_S = ||C S||_S <= 1, so ||dW P^-1||_S is eta
    to within 0.5 eta^2."""
    W, grads = _data(12, 8, seed=3)
    w = W
    for t, (w_new, _) in enumerate(_run(_map(), W, grads), start=1):
        Q = _orth(w)
        P = 0.5 * (Q.T @ w + w.T @ Q)
        rel = float(torch.linalg.matrix_norm((w_new - w) @ torch.linalg.inv(P), ord=2))
        e = LR * ETA(t)
        assert abs(rel - e) <= 0.5 * e * e + 1e-12, (t, rel, e)
        w = w_new


def test_a_wide_unit_is_the_tall_path_on_its_transpose():
    W, grads = _data(8, 12, seed=4)
    wide = _run(_map(), W, grads)
    tall = _run(_map(), W.T.contiguous(), [G.T.contiguous() for G in grads])
    for (a, sa), (b, sb) in zip(wide, tall):
        assert torch.equal(a, b.T) and torch.equal(sa, sb)


def test_a_wide_unit_takes_the_right_form_to_first_order():
    """m < n: dW = -eta P' Orth(P' Mn) = W X, X = -eta Orth(W^T Mn), up to O(eta^2)."""
    W, grads = _data(8, 12, seed=5, steps=1)
    e = 1e-7
    ((w1, _),) = _run(_map(eta=None), W, grads, lr=e)
    Mn = 1.95 * grads[0]
    u, _, vh = torch.linalg.svd(
        W.T @ Mn
    )  # rank 8: the partial isometry is the top-8 part
    X = -(u[:, :8] @ vh[:8])
    assert float(((w1 - W) / e - W @ X).abs().max()) < 1e-5


def test_at_a_scaled_orthogonal_W_the_step_is_c_times_orth_of_the_momentum():  # noqa: N802
    """W = c Q: P = c I, C = Orth(Mn), dW = -eta c C + 0.5 eta^2 c C Q^T C."""
    gen = torch.Generator().manual_seed(6)
    Q = torch.linalg.qr(torch.randn(12, 8, generator=gen, dtype=torch.float64))[0]
    G = 0.1 * torch.randn(12, 8, generator=gen, dtype=torch.float64)
    c = 3.0
    ((w1, _),) = _run(_map(), c * Q, [G])
    C = _orth(1.95 * G)
    e = LR * ETA(1)
    assert (
        float((w1 - (c * Q - e * c * C + 0.5 * e * e * c * C @ Q.T @ C)).abs().max())
        < 1e-12
    )


def test_the_transport_keeps_the_pairing_to_third_order():
    """<M+, A dW> = <M, dW> + O(eta^3) for any dW, A = I - eta Y + 0.5 eta^2 Y^2, Y = C Q^T, M the
    momentum before the transport; without the transport the error is O(eta)."""
    W, grads = _data(12, 8, seed=7, steps=1)
    gen = torch.Generator().manual_seed(8)
    X = torch.randn(12, 8, generator=gen, dtype=torch.float64)
    for e in (1e-2, 1e-3):
        ((_, Mp),) = _run(_map(eta=None), W, grads, lr=e)
        M = grads[0]
        Q = _orth(W)
        C = _orth(1.95 * grads[0] @ (0.5 * (Q.T @ W + W.T @ Q)))
        Y = C @ Q.T
        A = torch.eye(12, dtype=W.dtype) - e * Y + 0.5 * e * e * Y @ Y
        scale = float(M.norm() * X.norm())
        assert abs(float((Mp * (A @ X)).sum() - (M * X).sum())) < 2 * e**3 * scale
        assert abs(float((M * (A @ X)).sum() - (M * X).sum())) > 1e-3 * e * scale


def test_a_non_finite_weight_is_an_error():
    @dataclass(frozen=True)
    class _PassThrough(Compute):
        def core(self, x):
            return x

    W, grads = _data(12, 8, seed=9, steps=1)
    G = grads[0].clone()
    G[1, 1] = float("inf")
    cm = LieRelativeMap(Rule((), _PassThrough()))
    with pytest.raises(RuntimeError, match="lie_relative: a weight is not finite"):
        cm.run(G[None], W[None], W.new_zeros(1, 12, 8), StepCtx(LR, 0.0, 1))


def test_the_preset_builds_it():
    r = rule_of(
        {
            "rule": "lie_relative",
            "args": {
                "compute": "svd",
                "eta": {"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0},
            },
        }
    )
    cm = r.compute
    assert isinstance(cm, LieRelativeMap) and cm.owns_momentum and r.output == "weight"
    assert (cm.state_shape(12, 8), cm.state_shape(8, 12)) == ((12, 8), (12, 8))
    row = rule_of(
        {"rule": "lie_relative", "args": {"compute": "row-l2", "scale": "sqrt_n"}}
    )
    assert (row.kind, row.scale) == (
        "rows",
        "sqrt_n",
    )  # an embedding group keeps its row rule
    with pytest.raises(ValueError, match="lie_relative's step is relative to W"):
        rule_of(
            {"rule": "lie_relative", "args": {"compute": "svd", "scale": "spectral"}}
        )


# ---------------------------------------------------------------- in DiSCO
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


def _disco(ws):
    return DiSCO(
        [
            {
                "params": ws,
                "param_names": ["a", "b"],
                "momentum": 1.0,
                "nesterov": False,
                "weight_decay": 0.0,
            }
        ],
        parallel_dims=_NoParallel(),
        lr=LR,
        rule="lie_relative",
        args={
            "compute": "svd",
            "eta": {"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0},
        },
        param_layouts={"a": ParamLayout("matrix"), "b": ParamLayout("matrix")},
    )


def test_a_disco_step_is_the_rule_for_tall_and_wide_units(one_rank):
    (Wa, ga), (Wb, gb) = _data(12, 8, seed=10, steps=3), _data(8, 12, seed=11, steps=3)
    wa, wb = nn.Parameter(Wa.float().clone()), nn.Parameter(Wb.float().clone())
    opt = _disco([wa, wb])
    for A, B in zip(ga, gb):
        wa.grad, wb.grad = A.float().clone(), B.float().clone()
        opt.step()
    assert float((wa.detach().double() - _reference(Wa, ga)[-1][0]).abs().max()) < 1e-5
    assert float((wb.detach().double() - _reference(Wb, gb)[-1][0]).abs().max()) < 1e-5


def test_a_resumed_run_is_the_continuous_one(one_rank):
    (Wa, ga), (Wb, gb) = _data(12, 8, seed=12, steps=5), _data(8, 12, seed=13, steps=5)
    pa = [nn.Parameter(Wa.float().clone()), nn.Parameter(Wb.float().clone())]
    a = _disco(pa)

    def steps(opt, ps, k0, k1):
        for A, B in zip(ga[k0:k1], gb[k0:k1]):
            ps[0].grad, ps[1].grad = A.float().clone(), B.float().clone()
            opt.step()

    steps(a, pa, 0, 3)
    saved = [p.detach().clone() for p in pa]
    sd = copy.deepcopy(a.state_dict())
    assert sd["state"][1]["rule_state"].shape == (
        1,
        12,
        8,
    )  # the wide unit's transposed momentum
    steps(a, pa, 3, 5)
    pb = [nn.Parameter(x.clone()) for x in saved]
    pc = [nn.Parameter(x.clone()) for x in saved]
    b, c = _disco(pb), _disco(pc)
    b.load_state_dict(sd)
    steps(b, pb, 3, 5)
    steps(c, pc, 3, 5)
    assert all(torch.equal(x.detach(), y.detach()) for x, y in zip(pa, pb))
    assert not all(torch.equal(x.detach(), y.detach()) for x, y in zip(pa, pc))
