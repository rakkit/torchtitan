# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A DiSCO group's scheduled momentum `momentum_k` (the head rule `hsched.k<k>`, ps2r E13).

From the definition: the group's momentum field (the weight of the NEW gradient) at its step
t = 1, 2, ... is alpha_t = min(1, k / t); alpha = 1 means no momentum (buf = g), then
beta_t = 1 - k / t. The buffer is buf_t = (1 - alpha_t) buf_{t-1} + alpha_t g_t and the Nesterov
direction is lerp(buf_t, g_t, alpha_t) = (1 - alpha_t) buf_t + alpha_t g_t. Everything is
recomputed here in float64 from those equations, not from the code under test."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.disco.optimizer import DiSCO

K = 3.0
M, N = 12, 8
LR = 0.125


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


def _head_group(w, **extra):
    """The lm_head group as cell.py writes it: DiSCO muon, row-l2 compute, inv_sqrt_n scale, Nesterov."""
    return {
        "params": [w],
        "param_names": ["lm_head.weight"],
        "lr": LR,
        "momentum": 0.05,
        "nesterov": True,
        "weight_decay": 0.0,
        "decoupled_wd": False,
        "rule": "muon",
        "args": {"compute": "row-l2", "scale": "inv_sqrt_n", "pre": "identity"},
        **extra,
    }


def _opt(groups, names):
    return DiSCO(
        groups,
        parallel_dims=_NoParallel(),
        lr=LR,
        rule="muon",
        args={"compute": "row-l2", "scale": "inv_sqrt_n", "pre": "identity"},
        param_layouts={n: ParamLayout("matrix") for n in names},
    )


def _data(seed, steps):
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(M, N, generator=gen, dtype=torch.float64)
    return W, [
        torch.randn(M, N, generator=gen, dtype=torch.float64) for _ in range(steps)
    ]


def _alpha(t, k=K):
    return min(1.0, k / t)


def _reference(W, grads, *, k, nesterov, fixed=None):
    """float64 weights and buffers after each step: W <- W - lr N^-1/2 RN(direction), RN = x / (||x|| + EPS)."""
    from torchtitan.optimizers.disco.presets import EPS

    w, buf, out = W.clone(), torch.zeros_like(W), []
    for t, g in enumerate(grads, start=1):
        a = _alpha(t, k) if fixed is None else fixed
        buf = (1.0 - a) * buf + a * g
        d = (1.0 - a) * buf + a * g if nesterov else buf
        w = w - LR * N**-0.5 * d / (d.norm(dim=-1, keepdim=True) + EPS)
        out.append((w.clone(), buf.clone()))
    return out


def _run(W, grads, nesterov=True, k=K):
    w = nn.Parameter(W.float().clone())
    g = _head_group(
        w, nesterov=nesterov, **({"momentum_k": k} if k is not None else {})
    )
    opt = _opt([g], ["lm_head.weight"])
    trace = []
    for G in grads:
        w.grad = G.float().clone()
        opt.step()
        trace.append(
            (
                w.detach().double().clone(),
                opt.state[w]["momentum_buffer"].detach().double().clone(),
                float(opt.param_groups[0]["momentum"]),
                dict(opt.momentum_used),
            )
        )
    return w, opt, trace


def test_the_momentum_field_is_min_1_k_over_t_at_every_step():
    W, grads = _data(1, 12)
    _, _, trace = _run(W, grads)
    for t, (_, _, m, used) in enumerate(trace, start=1):
        assert m == pytest.approx(_alpha(t), abs=1e-12), t
        assert used == {0: pytest.approx(_alpha(t), abs=1e-12)}, t
    assert [_alpha(t) for t in (1, 2, 3, 4, 12)] == [
        1.0,
        1.0,
        1.0,
        0.75,
        0.25,
    ]  # k = 3: no momentum up to t = k


@pytest.mark.parametrize("nesterov", [True, False])
def test_buffer_and_weights_follow_the_recursion(nesterov):
    W, grads = _data(2, 14)
    _, _, trace = _run(W, grads, nesterov=nesterov)
    ref = _reference(W, grads, k=K, nesterov=nesterov)
    for t, ((w, buf, _, _), (rw, rbuf)) in enumerate(zip(trace, ref), start=1):
        assert float((buf - rbuf).abs().max()) < 1e-5 * float(rbuf.abs().max()), (
            t,
            "buffer",
        )
        assert float((w - rw).abs().max()) < 1e-5, (t, "weights")


def test_at_alpha_1_the_buffer_is_the_gradient_so_the_ema_starts_there():
    W, grads = _data(3, 6)
    _, _, trace = _run(W, grads)
    for t in (1, 2, 3):  # alpha_t = 1: buf_t = g_t exactly
        assert torch.equal(trace[t - 1][1].float(), grads[t - 1].float()), t
    # t = 4 = k + 1: buf = (1 - 3/4) g_3 + (3/4) g_4 -- needs buf_3 = g_3 (not 0)
    want = 0.25 * grads[2] + 0.75 * grads[3]
    assert float((trace[3][1] - want).abs().max()) < 1e-6


def test_a_group_without_the_key_keeps_its_fixed_momentum():
    W, grads = _data(4, 8)
    w, opt, trace = _run(W, grads, k=None)
    assert all(m == 0.05 for _, _, m, _ in trace) and opt.momentum_used == {}
    ref = _reference(W, grads, k=None, nesterov=True, fixed=0.05)
    for (wt, buf, _, _), (rw, rbuf) in zip(trace, ref):
        assert (
            float((wt - rw).abs().max()) < 1e-5
            and float((buf - rbuf).abs().max()) < 1e-5
        )


def test_a_scheduled_group_does_not_move_a_fixed_group_bitwise():
    W, grads = _data(5, 8)
    W2, grads2 = _data(6, 8)
    a = nn.Parameter(W.float().clone())
    b = nn.Parameter(W2.float().clone())
    both = _opt(
        [
            _head_group(a, momentum_k=K),
            {**_head_group(b), "param_names": ["other.weight"]},
        ],
        ["lm_head.weight", "other.weight"],
    )
    solo_b = nn.Parameter(W2.float().clone())
    solo = _opt(
        [{**_head_group(solo_b), "param_names": ["other.weight"]}], ["other.weight"]
    )
    for G, G2 in zip(grads, grads2):
        a.grad, b.grad, solo_b.grad = (
            G.float().clone(),
            G2.float().clone(),
            G2.float().clone(),
        )
        both.step()
        solo.step()
    assert torch.equal(b.detach(), solo_b.detach())
    assert both.param_groups[1]["momentum"] == 0.05 and 1 not in both.momentum_used


def test_a_resume_continues_bitwise_and_the_schedule_follows_t():
    W, grads = _data(7, 10)

    def run(break_at):
        w = nn.Parameter(W.float().clone())
        opt = _opt([_head_group(w, momentum_k=K)], ["lm_head.weight"])
        for i, G in enumerate(grads):
            if i == break_at:
                sd = opt.state_dict()
                w2 = nn.Parameter(w.detach().clone())
                opt = _opt([_head_group(w2, momentum_k=K)], ["lm_head.weight"])
                opt.load_state_dict(sd)
                w = w2
            w.grad = G.float().clone()
            opt.step()
        return w.detach(), opt

    straight, o1 = run(-1)
    resumed, o2 = run(5)
    assert torch.equal(straight, resumed)
    assert o1.momentum_used == o2.momentum_used == {0: pytest.approx(_alpha(10))}


def test_a_nonpositive_k_is_refused():
    w = nn.Parameter(torch.randn(M, N))
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="momentum_k"):
            _opt([_head_group(w, momentum_k=bad)], ["lm_head.weight"])


def test_the_container_treats_momentum_k_as_a_config_only_disco_key():
    from torchtitan.optimizers.container import (
        _DISCO_ONLY_KEYS,
        DiSCOOptimizersContainer,
    )

    assert (
        "momentum_k" in _DISCO_ONLY_KEYS
        and "momentum_k" in DiSCOOptimizersContainer._CONFIG_ONLY_SCHEDULER_KEYS
    )
