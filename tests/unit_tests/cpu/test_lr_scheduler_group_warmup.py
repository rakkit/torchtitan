# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A param group's own `warmup_steps`: the shared schedule times
min(1, t / warmup_steps) at the optimizer's step t, for that group only.
Its own `decay_ratio`: the trainer's schedule with that decay share."""

import functools

import torch

from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer


def _wsd(current_step, *, stable=12, decay=4):
    """A stand-in shared schedule: flat, then a linear decay."""
    if current_step < stable:
        return 1.0
    return max(0.0, 1.0 - (current_step - stable + 1) / decay)


def _container(warmup_steps=None):
    a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    groups = [{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.04}]
    if warmup_steps is not None:
        groups[1]["warmup_steps"] = warmup_steps
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    opt._opt_called = True  # no optimizer.step() here; silence torch's order warning
    return opt, LRSchedulersContainer([opt], functools.partial(_wsd))


def _lrs(opt, sched, steps):
    """The lr each group runs at on the optimizer's steps 1..steps."""
    out = []
    for _ in range(steps):
        out.append([g["lr"] for g in opt.param_groups])
        sched.step()
    return out


def test_only_the_group_with_warmup_steps_warms_up():
    opt, sched = _container(warmup_steps=4)
    plain_opt, plain_sched = _container()
    got, plain = _lrs(opt, sched, 16), _lrs(plain_opt, plain_sched, 16)
    for t, ((lr_a, lr_b), (p_a, p_b)) in enumerate(zip(got, plain), start=1):
        assert lr_a == p_a  # the other group: the shared schedule, bitwise
        assert abs(lr_b - p_b * min(1.0, t / 4)) < 1e-15, t


def test_without_warmup_steps_the_schedulers_are_as_before():
    opt, sched = _container()
    assert sched.schedulers[0].lr_lambdas[0] is sched.schedulers[0].lr_lambdas[1]


def test_a_resumed_scheduler_continues_the_group_warmup():
    opt, sched = _container(warmup_steps=6)
    full = _lrs(opt, sched, 10)
    opt2, sched2 = _container(warmup_steps=6)
    _lrs(opt2, sched2, 3)
    state = sched2.state_dict()
    opt3, sched3 = _container(warmup_steps=6)
    sched3.load_state_dict(state)
    for g, lr in zip(opt3.param_groups, sched3.schedulers[0].get_last_lr()):
        g["lr"] = lr
    assert _lrs(opt3, sched3, 7) == full[3:]


# ------------------------------------------------------------ own decay_ratio
T = 40


def _trainer(group_b=None, *, decay_ratio=0.2, warmup_steps=0):
    """Two AdamW groups under the trainer's real schedule over T steps."""
    a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    groups = [
        {"params": [a], "lr": 0.1},
        {"params": [b], "lr": 0.04, **(group_b or {})},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    opt._opt_called = True
    cfg = LRSchedulersContainer.Config(
        warmup_steps=warmup_steps,
        decay_ratio=decay_ratio,
        decay_type="linear",
        min_lr_factor=0.0,
    )
    return opt, cfg.build(optimizers=[opt], training_steps=T)


def _wsd_factor(t, warmup, decay):
    """The trainer's warmup-stable-decay at the optimizer's step t (1-based),
    written from its definition: a linear warmup over `warmup` steps, 1 until
    the last `decay` steps, then linear to 0 at the virtual step T + 1."""
    if t <= warmup:
        return t / warmup
    return min(1.0, (T + 1 - t) / decay)


def test_a_group_decays_on_its_own_ratio_after_its_own_warmup():
    """warmup 8 then a linear decay over everything after it (decay_ratio 1)."""
    opt, sched = _trainer({"warmup_steps": 8, "decay_ratio": 1.0})
    plain_opt, plain_sched = _trainer()
    got, plain = _lrs(opt, sched, T), _lrs(plain_opt, plain_sched, T)
    for t, ((lr_a, lr_b), (p_a, _)) in enumerate(zip(got, plain), start=1):
        assert lr_a == p_a  # the other group: the trainer's schedule, bitwise
        assert abs(lr_a - 0.1 * _wsd_factor(t, 0, 8)) < 1e-15, t  # 20% of 40
        assert abs(lr_b - 0.04 * _wsd_factor(t, 8, T - 8)) < 1e-15, t
    assert got[7][1] == 0.04 and got[8][1] == 0.04 and got[9][1] < 0.04
    assert 0.0 < got[-1][1] < 0.04 / (T - 8) * 1.01


def test_a_decay_ratio_without_warmup_moves_only_the_decay_start():
    opt, sched = _trainer({"decay_ratio": 0.5})
    for t, (_, lr_b) in enumerate(_lrs(opt, sched, T), start=1):
        assert abs(lr_b - 0.04 * _wsd_factor(t, 0, 20)) < 1e-15, t


def test_a_group_decay_ratio_equal_to_the_trainers_changes_nothing():
    opt, sched = _trainer({"decay_ratio": 0.2})
    plain_opt, plain_sched = _trainer()
    assert _lrs(opt, sched, T) == _lrs(plain_opt, plain_sched, T)


def test_a_disco_group_takes_its_own_warmup():
    """DiSCO accepts the scheduler's group keys and the scheduler warms that
    group only; any other unknown key is still refused."""
    import os
    import socket

    import pytest
    import torch.distributed as dist

    from torchtitan.models.common.param_layout import ParamLayout
    from torchtitan.optimizers.disco.optimizer import DiSCO

    class _NoParallel:
        dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
        fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
        world_mesh = None

        def get_optional_mesh(self, dims):
            return None

    own = not dist.is_initialized()
    if own:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
    try:
        a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
        sign = {
            "rule": "sgd",
            "args": {"compute": "sign", "scale": "one"},
            "momentum": 0.1,
        }
        groups = [
            {"params": [a], "param_names": ["a"], "lr": 0.04, **sign},
            {
                "params": [b],
                "param_names": ["b"],
                "lr": 0.04,
                "warmup_steps": 4,
                **sign,
            },
        ]
        layouts = {n: ParamLayout("vector") for n in "ab"}
        opt = DiSCO(groups, parallel_dims=_NoParallel(), lr=0.04, param_layouts=layouts)
        opt._opt_called = True
        got = _lrs(opt, LRSchedulersContainer([opt], functools.partial(_wsd)), 16)
        for t, (lr_a, lr_b) in enumerate(got, start=1):
            assert lr_a == 0.04 * _wsd(t - 1)
            assert abs(lr_b - 0.04 * _wsd(t - 1) * min(1.0, t / 4)) < 1e-15, t
        with pytest.raises(ValueError, match="unknown keys"):
            DiSCO(
                [{**groups[0], "params": [a], "warmup": 4}],
                parallel_dims=_NoParallel(),
                lr=0.04,
                param_layouts=layouts,
            )
    finally:
        if own:
            dist.destroy_process_group()


# ------------------------------------------- own decay_type / min_lr_factor (r98)
import math  # noqa: E402

import pytest  # noqa: E402


def _cosine_trainer(group_b=None):
    """The same two groups under a trainer cosine from the end of the warmup (no stable phase)."""
    a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    groups = [
        {"params": [a], "lr": 0.1},
        {"params": [b], "lr": 0.04, **(group_b or {})},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    opt._opt_called = True
    cfg = LRSchedulersContainer.Config(
        warmup_steps=0, decay_ratio=None, decay_type="cosine", min_lr_factor=0.0
    )
    return opt, cfg.build(optimizers=[opt], training_steps=T)


def _cos_factor(t):
    """Cosine to 0 over all T steps, from the definition: 0.5 (1 + cos(pi x)), x = (t - 1) / T on the optimizer's step t."""
    return 0.5 * (1.0 + math.cos(math.pi * (t - 1) / T))


def test_a_group_cosine_in_a_linear_wsd_trainer():
    opt, sched = _trainer({"decay_type": "cosine", "decay_ratio": 1.0})
    plain_opt, plain_sched = _trainer()
    got, plain = _lrs(opt, sched, T), _lrs(plain_opt, plain_sched, T)
    for t, ((lr_a, lr_b), (p_a, _)) in enumerate(zip(got, plain), start=1):
        assert lr_a == p_a  # the other group keeps the trainer's linear WSD, bitwise
        assert abs(lr_b - 0.04 * _cos_factor(t)) < 1e-15, t


def test_a_group_linear_wsd_in_a_cosine_trainer():
    opt, sched = _cosine_trainer({"decay_type": "linear", "decay_ratio": 0.2})
    plain_opt, plain_sched = _cosine_trainer()
    got, plain = _lrs(opt, sched, T), _lrs(plain_opt, plain_sched, T)
    for t, ((lr_a, lr_b), (p_a, _)) in enumerate(zip(got, plain), start=1):
        assert lr_a == p_a  # the other group keeps the trainer's cosine, bitwise
        assert abs(lr_a - 0.1 * _cos_factor(t)) < 1e-15, t
        assert abs(lr_b - 0.04 * _wsd_factor(t, 0, 8)) < 1e-15, t


def test_a_group_floor_lifts_only_its_decay():
    opt, sched = _trainer({"min_lr_factor": 0.1})
    plain_opt, plain_sched = _trainer()
    got, plain = _lrs(opt, sched, T), _lrs(plain_opt, plain_sched, T)
    for t, ((lr_a, lr_b), (p_a, _)) in enumerate(zip(got, plain), start=1):
        assert lr_a == p_a  # the other group still decays to 0, bitwise
        w = _wsd_factor(t, 0, 8)
        want = (
            1.0 if w == 1.0 else 0.1 + 0.9 * w
        )  # the floor acts in the decay phase only: x -> floor + (1 - floor) x
        assert abs(lr_b - 0.04 * want) < 1e-15, t
    assert (
        abs(got[-1][1] - 0.04 * (0.1 + 0.9 * 0.125)) < 1e-15
        and abs(got[-1][0] - 0.1 * 0.125) < 1e-15
    )  # step T: 1/8 of the way down


def test_a_group_decay_type_and_floor_equal_to_the_trainers_change_nothing():
    for build in (_trainer, _cosine_trainer):
        kind = "linear" if build is _trainer else "cosine"
        ratio = {"decay_ratio": 0.2} if build is _trainer else {}
        opt, sched = build({"decay_type": kind, "min_lr_factor": 0.0, **ratio})
        plain_opt, plain_sched = build()
        assert _lrs(opt, sched, T) == _lrs(plain_opt, plain_sched, T)  # bitwise


def test_decay_ratio_zero_is_a_constant_lr_for_that_group():
    for build in (_trainer, _cosine_trainer):
        opt, sched = build({"decay_ratio": 0.0})
        assert all(lr_b == 0.04 for _, lr_b in _lrs(opt, sched, T))


def test_an_unknown_group_decay_type_is_refused_when_the_schedule_is_built():
    with pytest.raises(ValueError, match="decay_type"):
        _trainer({"decay_type": "exp", "decay_ratio": 1.0})


def test_a_disco_group_takes_its_own_decay_type_and_floor_and_resumes_with_them():
    """DiSCO accepts the two keys (not any other), the scheduler uses them for that group only, and a
    state_dict round trip keeps them."""
    import os
    import socket

    import torch.distributed as dist

    from torchtitan.models.common.param_layout import ParamLayout
    from torchtitan.optimizers.disco.optimizer import DiSCO

    class _NoParallel:
        dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
        fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
        world_mesh = None

        def get_optional_mesh(self, dims):
            return None

    own = not dist.is_initialized()
    if own:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
    try:
        layouts = {n: ParamLayout("vector") for n in "ab"}
        sign = {
            "rule": "sgd",
            "args": {"compute": "sign", "scale": "one"},
            "momentum": 0.1,
        }

        def build(extra):
            a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
            groups = [
                {"params": [a], "param_names": ["a"], "lr": 0.04, **sign},
                {"params": [b], "param_names": ["b"], "lr": 0.04, **sign, **extra},
            ]
            opt = DiSCO(
                groups, parallel_dims=_NoParallel(), lr=0.04, param_layouts=layouts
            )
            opt._opt_called = True
            cfg = LRSchedulersContainer.Config(
                warmup_steps=0, decay_ratio=0.2, decay_type="linear", min_lr_factor=0.0
            )
            return opt, cfg.build(optimizers=[opt], training_steps=T)

        extra = {"decay_type": "cosine", "decay_ratio": 1.0, "min_lr_factor": 0.1}
        opt, sched = build(extra)
        got = _lrs(opt, sched, T)
        for t, (lr_a, lr_b) in enumerate(got, start=1):
            assert abs(lr_a - 0.04 * _wsd_factor(t, 0, 8)) < 1e-15, t
            assert abs(lr_b - 0.04 * (0.1 + 0.9 * _cos_factor(t))) < 1e-15 or t == 1, t
        assert got[0][1] == 0.04
        with pytest.raises(ValueError, match="unknown keys"):
            build({"decay_kind": "cosine"})
        sd = opt.state_dict()
        assert (
            sd["param_groups"][1]["decay_type"] == "cosine"
            and sd["param_groups"][1]["min_lr_factor"] == 0.1
        )
        opt2, _ = build(extra)
        opt2.load_state_dict(sd)
        assert [g.get("decay_type") for g in opt2.param_groups] == [None, "cosine"]
    finally:
        if own:
            dist.destroy_process_group()
