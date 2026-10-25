# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""optimizer.spectral_norm_freq: the SVD-based metrics (rms_to_rms, stable_rank,
condition_number, effective_rank*, the spectra, the spectral radial metrics) only on
the norm-logging steps divisible by N, step 1 and the final step; the SVD-free norms
keep log_norm_freq. 0 (the default) changes nothing. Training is unchanged either way."""

import socket
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.optimizers.disco import metrics as disco_metrics
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.helpers.norm_helper import _SVD_FREE_NORMS, get_norms_to_log
from torchtitan.optimizers.helpers.radial_helper import RADIAL_METRIC_NAMES

NORMS = ["frobenius_norm", "stable_rank", "effective_rank", "rms_to_rms", "rms_to_inf"]
CHEAP = [n for n in NORMS if n in _SVD_FREE_NORMS]
SPECTRAL_RADIAL = {  # written out, not imported: the radial metrics that need sigma_max
    "spectral_radius",
    "spectral_radius_next",
    "spectral_growth",
    "spectral_relative_step",
    "radiality_rms_to_rms",
    "aus_rms_to_rms",
}


# ---------------------------------------------------------------- (a) the schedule
class _Fake:
    def __init__(self):
        self.has_head_geometry = False
        self.kw = None

    def calculate_norm_at_next_step(self, norms_to_log=None, gram_level=None, **kw):
        self.kw = {"norms_to_log": norms_to_log, **kw}


def _container(monkeypatch, freq, er_freq=0, log_norm_freq=10, dense=500, export=True):
    import torchtitan.optimizers.container as container_mod

    monkeypatch.setattr(container_mod, "DiSCO", _Fake)
    fake = _Fake()
    c = MagicMock(spec=DiSCOOptimizersContainer)
    c.optimizers, c.model_parts = [fake], []
    c.norms_to_log, c.effective_rank_freq, c.spectral_norm_freq = (
        list(NORMS),
        er_freq,
        freq,
    )
    c.log_norm_freq, c.dense_norm_steps = log_norm_freq, dense
    c.gram_level, c.log_metrics_locally = 0, False
    c.spectrum_logging_config = MagicMock(enable_export=export)
    c._head_input, c._model_stats_armed, c.gnoise = None, False, None
    c.arm_moe_publish, c._arm_model_stats = MagicMock(), MagicMock()
    c.wants_norms_at = lambda step, **kw: DiSCOOptimizersContainer.wants_norms_at(
        c, step, **kw
    )
    c.arm_norm_computation = (
        lambda norms=None, **kw: DiSCOOptimizersContainer.arm_norm_computation(
            c, norms, **kw
        )
    )
    c._norms_at = lambda step, final: DiSCOOptimizersContainer._norms_at(c, step, final)
    c._spectral_at = lambda step, final: DiSCOOptimizersContainer._spectral_at(
        c, step, final
    )
    return c, fake


def _armed(c, fake, step, final=False):
    fake.kw = None
    DiSCOOptimizersContainer.begin_step(c, step, should_log=True, is_final_step=final)
    return fake.kw


def test_the_default_changes_nothing(monkeypatch):
    c, fake = _container(monkeypatch, 0)
    for step in (1, 2, 499, 500, 510, 515, 600, 700):
        kw = _armed(c, fake, step)
        if step <= 500 or step % 10 == 0:
            assert kw["norms_to_log"] == NORMS and kw["spectral"] is True, step
            assert kw["track_spectrum"] is True, step
        else:
            assert kw is None, step


@pytest.mark.parametrize("freq", [100, 250])
def test_the_spectral_part_runs_on_its_cadence_only(monkeypatch, freq):
    c, fake = _container(monkeypatch, freq)
    for step in (1, 2, 50, 99, 100, 101, 250, 499, 500, 510, 520, 600, 1000):
        kw = _armed(c, fake, step)
        if kw is None:
            assert step > 500 and step % 10, step
            continue
        due = step == 1 or step % freq == 0
        assert kw["spectral"] is due, (freq, step)
        assert kw["norms_to_log"] == (NORMS if due else CHEAP), (freq, step)
        assert kw["track_spectrum"] is due, (
            freq,
            step,
        )  # the Parquet spectra need the SVD
    kw = _armed(c, fake, 777, final=True)
    assert kw["spectral"] is True and kw["norms_to_log"] == NORMS


def test_with_the_effective_rank_cadence(monkeypatch):
    c, fake = _container(monkeypatch, 100, er_freq=200)
    assert _armed(c, fake, 100)["norms_to_log"] == [
        n for n in NORMS if n != "effective_rank"
    ]
    assert _armed(c, fake, 200)["norms_to_log"] == NORMS
    assert _armed(c, fake, 210)["norms_to_log"] == CHEAP


@pytest.mark.parametrize(
    "freq,ok", [(0, True), (10, True), (100, True), (15, False), (-10, False)]
)
def test_the_knob_must_be_a_multiple_of_log_norm_freq(freq, ok):
    cfg = DiSCOOptimizersContainer.Config(log_norm_freq=10, spectral_norm_freq=freq)
    if ok:
        DiSCOOptimizersContainer._validate(cfg)
    else:
        with pytest.raises(ValueError, match="spectral_norm_freq"):
            DiSCOOptimizersContainer._validate(cfg)
    assert DiSCOOptimizersContainer.Config().spectral_norm_freq == 0
    assert DiSCOOptimizersContainer.spectral_norm_freq == 0


# ---------------------------------------------------------------- (b) DiSCO's keys and values
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


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


RULES = {
    "muon_wd": dict(
        rule="muon",
        args={"compute": "svd"},
        group=dict(momentum=0.1, nesterov=True, weight_decay=0.1),
    ),
    "f_mat_row_off": dict(
        rule="angular_muown",
        args=dict(
            compute="svd",
            scale="one",
            warmup=0,
            c=0.0,
            native_norm=False,
            sphere="matrix",
            tangent=False,
            radius=2.0,
            magnitude_lr=0.3,
        ),
        group=dict(momentum=1.0, nesterov=False, weight_decay=0.0),
    ),
}


def _run(case, spectral_steps, steps=4, spy=None):
    gen = torch.Generator().manual_seed(3)
    W = torch.randn(12, 8, generator=gen)
    grads = [0.1 * torch.randn(12, 8, generator=gen) for _ in range(steps)]
    w = nn.Parameter(W.clone())
    r = RULES[case]
    opt = DiSCO(
        [{"params": [w], "param_names": ["w"], **r["group"]}],
        parallel_dims=_NoParallel(),
        lr=0.02,
        rule=r["rule"],
        args=dict(r["args"]),
        param_layouts={"w": ParamLayout("matrix")},
    )
    all_norms = get_norms_to_log("all")
    logs = []
    for t, G in enumerate(grads, start=1):
        spectral = t in spectral_steps
        norms = (
            all_norms if spectral else [n for n in all_norms if n in _SVD_FREE_NORMS]
        )
        opt.calculate_norm_at_next_step(
            norms_to_log=norms, gram_level=0, track_spectrum=spectral, spectral=spectral
        )
        if spy is not None:
            spy.clear()
        w.grad = G.clone()
        opt.step()
        logs.append((opt.get_norms_at_current_step(), list(spy or [])))
    return w.detach().clone(), logs


@pytest.mark.parametrize("case", list(RULES))
def test_keys_appear_exactly_on_their_steps_and_training_is_unchanged(
    one_rank, case, monkeypatch
):
    calls = []
    real = disco_metrics.radial_spectral_inputs

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(disco_metrics, "radial_spectral_inputs", spy)
    w_all, every = _run(case, {1, 2, 3, 4})
    w_some, some = _run(case, {1, 3}, spy=calls)
    assert torch.equal(w_all, w_some)  # training unchanged
    svd_norms = [n for n in get_norms_to_log("all") if n not in _SVD_FREE_NORMS]
    assert {"rms_to_rms", "stable_rank", "condition_number"} <= set(svd_norms)
    dropped = {f"track_{k}_{n}/w" for k in ("update", "param") for n in svd_norms}
    dropped |= {f"track_radial_{n}/w" for n in SPECTRAL_RADIAL}
    dropped |= {"track_spectrum_update/w", "track_spectrum_param/w"}
    for t, ((full, _), (got, spy_calls)) in enumerate(zip(every, some), start=1):
        if t in (1, 3):
            assert set(got) == set(full), t
            assert spy_calls, t
        else:
            assert set(got) == set(full) - dropped, (
                t,
                sorted(set(full) - set(got) ^ dropped),
            )
            assert not spy_calls, t  # the spectral inputs are not computed at all
        for k in got:  # every key it logs has the value of the all-spectral run
            a, b = torch.as_tensor(full[k]), torch.as_tensor(got[k])
            if k.startswith(("track_update_", "track_param_")) and t not in (1, 3):
                # the SVD-free tier's fused kernel: the same norms, other rounding
                assert torch.allclose(a, b, rtol=1e-6, atol=0), (t, k)
            else:
                assert torch.equal(a, b), (t, k)
    assert dropped <= set(every[1][0]) and len(RADIAL_METRIC_NAMES) > len(
        SPECTRAL_RADIAL
    )
