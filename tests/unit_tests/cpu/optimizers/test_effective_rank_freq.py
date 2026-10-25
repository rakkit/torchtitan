# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""optimizer.effective_rank_freq: effective_rank (the one norm that needs the whole spectrum) is computed on the
norm-logging steps divisible by N, the first step and the final step; every other norm on every logging step. 0 (the
default) leaves norms_to_log untouched on every step."""

from unittest.mock import MagicMock

import pytest

from torchtitan.optimizers.container import DiSCOOptimizersContainer

NORMS = ["frobenius_norm", "stable_rank", "effective_rank", "rms_to_rms"]


class _Fake:
    def __init__(self):
        self.need_to_calculate_norm = False
        self.has_head_geometry = False
        self.arm_kwargs = None

    def calculate_norm_at_next_step(self, norms_to_log=None, gram_level=None, **kw):
        self.need_to_calculate_norm = True
        self.arm_kwargs = {"norms_to_log": norms_to_log, **kw}


def _container(monkeypatch, freq, norms=NORMS, log_norm_freq=10, dense=500):
    import torchtitan.optimizers.container as container_mod

    monkeypatch.setattr(container_mod, "DiSCO", _Fake)
    fake = _Fake()
    c = MagicMock(spec=DiSCOOptimizersContainer)
    c.optimizers, c.model_parts = [fake], []
    c.norms_to_log, c.effective_rank_freq, c.spectral_norm_freq = list(norms), freq, 0
    c.gnoise = None  # a container built with gnoise off
    c.log_norm_freq, c.dense_norm_steps = log_norm_freq, dense
    c.gram_level, c.log_metrics_locally = 0, False
    c.spectrum_logging_config = MagicMock(enable_export=False)
    c._head_input = None
    c._model_stats_armed = False
    c.arm_moe_publish = MagicMock()
    c._arm_model_stats = MagicMock()
    # the real methods over the stub
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
    fake.arm_kwargs, fake.need_to_calculate_norm = None, False
    DiSCOOptimizersContainer.begin_step(c, step, should_log=True, is_final_step=final)
    return None if fake.arm_kwargs is None else fake.arm_kwargs["norms_to_log"]


def test_the_default_leaves_every_logged_step_untouched(monkeypatch):
    c, fake = _container(monkeypatch, 0)
    for step in (1, 2, 499, 500, 510, 515, 600):
        got = _armed(c, fake, step)
        if step <= 500 or step % 10 == 0:
            assert got == NORMS, step
        else:
            assert got is None, step  # not a norm-logging step


@pytest.mark.parametrize("freq", [100, 250])
def test_effective_rank_is_dropped_off_cadence_only(monkeypatch, freq):
    c, fake = _container(monkeypatch, freq)
    without = [n for n in NORMS if n != "effective_rank"]
    for step in (1, 2, 50, 99, 100, 101, 250, 499, 500, 510, 520, 600, 1000):
        got = _armed(c, fake, step)
        if got is None:
            assert step > 500 and step % 10, step
            continue
        due = step == 1 or step % freq == 0
        assert got == (NORMS if due else without), (freq, step)
    assert _armed(c, fake, 777, final=True) == NORMS  # the final step always logs it


def test_a_list_without_effective_rank_is_never_changed(monkeypatch):
    c, fake = _container(monkeypatch, 100, norms=["frobenius_norm", "stable_rank"])
    for step in (1, 7, 100, 101):
        assert _armed(c, fake, step) == ["frobenius_norm", "stable_rank"]


def test_the_config_field_defaults_to_every_logging_step():
    assert DiSCOOptimizersContainer.Config().effective_rank_freq == 0
    assert (
        DiSCOOptimizersContainer.effective_rank_freq == 0
    )  # class default: a container built without the field
