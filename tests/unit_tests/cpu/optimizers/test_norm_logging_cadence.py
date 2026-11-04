# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The arm -> step -> collect handshake for DiSCO's per-parameter norms.

Nothing type-checks the sequence and every way of breaking it is silent: norms
that are never computed just do not appear. Pins the cadence against the
expression it replaced, and that one arm covers exactly one step.
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from torchtitan.optimizers.container import DiSCOOptimizersContainer


def _reference_wants_norms(
    log_norm_freq: int, step: int, *, should_log: bool, is_final_step: bool
) -> bool:
    """The expression this replaced, transcribed from trainer.py before the move.

    Written from the original source rather than from the new implementation:
    a reference derived from the code under test agrees with its bugs.
    """
    return log_norm_freq > 0 and (
        is_final_step or (should_log and (step == 1 or step % log_norm_freq == 0))
    )


@pytest.mark.parametrize("log_norm_freq", [0, 1, 5, 900])
@pytest.mark.parametrize("step", [1, 2, 5, 10, 899, 900, 901, 1800])
@pytest.mark.parametrize("should_log", [True, False])
@pytest.mark.parametrize("is_final_step", [True, False])
def test_cadence_matches_the_expression_it_replaced(
    log_norm_freq, step, should_log, is_final_step
):
    container = MagicMock(spec=DiSCOOptimizersContainer)
    container.log_norm_freq = log_norm_freq
    container.dense_norm_steps = 0

    got = DiSCOOptimizersContainer.wants_norms_at(
        container, step, should_log=should_log, is_final_step=is_final_step
    )

    assert got == _reference_wants_norms(
        log_norm_freq, step, should_log=should_log, is_final_step=is_final_step
    )


@pytest.mark.parametrize("step", [1, 2, 499, 500, 501, 505, 510])
def test_dense_window_arms_every_logged_step(step):
    """Every logged step up to dense_norm_steps arms, then the cadence again."""
    container = MagicMock(spec=DiSCOOptimizersContainer)
    container.log_norm_freq = 10
    container.dense_norm_steps = 500
    got = DiSCOOptimizersContainer.wants_norms_at(
        container, step, should_log=True, is_final_step=False
    )
    assert got == (step <= 500 or step % 10 == 0)
    assert not DiSCOOptimizersContainer.wants_norms_at(
        container, step, should_log=False, is_final_step=False
    )


def test_disabled_frequency_never_arms():
    """log_norm_freq=0 must win over the final step, or every run logs norms."""
    container = MagicMock(spec=DiSCOOptimizersContainer)
    container.log_norm_freq = 0
    assert not DiSCOOptimizersContainer.wants_norms_at(
        container, 100, should_log=True, is_final_step=True
    )


class _FakeDiSCO:
    """Stands in for a DiSCO optimizer, mimicking the two-call handshake.

    `calculate_norm_at_next_step` sets a flag and clears the stash; `step`
    fills the stash only if armed and then clears the flag -- which is what
    `DiSCO.step` does at its end.
    """

    def __init__(self, has_head_geometry: bool = True) -> None:
        self.has_head_geometry = has_head_geometry
        self.head_probe_h = None
        self.need_to_calculate_norm = False
        self.norms_at_current_step: dict[str, float] = {}
        self.arm_kwargs: dict | None = None
        self.steps_that_computed: list[int] = []
        self._step_index = 0
        self.eta_used: dict[int, tuple[float, float]] = {}

    def calculate_norm_at_next_step(self, norms_to_log=None, gram_level=None, **kw):
        self.need_to_calculate_norm = True
        self.norms_at_current_step = {}
        self.arm_kwargs = {
            "norms_to_log": norms_to_log,
            "gram_level": gram_level,
            **kw,
        }

    def step(self):
        self._step_index += 1
        if self.need_to_calculate_norm:
            self.norms_at_current_step = {"weight_norm/w": 1.0}
            self.steps_that_computed.append(self._step_index)
        self.need_to_calculate_norm = False

    def get_norms_at_current_step(self):
        return self.norms_at_current_step


def _container_with(fake, monkeypatch, **attrs):
    """A container stub whose `isinstance(opt, DiSCO)` checks see `_FakeDiSCO`."""
    import torchtitan.optimizers.container as container_mod

    monkeypatch.setattr(container_mod, "DiSCO", _FakeDiSCO)
    container = MagicMock(spec=DiSCOOptimizersContainer)
    container.optimizers = [fake]
    container.model_parts = [MagicMock()]
    container.norms_to_log = ["all"]
    container.gram_level = 2
    container.log_metrics_locally = True
    container.spectrum_logging_config = MagicMock(enable_export=True)
    for k, v in attrs.items():
        setattr(container, k, v)
    return container


def test_arming_reaches_the_optimizer_with_the_containers_settings(monkeypatch):
    fake = _FakeDiSCO()
    container = _container_with(fake, monkeypatch)

    DiSCOOptimizersContainer.arm_norm_computation(container)

    assert fake.need_to_calculate_norm is True
    assert fake.arm_kwargs["norms_to_log"] == ["all"]
    assert fake.arm_kwargs["gram_level"] == 2
    assert fake.arm_kwargs["log_metrics_locally"] is True
    # track_spectrum is derived, not copied: the optimizer can skip computing,
    # packing and all-gathering the spectra when nothing will consume them.
    # The plot path is gone, so the parquet export is the only consumer and
    # this now tracks enable_export alone.
    assert fake.arm_kwargs["track_spectrum"] is True


def test_a_step_consumes_the_arm_exactly_once(monkeypatch):
    """The regression that matters: one arm must not log norms forever after."""
    fake = _FakeDiSCO()
    container = _container_with(fake, monkeypatch)

    DiSCOOptimizersContainer.arm_norm_computation(container)
    fake.step()
    fake.step()
    fake.step()

    assert fake.steps_that_computed == [1], (
        "norms were computed on a step that was never armed; DiSCO.step must "
        "reset need_to_calculate_norm"
    )


def test_an_unarmed_step_produces_no_norms(monkeypatch):
    fake = _FakeDiSCO()
    _container_with(fake, monkeypatch)

    fake.step()

    assert fake.get_norms_at_current_step() == {}


def test_spectra_are_not_computed_when_nothing_consumes_them(monkeypatch):
    """The whole point of `track_spectrum`: skip the work when it is discarded.

    The spectra are ~99% of the logging gather's payload, so computing them
    with the export off would be pure waste.
    """
    fake = _FakeDiSCO()
    container = _container_with(fake, monkeypatch)
    container.spectrum_logging_config = MagicMock(enable_export=False)

    DiSCOOptimizersContainer.arm_norm_computation(container)

    assert fake.arm_kwargs["track_spectrum"] is False


def _live_container(fake, monkeypatch, *, log_norm_freq: int = 5):
    """The container's real methods over `fake`, built without a model."""
    import torchtitan.optimizers.container as container_mod

    monkeypatch.setattr(container_mod, "DiSCO", _FakeDiSCO)
    container = object.__new__(DiSCOOptimizersContainer)
    # torch wraps `step` with its hook runner once any container is built.
    container._optimizer_step_pre_hooks = OrderedDict()
    container._optimizer_step_post_hooks = OrderedDict()
    container.optimizers = [fake]
    container.model_parts = []
    container.log_norm_freq = log_norm_freq
    container.dense_norm_steps = 0
    container.norms_to_log = ["all"]
    container.gram_level = 0
    container.log_metrics_locally = False
    container.spectrum_logging_config = MagicMock(enable_export=False)
    container._moe_publish_armed = False
    container._norms_armed = False
    container._head_input = None
    container.parallel_dims = None
    container.log_residual_stats = False
    container.log_gain_stats = False
    container.log_sink_stats = False
    container._model_stats_armed = False
    container._gains_before = None
    container.head_diag_probes = 512
    container.get_parameter_norms = lambda step: dict(fake.norms_at_current_step)
    return container


def test_the_head_input_is_asked_for_and_handed_over_on_norm_steps_only(monkeypatch):
    """begin_step (before the forward) says how many rows to keep; step()
    hands them to DiSCO, for that step only."""
    fake = _FakeDiSCO()
    container = _live_container(fake, monkeypatch)
    calls = []

    def head_input():
        calls.append(1)
        return "h"

    rows = container.begin_step(
        4, should_log=True, is_final_step=False, head_input=head_input
    )
    container.step()
    assert rows == 0 and calls == [] and fake.head_probe_h is None
    rows = container.begin_step(
        5, should_log=True, is_final_step=False, head_input=head_input
    )
    assert rows == 512 and calls == []  # nothing is taken before the step
    container.step()
    assert calls == [1] and fake.head_probe_h == "h"


def test_begin_step_arms_the_moe_metrics_on_logging_steps_only(monkeypatch):
    container = _live_container(_FakeDiSCO(), monkeypatch)
    container.begin_step(4, should_log=False, is_final_step=False)
    assert not container.consume_moe_publish_arm()
    container.begin_step(4, should_log=True, is_final_step=False)
    assert container.consume_moe_publish_arm()


def test_collect_reads_norms_back_only_after_an_armed_step(monkeypatch):
    fake = _FakeDiSCO()
    container = _live_container(fake, monkeypatch)
    container.begin_step(5, should_log=True, is_final_step=False)
    fake.step()
    assert "weight_norm/w" in container.collect_metrics(5)
    # The fake keeps its stash; the container must not read it unarmed.
    container.begin_step(6, should_log=True, is_final_step=False)
    fake.step()
    assert container.collect_metrics(6) == {}


def test_no_head_input_when_no_disco_holds_the_head(monkeypatch):
    """A head on AdamW, or no head at all: nothing to keep."""
    fake = _FakeDiSCO(has_head_geometry=False)
    container = _live_container(fake, monkeypatch)
    calls = []
    rows = container.begin_step(
        5, should_log=True, is_final_step=False, head_input=lambda: calls.append(1)
    )
    assert rows == 0 and fake.need_to_calculate_norm  # the norms are still armed
    container.step()
    assert calls == []
