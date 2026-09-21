# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The arm -> step -> collect handshake for DiSCO's per-parameter norms.

The norms are a by-product of the optimizer update: `arm_norm_computation()`
sets a flag, `step()` fills `norms_at_current_step` and clears the flag again,
and only then can the trainer read them back. Nothing type-checks that
sequence, and every way of breaking it is silent -- norms that are never
computed simply do not appear on the dashboard.

Two things are pinned here:

1. `wants_norms_at` reproduces the cadence expression it replaced, which used
   to live inline in `Trainer.train_step`.
2. Arming reaches every DiSCO optimizer with the container's logging settings,
   and a step consumes it exactly once.
"""

from __future__ import annotations

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

    got = DiSCOOptimizersContainer.wants_norms_at(
        container, step, should_log=should_log, is_final_step=is_final_step
    )

    assert got == _reference_wants_norms(
        log_norm_freq, step, should_log=should_log, is_final_step=is_final_step
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
    `DiSCO.step` does at disco.py:3166, its last statement before `return`.
    """

    def __init__(self) -> None:
        self.need_to_calculate_norm = False
        self.norms_at_current_step: dict[str, float] = {}
        self.arm_kwargs: dict | None = None
        self.steps_that_computed: list[int] = []
        self._step_index = 0

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
