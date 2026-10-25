# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`averages.start_from_gamma` (angular WSD): after a checkpoint load, the weights become the renormalised power average of that
gamma (exactly what the validation scores as avg_g<gamma>), the averages stop updating, and the optimizer state is untouched."""

import pytest
import torch
from torchtitan.components.optimizer.averages import WeightAverages

from tests.unit_tests.cpu.test_weight_averages import _by_name, _cfg, _run, Toy

RENORM = [["qkv", "spectral"], [r"^o\.", "frobenius"], ["w13", "frobenius"]]


def _setup(mode_rows=False, steps=9):
    torch.manual_seed(0)
    model = Toy()
    renorm = [["qkv", "rows"], ["w13", "rows"]] if mode_rows else RENORM
    avg = WeightAverages(_cfg(start_from_gamma=7.0, renorm=renorm), model_parts=[model])
    _run(avg, model, steps)
    return model, avg


@pytest.mark.parametrize("rows", [False, True])
def test_the_weights_become_the_scored_average_and_averaging_stops(rows):
    model, avg = _setup(rows)
    scored = _by_name(
        avg, model, 3
    )  # what the validation scores as avg_g7 (gamma index 3)
    state = {
        n: [avg.optimizers[0].state[p][f"avg_{i}"].clone() for i in range(4)]
        for n, p in model.named_parameters()
    }
    avg.start_from_average()
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), scored[n]), n
    assert avg.gammas == []
    for s in (10, 11):  # no further update of the averages
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        avg.step(s)
    for n, p in model.named_parameters():
        for i in range(4):
            assert torch.equal(avg.optimizers[0].state[p][f"avg_{i}"], state[n][i]), (
                n,
                i,
            )


def test_the_start_gamma_must_be_a_declared_gamma():
    with pytest.raises(ValueError, match="start_from_gamma"):
        WeightAverages.Config(gammas=[0.0, 1.0, 3.0, 7.0], start_from_gamma=15.0)


class _Val:
    def __init__(self):
        self.seen = []

    def validate(self, model_parts, step, tag=None):
        self.seen.append(
            (
                step,
                tag,
                {n: p.detach().clone() for n, p in model_parts[0].named_parameters()},
            )
        )


def test_the_first_validation_after_the_start_is_the_scored_average():
    """The gate (iv) at toy scale: the weights a validation sees right after the start
    are the weights it scored as avg_g7 before."""
    model, avg = _setup()
    v = _Val()
    avg.validate(
        v, 9
    )  # scores avg_g0 .. avg_g7 (the main run's validation at the branch point)
    scored7 = [x for x in v.seen if x[1] == "avg_g7"][0][2]
    avg.start_from_average()
    v2 = _Val()
    v2.validate([model], 9, tag="avgstart")
    for n in scored7:
        assert torch.equal(v2.seen[0][2][n], scored7[n]), n
