# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regressions for the MoE router metrics pipeline.

Producer: ``NormMoE``/``NormRouter`` buffers. Consumer:
``optimizers/container.py``'s step pre-hook.
"""

import types

import torch
import torch.nn as nn

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.opt_moe.norm_moe import LoadBalanceLoss, make_norm_moe_config
from torchtitan.optimizers.moe_balance import (
    publish_moe_metrics,
    register_moe_load_balancing_hook,
)


def _moe(**kw):
    """An MoE whose expert compute is stubbed out.

    These tests are about the metrics buffers, which are all filled before the
    experts run. Keeping the real `routed_experts` would drag in the expert
    dispatcher, whose collective path needs an ambient DeviceMesh that a CPU
    unit test has no business creating.
    """
    moe = make_norm_moe_config(
        dim=32, hidden_dim=64, num_experts=8, top_k=2, **kw
    ).build()
    moe.routed_experts = _StubExperts()
    return moe


class _StubExperts(nn.Module):
    def forward(self, x_TD, *args, **kwargs):
        return torch.zeros_like(x_TD)


def _run_bias_hook(gas: int, *, arm: bool = True) -> list:
    """Drive the real step pre-hook and return the MoE layers it updated."""
    import torch.nn as nn

    class _Blk(nn.Module):
        def __init__(self, moe, lid):
            super().__init__()
            self.moe, self.moe_enabled, self.layer_id = moe, True, lid

    class _Part(nn.Module):
        def __init__(self, moes):
            super().__init__()
            self.layers = nn.ModuleDict(
                {str(i): _Blk(m, i) for i, m in enumerate(moes)}
            )

    class _StubOptimizers:
        def __init__(self):
            self.hooks = []
            self.armed = False

        def arm_moe_publish(self):
            self.armed = True

        def consume_moe_publish_arm(self):
            armed, self.armed = self.armed, False
            return armed

        def register_step_pre_hook(self, fn):
            self.hooks.append(fn)

        def fire(self):
            for h in self.hooks:
                h()

    torch.manual_seed(0)
    moes = [_moe(aux_loss=LoadBalanceLoss.Config(coeff=1e-3)) for _ in range(2)]
    for m in moes:
        m.train()
    opt = _StubOptimizers()
    register_moe_load_balancing_hook(
        opt,
        [_Part(moes)],
        types.SimpleNamespace(
            get_optional_mesh=lambda _n: None, ep_enabled=False, tp=1
        ),
    )
    batches = [torch.randn(64, 32) for _ in range(gas)]
    AuxLoss.set_step_denominator(torch.tensor(float(sum(b.shape[0] for b in batches))))
    for b in batches:
        for m in moes:
            m(b)
    if arm:
        opt.arm_moe_publish()
    opt.fire()
    return moes


def test_published_load_balance_loss_is_gradient_accumulation_invariant():
    """`instance_acc` already carries 1/global_valid_tokens for the WHOLE step.

    The consumer therefore must not divide it by `acc_fwd_times`. It used to,
    which made `moe_load_balance_loss/L-*` scale as exactly 1/gas.
    """
    seen = {
        gas: _run_bias_hook(gas)[0]._log_expert_metrics["moe_load_balance_loss/L-0"]
        for gas in (1, 2, 4)
    }
    lo, hi = min(seen.values()), max(seen.values())
    assert hi / lo < 1.05, f"published metric should be gas-invariant, got {seen}"


def test_unarmed_step_updates_the_bias_without_publishing():
    """Off-cadence steps do the bias update only: no D2H copy, no queue put."""
    moes = _run_bias_hook(1, arm=False)
    for moe in moes:
        assert not getattr(moe, "_log_expert_metrics", None), "published unasked"
    assert any(float(m.expert_bias_E.abs().sum()) > 0 for m in moes)


def test_load_balance_loss_reduces_over_cp():
    """It is weighted by the CP-sharded `loss_mask.sum()`, so dp alone drops cp."""
    assert LoadBalanceLoss.Config(coeff=1e-3).reduce_mesh == "loss"


def test_publish_clears_stale_metrics_when_the_payload_arity_is_wrong():
    """A producer/consumer skew must produce a gap, not last step's numbers."""
    E, L = 4, 2
    info = [
        {"module": types.SimpleNamespace(_log_expert_metrics={}), "layer_id": i}
        for i in range(L)
    ]
    good = (
        info,
        [0.25] * (L * E),
        [0.1] * (L * E),
        [1.0] * L,
        [0.0] * L,
        [2.0] * L,
        [3.0] * L,
        [0.1] * L,
        [0.0] * L,
        E,
        False,
    )
    publish_moe_metrics(good)
    assert info[0]["module"]._log_expert_metrics["moe_entropy/L-0"] == 1.0

    publish_moe_metrics((info, [0.25] * (L * E)))  # too few fields: unpack raises
    for i in info:
        assert i["module"]._log_expert_metrics == {}, "stale metrics republished"


def test_track_router_metrics_still_gates_entropy():
    """The flag reaches the router even though NormMoE no longer stores it."""
    for flag, expect_zero in ((True, False), (False, True)):
        moe = _moe(track_router_metrics=flag)
        moe.train()
        moe(torch.randn(128, 32))
        assert moe.router.track_metrics is flag
        assert (float(moe.router_entropy) == 0.0) is expect_zero
        # the counter must advance either way, or the consumer divides by zero
        assert int(moe.acc_fwd_times) == 1


def test_loss_mask_marks_valid_tokens_not_padding():
    """Polarity the docstring used to state backwards."""
    moe = _moe()
    moe.train()
    mask = torch.zeros(128, dtype=torch.bool)
    mask[:30] = True
    moe(torch.randn(128, 32), loss_mask=mask)
    assert int(moe.tokens_per_expert_E.sum()) == 30 * 2  # valid tokens x top_k
