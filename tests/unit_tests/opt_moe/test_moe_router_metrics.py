# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regressions for the MoE router metrics pipeline.

Producer: ``OptMoE`` and its router's buffers. Consumer:
``optimizers/container.py``'s step pre-hook.
"""

import types

import pytest

import torch
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config
from torchtitan.optimizers.moe_balance import (
    publish_moe_metrics,
    register_moe_load_balancing_hook,
)


def _moe_config(*, dim, hidden_dim, num_experts, top_k, **kw):
    """One OptMoE layer's config: gauge-d8-w256 built with a single MoE layer
    (no depth scaling, as one layer alone)."""
    return (
        build_model_config(
            "gauge-d8-w256",
            parallelism=ParallelismConfig(),
            n_layers=1,
            dim=dim,
            n_dense_layers=0,
            depth_init="identity",
            moe_hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            **kw,
        )
        .layers[0]
        .feed_forward
    )


def _moe(**kw):
    """An MoE whose expert compute is stubbed out.

    These tests are about the metrics buffers, which are all filled before the
    experts run. Keeping the real `routed_experts` would drag in the expert
    dispatcher, whose collective path needs an ambient DeviceMesh that a CPU
    unit test has no business creating.
    """
    moe = _moe_config(dim=32, hidden_dim=64, num_experts=8, top_k=2, **kw).build()
    moe.routed_experts = _StubExperts()
    return moe


class _StubExperts(nn.Module):
    def forward(self, x_TD, *args, **kwargs):
        return torch.zeros_like(x_TD)


def _run_bias_hook(
    gas: int,
    *,
    arm: bool = True,
    moe_kw: dict | None = None,
    batches: list | None = None,
    paddings: list | None = None,
    device: str = "cpu",
    steps: int = 1,
) -> list:
    """Drive the real step pre-hook over `gas` microbatches (or `batches`,
    with optional padding masks), `steps` times, and return the MoE layers it
    updated. On CUDA every step after the first runs the hook in the sync
    debug mode "error": a host sync there raises."""
    import torch.nn as nn

    class _Blk(nn.Module):
        def __init__(self, moe):
            super().__init__()
            self.feed_forward = moe

    class _Part(nn.Module):
        def __init__(self, moes):
            super().__init__()
            self.layers = nn.ModuleDict({str(i): _Blk(m) for i, m in enumerate(moes)})

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
    if moe_kw is None:
        moe_kw = dict(aux_loss=dict(loss_type="microbatch_wise", coeff=1e-3))
    moes = [_moe(**moe_kw).to(device) for _ in range(2)]
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
    if batches is None:
        batches = [torch.randn(64, 32, device=device) for _ in range(gas)]
    if paddings is None:
        paddings = [None] * len(batches)
    den = torch.tensor(float(sum(b.shape[0] for b in batches)), device=device)
    for step in range(steps):
        for b, pad in zip(batches, paddings):
            for m in moes:
                m(b, padding_mask_T=pad, aux_loss_denominator=den)
        if arm:
            opt.arm_moe_publish()
        if step and device != "cpu":
            torch.cuda.set_sync_debug_mode("error")
        try:
            opt.fire()
        finally:
            if device != "cpu":
                torch.cuda.set_sync_debug_mode("default")
    return moes


def test_published_load_balance_loss_is_gradient_accumulation_invariant():
    """`instance_acc` already carries 1/global_valid_tokens for the WHOLE step:
    `moe_load_balance_loss/L-*` must not scale with the microbatch count."""
    seen = {
        gas: _run_bias_hook(gas)[0]._log_expert_metrics["moe_load_balance_loss/L-0"]
        for gas in (1, 2, 4)
    }
    lo, hi = min(seen.values()), max(seen.values())
    assert hi / lo < 1.05, f"published metric should be gas-invariant, got {seen}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "moe_kw",
    [
        dict(aux_loss=dict(loss_type="microbatch_wise", coeff=1e-3)),
        dict(quantile_balance_bins=128),
    ],
)
def test_unarmed_step_hook_never_syncs_the_host(moe_kw):
    """An off-cadence step's hook only queues GPU work: a host sync there
    (e.g. a blocking host-to-device copy) holds the CPU until the backward
    ends and the next launches start late."""
    _run_bias_hook(1, arm=False, moe_kw=moe_kw, device="cuda", steps=2)


def test_unarmed_step_updates_the_bias_without_publishing():
    """Off-cadence steps do the bias update only: no D2H copy, no queue put."""
    moes = _run_bias_hook(1, arm=False)
    for moe in moes:
        assert not getattr(moe, "_log_expert_metrics", None), "published unasked"
    assert any(float(m.expert_bias_E.abs().sum()) > 0 for m in moes)


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
        True,
    )
    publish_moe_metrics(good)
    assert info[0]["module"]._log_expert_metrics["moe_entropy/L-0"] == 1.0

    publish_moe_metrics((info, [0.25] * (L * E)))  # too few fields: unpack raises
    for i in info:
        assert i["module"]._log_expert_metrics == {}, "stale metrics republished"


def test_track_router_metrics_gates_entropy():
    for flag, expect_zero in ((True, False), (False, True)):
        moe = _moe(track_router_metrics=flag)
        moe.train()
        moe(torch.randn(128, 32))
        assert (float(moe.router_entropy) == 0.0) is expect_zero


def test_padding_mask_marks_padding_not_valid_tokens():
    """Upstream's polarity: True = padding, not counted."""
    moe = _moe()
    moe.train()
    padding = torch.ones(128, dtype=torch.bool)
    padding[:30] = False
    moe(torch.randn(128, 32), padding_mask_T=padding)
    assert int(moe.router.tokens_per_expert_E.sum()) == 30 * 2  # valid tokens x top_k


def test_activation_recompute_does_not_count_a_forward_twice():
    """Activation remat replays code outside a saved region in backward; the
    MoE's forward-only statistics (expert counts, entropy) must
    count each forward once, as without remat."""
    import torch_remat as remat

    stats = []
    for recompute in (False, True):
        torch.manual_seed(0)
        moe = _moe()
        moe.train()
        # the default policy's regions, named as a block names them
        moe.configure_remat_regions(["*"], ["*routed_experts.w13.*"])
        x = torch.randn(64, 32, requires_grad=True)
        forward = (
            remat.checkpoint(region_name="layers.0", preserve_rng_state=False)(
                moe.forward
            )
            if recompute
            else moe.forward
        )
        forward(x).sum().backward()
        stats.append(
            (
                moe.router.tokens_per_expert_E.clone(),
                moe.router_entropy.clone(),
            )
        )
    for plain, replayed in zip(*stats):
        assert torch.equal(plain, replayed), (plain, replayed)


def _token_entropies(moe, x):
    """Each token's entropy of its renormalized top-k weights (eval: no stats)."""
    moe.eval()
    with torch.no_grad():
        w, _, _ = moe.router(x, moe.expert_bias_E)
    moe.train()
    w = w / moe.router.route_scale
    return -(w * w.clamp_min(1e-20).log()).sum(dim=-1)


def test_published_entropy_is_the_mean_over_counted_tokens():
    """The step's entropy: the mean over every counted (non-padding) token of all
    microbatches, whatever their count; padding tokens do not enter."""
    torch.manual_seed(1)
    batches = [torch.randn(64, 32) for _ in range(3)]
    paddings = [torch.arange(64) >= n for n in (64, 40, 10)]
    # No bias: the reference routes as the training forwards did.
    moes = _run_bias_hook(
        3,
        moe_kw=dict(load_balance_coeff=None),
        batches=[b.clone() for b in batches],
        paddings=paddings,
    )
    for i, moe in enumerate(moes):
        ref = torch.cat(
            [_token_entropies(moe, b)[~p] for b, p in zip(batches, paddings)]
        ).mean()
        got = moe._log_expert_metrics[f"moe_entropy/L-{i}"]
        assert abs(got - float(ref)) < 1e-5, (got, float(ref))


def test_published_expert_usage_is_the_share_of_counted_tokens():
    """moe_ep_usage: each expert's share of the step's counted token-expert
    pairs (sums to 1 per layer); the counters are reset after the step."""
    torch.manual_seed(2)
    batches = [torch.randn(64, 32) for _ in range(2)]
    paddings = [torch.arange(64) >= 50, None]
    moes = _run_bias_hook(2, moe_kw={}, batches=batches, paddings=paddings)
    for i, moe in enumerate(moes):
        usage = [
            moe._log_expert_metrics[f"moe_ep_usage/L-{i}_EP-{e}"] for e in range(8)
        ]
        assert abs(sum(usage) - 1.0) < 1e-6
        assert float(moe.router.tokens_per_expert_E.sum()) == 0.0
        assert float(moe.router_entropy) == 0.0


def test_metrics_are_published_without_an_expert_bias():
    """No bias (load_balance_coeff None): usage, entropy and maxvio still
    publish; there are no bias metrics; the counters are still reset."""
    moes = _run_bias_hook(2, moe_kw=dict(load_balance_coeff=None))
    for i, moe in enumerate(moes):
        assert moe.expert_bias_E is None
        m = moe._log_expert_metrics
        assert f"moe_entropy/L-{i}" in m and f"moe_maxvio_batch/L-{i}" in m
        assert f"moe_ep_usage/L-{i}_EP-0" in m
        assert not any(k.startswith(("moe_bias/", "moe_bias_")) for k in m)
        assert float(moe.router.tokens_per_expert_E.sum()) == 0.0


def test_metrics_are_published_with_the_quantile_balance():
    """The quantile-balanced router: the bias is solved from its histogram, the
    same statistics publish, and the histogram is reset."""
    moes = _run_bias_hook(2, moe_kw=dict(quantile_balance_bins=64))
    for i, moe in enumerate(moes):
        m = moe._log_expert_metrics
        assert f"moe_entropy/L-{i}" in m and f"moe_bias/L-{i}_EP-0" in m
        assert float(moe.expert_bias_E.abs().sum()) > 0
        hist = moe.router.quantile_balancer.required_bias_histogram_EB
        assert float(hist.sum()) == 0.0
