# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A new optimizer is one `Compute` and one preset; nothing in the optimizer
or the transport changes.

The rule below is registered by this test alone. It declares everything a
second-order optimizer would -- the unit's weight on the computing rank,
per-unit state kept between steps, and the NEW WEIGHT as its output -- and
computes the plain Muon step written as `W_next`. So the distributed optimizer
must reproduce the single-rank Muon reference on every family: sharded (the
weight rides the all-to-all to the owner, the step rides back in the update's
slot), expert (whole on the rank) and replicated (owner plus all-gather).
"""

import os
import sys
import tempfile
from dataclasses import dataclass

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from reference import rules_from_container  # noqa: E402
from test_step_paths import _SameLMOReference, STEPS  # noqa: E402

from torchtitan.optimizers.disco.presets import plain  # noqa: E402
from torchtitan.optimizers.disco.rules import (  # noqa: E402
    Compute,
    register,
    Result,
    Rule,
)

pytestmark = pytest.mark.multi_gpu


@dataclass(frozen=True)
class MuonAsWeight(Compute):
    """`W_next = (1 - lr wd) W - lr U`, with a step counter as the state."""

    inner: Rule
    needs_weight = True
    has_state = True
    output = "weight"

    def run(self, x, w, state, ctx):
        u = self.inner.run_whole(x).update.float()
        W = w.float() * (1.0 - ctx.wd * ctx.lr) - ctx.lr * u
        return Result(weight=W, state=state + 1)  # zeros on the first step


@register("muon_as_weight")
def _muon_as_weight(pre="identity", compute="svd", steps=5, scale="spectral"):
    return Rule((), MuonAsWeight(plain(pre, compute, steps, scale)))


ROW_GROUPS = {  # embeddings and head keep their row rules
    "embedding": {
        "match": r"tok_embeddings\.weight",
        "rule": "sgd",
        "args": {"compute": "row-l2", "scale": "sqrt_n"},
    },
    "lm_head": {
        "match": r"(output|lm_head)\.weight",
        "rule": "sgd",
        "args": {"compute": "row-l2", "scale": "inv_sqrt_n"},
    },
}


class TestRuleExtension(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, topo: dict, *, moe: bool):
        device = self._device()
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        cfg = harness.trainer_config(
            harness.tiny_model_config(moe=moe),
            compute="svd",
            communication_dtype="float32",
            weight_decay=0.1,
            extra_groups=ROW_GROUPS,
            **topo,
        )
        cfg.optimizer.groups["default"]["rule"] = "muon_as_weight"
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        opt = container.optimizers[0]
        custom = [e for e in opt.entries if isinstance(e.rule.compute, MuonAsWeight)]
        assert custom, "the registered rule reached no parameter"
        params = {
            n: harness.full_tensor(p).float() for n, p in harness.trainable(model)
        }
        ref = _SameLMOReference(params, rules_from_container(container, model))
        failures = []
        for step in range(1, STEPS + 1):
            grads = harness.set_synthetic_grads(model, step, device)
            container.step()
            container.zero_grad()
            ref.step(grads)
            for name, p in harness.trainable(model):
                got, want = harness.full_tensor(p).float(), ref.params[name]
                d = (got - want).abs().max().item()
                if d > 1e-5 * max(1.0, want.abs().max().item()):
                    failures.append(f"step {step} {name}: max |diff| {d:.3e}")
        # every unit this rank computed kept its own state across the steps
        states = [  # the units this rank computed moved off zero
            float(v)
            for st in opt.state.values()
            if "rule_state" in st
            for t in (
                st["rule_state"].values()  # per unit, on its owner only
                if isinstance(st["rule_state"], dict)
                else [getattr(st["rule_state"], "_local_tensor", st["rule_state"])]
            )
            for v in t.flatten()  # experts: one slot per unit, [E, 2] for w13
            if float(v) != 0.0
        ]
        if not states or any(s != STEPS for s in states):
            failures.append(f"state not carried per unit: {sorted(map(float, states))}")
        ok = harness.all_ranks_agree(not failures, device)
        if failures and self.rank == 0:
            print("\n".join(failures[:20]))
        self.assertTrue(ok, failures[:3])

    @with_comms
    def test_fsdp4_sharded_and_experts(self):
        self._check(dict(dp_shard=4), moe=True)

    @with_comms
    def test_ddp4_replicated(self):
        self._check(dict(dp_replicate=4), moe=False)
