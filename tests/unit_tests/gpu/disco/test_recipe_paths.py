# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The production recipe on 4 ranks under FSDP against its single-rank meaning.

Groups: `opt_moe_mis_8b()`'s (amphione on every hidden matrix with the
router gate's row lengths frozen, sgd sign on the RMSNorm gains, row-l2 on the
embedding and the head), plus the conv group (amphione, row-l2 direction) and
the sinks group (sgd sign). Model: a tiny MoE with learnable-gain RMSNorm,
Canon conv and sinks. Reference: each group's momentum from its definition,
then every unit's rule on whole tensors in one process (`Rule.reference`, the
meaning every distributed path must reproduce), then the apply.
Runs on CPU/gloo when fewer than 4 accelerators are visible."""

import copy
import os
import sys
import tempfile

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from torchtitan.optimizers.disco.optimizer import _unit_type  # noqa: E402
from torchtitan.optimizers.disco.rules import StepCtx  # noqa: E402

from torchtitan_recipes.models.opt_moe import opt_moe_mis_8b  # noqa: E402

pytestmark = pytest.mark.multi_gpu

FSDP = dict(dp_replicate=1, dp_shard=4, cp=1, ep=1)
STEPS, TOL = 3, 1e-5
CONV_AND_SINKS = {
    "conv": {  # as reproduce_cfgs/configs/0.5.4-debug/w1-c3-*: amphione, row-l2 direction
        "match": r"(^|\.)attention\.(q|k|v)_conv\.weight$",
        "rule": "amphione",
        "lr": 0.02,
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.0,
        "args": {
            "compute": "row-l2",
            "polar_express_safety": None,
            "frozen_scale_units": [],
        },
    },
    "sinks": {
        "match": r"(^|\.)attention\.sinks$",
        "rule": "sgd",
        "lr": 0.02,
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.0,
        "args": {"pre": "identity", "compute": "sign", "scale": "one"},
    },
}


def _momentum(buf, g, b, nesterov):
    """A group's momentum by its definition: `momentum` b is the memory weight,
    buf <- b buf + (1 - b) g, Nesterov b buf + (1 - b) g; b = 0 is the raw
    gradient."""
    if b == 0.0:
        return buf, g
    buf = b * buf + (1.0 - b) * g
    return buf, (b * buf + (1.0 - b) * g if nesterov else buf)


class TestRecipePaths(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, n_kv_heads: int, *, log_steps=(), gains_scale=None):
        """`log_steps`: the steps armed for the norms (the owners' weight copies
        fill on step 1 from the logging gather, as in production). Every other
        step must gather nothing whole (the split gains step on their slice).
        `gains_scale`: the gains group's scale (the split slices must scale by
        the whole row's size)."""
        device = self._device()
        pd = harness.parallelism_context(world_size=4, device_type=device.type, **FSDP)
        model_cfg = harness.tiny_model_config(
            moe=True, conv=4, sinks=True, num_experts=8
        )
        model_cfg["norm_type"] = "rmsnorm"  # learnable gains
        model_cfg["n_kv_heads"] = n_kv_heads
        cfg = harness.trainer_config(model_cfg, **FSDP)
        groups = copy.deepcopy(opt_moe_mis_8b().optim.optimizer.groups)
        if device.type == "cpu":  # gram_polar_express runs quack's CUDA kernels only
            args = groups["default"]["args"]
            args["compute"] = "svd"
            args.pop("polar_express_safety")
        if gains_scale is not None:
            groups["gains"]["args"]["scale"] = gains_scale
        cfg.optim.optimizer.groups = {**copy.deepcopy(CONV_AND_SINKS), **groups}
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        names = {id(p): n for n, p in harness.trainable(model)}
        opts = container.optimizers
        entries = [(opt, e) for opt in opts for e in opt.entries]
        rules = {type(e.rule.compute).__name__ for _, e in entries}
        self.assertTrue({"AmphioneMap", "Sign", "RowL2"} <= rules, rules)
        ref = {
            names[id(e.param)]: harness.full_tensor(e.param).float().cpu().clone()
            for _, e in entries
        }
        buf = {n: torch.zeros_like(w) for n, w in ref.items()}
        state = {}
        failures = []
        from torch.distributed.tensor import DTensor

        gathers = []
        real_full = DTensor.full_tensor

        def counting(self_, *a, **k):
            gathers.append(1)
            return real_full(self_, *a, **k)

        for step in range(1, STEPS + 1):
            grads = harness.set_synthetic_grads(model, step, device)
            if step in log_steps:
                container.arm_norm_computation()
            gathers.clear()
            DTensor.full_tensor = counting
            try:
                container.step()
            finally:
                DTensor.full_tensor = real_full
            if step in log_steps:
                container.collect_metrics(step)
            elif gathers:
                failures.append(f"step {step}: {len(gathers)} full_tensor calls")
            container.zero_grad()
            for opt, e in entries:
                g = opt.param_groups[e.group_idx]
                name, cm = names[id(e.param)], e.rule.compute
                lr, wd = float(g["lr"]), float(g["weight_decay"])
                beta, nes = float(g["momentum"]), bool(g["nesterov"])
                if cm.owns_momentum:  # amphione: the raw gradient, its own averages
                    eff = grads[name].float().cpu()
                else:
                    buf[name], eff = _momentum(
                        buf[name], grads[name].float().cpu(), beta, nes
                    )
                W2 = ref[name].reshape(-1, e.view[1])
                G2 = eff.reshape(-1, e.view[1])
                out = W2.clone()
                for k, u in enumerate(e.units):
                    rows = [(a, a + n) for a, n in u.rows.runs]
                    Wu = torch.cat([W2[a:b] for a, b in rows])
                    Gu = torch.cat([G2[a:b] for a, b in rows])
                    t = _unit_type(e, u)  # the masks from their definitions
                    ctx = StepCtx(
                        lr,
                        wd,
                        step,
                        momentum=beta,
                        nesterov=nes,
                        frozen=torch.tensor([t in getattr(cm, "frozen_units", ())])
                        if getattr(cm, "frozen_units", ())
                        else None,
                        colmask=torch.tensor([t in cm.col_units])
                        if getattr(cm, "col_units", ())
                        else None,
                    )
                    if e.rule.kind == "unit":  # whole-unit computes take a stack
                        st = state.get((name, k))
                        if st is None and e.rule.has_state:
                            st = torch.zeros(1, *cm.state_shape(*u.shape))
                        res = e.rule.reference(
                            Gu[None], Wu[None] if e.rule.needs_weight else None, ctx, st
                        )
                        if e.rule.has_state:
                            state[name, k] = res.state
                        new = (
                            res.weight[0]
                            if e.rule.output == "weight"
                            else ctx.decay * Wu - lr * res.update[0]
                        )
                    else:
                        new = ctx.decay * Wu - lr * e.rule.reference(Gu).update
                    off = 0
                    for a, b in rows:
                        out[a:b] = new[off : off + b - a]
                        off += b - a
                ref[name] = out.reshape(ref[name].shape)
            for _, e in entries:
                name = names[id(e.param)]
                got, want = harness.full_tensor(e.param).float().cpu(), ref[name]
                d = (got - want).abs().max().item()
                if d > TOL * max(1.0, want.abs().max().item()):
                    failures.append(f"step {step} {name}: max |diff| {d:.3e}")
        ok = harness.all_ranks_agree(not failures, device)
        self.assertTrue(ok, "; ".join(failures[:5]))

    @with_comms
    def test_fsdp4_moe_mha(self):
        self._check(n_kv_heads=2)

    @with_comms
    def test_fsdp4_moe_gqa(
        self,
    ):  # GQA: wqkv's q unit larger than k / v (per-unit state)
        self._check(n_kv_heads=1)

    @with_comms
    def test_fsdp4_moe_gqa_logging_steps(self):
        """Production's step 1 logs: the owner copies fill from the logging
        gather; step 3 logs again and checks them."""
        self._check(n_kv_heads=1, log_steps=(1, 3))

    @with_comms
    def test_fsdp4_moe_gains_scaled_by_the_whole_row(self):
        self._check(n_kv_heads=2, gains_scale="sqrt_n")
