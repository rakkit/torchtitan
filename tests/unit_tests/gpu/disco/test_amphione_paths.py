# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""amphione (the production rule) on 4 ranks under FSDP and DDP, dense and MoE
(experts, router with frozen row lengths), against its compute run unit by unit
on whole tensors in one process: the transport of the weights, the per-unit
state, the masks and the returned weights.
Runs on CPU/gloo when fewer than 4 accelerators are visible."""

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from torchtitan.optimizers.disco.optimizer import _unit_type  # noqa: E402
from torchtitan.optimizers.disco.presets import AmphioneMap  # noqa: E402
from torchtitan.optimizers.disco.rules import StepCtx  # noqa: E402

pytestmark = pytest.mark.multi_gpu

FSDP = dict(dp_replicate=1, dp_shard=4, cp=1, ep=1)
DDP = dict(dp_replicate=4, dp_shard=1, cp=1, ep=1)
HSDP = dict(dp_replicate=2, dp_shard=2, cp=1, ep=1)
FSDP_EP2 = dict(dp_replicate=1, dp_shard=4, cp=1, ep=2)
STEPS, LR, TOL = 3, 0.02, 1e-5
BETA = 0.95  # the group's momentum (the recipe's, Nesterov on): amphione keeps the averages
# (topology, MoE, amphione args); no MoE under pure DDP (not a supported setup)
CASES = {
    "fsdp4_dense": (FSDP, False, {}),
    "ddp4_dense": (DDP, False, {}),
    "fsdp4_dense_scale_nesterov": (FSDP, False, {"scale_nesterov": True}),
    "fsdp4_dense_column_scales": (FSDP, False, {"column_scales": True}),
    "ddp4_dense_column_scales": (DDP, False, {"column_scales": True}),
    "fsdp4_moe": (FSDP, True, {}),
    "fsdp4_moe_frozen_router": (
        FSDP,
        True,
        {"frozen_scale_units": ["moe.router.gate"]},
    ),
    # the expert rule state on an EP mesh (local experts only) and HSDP steps
    "fsdp4_ep2_moe": (FSDP_EP2, True, {}),
    "hsdp22_moe": (HSDP, True, {}),
}


class TestAmphionePaths(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, topo, moe, args):
        device = self._device()
        pd = harness.parallelism_context(world_size=4, device_type=device.type, **topo)
        model_cfg = harness.tiny_model_config(moe=moe)
        model_cfg["attention"]["n_kv_heads"] = 2  # MHA: one state shape per wqkv
        cfg = harness.trainer_config(
            model_cfg,
            compute="svd",
            communication_dtype="float32",
            rule="amphione",
            rule_args={"pre": None, "scale": None, **args},  # amphione has neither
            lr=LR,
            momentum=BETA,
            nesterov=True,
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        names = {id(p): n for n, p in harness.trainable(model)}
        entries = [
            e
            for opt in container.optimizers
            for e in opt.entries
            if isinstance(e.rule.compute, AmphioneMap)
        ]
        self.assertTrue(entries, "no parameter runs amphione")
        if moe:
            self.assertTrue(any("router" in names[id(e.param)] for e in entries))
            self.assertTrue(any("experts" in names[id(e.param)] for e in entries))
        frozen = set(args.get("frozen_scale_units", ()))
        if frozen:
            self.assertTrue(
                any(_unit_type(e, u) in frozen for e in entries for u in e.units)
            )
        ref = {
            names[id(e.param)]: harness.full_tensor(e.param).float().clone()
            for e in entries
        }
        state = {
            (names[id(e.param)], k): torch.zeros(*e.rule.compute.state_shape(*u.shape))
            for e in entries
            for k, u in enumerate(e.units)
        }
        failures = []
        for step in range(1, STEPS + 1):
            grads = harness.set_synthetic_grads(model, step, device)
            container.step()
            container.zero_grad()
            for e in entries:
                cm = e.rule.compute
                name = names[id(e.param)]
                W2 = ref[name].reshape(-1, e.view[1])
                G2 = grads[name].float().cpu().reshape(-1, e.view[1])
                out = W2.clone()
                for k, u in enumerate(e.units):
                    rows = [(a, a + n) for a, n in u.rows.runs]
                    Wu = torch.cat([W2[a:b] for a, b in rows])
                    Gu = torch.cat([G2[a:b] for a, b in rows])
                    t = _unit_type(e, u)  # the masks from their definitions
                    ctx = StepCtx(
                        LR,
                        0.0,
                        step,
                        momentum=BETA,
                        nesterov=True,
                        colmask=torch.tensor([t in cm.col_units])
                        if cm.column_scales
                        else None,
                        frozen=torch.tensor([t in frozen]) if frozen else None,
                    )
                    res = cm.run(Gu[None], Wu[None], state[name, k][None], ctx)
                    state[name, k] = res.state[0]
                    off = 0
                    for a, b in rows:
                        out[a:b] = res.weight[0][off : off + b - a]
                        off += b - a
                ref[name] = out.reshape(ref[name].shape)
            for e in entries:
                name = names[id(e.param)]
                got, want = harness.full_tensor(e.param).float().cpu(), ref[name]
                d = (got - want).abs().max().item()
                if d > TOL * max(1.0, want.abs().max().item()):
                    failures.append(f"step {step} {name}: max |diff| {d:.3e}")
        ok = harness.all_ranks_agree(not failures, device)
        self.assertTrue(ok, "; ".join(failures[:5]))

    def _build(self, topo):
        """The MoE recipe's amphione (GQA: wqkv's q unit larger than k / v)."""
        device = self._device()
        pd = harness.parallelism_context(world_size=4, device_type=device.type, **topo)
        model_cfg = harness.tiny_model_config(moe=True)
        model_cfg["attention"]["n_kv_heads"] = 1
        cfg = harness.trainer_config(
            model_cfg,
            compute="svd",
            communication_dtype="float32",
            rule="amphione",
            rule_args={
                "pre": None,
                "scale": None,
                "frozen_scale_units": ["moe.router.gate"],
            },
            lr=LR,
            momentum=BETA,
            nesterov=True,
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            return harness.build_model_and_optimizer(cfg, pd, device, tmp)

    @with_comms
    def test_state_lives_on_the_owner_and_resumes_on_another_layout(self):
        """Each non-expert unit's state exists on exactly one rank (its owner),
        one tensor per unit. A run saved after step 2 under FSDP 4 and resumed
        under HSDP 2 x 2 (other owners, other expert shards) takes steps 3-4
        as the uninterrupted run does."""
        import torch.distributed.checkpoint as dcp

        model, container = self._build(FSDP)
        opt = container.optimizers[0]
        mine = sorted(
            (e.clean, k)
            for e in opt.entries
            if e.rule.has_state
            and isinstance(opt.state[e.param].get("rule_state"), dict)
            for k in opt.state[e.param]["rule_state"]
        )
        every = [None] * 4
        dist.all_gather_object(every, mine)
        units = sorted(
            (e.clean, str(u.pos))
            for e in opt.entries
            if e.rule.has_state and e.family.value != "expert"
            for u in e.units
        )
        flat = sorted(x for r in every for x in r)
        self.assertEqual(flat, units)  # each unit exactly once over the ranks
        for step in (1, 2):
            harness.set_synthetic_grads(model, step, self._device())
            container.step()
            container.zero_grad()
        path = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(path)
        dcp.save(
            {"model": model.state_dict(), "optim": container.state_dict()},
            checkpoint_id=path[0],
        )
        for step in (3, 4):
            harness.set_synthetic_grads(model, step, self._device())
            container.step()
            container.zero_grad()
        want = {
            n: harness.full_tensor(p).float().cpu() for n, p in harness.trainable(model)
        }
        model2, container2 = self._build(dict(dp_replicate=2, dp_shard=2, cp=1, ep=1))
        sd = {"model": model2.state_dict(), "optim": container2.state_dict()}
        dcp.load(sd, checkpoint_id=path[0])
        model2.load_state_dict(sd["model"])
        container2.load_state_dict(sd["optim"])
        for step in (3, 4):
            harness.set_synthetic_grads(model2, step, self._device())
            container2.step()
            container2.zero_grad()
        failures = []
        for n, p in harness.trainable(model2):
            got = harness.full_tensor(p).float().cpu()
            d = (got - want[n]).abs().max().item()
            if d > TOL * max(1.0, want[n].abs().max().item()):
                failures.append(f"{n}: max |diff| {d:.3e}")
        ok = harness.all_ranks_agree(not failures, self._device())
        self.assertTrue(ok, "; ".join(failures[:5]))

    @with_comms
    def test_the_weights_travel_once_and_the_owner_copy_is_checked(self):
        """Under FSDP the owners keep the weights of the units they step: the
        weight all-to-all runs at the first step only; a logging step that
        gathers the weights anyway finds the copies bitwise equal, and fails on
        every rank once the parameters changed outside the optimizer."""
        model, container = self._build(FSDP)
        opt = container.optimizers[0]
        calls = []
        real = opt._ex_w.to_owner
        opt._ex_w.to_owner = lambda *a, **k: calls.append(1) or real(*a, **k)
        for step in (1, 2, 3):
            harness.set_synthetic_grads(model, step, self._device())
            container.step()
            container.zero_grad()
        self.assertEqual(len(calls), 1)
        harness.set_synthetic_grads(model, 4, self._device())
        container.arm_norm_computation()  # a logging step: every weight travels
        container.step()
        container.zero_grad()
        container.collect_metrics(4)
        with torch.no_grad():
            for n, p in harness.trainable(model):
                if "attention.wo" in n or "qkv" in n:
                    getattr(p, "_local_tensor", p).add_(1e-3)
        harness.set_synthetic_grads(model, 5, self._device())
        container.arm_norm_computation()
        with self.assertRaisesRegex(RuntimeError, "changed outside the optimizer"):
            container.step()

    def _drift_on_one_rank(self, topo):
        """The copies are checked every `_COPY_CHECK_EVERY` steps without any
        norm logging; a parameter changed on one rank only (a shard of one
        replica) makes every rank raise at the next check, not hang."""
        model, container = self._build(topo)
        container._COPY_CHECK_EVERY = 3
        for step in (1, 2):
            harness.set_synthetic_grads(model, step, self._device())
            container.step()
            container.zero_grad()
        if self.rank == 0:
            with torch.no_grad():
                for n, p in harness.trainable(model):
                    if "attention.wo" in n:
                        getattr(p, "_local_tensor", p).add_(1e-3)
        harness.set_synthetic_grads(model, 3, self._device())
        with self.assertRaisesRegex(RuntimeError, "changed outside the optimizer"):
            container.step()

    @with_comms
    def test_a_change_on_one_rank_fails_every_rank_fsdp(self):
        self._drift_on_one_rank(FSDP)

    @with_comms
    def test_a_change_on_one_rank_fails_every_rank_hsdp(self):
        self._drift_on_one_rank(HSDP)


def _case(name):
    @with_comms
    def test(self):
        self._check(*CASES[name])

    return test


for _name in CASES:  # one test per case: each builds its meshes in fresh processes
    setattr(TestAmphionePaths, f"test_{_name}", _case(_name))
