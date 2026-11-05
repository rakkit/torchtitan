# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every DiSCO step path against a single-process reference, on 4 ranks.

Two arms per topology:

* `comm=float32`: the all-to-all / all-gather transports are lossless, so the
  only thing that can differ from a reference that calls the SAME `lmo` on the
  whole matrix is the transport itself (wrong rows, wrong owner, wrong offset).
  Compared to rounding (1e-7), row-block-wise, so a misrouted run cannot average away.
* `comm=bfloat16` (production): compared to a reference built from the
  transcribed formulas, under an a-priori bound: bf16 rounds once on the way
  out and once on the way back, so ||W - W_ref||_F / ||W_ref||_F is bounded by
  k * 2^-8 * sqrt(steps) with k = 4 leaving margin for the NS conditioning.

Runs on CPU/gloo when fewer than 4 accelerators are visible (login node) and
on CUDA/NCCL otherwise.
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from reference import ReferenceDiSCO, rules_from_container  # noqa: E402

from torchtitan.optimizers.disco.presets import Blocks  # noqa: E402
from torchtitan.optimizers.disco.rules import lmo  # noqa: E402

pytestmark = pytest.mark.multi_gpu

TOPOLOGIES = {
    "fsdp4": dict(dp_replicate=1, dp_shard=4, cp=1, ep=1),
    "ddp4": dict(dp_replicate=4, dp_shard=1, cp=1, ep=1),
    "hsdp22": dict(dp_replicate=2, dp_shard=2, cp=1, ep=1),
    "fsdp4_ep2": dict(dp_replicate=1, dp_shard=4, cp=1, ep=2),
    "cp2_fsdp2": dict(dp_replicate=1, dp_shard=2, cp=2, ep=1),
}
STEPS = 3
BF16_K = 4.0
FP32_TOL = 1e-7  # batched-vs-single svd rounding on GPU; misrouting is 1e-2


def _same_lmo(g, r):
    """The optimizer's own LMO on a whole matrix (transport arm)."""
    if g.numel() == 1:
        return torch.sign(g)
    rule = r.get("rule_obj")
    if rule is not None and (rule.kind == "gram" or rule.pre or rule.rest_tail):
        return rule.reference(g).update  # a rule with its own phases, whole
    return lmo(g, r["compute"], r["scale"], r["steps"])


class _SameLMOReference(ReferenceDiSCO):
    def matrix_lmo(self, g, r):
        return _same_lmo(g, r)


def _reload_and_check_aliasing(container) -> list[str]:
    """Save with DCP the way the trainer does (async_save, whose staging
    deep-copies every tensor), wipe the live moments, load them back: the
    expert moments must return as views of the block buffer the batched rule
    call reads, and the next steps must still match the reference."""
    path = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(path)
    pg = dist.new_group(
        backend="gloo"
    )  # async_save stages on the CPU, as the trainer's
    ret = dcp.async_save(
        container.state_dict(), checkpoint_id=path[0], process_group=pg
    )
    getattr(ret, "upload_completion", ret).result()
    for opt in container.optimizers:
        for st in opt.state.values():
            if "momentum_buffer" in st:  # only where DiSCO averages
                st["momentum_buffer"].zero_()
    sd = container.state_dict()
    dcp.load(sd, checkpoint_id=path[0])
    container.load_state_dict(sd)
    failures = []
    for opt in container.optimizers:
        for bi, block in enumerate(opt._blocks):
            big = opt._big.get(bi)
            for j, e in enumerate(block):
                if "momentum_buffer" not in opt.state[e.param]:
                    continue  # the block averages nothing
                buf = opt.state[e.param]["momentum_buffer"]
                local = buf.to_local() if hasattr(buf, "to_local") else buf
                n = local.shape[0]
                if n and (big is None or local.data_ptr() != big[j * n].data_ptr()):
                    failures.append(f"{e.name}: moment not a view of block {bi}")
    return failures


def _restart(model, container, cfg, pd, device):
    """Save model and optimizer with DCP, build a fresh pair, load into it:
    what a resumed job sees, unlike reloading into the live optimizer."""
    path = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(path)
    pg = dist.new_group(backend="gloo")  # async_save stages on the CPU
    ret = dcp.async_save(
        {"model": model.state_dict(), "optim": container.state_dict()},
        checkpoint_id=path[0],
        process_group=pg,
    )
    getattr(ret, "upload_completion", ret).result()
    with tempfile.TemporaryDirectory() as tmp:
        model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
    sd = {"model": model.state_dict(), "optim": container.state_dict()}
    dcp.load(sd, checkpoint_id=path[0])
    model.load_state_dict(sd["model"])
    container.load_state_dict(sd["optim"])
    return model, container


class TestDiSCOStepPaths(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check_topology(
        self,
        topo: dict,
        *,
        comm_dtype: torch.dtype,
        moe: bool = True,
        weight_decay=0.0,
        reload_after: int = 0,
        n_layers: int = 2,
        extra_groups: dict | None = None,
        one_per_owner: bool = False,
        num_experts: int = 4,
        restart_after: int = 0,
        on_build=None,
        model: dict | None = None,
    ):
        device = self._device()
        pd = harness.parallel_dims(
            world_size=self.world_size, device_type=device.type, **topo
        )
        # The transport arm needs an LMO whose result does not depend on how
        # DiSCO batches same-shape matrices: NS5 in bf16 differs by ~2.5% rel
        # between a batched and a per-matrix call, the SVD polar factor is
        # bitwise batch-independent.
        backend = "svd" if comm_dtype == torch.float32 else "newtonschulz5"
        cfg = harness.trainer_config(
            harness.tiny_model_config(
                moe=moe, n_layers=n_layers, num_experts=num_experts, **(model or {})
            ),
            compute=backend,
            communication_dtype=str(comm_dtype).removeprefix("torch."),
            weight_decay=weight_decay,
            extra_groups=extra_groups,
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        if on_build is not None:
            on_build(container)
        if one_per_owner:  # the step exchange lands in place on those owners
            ex = container.optimizers[0]._ex_step
            single = len(ex.owned) == 1
            self.assertTrue(not single or ex.in_place, f"rank {self.rank}")
            any_single = torch.tensor(float(single), device=device)
            dist.all_reduce(any_single, op=dist.ReduceOp.MAX)
            self.assertEqual(any_single.item(), 1.0, "no owner holds a single unit")

        params = {
            n: harness.full_tensor(p).float() for n, p in harness.trainable(model)
        }
        rules = rules_from_container(container, model)
        ref_cls = _SameLMOReference if comm_dtype == torch.float32 else ReferenceDiSCO
        ref = ref_cls(params, rules)

        worst = 0.0
        failures = []
        for step in range(1, STEPS + 1):
            grads = harness.set_synthetic_grads(model, step, device)
            container.step()
            container.zero_grad()
            ref.step(grads)
            if step == reload_after:
                failures += _reload_and_check_aliasing(container)
            if step == restart_after:
                model, container = _restart(model, container, cfg, pd, device)
            for name, p in harness.trainable(model):
                got = harness.full_tensor(p).float()
                want = ref.params[name]
                if comm_dtype == torch.float32:
                    # Same-shape units run as one batched svd on their owner; on
                    # GPU cuSOLVER's batched driver differs from the single one at
                    # ~1e-9. A misrouted row is off by 1e-2, so the arm keeps its
                    # point under a rounding-level tolerance.
                    d = (got - want).abs()
                    if d.max().item() > FP32_TOL * max(1.0, want.abs().max().item()):
                        bad = (
                            (d > FP32_TOL * max(1.0, want.abs().max().item()))
                            .reshape(got.shape[0], -1)
                            .any(dim=1)
                            .nonzero()
                            .flatten()
                        )
                        failures.append(
                            f"step {step} {name}: {bad.numel()}/{got.shape[0]} rows differ "
                            f"(first {bad[:8].tolist()}), max |diff| {(got - want).abs().max().item():.3e}"
                        )
                else:
                    rel = ((got - want).norm() / want.norm().clamp_min(1e-30)).item()
                    worst = max(worst, rel)
                    bound = BF16_K * 2.0**-8 * step**0.5
                    if rel > bound:
                        failures.append(
                            f"step {step} {name}: rel {rel:.3e} > bound {bound:.3e}"
                        )
        ok = harness.all_ranks_agree(not failures, device)
        if self.rank == 0 and failures:
            print("\n".join(failures[:20]))
        self.assertTrue(
            ok, f"{len(failures)} mismatches on rank {self.rank}: {failures[:3]}"
        )
        if self.rank == 0 and comm_dtype == torch.bfloat16:
            print(f"[{topo}] worst rel err over {STEPS} steps: {worst:.3e}")

    # ---- fp32 transport arm: the same LMO, equal to rounding ----
    @with_comms
    def test_fsdp4_fp32(self):
        self._check_topology(TOPOLOGIES["fsdp4"], comm_dtype=torch.float32)

    @with_comms
    def test_fsdp4_headwise_fp32(self):
        """Head-wise Muon: q, k, v blocked by head rows, wo by head columns."""
        head_dim = harness.tiny_model_config(moe=True).layer.attention.head_dim
        groups = {
            "qkv": {"match": r"\.wqkv\.", "args": {"block_rows": head_dim}},
            "wo": {"match": r"attention\.wo\.", "args": {"block_cols": head_dim}},
        }

        def blocked(container):
            names = {
                e.clean.rsplit(".", 1)[-1]
                for e in container.optimizers[0].entries
                if isinstance(e.rule.compute, Blocks)
            }
            self.assertEqual(names, {"wqkv", "wo"})

        self._check_topology(
            TOPOLOGIES["fsdp4"],
            comm_dtype=torch.float32,
            extra_groups=groups,
            on_build=blocked,
        )

    # ---- the attention's short conv (kind "conv", [C, K] units) and sinks ----
    @with_comms
    def test_fsdp4_conv_sinks_fp32(self):
        """Conv units under the default rule: gathered to their owners."""
        self._check_topology(
            TOPOLOGIES["fsdp4"],
            comm_dtype=torch.float32,
            model={"conv": 4, "sinks": True},
            on_build=self._assert_conv_family("sharded"),
        )

    @with_comms
    def test_fsdp4_conv_row_rule_fp32(self):
        """A row rule on the conv's channels: no gather."""
        self._check_topology(
            TOPOLOGIES["fsdp4"],
            comm_dtype=torch.float32,
            model={"conv": 4, "sinks": True},
            extra_groups={
                "conv": {"kind": "conv", "args": {"compute": "row-l2", "scale": "one"}}
            },
            on_build=self._assert_conv_family("local"),
        )

    @with_comms
    def test_hsdp22_conv_sinks_fp32(self):
        self._check_topology(
            TOPOLOGIES["hsdp22"],
            comm_dtype=torch.float32,
            model={"conv": 4, "sinks": True},
        )

    @with_comms
    def test_ddp4_dense_conv_sinks_fp32(self):
        self._check_topology(
            TOPOLOGIES["ddp4"],
            comm_dtype=torch.float32,
            moe=False,
            model={"conv": 4, "sinks": True},
        )

    def _assert_conv_family(self, family: str):
        def check(container):
            fams = {
                e.family.value
                for e in container.optimizers[0].entries
                if e.kind == "conv"
            }
            self.assertEqual(fams, {family})

        return check

    @with_comms
    def test_fsdp4_one_matrix_per_owner_fp32(self):
        """Owners holding a single travelling matrix: the step exchange lands
        in place and the updates come back through `hold` -- reached in
        production only at FSDP >= ~256."""
        ffn_rows = {"match": r"feed_forward\.", "args": {"compute": "row-l2"}}
        self._check_topology(
            TOPOLOGIES["fsdp4"],
            comm_dtype=torch.float32,
            moe=False,
            n_layers=1,
            extra_groups={"ffn": ffn_rows},
            one_per_owner=True,
        )

    @with_comms
    def test_fsdp4_fewer_experts_than_ranks_fp32(self):
        """dp_shard > num_experts: FSDP's Shard(0) leaves two ranks with no
        expert rows; they must build and step."""
        self._check_topology(
            TOPOLOGIES["fsdp4"], comm_dtype=torch.float32, num_experts=2
        )

    @with_comms
    def test_fsdp4_fewer_experts_than_ranks_resumes_fp32(self):
        """A restart with uneven expert shards (ranks with none) continues exactly."""
        self._check_topology(
            TOPOLOGIES["fsdp4"],
            comm_dtype=torch.float32,
            num_experts=2,
            restart_after=1,
        )

    @with_comms
    def test_fsdp4_fewer_experts_than_ranks_logs_real_experts_only(self):
        """The ranks holding no expert must not register phantom `ep_2`, `ep_3`."""
        device = self._device()
        pd = harness.parallel_dims(
            world_size=self.world_size, device_type=device.type, **TOPOLOGIES["fsdp4"]
        )
        cfg = harness.trainer_config(
            harness.tiny_model_config(moe=True, num_experts=2),
            log_norm_freq=1,
            **TOPOLOGIES["fsdp4"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        container.log_metrics_locally = True  # each rank reports only its own
        harness.set_synthetic_grads(model, 1, device)
        container.begin_step(1, should_log=True, is_final_step=False)
        container.step()
        mine = sorted(
            int(k.split("/ep_")[1].split("/")[0])
            for k in container.collect_metrics(1)
            if "/ep_" in k
        )
        seen: list = [None] * self.world_size
        dist.all_gather_object(seen, mine)
        found = sorted(set().union(*map(set, seen)))
        self.assertEqual(found, [0, 1], f"expert keys per rank: {seen}")

    @with_comms
    def test_ddp4_fp32(self):
        self._check_topology(TOPOLOGIES["ddp4"], comm_dtype=torch.float32)

    @with_comms
    def test_ddp4_dense_fp32(self):
        self._check_topology(TOPOLOGIES["ddp4"], comm_dtype=torch.float32, moe=False)

    @with_comms
    def test_hsdp22_fp32(self):
        self._check_topology(TOPOLOGIES["hsdp22"], comm_dtype=torch.float32)

    @with_comms
    def test_fsdp4_ep2_fp32(self):
        self._check_topology(TOPOLOGIES["fsdp4_ep2"], comm_dtype=torch.float32)

    @with_comms
    def test_cp2_fsdp2_fp32(self):
        self._check_topology(TOPOLOGIES["cp2_fsdp2"], comm_dtype=torch.float32)

    @with_comms
    def test_fsdp4_fp32_weight_decay(self):
        self._check_topology(
            TOPOLOGIES["fsdp4"], comm_dtype=torch.float32, weight_decay=0.1
        )

    @with_comms
    def test_fsdp4_ep2_fp32_reload(self):
        """A state-dict round trip mid-run: the experts' moments stay the
        block buffers, and the next steps still match the reference."""
        self._check_topology(
            TOPOLOGIES["fsdp4_ep2"], comm_dtype=torch.float32, reload_after=1
        )

    @with_comms
    def test_fsdp4_dense_fp32(self):
        self._check_topology(TOPOLOGIES["fsdp4"], comm_dtype=torch.float32, moe=False)

    # ---- bf16 production arm: formula reference under the a-priori bound ----
    @with_comms
    def test_fsdp4_bf16(self):
        self._check_topology(TOPOLOGIES["fsdp4"], comm_dtype=torch.bfloat16)

    @with_comms
    def test_ddp4_dense_bf16(self):
        self._check_topology(TOPOLOGIES["ddp4"], comm_dtype=torch.bfloat16, moe=False)

    @with_comms
    def test_fsdp4_ep2_bf16(self):
        self._check_topology(TOPOLOGIES["fsdp4_ep2"], comm_dtype=torch.bfloat16)

    # ---- needs decide the transport: a row-norm rule never enters the a2a ----
    @with_comms
    def test_rownorm_rule_needs_no_all_to_all(self):
        device = self._device()
        topo = TOPOLOGIES["fsdp4"]
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        cfg = harness.trainer_config(
            harness.tiny_model_config(moe=True),
            compute="row-l2",
            scale="inv_sqrt_n",
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        opt = container.optimizers[0]
        assert opt._ex_step is None, "row-separable rules must not use the per-step a2a"
        assert (
            opt._ex_all is not None
        ), "their units still reach the owner on logging steps"
        harness.set_synthetic_grads(model, 1, device)
        container.step()  # no collective on the update path: must not hang or raise

    # ---- sphere rules: weights on the owner, coefficients back ----
    @with_comms
    def test_fsdp4_hyperball_fp32(self):
        self._check_sphere(TOPOLOGIES["fsdp4"], "hyperball", {"scale": None})

    @with_comms
    def test_fsdp4_hyperball_exact_tangent_fp32(self):
        args = {"scale": None, "exact_angle": True, "tangent": True}
        self._check_sphere(TOPOLOGIES["fsdp4"], "hyperball", args)

    @with_comms
    def test_fsdp4_ep2_hyperball_exact_fp32(self):
        args = {"scale": None, "exact_angle": True}
        self._check_sphere(TOPOLOGIES["fsdp4_ep2"], "hyperball", args)

    @with_comms
    def test_sphere_rules_refuse_weight_decay(self):
        device = self._device()
        topo = TOPOLOGIES["fsdp4"]
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        for rule, args in (("hyperball", {"scale": None}),):
            cfg = harness.trainer_config(
                harness.tiny_model_config(moe=False),
                rule=rule,
                rule_args=args,
                weight_decay=0.1,
                **topo,
            )
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "weight_decay: 0"):
                    harness.build_model_and_optimizer(cfg, pd, device, tmp)

    def _check_sphere(
        self,
        topo,
        rule,
        rule_args,
        *,
        moe=True,
        weight_decay=0.0,
        restart_after=0,
    ):
        device = self._device()
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        cfg = harness.trainer_config(
            harness.tiny_model_config(moe=moe),
            compute="svd",
            communication_dtype="float32",
            rule=rule,
            rule_args=rule_args,
            lr=0.02,
            weight_decay=weight_decay,
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        params = {
            n: harness.full_tensor(p).float() for n, p in harness.trainable(model)
        }
        ref = _SameLMOReference(params, rules_from_container(container, model))
        spheres = [n for n, r in ref.rules.items() if r["sphere"] is not None]
        self.assertTrue(spheres, "no parameter runs the sphere rule")
        failures = []
        for step in range(1, STEPS + 1):
            grads = harness.set_synthetic_grads(model, step, device)
            container.step()
            container.zero_grad()
            ref.step(grads)
            if step == restart_after:
                model, container = _restart(model, container, cfg, pd, device)
            for name, p in harness.trainable(model):
                got, want = harness.full_tensor(p).float(), ref.params[name]
                d = (got - want).abs().max().item()
                if d > 1e-5 * max(1.0, want.abs().max().item()):
                    failures.append(f"step {step} {name}: max |diff| {d:.3e}")
        ok = harness.all_ranks_agree(not failures, device)
        if failures and self.rank == 0:
            print("\n".join(failures[:40]))
        self.assertTrue(ok, failures[:3])

    # ---- a reduced-Gram rule on the tall units: one all-reduce, no a2a ----
    @with_comms
    def test_fsdp4_gram_polar_on_embeddings(self):
        device = self._device()
        topo = TOPOLOGIES["fsdp4"]
        pd = harness.parallel_dims(world_size=4, device_type=device.type, **topo)
        groups = {
            "embedding": {
                "match": r"tok_embeddings\.weight",
                "rule": "muon",
                "args": {
                    "steps": 8,
                    "compute": "gram_polar+row-l2",
                    "scale": "sqrt_n",
                },
            },
            "lm_head": {
                "match": r"(output|lm_head)\.weight",
                "rule": "muon",
                "args": {
                    "steps": 8,
                    "compute": "gram_polar+row-l2",
                    "scale": "inv_sqrt_n",
                },
            },
        }
        cfg = harness.trainer_config(
            harness.tiny_model_config(moe=False),
            compute="svd",
            communication_dtype="float32",
            extra_groups=groups,
            **topo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
        opt = container.optimizers[0]
        heads = [
            e for e in opt.entries if e.family.value == "local" and e.param.ndim == 2
        ]
        assert len(heads) == 2 and all(e.rule.kind == "gram" for e in heads)
        params = {
            n: harness.full_tensor(p).float() for n, p in harness.trainable(model)
        }
        ref = _SameLMOReference(params, rules_from_container(container, model))
        GRAM_UNITS = {
            n
            for n, _ in harness.trainable(model)
            if n.endswith(("tok_embeddings.weight", "output.weight", "lm_head.weight"))
        }
        assert len(GRAM_UNITS) == 2
        failures = []
        for step in range(1, STEPS + 1):
            grads = harness.set_synthetic_grads(model, step, device)
            container.step()
            container.zero_grad()
            ref.step(grads)
            for name, p in harness.trainable(model):
                got, want = harness.full_tensor(p).float(), ref.params[name]
                d = (got - want).abs().max().item()
                # the Gram rule iterates in fp16 (GramPolar's work dtype): DiSCO's
                # all-reduced partial Grams and the reference's one Gram round
                # apart by ~1e-3 of the step (measured max 8.9e-4 at lr 0.1); a
                # transport fault moves the whole step (~0.1)
                tol = 3e-3 if name in GRAM_UNITS else 1e-4
                if d > tol * max(1.0, want.abs().max().item()):
                    failures.append(f"step {step} {name}: max |diff| {d:.3e}")
        ok = harness.all_ranks_agree(not failures, device)
        if failures and self.rank == 0:
            print("\n".join(failures[:12]))
        self.assertTrue(ok, failures[:3])
