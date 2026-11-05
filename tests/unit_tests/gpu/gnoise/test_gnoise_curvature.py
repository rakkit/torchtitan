# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Curvature against its equations on real forward / backward passes under FSDP2.

For a step D = W1 - W0 on the step's batch:
    model level  c   = 2 (L(W1) - L(W0) - <G0, D>) / ||D||^2,
    per unit     c_k = <D_k, G1_k - G0_k> / ||D_k||^2,
with L the global sum of loss / N and G the reduced gradient. The reference is taken
from full tensors in fp64 through the harness's own forward (not the engine path that
gnoise's `_batch_loss` uses), and the call must leave buffers, RNG and gradients as
they were. On CPU (gloo, dense test attention) or 4 GPUs."""

import math
import os
import sys
import tempfile
import types

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gn_harness as H  # noqa: E402

from torchtitan.models.common.param_layout import by_rank  # noqa: E402
from torchtitan.optimizers.disco.layout import resolve_units  # noqa: E402

pytestmark = pytest.mark.multi_gpu

TOL = 1e-6


def _full(t):
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().double()


class _Microbatch:
    """The call `_batch_loss` makes on the trainer's microbatch objects."""

    def __init__(self, inp, lab):
        self.inp, self.lab = inp, lab

    def to_input_dict(self, device, non_blocking=False):
        return {
            "input": self.inp,
            "labels": self.lab,
            "positions": H.positions(self.inp.numel(), device),
        }


def _engine(model, pd, cfg, dev):
    """The engine attributes `_batch_loss` reads; the loss is the harness's CE (or its
    chunked loss over the grouped [norm, lm_head])."""

    def loss_fn(pred, labels, *, global_loss_token_counts):
        if model._gn_chunked is not None:
            return model._gn_chunked(
                pred, labels, global_loss_token_counts=global_loss_token_counts
            )
        logits = pred.to_local() if hasattr(pred, "to_local") else pred
        y = labels.to_local() if hasattr(labels, "to_local") else labels
        loss = (
            F.cross_entropy(logits.float(), y, reduction="sum", ignore_index=-100)
            / global_loss_token_counts
        )
        return loss, {}

    return types.SimpleNamespace(
        model_parts=[model],
        device=dev,
        parallelism_context=pd,
        config=cfg,
        max_num_documents=None,
        preprocess_inputs_kwargs={},
        loss_fn=loss_fn,
    )


def _loss_and_grads(model, pd, cfg, mbs, nd, covered):
    """Global loss and full reduced gradients of the covered params, via the harness."""
    model.zero_grad(set_to_none=True)
    local = torch.zeros((), dtype=torch.float64, device=mbs[0][0].device)
    for inp, lab in mbs:
        loss = H.forward_loss(model, pd, cfg, inp, lab, nd)
        with spmd.no_typecheck():
            loss.backward()
        local += (
            (loss.to_local() if hasattr(loss, "to_local") else loss).detach().double()
        )
    total = local.clone()
    dist.all_reduce(total)
    grads = {n: _full(p.grad) for n, p in model.named_parameters() if n in covered}
    model.zero_grad(set_to_none=True)
    return float(total), local, grads


class TestCurvature(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self):
        if self.device_type == "cpu":
            H.patch_dense_attention()
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _scenario(self, topo, *, moe=False, param_dtype="float32", chunks=0, accum=2):
        dev = self._device()
        knobs = dict(
            enable=True,
            research_freq=1,
            curvature_model_freq=1,
            curvature_matrix_freq=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                topo,
                dev,
                tmp,
                gnoise=knobs,
                chunks=chunks,
                param_dtype=param_dtype,
                model_cfg=H.gauge_config(moe=moe, num_experts=8) if moe else None,
            )
            gn = cont.gnoise
            gn.validate_layout(accum=accum)
            mbs = []
            for k in range(accum):
                n = H.SEQ * (1 + self.rank % 3)
                mbs.append(H.batch(100 * k + self.rank, n, dev))
            local_n = torch.tensor(
                float(sum(int((lab != -100).sum()) for _, lab in mbs)), device=dev
            )
            dist.all_reduce(local_n)
            n_den = int(local_n)
            nd = torch.tensor(n_den, dtype=torch.int64, device=dev)
            by_name = {pp.name: pp for pp in gn.params}
            W0 = {n: _full(p) for n, p in model.named_parameters() if n in by_name}

            L0, L0_local, G0 = _loss_and_grads(model, pd, cfg, mbs, nd, by_name)
            H.run_step(
                model, pd, cfg, cont, 1, mbs, n_den
            )  # arms the step; saves (W0, G0)

            gen = torch.Generator().manual_seed(4321)
            D = {}
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n not in by_name:
                        continue
                    d = (
                        0.02
                        * (W0[n].square().mean().sqrt() + 1e-2)
                        * torch.randn(
                            tuple(p.shape), generator=gen, dtype=torch.float64
                        ).to(W0[n].device)
                    ).float()
                    D[n] = d.double()
                    if isinstance(p, DTensor):
                        p.add_(
                            distribute_tensor(d.to(dev), p.device_mesh, p.placements)
                        )
                    else:
                        p.add_(d.to(dev))
                    D[n] = (
                        _full(p) - W0[n]
                    )  # the step as stored (fp32 rounding of W0 + d)

            bufs = [(b, b.detach().clone()) for b in model.buffers()]
            rng = torch.get_rng_state()
            assert gn.curvature_due
            gn.metrics = type(gn.metrics)()
            gn.curvature(
                _engine(model, pd, cfg, dev),
                [[_Microbatch(i, l) for i, l in mbs]],
                # the step's counts as the harness's forward_loss uses them
                global_loss_token_counts=nd,
                global_routing_token_counts=nd.reshape(1),
                local_loss=L0_local,
            )
            m = dict(gn.metrics)

            errors = []
            for b, v in bufs:
                if not torch.equal(b, v):
                    errors.append(f"buffer of shape {tuple(b.shape)} changed")
            if not torch.equal(torch.get_rng_state(), rng):
                errors.append("CPU RNG state changed")
            left = [n for n, p in model.named_parameters() if p.grad is not None]
            if left:
                errors.append(f"gradients left set: {left[:3]}")

            L1, _, G1 = _loss_and_grads(model, pd, cfg, mbs, nd, by_name)
            gd = sum(float((G0[n] * D[n]).sum()) for n in D)
            dd = sum(float(D[n].square().sum()) for n in D)
            want = 2.0 * (L1 - L0 - gd) / dd
            got = float(m["gnoise_curv/model"])
            scale = 2.0 * (abs(L1) + abs(L0) + abs(gd)) / dd * 1e-6  # fp32 loss sums
            if not abs(got - want) <= scale:
                errors.append(
                    f"curv/model: got {got:.6e} want {want:.6e} (scale {scale:.2e})"
                )
            got_n = float(m["gnoise_curv/step_norm"])
            if not abs(got_n - math.sqrt(dd)) <= TOL * math.sqrt(dd):
                errors.append(
                    f"curv/step_norm: got {got_n:.6e} want {math.sqrt(dd):.6e}"
                )

            layouts = model.param_layouts()
            n_units = 0
            for name, p in model.named_parameters():
                if name not in by_name or by_name[name].expert:
                    continue
                if p.ndim <= 1:
                    units, view = [("", torch.arange(p.numel()))], (p.numel(), 1)
                else:
                    view = (math.prod(p.shape[:-1]), p.shape[-1])
                    lay = layouts.get(name) or by_rank(p.shape)
                    units = [
                        (
                            s.name,
                            torch.cat([torch.arange(a, a + c) for a, c in s.rows.runs]),
                        )
                        for s in resolve_units(name, p.shape, lay)
                    ]
                d2, h2 = D[name].reshape(view), (G1[name] - G0[name]).reshape(view)
                base = name.removesuffix(".weight")
                for uname, rows in units:
                    key = (
                        f"gnoise_curv/{base}.{uname}"
                        if uname
                        else f"gnoise_curv/{base}"
                    )
                    du, hu = d2[rows], h2[rows]
                    den = float(du.square().sum())
                    want = float((du * hu).sum()) / den
                    bound = (
                        float(du.norm() * hu.norm()) / den
                    )  # Cauchy-Schwarz: |c_k| <= this
                    got = float(m[key])
                    n_units += 1
                    if not abs(got - want) <= TOL * bound + 1e-12:
                        errors.append(
                            f"{key}: got {got:.6e} want {want:.6e} (bound {bound:.2e})"
                        )
            assert n_units > 10, "no per-unit curvature checked"
        return errors

    def _assert(self, errors):
        assert not errors, f"{len(errors)} mismatches:\n" + "\n".join(errors[:25])

    @with_comms
    def test_curvature_fsdp4(self):
        self._assert(self._scenario(dict(dp_replicate=1, dp_shard=4)))

    @with_comms
    def test_curvature_ddp4(self):
        """Replicated params: every rank holds every element, so the sums divide by 4."""
        self._assert(self._scenario(dict(dp_replicate=4, dp_shard=1)))

    @with_comms
    def test_curvature_hsdp22_bf16_chunked(self):
        self._assert(
            self._scenario(
                dict(dp_replicate=2, dp_shard=2), param_dtype="bfloat16", chunks=4
            )
        )

    @with_comms
    def test_curvature_moe_fsdp4(self):
        """The forward adds to tokens_per_expert_E; the call must put the counts back."""
        self._assert(
            self._scenario(dict(dp_replicate=1, dp_shard=4), moe=True, accum=1)
        )
