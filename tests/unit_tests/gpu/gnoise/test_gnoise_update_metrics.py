# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gnoise update metrics against their equations, on real FSDP2 steps (gloo, 4 ranks):

    Psi_t       = <O_t, G_t> / (|O_t| |G_t|)                  every E1 step
    lag_h       = <O_t, O_{t-h}> / (|O_t| |O_{t-h}|), h = 1, 2, 4, 8   E1 steps divisible by
                  the smallest multiple of the E1 interval >= 8
    psi_half / psi_quarter: Psi with G_S = sum of the small batches of group a, groups
                  {b : b mod g = a}, g = 2 / 4, b the rank-major small-batch index; mean over a
    ntop_k      = top-k eigenvalue share of the per-token noise covariance (sqrt(n_b) d_b,
                  d_b = (N_den / n_b) g_b - (N_den / N) sum g)
    bs1         = ||C_row^(1/2)||_S1^2 / ||Ghat||_S1^2, C_row = sum_b n_b d_b d_b^T / (K - 1)

O_t is taken here from the weights, W_t - W_{t+1} (the default group: muon, no decay),
G_t from the reduced gradient, the small-batch gradients from FSDP's own reduce with the
other ranks' labels masked (bf16 as research mode's exchange). Also: no param keeps
more than four update copies. (On = off for training: the GPU runs, as T1.)"""

import math
import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gn_harness as H  # noqa: E402

from torchtitan.models.common.param_layout import by_rank  # noqa: E402
from torchtitan.optimizers.disco.layout import resolve_units  # noqa: E402

pytestmark = pytest.mark.multi_gpu
LAGS = (1, 2, 4, 8)


def _full(t):
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().double()


def _cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


def _hidden_units(model):
    """{name: [(key, global rows)]} of the layout-kind matrix params, rows of the [R, C] view."""
    out, layouts = {}, model.param_layouts()
    for name, p in model.named_parameters():
        lay = layouts.get(name) or by_rank(p.shape)
        if lay.kind != "matrix":
            continue
        base = name.removesuffix(".weight")
        out[name] = [
            (
                f"{base}.{s.name}" if s.name else base,
                torch.cat([torch.arange(a, a + c) for a, c in s.rows.runs]),
            )
            for s in resolve_units(name, p.shape, lay)
        ]
    return out


class TestUpdateMetrics(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self):
        if self.device_type == "cpu":
            H.patch_dense_attention()
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _scenario(
        self,
        topo,
        *,
        accum=2,
        steps=16,
        research_at=8,
        e1=None,
        param_dtype="float32",
        tol=1e-6,
    ):
        dev = self._device()
        knobs = dict(enable=True, research_freq=research_at, **(e1 or {}))
        tok = [H.SEQ * (1 + r % 3) for r in range(4)]
        n_den = accum * sum(tok)
        K = 4 * accum
        errors, most = [], 0
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                topo, dev, tmp, gnoise=knobs, param_dtype=param_dtype
            )
            gn = cont.gnoise
            gn.validate_layout(accum=accum)
            units = _hidden_units(model)
            params = dict(model.named_parameters())
            armed = gn.due  # E1 steps: the gnoise schedule itself is tested elsewhere
            O_hist = {}
            for t in range(1, steps + 1):
                mbs = [
                    H.batch(1000 * t + 100 * k + self.rank, tok[self.rank], dev)
                    for k in range(accum)
                ]
                solo = None
                if t == research_at:
                    solo, _ = H.solo_grads(model, pd, cfg, mbs, n_den, 4, self.rank)
                W0 = {n: _full(params[n]) for n in units}
                H.run_step(model, pd, cfg, cont, t, mbs, n_den)
                G = {n: _full(params[n].grad) for n in units}
                cont.step()
                m = dict(gn.metrics)
                gn.metrics = type(
                    gn.metrics
                )()  # what collect() does on the trainer's logging step
                cont.zero_grad()
                most = max([most] + [len(d) for d in gn.dirs.values()])
                O = {n: W0[n] - _full(params[n]) for n in units}
                O_hist[t] = O
                is_e1 = armed(t, dense=t == 1)
                lag_step = (
                    is_e1 and t % 8 == 0
                )  # 8: the smallest multiple of 1 and of 4 >= 8
                for n, us in units.items():
                    C = params[n].shape[-1]
                    o2, g2 = O[n].reshape(-1, C), G[n].reshape(-1, C)
                    for key, rows in us:
                        k_psi = f"gnoise_psi/{key}"
                        if not is_e1:
                            if k_psi in m:
                                errors.append(f"step {t}: {k_psi} on a step without E1")
                            continue
                        want = _cos(o2[rows], g2[rows])
                        if not abs(float(m[k_psi]) - want) <= tol:
                            errors.append(
                                f"step {t} {k_psi}: got {float(m[k_psi]):.9f} want {want:.9f}"
                            )
                        for h in LAGS:
                            k_lag = f"gnoise_lag{h}/{key}"
                            if not lag_step:
                                if k_lag in m:
                                    errors.append(f"step {t}: {k_lag} off its steps")
                                continue
                            got = float(m[k_lag])
                            if t - h < 1:
                                if not math.isnan(got):
                                    errors.append(
                                        f"step {t} {k_lag}: {got}, want NaN (no step {t - h})"
                                    )
                                continue
                            want = _cos(o2[rows], O_hist[t - h][n].reshape(-1, C)[rows])
                            if not abs(got - want) <= tol:
                                errors.append(
                                    f"step {t} {k_lag}: got {got:.9f} want {want:.9f}"
                                )
                if solo is not None:
                    self._research(
                        errors, m, solo, O, units, params, tok, n_den, K, accum, tol
                    )
        if most > 4:
            errors.append(f"{most} stored update copies for one param (at most 4)")
        return errors

    def _research(self, errors, m, solo, O, units, params, tok, n_den, K, accum, tol):
        keys = sorted(solo)  # (rank, microbatch), rank-major
        nb = torch.tensor([float(tok[r]) for r, _ in keys], dtype=torch.float64)
        N = float(nb.sum())
        for name, us in units.items():
            C = params[name].shape[-1]
            o2 = O[name].reshape(-1, C)
            g = [solo[b][name].to(torch.bfloat16).double().reshape(-1, C) for b in keys]
            for key, rows in us:
                gu, Ou = [x[rows] for x in g], o2[rows]
                for frac, label in ((2, "half"), (4, "quarter")):
                    want = (
                        sum(
                            _cos(Ou, sum(gu[b] for b in range(K) if b % frac == a))
                            for a in range(frac)
                        )
                        / frac
                    )
                    got = float(m[f"gnoise_psi_{label}/{key}"])
                    if not abs(got - want) <= 1e-5:
                        errors.append(
                            f"psi_{label}/{key}: got {got:.8f} want {want:.8f}"
                        )
                Gh = (n_den / N) * sum(gu)
                d = [(n_den / float(nb[b])) * gu[b] - Gh for b in range(K)]
                Dm = torch.stack(
                    [float(nb[b]) ** 0.5 * d[b].reshape(-1) for b in range(K)]
                )
                sv2 = torch.linalg.svdvals(
                    Dm
                ).square()  # the covariance's nonzero eigenvalues
                for k in (1, 2, 3):
                    name_k = f"gnoise_ntop{k}/{key}"
                    if k > K - 1:
                        if name_k in m:
                            errors.append(f"{name_k} logged at K = {K}")
                        continue
                    want = float(sv2[:k].sum() / sv2.sum())
                    if not abs(float(m[name_k]) - want) <= 1e-9:
                        errors.append(
                            f"{name_k}: got {float(m[name_k]):.10f} want {want:.10f}"
                        )
                if K < 8:
                    for x in ("bs1", "bs1_noise", "bs1_grad"):
                        if f"gnoise_{x}/{key}" in m:
                            errors.append(f"gnoise_{x}/{key} logged at K = {K} < 8")
                    continue
                Crow = sum(float(nb[b]) * d[b] @ d[b].T for b in range(K)) / (K - 1)
                lam, V = torch.linalg.eigh(Crow)
                root = (V * lam.clamp_min(0).sqrt()) @ V.T  # C_row^(1/2)
                noise = float(torch.linalg.svdvals(root).sum()) ** 2
                grad = float(torch.linalg.svdvals(Gh).sum()) ** 2
                for x, want in (
                    ("bs1", noise / grad),
                    ("bs1_noise", noise),
                    ("bs1_grad", grad),
                ):
                    got = float(m[f"gnoise_{x}/{key}"])
                    if not abs(got - want) <= 1e-7 * abs(want):
                        errors.append(f"{x}/{key}: got {got:.10e} want {want:.10e}")

    def _assert(self, errors):
        ok = torch.tensor([float(not errors)])
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        assert not errors, f"{len(errors)} mismatches:\n" + "\n".join(errors[:25])
        assert float(ok) == 1.0, "another rank failed"

    @with_comms
    def test_fsdp4_e1_freq4_no_dense_k8(self):
        """E1 on steps 4, 8, 12, 16; the lag copies of steps 7, 6, 4, 15, 14, 12 are taken
        on steps without E1; research (K = 8, B_S1 on) at step 8."""
        self._assert(
            self._scenario(
                dict(dp_replicate=1, dp_shard=4),
                e1=dict(e1_freq=4, e1_dense_phase=False),
            )
        )

    @with_comms
    def test_ddp4_every_step_k4(self):
        """E1 every step (interval 1, lag steps 8 and 16); K = 4: no B_S1, top-k up to 3."""
        self._assert(self._scenario(dict(dp_replicate=4, dp_shard=1), accum=1))

    @with_comms
    def test_hsdp22_bf16_k8(self):
        self._assert(
            self._scenario(
                dict(dp_replicate=2, dp_shard=2),
                param_dtype="bfloat16",
                e1=dict(e1_freq=4, e1_dense_phase=False),
            )
        )
