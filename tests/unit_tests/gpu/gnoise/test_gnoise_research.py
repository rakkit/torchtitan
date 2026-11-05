# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gnoise research mode against the equations: at K = 4 every pair (s = 1) and every
pairing (s = 2) is used, so each statistic is a fixed function of the four small-batch
gradients, recomputed here with exact SVD polars from FSDP's own reduce."""

import itertools
import math
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
import gn_harness as H  # noqa: E402
import gn_reference as ref  # noqa: E402

from torchtitan.models.common.param_layout import by_rank  # noqa: E402
from torchtitan.optimizers.disco.layout import resolve_units  # noqa: E402

pytestmark = pytest.mark.multi_gpu
TOPK = [1, 4, 16]


def polar(X):
    U, S, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    return U @ Vh


def close(errors, name, got, want, tol):
    if not (math.isfinite(got) and abs(got - want) <= tol):
        errors.append(f"{name}: got {got} want {want}")


def agree(errors, m, sizes, s, key, want, tol):
    """agree_s<s> as `close`, or absent when `research_sizes` leaves s out."""
    name = f"gnoise_agree_s{s}/{key}"
    if s in sizes:
        close(errors, f"agree_s{s}/{key}", float(m[name]), want, tol)
    elif name in m:
        errors.append(f"{name} logged with research_sizes {sorted(sizes)}")


class TestResearch(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self):
        if self.device_type == "cpu":
            H.patch_dense_attention()
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _check(self, topo, sizes=None):
        dev = self._device()
        knobs = dict(enable=True, research_freq=1, research_topk=TOPK)
        if sizes:
            knobs["research_sizes"] = sizes
        sizes = set(sizes or (1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(topo, dev, tmp, gnoise=knobs)
            gn = cont.gnoise
            gn.validate_layout(accum=1)
            n_tok = H.SEQ * (1 + self.rank % 3)
            mbs = [H.batch(self.rank, n_tok, dev)]
            nl = [float(H.SEQ * (1 + r % 3)) for r in range(4)]
            n_den = int(sum(nl))
            W = H.full_weights(model)
            solo, _ = H.solo_grads(model, pd, cfg, mbs, n_den, 4, self.rank)
            m = H.run_step(model, pd, cfg, cont, 1, mbs, n_den)
        keys = sorted(solo)
        errors = []
        layouts = model.param_layouts()
        for name, p in model.named_parameters():
            lay = layouts.get(name) or by_rank(p.shape)
            base = name.removesuffix(".weight")
            g = [
                solo[b][name].to(torch.bfloat16).double() for b in keys
            ]  # the exchange is bf16
            if p.ndim <= 1:  # sign rule: agreement of the signs of disjoint group sums
                for s, groups in ((1, [[0], [1], [2], [3]]), (2, None)):
                    pairs = (
                        list(itertools.combinations(range(4), 2)) if s == 1 else None
                    )
                    if s == 1:
                        want = sum(
                            float((g[a].sign() * g[b].sign()).mean()) for a, b in pairs
                        ) / len(pairs)
                    else:
                        parts = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
                        want = (
                            sum(
                                float(
                                    ((g[a] + g[b]).sign() * (g[c] + g[d]).sign()).mean()
                                )
                                for (a, b), (c, d) in parts
                            )
                            / 3
                        )
                    agree(errors, m, sizes, s, base, want, 1e-9)
                continue
            view = (math.prod(p.shape[:-1]), p.shape[-1])
            Wv = W[name].reshape(view)
            gv = [x.reshape(view) for x in g]
            for s_ in resolve_units(name, p.shape, lay):
                rows = torch.cat([torch.arange(a, a + c) for a, c in s_.rows.runs])
                key = f"{base}.{s_.name}" if s_.name else base
                gu, Wu = [x[rows] for x in gv], Wv[rows]
                if lay.kind in ("embedding", "unembedding", "router"):

                    def rowcos(A, B):
                        na, nb = A.norm(dim=1), B.norm(dim=1)
                        c = (A * B).sum(1) / (na * nb).clamp_min(1e-300)
                        return float(torch.where((na > 0) & (nb > 0), c, 0.0).mean())

                    want1 = (
                        sum(
                            rowcos(gu[a], gu[b])
                            for a, b in itertools.combinations(range(4), 2)
                        )
                        / 6
                    )
                    parts = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
                    want2 = (
                        sum(
                            rowcos(gu[a] + gu[b], gu[c] + gu[d])
                            for (a, b), (c, d) in parts
                        )
                        / 3
                    )
                    agree(errors, m, sizes, 1, key, want1, 1e-6)
                    agree(errors, m, sizes, 2, key, want2, 1e-6)
                    continue
                mm, nn = Wu.shape
                # total, exact jackknife
                tr, sg = ref.estimate(gu, nl, n_den)
                se_tr, se_s = ref.jackknife(gu, nl, n_den)
                A = (
                    sum(
                        float(((n_den / nb) * x).square().sum())
                        for x, nb in zip(gu, nl)
                    )
                    / 4
                )
                den = sum(1 / x for x in nl) / 4 - 1 / sum(nl)
                close(
                    errors,
                    f"rs_tr/{key}",
                    float(m[f"gnoise_rs_tr/{key}"]),
                    tr,
                    1e-4 * A / den,
                )
                close(
                    errors,
                    f"rs_se_tr/{key}",
                    float(m[f"gnoise_rs_se_tr/{key}"]),
                    se_tr,
                    1e-4 * A / den,
                )
                close(
                    errors,
                    f"rs_se_sig/{key}",
                    float(m[f"gnoise_rs_se_sig/{key}"]),
                    se_s,
                    1e-4 * A,
                )
                # spectral radial: P g = <g, u1 v1^T> u1 v1^T with W's exact top pair
                U, S, Vh = torch.linalg.svd(Wu, full_matrices=False)
                u1, v1 = U[:, 0], Vh[0]
                proj = [
                    (u1 @ x @ v1) * torch.ones(1, dtype=torch.float64, device=x.device)
                    for x in gu
                ]
                tr, sg = ref.estimate(proj, nl, n_den)
                close(
                    errors,
                    f"srad_tr/{key}",
                    float(m[f"gnoise_srad_tr/{key}"]),
                    tr,
                    1e-4 * A / den,
                )
                close(
                    errors,
                    f"srad_sig/{key}",
                    float(m[f"gnoise_srad_sig/{key}"]),
                    sg,
                    1e-4 * A,
                )
                # top-k share on the smaller side
                tr_tot, _ = ref.estimate(gu, nl, n_den)
                for k in TOPK:
                    if k > min(mm, nn):
                        continue
                    if mm <= nn:
                        Uk = U[:, :k]
                        proj = [Uk @ (Uk.T @ x) for x in gu]
                    else:
                        Vk = Vh[:k].T
                        proj = [(x @ Vk) @ Vk.T for x in gu]
                    tr_k, _ = ref.estimate(proj, nl, n_den)
                    close(
                        errors,
                        f"topk{k}/{key}",
                        float(m[f"gnoise_topk{k}/{key}"]),
                        tr_k / tr_tot,
                        1e-4 * A / den / abs(tr_tot) + 1e-9,
                    )
                # LMO agreement: all 6 pairs at s = 1, all 3 pairings at s = 2
                Os = [polar(x) for x in gu]
                want1 = (
                    sum(
                        float((Os[a] * Os[b]).sum())
                        for a, b in itertools.combinations(range(4), 2)
                    )
                    / 6
                    / min(mm, nn)
                )
                parts = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
                want2 = (
                    sum(
                        float((polar(gu[a] + gu[b]) * polar(gu[c] + gu[d])).sum())
                        for (a, b), (c, d) in parts
                    )
                    / 3
                    / min(mm, nn)
                )
                agree(errors, m, sizes, 1, key, want1, 1e-5)
                agree(errors, m, sizes, 2, key, want2, 1e-5)
        assert not errors, f"{len(errors)} mismatches:\n" + "\n".join(errors[:25])

    @with_comms
    def test_research_fsdp4(self):
        self._check(dict(dp_replicate=1, dp_shard=4))

    @with_comms
    def test_research_sizes_subset_fsdp4(self):
        """research_sizes [2]: the half-batch agreement as in the full run, no s = 1."""
        self._check(dict(dp_replicate=1, dp_shard=4), sizes=[2])

    @with_comms
    def test_research_ddp4(self):
        self._check(dict(dp_replicate=4, dp_shard=1))

    @with_comms
    def test_research_hsdp22(self):
        self._check(dict(dp_replicate=2, dp_shard=2))


class TestResearchExperts(DTensorTestBase):
    """T27 (EP 1 and EP 2): the owner's K_e small-batch gradients of an expert equal FSDP's own
    per-rank expert gradients, with n_b,e = the routed counts."""

    @property
    def world_size(self) -> int:
        return 4

    def _experts(self, topo):
        """EP 1: small batch = rank; EP > 1: = EP group (round-robin routing, so the
        all-to-all splits are equal as gloo needs; groups get unequal token counts)."""
        if self.device_type == "cpu":
            H.patch_dense_attention()
            dev = torch.device("cpu")
        else:
            dev = torch.device(self.device_type, self.rank)
        ep = topo.get("ep", 1)
        sel = (0, 1) if ep == 1 else (0, 2)  # at EP 2 one expert on each EP shard
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                topo,
                dev,
                tmp,
                model_cfg=H.gauge_config(moe=True, num_experts=4),
                force_balance=ep > 1,
                gnoise=dict(
                    enable=True,
                    research_freq=1,
                    research_topk=[1],
                    research_matrices=r"routed_experts\..*\.e[%d%d]\b" % sel,
                ),
            )
            gn = cont.gnoise
            gn.validate_layout(accum=1)
            if ep > 1:
                mine = tuple(
                    dist.get_process_group_ranks(pd.get_mesh("ep").get_group())
                )
                allg = [None] * 4
                dist.all_gather_object(allg, mine)
                groups = sorted(set(allg))
                gi = next(i for i, G in enumerate(groups) if self.rank in G)
                mbs = [H.batch(self.rank, (H.SEQ // 2) * (1 + gi), dev)]
            else:
                groups = [(r,) for r in range(4)]
                mbs = [H.batch(self.rank, 2 * H.SEQ, dev)]
            nloc = torch.tensor(float(mbs[0][1].numel()))
            dist.all_reduce(nloc)
            n_den = int(nloc)
            solo, counts = H.solo_grads(model, pd, cfg, mbs, n_den, 4, self.rank)
            m = H.run_step(model, pd, cfg, cont, 1, mbs, n_den)
            names = {n for n, _ in model.named_parameters() if ".routed_experts." in n}
            gsolo = (
                H.group_grads(model, pd, cfg, mbs, n_den, groups, self.rank, names)
                if ep > 1
                else None
            )
        keys = [(G, 0) for G in groups]
        errors = []
        moes = sorted(
            n for n, mod in model.named_modules() if type(mod).__name__ == "OptMoE"
        )
        off = {k: 4 * i for i, k in enumerate(moes)}
        K = len(groups)
        for name, p in model.named_parameters():
            if ".routed_experts." not in name:
                continue
            moe = name.split(".routed_experts.")[0]
            for e in sel:
                n_e = [
                    sum(float(counts[(r, k)][r][off[moe] + e]) for r in G)
                    for (G, k) in keys
                ]
                src = [
                    gsolo[b] if gsolo is not None else solo[(b[0][0], b[1])]
                    for b in keys
                ]
                subs = ("w1", "w3") if name.endswith("w13.weight") else ("",)
                for si, sub in enumerate(subs):
                    g = [
                        (x[name][e][si] if sub else x[name][e])
                        .to(torch.bfloat16)
                        .double()
                        for x in src
                    ]
                    key = f"{name.removesuffix('.weight')}.e{e}" + (
                        f".{sub}" if sub else ""
                    )
                    tr, _ = ref.estimate(g, n_e, n_den)
                    keep = [i for i, c in enumerate(n_e) if c > 0]
                    if len(keep) < 2:
                        if math.isfinite(float(m[f"gnoise_rs_tr/{key}"])):
                            errors.append(
                                f"{key}: {len(keep)} small batches with tokens, logged a value"
                            )
                        continue
                    A = sum(
                        float(((n_den / n_e[i]) * g[i]).square().sum()) for i in keep
                    ) / len(keep)
                    den = sum(1 / n_e[i] for i in keep) / len(keep) - 1 / sum(
                        n_e[i] for i in keep
                    )
                    close(
                        errors,
                        f"rs_tr/{key}",
                        float(m[f"gnoise_rs_tr/{key}"]),
                        tr,
                        1e-4 * A / den,
                    )
                    if len(keep) == K:
                        Os = [polar(x) for x in g]
                        pairs = list(itertools.combinations(range(K), 2))
                        want = (
                            sum(float((Os[a] * Os[b]).sum()) for a, b in pairs)
                            / len(pairs)
                            / min(g[0].shape)
                        )
                        close(
                            errors,
                            f"agree_s1/{key}",
                            float(m[f"gnoise_agree_s1/{key}"]),
                            want,
                            1e-5,
                        )
            other = next(e for e in range(4) if e not in sel)
            assert (
                f"gnoise_rs_tr/{name.removesuffix('.weight')}.e{other}"
                + (".w1" if name.endswith("w13.weight") else "")
                not in m
            )
        assert not errors, f"{len(errors)} mismatches:\n" + "\n".join(errors[:25])

    @with_comms
    def test_research_experts_ep1(self):
        self._experts(dict(dp_replicate=1, dp_shard=4))

    @with_comms
    def test_research_experts_ep2(self):
        self._experts(dict(dp_replicate=1, dp_shard=4, ep=2))
