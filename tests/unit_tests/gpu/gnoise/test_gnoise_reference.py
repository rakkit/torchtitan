# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gnoise against the equations (reference.py) on real forward / backward passes
under FSDP2: every small batch's gradient is taken through FSDP's own reduce with
the other ranks' losses zeroed, then every logged component is recomputed from
its projection. On CPU (gloo, dense test attention) or 4 GPUs."""

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

TOL = 1e-4


class _Base(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _device(self):
        if self.device_type == "cpu":
            H.patch_dense_attention()
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _microbatches(self, dev, *, uneven, masked, accum, identical=False):
        out = []
        for k in range(accum):
            r = 0 if identical else self.rank
            n = H.SEQ * (2 if not uneven else (1 + r % 3))
            inp, lab = H.batch(
                100 * k + r, n, dev, mask_frac=0.25 if masked and r == 2 else 0.0
            )
            out.append((inp, lab))
        return out

    def _counts(self, mbs):
        local = [float((lab != -100).sum()) for _, lab in mbs]
        t = torch.tensor(local, dtype=torch.float64)
        allt = [torch.zeros_like(t) for _ in range(self.world_size)]
        dist.all_gather_object(allt, t)
        return {
            (r, k): float(allt[r][k])
            for r in range(self.world_size)
            for k in range(len(mbs))
        }


def _compare(errors, name, got, want, scale):
    if not (math.isfinite(want) and math.isfinite(got)):
        if math.isfinite(want) != math.isfinite(got):
            errors.append(f"{name}: got {got} want {want}")
        return
    if abs(got - want) > TOL * scale:
        errors.append(f"{name}: got {got:.6e} want {want:.6e} (scale {scale:.3e})")


def check_dense(model, cfg, gn, m, solo, n, n_den, W, *, head_dim):
    """Every dense unit's components against reference.py. Returns a list of errors."""
    errors = []
    layouts = model.param_layouts()
    keys = sorted(solo)
    nlist = [n[b] for b in keys]
    by_name = {pp.name: pp for pp in gn.params}
    types = {}
    for name, p in model.named_parameters():
        if name not in by_name or by_name[name].expert:
            continue
        lay = layouts.get(name) or by_rank(p.shape)
        if p.ndim <= 1:
            units = [("", torch.arange(p.numel()))]
            view = (p.numel(), 1)
        else:
            view = (math.prod(p.shape[:-1]), p.shape[-1])
            units = [
                (s.name, torch.cat([torch.arange(a, a + c) for a, c in s.rows.runs]))
                for s in resolve_units(name, p.shape, lay)
            ]
        Wv = W[name].reshape(view)
        g = [solo[b][name].reshape(view) for b in keys]
        base = name.removesuffix(".weight")
        for uname, rows in units:
            key = f"{base}.{uname}" if uname else base
            gu, Wu = [x[rows] for x in g], Wv[rows]
            kept = [(x, nb) for x, nb in zip(gu, nlist) if nb > 0]
            A = sum(float(((n_den / nb) * x).square().sum()) for x, nb in kept) / len(
                kept
            )
            den = sum(1 / nb for _, nb in kept) / len(kept) - 1 / sum(nlist)
            st, ss = A / den, A
            tr, s = ref.estimate(gu, nlist, n_den)
            _compare(errors, f"tr/{key}", float(m[f"gnoise_tr/{key}"]), tr, st)
            _compare(errors, f"sig/{key}", float(m[f"gnoise_sig/{key}"]), s, ss)
            P = ref.unit_projection(Wu)
            tr, s = ref.estimate([P(x) for x in gu], nlist, n_den)
            _compare(errors, f"rad_tr/{key}", float(m[f"gnoise_rad_tr/{key}"]), tr, st)
            _compare(errors, f"rad_sig/{key}", float(m[f"gnoise_rad_sig/{key}"]), s, ss)
            se_tr, se_s = ref.jackknife([P(x) for x in gu], nlist, n_den)
            _compare(
                errors,
                f"se_rad_tr/{key}",
                float(m[f"gnoise_se_rad_tr/{key}"]),
                se_tr,
                st,
            )
            _compare(
                errors,
                f"se_rad_sig/{key}",
                float(m[f"gnoise_se_rad_sig/{key}"]),
                se_s,
                ss,
            )
            P = ref.tangent(Wu)
            tr, s = ref.estimate([P(x) for x in gu], nlist, n_den)
            _compare(errors, f"tan_tr/{key}", float(m[f"gnoise_tan_tr/{key}"]), tr, st)
            _compare(errors, f"tan_sig/{key}", float(m[f"gnoise_tan_sig/{key}"]), s, ss)
            P = ref.row_radial(Wu)
            tr, s = ref.estimate([P(x) for x in gu], nlist, n_den)
            _compare(
                errors, f"rrad_tr/{key}", float(m[f"gnoise_rrad_tr/{key}"]), tr, st
            )
            _compare(
                errors, f"rrad_sig/{key}", float(m[f"gnoise_rrad_sig/{key}"]), s, ss
            )
            t = types.setdefault(ref_type(key), [0.0, 0.0])
            tr, s = ref.estimate(gu, nlist, n_den)
            t[0] += tr
            t[1] += s
            if uname in ("wq", "wk", "wv"):
                pass  # head blocks below, per parameter
        # per row (the EMA inputs), total and row radial
        pp = by_name[name]
        rows_tr = gn.rows[f"row/{pp.i}/tr"]
        for i in range(0, view[0], max(1, view[0] // 7)):
            tr, s = ref.estimate([x[i] for x in g], nlist, n_den)
            kept = [(x, nb) for x, nb in zip(g, nlist) if nb > 0]
            A = sum(
                float(((n_den / nb) * x[i]).square().sum()) for x, nb in kept
            ) / len(kept)
            den = sum(1 / nb for _, nb in kept) / len(kept) - 1 / sum(nlist)
            _compare(
                errors, f"row{i}/{name}", float(rows_tr[i]), tr, max(A / den, 1e-30)
            )
        # heads: q / k / v row blocks (in unit order), wo column blocks
        if name.endswith("wqkv.weight"):
            blocks = []
            for uname, rows in units:
                for h in range(len(rows) // head_dim):
                    blocks.append(rows[h * head_dim : (h + 1) * head_dim])
            for h, rows in enumerate(blocks):
                tr, s = ref.estimate([x[rows] for x in g], nlist, n_den)
                _compare(
                    errors,
                    f"hd/{base}.h{h}",
                    float(m[f"gnoise_hd_tr/{base}.h{h}"]),
                    tr,
                    abs(tr) + 1e-30 + abs(s) * sum(nlist),
                )
        if name.endswith(".wo.weight"):
            for h in range(view[1] // head_dim):
                cols = slice(h * head_dim, (h + 1) * head_dim)
                tr, s = ref.estimate([x[:, cols] for x in g], nlist, n_den)
                _compare(
                    errors,
                    f"hd/{base}.h{h}",
                    float(m[f"gnoise_hd_tr/{base}.h{h}"]),
                    tr,
                    abs(tr) + abs(s) * sum(nlist),
                )
        # leave-one-out sign agreement of the per-row radial
        if p.ndim >= 1:
            w = Wv.double()
            nrm = w.norm(dim=1)
            d = [
                torch.where(nrm > 0, (x * w).sum(1) / nrm.clamp_min(1e-300), 0.0)
                for x in g
            ]
            want = ref.loo_sign(d)
            got = float(
                m[f"gnoise_sign_loo/{units[0][0] and f'{base}.{units[0][0]}' or base}"]
            )
            # a coordinate at the resolved / unresolved threshold may land either side
            if abs(got - want) > 5e-3:
                errors.append(f"sign_loo/{name}: got {got} want {want}")
    for tkey, (tr, s) in types.items():
        got = float(m[f"gnoise_type_tr/{tkey}"])
        if abs(got - tr) > TOL * (abs(tr) + 1e-30) * 10:
            errors.append(f"type_tr/{tkey}: got {got} want {tr}")
    return errors


def ref_type(key: str) -> str:
    import re

    return re.sub(r"layers\.\d+\.", "layers.*.", key)


def check_experts(
    model, gn, m, solo, counts, n_den, accum, world, groups=None, gsolo=None
):
    """Per routed expert: small batch = (EP group, microbatch), n = the group's routed
    count; at EP 1 every rank is its own group (g from `solo`), at EP > 1 g from `gsolo`."""
    errors = []
    from torchtitan.models.opt_moe.norm_moe import NormMoE

    moes = sorted(n for n, mod in model.named_modules() if isinstance(mod, NormMoE))
    E_of = {
        n: mod.num_experts
        for n, mod in model.named_modules()
        if isinstance(mod, NormMoE)
    }
    off, o = {}, 0
    for k in moes:
        off[k], o = o, o + E_of[k]
    groups = groups or [(r,) for r in range(world)]
    keys = [(G, k) for G in groups for k in range(accum)]
    for name, p in model.named_parameters():
        if ".routed_experts." not in name:
            continue
        moe = name.split(".routed_experts.")[0]
        E = p.shape[0]
        per = 2 if name.endswith("w13.weight") else 1
        subs = ("w1", "w3") if per == 2 else ("",)
        for e in range(E):
            n_e = [
                sum(float(counts[(r, k)][r][off[moe] + e]) for r in G)
                for (G, k) in keys
            ]
            for si, sub in enumerate(subs):
                src = [
                    gsolo[(G, k)] if gsolo is not None else solo[(G[0], k)]
                    for (G, k) in keys
                ]
                g = [x[name][e][si] if per == 2 else x[name][e] for x in src]
                tr, s = ref.estimate(g, n_e, n_den)
                key = f"{name.removesuffix('.weight')}.e{e}" + (
                    f".{sub}" if sub else ""
                )
                got_tr, got_s = float(m[f"gnoise_tr/{key}"]), float(
                    m[f"gnoise_sig/{key}"]
                )
                keep = [i for i, c in enumerate(n_e) if c > 0]
                if len(keep) < 2:
                    if math.isfinite(got_tr):
                        errors.append(
                            f"{key}: {len(keep)} small batches with tokens, logged {got_tr}"
                        )
                    continue
                N = sum(n_e[i] for i in keep)
                A = sum(
                    float(((n_den / n_e[i]) * g[i]).square().sum()) for i in keep
                ) / len(keep)
                den = sum(1 / n_e[i] for i in keep) / len(keep) - 1 / N
                _compare(errors, f"tr/{key}", got_tr, tr, A / den)
                _compare(errors, f"sig/{key}", got_s, s, A)
    return errors


class TestReference(_Base):
    def _scenario(
        self,
        topo,
        *,
        moe=False,
        uneven=True,
        masked=True,
        accum=2,
        chunks=4,
        param_dtype="bfloat16",
        identical=False,
        balanced=False,
    ):
        dev = self._device()
        with tempfile.TemporaryDirectory() as tmp:
            (
                pd,
                model,
                cont,
                cfg,
            ) = H.build(  # balanced: equal all-to-all splits (gloo needs them)
                topo,
                dev,
                tmp,
                gnoise=dict(enable=True),
                chunks=chunks,
                param_dtype=param_dtype,
                model_cfg=H.gauge_config(moe=moe, num_experts=8) if moe else None,
                force_balance=balanced,
            )
            gn = cont.gnoise
            gn.validate_layout(accum=accum)
            groups = None
            if (
                moe and topo.get("ep", 1) > 1
            ):  # the EP groups, from the mesh's process groups
                mine = tuple(
                    dist.get_process_group_ranks(pd.get_mesh("ep").get_group())
                )
                allg = [None] * self.world_size
                dist.all_gather_object(allg, mine)
                groups = sorted(set(allg))
                assert len(groups) == self.world_size // topo["ep"], groups
            mbs = self._microbatches(
                dev, uneven=uneven, masked=masked, accum=accum, identical=identical
            )
            if (
                moe
            ):  # 16 tokens x top-2 over 8 experts: empty (expert, rank) pairs occur
                ntok = H.SEQ // 2
                if (
                    groups is not None
                ):  # equal within a group (gloo), unequal across groups
                    ntok *= 1 + next(i for i, G in enumerate(groups) if self.rank in G)
                mbs = [(i[:ntok], l[:ntok]) for i, l in mbs]
            n = self._counts(mbs)
            n_den = int(sum(n.values()))
            W = H.full_weights(model, getattr(torch, param_dtype))
            solo, counts = H.solo_grads(
                model, pd, cfg, mbs, n_den, self.world_size, self.rank
            )
            m = H.run_step(model, pd, cfg, cont, 1, mbs, n_den)
            errors = check_dense(model, cfg, gn, m, solo, n, n_den, W, head_dim=16)
            if moe:
                gsolo = None
                if groups is not None:
                    names = {
                        n
                        for n, _ in model.named_parameters()
                        if ".routed_experts." in n
                    }
                    gsolo = H.group_grads(
                        model, pd, cfg, mbs, n_den, groups, self.rank, names
                    )
                errors += check_experts(
                    model,
                    gn,
                    m,
                    solo,
                    counts,
                    n_den,
                    accum,
                    self.world_size,
                    groups,
                    gsolo,
                )
                if not balanced:
                    empty = sum(int((c[r] == 0).sum()) for (r, k), c in counts.items())
                    assert (
                        empty > 0
                    ), "no empty (expert, small batch): the exclusion rule is untested"
            gauge = {k: float(v) for k, v in m.items() if k.startswith("gnoise_gauge")}
            if float(m["gnoise/N_check"]) != 0.0:
                errors.append(f"N_check {float(m['gnoise/N_check'])}")
        return errors, gauge

    def _assert(self, errors):
        assert not errors, f"{len(errors)} mismatches:\n" + "\n".join(errors[:25])

    @with_comms
    def test_fsdp4_bf16_uneven_masked_accum2_chunked(self):
        self._assert(self._scenario(dict(dp_replicate=1, dp_shard=4))[0])

    @with_comms
    def test_ddp4_bf16_uneven_masked_accum2_chunked(self):
        self._assert(self._scenario(dict(dp_replicate=4, dp_shard=1))[0])

    @with_comms
    def test_hsdp22_bf16_uneven_masked_accum2_chunked(self):
        self._assert(self._scenario(dict(dp_replicate=2, dp_shard=2))[0])

    @with_comms
    def test_fsdp4_fp32_gauges(self):
        """T7: the gauge identities hold per small batch to fp32 precision."""
        errors, gauge = self._scenario(
            dict(dp_replicate=1, dp_shard=4), param_dtype="float32", chunks=0
        )
        bad = {k: v for k, v in gauge.items() if not v < 1e-5}
        self._assert(errors + [f"gauge {k} = {v}" for k, v in bad.items()])

    @with_comms
    def test_ddp4_fp32_gauges(self):
        errors, gauge = self._scenario(
            dict(dp_replicate=4, dp_shard=1), param_dtype="float32", chunks=0
        )
        bad = {k: v for k, v in gauge.items() if not v < 1e-5}
        self._assert(errors + [f"gauge {k} = {v}" for k, v in bad.items()])

    @with_comms
    def test_moe_fsdp4_ep2_balanced(self):
        """EP 2: an expert's small batch is its EP group's tokens (K_e = 2 x accum)."""
        self._assert(
            self._scenario(
                dict(dp_replicate=1, dp_shard=4, ep=2),
                moe=True,
                accum=2,
                chunks=0,
                uneven=False,
                masked=True,
                balanced=True,
            )[0]
        )

    @with_comms
    def test_moe_fsdp4_ep1(self):
        """8 experts, few tokens per rank: some (expert, rank) small batches are empty."""
        self._assert(
            self._scenario(
                dict(dp_replicate=1, dp_shard=4), moe=True, accum=1, chunks=0
            )[0]
        )


class TestKnownAnswers(_Base):
    """T4 / T5: equal mean gradients on every small batch give tr = 0 and S = ||P G||^2."""

    def _known(self, topo, *, duplicate, masked, moe=False, param_dtype="float32"):
        dev = self._device()
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                topo,
                dev,
                tmp,
                moe=moe,
                gnoise=dict(enable=True),
                param_dtype=param_dtype,
            )
            cont.gnoise.validate_layout(accum=1)
            inp, lab = H.batch(7, H.SEQ * 2, dev)
            if (
                duplicate and self.rank == 1
            ):  # the same sequences twice, as separate documents
                inp, lab = torch.cat([inp, inp]), torch.cat([lab, lab])
            if (
                masked and self.rank == 2
            ):  # plus a separate document whose labels are all masked
                extra_inp, _ = H.batch(9, H.SEQ, dev)
                inp = torch.cat([inp, extra_inp])
                lab = torch.cat([lab, torch.full((H.SEQ,), -100, device=dev)])
            mbs = [(inp, lab)]
            n = self._counts(mbs)
            m = H.run_step(model, pd, cfg, cont, 1, mbs, int(sum(n.values())))
        errors = []
        for k, v in m.items():
            if not k.startswith(
                ("gnoise_tr/", "gnoise_rad_tr/", "gnoise_rrad_tr/", "gnoise_hd_tr/")
            ):
                continue
            key = k.split("/", 1)[1]
            base = key
            tot_sig = m.get(f"gnoise_sig/{base}")
            if tot_sig is None:  # heads: compare with the parameter's first unit total
                tot_sig = max(
                    (
                        m[x]
                        for x in m
                        if x.startswith("gnoise_sig/" + key.rsplit(".h", 1)[0])
                    ),
                    default=None,
                )
            sig = float(m[k.replace("_tr/", "_sig/", 1)])
            tr = float(v)
            # tr * (H - 1/N) = A - F must vanish against A ~ S of the unit (tokens: n ~ SEQ)
            if abs(tr) > 1e-3 * H.SEQ * 2 * abs(float(tot_sig)):
                errors.append(
                    f"{k} = {tr:.3e} (S {sig:.3e}, unit S {float(tot_sig):.3e})"
                )
        return errors

    @with_comms
    def test_identical_fsdp4(self):
        self._assert_empty(
            self._known(dict(dp_replicate=1, dp_shard=4), duplicate=False, masked=False)
        )

    @with_comms
    def test_duplicate_uneven_fsdp4(self):
        """T4: rank 1 holds rank 0's sequences twice (n_1 = 2 n_0)."""
        self._assert_empty(
            self._known(dict(dp_replicate=1, dp_shard=4), duplicate=True, masked=False)
        )

    @with_comms
    def test_masked_extra_document_hsdp22(self):
        """T5: rank 2 adds a document whose labels are masked; n_b counts valid tokens."""
        self._assert_empty(
            self._known(dict(dp_replicate=2, dp_shard=2), duplicate=False, masked=True)
        )

    @with_comms
    def test_duplicate_moe_fsdp4(self):
        self._assert_empty(
            self._known(
                dict(dp_replicate=1, dp_shard=4), duplicate=True, masked=False, moe=True
            )
        )

    def _assert_empty(self, errors):
        assert not errors, f"{len(errors)} components with tr != 0:\n" + "\n".join(
            errors[:25]
        )
