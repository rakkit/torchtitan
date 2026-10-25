# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gnoise research mode: every small-batch gradient of a unit on one owner.

At the hook each rank copies its small batch's gradient of every covered unit
(bf16) into a send buffer laid out by owner; after backward one all-to-all over
the dp world moves them. A matrix unit goes whole to one owner (a cost-balanced
deal over the ranks that hold a copy: every dp rank for dense units, the ranks
of that expert's EP index for routed experts), a vocabulary or router unit by
row blocks (its LMO is row-local), a vector whole. The owner then has all K
small batches and W, so every estimate is exact, with its jackknife SE, and the
LMO agreement is taken at every subset size.
"""

import logging
import math
import re
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.optimizers import gnoise_math as gm

logger = logging.getLogger(__name__)

_NOISE_TOPK = (1, 2, 3)  # the noise's own top-k share, k <= K - 1
_BS1_MIN_K = 8
_ROW_KINDS = ("embedding", "unembedding", "router")


@dataclass
class _Item:
    j: int
    pp: object
    u: int  # global unit index in pp
    lmo: str  # "polar" | "row" | "sign"
    owner: int  # dp position; row items: -1 (row blocks over all positions)
    holders: list[int]  # dp positions holding a copy
    rows: torch.Tensor  # rows of the unit in the hook's [R_local, C] view (holder side)
    shape: tuple[int, int]  # (rows, C) of the unit
    expert: int = -1
    slot: int = 0
    span: tuple[int, int] | None = None  # rows as one contiguous slice


def _span(rows: torch.Tensor) -> tuple[int, int] | None:
    if rows.numel() == 0:
        return None
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    return (
        (lo, hi) if hi - lo == rows.numel() and bool((rows.diff() == 1).all()) else None
    )


def _divisors(K: int) -> list[int]:
    return [s for s in range(1, K // 2 + 1) if K % s == 0]


class Research:
    def __init__(self, gn) -> None:
        self.gn = gn
        cfg = gn.config
        pat = re.compile(cfg.research_matrices) if cfg.research_matrices else None
        W = gn.world
        self.items: list[_Item] = []
        load = [0.0] * W
        for pp in gn.params:
            for u, key in enumerate(pp.keys):
                if pat is not None and not pat.search(key):
                    continue
                m, n = pp.shapes[u]
                rows_g = (pp.row_unit == u).nonzero().flatten()
                if pp.kind in ("vector", "scalar"):
                    lmo, holders = "sign", list(range(W))
                elif pp.kind in _ROW_KINDS:
                    lmo, holders = "row", list(range(W))
                else:
                    lmo = "polar"
                    if pp.expert:
                        e = u // pp.per
                        c = e // pp.local_experts
                        holders = [q for q in range(W) if gn.ep_of[q] == c]
                    else:
                        holders = list(range(W))
                if pp.expert:
                    e, sub = divmod(u, pp.per)
                    loc = (
                        e - (e // pp.local_experts) * pp.local_experts
                    ) * pp.per + sub
                    rows = (pp.local_row_unit == loc).nonzero().flatten()
                else:
                    rows = rows_g
                owner = -1
                if lmo != "row":
                    cost = float(m * n * min(m, n)) if lmo == "polar" else float(m)
                    owner = min(holders, key=lambda q: (load[q], q))
                    load[owner] += cost
                self.items.append(
                    _Item(
                        len(self.items),
                        pp,
                        u,
                        lmo,
                        owner,
                        holders,
                        rows,
                        (m, n),
                        expert=u // pp.per if pp.expert else -1,
                        span=_span(rows),
                    )
                )
        if not self.items:
            raise ValueError("optimizer.gnoise.research_matrices matches no unit")
        self.accum = None
        # hidden matrices: Psi from the 1/2- and 1/4-batch subset sums, after the step
        self.psi_items = [
            it
            for it in self.items
            if it.lmo == "polar" and not it.pp.expert and it.pp.kind == "matrix"
        ]
        self.psi_js = {it.j for it in self.psi_items}
        self.psi_pending = False
        self.recv_kept, self.psi_store, self.psi_done = None, {}, 0

    # ------------------------------------------------------------- layout
    def set_layout(self, accum: int) -> None:
        """The static send / receive layout for `accum` microbatches per rank."""
        gn, W, me = self.gn, self.gn.world, self.gn.rank
        self.accum = accum
        K = W * accum
        cfg = gn.config
        self.sizes = sorted(cfg.research_sizes) if cfg.research_sizes else _divisors(K)
        bad = [s for s in self.sizes if K % s or s > K // 2]
        if bad:
            raise ValueError(
                f"optimizer.gnoise.research_sizes {bad}: each must divide K = {K} "
                "and be <= K / 2"
            )
        # pieces[(src, dst)] = [(item, row lo, row hi)] in a fixed order
        self.send_off, self.recv_off = {}, {}
        send_sizes, recv_sizes = [0] * W, [0] * W
        self.piece: dict[
            tuple[int, int, int], tuple[int, int]
        ] = {}  # (item, src, dst) -> rows
        for d in range(W):
            for it in self.items:
                for lo, hi, dst in self._targets(it):
                    if dst != d:
                        continue
                    numel = (hi - lo) * it.shape[1]
                    if me in it.holders:
                        self.send_off[(it.j, d)] = send_sizes[d]
                        send_sizes[d] += numel * accum
                    for s in it.holders:
                        if d == me:
                            self.recv_off[(it.j, s)] = recv_sizes[s]
                            recv_sizes[s] += numel * accum
        self.K = K
        self.ntop = [k for k in _NOISE_TOPK if k <= K - 1]
        self.bs1 = K >= _BS1_MIN_K
        if not self.bs1:
            logger.info(
                f"[gnoise] B_S1 not computed: K = {K} small batches < {_BS1_MIN_K}"
            )
        self.send_sizes, self.recv_sizes = send_sizes, recv_sizes
        self.send_base = [sum(send_sizes[:d]) for d in range(W)]
        self.recv_base = [sum(recv_sizes[:s]) for s in range(W)]
        self._slots()

    def _targets(self, it) -> list[tuple[int, int, int]]:
        """(row lo, row hi, destination) blocks of an item, rows within the unit."""
        m = it.shape[0]
        if it.lmo != "row":
            return [(0, m, it.owner)]
        W = self.gn.world
        b = math.ceil(m / W)
        return [(d * b, min(m, (d + 1) * b), d) for d in range(W) if d * b < m]

    def _slots(self) -> None:
        nk, ns = len(self.gn.config.research_topk), len(self.sizes)
        extra = len(self.ntop) + (3 if self.bs1 else 0)
        width = {"polar": 8 + nk + 3 * ns + 1 + extra, "row": 3 * ns, "sign": 2 * ns}
        off = 0
        for it in self.items:
            it.slot = off
            off += width[it.lmo]
        self.n_res = off

    # ------------------------------------------------------------- per step
    def begin(self) -> None:
        dev = self.gn._device()
        self.send = torch.empty(sum(self.send_sizes), dtype=torch.bfloat16, device=dev)
        self.w_own: dict[int, torch.Tensor] = {}
        self.psi_pending = False
        self.recv_kept, self.psi_store, self.psi_done = None, {}, 0

    def snapshot(self, pp, grad: torch.Tensor, W: torch.Tensor, mb: int) -> None:
        me = self.gn.rank
        g2 = grad.reshape(-1, pp.C)
        for it in self._items_of(pp):
            if me not in it.holders:
                continue
            gu = g2[it.span[0] : it.span[1]] if it.span else g2[it.rows]
            for lo, hi, d in self._targets(it):
                numel = (hi - lo) * it.shape[1]
                at = self.send_base[d] + self.send_off[(it.j, d)] + mb * numel
                self.send[at : at + numel].copy_(gu[lo:hi].reshape(-1))
            if it.owner == me and it.lmo == "polar" and mb == 0:
                w2 = W.reshape(-1, pp.C)
                self.w_own[it.j] = (
                    (w2[it.span[0] : it.span[1]] if it.span else w2[it.rows])
                    .detach()
                    .clone()
                )

    def _items_of(self, pp):
        if not hasattr(self, "_by_param"):
            self._by_param = {}
            for it in self.items:
                self._by_param.setdefault(it.pp.i, []).append(it)
        return self._by_param.get(pp.i, [])

    # ---------------------------------------------------------------- finish
    @torch.no_grad()
    def finish(self, res, n_b, moe_d, accum) -> None:
        gn = self.gn
        if accum != self.accum:
            raise RuntimeError(
                f"[gnoise] research layout for {self.accum} microbatches, got {accum}"
            )
        dev = self.send.device
        recv = torch.empty(sum(self.recv_sizes), dtype=torch.bfloat16, device=dev)
        if gn.group is not None:
            dist.all_to_all_single(
                recv, self.send, self.recv_sizes, self.send_sizes, group=gn.group
            )
        else:
            recv.copy_(self.send)
        self.send = None
        counts = (
            gn.expert_counts(moe_d, accum).reshape(gn.world * accum, -1)
            if gn.moe_list
            else None
        )
        out = torch.zeros(self.n_res, dtype=torch.float64, device=dev)
        me = gn.rank
        gen = torch.Generator(device="cpu")
        for it in self.items:
            mine = it.owner == me or (
                it.lmo == "row" and any(d == me for _, _, d in self._targets(it))
            )
            if not mine:
                continue
            gen.manual_seed(gn.step * 1_000_003 + it.j)
            lo, hi = next((a, b) for a, b, d in self._targets(it) if d == me)
            g = self._gather(recv, it, lo, hi)  # [K, rows, C] fp32
            n = self._counts(it, n_b, counts, accum)
            if it.lmo == "polar":
                self._polar(out, it, g, n, gn.n_den.double(), gen)
            elif it.lmo == "row":
                self._row(out, it, g, n, gen)
            else:
                self._sign(out, it, g, n, gen)
        if gn.group is not None:
            dist.all_reduce(out, group=gn.group)
        self._emit(res, out)
        if self.psi_items:  # read again after the optimizer step (`psi`)
            self.recv_kept, self.psi_pending = recv, True

    def _gather(self, recv, it, lo, hi) -> torch.Tensor:
        numel = (hi - lo) * it.shape[1]
        parts = []
        for s in it.holders:
            at = self.recv_base[s] + self.recv_off[(it.j, s)]
            parts.append(
                recv[at : at + numel * self.accum].view(
                    self.accum, hi - lo, it.shape[1]
                )
            )
        return torch.cat(parts).float()

    def _counts(self, it, n_b, counts, accum) -> torch.Tensor:
        rows = [s * accum + k for s in it.holders for k in range(accum)]
        if it.expert < 0:
            return n_b[rows]
        pp = it.pp
        return counts[rows, self.gn.moe_off[pp.moe] + it.expert]

    def _partitions(self, valid: torch.Tensor, s: int, gen) -> list[list[list[int]]]:
        """Disjoint groups of s small batches: one random partition per size, three at the
        half (all three pairings at K = 4); leftovers (experts, K not divisible) dropped."""
        idx = valid.nonzero().flatten().tolist()
        k = len(idx)
        if s > k // 2:
            return []
        if k == 4 and s == 2:
            a, b, c, d = idx
            return [[[a, b], [c, d]], [[a, c], [b, d]], [[a, d], [b, c]]]
        parts = 3 if s == k // 2 else 1
        out = []
        for _ in range(parts):
            perm = torch.randperm(k, generator=gen).tolist()
            g = k // s
            out.append([[idx[perm[a * s + b]] for b in range(s)] for a in range(g)])
        return out

    def _polar(self, out, it, g, n, n_den, gen) -> None:
        cfg, o = self.gn.config, it.slot
        valid = n > 0
        G = g.sum(0)
        flat, Gf = g.reshape(g.shape[0], -1).double(), G.reshape(-1).double()
        tot = gm.estimate(
            (flat * flat).sum(1)[None],
            (Gf @ Gf)[None],
            n[None],
            n_den,
            cross=(flat @ Gf)[None],
        )
        Wt = self.w_own[it.j].float()
        kmax = min(max(cfg.research_topk), min(it.shape))
        Uk, u1, v1 = gm.top_left(Wt, kmax)
        x = torch.einsum("m,kmn,n->k", u1, g, v1).double()
        X = x.sum()
        srad = gm.estimate(
            (x * x)[None], (X * X)[None], n[None], n_den, cross=(X * x)[None]
        )
        vals = [
            tot["tr"][0],
            tot["sig"][0],
            tot["se_tr"][0],
            tot["se_sig"][0],
            srad["tr"][0],
            srad["sig"][0],
            srad["se_tr"][0],
            srad["se_sig"][0],
        ]
        left = it.shape[0] <= it.shape[1]
        for k in cfg.research_topk:
            if k > min(it.shape):
                vals.append(
                    torch.tensor(math.nan, dtype=torch.float64, device=g.device)
                )
                continue
            U = Uk[:, :k]
            p = (
                torch.einsum("sk,bsn->bkn", U, g)
                if left
                else torch.einsum("bms,sk->bmk", g, U)
            )
            P = p.sum(0)
            pf, Pf = p.reshape(p.shape[0], -1).double(), P.reshape(-1).double()
            est = gm.estimate(
                (pf * pf).sum(1)[None],
                (Pf @ Pf)[None],
                n[None],
                n_den,
                cross=(pf @ Pf)[None],
            )
            vals.append(est["tr"][0] / tot["tr"][0])
        sums, meta = [], []
        for si, s in enumerate(self.sizes):
            for part in self._partitions(valid, s, gen):
                for grp in part:
                    sums.append(g[grp].sum(0))
                meta.append((si, len(part)))
        a = [[] for _ in self.sizes]
        rank_frac = torch.tensor(math.nan, dtype=torch.float64, device=g.device)
        if sums:
            S = torch.stack(sums)
            O, rank = gm.exact_polar(S)
            rank_frac = (rank.double() / min(it.shape)).mean()
            at = 0
            for si, groups in meta:
                a[si].append(gm.lmo_agreement(O[at : at + groups], groups))
                at += groups
        n_mean = n[valid].double().mean() if valid.any() else torch.tensor(math.nan)
        for si, s in enumerate(self.sizes):
            vals.append(torch.stack(a[si]).mean() if a[si] else torch.tensor(math.nan))
            vals.append(s * n_mean)
            pairs = sum(groups * (groups - 1) // 2 for i, groups in meta if i == si)
            vals.append(torch.tensor(float(pairs)))
        vals.append(rank_frac)
        vals += self._noise_shape(it, flat, n, n_den, valid)
        if it.j in self.psi_js:
            self.psi_store[it.j] = (flat @ flat.T, n.clone())
        out[o : o + len(vals)] = torch.stack(
            [
                torch.as_tensor(v, dtype=torch.float64, device=out.device).reshape(())
                for v in vals
            ]
        )

    def _noise_shape(self, it, flat, n, n_den, valid) -> list:
        """The noise's own top-k share and B_S1 from d_b = ghat_b - Ghat (valid b):
        eigenvalues of the K x K Gram of sqrt(n_b) d_b (the per-token covariance's), and
        C_row = sum_b n_b d_b d_b^T / (K - 1), B_S1 = ||C_row^(1/2)||_S1^2 / ||Ghat||_S1^2."""
        dev, nan = flat.device, math.nan
        vi = valid.nonzero().flatten()
        k = vi.numel()
        nb = n[vi].double()
        fv = flat[vi]
        Gh = (n_den / nb.sum()) * fv.sum(0)
        wD = nb.sqrt()[:, None] * ((n_den / nb)[:, None] * fv - Gh)
        lam = torch.linalg.eigvalsh(wD @ wD.T).flip(0).clamp_min(0) if k else None
        vals = [
            lam[:q].sum() / lam.sum() if q <= k - 1 else torch.tensor(nan, device=dev)
            for q in self.ntop
        ]
        if not self.bs1:
            return vals
        if k < _BS1_MIN_K:
            return vals + [torch.tensor(nan, device=dev)] * 3
        m, c = it.shape
        D = (wD / math.sqrt(k - 1)).view(k, m, c)
        if m <= k * c:
            C = torch.einsum("bik,bjk->ij", D, D)
        else:  # the same nonzero eigenvalues from the smaller side
            M2 = D.permute(1, 0, 2).reshape(m, k * c)
            C = M2.T @ M2
        noise = torch.linalg.eigvalsh(C).clamp_min(0).sqrt().sum().square()
        grad = torch.linalg.svdvals(Gh.view(m, c)).sum().square()
        return vals + [noise / grad, noise, grad]

    @torch.no_grad()
    def psi(self, res, disco, u) -> None:
        """After the optimizer step: Psi_S = <O, G_S> / (|O| |G_S|) for the 1/2- and
        1/4-batch subset sums G_S, O the update DiSCO applied (gathered whole); the
        groups are b = position among the valid small batches (rank-major), group a =
        {b : b mod (K / s) = a}; each statistic is the mean over its K / s groups."""
        gn = self.gn
        ents = {id(e.param): e for e in disco.entries}
        items = [it for it in self.psi_items if id(it.pp.p) in ents]
        if not items:
            return
        dev, me = gn._device(), gn.rank
        out = torch.zeros(len(items), 2, dtype=torch.float64, device=dev)
        full_of: dict[int, torch.Tensor] = {}
        for slot, it in enumerate(items):
            pp = it.pp
            if pp.i not in full_of:  # every rank joins each param's gather once
                full_of.clear()
                e, p = ents[id(pp.p)], pp.p
                loc = u[e.i].reshape(e.local.shape).float()
                if isinstance(p, DTensor):
                    loc = DTensor.from_local(
                        loc,
                        p.device_mesh,
                        p.placements,
                        run_check=False,
                        shape=p.shape,
                        stride=p.stride(),
                    ).full_tensor()
                full_of[pp.i] = loc.reshape(-1, pp.C)
            if it.owner != me:
                continue
            full = full_of[pp.i]
            R, n = self.psi_store[it.j]
            O = (full[it.span[0] : it.span[1]] if it.span else full[it.rows]).double()
            g = self._gather(self.recv_kept, it, 0, it.shape[0]).double()
            dots, oo = torch.einsum("bmn,mn->b", g, O), O.square().sum()
            idx = (n > 0).nonzero().flatten().tolist()
            k = len(idx)
            for j, groups in enumerate((2, 4)):
                if k < groups or k % groups:
                    out[slot, j] = math.nan
                    continue
                v = []
                for a in range(groups):
                    gi = torch.tensor(
                        [idx[q] for q in range(k) if q % groups == a], device=dev
                    )
                    v.append(dots[gi].sum() / (oo * R[gi][:, gi].sum()).sqrt())
                out[slot, j] = torch.stack(v).mean()
        if gn.group is not None:
            dist.all_reduce(out, group=gn.group)
        for slot, it in enumerate(items):
            key = it.pp.keys[it.u]
            gn._put(res, f"gnoise_psi_half/{key}", out[slot, 0], it.pp.i)
            gn._put(res, f"gnoise_psi_quarter/{key}", out[slot, 1], it.pp.i)
        self.psi_done += len(items)
        if self.psi_done >= len(self.psi_items):
            self.psi_pending = False
            self.recv_kept, self.psi_store, self.psi_done = None, {}, 0

    def _row(self, out, it, g, n, gen) -> None:
        valid = n > 0
        for si, s in enumerate(self.sizes):
            num, cnt, both = 0.0, 0, 0.0
            for part in self._partitions(valid, s, gen):
                S = torch.stack([g[grp].sum(0) for grp in part])
                cos, present = gm.row_agreement(S, len(part))
                pairs = len(part) * (len(part) - 1) // 2
                num = num + cos * pairs * g.shape[1]
                both = both + present * pairs * g.shape[1]
                cnt += pairs * g.shape[1]
            o = it.slot + 3 * si
            out[o] = num
            out[o + 1] = cnt
            out[o + 2] = both

    def _sign(self, out, it, g, n, gen) -> None:
        valid = n > 0
        for si, s in enumerate(self.sizes):
            agree = []
            for part in self._partitions(valid, s, gen):
                S = (
                    torch.sign(torch.stack([g[grp].sum(0) for grp in part]))
                    .reshape(len(part), -1)
                    .double()
                )
                iu = torch.triu_indices(len(part), len(part), 1)
                agree.append(((S[iu[0]] * S[iu[1]]).mean(1)).mean())
            o = it.slot + 2 * si
            out[o] = torch.stack(agree).mean() if agree else math.nan
            out[o + 1] = s * float(n[valid].double().mean())

    # ---------------------------------------------------------------- emit
    def _emit(self, res, out) -> None:
        gn = self.gn
        cfg = gn.config
        from torchtitan.optimizers.gnoise import type_key

        types: dict[str, list] = {}
        for it in self.items:
            pp, key, o = it.pp, it.pp.keys[it.u], it.slot
            put = lambda name, v: gn._put(
                res, f"gnoise_{name}/{key}", v, pp.i
            )  # noqa: E731
            if it.lmo == "polar":
                for i, name in enumerate(
                    (
                        "rs_tr",
                        "rs_sig",
                        "rs_se_tr",
                        "rs_se_sig",
                        "srad_tr",
                        "srad_sig",
                        "se_srad_tr",
                        "se_srad_sig",
                    )
                ):
                    put(name, out[o + i])
                for i, k in enumerate(cfg.research_topk):
                    put(f"topk{k}", out[o + 8 + i])
                base = o + 8 + len(cfg.research_topk)
                for si, s in enumerate(self.sizes):
                    a = out[base + 3 * si]
                    put(f"agree_s{s}", a)
                    put(f"agree_ntok_s{s}", out[base + 3 * si + 1])
                    put(f"agree_pairs_s{s}", out[base + 3 * si + 2])
                    types.setdefault(f"{type_key(key)}|{s}", []).append(a)
                put("polar_rank", out[base + 3 * len(self.sizes)])
                at = base + 3 * len(self.sizes) + 1
                for i, k in enumerate(self.ntop):
                    put(f"ntop{k}", out[at + i])
                if self.bs1:
                    at += len(self.ntop)
                    put("bs1", out[at])
                    put("bs1_noise", out[at + 1])
                    put("bs1_grad", out[at + 2])
            elif it.lmo == "row":
                for si, s in enumerate(self.sizes):
                    num, cnt, both = out[it.slot + 3 * si : it.slot + 3 * si + 3]
                    put(f"agree_s{s}", num / cnt)
                    put(f"agree_present_s{s}", both / cnt)
            else:
                for si, s in enumerate(self.sizes):
                    put(f"agree_s{s}", out[it.slot + 2 * si])
                    put(f"agree_ntok_s{s}", out[it.slot + 2 * si + 1])
        for tk, vals in types.items():
            tkey, s = tk.split("|")
            gn._put(
                res, f"gnoise_type_agree_s{s}/{tkey}", torch.stack(vals).nanmean(), 0
            )
