# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Moving units between the ranks that hold their rows and the rank that
computes on them.

`Exchange` is one set of tables over a subset of units: which rows of which
parameter each rank sends to each owner, in one packed all-to-all. The same
tables serve every direction and every payload (gradients out, updates back,
weights and raw moments to the owner on logging steps), so a rule that needs
more round trips adds a call, not a mechanism. All copies are `_foreach_copy_`
over precomputed contiguous views; owned units of one shape share a
`[B, m, n]` buffer so the rule runs once per batch.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .layout import rows_of_coordinate, RunList


@dataclass
class Unit:
    """One matrix the rule operates on, and where it lives."""

    index: int
    param_index: int
    param: torch.Tensor
    name: str
    rows: RunList  # global rows of the param's [R, C] view
    shape: tuple[int, int]
    view: tuple[int, int]  # the param's [R, C] view
    batch: object = None  # units with equal (shape, batch) run one rule call
    pos: int = 0  # position among its parameter's units


@dataclass
class _Segment:
    unit: int
    src: int
    dst: int
    in_unit: RunList  # rows within the unit's [m, n] matrix
    in_local: RunList  # rows within the source rank's local [nl, C] view
    n_rows: int


def unit_rows(rows2d: torch.Tensor, u: Unit) -> torch.Tensor:
    """The unit's rows out of a `[R, C]` (or local `[nl, C]`) view."""
    runs = u.rows.runs
    if len(runs) == 1 and runs[0] == (0, rows2d.shape[0]):
        return rows2d
    return torch.cat([rows2d[a : a + n] for a, n in runs], dim=0)


def deal(units: list[Unit], world: int) -> dict[int, int]:
    """Owner per unit index: the largest orthogonalisation cost `m n min(m, n)`
    first, dealt in a snake so each rank gets big and small units. Units, not
    parameters: a fused parameter's units are dealt apart."""
    order = sorted(units, key=lambda u: -u.shape[0] * u.shape[1] * min(u.shape))
    return {
        u.index: world - 1 - k % world if k // world % 2 else k % world
        for k, u in enumerate(order)
    }


def batches_of(owned: list[Unit]) -> list[list[Unit]]:
    """Owned units grouped by (shape, batch key), in first-seen order."""
    groups: dict[tuple, list[Unit]] = {}
    for u in owned:
        groups.setdefault((u.shape, u.batch), []).append(u)
    return list(groups.values())


class Exchange:
    """Packed all-to-all tables for `units` (row-sharded on mesh `axis`)."""

    def __init__(self, units, *, axis, group, rank, world, dtype, owner, returns=False):
        self.units, self.group, self.rank, self.world, self.dtype = (
            units,
            group,
            rank,
            world,
            dtype,
        )
        self.owner = owner  # unit index -> rank
        self.returns = returns  # results travel back (`to_holders`)
        self.unit_of = {u.index: u for u in units}
        self.owned = [u for u in units if owner[u.index] == rank]
        self.batches = batches_of(self.owned)
        peer_rows: dict[tuple[int, int], RunList] = {}
        for u in units:
            coord = list(u.param.device_mesh.get_coordinate())
            for r in range(world):
                if (u.param_index, r) not in peer_rows:
                    c = list(coord)
                    c[axis] = r
                    peer_rows[(u.param_index, r)] = rows_of_coordinate(
                        u.param.shape, u.view, u.param, c
                    )
        segments = []
        for u in units:
            for r in range(world):
                inter = peer_rows[(u.param_index, r)].intersect(u.rows)
                if inter.n_rows:
                    segments.append(
                        _Segment(
                            u.index,
                            r,
                            owner[u.index],
                            inter.positions_within(u.rows),
                            inter.positions_within(peer_rows[(u.param_index, r)]),
                            inter.n_rows,
                        )
                    )
        self.send_segs = sorted(
            (s for s in segments if s.src == rank), key=lambda s: (s.dst, s.unit)
        )
        self.recv_segs = sorted(
            (s for s in segments if s.dst == rank), key=lambda s: (s.src, s.unit)
        )
        self.in_splits, self.out_splits = [0] * world, [0] * world
        for s in self.send_segs:
            self.in_splits[s.dst] += s.n_rows * self._cols(s)
        for s in self.recv_segs:
            self.out_splits[s.src] += s.n_rows * self._cols(s)
        self.send_elems, self.recv_elems = sum(self.in_splits), sum(self.out_splits)
        # Where each owned unit starts in the whole-unit (stacks) layout.
        self._base, off = {}, 0
        for us in self.batches:
            for u in us:
                self._base[u.index] = off
                off += u.shape[0] * u.shape[1]
        self.stack_elems = off
        # Rows arrive already in that layout (one owned unit, rows in rank
        # order): the stacks are views of `recv`, no second buffer or copy.
        self.in_place = self.stack_elems > 0 and self._arrival_is_stacks()
        self._ws: dict = {}

    def _arrival_is_stacks(self) -> bool:
        off = 0
        for s in self.recv_segs:
            c = self._cols(s)
            for a, n in s.in_unit.runs:
                if self._base[s.unit] + a * c != off:
                    return False
                off += n * c
        return off == self.stack_elems

    def _cols(self, s: _Segment) -> int:
        return self.unit_of[s.unit].view[1]

    def _slot(self, name: str, device: torch.device, dtype=None) -> dict:
        """Buffers and copy plans for one payload: allocated once, kept.

        Two buffers, shared by lifetime: `send` and the whole units (`stacks`)
        -- `send` is spent once the outbound all-to-all is waited, the stacks
        once copied into `recv` for the way back. With `returns`, `recv` also
        has room for this rank's rows of every unit (`hold`), where the caller
        copies the updates once the return all-to-all is waited."""
        dt = dtype or self.dtype
        ws = self._ws.get((name, dt))
        if ws is not None:
            return ws
        out_buf = torch.empty(
            max(self.send_elems, 0 if self.in_place else self.stack_elems),
            dtype=dt,
            device=device,
        )
        in_buf = torch.empty(
            max(self.recv_elems, self.send_elems if self.returns else 0),
            dtype=dt,
            device=device,
        )
        send, recv = out_buf[: self.send_elems], in_buf[: self.recv_elems]
        home = in_buf if self.in_place else out_buf
        stacks, off = [], 0
        for us in self.batches:
            k = len(us) * us[0].shape[0] * us[0].shape[1]
            stacks.append(home[off : off + k].view(len(us), *us[0].shape))
            off += k
        full = {
            u.index: st[j]
            for us, st in zip(self.batches, stacks)
            for j, u in enumerate(us)
        }
        # send views <- (param_index, run) of the local rows
        pack_dst, pack_rows = [], []
        off = 0
        for s in self.send_segs:
            c = self._cols(s)
            for a, n in s.in_local.runs:
                pack_dst.append(send[off : off + n * c].view(n, c))
                pack_rows.append((self.unit_of[s.unit].param_index, a, n))
                off += n * c
        recv_views, full_views = [], []  # recv views <-> rows of the owned units
        off = 0
        for s in [] if self.in_place else self.recv_segs:
            c = self._cols(s)
            for a, n in s.in_unit.runs:
                recv_views.append(recv[off : off + n * c].view(n, c))
                full_views.append(full[s.unit][a : a + n])
                off += n * c
        ws = dict(
            pack_pis=sorted({pi for pi, _, _ in pack_rows}),
            send=send,
            recv=recv,
            hold=in_buf,
            stacks=stacks,
            full=full,
            pack_dst=pack_dst,
            pack_rows=pack_rows,
            recv_views=recv_views,
            full_views=full_views,
        )
        self._ws[(name, dt)] = ws
        return ws

    @staticmethod
    def _rows(ws: dict, which: str, tensors: dict[int, torch.Tensor], keep: bool):
        """`tensors[pi][a : a + n]` for every packed run. With `keep` (tensors
        that outlive the call: the momentum, the holders' updates) the views
        are reused while the same memory comes back; a kept view would pin a
        temporary."""
        key = tuple(
            (t.data_ptr(), t.shape, t.stride())
            for t in (tensors[pi] for pi in ws["pack_pis"])
        )
        seen = ws.get(which)
        if seen is not None and seen[0] == key:
            return seen[1]
        views = [tensors[pi][a : a + n] for pi, a, n in ws["pack_rows"]]
        if keep:
            ws[which] = (key, views)
        return views

    def release(self, name: str, dtype=None) -> None:
        """Drop a payload's buffers (a logging-only payload between logging steps)."""
        self._ws.pop((name, dtype or self.dtype), None)

    def to_owner(
        self,
        rows2d: dict[int, torch.Tensor],
        name: str,
        device,
        dtype=None,
        *,
        async_op=False,
        keep_views=False,
    ):
        """Local rows per param -> full units on their owners, in the exchange's
        dtype unless `dtype` says otherwise (the logging-step weights go fp32).
        Returns the slot (`full`: unit index -> [m, n]; `stacks`: per batch),
        or with `async_op` a function that waits for the exchange and returns it."""
        ws = self._slot(name, device, dtype)
        if ws["pack_dst"]:
            torch._foreach_copy_(
                ws["pack_dst"], self._rows(ws, "src_views", rows2d, keep_views)
            )
        work = dist.all_to_all_single(
            ws["recv"],
            ws["send"],
            output_split_sizes=self.out_splits,
            input_split_sizes=self.in_splits,
            group=self.group,
            async_op=async_op,
        )

        def finish() -> dict:
            if work is not None:
                work.wait()
            if ws["full_views"]:
                torch._foreach_copy_(ws["full_views"], ws["recv_views"])
            return ws

        return finish if async_op else finish()

    def to_holders(
        self,
        name: str,
        out: dict[int, torch.Tensor],
        dtype=None,
        *,
        async_op=False,
        keep_views=False,
    ):
        """The owned units in slot `name` (already overwritten with the result)
        -> each rank's rows, written into `out[param_index]` (fp32 `[nl, C]`).
        With `async_op`, returns a function that waits and writes them."""
        ws = self._slot(name, next(iter(out.values())).device, dtype)
        if ws["recv_views"]:
            torch._foreach_copy_(ws["recv_views"], ws["full_views"])
        work = dist.all_to_all_single(
            ws["send"],
            ws["recv"],
            output_split_sizes=self.in_splits,
            input_split_sizes=self.out_splits,
            group=self.group,
            async_op=async_op,
        )

        def finish() -> None:
            if work is not None:
                work.wait()
            if ws["pack_dst"]:
                torch._foreach_copy_(
                    self._rows(ws, "dst_views", out, keep_views),
                    ws["pack_dst"],
                )

        return finish if async_op else finish()


class Replicated:
    """Every rank holds every unit whole: owners compute, one all-gather of
    the updates. Owners from `deal`."""

    def __init__(self, units, *, group, rank, world, dtype):
        self.units, self.group, self.rank, self.world, self.dtype = (
            units,
            group,
            rank,
            world,
            dtype,
        )
        owner = deal(units, world)
        self.owner = [owner[u.index] for u in units]
        self.owned = [u for u in units if self.owner[u.index] == rank]
        self.batches = batches_of(self.owned)
        per_rank, self.offset = [0] * world, {}
        for u in units:
            self.offset[u.index] = per_rank[self.owner[u.index]]
            per_rank[self.owner[u.index]] += u.shape[0] * u.shape[1]
        self.chunk = max(per_rank) if per_rank else 0
        self._ws = None

    def _slot(self, device):
        if self._ws is None:
            dt = self.dtype
            send = torch.empty(self.chunk, dtype=dt, device=device)
            recv = torch.empty(self.world * self.chunk, dtype=dt, device=device)
            stacks = [
                torch.empty((len(us), *us[0].shape), dtype=dt, device=device)
                for us in self.batches
            ]
            full = {
                u.index: st[j]
                for us, st in zip(self.batches, stacks)
                for j, u in enumerate(us)
            }
            chunks = (
                list(recv.view(self.world, self.chunk).unbind(0)) if self.chunk else []
            )
            self._ws = dict(
                send=send,
                recv=recv,
                stacks=stacks,
                full=full,
                chunks=chunks,
                gathered={  # each unit's update, whole, once gathered
                    u.index: chunks[self.owner[u.index]][
                        self.offset[u.index] : self.offset[u.index]
                        + u.shape[0] * u.shape[1]
                    ].view(u.shape)
                    for u in self.units
                },
            )
        return self._ws

    def inputs(self, rows2d: dict[int, torch.Tensor], device) -> dict:
        """Owned units' rows into the stacks (one cast); returns the slot."""
        ws = self._slot(device)
        if ws["full"]:
            torch._foreach_copy_(
                [ws["full"][u.index] for us in self.batches for u in us],
                [
                    unit_rows(rows2d[u.param_index], u)
                    for us in self.batches
                    for u in us
                ],
            )
        return ws

    def all_gather(self, out: dict[int, torch.Tensor]) -> None:
        """Owned results (in the slot's `full`) -> every rank's `out[param_index]`
        rows, for the params in `out`; the rest are read from `gathered`."""
        ws = self._ws
        for u in self.owned:
            m, n = u.shape
            ws["send"][self.offset[u.index] : self.offset[u.index] + m * n].view(
                m, n
            ).copy_(ws["full"][u.index])
        if self.world > 1:
            dist.all_gather(ws["chunks"], ws["send"], group=self.group)
        else:
            ws["chunks"][0].copy_(ws["send"])
        for u in self.units:
            if u.param_index not in out:
                continue
            m, n = u.shape
            src = ws["chunks"][self.owner[u.index]][
                self.offset[u.index] : self.offset[u.index] + m * n
            ].view(m, n)
            dst = out[u.param_index]
            if len(u.rows.runs) == 1:
                a, k = u.rows.runs[0]
                dst[a : a + k].copy_(src)
            else:
                off = 0
                for a, k in u.rows.runs:
                    dst[a : a + k].copy_(src[off : off + k])
                    off += k
