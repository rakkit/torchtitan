# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-rank pieces of the transport: batching and the replicated inputs."""

import torch

from torchtitan.optimizers.disco.layout import RunList
from torchtitan.optimizers.disco.transport import (
    batches_of,
    deal,
    Exchange,
    Replicated,
    Unit,
)


def _unit(index, param_index, shape, batch, rows=None):
    p = torch.zeros(shape)
    return Unit(
        index,
        param_index,
        p,
        "",
        rows or RunList.contiguous(0, shape[0]),
        shape,
        shape,
        batch,
    )


def test_batches_group_by_shape_and_key_in_first_seen_order():
    us = [
        _unit(0, 0, (4, 8), 0),
        _unit(1, 1, (2, 8), 0),
        _unit(2, 2, (4, 8), 0),
        _unit(3, 3, (4, 8), 1),
    ]
    assert [[u.index for u in b] for b in batches_of(us)] == [[0, 2], [1], [3]]


def test_replicated_inputs_land_in_their_own_stack_slot():
    # Owned order 0,1,2 with shapes A,B,A batches as [0,2],[1]: unit 2's rows
    # must reach batch 0 slot 1, not batch 1 slot 0.
    us = [_unit(0, 0, (4, 8), 0), _unit(1, 1, (2, 8), 0), _unit(2, 2, (4, 8), 0)]
    rep = Replicated(us, group=None, rank=0, world=1, dtype=torch.float32)
    rows = {i: torch.randn(u.shape) for i, u in enumerate(us)}
    ws = rep.inputs(rows, torch.device("cpu"))
    for u in us:
        assert torch.equal(ws["full"][u.index], rows[u.param_index])


def test_replicated_fused_units_take_their_rows_and_write_them_back():
    # One [6, 4] parameter split into units rows [0,2)+[4,6) and [2,4).
    p_rows = torch.randn(6, 4)
    a = Unit(0, 0, p_rows, "a", RunList(((0, 2), (4, 2))), (4, 4), (6, 4), 0)
    b = Unit(1, 0, p_rows, "b", RunList(((2, 2),)), (2, 4), (6, 4), 0)
    rep = Replicated([a, b], group=None, rank=0, world=1, dtype=torch.float32)
    ws = rep.inputs({0: p_rows}, torch.device("cpu"))
    assert torch.equal(ws["full"][0], torch.cat([p_rows[0:2], p_rows[4:6]]))
    assert torch.equal(ws["full"][1], p_rows[2:4])
    ws["full"][0].mul_(2.0)
    ws["full"][1].mul_(3.0)
    out = {0: torch.empty(6, 4)}
    rep.all_gather(out)
    want = p_rows.clone()
    want[0:2] *= 2
    want[4:6] *= 2
    want[2:4] *= 3
    assert torch.equal(out[0], want)


def test_fused_units_are_dealt_apart():
    # Three layers of a fused [q; k; v], q four times the rows of k or v. In
    # parameter order (q k v q k v q k v), `index % 3` put every q on rank 0.
    us = [
        _unit(3 * layer + i, layer, shape, 0)
        for layer in range(3)
        for i, shape in enumerate(((8, 4), (2, 4), (2, 4)))
    ]
    owner = deal(us, 3)
    assert sorted(owner[u.index] for u in us if u.shape == (8, 4)) == [0, 1, 2]
    loads = [0] * 3
    for u in us:
        m, n = u.shape
        loads[owner[u.index]] += m * n * min(m, n)
    assert len(set(loads)) == 1, loads


def test_packed_row_views_are_kept_only_for_memory_that_persists():
    """The step exchange rebuilt ~2 views per unit every call; kept views are
    reused while the same memory comes back, and never pin a temporary."""
    ws = {"pack_pis": [0, 1], "pack_rows": [(0, 0, 2), (1, 1, 3)]}
    a, b = torch.zeros(4, 3), torch.zeros(5, 3)
    first = Exchange._rows(ws, "src", {0: a.view(4, 3), 1: b.view(5, 3)}, keep=True)
    again = Exchange._rows(ws, "src", {0: a.view(4, 3), 1: b.view(5, 3)}, keep=True)
    assert again is first  # new view objects, same memory: reused
    moved = Exchange._rows(ws, "src", {0: a.clone(), 1: b}, keep=True)
    assert moved is not first and moved[0].data_ptr() != first[0].data_ptr()
    ws2 = {"pack_pis": [0], "pack_rows": [(0, 0, 2)]}
    Exchange._rows(ws2, "src", {0: a}, keep=False)
    assert "src" not in ws2  # a temporary is never held
