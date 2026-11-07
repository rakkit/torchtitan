# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Node-local attention balancing: the assignment keeps every token exactly
once and every row's length, cuts only what fits nowhere whole, and lowers the
largest per-rank attention cost; the balancer on 4 Gloo ranks returns rows
whose union is the node's tokens with their labels, positions restarting at
every segment."""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.components.data.balance import (
    assign_segments,
    NodeAttentionBalancer,
    segment_attention_cost,
    split_segments,
)
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import IGNORE_INDEX

WINDOWS = [None, 4, 4, 4]  # one full layer, three sliding-window layers


def _row(lengths):
    return np.concatenate([np.arange(n) for n in lengths])


def _cost(rows_positions, pieces):
    return [
        float(
            sum(segment_attention_cost(np.array([n]), WINDOWS)[0] for _, _, n in rank)
        )
        for rank in pieces
    ]


def _packed_cost(rows_positions):
    out = []
    for positions in rows_positions:
        starts = split_segments(positions)
        lengths = np.diff(np.concatenate([starts, [len(positions)]]))
        out.append(float(segment_attention_cost(lengths, WINDOWS).sum()))
    return out


def test_cost_counts_query_key_pairs():
    # full: L (L + 1) / 2; window 4: 1 + 2 + 3 + 4 then 4 per further token
    got = segment_attention_cost(np.array([3, 10]), [None, 4])
    assert got.tolist() == [6 + 6, 55 + (10 + 6 * 4)]


def test_segments_start_at_position_resets_and_at_the_row_start():
    # a row may start inside a document (positions 5, 6)
    assert split_segments(np.array([5, 6, 0, 1, 2, 0, 0])).tolist() == [0, 2, 5, 6]


def test_assignment_keeps_every_token_once_and_balances():
    # two long documents on one rank: one moves (a single document longer than
    # a fair share would stay the bottleneck: segments are not split for cost)
    rows = [_row([32, 32]), _row([4] * 16), _row([8] * 8), _row([2] * 32)]
    pieces = assign_segments(rows, WINDOWS)
    assert [sum(n for _, _, n in rank) for rank in pieces] == [64] * 4
    covered = sorted(
        (src, start + i) for rank in pieces for src, start, n in rank for i in range(n)
    )
    assert covered == [(r, i) for r in range(4) for i in range(64)]
    assert max(_cost(rows, pieces)) < max(_packed_cost(rows))


def test_assignment_cuts_only_what_fits_nowhere_whole():
    # equal rows of whole segments that fit: no cut, nothing moves across a
    # document boundary
    rows = [_row([16, 16, 16, 16])] * 4
    pieces = assign_segments(rows, WINDOWS)
    assert all(n == 16 and start % 16 == 0 for rank in pieces for _, start, n in rank)
    # fragmented room: the last 21-token document fits in no rank whole and is
    # cut over the ranks with room left; every token is still placed once
    rows = [_row([40, 24]), _row([40, 24]), _row([22, 21, 21])]
    pieces = assign_segments(rows, WINDOWS)
    assert [sum(n for _, _, n in rank) for rank in pieces] == [64, 64, 64]
    covered = sorted(
        (src, start + i) for rank in pieces for src, start, n in rank for i in range(n)
    )
    assert covered == [(r, i) for r in range(3) for i in range(64)]
    assert sum(len(rank) for rank in pieces) > 7  # 7 segments, at least one cut


def test_assignment_is_deterministic():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(4):
        lengths = rng.integers(1, 20, size=40)
        lengths = lengths[np.cumsum(lengths) <= 256]
        rows.append(_row(list(lengths) + [256 - int(lengths.sum())]))
    assert assign_segments(rows, WINDOWS) == assign_segments(rows, WINDOWS)


def _balance_rank(rank, world, init_file, rows, out_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        from torchtitan.config.parallelism import ParallelismConfig
        from torchtitan.models.llama3 import build_model_config

        balancer = NodeAttentionBalancer.Config().build(
            model_config=build_model_config(
                "debugmodel", parallelism=ParallelismConfig()
            ),
            num_mtp_layers=0,
        )
        positions = torch.from_numpy(rows[rank])
        n = len(positions)
        labels = torch.arange(n) + 1000 * rank
        labels[-1] = IGNORE_INDEX
        mb = TokenizedTrainingMicrobatch(
            input=torch.arange(n) + 1000 * rank,
            labels=labels,
            positions=positions,
            padding_mask=torch.zeros(n, dtype=torch.bool),
            loss_token_counts=(labels != IGNORE_INDEX).sum(),
            routing_token_counts=torch.tensor([n]),
        )
        new = balancer(mb)
        torch.save(
            {
                "input": new.input,
                "labels": new.labels,
                "positions": new.positions,
                "loss_token_counts": new.loss_token_counts,
            },
            f"{out_file}.{rank}",
        )
    finally:
        dist.destroy_process_group()


def test_balancer_on_four_gloo_ranks_keeps_the_node_tokens():
    rows = [_row([60, 4]), _row([4] * 16), _row([8] * 8), _row([2] * 32)]
    with tempfile.TemporaryDirectory() as tmp:
        init_file, out = os.path.join(tmp, "init"), os.path.join(tmp, "out")
        mp.spawn(_balance_rank, args=(4, init_file, rows, out), nprocs=4, join=True)
        results = [torch.load(f"{out}.{r}", weights_only=True) for r in range(4)]
    pairs = sorted(
        (int(i), int(l))
        for res in results
        for i, l in zip(res["input"], res["labels"], strict=True)
    )
    want = sorted(
        (1000 * r + i, IGNORE_INDEX if i == 63 else 1000 * r + i)
        for r in range(4)
        for i in range(64)
    )
    assert pairs == want
    assert all(len(res["input"]) == 64 for res in results)
    total = sum(int(res["loss_token_counts"]) for res in results)
    assert total == 4 * 63
    for res in results:
        pos = res["positions"].numpy()
        starts = split_segments(pos)
        for a, b in zip(starts, list(starts[1:]) + [len(pos)], strict=True):
            assert pos[a:b].tolist() == list(range(b - a))


if __name__ == "__main__":
    pytest.main([__file__])
