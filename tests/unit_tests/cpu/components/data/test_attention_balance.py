# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Node-local attention balancing: the assignment keeps every token exactly
once and every row's length (variable counts: within the cap), cuts only what
fits nowhere whole, and lowers the largest per-rank cost; both balancers on 4
Gloo ranks return rows whose union is the node's tokens with their labels,
positions restarting at every segment."""

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
    NodeComputeBalancer,
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


def _total_cost(pieces, token_cost):
    return [
        float(
            sum(
                segment_attention_cost(np.array([n]), WINDOWS)[0] + token_cost * n
                for _, _, n in rank
            )
        )
        for rank in pieces
    ]


def test_variable_counts_move_short_documents_off_the_long_document_rank():
    # with equal counts the rank holding the 60-token document stays the
    # slowest whatever it keeps; with variable counts it keeps fewer tokens
    rows = [_row([60, 4]), _row([4] * 16), _row([8] * 8), _row([2] * 32)]
    token_cost = 4.0
    equal = assign_segments(rows, WINDOWS)
    variable = assign_segments(rows, WINDOWS, token_cost=token_cost, max_tokens=80)
    counts = [sum(n for _, _, n in rank) for rank in variable]
    assert sum(counts) == 4 * 64 and max(counts) <= 80
    covered = sorted(
        (src, start + i)
        for rank in variable
        for src, start, n in rank
        for i in range(n)
    )
    assert covered == [(r, i) for r in range(4) for i in range(64)]
    long_rank = next(i for i, rank in enumerate(variable) if (0, 0, 60) in rank)
    assert counts[long_rank] < 64
    assert max(_total_cost(variable, token_cost)) < max(_total_cost(equal, token_cost))


def test_compute_balancer_config_refuses_no_token_cost():
    with pytest.raises(ValueError, match="token_cost"):
        NodeComputeBalancer.Config(token_cost=0.0)
    with pytest.raises(ValueError, match="max_tokens_ratio"):
        NodeComputeBalancer.Config(token_cost=1.0, max_tokens_ratio=0.5)


def _balance_rank(rank, world, init_file, rows, out_file, balancer_config):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        from torchtitan.config.parallelism import ParallelismConfig
        from torchtitan.models.llama3 import build_model_config

        balancer = balancer_config.build(
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
                "padding_mask": new.padding_mask,
                "loss_token_counts": new.loss_token_counts,
            },
            f"{out_file}.{rank}",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "balancer_config",
    [
        NodeAttentionBalancer.Config(),
        NodeComputeBalancer.Config(token_cost=4.0, row_multiple=8),
    ],
    ids=["equal_counts", "variable_counts"],
)
def test_balancer_on_four_gloo_ranks_keeps_the_node_tokens(balancer_config):
    rows = [_row([60, 4]), _row([4] * 16), _row([8] * 8), _row([2] * 32)]
    with tempfile.TemporaryDirectory() as tmp:
        init_file, out = os.path.join(tmp, "init"), os.path.join(tmp, "out")
        mp.spawn(
            _balance_rank,
            args=(4, init_file, rows, out, balancer_config),
            nprocs=4,
            join=True,
        )
        results = [torch.load(f"{out}.{r}", weights_only=True) for r in range(4)]
    # the variable rows end in a padding segment up to a multiple of 8 tokens
    pairs = sorted(
        (int(i), int(l))
        for res in results
        for i, l, pad in zip(
            res["input"], res["labels"], res["padding_mask"], strict=True
        )
        if not pad
    )
    want = sorted(
        (1000 * r + i, IGNORE_INDEX if i == 63 else 1000 * r + i)
        for r in range(4)
        for i in range(64)
    )
    assert pairs == want
    counts = [len(res["input"]) for res in results]
    if isinstance(balancer_config, NodeComputeBalancer.Config):
        assert all(n % 8 == 0 for n in counts)
        tokens = [int((~res["padding_mask"]).sum()) for res in results]
        assert min(tokens) < 64 <= max(tokens) <= 80  # cap 1.25 x 64
        for res in results:  # padding: ignored labels
            assert (res["labels"][res["padding_mask"]] == IGNORE_INDEX).all()
    else:
        assert counts == [64] * 4
    total = sum(int(res["loss_token_counts"]) for res in results)
    assert total == 4 * 63
    for res in results:
        pos = res["positions"].numpy()
        starts = split_segments(pos)
        for a, b in zip(starts, list(starts[1:]) + [len(pos)], strict=True):
            assert pos[a:b].tolist() == list(range(b - a))


if __name__ == "__main__":
    pytest.main([__file__])
