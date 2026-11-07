# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Node-local attention balancing of packed document microbatches.

Each data-parallel rank packs its own documents, so the attention work of a
microbatch depends on its document lengths: a full-attention layer costs
~L^2 / 2 per document of length L, a sliding-window layer ~L * W. With every
collective waiting for the slowest rank, the rank that drew the long documents
sets the step time. The ranks of one node regroup their microbatches' document
segments so each keeps the same token count with about equal attention work:

1. the node's ranks all-gather their microbatches (token ids, labels,
   positions, padding mask; CPU, Gloo) -- the same rows on every rank;
2. every rank runs the same deterministic assignment: segments by descending
   attention cost, each to the rank with the least cost that still has room
   for it whole; a segment that fits nowhere whole is cut over the ranks with
   room left (the packer already cuts documents at row ends the same way);
3. each rank builds its row from its segments; positions restart at 0 for
   every segment (document starts are where positions reset to 0, and the
   attention within a segment depends only on relative positions).

The step's documents, labels and loss are unchanged; only which rank computes
which document changes (and the rare extra cut). Off by default
(``Trainer.Config.attention_balancer``).
"""

from __future__ import annotations

import heapq
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import Configurable
from torchtitan.models.deepseek_v3.mtp import get_mtp_token_counts


def segment_attention_cost(
    lengths: np.ndarray, windows: Sequence[int | None]
) -> np.ndarray:
    """Causal attention work of documents of ``lengths`` summed over layers
    with the given sliding windows (``None``: full attention): per layer and
    document, the number of query-key pairs ``sum_{i=1..L} min(i, W)``."""
    L = lengths.astype(np.float64)
    cost = np.zeros_like(L)
    for window in windows:
        if window is None:
            cost += L * (L + 1) / 2
        else:
            W = float(window)
            cost += np.where(L <= W, L * (L + 1) / 2, W * (W + 1) / 2 + (L - W) * W)
    return cost


def split_segments(positions: np.ndarray) -> np.ndarray:
    """Start offsets of the document segments of one row: index 0 and every
    position reset to 0 (a row may start inside a document)."""
    starts = np.flatnonzero(positions == 0)
    if starts.size == 0 or starts[0] != 0:
        starts = np.concatenate([[0], starts])
    return starts


def assign_segments(
    rows_positions: Sequence[np.ndarray], windows: Sequence[int | None]
) -> list[list[tuple[int, int, int]]]:
    """The new rows of a group, as ``(source row, start, length)`` pieces per
    rank. Every row keeps its token count (all rows are equally long). Ties
    are broken by (source row, start), so every rank computes the same
    assignment."""
    num_rows = len(rows_positions)
    capacity = len(rows_positions[0])
    segments = []
    for row, positions in enumerate(rows_positions):
        starts = split_segments(positions)
        lengths = np.diff(np.concatenate([starts, [len(positions)]]))
        costs = segment_attention_cost(lengths, windows)
        segments += [
            (-c, row, int(s), int(n))
            for c, s, n in zip(costs.tolist(), starts, lengths, strict=True)
        ]
    segments.sort()
    room = [capacity] * num_rows
    load = [0.0] * num_rows
    pieces: list[list[tuple[int, int, int]]] = [[] for _ in range(num_rows)]
    for neg_cost, row, start, length in segments:
        fits = [r for r in range(num_rows) if room[r] >= length]
        if fits:
            r = min(fits, key=lambda q: (load[q], q))
            pieces[r].append((row, start, length))
            room[r] -= length
            load[r] -= neg_cost
            continue
        # fits nowhere whole: cut it over the ranks with the most room
        left, offset = length, start
        by_room = [(-room[q], q) for q in range(num_rows) if room[q] > 0]
        heapq.heapify(by_room)
        while left:
            _, r = heapq.heappop(by_room)
            take = min(left, room[r])
            pieces[r].append((row, offset, take))
            room[r] -= take
            load[r] += segment_attention_cost(np.array([take]), windows)[0]
            left -= take
            offset += take
    assert all(r == 0 for r in room), room
    return pieces


class NodeAttentionBalancer(Configurable):
    """Regroups each training microbatch's document segments over the
    data-parallel ranks of one node. Every rank of the node calls it once per
    microbatch, in the same order (the trainer's microbatch stream). Needs
    every rank to be its own data-parallel rank (no TP / CP / PP: the Trainer
    config refuses them)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Set to balance; ``None`` (the default) leaves microbatches as
        packed. The attention costs come from the model's layers."""

    def __init__(
        self, config: Config, *, model_config: Any, num_mtp_layers: int
    ) -> None:
        # Hosts gathered once over the world: one task per GPU leaves no
        # node-local launcher view (LOCAL_WORLD_SIZE 1 per task).
        hosts: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(hosts, socket.gethostname())
        nodes: dict[str, list[int]] = {}
        for rank, host in enumerate(hosts):
            nodes.setdefault(host, []).append(rank)
        self.group, _ = dist.new_subgroups_by_enumeration(
            list(nodes.values()), backend="gloo"
        )
        self.rank = dist.get_rank(self.group)
        self.size = dist.get_world_size(self.group)
        self.windows = attention_windows(model_config)
        self.num_mtp_layers = num_mtp_layers

    def __call__(
        self, microbatch: TokenizedTrainingMicrobatch
    ) -> TokenizedTrainingMicrobatch:
        if microbatch.model_kwargs:
            raise ValueError(
                "attention balancing moves token rows only; this microbatch "
                f"carries model_kwargs {sorted(microbatch.model_kwargs)}"
            )
        local = torch.stack(
            [
                microbatch.input,
                microbatch.labels,
                microbatch.positions,
                microbatch.padding_mask.to(torch.int64),
            ]
        )
        gathered = [torch.empty_like(local) for _ in range(self.size)]
        dist.all_gather(gathered, local, group=self.group)
        rows = [g.numpy() for g in gathered]
        mine = assign_segments([row[2] for row in rows], self.windows)[self.rank]
        out = np.concatenate(
            [rows[src][:, start : start + n] for src, start, n in mine], axis=1
        )
        out[2] = np.concatenate([np.arange(n) for _, _, n in mine])
        new = torch.from_numpy(out)
        labels, positions = new[1], new[2]
        padding_mask = new[3].to(torch.bool)
        loss_token_counts, routing_token_counts = get_mtp_token_counts(
            target_mask=labels != IGNORE_INDEX,
            positions=positions,
            padding_mask=padding_mask,
            num_mtp_layers=self.num_mtp_layers,
        )
        if self.num_mtp_layers == 0:
            loss_token_counts = loss_token_counts[0]
        return TokenizedTrainingMicrobatch(
            input=new[0],
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
            loss_token_counts=loss_token_counts,
            routing_token_counts=routing_token_counts,
        )


def attention_windows(model_config) -> list[int | None]:
    """The sliding window of every decoder (and MTP) layer's attention
    (``None``: full attention), read from the model config's inner-attention
    configs."""
    from torchtitan.models.common.attention import InnerAttention

    windows = []
    for layer in list(model_config.layers) + list(
        getattr(model_config, "mtp_layers", None) or []
    ):
        found = [cfg for _, cfg, _, _ in layer.traverse(InnerAttention.Config)]
        if not found:
            continue
        windows.append(getattr(found[0], "sliding_window_size", None))
    return windows
