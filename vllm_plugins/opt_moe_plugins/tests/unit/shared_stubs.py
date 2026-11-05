# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test stubs for opt_moe_plugins unit tests."""

import torch
import torch.nn as nn


class DummyPPGroup:
    is_first_rank = True
    is_last_rank = True
    rank_in_group = 0
    world_size = 1


class DummyEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, hidden_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class DummyLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, positions, hidden_states, residual):
        return hidden_states, residual


class DummyFusedMoE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class DummyNormEverywhereFusedMoE(DummyFusedMoE):
    supports_norm_everywhere = True


class DummyRaisingNormEverywhereFusedMoE(DummyFusedMoE):
    def __init__(self, **kwargs):
        raise RuntimeError("dummy fused moe init failure")


class DummyRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, prefix: str):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            hidden_states.shape[0],
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )


class DummyExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        rms_norm_eps: float,
        norm_everywhere: bool,
        bias: bool,
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.mid_norm = nn.ReLU() if norm_everywhere else nn.Identity()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.intermediate_size = intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OptMoEDummyRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, prefix: str):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            hidden_states.shape[0],
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
