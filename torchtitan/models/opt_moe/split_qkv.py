# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Three separate Q/K/V projections.

Upstream consolidated ``QKVLinear`` onto a single FUSED ``wqkv`` parameter
(#4526 removed ``BaseQKVLinear``/the split variant, #4815 moved the Q/K/V
conversion into the HF adapters), so the split projection has no upstream
equivalent any more. opt_moe keeps it for its checkpoints: every existing one
stores ``wq``/``wk``/``wv`` as separate parameters, and a fused ``wqkv``
changes the native DCP layout.

Ported verbatim from upstream ``models/common/attention.py`` as of
``3dadef77c``, with the dead ``get_spmd_backend() == "spmd_types"`` guard
dropped (``spmd_types`` is the only backend since #4217/#4419).

TODO(fused-qkv): evaluate adopting upstream's fused ``QKVLinear`` (the
``fuse_qkv`` flag) and deleting this file. DiSCO is not in the way: the fused
``wqkv`` is declared as three units (``qkv_units``), so its update is the same
as for split ``wq``/``wk``/``wv``. What remains is the native DCP layout (one
``wqkv`` where there were three tensors), so existing checkpoints do not load;
HF export is unaffected, the adapters split Q/K/V at the external boundary.
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


class SplitQKVLinear(Module):
    """Three separate linear projections for Q, K and V."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        wq: Linear.Config
        wkv: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.wq = config.wq.build()
        self.wk = config.wkv.build()
        self.wv = config.wkv.build()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input into Q, K, V.

        Returns:
            xq and xk have shape ``[T, H, K]``; xv has shape ``[T, H, V]``.
        """
        num_tokens = x.shape[0]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of n_heads (or n_kv_heads) to infer the actual local
        # heads from sizes, as TP may have sharded them.
        def local_qkv_head_split(t: torch.Tensor) -> torch.Tensor:
            # Drop into the local region: S(1) cannot be propagated through the
            # qkv head unflatten.
            with spmd.local():
                t = t.view(num_tokens, -1, self.head_dim)
                if spmd.is_type_checking():
                    spmd.assert_type(
                        t,
                        spmd.V,
                        spmd.PartitionSpec(("dp", "cp"), "tp", None),
                    )
            return t

        return (
            local_qkv_head_split(xq),
            local_qkv_head_split(xk),
            local_qkv_head_split(xv),
        )
