# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Three separate Q/K/V projections.

Upstream consolidated ``QKVLinear`` onto a single FUSED ``wqkv`` parameter
(#4526 removed ``BaseQKVLinear``/the split variant, #4815 moved the Q/K/V
conversion into the HF adapters), so the split projection has no upstream
equivalent any more. opt_moe keeps it for two reasons:

* **Checkpoints.** Every existing opt_moe checkpoint stores ``wq``/``wk``/``wv``
  as separate parameters. A fused ``wqkv`` changes the native DCP layout.
* **DiSCO.** The optimizer orthogonalizes each 2-D parameter on its own.
  Orthogonalizing one stacked ``[(n_heads + 2 * n_kv) * head_dim, dim]`` matrix
  is not the same operation as orthogonalizing ``wq``, ``wk`` and ``wv``
  separately, so fusing would silently change the update.

Ported verbatim from upstream ``models/common/attention.py`` as of
``3dadef77c``, with the dead ``get_spmd_backend() == "spmd_types"`` guard
dropped (``spmd_types`` is the only backend since #4217/#4419).

TODO(fused-qkv): evaluate adopting upstream's fused ``QKVLinear`` and deleting
this file. DiSCO handles it MECHANICALLY already -- ``wqkv`` is 2-D
``[(n_heads + 2 * n_kv) * head_dim, dim]``, so ``get_param_type`` files it
FSDP/DDP and orthogonalizes it like any other matrix; nothing needs changing in
the optimizer. Two open questions, both numerics rather than plumbing:

  1. Orthogonalizing the CONCATENATION is not orthogonalizing wq/wk/wv
     separately. Whether that is better, worse or neutral for DiSCO is an
     experiment, not a refactor -- run it at fixed seed against a split-QKV arm
     before switching.
  2. The native DCP layout changes (one ``wqkv`` where there were three
     tensors), so existing checkpoints do not load. HF export is unaffected:
     the adapters split Q/K/V at the external boundary.

The throughput argument for fusing is one GEMM instead of three. Pair this with
TODO(fused-swiglu) in ``norm_ffn.py``; they are the same decision twice.
"""

from dataclasses import dataclass

import torch

import spmd_types as spmd

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
