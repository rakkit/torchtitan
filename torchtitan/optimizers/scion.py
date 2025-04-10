# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.optimizers.muon_utils import (
    gather_full_grad,
    shard_full_grad,
    zeropower_backends,
)
from torchtitan.tools.logging import logger

__all__ = [
    "Scion",
]


class Scion(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        is_light,
        is_unconstrained,
        lr,
        momentum,
        nesterov,
        eps,
        norm_factor,
        backend,
        backend_steps,
        parallel_dims,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            norm_factor=norm_factor,
            backend=backend,
            backend_steps=backend_steps,
        )
        self.is_light = is_light
        self.is_unconstrained = is_unconstrained
        self.fsdp_enabled = parallel_dims.fsdp_enabled
        logger.info(
            f"Scion optimizer (is_light={self.is_light}, is_unconstrained={self.is_unconstrained}) "
            f"is enabled with world_mesh={parallel_dims.world_mesh} | "
            f"fsdp_enabled={self.fsdp_enabled}"
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"],
                "zeropower_backend": zeropower_backends[group["backend"]],
                "backend_steps": group["backend_steps"],
            }
            if self.is_light and nesterov:
                raise NotImplementedError(
                    "Nesterov momentum is not supported for light mode. "
                    "Please set nesterov=False."
                )
            for p in group["params"]:
                g = p.grad
                if g is None or not p.requires_grad:
                    continue

                if not self.is_light and momentum != 1:
                    state = self.state[p]
                    if "momentum_buffer" not in state.keys():
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = (
                        buf
                        if not nesterov
                        else buf.mul(1 - momentum).add(g, alpha=momentum)
                    )

                if self.fsdp_enabled:
                    g = gather_full_grad(g)
                update = self.lmo(g, **param_kwargs)
                if self.fsdp_enabled:
                    update = shard_full_grad(update)
                if update.shape != p.data.shape:
                    raise RuntimeError(
                        f"Shape mismatch: g.shape={g.shape}, p.data.shape={p.data.shape}"
                    )

                if not self.is_unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

                if momentum != 1 and self.is_light:
                    g.mul_(1 - momentum)

        return loss

    @torch.no_grad()
    def lmo(self, g, eps, norm_factor, zeropower_backend, backend_steps):
        # NB: make sure this function does not modify the grad inplace
        #     since it is also called during the log of gradients
        g = zeropower_backend(g, steps=backend_steps, eps=eps)
        g = self.normalise_grad(g, norm_factor=norm_factor, eps=eps)

        return g

    @torch.no_grad()
    def normalise_grad(self, g, norm_factor, eps):
        if norm_factor == "spectral":
            g = g * (g.size(0) / g.size(1)) ** 0.5
        elif norm_factor.startswith("embed"):
            # NB: here assume shape [vocab_size, embed_dim]
            g = g * torch.rsqrt(g.pow(2).sum(axis=1, keepdim=True) + eps)
            if norm_factor == "embed_linear":
                g = g * g.size(1)
            elif norm_factor == "embed_sqrt":
                g = g * g.size(1) ** 0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor.startswith("unembed"):
            g = g * torch.rsqrt(g.pow(2).sum(axis=1, keepdim=True) + eps)
            if norm_factor == "unembed_linear":
                g = g / g.size(1)
            elif norm_factor == "unembed_sqrt":
                g = g / g.size(1) ** 0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor == "none":
            pass
        else:
            raise ValueError(f"Unknown norm_factor: {norm_factor}")

        return g
