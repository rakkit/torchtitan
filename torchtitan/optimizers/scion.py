# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed.tensor

from torchtitan.optimizers.muon_utils import gather_full_grad, zeropower_backends
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
        # NB: use default momentum here, param groups can have its own
        #     values
        self.use_momentum = momentum > 0 and momentum < 1
        self.is_unconstrained = is_unconstrained
        self.fsdp_enabled = parallel_dims.fsdp_enabled
        logger.info(
            f"Scion optimizer (is_light={self.is_light}, is_unconstrained={self.is_unconstrained}) "
            f"is enabled with world_mesh={parallel_dims.world_mesh} | "
            f"fsdp_enabled={self.fsdp_enabled}"
        )
        super().__init__(params, defaults)
        if self.is_light:
            # Initialize state
            self._store_grads_in_state()
            # Do not pass `self` through syntactic sugar. We need the
            # argument to not be populated.
            self.register_state_dict_pre_hook(
                type(self)._store_grads_in_state,
            )
            self.register_load_state_dict_post_hook(
                type(self)._load_grads_from_state,
            )

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
                "zeropower_backend": group["backend"],
                "backend_steps": group["backend_steps"],
            }
            # NB: here assume that normalisation with norm_factor is
            #     done across non-sharded axis, so can skip
            #     communication
            need_to_gather_and_shard = not (
                group["backend"] == "identity" and "embed" in group["norm_factor"]
            )
            if self.is_light and nesterov:
                raise NotImplementedError(
                    "Nesterov momentum is not supported for light mode. "
                    "Please set nesterov=False."
                )

            for p in group["params"]:
                g = self.get_momentum_or_grad(
                    p,
                    momentum,
                    nesterov,
                    update_buffer=True,
                    gather_to_local=need_to_gather_and_shard and self.fsdp_enabled,
                )
                if g is None:
                    continue
                update = self.lmo(g, **param_kwargs)

                if self.fsdp_enabled and need_to_gather_and_shard:
                    # update = shard_full_grad(update)
                    device_mesh = p.grad.device_mesh
                    placements = p.grad.placements
                    update = torch.distributed.tensor.distribute_tensor(
                        update,
                        device_mesh=device_mesh,
                        placements=placements,
                    )
                if update.shape != p.data.shape:
                    raise RuntimeError(
                        f"Shape mismatch: g.shape={g.shape}, p.data.shape={p.data.shape}"
                    )

                if not self.is_unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

                if self.is_light and self.use_momentum:
                    p.grad.mul_(1 - momentum)

        return loss

    @torch.no_grad()
    def lmo(
        self,
        g,
        eps,
        norm_factor,
        zeropower_backend,
        backend_steps,
        is_grouped_experts=False,
    ):
        # NB: make sure this function does not modify the grad inplace
        #     since it is also called during the log of gradients
        def _lmo_for_2d_tensor(g, need_transpose=False):
            g = g if not need_transpose else g.transpose(0, 1)
            g = zeropower_backends[zeropower_backend](g, steps=backend_steps, eps=eps)
            g = self.normalise_grad(g, norm_factor=norm_factor, eps=eps)
            return g if not need_transpose else g.transpose(0, 1)

        if not is_grouped_experts:
            # double check if the grad is grouped experts
            is_grouped_experts = g.ndim == 3

        if g.ndim == 2:
            g = _lmo_for_2d_tensor(g, need_transpose=is_grouped_experts)
        elif g.ndim == 3:
            g = torch.stack(
                [
                    _lmo_for_2d_tensor(g[i], need_transpose=is_grouped_experts)
                    for i in range(g.shape[0])
                ],
                dim=0,
            )
        else:
            raise ValueError(f"Unknown grad shape: {g.shape}")

        return g

    @torch.no_grad()
    def normalise_grad(self, g, norm_factor, eps):
        if norm_factor == "spectral":
            g = g * (g.size(0) / g.size(1)) ** 0.5
        elif norm_factor == "image_spectral":
            # Image domain norm as described in Scion paper.
            g = g * max((g.size(0) / g.size(1)) ** 0.5, 1)
        elif norm_factor.startswith("embed"):
            # NB: here assume shape [vocab_size, embed_dim]
            rms_values = torch.sqrt(g.pow(2).sum(axis=1, keepdim=True))
            g = g / (rms_values + eps)
            if norm_factor == "embed_linear":
                g = g * g.size(1)
            elif norm_factor == "embed_sqrt":
                g = g * g.size(1) ** 0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor.startswith("unembed"):
            rms_values = torch.sqrt(g.pow(2).sum(axis=1, keepdim=True))
            g = g / (rms_values + eps)
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

    @torch.no_grad()
    def get_momentum_or_grad(
        self, p, momentum, nesterov, update_buffer=True, gather_to_local=True
    ):
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        if not self.is_light and self.use_momentum:
            state = self.state[p]
            if "momentum_buffer" not in state.keys():
                if update_buffer:
                    state["momentum_buffer"] = torch.zeros_like(g)
                else:
                    raise ValueError(
                        "Momentum buffer not found in optimizer state. "
                        "Please check if the optimizer is initialized correctly."
                    )
            buf = state["momentum_buffer"]
            if update_buffer:
                buf.mul_(1 - momentum).add_(g, alpha=momentum)
            else:
                buf = buf.mul(1 - momentum).add(g, alpha=momentum)
            g = buf if not nesterov else buf.mul(1 - momentum).add(g, alpha=momentum)

        if gather_to_local:
            g = gather_full_grad(g).to_local()

        return g

    def __getstate__(self):
        self._store_grads_in_state()
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self._load_grads_from_state()

    def _store_grads_in_state(self):
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    self.state.setdefault(param, {})["grad_state"] = param.grad

    def _load_grads_from_state(self):
        for param, state in self.state.items():
            if "grad_state" in state:
                param.grad = state["grad_state"]
            elif isinstance(param, torch.Tensor):
                param.grad = None
