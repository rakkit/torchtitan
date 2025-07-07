# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed.tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate

from torchtitan.tools.logging import logger

from .muon_utils import zeropower_backends

__all__ = [
    "DistributedScion",
]


class ParamType(Enum):
    DDP = 0
    FSDP = 1
    Expert = 2
    Unknown = 3


def get_param_type(p, fsdp_enabled, expert_enabled):
    if p.grad is None:
        return ParamType.Unknown
    if not fsdp_enabled and not expert_enabled and isinstance(p, torch.Tensor):
        return ParamType.DDP
    device_mesh = p.device_mesh
    if p.ndim == 3:
        return ParamType.Expert
    if (
        len(device_mesh.mesh.shape) == 1
        and device_mesh.mesh_dim_names[0] == "dp_shard_cp"
    ):
        return ParamType.FSDP
    elif (
        len(device_mesh.mesh.shape) == 2
        and device_mesh.mesh_dim_names[0] == "dp_shard_1"
        and device_mesh.mesh_dim_names[1] == "dp_shard_2"
    ):
        return ParamType.Expert
    else:
        return ParamType.Unknown


def calculate_shard_shape(shape, rank, world_size):
    full = shape[0]
    base = full // world_size
    extra = full % world_size
    dim0 = base + 1 if rank < extra else base
    return (dim0, *shape[1:])


class DistributedScion(torch.optim.Optimizer):
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
        communication_dtype=torch.bfloat16,
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
        self.use_momentum = (
            momentum > 0 and momentum < 1
        )  # NB: use default momentum here, param groups can have its own values

        self.is_unconstrained = is_unconstrained
        self.world_mesh = parallel_dims.world_mesh

        self.fsdp_enabled = parallel_dims.fsdp_enabled
        self.expert_enabled = parallel_dims.ep_enabled
        self.dp_replicate_enabled = parallel_dims.dp_replicate_enabled
        self.tp_enabled = parallel_dims.tp_enabled
        logger.info(
            f"Distributed Scion optimizer "
            f"(is_light={self.is_light}, is_unconstrained={self.is_unconstrained}) "
            f"is enabled with world_mesh={self.world_mesh} | fsdp_enabled={self.fsdp_enabled} | "
            f"expert_enabled={self.expert_enabled}"
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

        self.communication_dtype = communication_dtype
        self.groups_info = {}
        self.paramters_to_groups = {}
        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"],
                "zeropower_backend": group["backend"],
                "backend_steps": group["backend_steps"],
            }
            self.groups_info[group_idx] = [lr, nesterov, momentum, param_kwargs]
            for param in group["params"]:
                self.paramters_to_groups[id(param)] = group_idx

            if self.is_light and nesterov:
                raise RuntimeError(
                    "Nesterov momentum is not supported for Scion's light mode. "
                    "Please set nesterov=False."
                )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        sgd_params = []
        ddp_params = []
        fsdp_params = []
        expert_params = []

        for group_idx, group in enumerate(self.param_groups):
            # we should update self.groups_info here incase we have LR and momentum scheduler
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"],
                "zeropower_backend": group["backend"],
                "backend_steps": group["backend_steps"],
            }
            self.groups_info[group_idx] = [lr, nesterov, momentum, param_kwargs]

            for p in group["params"]:
                param_kwargs = self.groups_info[self.paramters_to_groups[id(p)]][-1]
                norm_factor = param_kwargs["norm_factor"]
                backend = param_kwargs["zeropower_backend"]
                is_embed_norm = norm_factor.startswith(
                    "embed"
                ) or norm_factor.startswith("unembed")

                if backend == "identity" and is_embed_norm and self.fsdp_enabled:
                    sgd_params.append(p)
                    # fsdp_params.append(p)
                    continue

                param_type = get_param_type(p, self.fsdp_enabled, self.expert_enabled)
                if param_type == ParamType.DDP:
                    ddp_params.append(p)
                elif param_type == ParamType.FSDP:
                    fsdp_params.append(p)
                elif param_type == ParamType.Expert:
                    expert_params.append(p)
                elif param_type == ParamType.Unknown:
                    continue
                else:
                    raise ValueError("param_type")

        fsdp_params.sort(key=lambda x: x.numel(), reverse=True)
        expert_params.sort(key=lambda x: (x.numel(), x.shape[1]), reverse=True)

        self.step_sgd(sgd_params)
        self.step_ddp(ddp_params)
        self.step_expert(expert_params)
        self.step_fsdp(fsdp_params)
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

    def update_bucket_params(self, params, updates, start_idx, end_idx):

        for idx_in_bucket in range(start_idx, end_idx):
            shift = idx_in_bucket - start_idx
            p = params[idx_in_bucket]
            u = updates[shift]
            lr, nesterov, momentum, param_kwargs = self.groups_info[
                self.paramters_to_groups[id(p)]
            ]

            if not self.is_unconstrained:
                p.data.mul_(1 - lr)

            if isinstance(p, DTensor):
                p.to_local().add_(u, alpha=-lr)
            else:
                p.data.add_(u, alpha=-lr)

            if momentum != 1 and self.is_light:
                p.grad.mul_(1 - momentum)

    def step_sgd(self, sgd_params):
        if len(sgd_params) == 0:
            return

        for param_idx in range(len(sgd_params)):
            p = sgd_params[param_idx]
            lr, nesterov, momentum, param_kwargs = self.groups_info[
                self.paramters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )
            if isinstance(g, DTensor):
                u = self.lmo(g.to_local(), **param_kwargs)
                if not self.is_unconstrained:
                    p.data.to_local().mul_(1 - lr)
                p.data.to_local().add_(u, alpha=-lr)

            else:
                u = self.lmo(g, **param_kwargs)
                if not self.is_unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(u, alpha=-lr)

            if momentum != 1 and self.is_light:
                g.mul_(1 - momentum)

    def step_ddp(self, ddp_params):
        if len(ddp_params) == 0:
            return
        world_size = self.world_mesh.size()
        rank = self.world_mesh.get_rank()

        # @ THIS IS A HACK
        bucket_size = world_size
        total_buckets = math.ceil(len(ddp_params) / bucket_size)

        device = ddp_params[0].device
        cast_dtype = self.communication_dtype
        zero_tensor = partial(torch.zeros, dtype=cast_dtype, device=device)

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(ddp_params))
            current_rank_idx = start_idx + rank
            if current_rank_idx < len(ddp_params):
                p = ddp_params[current_rank_idx]
                # Step 1: Get the gradient
                _, nesterov, momentum, _ = self.groups_info[
                    self.paramters_to_groups[id(p)]
                ]
                g = self.get_momentum_or_grad(
                    p, momentum, nesterov, update_buffer=True, gather_to_local=False
                )

            else:
                """
                To avoid idle stream, we can randomly generate on last ranks
                """
                g = zero_tensor(ddp_params[end_idx - 1].shape)

            _IDX_OF_NS5 = min(start_idx + rank, end_idx - 1)
            _, _, _, param_kwargs = self.groups_info[
                self.paramters_to_groups[id(ddp_params[_IDX_OF_NS5])]
            ]
            u = self.lmo(g, **param_kwargs)

            # Step 3: FOR DDP, we do all-gather
            gather_lists = [None] * world_size

            for i in range(world_size):
                param_idx = start_idx + i
                if i == rank or param_idx >= len(ddp_params):
                    gather_lists[i] = u.to(dtype=cast_dtype)
                elif param_idx < len(ddp_params):
                    p = ddp_params[start_idx + i]
                    gather_lists[i] = zero_tensor(p.shape)

            dist.all_gather(gather_lists, u.to(dtype=cast_dtype))
            # Step 4: Update the parameters
            self.update_bucket_params(ddp_params, gather_lists, start_idx, end_idx)

    def step_fsdp(self, fsdp_params):
        if len(fsdp_params) == 0:
            return
        world_size = self.world_mesh.size()
        rank = self.world_mesh.get_rank()

        # @ THIS IS A HACK
        bucket_size = world_size
        total_buckets = math.ceil(len(fsdp_params) / bucket_size)

        device = fsdp_params[0].device
        cast_dtype = self.communication_dtype
        zero_tensor = partial(torch.empty, dtype=cast_dtype, device=device)

        # Process each bucket
        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(fsdp_params))

            # Step 1: Prepare data for first all_to_all
            send_list = []
            send_shapes = []
            target_shape, param_kwargs = None, None

            for rank_idx in range(world_size):
                current_rank_idx = start_idx + rank_idx

                if current_rank_idx < len(fsdp_params):
                    p = fsdp_params[current_rank_idx]
                    _, nesterov, momentum, param_kwargs = self.groups_info[
                        self.paramters_to_groups[id(p)]
                    ]
                    g = (
                        self.get_momentum_or_grad(
                            p,
                            momentum,
                            nesterov,
                            update_buffer=True,
                            gather_to_local=False,
                        )
                        .to_local()
                        .to(dtype=cast_dtype)
                    )
                    # Save the shape info for this parameter
                    if rank == rank_idx:
                        target_shape = p.shape
                else:
                    # Use a dummy shape for parameters beyond our range
                    p = fsdp_params[end_idx - 1]
                    g = zero_tensor(p.to_local().shape)

                send_list.append(g)
                send_shapes.append(g.shape)

            # Make sure target_shape is initialized
            if target_shape is None and end_idx > 0:
                target_shape = fsdp_params[end_idx - 1].shape
                param_kwargs = self.groups_info[
                    self.paramters_to_groups[id(fsdp_params[end_idx - 1])]
                ][-1]

            recv_shapes = [
                calculate_shard_shape(target_shape, rank_idx, world_size)
                for rank_idx in range(world_size)
            ]
            recv_list = [zero_tensor(shape) for shape in recv_shapes]

            # Step 3: First all_to_all - using ASYNC version
            dist.all_to_all(recv_list, send_list)

            # Step 5: Concatenate received gradients along dimension 0 and perform NS5
            # All tensors in recv_list should have the same dimensions except for dim 0

            full_g = torch.cat(recv_list, dim=0)

            u = self.lmo(full_g, **param_kwargs)

            # Step 6: Split the processed tensor back for second all_to_all
            split_sizes = [shape[0] for shape in recv_shapes]

            send_list = list(torch.split(u, split_sizes, dim=0))
            recv_list = [zero_tensor(shape) for shape in send_shapes]

            # Step 8: Second all_to_all - using ASYNC version
            dist.all_to_all(recv_list, send_list)
            del send_list
            # Step 10: Update parameters using the results
            self.update_bucket_params(
                fsdp_params,
                recv_list,
                start_idx,
                end_idx,
            )

    def step_expert(self, expert_params):
        if len(expert_params) == 0:
            return
        # No communication for expert params
        # Just do update locally on each rank

        for param_idx in range(len(expert_params)):
            p = expert_params[param_idx]
            lr, nesterov, momentum, param_kwargs = self.groups_info[
                self.paramters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )
            u = self.lmo(g.to_local(), is_grouped_experts=True, **param_kwargs)

            if not self.is_unconstrained:
                p.data.to_local().mul_(1 - lr)
            p.data.to_local().add_(u, alpha=-lr)

            if momentum != 1 and self.is_light:
                g.mul_(1 - momentum)

    @torch.no_grad()
    def get_momentum_or_grad(
        self, p, momentum, nesterov, update_buffer=False, gather_to_local=True
    ):
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        if not self.is_light and momentum != 1:
            state = self.state[p]
            if "momentum_buffer" not in state.keys():
                if update_buffer:
                    state["momentum_buffer"] = torch.zeros_like(g)
                else:
                    """
                    When you using DDP + Dist-muon,you might trieer an error here.
                    Because in the optimizer.log you try to log all gradient's norm.
                    But for DDP + Dist-muon, each rank only has a part of the gradient.

                    --
                    For debug, you can return None here.
                    """
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


def gather_full_grad(g):
    """Gathers the full gradient across all distributed processes using DTensor."""
    assert isinstance(g, DTensor), "Expected gradient to be a DTensor"
    replicated_grad = g.redistribute(
        placements=[Replicate()] * g.device_mesh.ndim
    )  # make sure all rank has the same shape
    return replicated_grad
