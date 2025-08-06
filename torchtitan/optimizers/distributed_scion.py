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
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from torchtitan.tools.logging import logger

from .muon_utils import zeropower_backends
from .norm_helper import calculate_norm, NORM_FUNCTIONS
from .utils import remove_orig_mod_and_weight_for_p_name

__all__ = [
    "DistributedScion",
]


def gather_and_merge(local_stats: dict, dst: int = 0):
    world = dist.get_world_size()
    rank = dist.get_rank()
    dtype = torch.bfloat16

    my_keys = list(local_stats.keys())

    if len(my_keys) > 0:
        val_tensor = torch.stack([local_stats[k].to(dtype) for k in my_keys])
    else:
        my_keys = "padding"
        val_tensor = None

    key_bucket = [None] * world if rank == dst else None
    val_bucket = [None] * world if rank == dst else None

    dist.gather_object(my_keys, key_bucket, dst=dst)
    # dist.barrier()
    dist.gather_object(val_tensor, val_bucket, dst=dst)
    dist.barrier()

    merged = {}
    if rank == dst:
        for peer, keys in enumerate(key_bucket):
            if val_bucket[peer] is None:
                continue
            for k, v in zip(keys, val_bucket[peer]):
                if k != "padding":
                    merged[k] = v

    dist.barrier()
    if rank == dst:
        return merged
    else:
        return {}


class ParamType(Enum):
    DDP = 0
    FSDP = 1
    Expert = 2
    Unknown = 3


def get_param_type(p, fsdp_enabled, expert_enabled):
    """
    We can aggressively assume that the param is FSDP-Sharded
    """
    if p.grad is None:
        return ParamType.Unknown
    if not fsdp_enabled and not expert_enabled and isinstance(p, torch.Tensor):
        return ParamType.DDP
    if p.ndim == 3:
        return ParamType.Expert
    elif fsdp_enabled:
        return ParamType.FSDP
    else:
        return ParamType.Unknown


def tp_axis(placements: tuple) -> int | None:
    """
    Return the index in `placements` that belongs to *tensor-parallel* (TP).

    Heuristics (PyTorch-TP default layouts):
      1. Row-parallel weights ⇒ `_StridedShard`  ⟶ that axis is TP.
      2. Col-parallel weights ⇒ `Shard(dim != 0)` ⟶ that axis is TP
         (FSDP shards dim-0, so a non-zero dim means TP).
    """
    # rule 1 – row-parallel
    for i, p in enumerate(placements):
        if isinstance(p, _StridedShard):
            return i

    # rule 2 – col-parallel
    for i, p in enumerate(placements):
        if isinstance(p, Shard) and p.dim != 0:
            return i

    return None  # could not infer


def gather_tp_shard(tensor, tp_group, tp_world_size, original_placements):
    # TP is used, we need to gather the TP-shard params first
    tp_mesh_dim = tp_axis(original_placements)
    assert tp_mesh_dim is not None, "something wrong here"
    shard_dim = original_placements[tp_mesh_dim].dim

    output_tensors = [torch.empty_like(tensor) for _ in range(tp_world_size)]
    dist.all_gather(output_tensors, tensor, group=tp_group)
    return torch.cat(output_tensors, dim=shard_dim)

    # # below is another version using all_gather_into_tensor
    # local = tensor if shard_dim == 0 else tensor.movedim(shard_dim, 0).contiguous()

    # out_shape = list(local.shape)
    # out_shape[0] *= tp_world_size
    # full_flat = torch.empty(*out_shape, dtype=local.dtype, device=local.device)

    # dist.all_gather_into_tensor(full_flat, local, group=tp_group)

    # full = full_flat if shard_dim == 0 else full_flat.movedim(0, shard_dim)
    # return full.contiguous()


def calculate_shard_shape(shape, rank, world_size):
    full = shape[0]
    splits = torch.arange(full).chunk(world_size)
    if rank >= len(splits):
        dim0 = 0
    else:
        dim0 = len(splits[rank])

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
        extra_reduce_for_HSDP=False,
    ):
        self.need_to_calculate_norm = False
        self.norms_to_log: list[str] = list(NORM_FUNCTIONS.keys())
        self.norms_at_current_step = {}
        self.extra_reduce_for_HSDP = False
        self.log_parameters_types = True

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

        self.extra_reduce_for_HSDP = extra_reduce_for_HSDP

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
        self.parameters_to_groups = {}
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
                self.parameters_to_groups[id(param)] = group_idx

            if self.is_light and nesterov:
                raise RuntimeError(
                    "Nesterov momentum is not supported for Scion's light mode. "
                    "Please set nesterov=False."
                )

    def calculate_norm_at_next_step(self, norms_to_log: list[str]):
        self.need_to_calculate_norm = True
        self.norms_to_log = norms_to_log
        self.norms_at_current_step = {}

    def get_norms_at_current_step(self):
        return self.norms_at_current_step

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        embed_params, embed_param_names = [], []
        ddp_params, ddp_param_names = [], []
        fsdp_params, fsdp_param_names = [], []
        expert_params, expert_param_names = [], []

        for group_idx, group in enumerate(self.param_groups):
            # we should update self.groups_info here incase we have LR and momentum scheduler
            # We can also optionally do norm_factor and backend scheduler if we want to
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

            for p_name, p in zip(group["param_names"], group["params"]):
                norm_factor = group["norm_factor"]
                backend = group["backend"]
                is_embed_norm = norm_factor.startswith(
                    "embed"
                ) or norm_factor.startswith("unembed")

                if backend == "identity" and is_embed_norm:
                    # for these Row/Col-wise norm, there is no need to gather the gradient
                    embed_params.append(p)
                    embed_param_names.append(p_name)
                    continue

                param_type = get_param_type(p, self.fsdp_enabled, self.expert_enabled)
                if param_type == ParamType.DDP:
                    ddp_params.append(p)
                    ddp_param_names.append(p_name)
                elif param_type == ParamType.FSDP:
                    fsdp_params.append(p)
                    fsdp_param_names.append(p_name)
                elif param_type == ParamType.Expert:
                    expert_params.append(p)
                    expert_param_names.append(p_name)
                elif param_type == ParamType.Unknown:
                    logger.warning(
                        f"Unknown param type: {p_name}, the optimizer will skip this param"
                    )
                    continue
                else:
                    raise ValueError("param_type")

        # Sort fsdp_params and their names together
        fsdp_pairs = list(zip(fsdp_params, fsdp_param_names))
        fsdp_pairs.sort(key=lambda x: x[0].numel(), reverse=True)
        fsdp_params, fsdp_param_names = zip(*fsdp_pairs) if fsdp_pairs else ([], [])
        # Sort expert_params and their names together
        expert_pairs = list(zip(expert_params, expert_param_names))
        expert_pairs.sort(key=lambda x: (x[0].numel(), x[0].shape[1]), reverse=True)
        expert_params, expert_param_names = (
            zip(*expert_pairs) if expert_pairs else ([], [])
        )
        if self.log_parameters_types:
            # only log once
            logger.info(
                f"fsdp_params: {len(fsdp_params)} | expert_params: {len(expert_params)} | "
                f"ddp_params: {len(ddp_params)} | embed_params: {len(embed_params)}"
            )
            self.log_parameters_types = False

        """
        We could merge `embed_params` and `expert_params` into one list.
        The diff is, we are sure expert_params have bunch of 2D full-matrixs
        But we might need to gather the `embed_params` to 2D full-matrixs
        if we wanna to get the norm of the gradient.
        """
        # reset the flag for the next step
        need_to_calculate_norm = self.need_to_calculate_norm
        self.need_to_calculate_norm = False
        self.step_embedding(
            embed_params,
            embed_param_names,
            need_to_calculate_norm=need_to_calculate_norm,
        )
        self.step_experts(
            expert_params,
            expert_param_names,
            need_to_calculate_norm=need_to_calculate_norm,
        )
        self.step_ddp(
            ddp_params, ddp_param_names, need_to_calculate_norm=need_to_calculate_norm
        )
        self.step_fsdp(
            fsdp_params, fsdp_param_names, need_to_calculate_norm=need_to_calculate_norm
        )

        return loss

    @torch.no_grad()
    def lmo(
        self,
        g,
        eps,
        norm_factor,
        zeropower_backend,
        backend_steps,
    ):
        g = g.to_local() if isinstance(g, DTensor) else g

        # NB: make sure this function does not modify the grad inplace
        #     since it is also called during the log of gradients
        def _lmo_for_2d_tensor(g, need_transpose=False):
            g = g if not need_transpose else g.transpose(0, 1)
            g = zeropower_backends[zeropower_backend](g, steps=backend_steps, eps=eps)
            g = self.normalise_grad(g, norm_factor=norm_factor, eps=eps)
            return g if not need_transpose else g.transpose(0, 1)

        is_grouped_experts = g.ndim == 3

        if g.ndim == 2:
            g = _lmo_for_2d_tensor(g, need_transpose=is_grouped_experts)
        elif g.ndim == 3:
            if g.shape[0] > 0:
                # When world_size [fsdp x EP] > Total number of experts,
                # some ranks may have 0 experts that shape will be [0, d-in, d-out]
                # We should return the original grad here and **do not** do stack
                g = torch.stack(
                    [
                        _lmo_for_2d_tensor(g[i], need_transpose=is_grouped_experts)
                        for i in range(g.shape[0])
                    ],
                    dim=0,
                )
            else:
                pass
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

    def update_bucket_params(self, params, updates, start_idx, end_idx, tp_group=None):

        for idx_in_bucket in range(start_idx, end_idx):
            shift = idx_in_bucket - start_idx
            p = params[idx_in_bucket]
            u = updates[shift]
            lr, nesterov, momentum, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]

            if not self.is_unconstrained:
                # p.data.mul_(1 - lr)
                p.mul_(1 - lr)

            original_placements = p.placements
            tp_mesh_dim = tp_axis(original_placements)
            if isinstance(p, DTensor):
                if tp_group is None or tp_mesh_dim is None:
                    p.to_local().add_(u, alpha=-lr)
                else:
                    tp_rank = tp_group.rank()
                    tp_sharded_dim = original_placements[tp_mesh_dim].dim
                    chunk_size = p.to_local().shape[tp_sharded_dim]
                    start_offset = tp_rank * chunk_size

                    slicer = [slice(None)] * u.dim()
                    slicer[tp_sharded_dim] = slice(
                        start_offset, start_offset + chunk_size
                    )
                    u_sliced = u[slicer]
                    p.to_local().add_(u_sliced, alpha=-lr)
            else:
                p.add_(u, alpha=-lr)

            if momentum != 1 and self.is_light and p.grad is not None:
                p.grad.mul_(1 - momentum)

    def step_embedding(
        self,
        embed_params,
        embed_param_names,
        skip_update=False,
        need_to_calculate_norm=False,
        apply_on_weight=True,
    ):
        if len(embed_params) == 0:
            return {}

        dp_replicate_group = None
        if "dp_replicate" in self.world_mesh.mesh_dim_names:
            dp_replicate_group = self.world_mesh["dp_replicate"].get_group()

        norms_of_update, norms_of_weight, final_norms = [], [], {}
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        for param_idx in range(len(embed_params)):
            p = embed_params[param_idx]
            lr, nesterov, momentum, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )

            u = self.lmo(g, **param_kwargs)

            #########################################################
            # # As of we use norm for Embedding, maybe we should not do Reduce here
            # if (
            #     dp_replicate_group is not None
            #     and self.extra_reduce_for_HSDP
            #     and self.fsdp_enabled
            # ):
            #     dist.all_reduce(u, group=dp_replicate_group, op=dist.ReduceOp.AVG)
            #     dist.barrier(group=dp_replicate_group)

            if not skip_update:
                self.update_bucket_params([p], [u], 0, 1)

        if not need_to_calculate_norm:
            return {}

        # for the embedding, if we want to calculate the norm, we need to gather the gradient
        for param_idx in range(len(embed_params)):
            p, p_name = embed_params[param_idx], embed_param_names[param_idx]
            lr, nesterov, momentum, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            # this is important, *Do NOT* update buffer twice here
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=False, gather_to_local=True
            )
            """
            TODO(JSC): maybe we can improve this [?]
            Rather than Gather - LMO, we can do LMO - Gather such that we can avoid compute
            lmo twice [?]. though lmo of embedding is not expensive
            """
            u = self.lmo(g, **param_kwargs)

            if apply_on_weight and isinstance(p, DTensor):
                p = p.full_tensor()

            norm_need_transpose = "tok_embeddings" in p_name
            norms_of_update = calculate_norm(
                -lr * u, self.norms_to_log, transpose=norm_need_transpose
            )
            if apply_on_weight:
                norms_of_weight: dict = calculate_norm(
                    p, self.norms_to_log, transpose=norm_need_transpose
                )
            else:
                norms_of_weight = None

            # This should _not_ be an f-string since the variable names
            # will be interpolated later.
            embed_norm_key_template = "track_{task_name}_{norm_name}/{cleaned_p_name}"
            cleaned_p_name = remove_orig_mod_and_weight_for_p_name(p_name)
            for norm_name in self.norms_to_log:
                final_norms[
                    embed_norm_key_template.format(
                        task_name="update",
                        norm_name=norm_name,
                        cleaned_p_name=cleaned_p_name,
                    )
                ] = norms_of_update[norm_name]
                if apply_on_weight:
                    final_norms[
                        embed_norm_key_template.format(
                            task_name="param",
                            norm_name=norm_name,
                            cleaned_p_name=cleaned_p_name,
                        )
                    ] = norms_of_weight[norm_name]
        self.norms_at_current_step.update(final_norms)

    def step_experts(
        self,
        expert_params,
        expert_param_names,
        skip_update=False,
        need_to_calculate_norm=False,
        apply_on_weight=True,
    ):
        if len(expert_params) == 0:
            return {}

        norms_of_update, norms_of_weight, final_norms = {}, {}, {}
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        for param_idx in range(len(expert_params)):
            p = expert_params[param_idx]
            lr, nesterov, momentum, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )
            u = self.lmo(g, **param_kwargs)

            if not skip_update:
                self.update_bucket_params([p], [u], 0, 1)

            if need_to_calculate_norm:
                local_rank = dist.get_rank()
                world_size = dist.get_world_size()
                ep_per_rank = math.ceil(u.shape[0] / world_size)
                cleaned_p_name = remove_orig_mod_and_weight_for_p_name(
                    expert_param_names[param_idx]
                )
                # This should _not_ be an f-string since the variable
                # names will be interpolated later.
                moe_norm_key_template = (
                    "track_{task}_{norm_name}/ep_{actual_ep_idx}/{cleaned_p_name}"
                )
                assert u.ndim == 3
                for ep_idx in range(u.shape[0]):
                    actual_ep_idx = ep_idx + local_rank * ep_per_rank
                    update_norms = calculate_norm(
                        u[ep_idx], self.norms_to_log, transpose=True
                    )
                    # Template for MoE norm keys
                    norms_of_update.update(
                        {
                            moe_norm_key_template.format(
                                task="update",
                                norm_name=norm_name,
                                actual_ep_idx=actual_ep_idx,
                                cleaned_p_name=cleaned_p_name,
                            ): norm_value
                            for norm_name, norm_value in update_norms.items()
                        }
                    )

        # now, each rank has a dict of norms_of_update
        # we need to all-gather the norms_of_update
        final_norms = gather_and_merge(norms_of_update)
        self.norms_at_current_step.update(final_norms)

    def step_ddp(
        self,
        ddp_params,
        ddp_param_names,
        skip_update=False,
        need_to_calculate_norm=False,
        apply_on_weight=True,
    ):
        # we should only call this function on "DDP-only" case?
        if len(ddp_params) == 0:
            return {}

        world_size = self.world_mesh.size()
        rank = self.world_mesh.get_rank()

        # @ THIS IS A HACK
        bucket_size = world_size
        total_buckets = math.ceil(len(ddp_params) / bucket_size)

        device = ddp_params[0].device
        cast_dtype = self.communication_dtype
        zero_tensor = partial(torch.zeros, dtype=cast_dtype, device=device)

        norms_of_update, norms_of_weight, final_norms = [], [], {}
        padding_norms = {
            norm_name: torch.tensor(0.0, device=device)
            for norm_name in self.norms_to_log
        }
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        # for DDP, we need to first update the buffer
        for param_idx in range(len(ddp_params)):
            p = ddp_params[param_idx]
            _, nesterov, momentum, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )

        #  then we do scion stuff
        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(ddp_params))
            current_rank_idx = start_idx + rank
            if current_rank_idx < len(ddp_params):
                p = ddp_params[current_rank_idx]
                # Step 1: Get the gradient
                _, nesterov, momentum, param_kwargs = self.groups_info[
                    self.parameters_to_groups[id(p)]
                ]
                g = self.get_momentum_or_grad(
                    p, momentum, nesterov, update_buffer=False, gather_to_local=False
                )
            else:
                # To avoid idle stream, we pad the last rank
                p = ddp_params[end_idx - 1]
                g = zero_tensor(p.shape)
                _, nesterov, momentum, param_kwargs = self.groups_info[
                    self.parameters_to_groups[id(p)]
                ]

            # step 2: lmo
            u = self.lmo(g, **param_kwargs)

            if not skip_update:
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

            if need_to_calculate_norm:
                # so here, we already have update of each rank
                p = ddp_params[min(current_rank_idx, len(ddp_params) - 1)]
                lr, *_ = self.groups_info[self.parameters_to_groups[id(p)]]

                if current_rank_idx < end_idx:
                    norms_of_update.extend(
                        calculate_norm(-lr * u, self.norms_to_log).values()
                    )
                else:
                    norms_of_update.extend(padding_norms.values())
                if apply_on_weight:
                    if current_rank_idx < end_idx:
                        norms_of_weight.extend(
                            calculate_norm(p, self.norms_to_log).values()
                        )
                    else:
                        norms_of_weight.extend(padding_norms.values())

        if need_to_calculate_norm and len(norms_of_update) > 0:

            norms_tensor = torch.stack(norms_of_update).to(device=device).float()
            gathered_update_norms = torch.empty(
                world_size * norms_tensor.shape[0],
                dtype=norms_tensor.dtype,
                device=norms_tensor.device,
            )
            dist.barrier()
            dist.all_gather_into_tensor(gathered_update_norms, norms_tensor)
            if apply_on_weight:
                norms_tensor = torch.stack(norms_of_weight).to(device=device).float()
                gathered_weight_norms = torch.empty(
                    world_size * norms_tensor.shape[0],
                    dtype=norms_tensor.dtype,
                    device=norms_tensor.device,
                )
                dist.barrier()
                dist.all_gather_into_tensor(gathered_weight_norms, norms_tensor)

            if rank == 0:
                # This should _not_ be an f-string since the variable
                # names will be interpolated later.
                ddp_norm_key_template = "track_{task_name}_{norm_name}/{cleaned_p_name}"
                reordered_names = []
                # for param_idx in range(len(ddp_params)):
                #     cleaned_p_name = remove_orig_mod_and_weight_for_p_name(
                #         ddp_param_names[param_idx]
                #     )
                #     reordered_names.append(cleaned_p_name)
                for rank_idx in range(world_size):
                    # The inner loop jumps by world_size to get all params for a given rank
                    for param_idx in range(rank_idx, len(ddp_params), world_size):
                        cleaned_p_name = remove_orig_mod_and_weight_for_p_name(
                            ddp_param_names[param_idx]
                        )
                        reordered_names.append(cleaned_p_name)
                valid_norms = len(ddp_params) * len(self.norms_to_log)
                gathered_update_norms = gathered_update_norms[:valid_norms]
                count = 0
                for param_idx in range(len(ddp_params)):
                    cleaned_p_name = reordered_names[param_idx]
                    for norm_name in self.norms_to_log:
                        final_norms[
                            ddp_norm_key_template.format(
                                task_name="update",
                                norm_name=norm_name,
                                cleaned_p_name=cleaned_p_name,
                            )
                        ] = gathered_update_norms[count]
                        if apply_on_weight:
                            # gathered_weight_norms = gathered_weight_norms[:valid_norms]
                            final_norms[
                                ddp_norm_key_template.format(
                                    task_name="param",
                                    norm_name=norm_name,
                                    cleaned_p_name=cleaned_p_name,
                                )
                            ] = gathered_weight_norms[count]
                        count += 1

        self.norms_at_current_step.update(final_norms)

    def step_fsdp(
        self,
        fsdp_params,
        fsdp_param_names,
        skip_update=False,
        need_to_calculate_norm=False,
        apply_on_weight=True,
    ):
        if len(fsdp_params) == 0:
            return {}
        tp_group, dp_replicate_group = None, None
        """
        To make FSDP+DP works, we lets step_fsdp work on each dp_replicate separately.
        Hence, we only care about the world size inside the dp_replicate.
        """

        # due to the werid implementation of parallel_dims.py (upstream)
        # here we should use `dp_shard_cp` rather then `dp_shard` as of
        # CP is also part of the dp_shard
        fsdp_group = self.world_mesh["dp_shard_cp"].get_group()

        if "dp_replicate" in self.world_mesh.mesh_dim_names:
            dp_replicate_group = self.world_mesh["dp_replicate"].get_group()

        if "tp" in self.world_mesh.mesh_dim_names:
            tp_group = self.world_mesh["tp"].get_group()
            tp_world_size = dist.get_world_size(group=tp_group)

        world_size = dist.get_world_size(fsdp_group)
        rank = dist.get_rank(fsdp_group)

        # @ THIS IS A HACK
        bucket_size = world_size
        total_buckets = math.ceil(len(fsdp_params) / bucket_size)

        device = fsdp_params[0].device
        cast_dtype = self.communication_dtype
        zero_tensor = partial(torch.empty, dtype=cast_dtype, device=device)

        norms_of_update, norms_of_weight, final_norms = [], [], {}

        padding_norms = {
            norm_name: torch.tensor(0.0, device=device)
            for norm_name in self.norms_to_log
        }

        apply_on_weight = apply_on_weight and need_to_calculate_norm

        # Process each bucket
        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(fsdp_params))

            # Step 1: Prepare data for first all_to_all
            grads_send_list, send_shapes = [], []
            target_shape, param_kwargs = None, None

            for rank_idx in range(world_size):
                current_rank_idx = start_idx + rank_idx

                if current_rank_idx < len(fsdp_params):
                    p = fsdp_params[current_rank_idx]
                    _, nesterov, momentum, param_kwargs = self.groups_info[
                        self.parameters_to_groups[id(p)]
                    ]

                    g = self.get_momentum_or_grad(
                        p,
                        momentum,
                        nesterov,
                        update_buffer=True,
                        gather_to_local=False,
                    )

                    original_placements = g.placements
                    tp_mesh_dim = tp_axis(original_placements)
                    if tp_group is not None and tp_mesh_dim is not None:
                        # the reason we need `tp_mesh_dim` is we want a flexible solution
                        # that Attention go TP and MLP go EP
                        g = gather_tp_shard(
                            g.to_local(), tp_group, tp_world_size, original_placements
                        ).to(dtype=cast_dtype)
                    else:
                        g = g.to_local().to(dtype=cast_dtype)

                    # Save the shape info for this parameter
                    if rank == rank_idx:
                        target_shape = p.shape
                else:
                    # Use a dummy shape for parameters beyond our range
                    p = fsdp_params[end_idx - 1]
                    g = zero_tensor(p.to_local().shape)

                grads_send_list.append(g)
                send_shapes.append(g.shape)

            # Make sure target_shape is initialized
            # (trigger by the padding of the last ranks)
            if target_shape is None and end_idx > 0:
                target_shape = fsdp_params[end_idx - 1].shape
                param_kwargs = self.groups_info[
                    self.parameters_to_groups[id(fsdp_params[end_idx - 1])]
                ][-1]

            recv_shapes = [
                calculate_shard_shape(target_shape, rank_idx, world_size)
                for rank_idx in range(world_size)
            ]
            recv_list = [zero_tensor(shape) for shape in recv_shapes]
            # Step 3: First all_to_all - using ASYNC version
            dist.barrier()
            dist.all_to_all(recv_list, grads_send_list, group=fsdp_group)
            # Step 5: Concatenate received gradients along dimension 0 and perform NS5
            # All tensors in recv_list should have the same dimensions except for dim 0

            full_g = torch.cat(recv_list, dim=0)
            u = self.lmo(full_g, **param_kwargs)
            dist.barrier(group=fsdp_group)

            if dp_replicate_group is not None and self.extra_reduce_for_HSDP:
                dist.all_reduce(u, group=dp_replicate_group, op=dist.ReduceOp.AVG)
                dist.barrier(group=dp_replicate_group)
            # in case of FSDP+DP, we can do a All-Reduce here sync the grads
            if not skip_update:
                # Step 6: Split the processed tensor back for second all_to_all
                split_sizes = [shape[0] for shape in recv_shapes]

                grads_send_list = list(torch.split(u, split_sizes, dim=0))
                recv_list = [zero_tensor(shape) for shape in send_shapes]
                # Step 8: Second all_to_all - using ASYNC version
                dist.all_to_all(recv_list, grads_send_list, group=fsdp_group)
                del grads_send_list
                # Step 10: Update parameters using the results
                self.update_bucket_params(
                    fsdp_params,
                    recv_list,
                    start_idx,
                    end_idx,
                    tp_group=tp_group,
                )

            if need_to_calculate_norm:
                if start_idx + rank < end_idx:
                    lr, *_ = self.groups_info[
                        self.parameters_to_groups[id(fsdp_params[start_idx + rank])]
                    ]
                    norms = calculate_norm(-lr * u, self.norms_to_log)
                else:
                    norms = padding_norms
                norms_of_update.extend(norms.values())

                if apply_on_weight:
                    params_send_list = []
                    for rank_idx in range(world_size):
                        current_rank_idx = start_idx + rank_idx
                        if current_rank_idx < len(fsdp_params):
                            p = fsdp_params[current_rank_idx]
                        else:
                            p = fsdp_params[end_idx - 1]

                        # her is patch for FSDP+TP
                        original_placements = p.placements
                        tp_mesh_dim = tp_axis(original_placements)
                        if tp_group is not None and tp_mesh_dim is not None:
                            p = gather_tp_shard(
                                p.to_local(),
                                tp_group,
                                tp_world_size,
                                original_placements,
                            ).to(dtype=cast_dtype)
                        else:
                            p = p.to_local().to(dtype=cast_dtype)

                        params_send_list.append(p)

                    recv_list = [zero_tensor(shape) for shape in recv_shapes]

                    dist.barrier(group=fsdp_group)
                    dist.all_to_all(recv_list, params_send_list, group=fsdp_group)

                    full_weight = torch.cat(recv_list, dim=0)

                    if start_idx + rank < end_idx:
                        norms = calculate_norm(full_weight, self.norms_to_log)
                    else:
                        norms = padding_norms
                    norms_of_weight.extend(norms.values())

        # Below we need to all-gather the norms of update to rank-0
        if need_to_calculate_norm and len(norms_of_update) > 0:
            # Convert norms_of_update to a flat tensor for all-gather
            # Each rank has bucket_size * len(self.norms_to_log) norm values
            norms_tensor = torch.stack(norms_of_update).to(device=device).float()
            gathered_update_norms = torch.empty(
                world_size * norms_tensor.shape[0],
                dtype=norms_tensor.dtype,
                device=norms_tensor.device,
            )
            dist.barrier(group=fsdp_group)
            dist.all_gather_into_tensor(
                gathered_update_norms, norms_tensor, group=fsdp_group
            )

            if apply_on_weight:
                norms_tensor = torch.stack(norms_of_weight).to(device=device).float()
                gathered_weight_norms = torch.empty(
                    world_size * norms_tensor.shape[0],
                    dtype=norms_tensor.dtype,
                    device=norms_tensor.device,
                )
                dist.barrier(group=fsdp_group)
                dist.all_gather_into_tensor(
                    gathered_weight_norms, norms_tensor, group=fsdp_group
                )

            if rank == 0:
                # Only rank 0 processes the gathered data
                reordered_names = []
                # Loop through ranks, then through the parameters handled by each rank
                for rank_idx in range(world_size):
                    # The inner loop jumps by world_size to get all params for a given rank
                    for param_idx in range(rank_idx, len(fsdp_params), world_size):
                        cleaned_p_name = remove_orig_mod_and_weight_for_p_name(
                            fsdp_param_names[param_idx]
                        )
                        reordered_names.append(cleaned_p_name)

                valid_norms = len(fsdp_params) * len(self.norms_to_log)
                gathered_update_norms = gathered_update_norms[:valid_norms]

                count = 0
                # This should _not_ be an f-string since the variable
                # names will be interpolated later.
                fsdp_norm_key_template = (
                    "track_{task_name}_{norm_name}/{cleaned_p_name}"
                )

                for param_idx in range(len(fsdp_params)):
                    cleaned_p_name = reordered_names[param_idx]
                    for norm_name in self.norms_to_log:
                        final_norms[
                            fsdp_norm_key_template.format(
                                task_name="update",
                                norm_name=norm_name,
                                cleaned_p_name=cleaned_p_name,
                            )
                        ] = gathered_update_norms[count]

                        if apply_on_weight:
                            # gathered_weight_norms = gathered_weight_norms[:valid_norms]
                            final_norms[
                                fsdp_norm_key_template.format(
                                    task_name="param",
                                    norm_name=norm_name,
                                    cleaned_p_name=cleaned_p_name,
                                )
                            ] = gathered_weight_norms[count]

                        count += 1
            dist.barrier(group=fsdp_group)
        self.norms_at_current_step.update(final_norms)

    @torch.no_grad()
    def get_momentum_or_grad(
        self, p, momentum, nesterov, update_buffer=False, gather_to_local=False
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

        if gather_to_local and isinstance(g, DTensor):
            g = g.redistribute(placements=[Replicate()] * g.device_mesh.ndim).to_local()

        return g
