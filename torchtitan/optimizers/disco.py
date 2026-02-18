# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from enum import Enum

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from torch.profiler import record_function  # labels in PyTorch profiler

from torchtitan.tools.logging import logger
from .abstract_disco import AbstractDiSCO

from .norm_helper import calculate_norm
from .utils import remove_orig_mod_and_weight_for_p_name

__all__ = [
    "DiSCO",
]


# these variables are hardcoded for now, taken from torchtitan
CONST_NAME_OF_EMBEDDING = "tok_embeddings"

# Environment variables and default values
# DISCO_DEBUG_MODE = "0"
# DISCO_ENABLE_PERSISTENT_CACHE = "1"
# DISCO_FSDP_A2A_MODE = "once"


class ParamType(Enum):
    DDP = 0
    FSDP = 1
    Expert = 2


def get_param_type(p, fsdp_enabled, expert_enabled):
    """Classify parameter bucket type from static structure."""
    if p.numel() == 1:
        # treat scalars separately in _build_param_lists(); return DDP here by default
        return ParamType.DDP
    if p.ndim == 3 and (expert_enabled or fsdp_enabled):
        return ParamType.Expert
    if fsdp_enabled:
        return ParamType.FSDP
    return ParamType.DDP


def tp_axis(placements: tuple, tp_enabled: bool = False) -> int | None:
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

    # this is a special case, We do TP only
    if tp_enabled and len(placements) == 1:
        if isinstance(placements[0], Shard):
            return 0
    return None  # could not infer


def gather_tp_shard(tensor, tp_mesh, tp_world_size, original_placements):
    # TP is used, we need to gather the TP-shard params first
    tp_mesh_dim = tp_axis(original_placements, True)
    assert tp_mesh_dim is not None, "TP mesh dimension not found"
    shard_dim = original_placements[tp_mesh_dim].dim

    # output_tensors = [torch.empty_like(tensor) for _ in range(tp_world_size)]
    # dist.all_gather(output_tensors, tensor, group=tp_group)
    # return torch.cat(output_tensors, dim=shard_dim)
    return funcol.all_gather_tensor(tensor, gather_dim=shard_dim, group=tp_mesh)


def calculate_shard_shape(shape, rank, world_size):
    full = shape[0]
    splits = torch.arange(full).chunk(world_size)
    if rank >= len(splits):
        dim0 = 0
    else:
        dim0 = len(splits[rank])

    return (dim0, *shape[1:])


def parse_env_var():
    debug_mode = os.environ.get("DISCO_DEBUG_MODE", "0") == "1"
    persistent_cache_enabled = (
        os.environ.get("DISCO_ENABLE_PERSISTENT_CACHE", "1") == "1"
    )

    a2a_mode = os.environ.get("DISCO_FSDP_A2A_MODE", "once").strip().lower()
    if a2a_mode not in {"once", "bucket"}:
        raise ValueError(
            f"Unknown DISCO_FSDP_A2A_MODE={a2a_mode}. Supported: once, bucket"
        )

    env_vars = {
        "debug_mode": debug_mode,
        "persistent_cache_enabled": persistent_cache_enabled,
        "a2a_mode": a2a_mode,
    }
    return env_vars


class DiSCO(AbstractDiSCO):
    def __init__(
        self,
        params,
        is_light,
        weight_decay,
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
        experts_weights_layout="G-D_out-D_in",
    ):
        env_vars = parse_env_var()
        logger.info(f"[DiSCO] Environment variables: {env_vars}")
        debug_mode = env_vars["debug_mode"]
        self.persistent_cache_enabled = env_vars["persistent_cache_enabled"]
        self.fsdp_a2a_mode = env_vars["a2a_mode"]

        # Initialize base optimizer and common state
        self.log_parameters_types = True
        self.is_light = is_light

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            norm_factor=norm_factor if not debug_mode else "none",
            backend=backend if not debug_mode else "identity",
            backend_steps=backend_steps,
            splits_into=None,  # should be explicitly set in the extra_param_group_split_rules
            splits_dim=None,  # should be explicitly set in the extra_param_group_split_rules
        )
        assert is_light is False, "is_light must be False"

        is_unconstrained = weight_decay == 0

        self.parallel_dims = parallel_dims

        self.fsdp_enabled = parallel_dims.fsdp_enabled
        self.expert_enabled = parallel_dims.ep_enabled
        self.dp_replicate_enabled = parallel_dims.dp_replicate_enabled
        self.tp_enabled = parallel_dims.tp_enabled

        # this is used to ensure only the DP or FSDP rank 0 will have norms
        if self.dp_replicate_enabled or self.fsdp_enabled:
            self.is_dp_rank_0 = (
                parallel_dims.get_optional_mesh("loss").get_local_rank() == 0
            )
        else:
            # only PP (and/or) TP enabled
            self.is_dp_rank_0 = dist.get_rank() == 0

        assert experts_weights_layout in [
            "G-D_in-D_out",
            "G-D_out-D_in",
        ], f"Unknown experts weights layout: {experts_weights_layout}"
        self.experts_need_transpose = experts_weights_layout == "G-D_in-D_out"
        self.extra_reduce_for_HSDP = extra_reduce_for_HSDP

        logger.info(
            f"[DiSCO] "
            f"(is_light={self.is_light}, is_unconstrained={is_unconstrained}) "
            f"is enabled with world_mesh={self.parallel_dims.world_mesh} | fsdp_enabled={self.fsdp_enabled} | "
            f"EP={self.expert_enabled} | TP={self.tp_enabled} | DP={self.dp_replicate_enabled}"
        )

        super().__init__(params, defaults, is_light=is_light)
        # Register light-mode grad state hooks if needed
        self.setup_light_state_hooks()

        self.communication_dtype = communication_dtype
        self.groups_info = {}
        self.parameters_to_groups = {}
        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"] if not debug_mode else "none",
                "zeropower_backend": group["backend"] if not debug_mode else "identity",
                "backend_steps": group["backend_steps"],
                "splits_into": group["splits_into"],
                "splits_dim": group["splits_dim"],
            }
            self.groups_info[group_idx] = [lr, nesterov, momentum, wd, param_kwargs]
            for param in group["params"]:
                self.parameters_to_groups[id(param)] = group_idx

            logger.info(
                f"[DiSCO] group_idx: {group_idx} have {len(group['params'])} "
                f"weights || lr: {lr} || nesterov: {nesterov} || "
                f"momentum: {momentum} || wd: {wd} || {param_kwargs}"
            )

        # public caches
        self.scale_params, self.scale_param_names = [], []
        self.embed_params, self.embed_param_names = [], []
        self.ddp_params, self.ddp_param_names = [], []
        self.fsdp_params, self.fsdp_param_names = [], []
        self.expert_params, self.expert_param_names = [], []
        # build once now
        self._build_param_lists()

    def _build_param_lists(self):
        # clear
        self.scale_params.clear()
        self.scale_param_names.clear()
        self.embed_params.clear()
        self.embed_param_names.clear()
        self.ddp_params.clear()
        self.ddp_param_names.clear()
        self.fsdp_params.clear()
        self.fsdp_param_names.clear()
        self.expert_params.clear()
        self.expert_param_names.clear()

        # decide embedding per-group exactly like in step()
        # (backend == "identity" and norm_factor startswith embed/unembed)
        def _is_embed_group(g):
            nf, be = g["norm_factor"], g["backend"]
            return (be == "identity") and (
                nf.startswith("embed") or nf.startswith("unembed")
            )

        for group in self.param_groups:
            route_to_embed = _is_embed_group(group)
            for p_name, p in zip(group["param_names"], group["params"]):
                if not p.requires_grad:
                    # ignore the non-trainable parameters
                    continue

                # Initialize the momentum buffer if it's the first time.
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)

                # 1) scalar branch identical to step()
                if p.numel() == 1:
                    assert (
                        group["backend"] == "identity"
                    ), "scale params must use identity backend"
                    assert (
                        group["norm_factor"] == "sign"
                    ), "scale params must use sign norm factor"
                    self.scale_params.append(p)
                    self.scale_param_names.append(p_name)
                    continue
                # 2) embedding fast path identical to step() predicate
                if route_to_embed:
                    self.embed_params.append(p)
                    self.embed_param_names.append(p_name)
                    continue
                # 3) structural type without reading p.grad (init-time)
                ptype = get_param_type(p, self.fsdp_enabled, self.expert_enabled)
                if ptype == ParamType.DDP:
                    self.ddp_params.append(p)
                    self.ddp_param_names.append(p_name)
                elif ptype == ParamType.FSDP:
                    self.fsdp_params.append(p)
                    self.fsdp_param_names.append(p_name)
                elif ptype == ParamType.Expert:
                    self.expert_params.append(p)
                    self.expert_param_names.append(p_name)
                else:
                    # static classifier should not return Unknown
                    pass
        if self.ddp_params:
            pairs = list(zip(self.ddp_params, self.ddp_param_names))
            # sort big → small to reduce padding and make buckets well-conditioned
            pairs.sort(key=lambda x: x[0].numel(), reverse=True)

            # snake interleave across buckets to balance per-rank Phase-A compute
            dp_replicate_mesh = (
                self.parallel_dims.get_optional_mesh("dp_replicate")
                if self.dp_replicate_enabled
                else None
            )
            w = dp_replicate_mesh.size() if dp_replicate_mesh is not None else 1
            if w > 1:
                blocks = [pairs[i : i + w] for i in range(0, len(pairs), w)]
                for b, blk in enumerate(blocks):
                    if b % 2 == 1:
                        blk.reverse()
                pairs = [p for blk in blocks for p in blk]

            self.ddp_params, self.ddp_param_names = (
                (list(t) if pairs else [] for t in zip(*pairs)) if pairs else ([], [])
            )

        if self.fsdp_params:
            pairs = list(zip(self.fsdp_params, self.fsdp_param_names))
            pairs.sort(key=lambda x: x[0].numel(), reverse=True)
            self.fsdp_params, self.fsdp_param_names = list(zip(*pairs))
            self.fsdp_params, self.fsdp_param_names = (
                list(self.fsdp_params),
                list(self.fsdp_param_names),
            )

        if self.expert_params:
            pairs = list(zip(self.expert_params, self.expert_param_names))
            pairs.sort(key=lambda x: (x[0].numel(), x[0].shape[1]), reverse=True)
            self.expert_params, self.expert_param_names = list(zip(*pairs))
            self.expert_params, self.expert_param_names = (
                list(self.expert_params),
                list(self.expert_param_names),
            )

        if self.embed_params:
            pairs = list(zip(self.embed_params, self.embed_param_names))
            pairs.sort(key=lambda x: x[0].numel(), reverse=True)
            self.embed_params, self.embed_param_names = map(list, zip(*pairs))

        if self.log_parameters_types:
            logger.info(
                f"[DiSCO] fsdp_params: {len(self.fsdp_params)} | expert_params: {len(self.expert_params)} | "
                f"ddp_params: {len(self.ddp_params)} | embed_params: {len(self.embed_params)} | "
                f"scale_params: {len(self.scale_params)}"
            )
            self.log_parameters_types = False

        # Pre-compute structural metadata so hot paths do zero redundant work per step.
        self._precompute_runtime_caches()
        self._precompute_update_slicing()
        self._precompute_fsdp_metadata()
        self._precompute_experts_metadata()
        self._precompute_ddp_metadata()
        self._precompute_embed_metadata()
        self._precompute_momentum_bufs()

    # ------------------------------------------------------------------
    # Pre-compute helpers (called once from _build_param_lists at init)
    # ------------------------------------------------------------------

    def _precompute_runtime_caches(self):
        """
        Build direct per-parameter runtime caches to avoid repeated dict lookups and
        repeated DTensor -> local view construction in hot paths.
        """
        self._param_local_views: dict[int, torch.Tensor] = {}
        self._momentum_buffer_by_param_id: dict[int, torch.Tensor] = {}
        self._zero_scalar: torch.Tensor | None = None
        self._padding_norms: dict[str, torch.Tensor] | None = None

        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if isinstance(p, DTensor):
                    self._param_local_views[pid] = p.to_local()
                else:
                    self._param_local_views[pid] = p
                if p.requires_grad:
                    self._momentum_buffer_by_param_id[pid] = self.state[p][
                        "momentum_buffer"
                    ]

    def _precompute_update_slicing(self):
        """
        Pre-compute TP slicing info for every parameter so update_bucket_params
        avoids isinstance / placements / tp_axis checks per param each step.

        Stores: self._tp_slice_info[id(p)] = None  (no slicing needed)
                                             or (shard_dim, chunk_size, start, slicer_tuple,
                                                 local_shape)
        where local_shape is the expected shape of p.to_local(), used to decide
        whether slicing is needed without calling p.to_local() per step.
        """
        self._tp_slice_info: dict[int, tuple | None] = {}
        if not self.tp_enabled:
            return
        tp_mesh = self.parallel_dims.get_optional_mesh("tp")
        tp_rank = tp_mesh.get_local_rank()
        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, DTensor):
                    self._tp_slice_info[id(p)] = None
                    continue
                placements = p.placements
                tp_dim = tp_axis(placements, tp_enabled=True)
                if tp_dim is None:
                    self._tp_slice_info[id(p)] = None
                else:
                    shard_dim = placements[tp_dim].dim
                    p_local = self._get_param_local_view(p)
                    local_shape = p_local.shape
                    chunk_size = local_shape[shard_dim]
                    start = tp_rank * chunk_size
                    slicer: list = [slice(None)] * p.dim()
                    slicer[shard_dim] = slice(start, start + chunk_size)
                    self._tp_slice_info[id(p)] = (
                        shard_dim,
                        chunk_size,
                        start,
                        tuple(slicer),
                        local_shape,
                    )

    def _get_cached_zero_scalar(self, device):
        if self._zero_scalar is None:
            self._zero_scalar = torch.zeros((), device=device)
        return self._zero_scalar

    def _get_cached_padding_norms(self, device):
        if self._padding_norms is None:
            z = self._get_cached_zero_scalar(device)
            self._padding_norms = {k: z for k in self.norms_to_log}
        return self._padding_norms

    def _get_param_local_view(self, p):
        """
        Return cached local tensor view for `p`. Refresh lazily only if missing.
        """
        pid = id(p)
        p_local = self._param_local_views.get(pid)
        if p_local is None:
            p_local = p.to_local() if isinstance(p, DTensor) else p
            self._param_local_views[pid] = p_local
        return p_local

    def _precompute_embed_metadata(self):
        """
        Pre-compute embed extra-param groups, shape batching metadata, and update plan.
        CPU-only; called once from _build_param_lists at init.
        """
        self._embed_extra_indices: list[int] = []
        self._embed_extra_by_group: list[tuple[int, list[int], list]] = []
        self._embed_extra_shape_groups: dict[tuple, list[int]] = {}
        self._embed_shape_group_gidx: dict[tuple, int] = {}
        self._embed_shape_group_locals: dict[tuple, list[torch.Tensor]] = {}
        self._embed_update_plan: list[
            tuple[
                int,
                torch.device,
                torch.dtype,
                list[int],
                list[torch.Tensor],
                list[torch.Tensor],
            ]
        ] = []
        self._embed_step_workspace_cache: dict | None = None

        if not self.embed_params:
            return

        # Build update apply plan for ALL embed params (grouped by group/device/dtype).
        # Same structure as _expert_update_plan / _ddp_update_plan.
        # Fast-path extras (big_us_by_shape) skip updates[] and are applied separately,
        # so their updates[i] stays None — they are automatically skipped here.
        update_buckets: dict[tuple, dict] = {}
        for param_idx, p in enumerate(self.embed_params):
            group_idx = self.parameters_to_groups[id(p)]
            p_local = self._get_param_local_view(p)
            key = (group_idx, p_local.device, p_local.dtype)
            if key not in update_buckets:
                update_buckets[key] = {"indices": [], "params": [], "locals": []}
            b = update_buckets[key]
            b["indices"].append(param_idx)
            b["params"].append(p)
            b["locals"].append(p_local)
        self._embed_update_plan = [
            (
                group_idx,
                device,
                dtype,
                data["indices"],
                data["params"],
                data["locals"],
            )
            for (group_idx, device, dtype), data in update_buckets.items()
        ]

        if len(self.embed_params) <= 2:
            return

        # Build extra indices (sorted by numel already from embed_params sort above)
        for i in range(2, len(self.embed_params)):
            self._embed_extra_indices.append(i)

        # Group extras by group_idx for batched get_momentum_or_grad_list
        by_group: dict[int, dict] = {}
        for i in self._embed_extra_indices:
            p = self.embed_params[i]
            gidx = self.parameters_to_groups[id(p)]
            if gidx not in by_group:
                by_group[gidx] = {"indices": [], "params": []}
            by_group[gidx]["indices"].append(i)
            by_group[gidx]["params"].append(p)
        self._embed_extra_by_group = [
            (gidx, d["indices"], d["params"]) for gidx, d in by_group.items()
        ]

        # Group by local shape for batched LMO (fill big_g → single lmo() per group)
        for i in self._embed_extra_indices:
            p_local = self._get_param_local_view(self.embed_params[i])
            shape = tuple(p_local.shape)
            self._embed_extra_shape_groups.setdefault(shape, []).append(i)

        # Per shape group: pre-compute group_idx and param local views for fast apply.
        for shape, indices in self._embed_extra_shape_groups.items():
            gidx0 = self.parameters_to_groups[id(self.embed_params[indices[0]])]
            self._embed_shape_group_gidx[shape] = gidx0
            self._embed_shape_group_locals[shape] = [
                self._get_param_local_view(self.embed_params[i]) for i in indices
            ]

    def _precompute_fsdp_metadata(self):
        """
        Pre-compute structural FSDP bucket metadata (shapes/splits/offsets) so
        step_fsdp can switch between:
          - global 2-call all_to_all_single
          - bucketed all_to_all_single fallback
        without recomputing layout every step.
        """
        self._fsdp_total_buckets = 0
        self._fsdp_bucket_ranges: list[tuple[int, int]] = []
        self._fsdp_target_shapes: list[tuple] = []
        self._fsdp_recv_shapes: list[list[tuple]] = []
        self._fsdp_split_rows: list[list[int]] = []
        self._fsdp_param_kwargs_me: list[dict] = []
        self._fsdp_send_shapes: list[list[tuple]] = []
        self._fsdp_tp_gather_info: list[list] = []
        self._fsdp_bucket_params: list[list[torch.Tensor]] = []
        self._fsdp_bucket_group_indices: list[list[int]] = []
        self._fsdp_send_numels: list[list[int]] = []
        self._fsdp_recv_numels: list[list[int]] = []
        self._fsdp_bucket_send_total_elems: list[int] = []
        self._fsdp_bucket_recv_total_elems: list[int] = []
        self._fsdp_bucket_send_chunk_offsets: list[list[int]] = []
        self._fsdp_bucket_recv_chunk_offsets: list[list[int]] = []
        self._fsdp_global_send_offsets: list[list[int]] = []
        self._fsdp_global_recv_offsets: list[list[int]] = []
        self._fsdp_global_input_splits_elems: list[int] = []
        self._fsdp_global_output_splits_elems: list[int] = []
        self._fsdp_global_input_chunk_offsets: list[int] = []
        self._fsdp_global_output_chunk_offsets: list[int] = []
        self._fsdp_global_forward_send_elems = 0
        self._fsdp_global_forward_recv_elems = 0
        self._fsdp_global_reverse_send_elems = 0
        self._fsdp_global_reverse_recv_elems = 0
        self._fsdp_max_bucket_send_elems = 0
        self._fsdp_max_bucket_recv_elems = 0

        self._fsdp_grad_send_slot_bases: list[list[int]] = []
        self._fsdp_upd_send_slot_bases: list[list[int]] = []
        self._fsdp_u_row_offsets: list[list[int]] = []
        self._fsdp_u_flat_offsets: list[list[int]] = []
        self._fsdp_full_g_copy_plan: list[list[tuple[int, int, int]]] = []
        self._fsdp_pack_copy_plan: list[
            tuple[int, int, int, tuple, tuple | None, int]
        ] = []
        self._fsdp_upd_recv_param_plan: list[tuple[int, int, int, tuple]] = []
        self._fsdp_param_local_dtypes: list[torch.dtype] = []
        self._fsdp_uniform_param_dtype: torch.dtype | None = None
        self._fsdp_upd_recv_cast_offset_by_param: list[int] = []
        self._fsdp_upd_recv_cast_numels_by_dtype: dict[torch.dtype, int] = {}
        self._fsdp_upd_recv_cast_plan_by_dtype: dict[
            torch.dtype, list[tuple[int, int, int]]
        ] = {}
        # Persistent workspace: created on first step, reused every subsequent step.
        self._fsdp_once_workspace_cache: dict[str, object] | None = None

        if not self.fsdp_params:
            return

        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        world_size = fsdp_mesh.size()
        rank = fsdp_mesh.get_local_rank()
        self._fsdp_world_size = world_size  # cached for _prepare_fsdp_lmo()
        tp_mesh = (
            self.parallel_dims.get_optional_mesh("tp") if self.tp_enabled else None
        )

        total_buckets = math.ceil(len(self.fsdp_params) / world_size)
        self._fsdp_total_buckets = total_buckets

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * world_size
            end_idx = min(start_idx + world_size, len(self.fsdp_params))
            self._fsdp_bucket_ranges.append((start_idx, end_idx))

            # Determine target shape and kwargs for this rank's owned param.
            my_idx = start_idx + rank
            ref_idx = my_idx if my_idx < end_idx else end_idx - 1
            p_ref = self.fsdp_params[ref_idx]
            target_shape = p_ref.shape  # global (logical) shape
            g_idx = self.parameters_to_groups[id(p_ref)]
            # Structural kwargs (eps/norm_factor/backend/…) are fixed; safe to cache.
            param_kwargs_me = dict(self.groups_info[g_idx][-1])
            self._fsdp_target_shapes.append(target_shape)
            self._fsdp_param_kwargs_me.append(param_kwargs_me)

            # recv_shapes[r]: shape of the shard that rank r holds of my full param.
            recv_shapes = [
                calculate_shard_shape(target_shape, r, world_size)
                for r in range(world_size)
            ]
            self._fsdp_recv_shapes.append(recv_shapes)
            self._fsdp_split_rows.append([s[0] for s in recv_shapes])
            recv_numels = [int(math.prod(s)) for s in recv_shapes]
            recv_chunk_offsets = []
            off = 0
            for n in recv_numels:
                recv_chunk_offsets.append(off)
                off += n
            self._fsdp_bucket_recv_chunk_offsets.append(recv_chunk_offsets)

            send_chunk_offsets = []
            # send_shapes[i] + TP gather info per slot.
            send_shapes: list[tuple] = []
            send_numels: list[int] = []
            tp_infos: list = []
            bucket_params: list[torch.Tensor] = []
            bucket_group_indices: list[int] = []
            for i in range(world_size):
                p_idx = start_idx + i
                if p_idx < end_idx:
                    p = self.fsdp_params[p_idx]
                    g_idx_i = self.parameters_to_groups[id(p)]
                    tp_info = None
                    if isinstance(p, DTensor):
                        placements = p.placements
                        tp_dim = tp_axis(placements)
                        if tp_mesh and tp_dim is not None:
                            shard_dim_tp = placements[tp_dim].dim
                            tp_info = (shard_dim_tp, tp_mesh)
                            send_shape = calculate_shard_shape(
                                p.shape, rank, world_size
                            )
                        else:
                            send_shape = self._get_param_local_view(p).shape
                    else:
                        send_shape = p.shape
                else:
                    # Padding slot – mirror the last real param's local shape.
                    p = self.fsdp_params[end_idx - 1]
                    g_idx_i = self.parameters_to_groups[id(p)]
                    send_shape = (
                        self._get_param_local_view(p).shape
                        if isinstance(p, DTensor)
                        else p.shape
                    )
                    tp_info = None

                send_shapes.append(send_shape)
                send_numels.append(int(math.prod(send_shape)))
                tp_infos.append(tp_info)
                bucket_params.append(p)
                bucket_group_indices.append(g_idx_i)
            off = 0
            for n in send_numels:
                send_chunk_offsets.append(off)
                off += n

            self._fsdp_send_shapes.append(send_shapes)
            self._fsdp_tp_gather_info.append(tp_infos)
            self._fsdp_bucket_params.append(bucket_params)
            self._fsdp_bucket_group_indices.append(bucket_group_indices)
            self._fsdp_send_numels.append(send_numels)
            self._fsdp_recv_numels.append(recv_numels)
            self._fsdp_bucket_send_chunk_offsets.append(send_chunk_offsets)
            self._fsdp_bucket_send_total_elems.append(int(sum(send_numels)))
            self._fsdp_bucket_recv_total_elems.append(int(sum(recv_numels)))

        if total_buckets == 0:
            return

        running_send = [0] * world_size
        running_recv = [0] * world_size
        for bucket_idx in range(total_buckets):
            bucket_send_offsets = []
            bucket_recv_offsets = []
            for i in range(world_size):
                bucket_send_offsets.append(running_send[i])
                bucket_recv_offsets.append(running_recv[i])
                running_send[i] += self._fsdp_send_numels[bucket_idx][i]
                running_recv[i] += self._fsdp_recv_numels[bucket_idx][i]
            self._fsdp_global_send_offsets.append(bucket_send_offsets)
            self._fsdp_global_recv_offsets.append(bucket_recv_offsets)

        self._fsdp_global_input_splits_elems = list(running_send)
        self._fsdp_global_output_splits_elems = list(running_recv)

        input_chunk_offsets = []
        off = 0
        for n in self._fsdp_global_input_splits_elems:
            input_chunk_offsets.append(off)
            off += n
        self._fsdp_global_input_chunk_offsets = input_chunk_offsets

        output_chunk_offsets = []
        off = 0
        for n in self._fsdp_global_output_splits_elems:
            output_chunk_offsets.append(off)
            off += n
        self._fsdp_global_output_chunk_offsets = output_chunk_offsets

        self._fsdp_global_forward_send_elems = int(
            sum(self._fsdp_global_input_splits_elems)
        )
        self._fsdp_global_forward_recv_elems = int(
            sum(self._fsdp_global_output_splits_elems)
        )
        self._fsdp_global_reverse_send_elems = self._fsdp_global_forward_recv_elems
        self._fsdp_global_reverse_recv_elems = self._fsdp_global_forward_send_elems
        self._fsdp_max_bucket_send_elems = max(self._fsdp_bucket_send_total_elems)
        self._fsdp_max_bucket_recv_elems = max(self._fsdp_bucket_recv_total_elems)

        self._fsdp_grad_send_slot_bases = [
            [
                self._fsdp_global_input_chunk_offsets[i]
                + self._fsdp_global_send_offsets[b][i]
                for i in range(world_size)
            ]
            for b in range(total_buckets)
        ]
        self._fsdp_upd_send_slot_bases = [
            [
                self._fsdp_global_output_chunk_offsets[d]
                + self._fsdp_global_recv_offsets[b][d]
                for d in range(world_size)
            ]
            for b in range(total_buckets)
        ]
        self._fsdp_u_row_offsets = []
        self._fsdp_u_flat_offsets = []
        for b in range(total_buckets):
            row_off, offsets = 0, []
            for rows in self._fsdp_split_rows[b]:
                offsets.append(row_off)
                row_off += rows
            self._fsdp_u_row_offsets.append(offsets)
            target_shape = self._fsdp_target_shapes[b]
            cols = int(math.prod(target_shape[1:])) if len(target_shape) > 1 else 1
            self._fsdp_u_flat_offsets.append([int(r * cols) for r in offsets])

        self._fsdp_full_g_copy_plan = []
        for b in range(total_buckets):
            recv_numels = self._fsdp_recv_numels[b]
            recv_offsets = self._fsdp_global_recv_offsets[b]
            bucket_plan: list[tuple[int, int, int]] = []
            dst_offset = 0
            for src in range(world_size):
                numel = recv_numels[src]
                src_base = (
                    self._fsdp_global_output_chunk_offsets[src] + recv_offsets[src]
                )
                bucket_plan.append((src_base, dst_offset, numel))
                dst_offset += numel
            target_numel = int(math.prod(self._fsdp_target_shapes[b]))
            if dst_offset != target_numel:
                raise RuntimeError(
                    "FSDP full_g assembly total mismatch: "
                    f"bucket={b}, assembled={dst_offset}, expected={target_numel}."
                )
            self._fsdp_full_g_copy_plan.append(bucket_plan)

        self._fsdp_pack_copy_plan = []
        for b in range(total_buckets):
            start_idx, end_idx = self._fsdp_bucket_ranges[b]
            send_bases = self._fsdp_grad_send_slot_bases[b]
            send_numels = self._fsdp_send_numels[b]
            send_shapes = self._fsdp_send_shapes[b]
            tp_infos = self._fsdp_tp_gather_info[b]
            group_indices = self._fsdp_bucket_group_indices[b]
            for i in range(end_idx - start_idx):
                param_idx = start_idx + i
                self._fsdp_pack_copy_plan.append(
                    (
                        param_idx,
                        send_bases[i],
                        send_numels[i],
                        send_shapes[i],
                        tp_infos[i],
                        group_indices[i],
                    )
                )

        self._fsdp_upd_recv_param_plan = []
        for b in range(total_buckets):
            start_idx, end_idx = self._fsdp_bucket_ranges[b]
            send_numels = self._fsdp_send_numels[b]
            send_shapes = self._fsdp_send_shapes[b]
            send_offsets = self._fsdp_global_send_offsets[b]
            for i in range(end_idx - start_idx):
                param_idx = start_idx + i
                base = self._fsdp_global_input_chunk_offsets[i] + send_offsets[i]
                self._fsdp_upd_recv_param_plan.append(
                    (param_idx, base, send_numels[i], send_shapes[i])
                )

        self._fsdp_param_local_dtypes = [
            self._get_param_local_view(p).dtype for p in self.fsdp_params
        ]
        self._fsdp_upd_recv_cast_offset_by_param = [0] * len(self.fsdp_params)
        self._fsdp_upd_recv_cast_numels_by_dtype = {}
        self._fsdp_upd_recv_cast_plan_by_dtype = {}
        if self._fsdp_param_local_dtypes and all(
            dt == self._fsdp_param_local_dtypes[0]
            for dt in self._fsdp_param_local_dtypes
        ):
            self._fsdp_uniform_param_dtype = self._fsdp_param_local_dtypes[0]
        else:
            self._fsdp_uniform_param_dtype = None
            offsets_by_dtype: dict[torch.dtype, int] = {}
            for param_idx, base, numel, _ in self._fsdp_upd_recv_param_plan:
                dtype = self._fsdp_param_local_dtypes[param_idx]
                off = offsets_by_dtype.get(dtype, 0)
                self._fsdp_upd_recv_cast_offset_by_param[param_idx] = off
                self._fsdp_upd_recv_cast_plan_by_dtype.setdefault(dtype, []).append(
                    (base, numel, off)
                )
                offsets_by_dtype[dtype] = off + numel
            self._fsdp_upd_recv_cast_numels_by_dtype = offsets_by_dtype

        # Safety guard: all global_upd_send write ranges must be disjoint.
        update_ranges: list[tuple[int, int, int, int]] = []
        total_upd_send_elems = int(self._fsdp_global_reverse_send_elems)
        for b in range(total_buckets):
            for dst in range(world_size):
                start = int(self._fsdp_upd_send_slot_bases[b][dst])
                end = start + int(self._fsdp_recv_numels[b][dst])
                if not (0 <= start <= end <= total_upd_send_elems):
                    raise RuntimeError(
                        "FSDP global_upd_send range out of bounds: "
                        f"bucket={b}, dst={dst}, range=[{start}, {end}), "
                        f"capacity={total_upd_send_elems}."
                    )
                update_ranges.append((start, end, b, dst))

        update_ranges.sort(key=lambda x: (x[0], x[1]))
        prev_end = 0
        prev_meta: tuple[int, int] | None = None
        for start, end, b, dst in update_ranges:
            if prev_meta is not None and start < prev_end:
                pb, pd = prev_meta
                raise RuntimeError(
                    "FSDP global_upd_send write overlap detected: "
                    f"prev=(bucket={pb}, dst={pd}, end={prev_end}) "
                    f"curr=(bucket={b}, dst={dst}, start={start})."
                )
            prev_end = end
            prev_meta = (b, dst)

    def _precompute_experts_metadata(self):
        """
        Pre-compute structural expert-batch metadata so step_experts avoids
        per-step recomputation of blocks, ep_per_rank, kwargs, etc.
        """
        self._expert_blocks: list[tuple[int, int]] = []
        self._expert_ep_per_rank: int = 0
        self._expert_kinds_of_norms: int = 0
        self._expert_transpose: bool = False
        self._expert_kwargs_per_block: list[dict | None] = []
        self._expert_block_group_idx: list[int | None] = []
        self._expert_big_g_specs: list[
            tuple[int, int, int, torch.dtype, torch.device] | None
        ] = []
        self._expert_step_workspace_cache: dict[str, object] | None = None
        self._expert_update_plan: list[
            tuple[
                int,
                torch.device,
                torch.dtype,
                list[int],
                list[torch.Tensor],
                list[torch.Tensor],
            ]
        ] = []

        if not self.expert_params:
            return

        total = len(self.expert_params)
        L = total // 3
        assert total == 3 * L, f"Expected 3*L expert params, got {total}"

        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        world_size = fsdp_mesh.size()

        # Keep the original expert split (2 blocks). Stream pool can still be larger.
        self._expert_blocks = [(0, L), (L, 3 * L)]
        self._expert_ep_per_rank = math.ceil(
            self.expert_params[0].shape[0] / world_size
        )
        self._expert_kinds_of_norms = len(self.norms_to_log)
        self._expert_transpose = self.experts_need_transpose

        ep_per_rank = self._expert_ep_per_rank
        for block_idx, (start, end) in enumerate(self._expert_blocks):
            block_params = self.expert_params[start:end]
            if block_params:
                g_idx = self.parameters_to_groups[id(block_params[0])]
                kwargs = dict(self.groups_info[g_idx][-1])
                self._expert_block_group_idx.append(g_idx)
                K = end - start
                p0 = block_params[0]
                loc = self._get_param_local_view(p0)  # Tensor [ep_per_rank, A, B]
                self._expert_big_g_specs.append(
                    (
                        K * ep_per_rank,
                        int(loc.shape[1]),
                        int(loc.shape[2]),
                        loc.dtype,
                        loc.device,
                    )
                )
            else:
                kwargs = None
                self._expert_block_group_idx.append(None)
                self._expert_big_g_specs.append(None)
            self._expert_kwargs_per_block.append(kwargs)

        # Precompute expert update apply plan (grouped by group/device/dtype).
        update_buckets: dict[
            tuple[int, torch.device, torch.dtype], dict[str, list]
        ] = {}
        for param_idx, p in enumerate(self.expert_params):
            group_idx = self.parameters_to_groups[id(p)]
            p_local = self._get_param_local_view(p)
            key = (group_idx, p_local.device, p_local.dtype)
            if key not in update_buckets:
                update_buckets[key] = {"indices": [], "params": [], "locals": []}
            b = update_buckets[key]
            b["indices"].append(param_idx)
            b["params"].append(p)
            b["locals"].append(p_local)
        self._expert_update_plan = [
            (
                group_idx,
                device,
                dtype,
                data["indices"],
                data["params"],
                data["locals"],
            )
            for (group_idx, device, dtype), data in update_buckets.items()
        ]

    def _precompute_ddp_metadata(self):
        """
        Pre-compute structural DDP metadata so step_ddp avoids per-step
        bucket math, repeated param->group dict lookups, and runtime
        update/logging planning overhead.
        """
        self._ddp_total_buckets: int = 0
        self._ddp_world_size: int = 1
        self._ddp_rank: int = 0
        self._ddp_owned_indices: list[int] = []
        self._ddp_owned_indices_by_group: list[
            tuple[int, list[int], list[torch.Tensor]]
        ] = []
        self._ddp_rank_owned_param_indices: list[list[int]] = []
        self._ddp_rank_owned_numels: list[int] = []
        self._ddp_max_rank_owned_numel: int = 0
        self._ddp_owned_pack_offsets: list[int] = []
        self._ddp_unpack_offsets_by_rank: list[dict[int, int]] = []
        self._ddp_param_group_idx: list[int] = []
        self._ddp_tp_slice_info_by_idx: list[tuple | None] = []
        self._ddp_tp_gather_info_by_idx: list[tuple | None] = []
        self._ddp_local_shape_by_idx: list[tuple[int, ...]] = []
        self._ddp_full_shape_by_idx: list[tuple[int, ...]] = []
        self._ddp_param_local_shapes: list[tuple[int, ...]] = []  # alias for apply path
        self._ddp_cast_indices_by_dtype: dict[torch.dtype, list[int]] = {}
        self._ddp_lmo_non_tp_precast_indices_by_src_dtype: dict[
            torch.dtype, list[int]
        ] = {}
        self._ddp_owner_rank_by_param: list[int] = []
        self._ddp_owner_bucket_by_param: list[int] = []
        self._ddp_clean_param_names: list[str] = []
        self._ddp_update_plan: list[
            tuple[
                int,
                torch.device,
                torch.dtype,
                list[int],
                list[torch.Tensor],
                list[torch.Tensor],
            ]
        ] = []
        self._ddp_step_workspace_cache: dict[str, object] | None = None

        if not self.ddp_params:
            return

        # TODO(jsc): For now, we force MoE training is with FSDP.
        # SO no MoE training with DDP.
        invalid_ddp_params: list[str] = []
        for p, p_name in zip(self.ddp_params, self.ddp_param_names):
            if p.ndim > 2:
                invalid_ddp_params.append(
                    f"{p_name} shape={tuple(p.shape)} ndim={p.ndim}"
                )
        if invalid_ddp_params:
            raise ValueError(
                "DDP norm path only supports parameters with ndim <= 2. "
                "Move higher-rank tensors to a non-DDP norm path. "
                f"Offending params: {', '.join(invalid_ddp_params)}"
            )

        dp_replicate_mesh = (
            self.parallel_dims.get_optional_mesh("dp_replicate")
            if self.dp_replicate_enabled
            else None
        )
        world_size = dp_replicate_mesh.size() if dp_replicate_mesh else 1
        rank = dp_replicate_mesh.get_local_rank() if dp_replicate_mesh else 0
        self._ddp_world_size = world_size
        self._ddp_rank = rank

        n_params = len(self.ddp_params)
        bucket_size = world_size
        total_buckets = (
            math.ceil(n_params / bucket_size) if world_size > 1 else n_params
        )
        self._ddp_total_buckets = total_buckets

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, n_params)
            my_idx = start_idx + rank
            if my_idx < end_idx:
                self._ddp_owned_indices.append(my_idx)

        self._ddp_rank_owned_param_indices = [
            [i for i in range(n_params) if i % world_size == r]
            for r in range(world_size)
        ]
        self._ddp_rank_owned_numels = [
            sum(int(self.ddp_params[i].numel()) for i in indices)
            for indices in self._ddp_rank_owned_param_indices
        ]
        self._ddp_max_rank_owned_numel = (
            max(self._ddp_rank_owned_numels) if self._ddp_rank_owned_numels else 0
        )

        offset = 0
        for param_idx in self._ddp_owned_indices:
            self._ddp_owned_pack_offsets.append(offset)
            offset += int(self.ddp_params[param_idx].numel())

        for r in range(world_size):
            rank_offsets: dict[int, int] = {}
            offset = 0
            for param_idx in self._ddp_rank_owned_param_indices[r]:
                rank_offsets[param_idx] = offset
                offset += int(self.ddp_params[param_idx].numel())
            self._ddp_unpack_offsets_by_rank.append(rank_offsets)

        update_buckets: dict[
            tuple[int, torch.device, torch.dtype], dict[str, list]
        ] = {}
        owned_by_group: dict[int, dict[str, list]] = {}
        tp_slice_info = self._tp_slice_info if self.tp_enabled else {}
        comm_dtype = self.communication_dtype

        for param_idx, p in enumerate(self.ddp_params):
            group_idx = self.parameters_to_groups[id(p)]
            self._ddp_param_group_idx.append(group_idx)

            p_local = self._get_param_local_view(p)
            local_shape = tuple(p_local.shape)
            full_shape = tuple(p.shape)
            self._ddp_local_shape_by_idx.append(local_shape)
            self._ddp_full_shape_by_idx.append(full_shape)
            self._ddp_param_local_shapes.append(local_shape)
            self._ddp_tp_slice_info_by_idx.append(tp_slice_info.get(id(p)))
            if self.tp_enabled and isinstance(p, DTensor):
                placements = p.placements
                self._ddp_tp_gather_info_by_idx.append(
                    placements
                    if tp_axis(placements, tp_enabled=True) is not None
                    else None
                )
            else:
                self._ddp_tp_gather_info_by_idx.append(None)

            if p_local.dtype != comm_dtype:
                self._ddp_cast_indices_by_dtype.setdefault(p_local.dtype, []).append(
                    param_idx
                )

            key = (group_idx, p_local.device, p_local.dtype)
            if key not in update_buckets:
                update_buckets[key] = {"indices": [], "params": [], "locals": []}
            b = update_buckets[key]
            b["indices"].append(param_idx)
            b["params"].append(p)
            b["locals"].append(p_local)

        for idx in self._ddp_owned_indices:
            group_idx = self._ddp_param_group_idx[idx]
            if group_idx not in owned_by_group:
                owned_by_group[group_idx] = {"indices": [], "params": []}
            owned_by_group[group_idx]["indices"].append(idx)
            owned_by_group[group_idx]["params"].append(self.ddp_params[idx])

            p_local = self._get_param_local_view(self.ddp_params[idx])
            tp_info = self._ddp_tp_gather_info_by_idx[idx]
            if tp_info is None and p_local.dtype != comm_dtype:
                self._ddp_lmo_non_tp_precast_indices_by_src_dtype.setdefault(
                    p_local.dtype, []
                ).append(idx)

        self._ddp_owned_indices_by_group = [
            (group_idx, data["indices"], data["params"])
            for group_idx, data in owned_by_group.items()
        ]

        self._ddp_owner_rank_by_param = [i % world_size for i in range(n_params)]
        self._ddp_owner_bucket_by_param = [i // world_size for i in range(n_params)]
        self._ddp_clean_param_names = [
            remove_orig_mod_and_weight_for_p_name(pn) for pn in self.ddp_param_names
        ]
        self._ddp_update_plan = [
            (
                group_idx,
                device,
                dtype,
                data["indices"],
                data["params"],
                data["locals"],
            )
            for (group_idx, device, dtype), data in update_buckets.items()
        ]

    def _create_embed_step_workspace(self, device: torch.device) -> dict:
        """
        Lazy CUDA alloc for embed step buffers. Step-wise by default;
        persistent when persistent_cache_enabled is True.
        """
        if (
            self.persistent_cache_enabled
            and self._embed_step_workspace_cache is not None
            and self._embed_step_workspace_cache["device"] == device
        ):
            return self._embed_step_workspace_cache

        n_params = len(self.embed_params)

        # Pre-alloc fill buffers for shape groups with ≥2 params (like expert big_g).
        # Views pre-created with unbind (ONE aten::unbind at init, not per step).
        batched_g_bufs: dict[tuple, torch.Tensor] = {}
        buf_views_by_shape: dict[tuple, list[torch.Tensor]] = {}
        for shape, indices in self._embed_extra_shape_groups.items():
            if len(indices) < 2:
                continue
            N = len(indices)
            p0 = self.embed_params[indices[0]]
            dtype = self._get_param_local_view(p0).dtype
            buf = torch.empty(N, *shape, dtype=dtype, device=device)
            batched_g_bufs[shape] = buf
            buf_views_by_shape[shape] = list(buf.unbind(0))  # ONE aten::unbind at init

        # Per-extra-param float32 scratch for Phase 2 norm (avoids -lr*u temp alloc)
        norm_scratch: list[torch.Tensor | None] = [None] * n_params
        for i in self._embed_extra_indices:
            p_local = self._get_param_local_view(self.embed_params[i])
            norm_scratch[i] = torch.empty(
                p_local.shape, dtype=torch.float32, device=device
            )

        ws = {
            "device": device,
            "n_params": n_params,
            "batched_g_bufs": batched_g_bufs,
            "buf_views_by_shape": buf_views_by_shape,
            "norm_scratch": norm_scratch,
            "updates": [None] * n_params,
            "effective_grads": [None] * n_params,
        }
        if self.persistent_cache_enabled:
            self._embed_step_workspace_cache = ws
        return ws

    def _create_ddp_step_workspace(
        self, cast_dtype: torch.dtype, device: torch.device
    ) -> dict[str, object]:
        n_params = len(self.ddp_params)
        norm_k = len(self.norms_to_log)
        if (
            self._ddp_step_workspace_cache is not None
            and self.persistent_cache_enabled
            and self._ddp_step_workspace_cache.get("device") == device
            and self._ddp_step_workspace_cache.get("cast_dtype") == cast_dtype
            and self._ddp_step_workspace_cache.get("n_params") == n_params
            and self._ddp_step_workspace_cache.get("norm_k") == norm_k
            and self._ddp_step_workspace_cache.get("world_size") == self._ddp_world_size
            and self._ddp_step_workspace_cache.get("rank") == self._ddp_rank
        ):
            return self._ddp_step_workspace_cache

        global_update_bufs: list[torch.Tensor] = [
            torch.empty(p.shape, dtype=cast_dtype, device=device)
            for p in self.ddp_params
        ]
        zero_by_shape: dict[tuple[tuple[int, ...], torch.dtype], torch.Tensor] = {}
        norm_scratch: list[torch.Tensor | None] = [None] * n_params

        for p in self.ddp_params:
            shape = tuple(p.shape)
            key = (shape, cast_dtype)
            if key not in zero_by_shape:
                zero_by_shape[key] = torch.zeros(shape, dtype=cast_dtype, device=device)

        for param_idx in self._ddp_owned_indices:
            p = self.ddp_params[param_idx]
            norm_scratch[param_idx] = torch.empty(
                p.shape, dtype=cast_dtype, device=device
            )

        if cast_dtype == self.communication_dtype:
            cast_indices_by_dtype = self._ddp_cast_indices_by_dtype
        else:
            cast_indices_by_dtype: dict[torch.dtype, list[int]] = {}
            for param_idx, p in enumerate(self.ddp_params):
                p_local = self._get_param_local_view(p)
                if p_local.dtype != cast_dtype:
                    cast_indices_by_dtype.setdefault(p_local.dtype, []).append(
                        param_idx
                    )

        cast_dst_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for dtype, indices in cast_indices_by_dtype.items():
            cast_dst_by_dtype[dtype] = [
                torch.empty(
                    self._ddp_param_local_shapes[param_idx], dtype=dtype, device=device
                )
                for param_idx in indices
            ]

        lmo_inputs_by_idx: list[torch.Tensor | None] = [None] * n_params
        tp_local_cast_bufs_by_idx: list[torch.Tensor | None] = [None] * n_params
        lmo_non_tp_cast_bufs_by_idx: list[torch.Tensor | None] = [None] * n_params
        tp_lmo_precast_indices: set[int] = {
            idx
            for indices in self._ddp_cast_indices_by_dtype.values()
            for idx in indices
            if self._ddp_tp_gather_info_by_idx[idx] is not None
        }
        for param_idx in self._ddp_owned_indices:
            if (
                param_idx in tp_lmo_precast_indices
                and self._ddp_tp_gather_info_by_idx[param_idx] is not None
            ):
                tp_local_cast_bufs_by_idx[param_idx] = torch.empty(
                    self._ddp_local_shape_by_idx[param_idx],
                    dtype=cast_dtype,
                    device=device,
                )
        for indices in self._ddp_lmo_non_tp_precast_indices_by_src_dtype.values():
            for param_idx in indices:
                lmo_non_tp_cast_bufs_by_idx[param_idx] = torch.empty(
                    self._ddp_full_shape_by_idx[param_idx],
                    dtype=cast_dtype,
                    device=device,
                )

        max_owned_numel = self._ddp_max_rank_owned_numel
        ddp_flat_send = torch.empty(max_owned_numel, dtype=cast_dtype, device=device)
        ddp_flat_recv_chunks = [
            torch.empty(max_owned_numel, dtype=cast_dtype, device=device)
            for _ in range(self._ddp_world_size)
        ]
        ddp_pack_dst_views: list[torch.Tensor] = []
        for j, param_idx in enumerate(self._ddp_owned_indices):
            offset = self._ddp_owned_pack_offsets[j]
            numel = int(self.ddp_params[param_idx].numel())
            ddp_pack_dst_views.append(
                ddp_flat_send.narrow(0, offset, numel).view(
                    self._ddp_full_shape_by_idx[param_idx]
                )
            )

        ddp_unpack_src_views: list[torch.Tensor] = []
        ddp_unpack_dst_views: list[torch.Tensor] = []
        for r in range(self._ddp_world_size):
            rank_offsets = self._ddp_unpack_offsets_by_rank[r]
            for param_idx in self._ddp_rank_owned_param_indices[r]:
                offset = rank_offsets[param_idx]
                numel = int(self.ddp_params[param_idx].numel())
                ddp_unpack_src_views.append(
                    ddp_flat_recv_chunks[r]
                    .narrow(0, offset, numel)
                    .view(self._ddp_full_shape_by_idx[param_idx])
                )
                ddp_unpack_dst_views.append(global_update_bufs[param_idx])

        total_norm_elems = self._ddp_total_buckets * norm_k
        upd_norm_local_flat = torch.empty(
            total_norm_elems, dtype=torch.float32, device=device
        )

        ws = {
            "device": device,
            "cast_dtype": cast_dtype,
            "n_params": n_params,
            "norm_k": norm_k,
            "world_size": self._ddp_world_size,
            "rank": self._ddp_rank,
            "global_update_bufs": global_update_bufs,
            "zero_by_shape": zero_by_shape,
            "norm_scratch": norm_scratch,
            "global_updates": [None] * n_params,
            "apply_updates": [None] * n_params,
            "cast_indices_by_dtype": cast_indices_by_dtype,
            "cast_dst_by_dtype": cast_dst_by_dtype,
            "lmo_inputs_by_idx": lmo_inputs_by_idx,
            "tp_local_cast_bufs_by_idx": tp_local_cast_bufs_by_idx,
            "lmo_non_tp_cast_bufs_by_idx": lmo_non_tp_cast_bufs_by_idx,
            "ddp_flat_send": ddp_flat_send,
            "ddp_flat_recv_chunks": ddp_flat_recv_chunks,
            "ddp_pack_dst_views": ddp_pack_dst_views,
            "ddp_unpack_src_views": ddp_unpack_src_views,
            "ddp_unpack_dst_views": ddp_unpack_dst_views,
            "upd_norm_local_flat": upd_norm_local_flat,
            "w_norm_local_flat": None,
        }
        if self.persistent_cache_enabled:
            self._ddp_step_workspace_cache = ws
        return ws

    def _create_expert_step_workspace(self) -> dict[str, object]:
        if (
            self.persistent_cache_enabled
            and self._expert_step_workspace_cache is not None
        ):
            return self._expert_step_workspace_cache

        def _allocate_expert_big_g_bufs() -> list[torch.Tensor | None]:
            bufs: list[torch.Tensor | None] = []
            for spec in self._expert_big_g_specs:
                if spec is None:
                    bufs.append(None)
                    continue
                n, a, b, dtype, device = spec
                bufs.append(torch.zeros((n, a, b), dtype=dtype, device=device))
            return bufs

        def _create_expert_big_g_views(
            big_g_bufs: list[torch.Tensor | None] | None,
        ) -> list[list[torch.Tensor]]:
            if not big_g_bufs:
                return []

            ep_per_rank = self._expert_ep_per_rank
            views_by_block: list[list[torch.Tensor]] = []
            for block_idx, (start, end) in enumerate(self._expert_blocks):
                big_g = big_g_bufs[block_idx]
                if big_g is None:
                    views_by_block.append([])
                    continue
                k_count = end - start
                block_views = []
                for k in range(k_count):
                    base = k * ep_per_rank
                    block_views.append(big_g[base : base + ep_per_rank])
                views_by_block.append(block_views)
            return views_by_block

        big_g_bufs = _allocate_expert_big_g_bufs()
        dst_views_by_block = _create_expert_big_g_views(big_g_bufs)
        ws: dict[str, object] = {
            "big_g_bufs": big_g_bufs,
            "dst_views_by_block": dst_views_by_block,
        }
        if self.persistent_cache_enabled:
            self._expert_step_workspace_cache = ws
        return ws

    def _allocate_fsdp_once_workspace(
        self, cast_dtype: torch.dtype, device: torch.device
    ) -> dict[str, object]:
        workspace: dict[str, object] = {
            "global_grad_send": torch.zeros(
                self._fsdp_global_forward_send_elems, dtype=cast_dtype, device=device
            ),
            "global_grad_recv": torch.empty(
                self._fsdp_global_forward_recv_elems, dtype=cast_dtype, device=device
            ),
            "full_g_bufs": [
                torch.empty(shape, dtype=cast_dtype, device=device)
                for shape in self._fsdp_target_shapes
            ],
        }
        """
        used for FSDP once mode, (all_to_all for all parameters)
        """
        workspace["global_upd_send"] = torch.empty(
            self._fsdp_global_reverse_send_elems, dtype=cast_dtype, device=device
        )
        workspace["global_upd_recv"] = torch.empty(
            self._fsdp_global_reverse_recv_elems, dtype=cast_dtype, device=device
        )
        return workspace

    def _allocate_fsdp_bucket_workspace(
        self,
        cast_dtype: torch.dtype,
        device: torch.device,
        apply_on_weight: bool,
        skip_update: bool,
    ) -> dict[str, torch.Tensor | None]:
        """
        used for FSDP bucketed mode, (all_to_all for each bucket)
        """
        workspace: dict[str, torch.Tensor | None] = {
            "grad_send_flat": torch.empty(
                self._fsdp_max_bucket_send_elems, dtype=cast_dtype, device=device
            ),
            "grad_recv_flat": torch.empty(
                self._fsdp_max_bucket_recv_elems, dtype=cast_dtype, device=device
            ),
            "upd_send_flat": None,
            "upd_recv_flat": None,
            "param_send_flat": None,
            "param_recv_flat": None,
        }
        if not skip_update:
            workspace["upd_send_flat"] = torch.empty(
                self._fsdp_max_bucket_recv_elems, dtype=cast_dtype, device=device
            )
            workspace["upd_recv_flat"] = torch.empty(
                self._fsdp_max_bucket_send_elems, dtype=cast_dtype, device=device
            )
        if apply_on_weight:
            workspace["param_send_flat"] = torch.empty(
                self._fsdp_max_bucket_send_elems, dtype=cast_dtype, device=device
            )
            workspace["param_recv_flat"] = torch.empty(
                self._fsdp_max_bucket_recv_elems, dtype=cast_dtype, device=device
            )
        return workspace

    def _create_fsdp_step_workspace(
        self, cast_dtype: torch.dtype, device: torch.device, skip_update: bool
    ) -> dict[str, object]:
        # Persistent workspace: created once, reused every step.
        # All view objects are pre-created once and remain valid because the underlying
        # workspace tensors never move (same Python objects across all steps).

        if self._fsdp_once_workspace_cache and self.persistent_cache_enabled:
            return self._fsdp_once_workspace_cache

        ws = self._allocate_fsdp_once_workspace(cast_dtype, device)

        grad_send = ws["global_grad_send"]
        grad_recv = ws["global_grad_recv"]
        full_g_bufs_ws = ws["full_g_bufs"]
        world_size = self._fsdp_world_size
        total_buckets = self._fsdp_total_buckets

        # Issue 1: dst views into global_grad_send for _prepare_fsdp_lmo.
        ws["pack_dst_views"] = [
            grad_send.narrow(0, base, numel).view(send_shape)
            for (_, base, numel, send_shape, _, _) in self._fsdp_pack_copy_plan
        ]

        # Issue 3a (flat): all full_g assembly in one _foreach_copy_ before LMO loop.
        full_g_prep_src_flat: list[torch.Tensor] = []
        full_g_prep_dst_flat: list[torch.Tensor] = []
        for b in range(total_buckets):
            full_g_flat = full_g_bufs_ws[b].view(-1)
            for src_base, dst_base, numel in self._fsdp_full_g_copy_plan[b]:
                full_g_prep_src_flat.append(grad_recv.narrow(0, src_base, numel))
                full_g_prep_dst_flat.append(full_g_flat.narrow(0, dst_base, numel))
        ws["full_g_prep_src_flat"] = full_g_prep_src_flat
        ws["full_g_prep_dst_flat"] = full_g_prep_dst_flat

        if skip_update:
            ws["global_upd_recv_cast"] = None
            ws["global_upd_recv_cast_by_dtype"] = None
            if self.persistent_cache_enabled:
                self._fsdp_once_workspace_cache = ws
            return ws

        upd_send = ws["global_upd_send"]
        upd_recv = ws["global_upd_recv"]

        # Issue 3b (flat): persistent u buffers + pre-created flat src views and
        # pre-created flat dst views into global_upd_send.
        ws["u_bufs"] = [
            torch.empty(shape, dtype=cast_dtype, device=device)
            for shape in self._fsdp_target_shapes
        ]
        u_src_views_flat: list[torch.Tensor] = []
        upd_pack_dst_views_flat: list[torch.Tensor] = []
        for b in range(total_buckets):
            recv_numels = self._fsdp_recv_numels[b]
            u_flat = ws["u_bufs"][b].view(-1)
            flat_offsets = self._fsdp_u_flat_offsets[b]
            for d in range(world_size):
                u_src_views_flat.append(
                    u_flat.narrow(0, flat_offsets[d], recv_numels[d])
                )
                upd_pack_dst_views_flat.append(
                    upd_send.narrow(
                        0,
                        self._fsdp_upd_send_slot_bases[b][d],
                        recv_numels[d],
                    )
                )
        ws["u_src_views_flat"] = u_src_views_flat
        ws["upd_pack_dst_views_flat"] = upd_pack_dst_views_flat
        if len(u_src_views_flat) != len(upd_pack_dst_views_flat):
            raise RuntimeError(
                "FSDP once-mode pack metadata mismatch: "
                f"src_views={len(u_src_views_flat)}, "
                f"dst_views={len(upd_pack_dst_views_flat)}."
            )

        target_dtype = self._fsdp_uniform_param_dtype
        if target_dtype is not None and target_dtype != cast_dtype:
            ws["global_upd_recv_cast"] = torch.empty(
                self._fsdp_global_reverse_recv_elems, dtype=target_dtype, device=device
            )
        else:
            ws["global_upd_recv_cast"] = None

        if target_dtype is None and self._fsdp_upd_recv_cast_numels_by_dtype:
            cast_by_dtype: dict[torch.dtype, torch.Tensor] = {}
            for dtype, numel in self._fsdp_upd_recv_cast_numels_by_dtype.items():
                if dtype == cast_dtype:
                    continue
                cast_by_dtype[dtype] = torch.empty(numel, dtype=dtype, device=device)
            ws["global_upd_recv_cast_by_dtype"] = (
                cast_by_dtype if cast_by_dtype else None
            )
        else:
            ws["global_upd_recv_cast_by_dtype"] = None

        # Issue 2a: src+dst views for per-dtype cast of global_upd_recv.
        cast_by_dtype_ws = ws.get("global_upd_recv_cast_by_dtype") or {}
        cast_src_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        cast_dst_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for dtype, cast_plan in self._fsdp_upd_recv_cast_plan_by_dtype.items():
            cast_flat = cast_by_dtype_ws.get(dtype)
            if cast_flat is None:
                continue
            cast_src_by_dtype[dtype] = [
                upd_recv.narrow(0, base, numel) for (base, numel, _) in cast_plan
            ]
            cast_dst_by_dtype[dtype] = [
                cast_flat.narrow(0, off, numel) for (_, numel, off) in cast_plan
            ]
        ws["cast_src_by_dtype"] = cast_src_by_dtype
        ws["cast_dst_by_dtype"] = cast_dst_by_dtype

        # Issue 2b: param views into update_source for global_updates extraction.
        cast_recv_uniform = ws.get("global_upd_recv_cast")
        update_source = cast_recv_uniform if cast_recv_uniform is not None else upd_recv
        upd_recv_views: list[torch.Tensor] = []
        for param_idx, base, numel, send_shape in self._fsdp_upd_recv_param_plan:
            if cast_by_dtype_ws:
                dtype = self._fsdp_param_local_dtypes[param_idx]
                cf = cast_by_dtype_ws.get(dtype)
                if cf is not None:
                    off = self._fsdp_upd_recv_cast_offset_by_param[param_idx]
                    upd_recv_views.append(cf.narrow(0, off, numel).view(send_shape))
                    continue
            upd_recv_views.append(update_source.narrow(0, base, numel).view(send_shape))
        ws["upd_recv_views"] = upd_recv_views

        if self.persistent_cache_enabled:
            self._fsdp_once_workspace_cache = ws
        return ws

    def _precompute_momentum_bufs(self):
        """
        Pre-build per-group momentum buffer lists so prepare_gradients_and_momentum
        avoids state dict lookups per param each step.
        """
        # List of (group_idx, buf_list, param_list). We keep all trainable params
        # so dynamic momentum schedules (e.g. 0.0 -> 0.9) are handled correctly.
        self._momentum_state: list[tuple[int, list, list]] = []
        for group_idx, group in enumerate(self.param_groups):
            bufs: list[torch.Tensor] = []
            params: list[torch.Tensor] = []
            for p in group["params"]:
                if p.requires_grad:
                    bufs.append(self._momentum_buffer_by_param_id[id(p)])
                    params.append(p)
            if bufs:
                self._momentum_state.append((group_idx, bufs, params))

        # Cache for per-step execution plan keyed by current momentum schedule.
        self._momentum_plan_key: tuple[float, ...] | None = None
        self._momentum_execution_plan: list[tuple[float, list, list]] = []

    @record_function("disco.prepare_ddp_lmo")
    def _prepare_ddp_lmo(
        self, ddp_params, workspace, tp_mesh=None, tp_world_size=1
    ) -> list[torch.Tensor | None]:
        """
        Prepare per-owned LMO inputs for DDP:
          1) grouped get_momentum_or_grad_list by group_idx
          2) cast to communication dtype in batched foreach copies
          3) TP gather on cast dtype for TP-sharded tensors
        """
        cast_dtype: torch.dtype = workspace["cast_dtype"]
        n_params = len(ddp_params)
        lmo_inputs: list[torch.Tensor | None] = workspace["lmo_inputs_by_idx"]
        if len(lmo_inputs) != n_params:
            raise ValueError("DDP LMO input workspace size mismatch.")

        for param_idx in self._ddp_owned_indices:
            lmo_inputs[param_idx] = None

        # 1) grouped effective-grad fetch
        for group_idx, owned_indices, params_bucket in self._ddp_owned_indices_by_group:
            _, nesterov, momentum, _, _ = self.groups_info[group_idx]
            grads = self.get_momentum_or_grad_list(
                params_bucket,
                momentum,
                nesterov,
                gather_to_local=False,
                to_local=False,
            )
            for j, param_idx in enumerate(owned_indices):
                lmo_inputs[param_idx] = grads[j]

        # 2) dtype cast staging
        non_tp_src_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        non_tp_idx_by_dtype: dict[torch.dtype, list[int]] = {}
        tp_src_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        tp_idx_by_dtype: dict[torch.dtype, list[int]] = {}
        tp_local_after_cast: dict[int, torch.Tensor] = {}

        for param_idx in self._ddp_owned_indices:
            g = lmo_inputs[param_idx]
            if g is None:
                continue

            tp_gather_info = (
                self._ddp_tp_gather_info_by_idx[param_idx]
                if tp_mesh is not None
                else None
            )
            if tp_gather_info is not None:
                g_local = g.to_local() if isinstance(g, DTensor) else g
                if g_local.dtype != cast_dtype:
                    tp_src_by_dtype.setdefault(g_local.dtype, []).append(g_local)
                    tp_idx_by_dtype.setdefault(g_local.dtype, []).append(param_idx)
                else:
                    tp_local_after_cast[param_idx] = g_local
                continue

            g_local = g.to_local() if isinstance(g, DTensor) else g
            if g_local.dtype != cast_dtype:
                non_tp_src_by_dtype.setdefault(g_local.dtype, []).append(g_local)
                non_tp_idx_by_dtype.setdefault(g_local.dtype, []).append(param_idx)
            else:
                lmo_inputs[param_idx] = g_local

        for src_dtype, src_views in non_tp_src_by_dtype.items():
            idxs = non_tp_idx_by_dtype[src_dtype]
            dst_views = [workspace["lmo_non_tp_cast_bufs_by_idx"][i] for i in idxs]
            if any(dst is None for dst in dst_views):
                raise ValueError(
                    "Missing non-TP cast buffers for DDP pre-LMO cast path."
                )
            torch._foreach_copy_(dst_views, src_views)
            for j, param_idx in enumerate(idxs):
                lmo_inputs[param_idx] = dst_views[j]

        if tp_mesh is not None:
            for src_dtype, src_views in tp_src_by_dtype.items():
                idxs = tp_idx_by_dtype[src_dtype]
                dst_views = [workspace["tp_local_cast_bufs_by_idx"][i] for i in idxs]
                if any(dst is None for dst in dst_views):
                    raise ValueError(
                        "Missing TP local cast buffers for DDP pre-LMO cast path."
                    )
                torch._foreach_copy_(dst_views, src_views)
                for j, param_idx in enumerate(idxs):
                    tp_local_after_cast[param_idx] = dst_views[j]

            for param_idx in self._ddp_owned_indices:
                if self._ddp_tp_gather_info_by_idx[param_idx] is None:
                    continue
                g_local = tp_local_after_cast.get(param_idx)
                if g_local is None:
                    lmo_inputs[param_idx] = None
                    continue
                lmo_inputs[param_idx] = gather_tp_shard(
                    g_local,
                    tp_mesh,
                    tp_world_size,
                    self._ddp_tp_gather_info_by_idx[param_idx],
                )

        return lmo_inputs

    @record_function("disco.prepare_experts_lmo_per_block")
    def _prepare_experts_lmo_per_block(
        self,
        expert_params,
        block_idx: int,
        big_g: torch.Tensor | None,
        dst_views_pre: list[torch.Tensor] | None = None,
    ) -> bool:
        start, end = self._expert_blocks[block_idx]
        block_params = expert_params[start:end]
        if not block_params or big_g is None:
            return False
        if dst_views_pre is None:
            raise ValueError(
                "dst_views_pre must be provided for expert fast fill path."
            )

        group_idx = self._expert_block_group_idx[block_idx]
        _, nesterov, momentum, _, _ = self.groups_info[group_idx]

        block_effective_grads = self.get_momentum_or_grad_list(
            block_params,
            momentum,
            nesterov,
            to_local=True,
        )

        dst_views: list[torch.Tensor] = []
        src_views: list[torch.Tensor] = []
        any_grad = False
        for k, g_local in enumerate(block_effective_grads):
            if g_local is None:
                continue
            assert (
                g_local.ndim == 3
            ), "Batching path assumes MoE expert weights are 3-D."
            if g_local.shape[0] == 0:
                continue
            if k >= len(dst_views_pre):
                raise ValueError(
                    "Expert destination views are shorter than block parameter list."
                )
            dst_views.append(dst_views_pre[k])
            src_views.append(g_local)
            any_grad = True

        if dst_views:
            torch._foreach_copy_(dst_views, src_views)
        return any_grad

    @record_function("disco.prepare_experts_lmo")
    def _prepare_experts_lmo(
        self, expert_params, big_g_bufs, dst_views_by_block=None
    ) -> list[bool]:
        """Fill per-step expert big_g buffers from expert param gradients/momentum."""
        if big_g_bufs is None:
            big_g_bufs = self._allocate_expert_big_g_bufs()
        if dst_views_by_block is None:
            dst_views_by_block = self._create_expert_big_g_views(big_g_bufs)

        self._expert_any_grad: list[bool] = [False] * len(self._expert_blocks)
        for block_idx, (start, end) in enumerate(self._expert_blocks):
            block_params = expert_params[start:end]
            if not block_params:
                continue
            big_g = big_g_bufs[block_idx]
            if big_g is not None:
                big_g.zero_()
            block_dst_views = (
                dst_views_by_block[block_idx]
                if dst_views_by_block is not None
                and block_idx < len(dst_views_by_block)
                else None
            )
            self._expert_any_grad[block_idx] = self._prepare_experts_lmo_per_block(
                expert_params,
                block_idx,
                big_g,
                block_dst_views,
            )
        return self._expert_any_grad

    @record_function("disco.prepare_fsdp_lmo")
    def _prepare_fsdp_lmo(
        self,
        global_grad_send: torch.Tensor,
        pack_dst_views: list | None = None,
    ):
        """Fill step-local global_grad_send for all FSDP buckets."""
        dst_active: list[torch.Tensor] = []
        src_active: list[torch.Tensor] = []
        for i, (
            param_idx,
            base,
            numel,
            send_shape,
            tp_info,
            group_idx,
        ) in enumerate(self._fsdp_pack_copy_plan):
            p = self.fsdp_params[param_idx]
            g = self._get_effective_grad_by_group(p, group_idx, param_idx)
            g_local = self._maybe_unpack_dtensor(g, tp_info)
            dst_active.append(
                pack_dst_views[i]
                if pack_dst_views is not None
                else global_grad_send.narrow(0, base, numel).view(send_shape)
            )
            src_active.append(g_local)
        if dst_active:
            torch._foreach_copy_(dst_active, src_active)

    @record_function("disco.prepare_ddp_apply_updates")
    def _prepare_ddp_apply_updates(
        self, global_updates, workspace, tp_mesh=None
    ) -> list[torch.Tensor | None]:
        """
        Prepare update tensors for DDP apply:
          1) optional TP slice to local shape
          2) dtype cast via batched foreach_copy_ into preallocated typed buffers
        """
        apply_updates: list[torch.Tensor | None] = workspace["apply_updates"]
        n_params = len(self.ddp_params)
        if len(apply_updates) != n_params:
            raise ValueError("DDP apply workspace size mismatch.")

        # Stage 1: gather source views (with optional TP slice) into apply_updates.
        for param_idx in range(n_params):
            u = global_updates[param_idx]
            if u is None:
                apply_updates[param_idx] = None
                continue

            if tp_mesh is not None:
                tp_info = self._ddp_tp_slice_info_by_idx[param_idx]
                if tp_info is not None:
                    _, _, _, slicer, local_shape = tp_info
                    if tuple(u.shape) != tuple(local_shape):
                        u = u[slicer]

            apply_updates[param_idx] = u

        # Stage 2: batched cast by destination dtype (FSDP-style cast phase).
        cast_indices_by_dtype: dict[torch.dtype, list[int]] = workspace[
            "cast_indices_by_dtype"
        ]
        cast_dst_by_dtype: dict[torch.dtype, list[torch.Tensor]] = workspace[
            "cast_dst_by_dtype"
        ]
        for dtype, indices in cast_indices_by_dtype.items():
            if not indices:
                continue
            src_views: list[torch.Tensor] = []
            dst_views_active: list[torch.Tensor] = []
            active_indices: list[int] = []
            dst_views_all = cast_dst_by_dtype[dtype]
            for j, param_idx in enumerate(indices):
                src = apply_updates[param_idx]
                if src is None:
                    continue
                src_views.append(src)
                dst_views_active.append(dst_views_all[j])
                active_indices.append(param_idx)
            if not active_indices:
                continue
            torch._foreach_copy_(dst_views_active, src_views)
            for j, param_idx in enumerate(active_indices):
                apply_updates[param_idx] = dst_views_active[j]

        return apply_updates

    @record_function("disco.update_embed_params_fast")
    def _update_embed_params_fast(
        self,
        updates: list,
        big_us_by_shape: dict,
    ):
        """Apply embed param updates using pre-computed update plan.

        Fast-path extras (all grads present) are applied directly from
        big_us_by_shape: for each shape group, big_u.unbind(0) gives N views
        and _foreach_add_ applies them in one CUDA dispatch.

        Canonicals [0, 1] and partial-fallback extras are applied from
        updates[] via _embed_update_plan (same structure as expert / DDP plan).
        Fast-path extras have updates[i]=None and are skipped automatically.
        """
        # 1. Fast-path extras: big_u per shape group → _foreach_add_ (no updates[] lookup)
        for shape, big_u in big_us_by_shape.items():
            gidx = self._embed_shape_group_gidx[shape]
            lr, _, _, wd, _ = self.groups_info[gidx]
            p_locals = self._embed_shape_group_locals[shape]
            if wd != 0.0:
                torch._foreach_mul_(p_locals, 1.0 - wd * lr)
            torch._foreach_add_(p_locals, list(big_u.unbind(0)), alpha=-lr)

        # 2. Canonicals + partial fallback extras: read from updates[] via plan
        for (
            group_idx,
            _device,
            _dtype,
            param_indices,
            params_bucket,
            locals_bucket,
        ) in self._embed_update_plan:
            lr, _, _, wd, _ = self.groups_info[group_idx]

            active_locals: list[torch.Tensor] = []
            active_updates: list[torch.Tensor] = []
            for j, param_idx in enumerate(param_indices):
                u = updates[param_idx]
                if u is None:
                    continue  # fast-path extra already applied, or no grad

                p = params_bucket[j]
                p_local = locals_bucket[j]
                if p_local.shape != u.shape:
                    if isinstance(p, DTensor):
                        p_local = p.to_local()
                        self._param_local_views[id(p)] = p_local
                        locals_bucket[j] = p_local
                    if p_local.shape != u.shape:
                        raise ValueError(
                            "Shape mismatch between embed parameter shard and update."
                        )

                if u.dtype != p_local.dtype:
                    u = u.to(p_local.dtype)

                active_locals.append(p_local)
                active_updates.append(u)

            if not active_locals:
                continue

            if wd != 0.0:
                torch._foreach_mul_(active_locals, 1.0 - wd * lr)
            torch._foreach_add_(active_locals, active_updates, alpha=-lr)

    @record_function("disco.update_ddp_params_fast")
    def _update_ddp_params_fast(self, apply_updates):
        if len(apply_updates) != len(self.ddp_params):
            raise ValueError("DDP updates length mismatch with parameter list.")

        for (
            group_idx,
            _device,
            _dtype,
            param_indices,
            params_bucket,
            locals_bucket,
        ) in self._ddp_update_plan:
            lr, _, _, wd, _ = self.groups_info[group_idx]

            active_locals: list[torch.Tensor] = []
            active_updates: list[torch.Tensor] = []
            for j, param_idx in enumerate(param_indices):
                u = apply_updates[param_idx]
                if u is None:
                    continue

                p = params_bucket[j]
                p_local = locals_bucket[j]
                if p_local.shape != u.shape:
                    if isinstance(p, DTensor):
                        p_local = p.to_local()
                        self._param_local_views[id(p)] = p_local
                        locals_bucket[j] = p_local
                        self._ddp_param_local_shapes[param_idx] = tuple(p_local.shape)
                    if p_local.shape != u.shape:
                        raise ValueError(
                            "Shape mismatch between DDP parameter shard and update."
                        )

                if u.dtype != p_local.dtype:
                    raise ValueError(
                        "Unexpected DDP update dtype mismatch after cast-prep: "
                        f"update={u.dtype}, param_local={p_local.dtype}."
                    )

                active_locals.append(p_local)
                active_updates.append(u)

            if not active_locals:
                continue

            if wd != 0.0:
                torch._foreach_mul_(active_locals, 1.0 - wd * lr)
            torch._foreach_add_(active_locals, active_updates, alpha=-lr)

    @record_function("disco.update_expert_params_fast")
    def _update_expert_params_fast(self, expert_params, all_updates):
        if len(all_updates) != len(expert_params):
            raise ValueError(
                "Expert updates length mismatch with expert parameter list."
            )

        for (
            group_idx,
            _device,
            _dtype,
            param_indices,
            params_bucket,
            locals_bucket,
        ) in self._expert_update_plan:
            lr, _, _, wd, _ = self.groups_info[group_idx]

            active_locals: list[torch.Tensor] = []
            active_updates: list[torch.Tensor] = []
            for j, param_idx in enumerate(param_indices):
                u = all_updates[param_idx]
                if u is None:
                    continue

                p = params_bucket[j]
                p_local = locals_bucket[j]
                if p_local.shape != u.shape:
                    if isinstance(p, DTensor):
                        p_local = p.to_local()
                        self._param_local_views[id(p)] = p_local
                        locals_bucket[j] = p_local
                    if p_local.shape != u.shape:
                        raise ValueError(
                            "Shape mismatch between expert parameter shard and update."
                        )

                if u.dtype != p_local.dtype:
                    u = u.to(p_local.dtype)

                active_locals.append(p_local)
                active_updates.append(u)

            if not active_locals:
                continue

            if wd != 0.0:
                torch._foreach_mul_(active_locals, 1.0 - wd * lr)
            torch._foreach_add_(active_locals, active_updates, alpha=-lr)

    def _refresh_momentum_execution_plan(self):
        """
        Rebuild foreach execution plan only when per-group momentum values change.
        Plan buckets tensors by (device, dtype, momentum) to maximize foreach fusion.
        """
        if not self._momentum_state:
            self._momentum_execution_plan = []
            self._momentum_plan_key = tuple()
            return

        plan_key = tuple(
            float(self.param_groups[group_idx]["momentum"])
            for group_idx, _, _ in self._momentum_state
        )
        if plan_key == self._momentum_plan_key:
            return

        buckets: dict[tuple[torch.device, torch.dtype, float], dict[str, list]] = {}
        for (group_idx, bufs, params), m in zip(self._momentum_state, plan_key):
            if not (0.0 < m < 1.0):
                continue
            for buf, p in zip(bufs, params):
                key = (buf.device, buf.dtype, m)
                if key not in buckets:
                    buckets[key] = {"bufs": [], "params": []}
                buckets[key]["bufs"].append(buf)
                buckets[key]["params"].append(p)

        self._momentum_execution_plan = [
            (m, data["bufs"], data["params"]) for (_, _, m), data in buckets.items()
        ]
        self._momentum_plan_key = plan_key

    def _maybe_unpack_dtensor(self, x, tp_info):
        if isinstance(x, DTensor):
            x_local = x.to_local()
            if tp_info is not None:
                _, tp_world_size, tp_mesh = tp_info
                x_local = gather_tp_shard(x_local, tp_mesh, tp_world_size, x.placements)
            return x_local
        return x

    def _get_effective_grad_by_group(self, p, group_idx, param_idx):
        """
        Fast path for effective grad retrieval in FSDP packing.
        Avoids state dict lookups and the generic helper call overhead.
        """
        g = p.grad
        if not p.requires_grad:
            p_name = (
                self.fsdp_param_names[param_idx]
                if param_idx < len(self.fsdp_param_names)
                else f"param_idx={param_idx}"
            )
            raise RuntimeError(
                f"[DiSCO] {p_name}. Does not require grad, why do you put it in the param_groups?"
            )

        if g is None:
            p_name = (
                self.fsdp_param_names[param_idx]
                if param_idx < len(self.fsdp_param_names)
                else f"param_idx={param_idx}"
            )
            raise RuntimeError(
                "[DiSCO] FSDP grad pack encountered missing gradient: "
                f"{p_name}. This optimizer expects all FSDP params to have grads."
            )

        _, nesterov, momentum, _, _ = self.groups_info[group_idx]
        use_momentum = (not self.is_light) and (0.0 < momentum < 1.0)
        if not use_momentum:
            return g

        buf = self._momentum_buffer_by_param_id.get(id(p))
        if buf is None:
            raise ValueError(
                "Momentum buffer missing; ensure pre-pass ran before FSDP packing."
            )
        if not nesterov:
            return buf
        return torch.lerp(buf, g, momentum)

    @record_function("disco.step")
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # only refresh schedulables each step (unchanged behaviour)
        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"],
                "zeropower_backend": group["backend"],
                "backend_steps": group["backend_steps"],
                "splits_into": group["splits_into"],
                "splits_dim": group["splits_dim"],
            }
            self.groups_info[group_idx] = [lr, nesterov, momentum, wd, param_kwargs]

        self.prepare_gradients_and_momentum()

        fsdp_workspace = None
        if self.fsdp_params and self.fsdp_a2a_mode == "once":
            fsdp_workspace = self._create_fsdp_step_workspace(
                cast_dtype=self.communication_dtype,
                device=self.fsdp_params[0].device,
                skip_update=False,
            )

        expert_workspace = (
            self._create_expert_step_workspace() if self.expert_params else None
        )
        ddp_workspace = None
        if self.ddp_params:
            ddp_workspace = (
                self._create_ddp_step_workspace(
                    cast_dtype=self.communication_dtype,
                    device=self.ddp_params[0].device,
                )
                if self.ddp_params
                else None
            )

        embed_workspace = (
            self._create_embed_step_workspace(device=self.embed_params[0].device)
            if self.embed_params
            else None
        )

        if self.embed_params:
            self.step_embedding(
                self.embed_params, self.embed_param_names, embed_workspace
            )

        # Expert LMO launches are coordinated inside step_experts.
        if self.expert_params:
            self.step_experts(
                self.expert_params,
                self.expert_param_names,
                workspace=expert_workspace,
            )

        if self.fsdp_params:
            self.step_fsdp(
                self.fsdp_params,
                self.fsdp_param_names,
                workspace=fsdp_workspace,
            )

        if self.ddp_params:
            self.step_ddp(
                self.ddp_params,
                self.ddp_param_names,
                workspace=ddp_workspace,
            )

        if self.scale_params:
            self.step_scalar(self.scale_params, self.scale_param_names)

        self.need_to_calculate_norm = False
        return loss

    @record_function("disco.step_scalar")
    @torch.compile()
    def step_scalar(
        self,
        scalar_params,
        scalar_param_names,
        skip_update=False,
        apply_on_weight=True,
    ):
        """
        We hardcode the update for scalar parameters to be the `sign` of the gradient.
        """
        if not scalar_params:
            return

        updates = []
        for p in scalar_params:
            _, nesterov, momentum, _, _ = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(p, momentum, nesterov)

            if g is None:
                updates.append(None)
                continue

            g_local = g.to_local() if isinstance(g, DTensor) else g
            u = torch.sign(g_local)
            updates.append(u)

        if not skip_update:
            # Scalar parameters are not TP-sharded.
            self.update_bucket_params(
                scalar_params, updates, 0, len(scalar_params), tp_mesh=None
            )

        if not self.need_to_calculate_norm:
            return

        final_norms = {}
        if apply_on_weight and self.need_to_calculate_norm:
            for i, p in enumerate(scalar_params):
                p_local = p.to_local() if isinstance(p, DTensor) else p
                cleaned_p_name = remove_orig_mod_and_weight_for_p_name(
                    scalar_param_names[i]
                )
                # The original code only logs the parameter's absolute value, as the
                # update norm is constant (learning_rate * 1.0).
                final_norms[f"scalar_param_supremum/{cleaned_p_name}"] = p_local.abs()

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_embedding")
    def step_embedding(
        self,
        embed_params,
        embed_param_names,
        workspace,
        skip_update=False,
        apply_on_weight=True,
    ):
        # Reuse pre-allocated Python lists (reset before use)
        effective_grads: list = workspace["effective_grads"]
        updates: list = workspace["updates"]
        for i in range(len(effective_grads)):
            effective_grads[i] = None
        for i in range(len(updates)):
            updates[i] = None

        tp_mesh = (
            self.parallel_dims.get_optional_mesh("tp") if self.tp_enabled else None
        )

        batched_g_bufs = workspace["batched_g_bufs"]
        buf_views = workspace["buf_views_by_shape"]

        # ===== PREPARE: fetch all grads + fill big_g buffers =====

        # Canonical params [0, 1] — per-param fetch (preserves TP edge-case handling)
        for i in range(min(2, len(embed_params))):
            p = embed_params[i]
            _, nesterov, momentum, _, _ = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(p, momentum, nesterov, gather_to_local=False)
            if g is not None and not self.fsdp_enabled and self.tp_enabled:
                # Edge case: TP-only (no FSDP) — grad is Replicate, weight is sharded
                original_placements = p.placements
                tp_mesh_dim = tp_axis(original_placements, True)
                tp_sharded_dim = original_placements[tp_mesh_dim].dim
                chunk_size = p.to_local().shape[tp_sharded_dim]
                start_offset = tp_mesh.get_local_rank() * chunk_size
                slicer = [slice(None)] * g.dim()
                slicer[tp_sharded_dim] = slice(start_offset, start_offset + chunk_size)
                g = g[tuple(slicer)]
            effective_grads[i] = g

        # Extra params [2+] — batched fetch grouped by group_idx
        for gidx, indices, params_in_group in self._embed_extra_by_group:
            _, nesterov, momentum, _, _ = self.groups_info[gidx]
            grads = self.get_momentum_or_grad_list(
                params_in_group,
                momentum,
                nesterov,
                gather_to_local=False,
                to_local=True,
            )
            for j, i in enumerate(indices):
                effective_grads[i] = grads[j]

        # Fill big_g buffers for all-valid shape groups — like expert _prepare_experts_lmo.
        # Fill happens BEFORE any LMO so GPU memory writes complete before compute.
        filled_shapes: set[tuple] = set()
        for shape, indices in self._embed_extra_shape_groups.items():
            if shape not in batched_g_bufs:
                continue
            if all(effective_grads[i] is not None for i in indices):
                torch._foreach_copy_(
                    buf_views[shape], [effective_grads[i] for i in indices]
                )
                filled_shapes.add(shape)

        # ===== LMO =====

        # Canonical LMO [0, 1] — per-param (unique shapes, no batching)
        for i in range(min(2, len(embed_params))):
            g = effective_grads[i]
            if g is None:
                continue
            *_, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(embed_params[i])]
            ]
            updates[i] = self.lmo(g, **param_kwargs)

        # Extra LMO by shape group — use pre-filled big_g or fallback to stack
        big_us_by_shape: dict[tuple, torch.Tensor] = {}
        for shape, indices in self._embed_extra_shape_groups.items():
            gidx = self._embed_shape_group_gidx[shape]
            *_, param_kwargs = self.groups_info[gidx]

            if shape in filled_shapes:
                # All-valid fast path: lmo on pre-filled buffer (no allocation).
                # Result collected in big_us_by_shape; applied by _update_embed_params_fast.
                big_us_by_shape[shape] = self.lmo(batched_g_bufs[shape], **param_kwargs)
            else:
                # Partial grads fallback — collect valid, stack, lmo, store in updates[]
                valid_pairs = [
                    (i, effective_grads[i])
                    for i in indices
                    if effective_grads[i] is not None
                ]
                if not valid_pairs:
                    continue
                valid_idx, valid_grads = zip(*valid_pairs)
                if len(valid_grads) >= 2:
                    big_u = self.lmo(torch.stack(list(valid_grads)), **param_kwargs)
                    u_views = big_u.unbind(0)
                    for k, i in enumerate(valid_idx):
                        updates[i] = u_views[k]
                else:
                    updates[valid_idx[0]] = self.lmo(valid_grads[0], **param_kwargs)

        # ===== UPDATE =====
        # Fast-path extras applied via big_us_by_shape (_foreach_add_ with unbind).
        # Canonicals [0,1] + partial fallback extras applied via update_plan.
        if not skip_update:
            self._update_embed_params_fast(updates, big_us_by_shape)

        #  Norm Calculation (on full tensors for correctness)
        if not self.need_to_calculate_norm:
            return
        final_norms = {}
        apply_on_weight = apply_on_weight and self.need_to_calculate_norm
        norm_scratch: list = workspace["norm_scratch"]

        for i, (p, p_name) in enumerate(zip(embed_params, embed_param_names)):
            lr, nesterov, momentum, _, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]

            # Gather full tensor for norm calculation
            g = self.get_momentum_or_grad(p, momentum, nesterov, gather_to_local=True)
            u = self.lmo(g, **param_kwargs)

            need_T = CONST_NAME_OF_EMBEDDING in p_name

            # Use pre-alloc float32 scratch to avoid -lr*u temp allocation (extras only)
            scratch = norm_scratch[i]
            if scratch is not None and scratch.shape == u.shape:
                torch.mul(u, -lr, out=scratch)
                upd_norms = calculate_norm(scratch, self.norms_to_log, transpose=need_T)
            else:
                upd_norms = calculate_norm(-lr * u, self.norms_to_log, transpose=need_T)

            # Gather the parameter itself to a full tensor if needed
            if apply_on_weight and isinstance(p, DTensor):
                p = p.full_tensor()

            if apply_on_weight:
                wnorm = calculate_norm(p, self.norms_to_log, transpose=need_T)

            cleaned_p_name = remove_orig_mod_and_weight_for_p_name(p_name)
            for norm_name in self.norms_to_log:
                final_norms[f"track_update_{norm_name}/{cleaned_p_name}"] = upd_norms[
                    norm_name
                ]
                if apply_on_weight:
                    final_norms[f"track_param_{norm_name}/{cleaned_p_name}"] = wnorm[
                        norm_name
                    ]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_experts")
    def step_experts(
        self,
        expert_params,
        expert_param_names,
        workspace,
        skip_update=False,
        apply_on_weight=True,
    ):
        (expert_big_g_bufs, expert_dst_views_by_block) = (
            workspace["big_g_bufs"],
            workspace["dst_views_by_block"],
        )

        need_to_calculate_norm = self.need_to_calculate_norm

        norms_of_update, norms_of_weight, final_norms = [], [], {}
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        device = expert_params[0].device
        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        world_size, local_rank = fsdp_mesh.size(), fsdp_mesh.get_local_rank()

        # Use pre-computed structural metadata from init (avoids per-step recomputation).
        ep_per_rank = self._expert_ep_per_rank
        kinds_of_norms = self._expert_kinds_of_norms
        transpose = self._expert_transpose
        blocks = self._expert_blocks

        padding_norms = self._get_cached_zero_scalar(device)

        # Fill all expert big_g buffers once before block LMO loop.

        self._prepare_experts_lmo(
            expert_params,
            expert_big_g_bufs,
            dst_views_by_block=expert_dst_views_by_block,
        )

        all_updates: list[torch.Tensor | None] = [None] * len(expert_params)

        for block_idx, (start, end) in enumerate(blocks):
            block_params = expert_params[start:end]
            if not block_params:
                continue

            # Pre-stored structural lmo kwargs (eps/norm_factor/backend…); lr/wd excluded.
            kwargs0 = self._expert_kwargs_per_block[block_idx]
            if kwargs0 is None:
                continue

            K = end - start

            any_grad = (
                self._expert_any_grad[block_idx]
                if block_idx < len(self._expert_any_grad)
                else False
            )

            big_g = expert_big_g_bufs[block_idx]  # [K*ep_per_rank, A, B]

            if not any_grad or big_g is None:
                all_updates[start:end] = [None] * K
                continue

            lmo_label = f"disco.expert_lmo_block{block_idx}"
            with record_function(lmo_label):
                big_u = self.lmo(big_g, **kwargs0, transpose_experts=transpose)
            all_updates[start:end] = big_u.view(K, ep_per_rank, *big_u.shape[1:])

        if need_to_calculate_norm:
            for block_idx, (start, end) in enumerate(blocks):
                block_params = expert_params[start:end]
                block_updates = all_updates[start:end]
                if not block_params or not block_updates:
                    continue
                for p, u in zip(block_params, block_updates):
                    if u is None:
                        continue
                    assert u.ndim == 3
                    p_local = p.to_local() if isinstance(p, DTensor) else p
                    for ep_idx in range(u.shape[0]):
                        update_norms = calculate_norm(
                            u[ep_idx], self.norms_to_log, transpose=transpose
                        )
                        norms_of_update.extend(update_norms.values())

                        if apply_on_weight:
                            weight_norms = calculate_norm(
                                p_local[ep_idx],
                                self.norms_to_log,
                                transpose=transpose,
                            )
                            norms_of_weight.extend(weight_norms.values())

        if not skip_update:
            if any(u is not None for u in all_updates):
                self._update_expert_params_fast(expert_params, all_updates)

        if need_to_calculate_norm:
            expected_total = len(expert_params) * ep_per_rank * kinds_of_norms
            pad_needed = expected_total - len(norms_of_update)
            if pad_needed > 0:
                norms_of_update.extend([padding_norms] * pad_needed)
                if apply_on_weight:  # keep weight-norms aligned
                    norms_of_weight.extend([padding_norms] * pad_needed)

            norms_tensor = torch.stack(norms_of_update).float().to(device)
            gathered_update_norms = funcol.all_gather_tensor(
                norms_tensor, gather_dim=0, group=fsdp_mesh
            )

            if apply_on_weight:
                norms_tensor = torch.stack(norms_of_weight).float().to(device)
                # TODO: This barrier may be removable because the subsequent
                # all_gather_tensor is a collective synchronization point.
                dist.barrier()
                gathered_weight_norms = funcol.all_gather_tensor(
                    norms_tensor, gather_dim=0, group=fsdp_mesh
                )

            if local_rank == 0:
                norm_names = list(self.norms_to_log)

                P = len(expert_params)  # parameters per rank
                E = ep_per_rank  # experts per rank
                K = kinds_of_norms  # norms per expert
                block = P * E * K  # values contributed by each rank

                for idx in range(world_size * block):
                    r, rem = divmod(idx, block)  # producing rank
                    p, rem = divmod(rem, E * K)  # parameter index
                    e, k = divmod(rem, K)  # expert, norm indices

                    actual_ep_idx = e + r * E
                    if actual_ep_idx >= expert_params[0].shape[0]:
                        continue  # skip pure padding slots

                    cleaned_name = remove_orig_mod_and_weight_for_p_name(
                        expert_param_names[p]
                    )
                    norm_name = norm_names[k]

                    key_update = (
                        f"track_update_{norm_name}/ep_{actual_ep_idx}/{cleaned_name}"
                    )
                    final_norms[key_update] = gathered_update_norms[idx]

                    if apply_on_weight:
                        key_param = (
                            f"track_param_{norm_name}/ep_{actual_ep_idx}/{cleaned_name}"
                        )
                        final_norms[key_param] = gathered_weight_norms[idx]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_ddp")
    def step_ddp(
        self,
        ddp_params,
        ddp_param_names,
        workspace,
        skip_update: bool = False,
        apply_on_weight: bool = True,
    ):

        need_to_calculate_norm = self.need_to_calculate_norm
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        # --- distributed groups ---
        dp_replicate_mesh = (
            self.parallel_dims.get_optional_mesh("dp_replicate")
            if self.dp_replicate_enabled
            else None
        )

        world_size, rank = (
            (dp_replicate_mesh.size(), dp_replicate_mesh.get_local_rank())
            if dp_replicate_mesh
            else (1, 0)
        )
        if world_size != self._ddp_world_size or rank != self._ddp_rank:
            self._precompute_ddp_metadata()

        tp_mesh = (
            self.parallel_dims.get_optional_mesh("tp") if self.tp_enabled else None
        )
        tp_world_size = tp_mesh.size() if tp_mesh is not None else 1

        device = ddp_params[0].device
        cast_dtype = self.communication_dtype  # comm/exchange dtype
        total_buckets = self._ddp_total_buckets
        num_norm_types = len(self.norms_to_log)
        final_norms = {}

        # -------- Phase A: precompute local LMO updates (no comm) --------
        local_updates: list[torch.Tensor | None] = [None] * len(ddp_params)
        lmo_inputs = self._prepare_ddp_lmo(
            ddp_params,
            workspace,
            tp_mesh=tp_mesh,
            tp_world_size=tp_world_size,
        )
        for i in self._ddp_owned_indices:
            g = lmo_inputs[i]
            if g is None:
                local_updates[i] = None
                continue
            group_idx = self._ddp_param_group_idx[i]
            param_kwargs = self.groups_info[group_idx][-1]
            u = self.lmo(g, **param_kwargs)
            local_updates[i] = u

        # -------- Phase B: DDP communication (one-shot flat all_gather) and global updates --------
        (global_updates, global_update_bufs, zero_by_shape, norm_scratch) = (
            workspace["global_updates"],
            workspace["global_update_bufs"],
            workspace["zero_by_shape"],
            workspace["norm_scratch"],
        )

        if skip_update:
            global_updates[:] = [None] * len(global_updates)
        if not skip_update:
            if dp_replicate_mesh is not None and world_size > 1:
                (
                    ddp_flat_send,
                    ddp_flat_recv_chunks,
                    ddp_pack_dst_views,
                    ddp_unpack_src_views,
                    ddp_unpack_dst_views,
                ) = (
                    workspace["ddp_flat_send"],
                    workspace["ddp_flat_recv_chunks"],
                    workspace["ddp_pack_dst_views"],
                    workspace["ddp_unpack_src_views"],
                    workspace["ddp_unpack_dst_views"],
                )

                pack_src_views: list[torch.Tensor] = []
                for param_idx in self._ddp_owned_indices:
                    u = local_updates[param_idx]
                    if u is None:
                        ref = ddp_params[param_idx]
                        u = zero_by_shape[(tuple(ref.shape), cast_dtype)]
                    pack_src_views.append(u)

                if pack_src_views:
                    torch._foreach_copy_(ddp_pack_dst_views, pack_src_views)

                owned_total_numel = self._ddp_rank_owned_numels[rank]
                max_owned_numel = self._ddp_max_rank_owned_numel
                if owned_total_numel < max_owned_numel:
                    ddp_flat_send.narrow(
                        0, owned_total_numel, max_owned_numel - owned_total_numel
                    ).zero_()

                dist.all_gather(
                    ddp_flat_recv_chunks,
                    ddp_flat_send,
                    group=dp_replicate_mesh.get_group(),
                )
                if ddp_unpack_src_views:
                    torch._foreach_copy_(ddp_unpack_dst_views, ddp_unpack_src_views)

                for param_idx, buf in enumerate(global_update_bufs):
                    global_updates[param_idx] = buf
            else:
                for param_idx, p in enumerate(ddp_params):
                    u = local_updates[param_idx]
                    if u is None:
                        global_updates[param_idx] = zero_by_shape[
                            (tuple(p.shape), cast_dtype)
                        ]
                    else:
                        global_updates[param_idx] = u

        upd_norm_local_flat = None
        w_norm_local_flat = None
        if need_to_calculate_norm:
            required_norm_elems = total_buckets * num_norm_types
            upd_norm_local_flat = workspace.get("upd_norm_local_flat")
            if (
                upd_norm_local_flat is None
                or upd_norm_local_flat.numel() != required_norm_elems
            ):
                upd_norm_local_flat = torch.empty(
                    required_norm_elems, dtype=torch.float32, device=device
                )
                workspace["upd_norm_local_flat"] = upd_norm_local_flat
            upd_norm_local_flat.zero_()

            if apply_on_weight:
                w_norm_local_flat = workspace.get("w_norm_local_flat")
                if (
                    w_norm_local_flat is None
                    or w_norm_local_flat.numel() != required_norm_elems
                ):
                    w_norm_local_flat = torch.empty(
                        required_norm_elems, dtype=torch.float32, device=device
                    )
                    workspace["w_norm_local_flat"] = w_norm_local_flat
                w_norm_local_flat.zero_()

        # ---- local update norms (owner slots only) ----
        if need_to_calculate_norm and upd_norm_local_flat is not None:
            for my_idx in self._ddp_owned_indices:
                u = local_updates[my_idx]
                if u is None:
                    continue
                group_idx = self._ddp_param_group_idx[my_idx]
                lr, *_ = self.groups_info[group_idx]
                scratch = norm_scratch[my_idx]
                if scratch is None:
                    raise ValueError("Missing DDP norm scratch buffer for owned index.")
                torch.mul(u, -lr, out=scratch)
                upd_norms = calculate_norm(scratch, self.norms_to_log)
                base = self._ddp_owner_bucket_by_param[my_idx] * num_norm_types
                upd_norm_local_flat[base : base + num_norm_types].copy_(
                    torch.stack(list(upd_norms.values()))
                )

        # -------- Phase C: apply once (pre-cast + grouped foreach apply) --------
        if not skip_update:
            apply_updates = self._prepare_ddp_apply_updates(
                global_updates,
                workspace,
                tp_mesh=tp_mesh,
            )
            self._update_ddp_params_fast(apply_updates)

        # -------- Phase C.5: Calculate Weight Norms (POST-UPDATE) --------
        if apply_on_weight and w_norm_local_flat is not None:
            for my_idx in self._ddp_owned_indices:
                w = ddp_params[my_idx]
                tp_gather_info = (
                    self._ddp_tp_gather_info_by_idx[my_idx]
                    if tp_mesh is not None
                    else None
                )
                if tp_gather_info is not None:
                    w_local = w.to_local() if isinstance(w, DTensor) else w
                    w = gather_tp_shard(w_local, tp_mesh, tp_world_size, tp_gather_info)
                elif isinstance(w, DTensor):
                    w = w.to_local()
                w_norms = calculate_norm(w, self.norms_to_log)
                base = self._ddp_owner_bucket_by_param[my_idx] * num_norm_types
                w_norm_local_flat[base : base + num_norm_types].copy_(
                    torch.stack(list(w_norms.values()))
                )

        # -------- Phase D: final norm gather/log --------
        if not need_to_calculate_norm:
            return

        if upd_norm_local_flat is None:
            return
        upd = upd_norm_local_flat
        if dp_replicate_mesh is not None and world_size > 1:
            gathered_upd = funcol.all_gather_tensor(
                upd, gather_dim=0, group=dp_replicate_mesh
            )
        else:
            gathered_upd = upd

        if apply_on_weight and w_norm_local_flat is not None:
            w = w_norm_local_flat
            if dp_replicate_mesh is not None and world_size > 1:
                gathered_w = funcol.all_gather_tensor(
                    w, gather_dim=0, group=dp_replicate_mesh
                )
            else:
                gathered_w = w
        else:
            gathered_w = None

        if self.is_dp_rank_0:
            cleaned_names = (
                self._ddp_clean_param_names
                if len(self._ddp_clean_param_names) == len(ddp_param_names)
                else [
                    remove_orig_mod_and_weight_for_p_name(p_name)
                    for p_name in ddp_param_names
                ]
            )
            owner_ranks = (
                self._ddp_owner_rank_by_param
                if len(self._ddp_owner_rank_by_param) == len(ddp_param_names)
                else [i % world_size for i in range(len(ddp_param_names))]
            )
            owner_buckets = (
                self._ddp_owner_bucket_by_param
                if len(self._ddp_owner_bucket_by_param) == len(ddp_param_names)
                else [i // world_size for i in range(len(ddp_param_names))]
            )
            for param_idx, cleaned in enumerate(cleaned_names):
                owner_rank = owner_ranks[param_idx]
                owner_bucket = owner_buckets[param_idx]
                base = (owner_rank * total_buckets + owner_bucket) * num_norm_types
                for k, norm_name in enumerate(self.norms_to_log):
                    idx = base + k
                    final_norms[f"track_update_{norm_name}/{cleaned}"] = gathered_upd[
                        idx
                    ]
                    if apply_on_weight:
                        final_norms[f"track_param_{norm_name}/{cleaned}"] = gathered_w[
                            idx
                        ]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    def _gather_and_log_fsdp_norms(
        self,
        norms_of_update,
        norms_of_weight,
        fsdp_mesh,
        device,
        fsdp_param_names,
        world_size,
        total_buckets,
        apply_on_weight,
    ):
        """
        Gathers FSDP norm tensors from all ranks and logs them on rank 0.
        This is a collective operation followed by a rank-0 logging step.
        """
        # --- 1. Collective Communication: All ranks must participate ---
        upd = torch.stack(norms_of_update).float().to(device)
        gathered_update_norms = funcol.all_gather_tensor(
            upd, gather_dim=0, group=fsdp_mesh
        )

        gathered_weight_norms = None
        if apply_on_weight and norms_of_weight:
            w = torch.stack(norms_of_weight).float().to(device)
            gathered_weight_norms = funcol.all_gather_tensor(
                w, gather_dim=0, group=fsdp_mesh
            )

        # --- 2. Local Processing: Only rank 0 processes and logs the results ---
        final_norms = {}
        if self.is_dp_rank_0:
            num_norm_types = len(self.norms_to_log)
            entries_per_rank = total_buckets * num_norm_types
            cleaned_names = [
                remove_orig_mod_and_weight_for_p_name(pn) for pn in fsdp_param_names
            ]

            for param_idx, cleaned_p_name in enumerate(cleaned_names):
                owner_rank = param_idx % world_size
                bucket_idx_on_owner = param_idx // world_size
                base = (
                    owner_rank * entries_per_rank + bucket_idx_on_owner * num_norm_types
                )

                for norm_idx, norm_name in enumerate(self.norms_to_log):
                    idx = base + norm_idx
                    final_norms[
                        f"track_update_{norm_name}/{cleaned_p_name}"
                    ] = gathered_update_norms[idx]
                    if apply_on_weight and gathered_weight_norms is not None:
                        final_norms[
                            f"track_param_{norm_name}/{cleaned_p_name}"
                        ] = gathered_weight_norms[idx]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_fsdp")
    def step_fsdp(
        self,
        fsdp_params,
        fsdp_param_names,
        workspace,
        skip_update=False,
        apply_on_weight=True,
    ):

        need_to_calculate_norm = self.need_to_calculate_norm
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        world_size, rank = fsdp_mesh.size(), fsdp_mesh.get_local_rank()
        device = fsdp_params[0].device
        cast_dtype = self.communication_dtype

        tp_mesh = (
            self.parallel_dims.get_optional_mesh("tp") if self.tp_enabled else None
        )

        dp_replicate_mesh = (
            self.parallel_dims.get_optional_mesh("dp_replicate")
            if self.dp_replicate_enabled
            else None
        )
        fsdp_group = fsdp_mesh.get_group()

        global_updates = [None] * len(fsdp_params)
        norms_of_update, norms_of_weight = [], []
        padding_norms = self._get_cached_padding_norms(device)

        # Use pre-computed total_buckets from init.
        total_buckets = self._fsdp_total_buckets

        use_global_fast_path = self.fsdp_a2a_mode == "once"
        bucket_workspace = None

        if use_global_fast_path:
            # Use persistent workspace cache; _workspace override for external callers.
            global_grad_send = workspace["global_grad_send"]
            global_grad_recv = workspace["global_grad_recv"]
            full_g_bufs = workspace["full_g_bufs"]
            global_upd_send = None if skip_update else workspace["global_upd_send"]
            global_upd_recv = None if skip_update else workspace["global_upd_recv"]
            global_upd_recv_cast = (
                None if skip_update else workspace["global_upd_recv_cast"]
            )
            if not skip_update and (global_upd_send is None or global_upd_recv is None):
                raise RuntimeError(
                    "Missing reverse A2A buffers for FSDP once-mode update."
                )

            self._prepare_fsdp_lmo(
                global_grad_send,
                pack_dst_views=workspace.get("pack_dst_views"),
            )

            dist.all_to_all_single(
                global_grad_recv,
                global_grad_send,
                output_split_sizes=self._fsdp_global_output_splits_elems,
                input_split_sizes=self._fsdp_global_input_splits_elems,
                group=fsdp_group,
            )

            # Global prep: fill all full_g_bufs in a single dispatch before LMO loop.
            # This moves all full_g assembly outside the LMO loop entirely.
            src_flat = workspace["full_g_prep_src_flat"]
            dst_flat = workspace["full_g_prep_dst_flat"]
            if src_flat:
                torch._foreach_copy_(dst_flat, src_flat)

            # Clean LMO loop: only lmo() + optional reduce + optional norm.
            # Reverse-pack source view creation is moved outside this loop.
            bucket_norm_dicts: list[dict | None] = [None] * total_buckets
            u_keepalive: list[torch.Tensor] = []

            with record_function("disco.fsdp_lmo_loop"):
                for bucket_idx in range(total_buckets):
                    start_idx, end_idx = self._fsdp_bucket_ranges[bucket_idx]
                    param_kwargs_me = self._fsdp_param_kwargs_me[bucket_idx]
                    bucket_group_indices = self._fsdp_bucket_group_indices[bucket_idx]
                    my_param_in_bucket = (start_idx + rank) < end_idx

                    u = self.lmo(full_g_bufs[bucket_idx], **param_kwargs_me)

                    if dp_replicate_mesh and self.extra_reduce_for_HSDP:
                        dist.all_reduce(
                            u, group=dp_replicate_mesh, op=dist.ReduceOp.AVG
                        )

                    if not skip_update:
                        u_keepalive.append(u)

                    if need_to_calculate_norm and my_param_in_bucket:
                        lr, *_ = self.groups_info[bucket_group_indices[rank]]
                        bucket_norm_dicts[bucket_idx] = calculate_norm(
                            -lr * u, self.norms_to_log
                        )

            if need_to_calculate_norm:
                for b in range(total_buckets):
                    d = bucket_norm_dicts[b]
                    norms_of_update.extend(
                        d.values() if d is not None else padding_norms.values()
                    )

            if not skip_update:
                if len(u_keepalive) != total_buckets:
                    raise RuntimeError(
                        "FSDP once-mode unexpected number of LMO outputs: "
                        f"got={len(u_keepalive)}, expected={total_buckets}."
                    )
                torch._foreach_copy_(workspace["u_bufs"], u_keepalive)
                # Batch pack: single _foreach_copy_ into persistent flat dst views.
                torch._foreach_copy_(
                    workspace["upd_pack_dst_views_flat"],
                    workspace["u_src_views_flat"],
                )

                with record_function("disco.fsdp_reverse_a2a"):
                    dist.all_to_all_single(
                        global_upd_recv,
                        global_upd_send,
                        output_split_sizes=self._fsdp_global_input_splits_elems,
                        input_split_sizes=self._fsdp_global_output_splits_elems,
                        group=fsdp_group,
                    )
                u_keepalive.clear()

                if global_upd_recv_cast is not None:
                    global_upd_recv_cast.copy_(global_upd_recv)
                elif workspace.get("cast_dst_by_dtype"):
                    for dtype, dvs in workspace["cast_dst_by_dtype"].items():
                        svs = workspace["cast_src_by_dtype"][dtype]
                        torch._foreach_copy_(dvs, svs)
                upd_recv_views = workspace["upd_recv_views"]
                for j, (param_idx, _, _, _) in enumerate(
                    self._fsdp_upd_recv_param_plan
                ):
                    global_updates[param_idx] = upd_recv_views[j]
        else:
            bucket_workspace = self._allocate_fsdp_bucket_workspace(
                cast_dtype=cast_dtype,
                device=workspace["device"],
                apply_on_weight=apply_on_weight,
                skip_update=skip_update,
            )

            grad_send_scratch = bucket_workspace["grad_send_flat"]
            grad_recv_scratch = bucket_workspace["grad_recv_flat"]
            upd_send_scratch = bucket_workspace["upd_send_flat"]
            upd_recv_scratch = bucket_workspace["upd_recv_flat"]

            for bucket_idx in range(total_buckets):
                start_idx, end_idx = self._fsdp_bucket_ranges[bucket_idx]
                recv_shapes = self._fsdp_recv_shapes[bucket_idx]
                send_shapes = self._fsdp_send_shapes[bucket_idx]
                split_rows = self._fsdp_split_rows[bucket_idx]
                tp_infos = self._fsdp_tp_gather_info[bucket_idx]
                bucket_params = self._fsdp_bucket_params[bucket_idx]
                bucket_group_indices = self._fsdp_bucket_group_indices[bucket_idx]
                param_kwargs_me = self._fsdp_param_kwargs_me[bucket_idx]
                send_numels = self._fsdp_send_numels[bucket_idx]
                recv_numels = self._fsdp_recv_numels[bucket_idx]
                my_param_in_bucket = (start_idx + rank) < end_idx

                send_chunk_offsets = self._fsdp_bucket_send_chunk_offsets[bucket_idx]
                recv_chunk_offsets = self._fsdp_bucket_recv_chunk_offsets[bucket_idx]
                send_total = self._fsdp_bucket_send_total_elems[bucket_idx]
                recv_total = self._fsdp_bucket_recv_total_elems[bucket_idx]
                grad_send_flat = grad_send_scratch[:send_total]
                grad_recv_flat = grad_recv_scratch[:recv_total]

                for i in range(world_size):
                    base = send_chunk_offsets[i]
                    numel = send_numels[i]
                    if start_idx + i < end_idx:
                        p = bucket_params[i]
                        group_idx = bucket_group_indices[i]
                        param_idx = start_idx + i
                        g = self._get_effective_grad_by_group(p, group_idx, param_idx)
                        g_local = self._maybe_unpack_dtensor(g, tp_infos[i])
                        if g_local.dtype != cast_dtype:
                            g_local = g_local.to(cast_dtype)
                        grad_send_flat[base : base + numel].copy_(g_local.reshape(-1))
                    else:
                        grad_send_flat[base : base + numel].zero_()

                dist.all_to_all_single(
                    grad_recv_flat,
                    grad_send_flat,
                    output_split_sizes=recv_numels,
                    input_split_sizes=send_numels,
                    group=fsdp_group,
                )

                recv_views = [
                    grad_recv_flat[
                        recv_chunk_offsets[src] : recv_chunk_offsets[src]
                        + recv_numels[src]
                    ].view(recv_shapes[src])
                    for src in range(world_size)
                ]
                full_g = torch.cat(recv_views, dim=0)
                u = self.lmo(full_g, **param_kwargs_me)

                if dp_replicate_mesh and self.extra_reduce_for_HSDP:
                    dist.all_reduce(u, group=dp_replicate_mesh, op=dist.ReduceOp.AVG)

                if not skip_update:
                    upd_send_flat = upd_send_scratch[:recv_total]
                    updates_send_list = list(torch.split(u, split_rows, dim=0))
                    for dst in range(world_size):
                        base = recv_chunk_offsets[dst]
                        u_part = updates_send_list[dst]
                        if u_part.dtype != cast_dtype:
                            u_part = u_part.to(cast_dtype)
                        upd_send_flat[base : base + recv_numels[dst]].copy_(
                            u_part.reshape(-1)
                        )

                    upd_recv_flat = upd_recv_scratch[:send_total]
                    dist.all_to_all_single(
                        upd_recv_flat,
                        upd_send_flat,
                        output_split_sizes=send_numels,
                        input_split_sizes=recv_numels,
                        group=fsdp_group,
                    )
                    for i in range(end_idx - start_idx):
                        base = send_chunk_offsets[i]
                        global_updates[start_idx + i] = upd_recv_flat[
                            base : base + send_numels[i]
                        ].view(send_shapes[i])

                if need_to_calculate_norm:
                    if my_param_in_bucket:
                        lr, *_ = self.groups_info[bucket_group_indices[rank]]
                        upd_norms = calculate_norm(-lr * u, self.norms_to_log)
                    else:
                        upd_norms = padding_norms
                    norms_of_update.extend(upd_norms.values())

        # Single vectorised apply.
        if not skip_update:
            self.update_bucket_params(
                fsdp_params,
                global_updates,
                0,
                len(fsdp_params),
                tp_mesh=tp_mesh,
            )

        # --- Calculate Weight Norms (POST-UPDATE) ---
        if apply_on_weight:
            param_send_scratch = (
                bucket_workspace["param_send_flat"] if bucket_workspace else None
            )
            param_recv_scratch = (
                bucket_workspace["param_recv_flat"] if bucket_workspace else None
            )
            for bucket_idx in range(total_buckets):
                start_idx, end_idx = self._fsdp_bucket_ranges[bucket_idx]
                my_param_in_bucket = (start_idx + rank) < end_idx

                recv_shapes = self._fsdp_recv_shapes[bucket_idx]
                recv_numels = self._fsdp_recv_numels[bucket_idx]
                send_numels = self._fsdp_send_numels[bucket_idx]
                tp_infos = self._fsdp_tp_gather_info[bucket_idx]
                send_total = self._fsdp_bucket_send_total_elems[bucket_idx]
                recv_total = self._fsdp_bucket_recv_total_elems[bucket_idx]
                send_chunk_offsets = self._fsdp_bucket_send_chunk_offsets[bucket_idx]
                recv_chunk_offsets = self._fsdp_bucket_recv_chunk_offsets[bucket_idx]
                if param_send_scratch is None or param_recv_scratch is None:
                    param_send_flat = torch.empty(
                        send_total, dtype=cast_dtype, device=device
                    )
                    param_recv_flat = torch.empty(
                        recv_total, dtype=cast_dtype, device=device
                    )
                else:
                    param_send_flat = param_send_scratch[:send_total]
                    param_recv_flat = param_recv_scratch[:recv_total]

                for i in range(world_size):
                    param_idx = start_idx + i
                    p = fsdp_params[param_idx if param_idx < end_idx else end_idx - 1]

                    tp_info = tp_infos[i]
                    p_local = self._maybe_unpack_dtensor(p, tp_info)
                    if p_local.dtype != cast_dtype:
                        p_local = p_local.to(cast_dtype)
                    base = send_chunk_offsets[i]
                    param_send_flat[base : base + send_numels[i]].copy_(
                        p_local.reshape(-1)
                    )

                dist.all_to_all_single(
                    param_recv_flat,
                    param_send_flat,
                    output_split_sizes=recv_numels,
                    input_split_sizes=send_numels,
                    group=fsdp_group,
                )
                recv_views = [
                    param_recv_flat[
                        recv_chunk_offsets[src] : recv_chunk_offsets[src]
                        + recv_numels[src]
                    ].view(recv_shapes[src])
                    for src in range(world_size)
                ]
                full_weight = torch.cat(recv_views, dim=0)

                w_norms = (
                    calculate_norm(full_weight, self.norms_to_log)
                    if my_param_in_bucket
                    else padding_norms
                )
                norms_of_weight.extend(w_norms.values())

        if need_to_calculate_norm and norms_of_update:
            self._gather_and_log_fsdp_norms(
                norms_of_update,
                norms_of_weight,
                fsdp_mesh,
                rank,
                device,
                fsdp_param_names,
                world_size,
                total_buckets,
                apply_on_weight,
            )

    @record_function("disco._prepare_gradients_and_momentum")
    @torch.no_grad()
    def prepare_gradients_and_momentum(self) -> None:
        """
        Fused pre-pass that updates momentum buffers for *all* parameters
        with available grads:
            buf <- (1 - m) * buf + m * g
        It performs foreach-kernel updates per (device, dtype, momentum).

        Uses pre-built momentum buffer lists (_momentum_state) to avoid
        per-step state dict lookups.

        ************************************************************************
        Notes:
        - momentum schedule can change at runtime (e.g. 0.0 -> 0.9)
        - params without grad in the current step are skipped
        ************************************************************************
        """
        self._refresh_momentum_execution_plan()

        for m, bufs, params in self._momentum_execution_plan:
            active_bufs = []
            active_grads = []
            for buf, p in zip(bufs, params):
                g = p.grad
                if g is not None:
                    active_bufs.append(buf)
                    active_grads.append(g)
            if not active_bufs:
                continue
            torch._foreach_mul_(active_bufs, 1.0 - m)
            torch._foreach_add_(active_bufs, active_grads, alpha=m)

    @record_function("disco.get_momentum_or_grad")
    def get_momentum_or_grad(self, p, momentum, nesterov, gather_to_local=False):
        """
        Retrieves the effective gradient for a parameter.
        Assumes the momentum buffer has already been updated in a pre-pass.
        """
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        use_momentum = momentum > 0 and momentum < 1

        if not self.is_light and use_momentum:
            buf = self._momentum_buffer_by_param_id.get(id(p))
            if buf is None:
                raise ValueError(
                    "Momentum buffer missing; ensure pre-pass ran before calling get_momentum_or_grad."
                )
            g = buf if not nesterov else torch.lerp(buf, g, momentum)

        if gather_to_local and isinstance(g, DTensor):
            g = g.redistribute(placements=[Replicate()] * g.device_mesh.ndim).to_local()

        return g

    @record_function("disco.get_momentum_or_grad_list")
    def get_momentum_or_grad_list(
        self,
        params,
        momentum,
        nesterov,
        gather_to_local: bool = False,
        to_local: bool = False,
    ):
        """
        Batched variant of get_momentum_or_grad for a parameter list.
        Returns effective gradients in input order (None where unavailable).
        """
        outputs: list[torch.Tensor | None] = [None] * len(params)
        use_momentum = (not self.is_light) and (0.0 < momentum < 1.0)

        for i, p in enumerate(params):
            g = p.grad
            if g is None or not p.requires_grad:
                continue

            if use_momentum:
                buf = self._momentum_buffer_by_param_id.get(id(p))
                if buf is None:
                    raise ValueError(
                        "Momentum buffer missing; ensure pre-pass ran before calling get_momentum_or_grad_list."
                    )
                g = buf if not nesterov else torch.lerp(buf, g, momentum)

            if gather_to_local and isinstance(g, DTensor):
                g = g.redistribute(
                    placements=[Replicate()] * g.device_mesh.ndim
                ).to_local()
            elif to_local and isinstance(g, DTensor):
                g = g.to_local()
            outputs[i] = g

        return outputs

    @record_function("disco.update_bucket_params")
    def update_bucket_params(self, params, updates, start_idx, end_idx, tp_mesh=None):
        slice_params = params[start_idx:end_idx]
        slice_updates = updates[: (end_idx - start_idx)]

        prepared = []
        if tp_mesh is not None:
            # Use pre-computed TP slicing info to skip per-param isinstance/placements/tp_axis
            # and avoid calling p.to_local() just for a shape check.
            tp_slice_info = self._tp_slice_info
            for p, u in zip(slice_params, slice_updates):
                if u is None:
                    prepared.append((p, None))
                    continue
                info = tp_slice_info.get(id(p))
                if info is not None:
                    _, _, _, slicer, local_shape = info
                    # Apply TP slicing only when the update is the full (non-sharded) tensor.
                    if u.shape != local_shape:
                        u = u[slicer]
                prepared.append((p, u))
        else:
            prepared = list(zip(slice_params, slice_updates))

        # ------------- Phase 2: foreach buckets -------------
        # Use a plain dict (pre-sized) instead of a defaultdict with a lambda closure.
        buckets: dict[tuple, dict] = {}
        param_group_map = self.parameters_to_groups

        for p, u in prepared:
            if u is None:
                continue
            lr, _, _, wd, _ = self.groups_info[param_group_map[id(p)]]
            p_local = self._get_param_local_view(p)

            # Robust shape check after TP slicing.
            if p_local.shape != u.shape:
                # DTensor local views are expected to be stable, but refresh lazily if
                # shape changed due to an unexpected runtime redistribution.
                if isinstance(p, DTensor):
                    p_local = p.to_local()
                    self._param_local_views[id(p)] = p_local
                if p_local.shape != u.shape:
                    raise ValueError(
                        f"Shape mismatch between parameter shard {p_local.shape} and update slice {u.shape}. "
                        f"Ensure you pass the same DDP/FSDP window and slice TP updates. "
                    )

            # dtype/device consistency for foreach.
            if u.dtype != p_local.dtype:
                u = u.to(p_local.dtype)

            key = (p_local.device, p_local.dtype, float(lr), float(wd))
            if key not in buckets:
                buckets[key] = {
                    "orig": [],
                    "locals": [],
                    "updates": [],
                    "lr": lr,
                    "wd": wd,
                }
            b = buckets[key]
            b["orig"].append(p)
            b["locals"].append(p_local)
            b["updates"].append(u)

        for (_, _, lr, wd), data in buckets.items():
            if not data["locals"]:
                continue
            if wd != 0.0:
                torch._foreach_mul_(data["locals"], 1.0 - wd * lr)
            torch._foreach_add_(data["locals"], data["updates"], alpha=-lr)
