# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Applies PT-D parallelisms (except pipeline parallelism) plus activation
# checkpointing and compile to the OPT MoE model.
#
# The old 668-line plan (explicit parallelize_module dicts, ExpertParallel /
# ExpertTensorParallel wrappers, a hand-rolled FSDP walk) is gone: upstream now
# resolves TP/EP from the declarative ShardingConfig set in sharding.py and
# provides a shared FSDP application for decoders.

import logging

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.models.opt_moe.model import OPTMoEModel


logger = logging.getLogger(__name__)


def parallelize_opt_moe(
    model: OPTMoEModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    model_compile_enabled = (
        compile_config is not None and "model" in compile_config.components
    )

    if parallel_dims.cp_enabled:
        # CP is wired but NOT validated on real hardware. In place:
        # `sharding.py` installs the inner-attention local_map (k/v Replicate on
        # the CP axis so they are all-gathered to match the BlockMask's kv dim)
        # and stamps a ShardingConfig on every Linear and on the MoE expert
        # state, so `fully_shard(dp_mesh_dims=...)` accepts the params under
        # spmd_types. Verified with a FAKE process group only -- dense and MoE
        # both reach 100% DTensor params at dp_shard=2/cp=2 -- so no collective
        # has actually run.
        #
        # KNOWN GAP: `rope.cache` stays a plain tensor (see sharding.py).
        logger.warning(
            "Context Parallel is enabled for OPT MoE. This path is newly wired "
            "and has NOT been validated on multiple GPUs -- verify loss against "
            "a cp=1 run before trusting it."
        )

    model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    # Skip FSDP for inference: FSDP's forward hooks are incompatible with the
    # torch.inference_mode() vLLM uses.
    if skip_dp:
        return model

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
        # Pin routed experts to Shard(0). DiSCO's step_experts runs the LMO on
        # each rank's LOCAL expert buffer with no gather, which is only the true
        # LMO when a rank owns whole expert matrices. Under Shard(1) a rank holds
        # a row-block of every expert instead, so the result would be silently
        # wrong (and today it crashes in _foreach_copy_ on the shape mismatch).
        # 0.4.0 had this implicitly: its Shard(1) branch was gated on
        # ep_degree > 1, so opt_moe always got Shard(0).
        expert_shard_dim=0,
    )

    return model
