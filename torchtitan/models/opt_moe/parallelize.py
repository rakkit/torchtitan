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
        # CP now runs on upstream's mechanism: #4322 moved the k/v all-gather
        # out of a local_map in Module.forward() and into the attention
        # implementation itself, and `OPTMoEModel.Config.update_from_config`
        # swaps in KVAllGatherCPFlexInnerAttention when the degree is > 1.
        # `config/validation.py` then REQUIRES a CPInnerAttention at cp > 1, so
        # a misconfigured run fails at config time rather than training wrong.
        #
        # What stays ours is document identity across the load-balancer seam:
        # `doc_id` is computed from GLOBAL positions before cp_shard runs (see
        # model.py preprocess_inputs), because "headtail" hands each rank two
        # NON-adjacent chunks whose join carries no `positions == 0` reset.
        logger.info(
            "Context Parallel enabled for OPT MoE (degree %d) using "
            "KVAllGatherCPFlexInnerAttention.",
            parallel_dims.cp,
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
