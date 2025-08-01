# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import copy

import torch
from torch.distributed.tensor.parallel import loss_parallel
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    ScheduleZBVZeroBubble,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline import (
    build_pipeline_schedule,
    generate_split_points,
    stage_ids_this_rank,
)
from torchtitan.protocols.train_spec import ParallelizeFunction
from torchtitan.tools.logging import logger
from functools import partial
from ..model.args import TransformerModelArgs


def pipeline_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_config: TransformerModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.world_mesh["pp"]

    stages, model_parts = pipeline_llama_manual_split(
        model, pp_mesh, parallel_dims, job_config, device, model_config
    )

    # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
    # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
    # optimizer, and checkpointing
    for i, m in enumerate(model_parts):
        # apply SPMD-style PT-D techniques
        m = parallelize_fn(m, parallel_dims, job_config)
        model_parts[i] = m
        # NOTE: this is to update the model in the stage
        #       in case the model is modified e.g. by torch.compile
        stages[i].submod = m

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def pipeline_llama_manual_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_config: TransformerModelArgs,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    parallelism_config = job_config.parallelism

    tp_size = parallel_dims.tp  # need tp_size for seq_len division
    # for DDP only, hidden will be float32, otherwise BF16
    ddp_only = not parallel_dims.dp_shard_enabled

    if model_config.num_mtp_modules > 0:
        # lets disable MTP for now
        raise ValueError("MTP is not supported yet")
    splits = parallelism_config.pipeline_parallel_split_points or generate_split_points(
        parallelism_config.pipeline_parallel_schedule,
        parallel_dims.pp,
        model_config.n_layers + model_config.num_mtp_modules,
        parallelism_config.pipeline_parallel_layers_per_stage,
        num_mtp_layers=model_config.num_mtp_modules,
    )

    def _build_stage(
        stage_idx: int,
        start_layer: str | None,
        stop_layer: str | None,
        is_first: bool = False,
        is_last: bool = False,
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)
        dim = model.tok_embeddings.weight.shape[1]
        vocab_size = model.tok_embeddings.weight.shape[0]

        if not is_first:
            model.tok_embeddings = None

        drop_layers = start_layer is not None
        for name in list(model.layers.keys()):
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                del model.layers[name]
        if model_config.num_mtp_modules > 0:
            for name in list(model.mtp_layers.keys()):
                # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
                if f"mtp_layers.{name}" == start_layer:
                    drop_layers = False
                if f"mtp_layers.{name}" == stop_layer:
                    drop_layers = True
                if drop_layers:
                    del model.mtp_layers[name]

        if not is_last and not stop_layer.startswith("mtp_layers."):
            model.norm = None
            model.output = None

        # ==== Here we infer the input and output shapes for the stage ====
        # see https://github.com/pytorch/torchtitan/issues/1492
        pp_mbs = job_config.parallelism.pipeline_parallel_microbatch_size
        seq_len = job_config.training.seq_len
        tp_seq_len = seq_len // max(tp_size, 1)
        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )
        # here is hard-coded for now, that we assume all hidden have Shard(1) placement on TP mesh

        hidden_dtype = torch.bfloat16 if not ddp_only else torch.float32
        empty_tensor = partial(torch.empty, dtype=hidden_dtype, device="meta")

        if is_first:
            example_input = empty_tensor((pp_mbs, seq_len), dtype=torch.long)
            example_output = empty_tensor((pp_mbs, tp_seq_len, dim))
        elif is_last:
            # Why we dont need this?
            # # output_dim = (
            #     vocab_size if not loss_parallel_enabled else vocab_size // tp_size
            # )
            example_input = empty_tensor((pp_mbs, tp_seq_len, dim))
            example_output = empty_tensor((pp_mbs, seq_len, vocab_size))
        else:
            example_input = empty_tensor((pp_mbs, tp_seq_len, dim))
            example_output = empty_tensor((pp_mbs, tp_seq_len, dim))
        # ============================================================
        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
            input_args=(example_input,),
            output_args=(example_output,),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    stages = []
    models = []

    schedule_class = get_schedule_class(parallelism_config.pipeline_parallel_schedule)
    style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"

    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style=style):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models
