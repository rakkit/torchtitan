# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    # parallelize_module, # we wrap it at the end of this file
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.expert_parallel import ExpertParallel, TensorParallel
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import (
    apply_ac,
    apply_ddp,
    PrepareMidNormInputOutput,
)
from torchtitan.models.llama3.model.bitnet_model import BitNetTransformerBlock
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    world_mesh = parallel_dims.world_mesh
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if (
        job_config.parallelism.context_parallel_degree > 1
        and model.model_args.use_flex_attn
    ):
        raise NotImplementedError("CP support for FlexAttention is still in progress.")

    if parallel_dims.tp_enabled:
        if "bitnet" in job_config.model.converters:
            raise RuntimeError("BitNet currently does not support tensor parallelism")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_approx_mid_norm_for_tensor_parallel=job_config.parallelism.enable_approx_mid_norm_for_tensor_parallel,
            tensor_parallel_only_attention=job_config.parallelism.tensor_parallel_only_attention,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])
    tp_only_attention = job_config.parallelism.tensor_parallel_only_attention
    # I dont think we need to apply TP for MOE?

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled and parallel_dims.ep_enabled
                else None
            ),
            tp_only_attention=tp_only_attention,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )
    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            dp_mod_ep_mesh=(
                world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
                if dp_mod_ep_mesh_dim_names
                else None
            ),
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = world_mesh
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_approx_mid_norm_for_tensor_parallel: bool = False,
    tensor_parallel_only_attention: bool = False,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        (
            rowwise_parallel,
            colwise_parallel,
            prepare_module_input,
            prepare_module_output,
        ) = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
            None,
        )
    else:
        (
            rowwise_parallel,
            colwise_parallel,
            prepare_module_input,
            prepare_module_output,
        ) = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
            PrepareModuleOutput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
        }
        if not isinstance(transformer_block.attention.o_norm, nn.Identity):
            if enable_approx_mid_norm_for_tensor_parallel:
                layer_plan["attention.o_norm"] = SequenceParallel(sequence_dim=-1)
            else:
                layer_plan["attention.o_norm"] = PrepareMidNormInputOutput()

        if isinstance(transformer_block, BitNetTransformerBlock):
            layer_plan.update(
                {
                    "attention.wo_norm": SequenceParallel(),
                    "feed_forward.w2_norm": SequenceParallel(),
                }
            )
        # dont want to bother the Mid-norm for now
        if not transformer_block.moe_enabled and not tensor_parallel_only_attention:
            layer_plan.update(
                {
                    "ffn_norm": SequenceParallel(),
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

            if not isinstance(transformer_block.feed_forward.out_norm, nn.Identity):
                if enable_approx_mid_norm_for_tensor_parallel:
                    layer_plan["feed_forward.out_norm"] = SequenceParallel(
                        sequence_dim=-1
                    )
                else:
                    layer_plan["feed_forward.out_norm"] = PrepareMidNormInputOutput()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
    tp_only_attention: bool = False,
    enable_approx_mid_norm_for_tensor_parallel: bool = False,
):
    """
    I will disble moe EP_TP for now.
    To make it work, we need
    1) Takeing the mid-norm things same same dense model
    2) We need to name the MoE layer as "moe" and dense layer as "feed_forward", we cannot use the
       same name "feed_forward" for both.

    """
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_only_attention:
            tp_mesh = None

        if tp_mesh is not None:
            # TODO(JSC): DO we really want to run TP for MoE?
            raise NotImplementedError(
                "Maybe we dont need TP for MoE, ->use --parallelism.tensor_parallel_only_attention"
            )

            # moe_layer_plan = {
            #     # input / output sharding on the seqlen dim
            #     # all-gather for input, reduce-scatter for output
            #     "feed_forward": PrepareModuleInputOutput(
            #         input_layouts=(Shard(1),),
            #         desired_input_layouts=(Replicate(),),
            #         use_local_input=True,
            #         output_layouts=(Partial(),),
            #         desired_output_layouts=(Shard(1),),
            #     ),
            #     # replicate computation for the router
            #     "feed_forward.router.gate": NoParallel(),
            #     # input Replicate, output Partial
            #     "feed_forward.shared_experts": TensorParallel(),
            # }
            # parallelize_module(
            #     module=transformer_block,
            #     device_mesh=tp_mesh,
            #     parallelize_plan=moe_layer_plan,
            # )

        # if ep_mesh is not None:
        experts_mesh, experts_plan = None, None
        if ep_mesh is None:  # TP only
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None:  # EP only
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
            transformer_block.feed_forward.experts.ep_enable = True
            total_experts = transformer_block.feed_forward.experts.num_experts
            ep_world_size = ep_mesh.size()
            ep_per_rank = total_experts // ep_world_size
            transformer_block.feed_forward.experts.expert_per_rank = ep_per_rank
            transformer_block.feed_forward.experts.ep_size = ep_world_size

        else:  # EP + TP
            # DONT THINK WE NEED THIS
            # TODO(JSC): DO we really want to run TP for MoE?
            raise NotImplementedError("Maybe we dont need TP for MoE")
            # experts_mesh = ep_tp_mesh
            # experts_plan = ExpertTensorParallel(tp_mesh=tp_mesh, ep_mesh=ep_mesh)

        if experts_mesh:
            parallelize_module(
                module=transformer_block.feed_forward.experts,
                device_mesh=experts_mesh,
                parallelize_plan=experts_plan,
            )


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    import torch._dynamo

    for layer_id, transformer_block in model.layers.named_children():
        torch._dynamo.config.capture_scalar_outputs = True
        # torch._dynamo.config.suppress_errors = True
        # torch._dynamo.config.cache_size_limit = 16
        transformer_block = torch.compile(
            transformer_block,
            fullgraph=True,
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    dp_mod_ep_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for layer_id, transformer_block in model.layers.items():
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(layer_id) < len(model.layers) - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

        # NOTE: in an MoE layer, the router and the shared experts
        #       are sharded together with the TransformerBlock
        if transformer_block.moe_enabled and dp_mod_ep_mesh:
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = dp_mod_ep_mesh
            fully_shard(
                transformer_block.feed_forward.experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
            )

            # NOTE: # Although the FSDP sharding of experts is done on a mesh of
            #       a different size than other parameters, the gradient division
            #       factor should be consistent with data.
            transformer_block.feed_forward.experts.set_reduce_scatter_divide_factor(
                gradient_divide_factor,
            )
            transformer_block.feed_forward.experts.ep_enable = True

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)


from typing import Optional, Union

# this is missing at pytorch 2.6
# for pytorch 2.7, we can import from pytorch directly
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/style.py#L704
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


class PrepareModuleInputOutput(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, tuple[Optional[Placement]]]
        ] = None,
        input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        use_local_input: bool = False,
        output_layouts: Union[Placement, tuple[Placement]],
        desired_output_layouts: Union[Placement, tuple[Placement]],
        use_local_output: bool = True,
    ):
        self.prepare_module_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.prepare_module_input._apply(module, device_mesh)
        self.prepare_module_output._apply(module, device_mesh)

        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.prepare_module_input.input_layouts}, "
        tmpstr += (
            f"desired_input_layouts={self.prepare_module_input.desired_input_layouts}, "
        )
        tmpstr += (
            f"input_kwarg_layouts={self.prepare_module_input.input_kwarg_layouts}, "
        )
        tmpstr += f"desired_input_kwarg_layouts={self.prepare_module_input.desired_input_kwarg_layouts}, "
        tmpstr += f"use_local_input={self.prepare_module_input.use_local_output}, "
        tmpstr += f"output_layouts={self.prepare_module_output.output_layouts}, "
        tmpstr += f"desired_output_layouts={self.prepare_module_output.desired_output_layouts}, "
        tmpstr += f"use_local_output={self.prepare_module_output.use_local_output}"
        tmpstr += ")"
        return tmpstr


# https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/api.py
# see commit https://github.com/pytorch/pytorch/commit/5633283574c458bd6a3cbb6a0a890f0cb9c8b2b5
# There is a werid bugs in `parallelize_module` for a while that it call the function `_validate_tp_mesh_dim`,
# how ever this function only exam the "master" mesh, but did not check the 'sub-mesh'
# thereforce, if we wana `parallelize_module` to work on device_mesh that is a sub-mesh, it will raise the error.

import warnings
from fnmatch import fnmatch
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle


def parallelize_module(  # type: ignore[return]
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, dict[str, ParallelStyle]]] = None,
    *,
    src_data_rank: Optional[int] = 0,
) -> nn.Module:
    # torch._C._log_api_usage_once("torch.distributed.tensor.parallel.parallelize_module")

    device_mesh = device_mesh or _mesh_resources.get_current_mesh()

    if parallelize_plan is None:
        warnings.warn(
            "No parallelize_plan is provided and auto-parallel is not supported "
            "at the moment, so this parallelize_module call will do nothing."
        )
        return module

    # note: The RNG tracker will be initialized in distribute_tensor() call if it hasn't
    # been initialized.

    if isinstance(parallelize_plan, ParallelStyle):
        parallelize_plan.src_data_rank = src_data_rank
        return parallelize_plan._apply(module, device_mesh)
    elif isinstance(parallelize_plan, dict):
        for module_path, parallelize_style in parallelize_plan.items():
            if module_path == "":
                # shortcut: empty string means to apply the plan to the current module
                parallelize_module(module, device_mesh, parallelize_style)
                continue

            path_splits = module_path.split(".")
            # Instead of blindly popping tokens, first check the match,
            # we only consume/pop the token if we found a match.
            token = path_splits[0]

            matched_children = list(
                filter(
                    # `t[0]` is child name
                    lambda t: fnmatch(t[0], token),
                    module.named_children(),
                )
            )
            if not matched_children:
                # No match at this level. Log a warning and process next plan entry.
                warnings.warn(
                    f"Parallelize plan key '{module_path}' could not be resolved: "
                    f"no submodule matching token '{token}' in module {module}, "
                    f"skipping this plan entry."
                )
                continue

            # Now that we have a match, we can consume the token.
            path_splits.pop(0)
            # apply the plan to all matched submodules
            for _, submodule in matched_children:
                if path_splits:
                    # we haven't reached the leaf, apply in dict style
                    leaf_path = ".".join(path_splits)  # rest of the path after `token`
                    parallelize_module(
                        submodule,
                        device_mesh,
                        {leaf_path: parallelize_style},
                        src_data_rank=src_data_rank,
                    )
                else:
                    # otherwise, directly apply style to this submodule
                    parallelize_module(
                        submodule,
                        device_mesh,
                        parallelize_style,
                        src_data_rank=src_data_rank,
                    )
        return module
    else:
        raise TypeError(  # pyre-ignore[7]
            "Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for"
            f" parallelize_plan, {type(parallelize_plan)} found!"
        )
