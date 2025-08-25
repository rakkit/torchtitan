# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

import itertools
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.config.job_config import ActivationCheckpoint as ACConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
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

        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            tensor_parallel_only_attention=job_config.parallelism.tensor_parallel_only_attention,
            enable_approx_mid_norm_for_tensor_parallel=job_config.parallelism.enable_approx_mid_norm_for_tensor_parallel,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )
    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model)

    if parallel_dims.fsdp_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
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
        apply_ddp(
            model,
            world_mesh,
            enable_compile=model_compile_enabled,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )

    return model


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    tensor_parallel_only_attention: bool = False,
    enable_approx_mid_norm_for_tensor_parallel: bool = False,
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

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
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
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(),
        }
        if isinstance(transformer_block, BitNetTransformerBlock):
            layer_plan.update(
                {
                    "attention.wo_norm": SequenceParallel(),
                    "feed_forward.w2_norm": SequenceParallel(),
                }
            )

        """
        PrepareMidNormInputOutput() is the "correct" way to do it,
        We need to either  disable the Async TP and compile together.
        Or use tensor_parallel_only_attention + compile + no_async_tp.

        SequenceParallel(sequence_dim=-1) is a hack to make it work with Async TP and compile together.
        This is an "approximate" solution, and might lead to error at some point.
        """
        if not isinstance(transformer_block.attention.o_norm, nn.Identity):
            if enable_approx_mid_norm_for_tensor_parallel:
                layer_plan["attention.o_norm"] = SequenceParallel(sequence_dim=-1)
            else:
                layer_plan["attention.o_norm"] = PrepareMidNormInputOutput()

        if not tensor_parallel_only_attention and not isinstance(
            transformer_block.feed_forward.out_norm, nn.Identity
        ):
            if enable_approx_mid_norm_for_tensor_parallel:
                layer_plan["feed_forward.out_norm"] = SequenceParallel(sequence_dim=-1)
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


# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}


def _apply_ac_to_transformer_block(
    module: nn.Module, ac_config: ACConfig, *, base_fqn: Optional[str] = None
):
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        mm_funs = [torch.ops.aten.mm.default]

        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        if isinstance(module, BitNetTransformerBlock):
            _save_list.update(
                {
                    torch.ops.torchao.scaled_int8_mm.default,
                    torch.ops.aten._int_mm.default,
                    torch.ops.aten.mean.default,
                }
            )
            mm_funs.extend(
                [
                    torch.ops.torchao.scaled_int8_mm.default,
                    torch.ops.aten._int_mm.default,
                ]
            )

        mm_recompute_shapes = set()
        if len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
            for module_fqn, submod in module.named_modules():
                fqn = module_fqn
                if base_fqn is not None:
                    fqn = f"{base_fqn}.{module_fqn}"
                if not any(
                    filter_fqn in fqn
                    for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
                ):
                    continue
                if not isinstance(submod, nn.Linear):
                    raise ValueError(
                        "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                        f"a nn.Linear, but got: {submod}"
                    )
                out_f, in_f = submod.weight.shape
                mm_recompute_shapes.add((in_f, out_f))
            logger.debug(
                f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
            )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func in mm_funs:
                    if args[1].shape in mm_recompute_shapes:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (
                    func in mm_funs and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(ac_config.selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module


def apply_ac(model: nn.Module, ac_config: ACConfig):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block, ac_config, base_fqn=f"layers.{layer_id}"
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
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

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    transformer_blocks = model.layers.items()
    if model.model_args.num_mtp_modules > 0:
        transformer_blocks = itertools.chain(
            transformer_blocks, model.mtp_layers.items()
        )
    for layer_id, transformer_block in transformer_blocks:
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    fully_shard(model, **fsdp_config)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")


class PrepareMidNormInputOutput(torch.distributed.tensor.parallel.ParallelStyle):
    """
    when `norm_everywhere=True`, we need to particularly handle
    the mid-norm in mid of FFN. (and norm before out-proj in attention)

    We need to
    1. Replicate[gather] the input to the norm layer,
    2. Run the norm layer
    3. Shard the output back

    But it does not work with async TP and compile together. (for FFN)

    ###########
    Insteard, Can we use SequenceParallel(dim=-1) here?
    it seems to be working + loss and norm are aligned
    """

    def __init__(
        self,
        shard_dim: int = -1,
    ):
        # fixed layouts for the MLP mid-norm case
        self._in_layout = (Shard(shard_dim),)
        self._desired_in = (Replicate(),)
        self._out_layout = (Replicate(),)
        self._desired_out = (Shard(shard_dim),)

    def _prep_in(self, inputs, mesh):
        x, *rest = inputs
        if not isinstance(x, torch.distributed.tensor.DTensor):
            x = torch.distributed.tensor.DTensor.from_local(
                x, mesh, self._in_layout, run_check=False
            )
        if self._in_layout != self._desired_in:
            x = x.redistribute(placements=self._desired_in)
        return (x.to_local(), *rest)  # hand local tensor to module

    def _prep_out(self, outputs, mesh):
        if not isinstance(outputs, torch.distributed.tensor.DTensor):
            outputs = torch.distributed.tensor.DTensor.from_local(
                outputs, mesh, self._out_layout, run_check=False
            )
        if self._out_layout != self._desired_out:
            outputs = outputs.redistribute(placements=self._desired_out)
        return outputs.to_local()  # keep local shard

    def _apply(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        module.register_forward_pre_hook(  # gather before norm
            lambda m, i: self._prep_in(i, mesh)
        )
        module.register_forward_hook(  # re-shard after norm
            lambda m, i, o: self._prep_out(o, mesh)
        )
        return module
