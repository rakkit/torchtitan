# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import pytest
import torch
from torchtitan import utils
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import get_train_spec

sys.path.insert("..")
from train import _batch_backward, _get_batch


@pytest.fixture
def job_config(
    data_parallel_replicate_degree,
    data_parallel_shard_degree,
    tensor_parallel_degree,
    pipeline_parallel_degree,
    context_parallel_degree,
    enable_loss_parallel,
    enable_async_tensor_parallel,
    activation_checkpoint_mode,
    enable_float8_linear,
):
    config = JobConfig()
    arg_list = [
        # Job
        "--job.dump_folder",
        "/tmp/test_tt/",
        # Model
        "--model.name",
        "llama",
        "--model.flavor",
        "debugmodel",
        "--model.tokenizer_path",
        "./tests/assets/test_tiktoken.model",
        # Training
        "--training.dataset",
        "c4_test",
        "--training.dataset_path",
        "./tests/assets/c4_test",
        "--training.batch_size",
        "1",
        "--training.seq_len",
        "128",
        # Parallelism settings
        "--training.data_parallel_replicate_degree",
        str(data_parallel_replicate_degree),
        "--training.data_parallel_shard_degree",
        str(data_parallel_shard_degree),
        "--training.tensor_parallel_degree",
        str(tensor_parallel_degree),
        "--experimental.pipeline_parallel_degree",
        str(pipeline_parallel_degree),
        "--experimental.context_parallel_degree",
        str(context_parallel_degree),
        # Determinism
        "--training.seed",
        str(0),
        "--training.deterministic",
        # Activation checkpoint
        "--activation_checkpoint.mode",
        activation_checkpoint_mode,
    ]
    if not enable_loss_parallel:
        arg_list.append("--training.disable_loss_parallel")
    if enable_async_tensor_parallel:
        arg_list.append("--experimental.enable_async_tensor_parallel")
    if enable_float8_linear:
        arg_list.append("--float8.enable_float8_linear")

    config.parse_args(arg_list)
    return config


# @pytest.fixture
# def transformer_config():
#     return ModelArgs(
#         dim=512,
#         vocab_size=10000,
#         n_layers=6,
#         n_heads=8,
#         n_kv_heads=4,
#         max_seq_len=512,
#         rope_theta=50000.0,
#     )


class TestGradientAccumulation:
    @pytest.fixture(autouse=True)
    def setup_class(self, transformer_config):
        self.model_args = transformer_config
        self.batch_size = 1
        # TODO vary this
        self.world_size = 1
        self.dim = self.model_args.dim
        self.vocab_size = self.model_args.vocab_size
        self.seq_len = 128
        self.input_ids = torch.arange(self.batch_size * self.seq_len).reshape(
            self.batch_size, self.seq_len
        )

    # def test_gradient_accumulation(self):
    #     model = Transformer(self.model_args)
    #     fixed_init_model(model, min_val=-1, max_val=1)
    #     output = model(**self.input)
    #     expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
    #     assert (
    #         output.shape == expected_shape
    #     ), f"Expected shape {expected_shape}, but got {output.shape}"

    def init_distributed(
        self,
        job_config,
    ):
        world_size = int(os.environ["WORLD_SIZE"])
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
        device_module, device_type = utils.device_module, utils.device_type
        device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(device)
        dist_utils.init_distributed(job_config)

        world_mesh = parallel_dims.build_mesh(device_type=device_type)
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        if parallel_dims.pp_enabled:
            pp_mesh = world_mesh["pp"]

        # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
        dist_utils.set_determinism(
            world_mesh,
            device,
            job_config.training.seed,
            job_config.training.deterministic,
        )

        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.device_type = device_type
        self.device = device
        self.world_mesh = world_mesh
        self.dp_degree, self.dp_rank = dp_degree, dp_rank
        self.pp_mesh = pp_mesh if parallel_dims.pp_enabled else None

    def init_train_spec(self):
        job_config = self.job_config

        train_spec = get_train_spec(job_config.model.name)

        self.train_spec = train_spec

    def build_tokenizer(self):
        train_spec = self.train_spec
        job_config = self.job_config

        tokenizer = train_spec.tokenizer_cls(job_config.model.tokenizer_path)

        self.tokenizer = tokenizer

    def build_dataloader(self, batch_size):
        train_spec = self.train_spec
        job_config = self.job_config
        dp_degree = self.dp_degree
        dp_rank = self.dp_rank
        tokenizer = self.tokenizer

        dataloader = train_spec.build_dataloader_fn(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            job_config=job_config,
        )

        return dataloader

    def init_model(self):
        train_spec = self.train_spec
        job_config = self.job_config
        tokenizer = self.tokenizer
        parallel_dims = self.parallel_dims

        # build model (using meta init)
        model_cls = train_spec.cls
        model_config = train_spec.config[job_config.model.flavor]
        # set the model configs from training inputs:
        # 1. norm type to decide which norm layer to use
        # 2. vocab size from tokenizer
        # 3. max_seq_len base on inputs
        model_config.norm_type = job_config.model.norm_type
        model_config.vocab_size = tokenizer.n_words
        model_config.max_seq_len = job_config.training.seq_len

        with torch.device("meta"):
            model = model_cls.from_model_args(model_config)

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        self.model_config = model_config
        self.model = model
        self.model_converters = model_converters

    def parallelize_model(self):
        device_type = self.device_type
        parallel_dims = self.parallel_dims
        train_spec = self.train_spec
        model = self.model
        pp_mesh = self.pp_mesh
        job_config = self.job_config
        device = self.device
        model_config = self.model_config
        world_mesh = self.world_mesh

        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        if parallel_dims.pp_enabled:
            # apply PT-D Pipeline Parallel
            (
                pp_schedule,
                eval_pp_schedule,
                model_parts,
                has_first_stage,
                has_last_stage,
            ) = train_spec.pipelining_fn(
                model,
                pp_mesh,
                parallel_dims,
                job_config,
                device,
                model_config,
                train_spec.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
            del model

            # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
            # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
            # optimizer, and checkpointing
            for m in model_parts:
                # apply SPMD-style PT-D techniques
                train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            model_parts = [model]

        self.model = None if parallel_dims.pp_enabled else model
        self.model_parts = model_parts
        self.pp_schedule = pp_schedule if parallel_dims.pp_enabled else None
        self.has_first_stage = has_first_stage
        self.has_last_stage = has_last_stage

    def training_step_original(self, data_iterator, train_context):
        device_type = self.device_type
        world_mesh = self.world_mesh
        model_parts = self.model_parts
        job_config = self.job_config
        parallel_dims = self.parallel_dims
        has_last_stage = self.has_last_stage
        has_first_stage = self.has_first_stage
        pp_schedule = self.pp_schedule
        device = self.device
        model = self.model
        train_spec = self.train_spec

        batch = next(data_iterator)
        input_ids, labels = batch
        next_batch = next(data_iterator)
        next_input_ids, next_labels = next_batch
        input_ids = torch.cat([input_ids, next_input_ids])
        labels = torch.cat([labels, next_labels])
        del next_batch, next_input_ids, next_labels

        input_ids = input_ids.to(device_type)
        labels = labels.to(device_type)

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[input_ids, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={input_ids, labels},
                cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with train_context(optional_context_parallel_ctx):
                targets, losses = (labels, []) if has_last_stage else (None, None)
                if has_first_stage:
                    pp_schedule.step(input_ids, target=targets, losses=losses)
                else:
                    pp_schedule.step(target=targets, losses=losses)

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(device)
                if has_last_stage
                else torch.tensor([-1.0], device=device)
            )
        else:
            # Non-PP forward / backward
            with train_context(optional_context_parallel_ctx):
                pred = model(input_ids)
                loss = train_spec.loss_fn(pred, labels)
                # pred.shape=(bs, seq_len, vocab_size)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss

    def training_step_refactored(self, data_iterator, train_context):
        device_type = self.device_type
        model_parts = self.model_parts
        # input_ids = self.input_ids
        # labels = self.labels
        train_spec = self.train_spec
        parallel_dims = self.parallel_dims
        world_mesh = self.world_mesh
        pp_schedule = self.pp_schedule
        has_first_stage = self.has_first_stage
        has_last_stage = self.has_last_stage
        device = self.device
        job_config = self.job_config

        (
            input_ids,
            labels,
            data_loading_time,
            batch_ntokens,
        ) = _get_batch(data_iterator, device_type)
        loss = _batch_backward(
            model_parts,
            input_ids,
            labels,
            train_spec,
            train_context,
            parallel_dims,
            world_mesh,
            pp_schedule,
            has_first_stage,
            has_last_stage,
            device,
            job_config,
        )
        return loss

    def post_training_step(self, loss):
        model_parts = self.model_parts
        job_config = self.job_config
        pp_mesh = self.pp_mesh
        parallel_dims = self.parallel_dims
        world_mesh = self.world_mesh

        # clip gradients
        total_norm = dist_utils.clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            job_config.training.max_norm,
            foreach=True,
            pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
        )

        if (
            parallel_dims.dp_replicate_enabled
            or parallel_dims.dp_shard_enabled
            or parallel_dims.cp_enabled
        ):
            loss = loss.detach()
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss, world_mesh["dp_cp"]),
                dist_utils.dist_max(loss, world_mesh["dp_cp"]),
            )
        else:
            global_avg_loss = global_max_loss = loss.item()

        return total_norm, global_avg_loss, global_max_loss

    def test_gradient_accumulation(self):
        self.init_distributed()
        self.init_train_spec()
        self.build_tokenizer()
        self.init_model()
        self.parallelize_model()

        # Original version

        train_spec = self.train_spec
        model_parts = self.model_parts
        job_config = self.job_config
        model_converters = self.model_converters
        dataloader = self.build_dataloader(self.job_config.training.batch_size)
        parallel_dims = self.parallel_dims

        optimizers = train_spec.build_optimizers_fn(model_parts, job_config)
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
        )
        data_iterator = iter(dataloader)
        train_context = dist_utils.get_train_context(
            parallel_dims.loss_parallel_enabled,
            job_config.experimental.enable_compiled_autograd,
        )

        optimizers.zero_grad()

        loss = self.train_step_original(data_iterator, train_context)
        (
            total_norm_original,
            global_avg_loss_original,
            global_max_loss_original,
        ) = self.post_training_step(loss)
        loss_original = loss.detach()

        del loss, optimizers

        # Refactored version

        train_spec = self.train_spec
        model_parts = self.model_parts
        job_config = self.job_config
        model_converters = self.model_converters
        dataloader = self.build_dataloader(self.job_config.training.batch_size)
        parallel_dims = self.parallel_dims

        optimizers = train_spec.build_optimizers_fn(model_parts, job_config)
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
        )
        data_iterator = iter(dataloader)
        train_context = dist_utils.get_train_context(
            parallel_dims.loss_parallel_enabled,
            job_config.experimental.enable_compiled_autograd,
        )

        optimizers.zero_grad()
        losses_refactored = []

        loss = self.train_step_refactored(data_iterator, train_context)
        losses_refactored.append(loss.detach())

        loss = self.train_step_refactored(data_iterator, train_context)
        losses_refactored.append(loss.detach())

        loss_refactored = torch.mean(torch.stack(losses_refactored))

        (
            total_norm_refactored,
            global_avg_loss_refactored,
            global_max_loss_refactored,
        ) = self.post_training_step(loss_refactored)

        # Compare

        assert torch.close(loss_original, loss_refactored), "loss does not match"
        assert torch.close(
            total_norm_original, total_norm_refactored
        ), "total norm does not match"
        assert torch.close(
            global_avg_loss_original, global_avg_loss_refactored
        ), "global avg loss does not match"
        assert torch.close(
            global_max_loss_original, global_max_loss_refactored
        ), "global max loss does not match"
