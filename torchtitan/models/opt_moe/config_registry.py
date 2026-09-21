# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.trainer import Trainer
from . import model_registry


def moe_template_config() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-120b",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("bsc-1B-7B-opt-g"),
        # No `dataloader=` default. Every recipe assigns `config.dataloader` from
        # `components/data/mix.make_pretrain_dataloader_config`, and a template default
        # that no run uses is a trap: it decides the shape of anything that forgets to.
        optimizer=DiSCOOptimizersContainer.Config(lr=8e-4),
        # Plain chunked cross-entropy. The load-balance term is no longer added
        # here: NormMoE injects its gradient at the layer via LoadBalanceLoss,
        # so the model returns logits alone.
        # ChunkedLossWrapper splits the token dim and runs lm_head + CE per
        # chunk, so the [T, vocab] logits never exist all at once. With
        # vocab=201088 that is the single largest activation in the step.
        # MEASURED at T=4096: peak fwd+bwd 12.02 GiB -> 3.02 GiB (-75%), with
        # bit-identical loss. `trainer.py` already looks one level into
        # `loss_fn.inner` to wire lm_head, and every token count we run
        # (1024/4096/8192/40960) divides by 8.
        loss=ChunkedLossWrapper.Config(num_chunks=8),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            # 4 sequences x 4096 tokens, expressed the way 0.5.0 expresses it.
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            steps=100,
            # CUDA graphs require fixed shapes and no host sync inside the
            # captured region. OPT MoE's token-choice routing computes
            # per-expert token counts on the host for the grouped-GEMM offsets,
            # so the capture is invalid -- upstream notes the same limitation
            # for EP backends that synchronize during dispatch.
            disable_cuda_graphs=True,
        ),
        checkpointer=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
            # Upstream defaults to "disabled" -- a synchronous save on the
            # training thread. "async" stages to CPU and uploads from a
            # background thread instead. Not the pinned-mem variant: that
            # additionally spawns a process and holds a pinned copy of the
            # local shard PER RANK, which is not worth it unmeasured.
            async_mode="async",
        ),
        # Upstream replaced the mode/selective_ac_option pair with a config
        # class per policy; per-op SAC is now SelectiveAC.Config.
        activation_checkpoint=SelectiveAC.Config(),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        # scripts/checkpoint_conversion/convert_to_hf.py reconstructs the model
        # spec from this file, so opt_moe runs always write it.
        debug=DebugConfig(save_config_file="job_config.json"),
    )
