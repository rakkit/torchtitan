# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified OPT MoE model recipes."""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import LRSchedulersContainer
from torchtitan.config import CommConfig, DebugConfig, PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.opt_moe import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.optimizers.optim import DiSCOOptim
from torchtitan.trainer import Trainer

# The hidden matrices' lr of the recipe (the study's one lr across sizes, batch 128 x 4096).
_LR = 0.01562


# The production init (the optimizer study's MoE recipe): RMSNorm gains, no depth
# or residual scaling, every hidden matrix (attention with its gate, FFN, experts,
# router gate) scaled-orthogonal at std 4 (amphione has no step-1 rescale: the x4
# is in the init), embedding scion-normal std 0.02, head scion-normal std 1.
_PRODUCTION_INIT = dict(
    depth_init="identity",
    embedding_init_std=0.02,
    **{
        f"{w}_init_{k}": v
        for w in ("wqkv", "wo", "w_gate", "w1", "w2", "w3", "router")
        for k, v in (("fn_type", "scaled_orthogonal"), ("std", 4.0))
    },
)


def opt_moe_mis_8b() -> Trainer.Config:
    """mis-8b on the optimizer study's production MoE recipe: amphione on every
    hidden matrix, the production init, FSDP over every GPU, 4 x 4096 tokens per
    microbatch."""
    parallelism = ParallelismConfig(data_parallel_shard_degree=-1)
    model = build_model_config(
        "mis-8b", parallelism=parallelism, seq_len=4096, **_PRODUCTION_INIT
    )
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-120b",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model,
        # No `dataloader=`: every run sets it from
        # `components/data/mix.make_pretrain_dataloader_config`.
        # Optimizer: amphione on every hidden matrix (the experts and the router
        # included; the router gate's row scales frozen), a sign step on the RMSNorm
        # gains, row-l2 on the embedding and the head. `momentum` of an sgd / muon
        # group is the memory weight beta (buf <- beta buf + (1 - beta) g); amphione
        # takes its group's momentum and keeps the averages.
        optim=DiSCOOptim.Config(
            max_norm=float("inf"),  # no clipping (x1.0 exactly)
            # One schedule for every group: no warmup, sqrt decay over the whole run.
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0, decay_ratio=1.0, decay_type="sqrt", min_lr_factor=0.0
            ),
            optimizer=DiSCOOptimizersContainer.Config(
                groups={
                    "gains": {
                        "match": r"(^|\.)(q_norm|k_norm|attention_norm|ffn_norm|norm|enorm|hnorm|mtp_norm)\.weight$",
                        "rule": "sgd",
                        "args": {"pre": "identity", "compute": "sign", "scale": "one"},
                        "lr": 0.5 * _LR,
                        "momentum": 0.95,
                        "nesterov": True,
                        "weight_decay": 0.0,
                    },
                    "default": {
                        "lr": _LR,
                        "weight_decay": 0.0,
                        "momentum": 0.95,
                        "nesterov": True,
                        "rule": "amphione",
                        "args": {
                            "scale_nesterov": False,  # scales: no Nesterov
                            "compute": "gram_polar_express",
                            "steps": 6,
                            "polar_express_safety": 1.05,
                            "frozen_scale_units": ["feed_forward.router.gate"],
                        },
                    },
                    "embedding": {
                        "match": r"tok_embeddings\.weight",
                        "rule": "sgd",
                        "args": {
                            "compute": "row-l2",
                            "scale": "sqrt_n",
                            "pre": "identity",
                        },
                        "lr": 0.001953,
                        "momentum": 0.95,
                        "nesterov": True,
                        "weight_decay": 0.0,
                    },
                    "lm_head": {
                        "match": r"(output|lm_head)\.weight",
                        "rule": "sgd",
                        "args": {
                            "compute": "row-l2",
                            "scale": "inv_sqrt_n",
                            "pre": "identity",
                        },
                        "lr": 0.125,
                        "momentum": 0.99,  # batch 128 x 4096
                        "nesterov": True,
                        "weight_decay": 0.0,
                    },
                }
            ),
        ),
        # Plain chunked cross-entropy (an MoE auxiliary loss, when configured,
        # injects its gradient at the router). ChunkedLossWrapper splits the
        # token dim and runs lm_head + CE per chunk (the engine hands it the
        # head: set_lm_head), so the [T, vocab] logits never exist all at once.
        # With vocab=201088 that is the single largest activation in the step.
        # MEASURED at T=4096: peak fwd+bwd 12.02 GiB -> 3.02 GiB (-75%), with
        # bit-identical loss. Every token count we run (1024/4096/8192/40960)
        # divides by 8.
        loss=ChunkedLossWrapper.Config(
            num_chunks=8,
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model.vocab_size),
        ),
        # fp32 with bf16 params and fp32 gradient reduction
        precision=PrecisionConfig(
            dtype="float32", param_dtype="bfloat16", reduce_dtype="float32"
        ),
        training=TrainingConfig(
            # 4 sequences x 4096 tokens
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            steps=100,
            # Accelerator graphs need fixed shapes and no host sync inside the
            # captured region: off for EP with the all-to-all dispatch (it syncs
            # with the host). Without EP the local dispatch has no host sync;
            # graphs on there are not measured yet.
            disable_accelerator_graphs=True,
        ),
        checkpointer=CheckpointManager.Config(
            interval=50,
            # The final save is a full one (optimizer included); with
            # last_save_model_only it would be the model at export_dtype (fp32:
            # the master weights).
            last_save_model_only=False,
            export_dtype="float32",
            # "async" stages to CPU and writes from a thread in the training
            # process; "async_with_pinned_mem" writes from its own process and
            # keeps a pinned host copy of the rank's shard. Measured (16 nodes,
            # 250 GB checkpoint): ~2.4-2.9 s added per save with "async"
            # (~7 s while the previous write still runs), ~0.2-0.4 s with
            # pinned memory, at ~4 GB pinned host memory per GPU.
            async_mode="async_with_pinned_mem",
        ),
        # torch_remat's default regions (code outside a region is recomputed)
        activation_remat=DefaultActivationRematConfig(),
        parallelism=parallelism,
        # FSDP's collectives at a fixed 16 CTAs and 16 SMs kept free for them
        # (README "Communication").
        comm=CommConfig(nccl_ctas={"dp_shard": 16}, sm_carveout=16),
        # scripts/checkpoint_conversion/convert_to_hf.py reconstructs the model
        # spec from this file, so opt_moe runs always write it.
        debug=DebugConfig(save_config_file="job_config.json"),
    )
