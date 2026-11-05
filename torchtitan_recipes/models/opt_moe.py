# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The OPT MoE training template: the production MoE recipe on mis-8b, which
run configs start from (``config = moe_template_config()``)."""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import LRSchedulersContainer
from torchtitan.config import DebugConfig, PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.opt_moe import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.optimizers.container import DiSCOOptimizersContainer
from torchtitan.optimizers.optim import DiSCOOptim
from torchtitan.trainer import Trainer

# The hidden matrices' lr of the recipe (the study's one lr across sizes, batch 128 x 4096).
_LR = 0.01562


def _production_model_args() -> dict:
    """The production MoE recipe's model arguments (as
    reproduce_cfgs/configs/0.5.4-debug/production-moe-336m.py; a run config still
    writes every setting): RMSNorm gains, no depth or residual scaling, every
    hidden matrix (attention with its gate, FFN, experts, router gate)
    scaled-orthogonal at std 4 (amphione has no step-1 rescale: the x4 is in the
    init), embedding scion-normal std 0.02, head scion-normal std 1."""

    def inits(*names: str) -> dict:
        out = {}
        for w in names:
            out[f"{w}_init_fn_type"] = "scaled_orthogonal"
            out[f"{w}_init_std"] = 4.0
        return out

    return dict(
        norm_type="rmsnorm",
        depth_init="identity",
        residual_scale="identity",
        init_gate_as_residual=False,
        first_in_init_fn_type="scion_normal_input",
        first_in_init_std=0.02,
        final_out_init_fn_type="scion_normal_output",
        final_out_init_std=1.0,
        attention=inits("wqkv", "wo", "w_gate"),
        feed_forward=inits("w1", "w2", "w3"),
        moe=inits("w1", "w2", "w3", "router"),
    )


def moe_template_config() -> Trainer.Config:
    parallelism = ParallelismConfig(data_parallel_shard_degree=-1)
    model = build_model_config(
        "mis-8b", parallelism=parallelism, seq_len=4096, **_production_model_args()
    )
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-120b",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model,
        # No `dataloader=` default. Every recipe assigns `config.dataloader` from
        # `components/data/mix.make_pretrain_dataloader_config`, and a template default
        # that no run uses is a trap: it decides the shape of anything that forgets to.
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
                        "match": r"(^|\.)(q_norm|k_norm|attention_norm|ffn_norm|norm)\.weight$",
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
            # with the host) and for gnoise (refuses them). Without EP the local
            # dispatch has no host sync; graphs on there are not measured yet.
            disable_accelerator_graphs=True,
        ),
        checkpointer=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            # A final model-only save (last_save_model_only) keeps the fp32
            # master weights.
            export_dtype="float32",
            # Upstream defaults to "disabled" -- a synchronous save on the
            # training thread. "async" stages to CPU and uploads from a
            # background thread instead. Not the pinned-mem variant: that
            # additionally spawns a process and holds a pinned copy of the
            # local shard PER RANK, which is not worth it unmeasured.
            async_mode="async",
        ),
        # torch_remat's default regions (code outside a region is recomputed)
        activation_remat=DefaultActivationRematConfig(),
        parallelism=parallelism,
        # scripts/checkpoint_conversion/convert_to_hf.py reconstructs the model
        # spec from this file, so opt_moe runs always write it.
        debug=DebugConfig(save_config_file="job_config.json"),
    )
