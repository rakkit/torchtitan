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

# The hidden matrices' lr of the recipe (the study's one lr across sizes, batch 128 x 4096).
_LR = 0.01562


def moe_template_config() -> Trainer.Config:
    # The production MoE recipe, model and optimizer side, on the mis-8b flavor (as
    # reproduce_cfgs/configs/0.5.4-debug/production-moe-336m.py; a run config still
    # writes every setting). Model: RMSNorm gains, no depth or residual scaling,
    # every hidden matrix (attention with its gate, FFN, experts) scaled-orthogonal
    # at std 4 (amphione has no step-1 rescale: the x4 is in the init), the router
    # gate too, embedding scion-normal std 0.02, head scion-normal std 1.
    m = model_registry("mis-8b")
    m.norm_type = m.layer.norm_type = m.layer.attention.norm_type = "rmsnorm"
    m.layer.depth_init = "identity"
    m.layer.residual_scale = "identity"
    m.layer.init_gate_as_residual = False
    m.first_in_init_fn_type, m.first_in_init_std = "scion_normal_input", 0.02
    m.final_out_init_fn_type, m.final_out_init_std = "scion_normal_output", 1.0
    for module, names in (
        (m.layer.attention, ("wq", "wk", "wv", "wo", "w_gate")),
        (m.layer.feed_forward, ("w1", "w2", "w3")),
        (m.layer.moe, ("w1", "w2", "w3")),
    ):
        for w in names:
            setattr(module, f"{w}_init_fn_type", "scaled_orthogonal")
            setattr(module, f"{w}_init_std", 4.0)
    m.layer.moe.router_init_fn_type = "scaled_orthogonal"
    m.layer.moe.router_init_std = 4.0
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-120b",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=m,
        # No `dataloader=` default. Every recipe assigns `config.dataloader` from
        # `components/data/mix.make_pretrain_dataloader_config`, and a template default
        # that no run uses is a trap: it decides the shape of anything that forgets to.
        # Optimizer: amphione on every hidden matrix (the experts and the router
        # included; the router gate's row lengths frozen), a sign step on the RMSNorm
        # gains, row-l2 on the embedding and the head. `momentum` of an sgd / muon
        # group is the new gradient's weight (0.05 = beta 0.95); amphione keeps its
        # own momentum (`beta`, `nesterov`).
        optimizer=DiSCOOptimizersContainer.Config(
            groups={
                "gains": {
                    "match": r"(^|\.)(q_norm|k_norm|attention_norm|ffn_norm|norm)\.weight$",
                    "rule": "sgd",
                    "args": {"pre": "identity", "compute": "sign", "scale": "one"},
                    "lr": 0.5 * _LR,
                    "momentum": 0.05,
                    "nesterov": True,
                    "weight_decay": 0.0,
                    "decoupled_wd": False,
                },
                "default": {
                    "lr": _LR,
                    "weight_decay": 0.0,
                    "decoupled_wd": False,
                    "momentum": 1.0,
                    "nesterov": False,
                    "rule": "amphione",
                    "args": {
                        "beta": 0.95,
                        "nesterov": True,
                        "gain_beta": None,
                        "compute": "gram_polar_express",
                        "steps": 6,
                        "polar_express_safety": 1.05,
                        "frozen_length_units": ["moe.router.gate"],
                    },
                },
                "embedding": {
                    "match": r"tok_embeddings\.weight",
                    "rule": "muon",
                    "args": {"compute": "row-l2", "scale": "sqrt_n", "pre": "identity"},
                    "lr": 0.001953,
                    "momentum": 0.05,
                    "nesterov": True,
                    "weight_decay": 0.0,
                    "decoupled_wd": False,
                },
                "lm_head": {
                    "match": r"(output|lm_head)\.weight",
                    "rule": "muon",
                    "args": {
                        "compute": "row-l2",
                        "scale": "inv_sqrt_n",
                        "pre": "identity",
                    },
                    "lr": 0.125,
                    "momentum": 0.01,  # beta 0.99 (batch 128 x 4096)
                    "nesterov": True,
                    "weight_decay": 0.0,
                    "decoupled_wd": False,
                },
            }
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
        loss=ChunkedLossWrapper.Config(num_chunks=8),
        # One schedule for every group: no warmup, sqrt decay over the whole run.
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0, decay_ratio=1.0, decay_type="sqrt", min_lr_factor=0.0
        ),
        training=TrainingConfig(
            # 4 sequences x 4096 tokens, expressed the way 0.5.0 expresses it.
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            steps=100,
            # fp32 with bf16 params and fp32 gradient reduction; no clipping
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            max_norm=0.0,
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
