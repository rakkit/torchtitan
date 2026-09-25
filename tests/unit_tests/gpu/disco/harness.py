# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a small opt_moe model and a DiSCO container the way the trainer does,
without the trainer: no dataloader, no tokenizer, no forward pass.

Gradients are synthetic (a deterministic function of the parameter name and the
step), so the optimizer can be exercised under any parallelism on any world
size and compared to a single-process reference.
"""

import dataclasses
import zlib

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.models.opt_moe import model_registry
from torchtitan.models.opt_moe.config_registry import moe_template_config
from torchtitan.optimizers.container import DiSCOOptimizersContainer

# The production rules from cfg-cpswa.py: embeddings, lm_head and router gates
# use the row-separable rules and never see the LMO.
EMBED_GROUPS = {  # `optimizer.groups` entries (`default` is added by trainer_config)
    "embedding": {
        "match": r"tok_embeddings\.weight",
        "args": {"compute": "row-l2", "scale": "sqrt_n"},
    },
    "lm_head": {
        "match": r"(output|lm_head)\.weight",
        "args": {"compute": "row-l2", "scale": "inv_sqrt_n"},
    },
}


def tiny_model_config(
    *,
    moe: bool = True,
    n_layers: int = 2,
    dim: int = 64,
    vocab_size: int = 256,
    num_experts: int = 4,
    ffn_hidden_dim: int = 128,
    moe_hidden_dim: int = 32,
    fuse_qkv: bool = False,
    fuse_w13: bool = False,
    flavor: str = "mis-8b-proxy",
):
    """mis-8b-proxy's architecture shrunk until a 4-rank test runs in seconds."""
    cfg = model_registry(flavor)
    cfg.dim = dim
    cfg.n_layers = n_layers
    cfg.vocab_size = vocab_size
    cfg.swa_pattern = ("SF" * n_layers)[:n_layers]
    cfg.rope_pattern = ("RN" * n_layers)[:n_layers]
    cfg.max_context_length = 64
    cfg.rope = dataclasses.replace(cfg.rope, dim=32, max_context_length=64)
    layer = cfg.layer
    layer.n_dense_layers = 0 if moe else n_layers
    layer.feed_forward.hidden_dim = ffn_hidden_dim
    layer.moe.hidden_dim = moe_hidden_dim
    layer.moe.num_experts = num_experts
    layer.moe.num_shared_experts = 1
    layer.moe.top_k = 2
    attn = layer.attention
    attn.n_heads = 2
    attn.n_kv_heads = 1
    attn.head_dim = 32
    attn.fuse_qkv = fuse_qkv
    layer.feed_forward.fuse_w13 = fuse_w13
    attn.sliding_window_size = 16
    return cfg


def trainer_config(
    model_cfg,
    *,
    dp_replicate: int = 1,
    dp_shard: int = 1,
    cp: int = 1,
    ep: int = 1,
    lr: float = 0.1,
    momentum: float = 0.1,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    decoupled_wd: bool = False,
    compute: str = "newtonschulz5",
    steps: int = 5,
    scale: str = "spectral",
    pre: str = "identity",
    aus: bool = False,
    rule: str | None = None,
    rule_args: dict | None = None,
    extra_groups: dict | None = None,
    log_norm_freq: int = 0,
    norms_to_log=("all",),
    gram_level: int = 0,
    communication_dtype: str = "float32",
):
    """A Trainer.Config whose model config has been expanded for `parallelism`.
    `rule` overrides the default group's rule; `rule_args` merge over its
    args, a None value dropping that arg."""
    config = moe_template_config()
    config.model = model_cfg
    par = config.parallelism
    par.data_parallel_replicate_degree = dp_replicate
    par.data_parallel_shard_degree = dp_shard
    par.context_parallel_degree = cp
    par.expert_parallel_degree = ep
    config.training.max_context_length = 64
    config.training.num_tokens_per_microbatch_per_dp_rank = 64
    config.training.steps = 10
    config.metrics.log_freq = 1
    opt = config.optimizer
    opt.groups = {
        "default": {
            "lr": lr,
            "weight_decay": weight_decay,
            "decoupled_wd": decoupled_wd,
            "momentum": momentum,
            "nesterov": nesterov,
            "rule": rule or ("aus" if aus else "muon"),
            "args": {
                k: v
                for k, v in {
                    "compute": compute,
                    "steps": steps,
                    "scale": scale,
                    "pre": pre,
                    **(rule_args or {}),
                }.items()
                if v is not None
            },
        },
        **{k: dict(v) for k, v in EMBED_GROUPS.items()},
        **(extra_groups or {}),
    }
    opt.log_norm_freq = log_norm_freq
    opt.norms_to_log = list(norms_to_log)
    opt.gram_level = gram_level
    opt.communication_dtype = communication_dtype
    config.model.apply_parallelism_transforms(parallelism=par)
    config.model.update_from_config(config=config)
    return config


def parallel_dims(
    *,
    world_size: int,
    dp_replicate: int = 1,
    dp_shard: int = 1,
    cp: int = 1,
    ep: int = 1,
    device_type: str = "cuda",
) -> ParallelDims:
    # `torchtitan.tools.utils.device_type` falls back to "cuda" when no
    # accelerator is visible; a CPU/gloo run needs the mesh built on "cpu".
    from torchtitan.distributed import parallel_dims as parallel_dims_mod

    parallel_dims_mod.device_type = device_type
    pd = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=dp_shard,
        cp=cp,
        tp=1,
        pp=1,
        ep=ep,
        world_size=world_size,
    )
    pd.build_mesh()
    return pd


def build_model_and_optimizer(
    config, pd: ParallelDims, device: torch.device, dump_folder: str
):
    """The training engine's lifecycle: meta build -> parallelize -> to_empty -> init."""
    with torch.device("meta"):
        model = config.model.build()
    model = model.parallelize(
        parallel_dims=pd,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=None,
        ac_config=None,
        dump_folder=dump_folder,
    )
    with get_spmd_context(parallel_dims=pd):
        model.to_empty(device=device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)
        model.train()
    container = DiSCOOptimizersContainer(
        config.optimizer, model_parts=[model], parallel_dims=pd, output_dir=dump_folder
    )
    return model, container


def full_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t


def trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def synthetic_grad(name: str, shape, step: int, *, seed: int = 0, scale: float = 1e-2):
    """Full-shape gradient, identical on every rank and every process."""
    gen = torch.Generator().manual_seed(
        (seed * 1_000_003 + step * 7_919 + zlib.crc32(name.encode())) % (2**63 - 1)
    )
    return torch.randn(tuple(shape), generator=gen, dtype=torch.float32) * scale


def set_synthetic_grads(model, step: int, device: torch.device, *, seed: int = 0):
    """Assign the synthetic gradient to every trainable param; returns the full grads."""
    full = {}
    for name, p in trainable(model):
        g = synthetic_grad(name, p.shape, step, seed=seed).to(
            device=device, dtype=p.dtype
        )
        full[name] = g
        if isinstance(p, DTensor):
            p.grad = distribute_tensor(g, p.device_mesh, p.placements)
        else:
            p.grad = g.clone()
    return full


def all_ranks_agree(value: bool, device: torch.device) -> bool:
    # On the collective's device: NCCL rejects CPU tensors, gloo accepts both.
    t = torch.tensor([int(value)], dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())
