# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Residual and gain statistics under data parallelism (4 GPUs, torchrun):

    torchrun --standalone --nproc_per_node 4 tests/unit_tests/gpu/residual_stats_dist_check.py

Three copies of one tiny model with the same weights: plain parameters, the
production DDP path (dp_replicate 4 x dp_shard 1) and FSDP (dp_shard 4), the
latter two built through the model's own ``parallelize`` (the trainer's path)
and fed through ``preprocess_inputs`` inside the SPMD context, as the engine
does. Each rank feeds its own batch. Pass: every copy reports the same residual
RMS, equal to a recomputation from all ranks' all-gathered captures, and the
same gain statistics after the same update, equal to formulas on the full
gains. Exits non-zero on any failure.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.models.opt_moe import build_model_config
from torchtitan.optimizers import model_stats

VOCAB, DOCS = 1024, (640, 384)


def config():
    return build_model_config(
        "gauge-d16-w512",
        parallelism=ParallelismConfig(),
        n_layers=2,
        vocab_size=VOCAB,
        norm_type="rmsnorm",
    )


def plain_model():
    with torch.device("meta"):
        model = config().build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    return model


def parallel_model(pd: ParallelismContext, rep: int, shard: int, plain):
    with torch.device("meta"):
        model = config().build()
    model.parallelize(
        parallelism_context=pd,
        training=TrainingConfig(
            mixed_precision_param="float32", mixed_precision_reduce="float32"
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=rep, data_parallel_shard_degree=shard
        ),
        compile_config=None,
        ac_config=None,
        dump_folder=tempfile.mkdtemp(),
    )
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
        full = dict(plain.named_parameters())
        for name, p in model.named_parameters():
            src = full[name].detach()
            if isinstance(p, DTensor):
                p.to_local().copy_(
                    distribute_tensor(src, p.device_mesh, p.placements).to_local()
                )
            else:
                p.copy_(src)
    return model


def batch(rank: int):
    g = torch.Generator().manual_seed(100 + rank)
    tokens = torch.randint(0, VOCAB, (sum(DOCS),), generator=g)
    positions = torch.cat([torch.arange(n) for n in DOCS])
    return {
        "input": tokens,
        "labels": torch.roll(tokens, -1),
        "positions": positions,
        "padding_mask": torch.zeros_like(tokens, dtype=torch.bool),
    }


def forward(model, pd, rep, shard, data):
    model_stats.arm_residual_capture([model])
    with torch.no_grad():
        if pd is None:
            pos = data["positions"].cuda()
            model(data["input"].cuda(), pos, model._get_attention_metadata(pos))
            return
        cuda = {k: v.cuda() for k, v in data.items()}
        parallelism = ParallelismConfig(
            data_parallel_replicate_degree=rep, data_parallel_shard_degree=shard
        )
        inputs, _, extra, _ = model.preprocess_inputs(
            cuda,
            parallelism_context=pd,
            parallelism=parallelism,
            max_context_length=4096,
        )
        with pd.activate_spmd():
            model(inputs, **extra)


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert world == 4, world
    dims = dict(cp=1, tp=1, pp=1, ep=1, world_size=4, enable_sequence_parallel=False)
    pd_ddp = ParallelismContext(dp_replicate=4, dp_shard=1, **dims)
    pd_fsdp = ParallelismContext(dp_replicate=1, dp_shard=4, **dims)
    pd_ddp.build_mesh()
    pd_fsdp.build_mesh()
    plain = plain_model()
    copies = {
        "plain": (plain, None, 1, 1),
        "ddp": (parallel_model(pd_ddp, 4, 1, plain), pd_ddp, 4, 1),
        "fsdp": (parallel_model(pd_fsdp, 1, 4, plain), pd_fsdp, 1, 4),
    }
    data = batch(rank)
    fails = []

    # Residual: each rank its own batch; the loss mesh reduces over all 4 ranks.
    residual = {}
    for name, (model, pd, rep, shard) in copies.items():
        forward(model, pd, rep, shard, data)
        local = torch.stack(model.residual_capture).sum(0)
        count = torch.tensor([float(len(model.residual_capture))], device=local.device)
        packed = torch.cat([local, count])
        gathered = [torch.empty_like(packed) for _ in range(world)]
        dist.all_gather(gathered, packed)
        total = torch.stack(gathered).sum(0)
        want = (total[:-1] / total[-1]).sqrt().tolist()
        got = model_stats.residual_metrics([model], pd_ddp if pd is None else pd)
        residual[name] = got
        keys = [
            k
            for i in range(2)
            for k in (f"residual_rms/in/L-{i}", f"residual_rms/write/L-{i}")
        ]
        keys.append("residual_rms/final_norm_in")
        for k, w in zip(keys, want, strict=True):
            if not math.isclose(got[k], w, rel_tol=1e-6):
                fails.append(f"{name} {k}: {got[k]} vs all-gathered {w}")
    for name in ("ddp", "fsdp"):
        for k, v in residual["plain"].items():
            if not math.isclose(residual[name][k], v, rel_tol=1e-5):
                fails.append(f"{name} vs plain {k}: {residual[name][k]} vs {v}")

    # Gains: the same deterministic update on every copy, applied shard by shard.
    gains_out = {}
    for name, (model, pd, _, _) in copies.items():
        before = model_stats.snapshot_gains([model])
        g = torch.Generator().manual_seed(5)
        full = {}
        with torch.no_grad():
            for mod_name, mod in model.named_modules():
                w = getattr(mod, "weight", None)
                if not isinstance(mod, torch.nn.RMSNorm) or w is None:
                    continue
                delta = 0.01 * torch.randn(w.shape, generator=g).cuda()
                if isinstance(w, DTensor):
                    w.to_local().add_(
                        distribute_tensor(delta, w.device_mesh, w.placements).to_local()
                    )
                    full[mod_name] = w.full_tensor().double()
                else:
                    w.add_(delta)
                    full[mod_name] = w.detach().double()
        got = model_stats.gain_metrics([model], before, pd)
        gains_out[name] = got
        for (
            mod_name,
            a,
        ) in full.items():  # mean / RMS vs the full gain; rel_update vs plain
            for key, want in (
                ("mean", float(a.mean())),
                ("rms", float(a.square().mean().sqrt())),
            ):
                if not math.isclose(got[f"gain_{key}/{mod_name}"], want, rel_tol=1e-6):
                    fails.append(
                        f"{name} gain_{key}/{mod_name}: {got[f'gain_{key}/{mod_name}']} vs {want}"
                    )
    for name in ("ddp", "fsdp"):
        for k, v in gains_out["plain"].items():
            if not math.isclose(gains_out[name][k], v, rel_tol=1e-5):
                fails.append(f"{name} vs plain {k}: {gains_out[name][k]} vs {v}")

    placements = {
        name: sorted(
            {
                str(p.placements)
                for _, p in model.named_parameters()
                if isinstance(p, DTensor) and p.ndim == 1
            }
        )
        for name, (model, _, _, _) in copies.items()
    }
    if rank == 0:
        print(f"1-D param placements: {placements}")
        print(
            f"residual keys {len(residual['plain'])}, gain keys {len(gains_out['plain'])}; e.g. "
            f"{next(iter(residual['plain'].items()))}, {next(iter(gains_out['plain'].items()))}"
        )
    ok = torch.tensor([0 if fails else 1], device="cuda")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    for f in fails[:10]:
        print(f"[rank {rank}] FAIL {f}")
    if rank == 0:
        print("TEST 9", "PASS" if ok.item() else "FAIL")
    dist.destroy_process_group()
    return 0 if ok.item() else 1


if __name__ == "__main__":
    sys.exit(main())
