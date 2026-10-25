# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A shrunken gauge model run through real forward / backward passes under FSDP2,
with the gnoise hooks driven the way the trainer drives them."""

import copy
import dataclasses
import os
import sys

import spmd_types as spmd

import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disco")
)
import harness  # noqa: E402

from torchtitan.models.opt_moe import model_registry  # noqa: E402
from torchtitan.models.opt_moe.norm_moe import MoE  # noqa: E402

VOCAB, SEQ = 128, 32


def patch_dense_attention() -> None:
    """CPU only: flex attention has no CPU backward, so evaluate the same BlockMask
    `mask_mod` densely. The GPU tests run the real flex kernel."""
    from torchtitan.models.common.attention import FlexInnerAttention

    def forward(
        self,
        q_THK,
        k_THK,
        v_THV,
        *,
        attention_masks,
        scale=None,
        enable_gqa=False,
        out_transform=None,
        **kwargs
    ):
        assert out_transform is None, "the dense test attention has no sinks"
        T, H, Kd = q_THK.shape
        qi = torch.arange(T)[:, None].expand(T, T)
        ki = torch.arange(T)[None, :].expand(T, T)
        z = torch.zeros((), dtype=torch.long)
        mask = attention_masks.mask_mod(z, z, qi, ki)
        k, v = k_THK, v_THV
        if k.shape[1] != H:
            k = k.repeat_interleave(H // k.shape[1], dim=1)
            v = v.repeat_interleave(H // v.shape[1], dim=1)
        sc = scale if scale is not None else Kd**-0.5
        att = torch.einsum("thk,shk->hts", q_THK.float(), k.float()) * sc
        att = att.masked_fill(~mask[None], float("-inf")).softmax(-1)
        return torch.einsum("hts,shv->thv", att, v.float()).to(q_THK.dtype)

    FlexInnerAttention.forward = forward


def gauge_config(
    *,
    moe: bool = False,
    dim: int = 32,
    n_layers: int = 2,
    num_experts: int = 4,
    gains: bool = True
):
    """gauge-d16-w512 shrunk: QK-norm, MHA, norm eps 1e-30; learnable gains on every
    RMSNorm as the study's GAINS on (cell.py)."""
    cfg = copy.deepcopy(
        model_registry("gauge-d16-w512")
    )  # the registry's object is shared
    if gains:
        cfg.layer.norm_type = cfg.layer.attention.norm_type = cfg.norm_type = "rmsnorm"
    cfg.dim, cfg.n_layers, cfg.vocab_size = dim, n_layers, VOCAB
    cfg.max_context_length = SEQ
    cfg.rope = dataclasses.replace(cfg.rope, dim=16, max_context_length=SEQ)
    layer = cfg.layer
    layer.n_dense_layers = n_layers
    layer.feed_forward.hidden_dim = 2 * dim
    attn = layer.attention
    attn.n_heads, attn.n_kv_heads, attn.head_dim = 2, 2, 16
    if moe:
        layer.moe = MoE.Config(
            hidden_dim=16,
            num_experts=num_experts,
            num_shared_experts=1,
            top_k=2,
            scaling_factor=1.0,
            score_func="sigmoid",
            load_balance_coeff=1e-3,
            aux_loss=None,
            bias_update_norm_factor="sign",
            track_router_metrics=False,
            quantile_balance_bins=None,
            moe_comm_backend="standard",
            norm_everywhere=False,
            norm_type="np_rmsnorm",
        )
        layer.n_dense_layers = 0
    return cfg


def build(
    topo: dict,
    device,
    tmp,
    *,
    moe=False,
    gnoise=None,
    model_cfg=None,
    world=4,
    param_dtype="bfloat16",
    chunks=0,
    force_balance=False
):
    """The model and container as the training engine builds them; returns (pd, model,
    container, cfg). chunks > 0: the engine's chunked loss over the grouped [norm, lm_head].
    force_balance: round-robin routing (debug.moe_force_load_balance), equal EP splits."""
    pd = harness.parallel_dims(world_size=world, device_type=device.type, **topo)
    cfg = harness.trainer_config(
        model_cfg or gauge_config(moe=moe), log_norm_freq=1, **topo
    )
    if force_balance:
        cfg.debug.moe_force_load_balance = True
        for lc in getattr(cfg.model, "layers", None) or []:
            if getattr(lc, "moe", None) is not None:
                lc.moe.router._debug_force_load_balance = True
    cfg.training.max_context_length = SEQ
    cfg.training.mixed_precision_param = param_dtype
    if gnoise is not None:
        for k, v in gnoise.items():
            setattr(cfg.optimizer.gnoise, k, v)
    torch.manual_seed(1234)  # the trainer seeds every rank alike (set_determinism)
    model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
    if (
        device.type == "cpu"
    ):  # gloo has no PREMUL_SUM; the divide factor is 1, so SUM is equal
        from torch.distributed.fsdp import FSDPModule

        for m in model.modules():
            if isinstance(m, FSDPModule):
                m.set_force_sum_reduction_for_comms(True)
    from torch.distributed.tensor import distribute_tensor, DTensor

    gen = torch.Generator().manual_seed(77)
    for n, p in model.named_parameters():
        if (
            ".router.gate." in n
        ):  # the default gate starts with equal scores: route by a random one
            full = torch.randn(tuple(p.shape), generator=gen) * 0.5
            with torch.no_grad():
                if isinstance(p, DTensor):
                    p.copy_(
                        distribute_tensor(full.to(device), p.device_mesh, p.placements)
                    )
                else:
                    p.copy_(full.to(device))
    model._gn_chunked = None
    if chunks:
        from torchtitan.components.loss import ChunkedLossWrapper

        loss = ChunkedLossWrapper.Config(num_chunks=chunks).build(compile_config=None)
        loss.set_lm_head(model.lm_head)
        model._skip_lm_head = True
        model._gn_chunked = loss
    return pd, model, container, cfg


def batch(seed: int, n_tokens: int, device, *, mask_frac: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randint(0, VOCAB, (n_tokens + 1,), generator=g)
    inp, lab = tok[:-1].clone(), tok[1:].clone()
    if mask_frac:
        lab[: int(mask_frac * n_tokens)] = -100
    return inp.to(device), lab.to(device)


def positions(n: int, device, doc_len: int = SEQ):
    return (torch.arange(n) % doc_len).to(device)


def forward_loss(model, pd, cfg, inp, lab, n_den):
    """The engine's path: preprocess_inputs (masks, SPMD types), forward, sum CE / N."""
    pos = positions(inp.numel(), inp.device)
    x, y, extra = model.preprocess_inputs(
        {"input": inp, "labels": lab, "positions": pos},
        parallel_dims=pd,
        parallelism=cfg.parallelism,
        max_context_length=SEQ,
    )
    from torchtitan.distributed.utils import get_spmd_context

    with get_spmd_context(parallel_dims=pd):
        out = model(x, **extra)
        if getattr(model, "_gn_chunked", None) is not None:
            return model._gn_chunked(out, y, global_valid_tokens=n_den)[0]
    logits = out.to_local() if hasattr(out, "to_local") else out
    y = y.to_local() if hasattr(y, "to_local") else y
    return (
        F.cross_entropy(logits.float(), y, reduction="sum", ignore_index=-100) / n_den
    )


def run_step(model, pd, cfg, container, step, microbatches, n_den, *, gn=None):
    """microbatches: this rank's [(inputs, labels)]; n_den the global valid-token count.
    Drives the trainer's calls; returns the gnoise metrics (tensors) of the step."""
    container.begin_step(step, should_log=True, is_final_step=False)
    gn = container.gnoise if gn is None else gn
    nd = torch.tensor(n_den, dtype=torch.int64, device=microbatches[0][0].device)
    from torchtitan.models.common.aux_loss import AuxLoss

    AuxLoss.set_step_denominator(nd)  # as the engine's prepare_step
    if gn is not None:
        gn.set_tokens(nd)
    for inp, lab in microbatches:
        if gn is not None:
            gn.begin_microbatch(int((lab != -100).sum()))
        loss = forward_loss(model, pd, cfg, inp, lab, nd)
        with spmd.no_typecheck():
            loss.backward()
    if gn is not None:
        gn.before_clip()
        return dict(gn.metrics)
    return {}


def solo_grads(model, pd, cfg, microbatches, n_den, world, rank):
    """Every small batch's gradient, through FSDP's own reduce and without gnoise: for
    each (rank r, microbatch k) only r's labels count (the others mask all theirs, so
    their gradient is exactly 0; the chunked loss runs its chunk backward inside the
    call, so scaling the returned loss would not do), and the reduced gradient is
    g_(r, k) bitwise. Returns {(r, k): {name: full fp64 tensor}} and the routed-count
    deltas {(r, k): [per rank: [sum E]]}."""
    from torchtitan.models.opt_moe.norm_moe import NormMoE

    moes = {n: m for n, m in model.named_modules() if isinstance(m, NormMoE)}
    nd = torch.tensor(n_den, dtype=torch.int64, device=microbatches[0][0].device)
    from torchtitan.models.common.aux_loss import AuxLoss

    AuxLoss.set_step_denominator(nd)
    out, counts = {}, {}
    for r in range(world):
        for k, (inp, lab) in enumerate(microbatches):
            model.zero_grad(set_to_none=True)
            snap = {n: m.tokens_per_expert_E.clone() for n, m in moes.items()}
            lab_r = lab if rank == r else torch.full_like(lab, -100)
            loss = forward_loss(model, pd, cfg, inp, lab_r, nd)
            with spmd.no_typecheck():
                loss.backward()
            delta = (
                torch.cat([moes[n].tokens_per_expert_E - snap[n] for n in sorted(moes)])
                if moes
                else torch.zeros(0)
            )
            allc = [torch.zeros_like(delta) for _ in range(world)]
            torch.distributed.all_gather(allc, delta)
            counts[(r, k)] = allc
            out[(r, k)] = {
                n: (
                    p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                ).double()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
    model.zero_grad(set_to_none=True)
    return out, counts


def group_grads(model, pd, cfg, microbatches, n_den, groups, rank, names):
    """For EP > 1: each (EP group G, microbatch k)'s gradient, the labels of every rank
    outside G masked. A rank's tokens only reach its own loss, so G's expert copies
    compute exactly what they compute in the full step. Returns {(G, k): {name: fp64}}."""
    nd = torch.tensor(n_den, dtype=torch.int64, device=microbatches[0][0].device)
    from torchtitan.models.common.aux_loss import AuxLoss

    AuxLoss.set_step_denominator(nd)
    out = {}
    for G in groups:
        for k, (inp, lab) in enumerate(microbatches):
            model.zero_grad(set_to_none=True)
            lab_g = lab if rank in G else torch.full_like(lab, -100)
            loss = forward_loss(model, pd, cfg, inp, lab_g, nd)
            with spmd.no_typecheck():
                loss.backward()
            out[(G, k)] = {
                n: (
                    p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                ).double()
                for n, p in model.named_parameters()
                if n in names and p.grad is not None
            }
    model.zero_grad(set_to_none=True)
    return out


def full_weights(model, param_dtype=torch.bfloat16):
    """The compute copy's values (param_dtype of the fp32 master), in fp64."""
    return {
        n: (p.full_tensor() if hasattr(p, "full_tensor") else p)
        .detach()
        .to(param_dtype)
        .double()
        for n, p in model.named_parameters()
    }
