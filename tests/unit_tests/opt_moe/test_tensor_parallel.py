# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe under Tensor Parallel: one packed batch through the same weights at
TP=2 (EP=2: upstream's MoE needs EP >= TP; the MoE sequence-shards its tokens
over TP) and on one rank gives the same loss and gradients, with and without
sequence parallelism. Each attention case (the element-wise gate, sinks,
neither with partial RoPE) with the Canon short conv on k and v, a
sliding-window and a full layer, a dense and an MoE layer (shared expert,
expert bias), and an MTP depth; many short documents (the MTP validity mask
and the conv's document offsets) and trailing padding. Each rank holds whole
KV groups (4 KV heads, 2 per rank); fp32 with flex attention (the routed
experts in bf16). The forward and the loss run under spmd_types type
checking, as the training engine runs them."""

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import TIMEOUT_OVERRIDE
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.opt_moe import build_model_config
from torchtitan.models.opt_moe.mtp import MTPCrossEntropyLoss, roll_mtp_sequence

TP = 2
T = 512
V = 512
# many short documents: the MTP validity mask and the conv offsets carry weight
DOCS = [16] * 22 + [1] * 8 + [120]
PAD = T - sum(DOCS)  # trailing padding: padding_mask True, labels ignored
ATTENTION = {
    "gate": dict(gated_attention_type="element-wise"),
    "sinks": dict(
        gated_attention_type=None,
        attention_sinks=True,
        sinks_init_fn_type="trunc_normal",
        sinks_init_std=0.5,
    ),
    "neither": dict(gated_attention_type=None, rope=dict(dim=64, theta=10000.0)),
}
# The largest |loss_tp - loss| / |loss| (fp32).
TOL = 1e-5
# The largest |g_tp - g| / |g| per parameter: the routed experts' grouped
# GEMMs run in bf16 (GroupedLinear._grouped_mm) and TP x EP groups the tokens
# differently from one rank, so their outputs round differently and every
# gradient upstream of an MoE moves (measured <= 1.4e-3); the sharding bugs
# this test caught moved gradients by 0.45 and 1.0.
GRAD_TOL = 5e-3
# Every rank compiles FlexAttention (forward and backward, full and sliding
# window) for each case: more than the harness's default 300 s per test.
TIMEOUT_OVERRIDE["test_tp_matches_one_rank"] = 1200


def _model(parallelism: ParallelismConfig, attention: dict) -> Module:
    cfg = build_model_config(
        "mis-8b-proxy",
        parallelism=parallelism,
        seq_len=T,
        n_layers=2,
        swa_pattern="SF",
        rope_pattern=None,
        attn_backend="flex",
        vocab_size=V,
        n_heads=8,
        n_kv_heads=4,
        n_dense_layers=1,
        dense_hidden_dim=256,
        num_experts=8,
        top_k=2,
        moe_hidden_dim=64,
        conv_kernel_size=4,
        num_mtp_layers=1,
        # the production recipe's router init: scores far apart, so the TP and
        # the one-rank summation orders cannot flip a token's top-k
        router_init_fn_type="scaled_orthogonal",
        router_init_std=4.0,
        **attention,
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    return model


def _placement(layout: spmd.SpmdType, axis: str):
    """``("shard", dim)``, ``("R", None)`` or ``("I", None)`` of ``layout`` on
    ``axis`` (None: the layout does not name the axis)."""
    if layout.partition_spec is not None:
        for dim, entry in enumerate(layout.partition_spec):
            axes = (
                () if entry is None else entry if isinstance(entry, tuple) else (entry,)
            )
            if axis in axes:
                return "shard", dim
    kind = layout.local_type.get(axis)
    if kind is None:
        return None
    if isinstance(kind, spmd.Shard):
        return "shard", kind.dim
    if kind == spmd.R:
        return "R", None
    if kind == spmd.I:
        return "I", None
    raise ValueError(f"no gradient rule for {kind} on {axis}")


def _relative(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


@pytest.mark.skipif(torch.cuda.device_count() < TP, reason=f"needs {TP} GPUs")
class TestTensorParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return TP

    @with_comms
    def test_tp_matches_one_rank(self):
        # this test's worker processes only: no max-autotune (compile time)
        FlexInnerAttention.inductor_configs["max_autotune"] = False
        positions = torch.cat([torch.arange(n) for n in DOCS + [PAD]]).cuda()
        padding = torch.zeros(T, dtype=torch.bool, device="cuda")
        padding[T - PAD :] = True
        torch.manual_seed(0)
        tokens = torch.randint(0, V, (T,), device="cuda")
        labels = torch.randint(0, V, (T,), device="cuda")
        labels[padding] = IGNORE_INDEX
        rolled, valid = roll_mtp_sequence(
            tokens, shift=1, fill_value=0, positions_T=positions, padding_mask_T=padding
        )
        mtp_labels, _ = roll_mtp_sequence(
            labels,
            shift=1,
            fill_value=IGNORE_INDEX,
            positions_T=positions,
            padding_mask_T=padding,
        )
        counts = torch.stack(
            [(labels != IGNORE_INDEX).sum(), (mtp_labels != IGNORE_INDEX).sum()]
        )
        loss_fn = MTPCrossEntropyLoss(
            MTPCrossEntropyLoss.Config(mtp_scale=0.3, global_vocab_size=V)
        )
        # every one-rank reference first: a TP x EP mesh built for a case leaves
        # process state that a later one-rank model would run under
        references = {}
        for case, attention in ATTENTION.items():
            torch.manual_seed(0)
            reference = _model(ParallelismConfig(), attention)
            with torch.no_grad():  # every conv tap off its init value
                for name, p in reference.named_parameters():
                    if "_conv." in name:
                        p.add_(0.3 * torch.randn_like(p))
            predictions = reference(
                (tokens, rolled),
                positions=positions,
                attention_metadata=reference._get_attention_metadata(
                    positions, padding_mask=padding
                ),
                mtp_input_valid_masks=(valid,),
                padding_mask=padding,
            )
            want_loss, _ = loss_fn(predictions, (labels, mtp_labels), counts)
            want_loss.backward()
            references[case] = (
                reference.state_dict(),
                want_loss.detach(),
                {n: p.grad for n, p in reference.named_parameters()},
            )
        # every case's errors, asserted together at the end (one failure would
        # hide the rest)
        failures = []
        for case, attention in ATTENTION.items():
            state, want_loss, want_grads = references[case]
            for sp in (False, True):
                tag = f"{case} sp={sp}"
                pc = ParallelismContext(
                    dp_replicate=1,
                    dp_shard=1,
                    cp=1,
                    tp=TP,
                    pp=1,
                    ep=TP,
                    world_size=TP,
                    enable_sequence_parallel=sp,
                )
                pc.build_mesh()
                parallelism = ParallelismConfig(
                    tensor_parallel_degree=TP,
                    expert_parallel_degree=TP,
                    enable_sequence_parallel=sp,
                )
                tp_group = pc.get_dense_tp_mesh().get_group()
                ep_group = pc.spmd_sparse_mesh()["ep"].get_group()
                # built and loaded whole, then sharded (an init under the
                # SPMD context would take the full weights for TP shards)
                model = _model(parallelism, attention)
                model.load_state_dict(state)
                annotate_replicated_parameters(model, pc)  # as the trainer
                with pc.activate_spmd():
                    # the sharding alone (OPTMoEModel._parallelize also lists
                    # the MTP blocks in `layers` for parallelize()'s remat /
                    # FSDP steps, which remove them again)
                    Module._parallelize(model, pc)
                    inputs, local_labels, model_kwargs, _ = model.preprocess_inputs(
                        {
                            "input": tokens,
                            "labels": labels,
                            "positions": positions,
                            "padding_mask": padding,
                        },
                        parallelism_context=pc,
                        parallelism=parallelism,
                    )
                # the forward and the loss type-checked, as the training engine
                with pc.activate_spmd(typechecking=True):
                    local_predictions = model(inputs, **model_kwargs)
                    loss, _ = loss_fn(local_predictions, local_labels, counts)
                    with spmd.no_typecheck():
                        loss.backward()
                rel = _relative(loss.detach(), want_loss)
                if not rel < TOL:
                    failures.append(f"{tag} loss {rel:.3g}")
                modules = dict(model.named_modules())
                for name, p in model.named_parameters():
                    owner, _, attr = name.rpartition(".")
                    layout = modules[owner]._sharding_config.state_shardings[attr]
                    want = want_grads[name]
                    grad = p.grad.clone()
                    for axis, group in (("tp", tp_group), ("ep", ep_group)):
                        placement = _placement(layout, axis)
                        if placement is None or placement[0] == "I":
                            continue
                        kind, dim = placement
                        if kind == "R":  # each rank's partial sum
                            dist.all_reduce(grad, group=group)
                        else:
                            want = want.chunk(dist.get_world_size(group), dim)[
                                dist.get_rank(group)
                            ]
                    rel = _relative(grad, want)
                    if not rel < GRAD_TOL:
                        failures.append(f"{tag} {name} {rel:.3g}")
        if failures and dist.get_rank() == 0:
            print("\n".join(["TP mismatches:", *failures]), flush=True)
        self.assertEqual(failures, [])
