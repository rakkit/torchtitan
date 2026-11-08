# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe under Context Parallel: one packed batch through the same weights at
CP=4 and on one rank gives the same loss and gradients. The attention has
everything the final model uses: the Canon short conv (each rank's fragments
start from the tokens before them, exchanged across the ranks), attention sinks
(all-gather K/V: every local query sees every key it attends to, so the
log-sum-exp the sinks rescale is complete), the element-wise gate, a
sliding-window and a full layer. Documents cross the ranks' token ranges;
contiguous sharding and head-tail load balancing, and Ulysses (heads sharded
over CP; the sinks rescale on each rank's heads); the flex (fp32),
varlen and flex_flash (FA4 through flex; bf16: FlashAttention has no fp32)
backends."""

import pytest
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
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.models.opt_moe import build_model_config
from torchtitan.models.opt_moe.mtp import MTPCrossEntropyLoss, roll_mtp_sequence

CP = 4
T = 1024
# Documents end inside the ranks' 256-token (contiguous) and 128-token (head-tail)
# chunks; one 1-token document; with max_num_documents above their count, the
# offsets carry padding slots as in training.
DOCS = [300, 1, 450, 273]
MAX_DOCUMENTS = 8
# backend -> (dtype, the largest |g_cp - g| / |g| per parameter and for the loss)
BACKENDS = {
    "flex": (torch.float32, 1e-5),
    "varlen": (torch.bfloat16, 2e-2),
    "flex_flash": (torch.bfloat16, 2e-2),
}
# The largest per-token cross-entropy difference CP vs one rank (nats): a bug at
# a shard or document edge moves a few tokens by O(0.1 - 1), which a
# whole-tensor norm above can dilute. Measured (4 GPUs, 11-06): flex fp32
# 4.8e-7 / 9.5e-7, varlen bf16 5.0e-4 (without / with the head-tail balancer);
# the bounds are 10x those.
TOKEN_TOL = {"flex": 1e-5, "varlen": 5e-3, "flex_flash": 5e-3}


# attention -> its CP cases (cp_attention, load balancer): the all-gather K/V
# (contiguous and head-tail) and Ulysses (heads sharded over CP: contiguous
# shards only; the sinks rescale on this rank's heads). 4 KV heads: Ulysses
# needs them divisible by CP.
CASES = (
    ("gate", (("ulysses", None),)),
    (
        "sinks",
        (
            ("ulysses", None),
            ("all_gather_kv", None),
            ("all_gather_kv", HeadTailCPLoadBalancer.Config()),
        ),
    ),
)


def _model(
    parallelism: ParallelismConfig,
    backend: str,
    dtype: torch.dtype,
    *,
    attention: str,
    cp_attention: str = "all_gather_kv",
):
    attention_kwargs = (
        dict(
            attention_sinks=True,
            sinks_init_fn_type="trunc_normal",
            sinks_init_std=0.5,
        )
        if attention == "sinks"
        else dict(gated_attention_type="element-wise")
    )
    cfg = build_model_config(
        "mis-8b-proxy",
        parallelism=parallelism,
        cp_attention=cp_attention,
        seq_len=T,
        n_layers=2,
        swa_pattern="SF",
        rope_pattern=None,
        attn_backend=backend,
        vocab_size=512,
        # dense layers: a near-tie in MoE routing must not decide the comparison
        n_dense_layers=2,
        dense_hidden_dim=512,
        conv_kernel_size=4,
        num_mtp_layers=1,
        n_kv_heads=4,
        **attention_kwargs,
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    return model.to(dtype)


def _relative(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


# One test per backend: every rank compiles FlexAttention (max_autotune: forward
# and backward, full and sliding-window) for each attention case and sharding,
# more than the harness's default 300 s per test.
for _backend in BACKENDS:
    TIMEOUT_OVERRIDE[f"test_cp_matches_one_rank_{_backend}"] = 1500


@pytest.mark.skipif(torch.cuda.device_count() < CP, reason=f"needs {CP} GPUs")
class TestContextParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return CP

    @with_comms
    def test_cp_matches_one_rank_flex(self):
        self._cp_matches_one_rank("flex")

    @with_comms
    def test_cp_matches_one_rank_varlen(self):
        self._cp_matches_one_rank("varlen")

    @with_comms
    def test_cp_matches_one_rank_flex_flash(self):
        self._cp_matches_one_rank("flex_flash")

    def _cp_matches_one_rank(self, backend: str):
        dtype, tol = BACKENDS[backend]
        pc = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=CP,
            tp=1,
            pp=1,
            ep=1,
            world_size=CP,
            enable_sequence_parallel=False,
        )
        pc.build_mesh()
        positions = torch.cat([torch.arange(n) for n in DOCS]).cuda()
        torch.manual_seed(0)
        tokens = torch.randint(0, 512, (T,), device="cuda")
        labels = torch.randint(0, 512, (T,), device="cuda")
        for attention, cases in CASES:
            torch.manual_seed(0)
            reference = _model(ParallelismConfig(), backend, dtype, attention=attention)
            with torch.no_grad():  # every conv tap off its init value
                for name, p in reference.named_parameters():
                    if "_conv." in name:
                        p.add_(0.3 * torch.randn_like(p))
            rolled, valid = roll_mtp_sequence(
                tokens, shift=1, fill_value=0, positions_T=positions
            )
            mtp_labels, _ = roll_mtp_sequence(
                labels, shift=1, fill_value=IGNORE_INDEX, positions_T=positions
            )
            counts = torch.stack(
                [(labels != IGNORE_INDEX).sum(), (mtp_labels != IGNORE_INDEX).sum()]
            )
            loss_fn = MTPCrossEntropyLoss(MTPCrossEntropyLoss.Config(mtp_scale=0.3))
            predictions = reference(
                (tokens, rolled),
                positions=positions,
                attention_metadata=reference._get_attention_metadata(
                    positions, max_num_documents=MAX_DOCUMENTS
                ),
                mtp_input_valid_masks=(valid,),
            )
            want_loss, _ = loss_fn(
                tuple(p.float() for p in predictions), (labels, mtp_labels), counts
            )
            want_loss.backward()
            want_grads = {n: p.grad for n, p in reference.named_parameters()}
            want_token_loss = torch.nn.functional.cross_entropy(
                predictions[0].detach().float(),
                labels,
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )
            for cp_attention, balancer in cases:
                with self.subTest(
                    backend=backend,
                    attention=attention,
                    cp_attention=cp_attention,
                    load_balancer=balancer,
                ):
                    parallelism = ParallelismConfig(
                        context_parallel_degree=CP,
                        context_parallel_load_balancer=balancer,
                    )
                    model = _model(
                        parallelism,
                        backend,
                        dtype,
                        attention=attention,
                        cp_attention=cp_attention,
                    )
                    model.load_state_dict(reference.state_dict())
                    with pc.activate_spmd():
                        # the token index rides along as a loss kwarg: sharded
                        # as the labels, so each rank knows its tokens
                        (
                            inputs,
                            local_labels,
                            model_kwargs,
                            extra,
                        ) = model.preprocess_inputs(
                            {
                                "input": tokens,
                                "labels": labels,
                                "positions": positions,
                                "loss_kwargs": {
                                    "token_index": torch.arange(T, device="cuda")
                                },
                            },
                            parallelism_context=pc,
                            parallelism=parallelism,
                            max_num_documents=MAX_DOCUMENTS,
                        )
                        local_predictions = model(inputs, **model_kwargs)
                        local_loss, _ = loss_fn(
                            tuple(p.float() for p in local_predictions),
                            local_labels,
                            counts,
                        )
                        local_loss.backward()
                        local_token_loss = torch.nn.functional.cross_entropy(
                            local_predictions[0].detach().float(),
                            local_labels[0],
                            reduction="none",
                            ignore_index=IGNORE_INDEX,
                        )
                    loss = local_loss.detach().clone()
                    dist.all_reduce(loss)
                    self.assertLess(_relative(loss, want_loss.detach()), tol)
                    index = torch.empty(T, dtype=torch.long, device="cuda")
                    token_loss = torch.empty(T, device="cuda")
                    dist.all_gather_into_tensor(
                        index, extra["token_index"].contiguous()
                    )
                    dist.all_gather_into_tensor(
                        token_loss, local_token_loss.contiguous()
                    )
                    got_token_loss = torch.full_like(want_token_loss, float("nan"))
                    got_token_loss[index] = token_loss
                    worst = (got_token_loss - want_token_loss).abs().max().item()
                    self.assertLess(worst, TOKEN_TOL[backend])
                    for name, p in model.named_parameters():
                        grad = p.grad.clone()
                        dist.all_reduce(grad)  # each rank holds its tokens' share
                        self.assertLess(
                            _relative(grad, want_grads[name]), tol, msg=name
                        )
