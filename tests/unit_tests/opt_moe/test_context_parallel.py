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
contiguous sharding and head-tail load balancing; the flex (fp32) and varlen
(bf16: FlashAttention has no fp32) backends."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.models.opt_moe import build_model_config

CP = 4
T = 1024
# Documents end inside the ranks' 256-token (contiguous) and 128-token (head-tail)
# chunks; one 1-token document; with max_num_documents above their count, the
# offsets carry padding slots as in training.
DOCS = [300, 1, 450, 273]
MAX_DOCUMENTS = 8
# backend -> (dtype, the largest |g_cp - g| / |g| per parameter and for the loss)
BACKENDS = {"flex": (torch.float32, 1e-5), "varlen": (torch.bfloat16, 2e-2)}


def _model(parallelism: ParallelismConfig, backend: str, dtype: torch.dtype):
    cfg = build_model_config(
        "mis-8b-proxy",
        parallelism=parallelism,
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
        attention_sinks=True,
        sinks_init_fn_type="trunc_normal",
        sinks_init_std=0.5,
        gated_attention_type="element-wise",
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    return model.to(dtype)


def _relative(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


@pytest.mark.skipif(torch.cuda.device_count() < CP, reason=f"needs {CP} GPUs")
class TestContextParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return CP

    @with_comms
    def test_cp_matches_one_rank(self):
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
        for backend, (dtype, tol) in BACKENDS.items():
            torch.manual_seed(0)
            reference = _model(ParallelismConfig(), backend, dtype)
            with torch.no_grad():  # every conv tap off its init value
                for name, p in reference.named_parameters():
                    if "_conv." in name:
                        p.add_(0.3 * torch.randn_like(p))
            logits = reference(
                tokens,
                positions=positions,
                attention_metadata=reference._get_attention_metadata(
                    positions, max_num_documents=MAX_DOCUMENTS
                ),
            )
            want_loss = F.cross_entropy(logits.float(), labels, reduction="sum") / T
            want_loss.backward()
            want_grads = {n: p.grad for n, p in reference.named_parameters()}
            for balancer in (None, HeadTailCPLoadBalancer.Config()):
                with self.subTest(backend=backend, load_balancer=balancer):
                    parallelism = ParallelismConfig(
                        context_parallel_degree=CP,
                        context_parallel_load_balancer=balancer,
                    )
                    model = _model(parallelism, backend, dtype)
                    model.load_state_dict(reference.state_dict())
                    with pc.activate_spmd():
                        inputs, local_labels, model_kwargs, _ = model.preprocess_inputs(
                            {"input": tokens, "labels": labels, "positions": positions},
                            parallelism_context=pc,
                            parallelism=parallelism,
                            max_num_documents=MAX_DOCUMENTS,
                        )
                        local_logits = model(inputs, **model_kwargs)
                        local_loss = (
                            F.cross_entropy(
                                local_logits.float(), local_labels, reduction="sum"
                            )
                            / T
                        )
                        local_loss.backward()
                    loss = local_loss.detach().clone()
                    dist.all_reduce(loss)
                    self.assertLess(_relative(loss, want_loss.detach()), tol)
                    for name, p in model.named_parameters():
                        grad = p.grad.clone()
                        dist.all_reduce(grad)  # each rank holds its tokens' share
                        self.assertLess(
                            _relative(grad, want_grads[name]), tol, msg=name
                        )
