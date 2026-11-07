# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe's MoE paths (flavor arguments token_dispatch, expert_gemm,
router_topk) train the same model: one step of a two-layer model, same init and
tokens, loss and every gradient close to upstream's paths (scatter-add dispatch,
torch._grouped_mm, torch.topk)."""

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config


def _step(**paths):
    cfg = build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=512,
        n_layers=2,
        swa_pattern="SF",
        rope_pattern=None,
        moe_hidden_dim=128,  # DeepGEMM: multiples of 64
        **paths,
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    positions = torch.cat([torch.arange(200), torch.arange(312)]).cuda()
    tokens = torch.randint(0, cfg.vocab_size, (512,), device="cuda")
    logits = model(
        tokens,
        positions=positions,
        attention_metadata=model._get_attention_metadata(positions),
    )
    loss = F.cross_entropy(logits.float(), tokens.roll(-1))
    loss.backward()
    return loss.item(), {n: p.grad.float() for n, p in model.named_parameters()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "paths",
    [
        dict(token_dispatch="gather"),
        dict(router_topk="triton"),
        dict(token_dispatch="gather", expert_gemm="deepgemm", router_topk="triton"),
    ],
)
def test_alternative_moe_paths_train_the_same_model(paths):
    """Each combination vs the upstream paths: loss and every gradient."""
    pytest.importorskip("deep_gemm") if paths.get("expert_gemm") else None
    upstream = dict(
        token_dispatch="scatter_add", expert_gemm="torch", router_topk="torch"
    )
    loss, grads = _step(**upstream)
    alt_loss, alt_grads = _step(**{**upstream, **paths})
    assert abs(alt_loss - loss) < 1e-3 * abs(loss)
    for name, grad in grads.items():
        rel = (alt_grads[name] - grad).norm() / grad.norm().clamp_min(1e-12)
        assert rel < 2e-2, (name, rel.item())
