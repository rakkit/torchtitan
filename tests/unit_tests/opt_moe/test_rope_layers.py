# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-layer attention: sliding window or full (`swa_pattern`) and RoPE, partial
RoPE or NoPE (`rope_pattern`, `rope` / `rope_of_swa` and their `dim`), interleaved."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.opt_moe import model_registry

HEAD = 128  # gauge-d8-w256's head_dim


def _cfg():
    cfg = model_registry("gauge-d8-w256")
    cfg.n_layers, cfg.vocab_size = 4, 512
    cfg.layer.n_dense_layers = 4
    cfg.layer.attention.sliding_window_size = 16
    cfg.swa_pattern = "SFSF"  # sliding, full, sliding, full
    cfg.rope_pattern = "RRNR"  # layer 2 (sliding) is NoPE
    cfg.rope = dataclasses.replace(cfg.rope, dim=HEAD // 2)  # full layers: partial RoPE
    cfg.rope_of_swa = CosSinRoPE.Config(dim=HEAD, max_context_length=4096, theta=5000.0)
    cfg.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=256),
        )
    )
    return cfg


def test_each_layer_gets_its_rope():
    layers = [lc.attention for lc in _cfg().layers]
    assert [a.sliding_window_size > 0 for a in layers] == [True, False, True, False]
    assert layers[0].rope.dim == HEAD and layers[0].rope.theta == 5000.0  # rope_of_swa
    assert layers[1].rope.dim == HEAD // 2 and layers[1].rope.theta == 10000.0  # rope
    assert layers[2].rope is None  # NoPE
    assert layers[3].rope.dim == HEAD // 2
    with torch.device("meta"):
        model = _cfg().build()
    attn = [model.layers[str(i)].attention for i in range(4)]
    assert attn[2].rope is None
    assert [a.partial_rope for a in attn] == [False, True, False, True]
    assert [a.qk_rope_dim for a in attn] == [HEAD, HEAD // 2, HEAD, HEAD // 2]


def test_partial_rope_rotates_the_first_rope_dim_dims_only():
    with torch.device("meta"):
        model = _cfg().build()
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    attn = model.layers["1"].attention
    t, h = 12, attn.n_heads
    q, k = torch.randn(t, h, HEAD), torch.randn(t, h, HEAD)
    positions = torch.arange(t)
    got_q, got_k = attn._apply_rope(q, k, positions)
    rope = CosSinRoPE.Config(dim=HEAD // 2, max_context_length=4096).build()
    rope.init_states(buffer_device=torch.device("cpu"))
    want_q, want_k = rope(q[..., : HEAD // 2], k[..., : HEAD // 2], positions)
    torch.testing.assert_close(got_q[..., : HEAD // 2], want_q)
    torch.testing.assert_close(got_k[..., : HEAD // 2], want_k)
    assert torch.equal(got_q[..., HEAD // 2 :], q[..., HEAD // 2 :])  # passed through
    assert torch.equal(got_k[..., HEAD // 2 :], k[..., HEAD // 2 :])


def test_rope_wider_than_the_head_is_refused():
    cfg = _cfg()
    cfg.layers[1].attention.rope = dataclasses.replace(cfg.rope, dim=2 * HEAD)
    with pytest.raises(ValueError, match="must be <= head_dim"):
        with torch.device("meta"):
            cfg.layers[1].attention.build()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention needs CUDA")
def test_interleaved_model_runs_forward_and_backward():
    cfg = _cfg()
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    positions = torch.cat([torch.arange(100), torch.arange(156)]).cuda()
    tokens = torch.randint(0, cfg.vocab_size, (256,), device="cuda")
    logits = model(
        tokens,
        positions=positions,
        attention_masks=model.get_attention_masks(positions),
    )
    logits.float().square().mean().backward()
    assert torch.isfinite(logits).all()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
