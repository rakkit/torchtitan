# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`fuse_qkv` and `fuse_w13` change layout only, never arithmetic."""

from __future__ import annotations

import pytest
import torch

from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.opt_moe.gated_norm_swattention import GatedNormSWAttention
from torchtitan.models.opt_moe.norm_ffn import FeedForward
from torchtitan.models.opt_moe.split_qkv import SplitQKVLinear


@pytest.mark.parametrize("norm_everywhere", [False, True])
def test_fused_w13_matches_the_split_pair(norm_everywhere: bool):
    def build(fuse: bool) -> FeedForward:
        module = FeedForward(
            FeedForward.Config(
                dim=8,
                hidden_dim=16,
                fuse_w13=fuse,
                norm_everywhere=norm_everywhere,
            )
        )
        module.to_empty(device="cpu")
        return module

    split, fused = build(False), build(True)
    torch.manual_seed(0)
    w1, w3, w2 = torch.randn(16, 8), torch.randn(16, 8), torch.randn(8, 16)
    split.w1.weight.data.copy_(w1)
    split.w3.weight.data.copy_(w3)
    split.w2.weight.data.copy_(w2)
    fused.w13.weight.data[0].copy_(w1)
    fused.w13.weight.data[1].copy_(w3)
    fused.w2.weight.data.copy_(w2)

    x = torch.randn(5, 8)
    assert torch.equal(split(x), fused(x))
    assert fused.w13.weight.shape == (2, 16, 8)


def test_the_flags_are_off_by_default():
    assert FeedForward.Config(dim=8, hidden_dim=16).fuse_w13 is False
    assert (
        GatedNormSWAttention.Config(
            dim=8, n_heads=4, n_kv_heads=2, head_dim=4, attn_backend="flex"
        ).fuse_qkv
        is False
    )


@pytest.mark.parametrize("fuse, expected", [(False, SplitQKVLinear), (True, QKVLinear)])
def test_fuse_qkv_selects_the_projection(fuse: bool, expected: type):
    module = GatedNormSWAttention.Config(
        dim=8, n_heads=4, n_kv_heads=2, head_dim=4, attn_backend="flex", fuse_qkv=fuse
    ).build()
    assert isinstance(module.qkv_linear, expected)


def test_fused_qkv_init_matches_the_split_projections():
    """Each projection must be drawn at its own split shape: `scaled_orthogonal`
    scales by the shape it is given.
    """
    from torchtitan.models.opt_moe.gated_norm_swattention import _fused_qkv_init
    from torchtitan.models.opt_moe.utils.inits import make_param_init

    n_heads, n_kv, head_dim, dim = 8, 2, 4, 16
    heads_per_kv = n_heads // n_kv

    def gram_diag(mat: torch.Tensor) -> float:
        return torch.diagonal(mat @ mat.T).mean().item()

    make = lambda: make_param_init("scaled_orthogonal", 1.0)  # noqa: E731

    split_q = torch.empty(n_heads * head_dim, dim)
    make()(split_q)
    split_k = torch.empty(n_kv * head_dim, dim)
    make()(split_k)

    fused = torch.empty((n_heads + 2 * n_kv) * head_dim, dim)
    _fused_qkv_init(
        make(), make(), make(), n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim
    )(fused)
    view = fused.view(n_kv, heads_per_kv + 2, head_dim, dim)

    # fan_out/fan_in per projection: q = 32/16 = 2, k = 8/16 = 0.5
    assert gram_diag(view[:, :heads_per_kv].reshape(-1, dim)) == pytest.approx(
        gram_diag(split_q), rel=1e-6
    )
    assert gram_diag(view[:, heads_per_kv].reshape(-1, dim)) == pytest.approx(
        gram_diag(split_k), rel=1e-6
    )

    # A single init over the whole matrix gets the scale wrong.
    naive = torch.empty((n_heads + 2 * n_kv) * head_dim, dim)
    make()(naive)
    naive_q = naive.view(n_kv, heads_per_kv + 2, head_dim, dim)[:, :heads_per_kv]
    assert gram_diag(naive_q.reshape(-1, dim)) != pytest.approx(
        gram_diag(split_q), rel=1e-3
    )


def test_fuse_qkv_allows_different_initializers_per_projection():
    module = GatedNormSWAttention.Config(
        dim=8,
        n_heads=4,
        n_kv_heads=2,
        head_dim=4,
        attn_backend="flex",
        fuse_qkv=True,
        wq_init_fn_type="scaled_orthogonal",
        wk_init_fn_type="scion_normal_input",
    ).build()
    assert isinstance(module.qkv_linear, QKVLinear)


def test_no_flavor_enables_either_flag():
    import torchtitan.models.opt_moe as opt_moe

    for name, config in opt_moe.moe_opt_moe_configs.items():
        attention = getattr(config.layer, "attention", None)
        feed_forward = getattr(config.layer, "feed_forward", None)
        assert not getattr(attention, "fuse_qkv", False), name
        assert not getattr(feed_forward, "fuse_w13", False), name
