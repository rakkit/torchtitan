# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe attention's optional short conv (KDA / Gated DeltaNet) and sinks
(gpt-oss), against references written from their equations."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.opt_moe import model_registry
from torchtitan.models.opt_moe.utils.short_conv import causal_short_conv


def _reference_conv(x, w, positions):
    """Per document: `silu(conv1d(x, w, padding=K-1))[:L]`, in fp64."""
    k, out = w.shape[-1], torch.empty_like(x, dtype=torch.float64)
    starts = (positions == 0).nonzero().flatten().tolist() + [x.shape[0]]
    for a, b in zip(starts[:-1], starts[1:]):
        doc = x[a:b].double().T[None]  # [1, C, L]
        y = F.conv1d(doc, w.double(), padding=k - 1, groups=w.shape[0])[0, :, : b - a]
        out[a:b] = F.silu(y).T
    return out


def _docs(lengths, c=6, seed=0):
    torch.manual_seed(seed)
    positions = torch.cat([torch.arange(n) for n in lengths])
    return torch.randn(len(positions), c), torch.randn(c, 1, 4), positions


def test_short_conv_is_a_causal_depthwise_conv_per_document():
    x, w, positions = _docs([1, 5, 2, 9])
    torch.testing.assert_close(
        causal_short_conv(x, w, positions, "kda").double(),
        _reference_conv(x, w, positions),
        rtol=1e-6,
        atol=1e-6,
    )


def test_short_conv_sees_neither_the_future_nor_the_previous_document():
    x, w, positions = _docs([7, 7])
    base = causal_short_conv(x, w, positions, "kda")
    later = x.clone()
    later[5:] += 1.0  # the future of token 4, and the whole second document
    torch.testing.assert_close(causal_short_conv(later, w, positions, "kda")[:5], base[:5])
    earlier = x.clone()
    earlier[:7] += 1.0  # the whole first document
    torch.testing.assert_close(causal_short_conv(earlier, w, positions, "kda")[7:], base[7:])


def test_short_conv_gradients_match_the_reference():
    x, w, positions = _docs([3, 8])
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    xb, wb = x.double().requires_grad_(), w.double().requires_grad_()
    g = torch.randn(x.shape, dtype=torch.float64)
    (causal_short_conv(xa, wa, positions, "kda").double() * g).sum().backward()
    (_reference_conv(xb, wb, positions) * g).sum().backward()
    torch.testing.assert_close(xa.grad.double(), xb.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(wa.grad.double(), wb.grad, rtol=1e-5, atol=1e-5)


def _tiny(conv=4, sinks=True):
    cfg = model_registry("mis-8b-proxy")
    cfg.n_layers, cfg.swa_pattern, cfg.rope_pattern = 2, "SF", None
    cfg.layer.attention.conv_kernel_size = conv
    cfg.layer.attention.conv_init_fn_type = "delay" if conv else None
    cfg.layer.attention.conv_init_gain = 1.0 if conv else None
    cfg.layer.attention.conv_streams = "qkv" if conv else None
    cfg.layer.attention.conv_format = "kda" if conv else None
    cfg.layer.attention.attention_sinks = sinks
    cfg.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(), training=TrainingConfig(max_context_length=256)
        )
    )
    return cfg


def test_conv_weights_are_declared_and_disco_takes_them_as_matrices():
    from torchtitan.optimizers.disco.layout import classify, Family

    with torch.device("meta"):
        model = _tiny().build()
    layouts = model.param_layouts()
    convs = [n for n, l in layouts.items() if l.kind == "conv"]
    assert len(convs) == 6 and all(
        n.endswith(("q_conv.weight", "k_conv.weight", "v_conv.weight")) for n in convs
    )
    assert layouts["layers.0.attention.sinks"].kind == "vector"
    p = dict(model.named_parameters())[convs[0]]
    assert classify(convs[0], p, layouts[convs[0]], row_rule=True) is Family.LOCAL
    assert classify(convs[0], p, layouts[convs[0]], row_rule=False) is Family.REPLICATED


def test_features_off_add_nothing():
    with torch.device("meta"):
        plain = _tiny(conv=0, sinks=False).build()
    names = {n for n, _ in plain.named_parameters()}
    assert not any("conv" in n or n.endswith("sinks") for n in names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention needs CUDA")
def test_sinks_are_a_softmax_with_one_more_logit():
    from torch.nn.attention.flex_attention import create_block_mask

    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.gpt_oss.model import apply_attention_sink_rescale

    torch.manual_seed(0)
    t, h, d = 256, 4, 32
    q, k, v = (torch.randn(t, h, d, device="cuda") for _ in range(3))
    sinks = torch.randn(h, device="cuda")
    mask = create_block_mask(lambda b, hh, qi, ki: qi >= ki, 1, None, t, t, device="cuda")
    inner = FlexInnerAttention(FlexInnerAttention.Config())
    got = inner(
        q, k, v, attention_masks=mask,
        out_transform=lambda o, lse: apply_attention_sink_rescale(o, lse, sinks),
    )
    s = torch.einsum("qhd,khd->hqk", q.double(), k.double()) / d**0.5
    s = s.masked_fill(~torch.ones(t, t, dtype=torch.bool, device="cuda").tril(), float("-inf"))
    logits = torch.cat([s, sinks.double()[:, None, None].expand(h, t, 1)], dim=-1)
    p = torch.softmax(logits, dim=-1)[..., :t]  # the sink carries no value
    want = torch.einsum("hqk,khd->qhd", p, v.double())
    torch.testing.assert_close(got.double(), want, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention needs CUDA")
def test_model_with_conv_and_sinks_trains_one_step():
    cfg = _tiny()
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    sinks = model.layers["0"].attention.sinks
    assert 0 < sinks.std().item() < 0.05
    positions = torch.cat([torch.arange(100), torch.arange(156)]).cuda()
    tokens = torch.randint(0, cfg.vocab_size, (256,), device="cuda")
    logits = model(tokens, positions=positions, attention_masks=model.get_attention_masks(positions))
    F.cross_entropy(logits.float(), tokens.roll(-1)).backward()
    attn = model.layers["0"].attention
    for p in (attn.q_conv.weight, attn.v_conv.weight, attn.sinks):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
