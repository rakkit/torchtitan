# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe attention's optional short conv (Canon: x + conv(x)) and sinks
(gpt-oss), against references written from their equations."""


import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config


def _reference_conv(x, w, positions):
    """Per document: `x + conv1d(x, w, padding=K-1)[:L]` (Canon), in fp64."""
    k, out = w.shape[-1], torch.empty_like(x, dtype=torch.float64)
    starts = (positions == 0).nonzero().flatten().tolist() + [x.shape[0]]
    for a, b in zip(starts[:-1], starts[1:]):
        doc = x[a:b].double().T[None]  # [1, C, L]
        y = F.conv1d(doc, w.double(), padding=k - 1, groups=w.shape[0])[0, :, : b - a]
        out[a:b] = x[a:b].double() + y.T
    return out


def _docs(lengths, c=6, seed=0):
    torch.manual_seed(seed)
    positions = torch.cat([torch.arange(n) for n in lengths]).cuda()
    return (
        torch.randn(len(positions), c, device="cuda"),
        torch.randn(c, 1, 4, device="cuda"),
        positions,
    )


def _conv(x, w, positions):
    from torchtitan.models.common.attention import create_varlen_metadata_for_document
    from torchtitan.models.common.attention.linear_attention import (
        LinearAttentionMetadata,
    )
    from torchtitan.models.opt_moe.gated_swa_attention import canon_short_conv

    meta = LinearAttentionMetadata(
        varlen=create_varlen_metadata_for_document(positions),
        num_conv_history_tokens=w.shape[-1] - 1,
    )
    return canon_short_conv(x, w, meta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the conv kernel needs CUDA")
def test_short_conv_is_a_causal_depthwise_conv_per_document():
    x, w, positions = _docs([1, 5, 2, 9])
    torch.testing.assert_close(
        _conv(x, w, positions).double(),
        _reference_conv(x, w, positions),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the conv kernel needs CUDA")
def test_short_conv_sees_neither_the_future_nor_the_previous_document():
    x, w, positions = _docs([7, 7])
    base = _conv(x, w, positions)
    later = x.clone()
    later[5:] += 1.0  # the future of token 4, and the whole second document
    torch.testing.assert_close(_conv(later, w, positions)[:5], base[:5])
    earlier = x.clone()
    earlier[:7] += 1.0  # the whole first document
    torch.testing.assert_close(_conv(earlier, w, positions)[7:], base[7:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the conv kernel needs CUDA")
def test_short_conv_gradients_match_the_reference():
    x, w, positions = _docs([3, 8])
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    xb, wb = x.double().requires_grad_(), w.double().requires_grad_()
    g = torch.randn(x.shape, dtype=torch.float64, device="cuda")
    (_conv(xa, wa, positions).double() * g).sum().backward()
    (_reference_conv(xb, wb, positions) * g).sum().backward()
    torch.testing.assert_close(xa.grad.double(), xb.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(wa.grad.double(), wb.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the conv kernel needs CUDA")
def test_short_conv_at_the_production_width_in_bf16():
    """The fused k / v conv of the 30B shapes (2 x 4 KV heads of 128: 1024
    channels), bf16 as in training, over a 32768-token row of packed
    documents (1 to 3000 tokens): the output and both gradients are the fp64
    reference's within bf16 rounding. The kernel accumulates in fp32 and rounds
    twice, as training does (conv(x) to bf16, then the bf16 add x + conv(x)),
    so the bound is in the two terms' sizes: 2^-7 (|x| + |conv(x)|)."""
    g = torch.Generator().manual_seed(0)
    lengths, left = [], 32768
    while left > 0:
        lengths.append(min(left, int(torch.randint(1, 3001, (1,), generator=g))))
        left -= lengths[-1]
    x, w, positions = _docs(lengths, c=1024)
    x, w = x.bfloat16(), (0.5 * w).bfloat16()
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    xb, wb = x.double().requires_grad_(), w.double().requires_grad_()
    out = _conv(xa, wa, positions)
    want = _reference_conv(xb, wb, positions)
    grad = torch.randn(x.shape, device="cuda").bfloat16()
    (out * grad).sum().backward()
    (want * grad.double()).sum().backward()
    assert out.dtype == torch.bfloat16

    def within_two_roundings(got, want, residual):
        bound = 2**-7 * (residual.abs() + (want - residual).abs()) + 1e-6
        assert ((got.double() - want).abs() <= bound).all()

    within_two_roundings(out, want.detach(), xb.detach())
    within_two_roundings(xa.grad, xb.grad, grad.double())
    # the weight gradient: fp32 sums over the row, one bf16 rounding
    scale = wb.grad.abs().max()
    torch.testing.assert_close(
        wa.grad.double() / scale, wb.grad / scale, rtol=8e-3, atol=1e-3
    )


def _tiny(conv=4, sinks=True, sinks_init_fn_type=None, sinks_init_std=None, gate=None):
    attention = dict(
        conv_kernel_size=conv,  # the other conv arguments: their defaults
        attention_sinks=sinks,
        gated_attention_type=gate,
    )
    if sinks_init_fn_type is not None:
        attention["sinks_init_fn_type"] = sinks_init_fn_type
    if sinks_init_std is not None:
        attention["sinks_init_std"] = sinks_init_std
    return build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=256,
        n_layers=2,
        swa_pattern="SF",
        rope_pattern=None,
        **attention,
    )


def test_conv_weights_are_declared_and_disco_takes_them_as_matrices():
    from torchtitan.optimizers.disco.layout import classify, Family

    with torch.device("meta"):
        model = _tiny().build()
    layouts = model.param_layouts()
    convs = [n for n, l in layouts.items() if l.kind == "conv"]
    assert len(convs) == 4 and all(
        n.endswith(("k_conv.weight", "v_conv.weight")) for n in convs
    )
    assert layouts["layers.0.attention.sinks"].kind == "vector"
    p = dict(model.named_parameters())[convs[0]]
    assert classify(convs[0], p, layouts[convs[0]], row_rule=True) is Family.LOCAL
    assert classify(convs[0], p, layouts[convs[0]], row_rule=False) is Family.REPLICATED


def test_the_gate_is_none_or_element_wise():
    """Only the element-wise gate exists: gate_proj maps dim to every output
    channel; no gate adds no parameter; any other value is refused at build."""
    with torch.device("meta"):
        gated = _tiny(conv=0, sinks=False, gate="element-wise").build()
        plain = _tiny(conv=0, sinks=False, gate=None).build()
    at = gated.layers["0"].attention
    dim = at.wo.weight.shape[0]  # wo: n_heads * head_dim -> dim
    assert tuple(at.gate_proj.weight.shape) == (at.n_heads * at.head_dim, dim)
    assert not any("gate_proj" in n for n, _ in plain.named_parameters())
    for bad in ("head-wise", "none"):
        with pytest.raises(ValueError, match="element-wise"):
            with torch.device("meta"):
                _tiny(conv=0, sinks=False, gate=bad).build()


def test_features_off_add_nothing():
    with torch.device("meta"):
        plain = _tiny(conv=0, sinks=False).build()
    names = {n for n, _ in plain.named_parameters()}
    assert not any("conv" in n or n.endswith("sinks") for n in names)


def test_sink_rescale_is_the_softmax_with_one_more_logit_on_cpu():
    """out * sigmoid(lse - s) is the softmax over the keys plus one logit s that carries no value."""
    from torchtitan.models.opt_moe import gated_swa_attention as attention_sinks

    torch.manual_seed(0)
    t, h, d = 12, 3, 8
    q, k, v = (torch.randn(t, h, d, dtype=torch.float64) for _ in range(3))
    sinks = torch.randn(h, dtype=torch.float64)
    s = torch.einsum("qhd,khd->hqk", q, k) / d**0.5
    s = s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))
    out = torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), v)
    lse = torch.logsumexp(s, dim=-1).T  # [t, h], as flex attention returns it
    got = attention_sinks.apply_attention_sink_rescale(out, lse, sinks)
    logits = torch.cat([s, sinks[:, None, None].expand(h, t, 1)], dim=-1)
    want = torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1)[..., :t], v)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def _init_on_cpu(cfg):
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cpu")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    return model


def test_sinks_are_zero_at_init_by_default():
    model = _init_on_cpu(_tiny(conv=0, sinks=True))
    sinks = [model.layers[str(i)].attention.sinks for i in range(2)]
    assert all(
        s.shape == (model.layers["0"].attention.n_heads,) and (s == 0).all()
        for s in sinks
    )
    assert all(s.requires_grad for s in sinks)  # still learned


def test_sinks_take_any_init_type_of_the_init_helper():
    model = _init_on_cpu(
        _tiny(
            conv=0, sinks=True, sinks_init_fn_type="trunc_normal", sinks_init_std=0.02
        )
    )
    sinks = torch.cat([model.layers[str(i)].attention.sinks for i in range(2)])
    assert 0 < sinks.std().item() < 0.05 and (sinks != 0).any()  # std 0.02
    with pytest.raises(NotImplementedError, match="init_fn_type"):
        with torch.device("meta"):
            _tiny(conv=0, sinks=True, sinks_init_fn_type="no_such_init").build()


def test_sink_metrics_log_every_value_of_every_layer():
    from torch import nn

    from torchtitan.optimizers import model_stats

    class Attn(nn.Module):
        def __init__(self, vals):
            super().__init__()
            self.sinks = nn.Parameter(torch.tensor(vals))

    class Part(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict(
                {
                    "0": nn.ModuleDict({"attention": Attn([0.0, 0.25])}),
                    "1": nn.ModuleDict({"attention": Attn([-1.5, 2.0])}),
                }
            )

    got = model_stats.sink_metrics([Part()], parallelism_context=None)
    assert got == {
        "sink_value/layers.0.attention/head-0": 0.0,
        "sink_value/layers.0.attention/head-1": 0.25,
        "sink_value/layers.1.attention/head-0": -1.5,
        "sink_value/layers.1.attention/head-1": 2.0,
    }
    assert model_stats.sink_metrics([nn.Linear(2, 2)], parallelism_context=None) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention needs CUDA")
def test_sinks_are_a_softmax_with_one_more_logit():
    from torch.nn.attention.flex_attention import create_block_mask

    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.opt_moe.gated_swa_attention import (
        apply_attention_sink_rescale,
    )

    torch.manual_seed(0)
    t, h, d = 256, 4, 32
    q, k, v = (torch.randn(t, h, d, device="cuda") for _ in range(3))
    sinks = torch.randn(h, device="cuda")
    mask = create_block_mask(
        lambda b, hh, qi, ki: qi >= ki, 1, None, t, t, device="cuda"
    )
    inner = FlexInnerAttention(FlexInnerAttention.Config())
    got = inner(
        q,
        k,
        v,
        attention_metadata=mask,
        out_transform=lambda o, lse: apply_attention_sink_rescale(o, lse, sinks),
    )
    s = torch.einsum("qhd,khd->hqk", q.double(), k.double()) / d**0.5
    s = s.masked_fill(
        ~torch.ones(t, t, dtype=torch.bool, device="cuda").tril(), float("-inf")
    )
    logits = torch.cat([s, sinks.double()[:, None, None].expand(h, t, 1)], dim=-1)
    p = torch.softmax(logits, dim=-1)[..., :t]  # the sink carries no value
    want = torch.einsum("hqk,khd->qhd", p, v.double())
    torch.testing.assert_close(got.double(), want, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention needs CUDA")
def test_model_with_conv_and_sinks_trains_one_step():
    # non-zero sinks (their rescale acts): the default std 0.0 would divide by zero
    cfg = _tiny(sinks_init_fn_type="trunc_normal", sinks_init_std=0.02)
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
    logits = model(
        tokens,
        positions=positions,
        attention_metadata=model._get_attention_metadata(positions),
    )
    F.cross_entropy(logits.float(), tokens.roll(-1)).backward()
    attn = model.layers["0"].attention
    for p in (attn.k_conv.weight, attn.v_conv.weight, attn.sinks):
        assert (
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention needs CUDA")
def test_training_and_no_grad_sinks_give_the_same_output():
    """Training applies the sink rescale after the attention's remat region (so
    the backward recomputes it); no-grad paths (validation, the RL generator's
    injected hook) apply it as the attention's out_transform. Same math, so the
    outputs are bitwise equal."""
    cfg = _tiny(sinks_init_fn_type="trunc_normal", sinks_init_std=0.5)
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    positions = torch.cat([torch.arange(100), torch.arange(156)]).cuda()
    tokens = torch.randint(0, cfg.vocab_size, (256,), device="cuda")
    metadata = model._get_attention_metadata(positions)
    trained = model(tokens, positions=positions, attention_metadata=metadata)
    with torch.no_grad():
        scored = model(tokens, positions=positions, attention_metadata=metadata)
    assert trained.requires_grad
    assert torch.equal(trained.detach(), scored)
