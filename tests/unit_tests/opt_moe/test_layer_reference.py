# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One whole opt_moe decoder layer against a reference written from its
equations (fp64, plain PyTorch, reading the layer's weights), forward and
backward, on packed documents with the real flex attention:

    h   = a x + b Attn(RMSNorm_1(x)),      out = a h + b FFN(RMSNorm_2(h))
    Attn(u): [q | k | v] = Wqkv u (rows per KV group: its q heads, k, v);
             g = Wg u; q, k, v <- x + conv(x) (Canon, causal, per document);
             q, k <- RMSNorm per head; q, k <- RoPE on the first rope.dim
             dims (or none); softmax(q k^T / sqrt(d)) over the same document,
             causal, within the window, with one value-less logit s_h per
             head (sinks); o <- o * sigmoid(g); Wo o
    FFN: SwiGLU, or the MoE (sigmoid scores, top-k on score + bias, weights
         the raw scores renormalized x route_scale, plus the shared expert).

The component tests (conv, sinks, MoE, RoPE, residual scale) check each piece;
this one checks how the layer wires them: the fused-QKV split per KV group,
the order conv -> qk-norm -> RoPE, the gate's input, SWA + document mask +
sinks together, the residual scales.
"""

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config

T, DIM, HEADS, KV_HEADS, HEAD_DIM, WINDOW, THETA = 256, 128, 4, 2, 32, 32, 10000.0
DOCS = (100, 3, 153)

# Each variant: (layer 0 dense, layer 1 MoE) settings.
VARIANTS = {
    "all-on": dict(
        swa_pattern="SF",
        rope_pattern="RN",
        rope_dim=16,  # partial RoPE on layer 0, NoPE on layer 1
        attention=dict(
            conv_kernel_size=4,
            conv_streams="qkv",
            gated_attention_type="element-wise",
            attention_sinks=True,
            sinks_init_fn_type="trunc_normal",
            sinks_init_std=0.5,
        ),
        residual_scale="depth_scale_N",  # a = b = 1/2 at 2 layers
    ),
    "kv-conv-full-rope": dict(
        swa_pattern="FS",
        rope_pattern="NR",
        rope_dim=HEAD_DIM,
        attention=dict(conv_kernel_size=4, conv_streams="kv"),
        residual_scale="complete_p_base1",  # a = 1, b = 1/2
    ),
    "plain": dict(
        swa_pattern="FF",
        rope_pattern="RR",
        rope_dim=HEAD_DIM,
        attention=dict(),
        residual_scale="identity",
    ),
}


def _model(variant: dict):
    cfg = build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=T,
        n_layers=2,
        n_dense_layers=1,
        dim=DIM,
        vocab_size=512,
        sliding_window_size=WINDOW,
        swa_pattern=variant["swa_pattern"],
        rope_pattern=variant["rope_pattern"],
        rope=dict(dim=variant["rope_dim"], theta=THETA),
        n_heads=HEADS,
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        **variant["attention"],
        dense_hidden_dim=96,
        moe_hidden_dim=48,
        num_experts=8,
        top_k=2,
        residual_scale=variant["residual_scale"],
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
        # Move every gain, conv tap and the expert bias off its init value, so
        # each enters the forward non-trivially.
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.add_(0.2 * torch.randn_like(p))
            if "_conv." in name:
                p.add_(0.3 * torch.randn_like(p))
        for layer in model.layers.values():
            bias = getattr(layer.feed_forward, "expert_bias_E", None)
            if bias is not None:
                bias.copy_(0.05 * torch.randn_like(bias))
    return cfg, model


def _rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _conv(x_TC, w_C1K, positions):
    """Per document: x + conv1d(x, w, padding K-1)[:L] (depthwise, causal)."""
    k, out = w_C1K.shape[-1], []
    starts = (positions == 0).nonzero().flatten().tolist() + [x_TC.shape[0]]
    for a, b in zip(starts[:-1], starts[1:]):
        doc = x_TC[a:b].T[None]
        y = F.conv1d(doc, w_C1K, padding=k - 1, groups=w_C1K.shape[0])[0, :, : b - a]
        out.append(x_TC[a:b] + y.T)
    return torch.cat(out)


def _rope(x_THK, positions, dim):
    """Rotate-half RoPE on the first `dim` dims of every head."""
    inv = 1.0 / THETA ** (
        torch.arange(0, dim, 2, dtype=torch.float64, device=x_THK.device) / dim
    )
    ang = positions.double()[:, None] * inv[None]
    ang = torch.cat([ang, ang], dim=-1)[:, None, :]
    r, rest = x_THK[..., :dim], x_THK[..., dim:]
    rot = torch.cat([-r[..., dim // 2 :], r[..., : dim // 2]], dim=-1)
    return torch.cat([r * ang.cos() + rot * ang.sin(), rest], dim=-1)


def _attention(q, k, v, positions, window, sinks):
    reps = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(reps, dim=1), v.repeat_interleave(reps, dim=1)
    doc = torch.cumsum((positions == 0).long(), dim=0)
    i = torch.arange(q.shape[0], device=q.device)
    mask = (doc[:, None] == doc[None, :]) & (i[:, None] >= i[None, :])
    if window is not None:
        mask &= (i[:, None] - i[None, :]) < window
    s = torch.einsum("qhd,khd->hqk", q, k) / q.shape[-1] ** 0.5
    s = s.masked_fill(~mask, float("-inf"))
    if sinks is not None:
        sink = sinks[:, None, None].expand(-1, q.shape[0], 1)
        p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1]
    else:
        p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, v)


def _swiglu(x, w13, w2):
    return (F.silu(x @ w13[0].T) * (x @ w13[1].T)) @ w2.T


def reference_layer(layer, layer_cfg, P, x, positions, eps, module_ids=None):
    """The layer's output from the equations; `P`: its parameters in fp64,
    `layer_cfg`: its config (the window: the inner attention keeps it there).
    `module_ids`: the layer's own top-k expert ids, used only for tokens whose
    fp64 choice is within 1e-6 of a tie (rounding decides those)."""
    a, b = layer.identity_scale, layer.block_scale
    att = layer.attention
    n = x.shape[0]
    u = _rms(x, P["attention_norm.weight"], eps)
    per_kv = HEADS // KV_HEADS
    qkv = (u @ P["attention.qkv_linear.wqkv.weight"].T).view(
        n, KV_HEADS, per_kv + 2, HEAD_DIM
    )
    q = qkv[:, :, :per_kv].reshape(n, HEADS, HEAD_DIM)
    k, v = qkv[:, :, per_kv], qkv[:, :, per_kv + 1]
    gate = (
        u @ P["attention.gate_proj.weight"].T
        if "attention.gate_proj.weight" in P
        else None
    )
    for name in ("q", "k", "v"):
        w = P.get(f"attention.{name}_conv.weight")
        if w is not None:
            t = {"q": q, "k": k, "v": v}[name]
            t = _conv(t.flatten(1), w, positions).view_as(t)
            q, k, v = (t if m == name else o for m, o in (("q", q), ("k", k), ("v", v)))
    q = _rms(q, P["attention.q_norm.weight"], eps)
    k = _rms(k, P["attention.k_norm.weight"], eps)
    if att.rope is not None:
        q, k = _rope(q, positions, att.qk_rope_dim), _rope(
            k, positions, att.qk_rope_dim
        )
    window = getattr(layer_cfg.attention.inner_attention, "sliding_window_size", None)
    o = _attention(q, k, v, positions, window, P.get("attention.sinks"))
    o = o.reshape(n, HEADS * HEAD_DIM)
    if gate is not None:
        o = o * torch.sigmoid(gate)
    h = a * x + b * (o @ P["attention.wo.weight"].T)
    u2 = _rms(h, P["ffn_norm.weight"], eps)
    if "feed_forward.w13.weight" in P:
        f = _swiglu(u2, P["feed_forward.w13.weight"], P["feed_forward.w2.weight"])
    else:
        moe = layer.feed_forward
        scores = torch.sigmoid(u2 @ P["feed_forward.router.gate.weight"].T)
        choice = scores + moe.expert_bias_E.double()
        top = torch.topk(choice, k=moe.router.top_k + 1, dim=-1)
        ids = top.indices[:, :-1]
        tie = (top.values[:, -2] - top.values[:, -1]) < 1e-6
        if tie.any():
            ids = torch.where(tie[:, None], module_ids, ids)
        w = scores.gather(-1, ids)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * moe.router.route_scale
        w13 = P["feed_forward.routed_experts.w13.weight"]
        w2 = P["feed_forward.routed_experts.w2.weight"]
        f = torch.zeros_like(u2)
        for slot in range(ids.shape[1]):
            for e in ids[:, slot].unique().tolist():
                rows = (ids[:, slot] == e).nonzero().flatten()
                y = _swiglu(u2[rows], w13[e], w2[e])
                f = f.index_add(0, rows, y * w[rows, slot, None])
        f = f + _swiglu(
            u2,
            P["feed_forward.shared_experts.w13.weight"],
            P["feed_forward.shared_experts.w2.weight"],
        )
    return a * h + b * f


def _rel(got, want):
    return ((got.double() - want).norm() / want.norm()).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention needs CUDA")
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_layer_matches_its_equations(variant):
    cfg, model = _model(VARIANTS[variant])
    eps = cfg.layers[0].attention_norm.eps
    positions = torch.cat([torch.arange(n) for n in DOCS]).cuda()
    metadata = model._get_attention_metadata(positions)
    torch.manual_seed(1)
    for layer_id, layer in model.layers.items():
        x = torch.randn(T, DIM, device="cuda", requires_grad=True)
        captured = {}
        router = getattr(layer.feed_forward, "router", None)
        hook = (
            router.register_forward_hook(
                lambda _m, _i, out: captured.update(ids=out[1].detach())
            )
            if router is not None
            else None
        )
        out = layer(x, metadata.get(type(layer.attention.inner_attention)), positions)
        if hook is not None:
            hook.remove()
        g = torch.randn_like(out)
        (out * g).sum().backward()

        P = {
            n: p.detach().double().requires_grad_() for n, p in layer.named_parameters()
        }
        xr = x.detach().double().requires_grad_()
        ref = reference_layer(
            layer, cfg.layers[int(layer_id)], P, xr, positions, eps, captured.get("ids")
        )
        (ref * g.double()).sum().backward()

        # Dense layers run in fp32 end to end; the MoE's grouped GEMM runs in bf16.
        tol = 5e-5 if router is None else 2e-2
        assert _rel(out, ref) < tol, (layer_id, "output", _rel(out, ref))
        assert _rel(x.grad, xr.grad) < tol, (layer_id, "input grad")
        for name, p in layer.named_parameters():
            err = _rel(p.grad, P[name].grad)
            # The router gate's gradient flows through the bf16 experts and
            # HiMidLo's fp32 emulation: the loosest of the MoE tolerances.
            assert err < (tol if "router" not in name else 5e-2), (
                layer_id,
                name,
                err,
            )
