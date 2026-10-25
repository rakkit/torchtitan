# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe's learned per-row output multipliers: y = m * (W x) must equal the
same projection with its rows pre-scaled, W_eff = Diag(m) W."""

from types import SimpleNamespace

import pytest

import torch
import torch.nn.functional as F

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.opt_moe import model_registry

MULT = (
    "v_multiplier",
    "o_multiplier",
    "w1_multiplier",
    "w3_multiplier",
    "w2_multiplier",
)
ALL = (("v", "o"), ("w1", "w3", "w2"))
RESIDUAL = (("o",), ("w2",))  # the two residual writes: attention and MLP block outputs
NONE = ((), ())
SEVEN = (
    ("q", "k", "v", "o"),
    ("w1", "w3", "w2"),
)  # every projection, as AngularMuown's magnitudes


def _model(mults, seed: int = 0, scalar: bool = False):
    cfg = model_registry("gauge-d8-w256")
    cfg.n_layers = 2
    cfg.layer.attention.row_multipliers, cfg.layer.feed_forward.row_multipliers = mults
    cfg.layer.attention.scalar_multipliers = scalar and bool(mults[0])
    cfg.layer.feed_forward.scalar_multipliers = scalar and bool(mults[1])
    cfg.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=64),
        )
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    return model.double()


class _CausalSDPA(torch.nn.Module):
    """Stand-in for flex attention on CPU: plain causal attention, [T, H, K]."""

    def forward(self, q, k, v, **kwargs):
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True
        )
        return out.transpose(0, 1)


def _v_rows(attn) -> torch.Tensor:
    """wqkv row indices of v, in v_multiplier order: rows are interleaved per KV
    group as [q_heads_per_kv, k, v] x n_kv_heads, head_dim rows each."""
    hd, n_kv = attn.head_dim, attn.n_kv_heads
    per_group = (attn.n_heads // n_kv + 2) * hd
    return torch.cat(
        [
            torch.arange(g * per_group + per_group - hd, (g + 1) * per_group)
            for g in range(n_kv)
        ]
    )


def test_off_adds_nothing_and_on_adds_only_the_multipliers():
    off = {n: p.shape for n, p in _model(NONE).named_parameters()}
    on = {n: p.shape for n, p in _model(ALL).named_parameters()}
    assert not any(n.endswith(MULT) for n in off)
    extra = {n: s for n, s in on.items() if n not in off}
    assert {n: s for n, s in on.items() if n in off} == off
    layer = _model(ALL).layers["0"]
    assert extra == {
        "layers.0.attention.v_multiplier": (
            layer.attention.n_kv_heads * layer.attention.head_dim,
        ),
        "layers.0.attention.o_multiplier": (256,),
        "layers.0.feed_forward.w1_multiplier": (1024,),
        "layers.0.feed_forward.w3_multiplier": (1024,),
        "layers.0.feed_forward.w2_multiplier": (256,),
        "layers.1.attention.v_multiplier": (
            layer.attention.n_kv_heads * layer.attention.head_dim,
        ),
        "layers.1.attention.o_multiplier": (256,),
        "layers.1.feed_forward.w1_multiplier": (1024,),
        "layers.1.feed_forward.w3_multiplier": (1024,),
        "layers.1.feed_forward.w2_multiplier": (256,),
    }


def test_multipliers_start_at_one_and_leave_every_other_weight_bitwise():
    off, on = dict(_model(NONE).named_parameters()), dict(
        _model(ALL).named_parameters()
    )
    for n, p in on.items():
        if n.endswith(MULT):
            assert torch.equal(p, torch.ones_like(p)), n
        else:
            assert torch.equal(p, off[n]), n


def test_ffn_multipliers_equal_prescaled_rows():
    torch.manual_seed(1)
    ffn_on, ffn_off = (
        _model(ALL).layers["0"].feed_forward,
        _model(NONE).layers["0"].feed_forward,
    )
    m1, m3, m2 = (
        torch.rand_like(getattr(ffn_on, f"{w}_multiplier")) + 0.5
        for w in ("w1", "w3", "w2")
    )
    with torch.no_grad():
        ffn_on.w1_multiplier.copy_(m1), ffn_on.w3_multiplier.copy_(
            m3
        ), ffn_on.w2_multiplier.copy_(m2)
        ffn_off.w13.weight[0] *= m1[:, None]
        ffn_off.w13.weight[1] *= m3[:, None]
        ffn_off.w2.weight *= m2[:, None]
    x = torch.randn(10, 256, dtype=torch.float64)
    torch.testing.assert_close(ffn_on(x), ffn_off(x), rtol=1e-12, atol=1e-12)


def test_attention_multipliers_equal_prescaled_rows():
    torch.manual_seed(2)
    a_on, a_off = _model(ALL).layers["0"].attention, _model(NONE).layers["0"].attention
    for a in (a_on, a_off):
        a.inner_attention = _CausalSDPA()
    mv, mo = (
        torch.rand_like(a_on.v_multiplier) + 0.5,
        torch.rand_like(a_on.o_multiplier) + 0.5,
    )
    with torch.no_grad():
        a_on.v_multiplier.copy_(mv), a_on.o_multiplier.copy_(mo)
        w = a_off.qkv_linear.wqkv.weight
        w[_v_rows(a_off)] *= mv[:, None]
        a_off.wo.weight *= mo[:, None]
    x, positions = torch.randn(12, 256, dtype=torch.float64), torch.arange(12)
    torch.testing.assert_close(
        a_on(x, None, positions), a_off(x, None, positions), rtol=1e-12, atol=1e-12
    )


def test_v_rows_are_the_v_block_of_wqkv():
    # The index map above against the model's own unit layout: scaling exactly
    # those rows must change only the v output of qkv_linear.
    a = _model(NONE).layers["0"].attention
    x = torch.randn(5, 256, dtype=torch.float64)
    q0, k0, v0 = a.qkv_linear(x)
    with torch.no_grad():
        a.qkv_linear.wqkv.weight[_v_rows(a)] *= 3.0
    q1, k1, v1 = a.qkv_linear(x)
    assert torch.equal(q0, q1) and torch.equal(k0, k1)
    torch.testing.assert_close(v1, 3.0 * v0)


def test_multipliers_at_one_are_an_exact_noop():
    m_on, m_off = _model(ALL), _model(NONE)
    x = torch.randn(9, 256, dtype=torch.float64)
    assert torch.equal(
        m_on.layers["0"].feed_forward(x), m_off.layers["0"].feed_forward(x)
    )
    a_on, a_off = m_on.layers["1"].attention, m_off.layers["1"].attention
    a_on.inner_attention, a_off.inner_attention = _CausalSDPA(), _CausalSDPA()
    positions = torch.arange(9)
    assert torch.equal(a_on(x, None, positions), a_off(x, None, positions))


def test_every_multiplier_gets_a_gradient():
    layer = _model(ALL).layers["0"]
    layer.attention.inner_attention = _CausalSDPA()
    x = torch.randn(8, 256, dtype=torch.float64)
    y = layer.feed_forward(x).sum() + layer.attention(x, None, torch.arange(8)).sum()
    y.backward()
    for name in MULT:
        mod = layer.attention if name[0] in "vo" else layer.feed_forward
        g = getattr(mod, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_residual_writes_only_add_o_and_w2():
    extra = {n for n, _ in _model(RESIDUAL).named_parameters()} - {
        n for n, _ in _model(NONE).named_parameters()
    }
    assert extra == {
        f"layers.{i}.{m}"
        for i in (0, 1)
        for m in ("attention.o_multiplier", "feed_forward.w2_multiplier")
    }


def test_residual_write_multipliers_equal_prescaled_rows():
    torch.manual_seed(3)
    on, off = _model(RESIDUAL).layers["0"], _model(NONE).layers["0"]
    on.attention.inner_attention, off.attention.inner_attention = (
        _CausalSDPA(),
        _CausalSDPA(),
    )
    mo = torch.rand_like(on.attention.o_multiplier) + 0.5
    m2 = torch.rand_like(on.feed_forward.w2_multiplier) + 0.5
    with torch.no_grad():
        on.attention.o_multiplier.copy_(mo), on.feed_forward.w2_multiplier.copy_(m2)
        off.attention.wo.weight *= mo[:, None]
        off.feed_forward.w2.weight *= m2[:, None]
    x, positions = torch.randn(11, 256, dtype=torch.float64), torch.arange(11)
    torch.testing.assert_close(
        on.attention(x, None, positions),
        off.attention(x, None, positions),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        on.feed_forward(x), off.feed_forward(x), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("mults", [(("wq",), ()), ((), ("w13",)), (("o",), ("v",))])
def test_unknown_multiplier_names_are_rejected(mults):
    with pytest.raises(ValueError, match="row_multipliers"):
        _model(mults)


@pytest.mark.parametrize(
    "mults, scalar",
    [
        (ALL, False),
        (RESIDUAL, False),
        (NONE, False),
        (ALL, True),
        (RESIDUAL, True),
        (SEVEN, False),
    ],
)
def test_every_parameter_has_a_sharding_placement(mults, scalar):
    """fully_shard under DP needs every parameter to be a DTensor; a direct
    parameter without a placement stays a plain tensor and crashes at startup."""
    from torchtitan.models.opt_moe.sharding import set_opt_moe_sharding_config

    cfg = model_registry("gauge-d8-w256")
    cfg.n_layers = 2
    cfg.layer.attention.row_multipliers, cfg.layer.feed_forward.row_multipliers = mults
    cfg.layer.attention.scalar_multipliers = scalar and bool(mults[0])
    cfg.layer.feed_forward.scalar_multipliers = scalar and bool(mults[1])
    cfg.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=64),
        )
    )
    set_opt_moe_sharding_config(cfg, enable_sp=False, enable_tp=False, enable_ep=False)
    with torch.device("meta"):
        model = cfg.build()
    uncovered = [
        f"{mn}.{pn}"
        for mn, mod in model.named_modules()
        for pn, _ in mod.named_parameters(recurse=False)
        if getattr(mod, "_sharding_config", None) is None
        or pn not in mod._sharding_config.state_shardings
    ]
    assert not uncovered, uncovered


def test_scalar_multipliers_are_one_element_each():
    shapes = {
        n: p.shape
        for n, p in _model(ALL, scalar=True).named_parameters()
        if n.endswith(MULT)
    }
    assert len(shapes) == 10 and set(shapes.values()) == {(1,)}


def test_scalar_multipliers_equal_a_prescaled_matrix():
    torch.manual_seed(4)
    on, off = _model(ALL, scalar=True).layers["0"], _model(NONE).layers["0"]
    on.attention.inner_attention, off.attention.inner_attention = (
        _CausalSDPA(),
        _CausalSDPA(),
    )
    m = {n: torch.rand(1, dtype=torch.float64) + 0.5 for n in MULT}
    with torch.no_grad():
        for n, v in m.items():
            getattr(on.attention if n[0] in "vo" else on.feed_forward, n).copy_(v)
        off.attention.qkv_linear.wqkv.weight[_v_rows(off.attention)] *= m[
            "v_multiplier"
        ]
        off.attention.wo.weight *= m["o_multiplier"]
        off.feed_forward.w13.weight[0] *= m["w1_multiplier"]
        off.feed_forward.w13.weight[1] *= m["w3_multiplier"]
        off.feed_forward.w2.weight *= m["w2_multiplier"]
    x, positions = torch.randn(10, 256, dtype=torch.float64), torch.arange(10)
    torch.testing.assert_close(
        on.attention(x, None, positions),
        off.attention(x, None, positions),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        on.feed_forward(x), off.feed_forward(x), rtol=1e-12, atol=1e-12
    )


def test_scalar_multipliers_at_one_are_an_exact_noop_and_get_gradients():
    on, off = _model(ALL, scalar=True).layers["0"], _model(NONE).layers["0"]
    on.attention.inner_attention, off.attention.inner_attention = (
        _CausalSDPA(),
        _CausalSDPA(),
    )
    x, positions = torch.randn(8, 256, dtype=torch.float64), torch.arange(8)
    ya, yf = on.attention(x, None, positions), on.feed_forward(x)
    assert torch.equal(ya, off.attention(x, None, positions))
    assert torch.equal(yf, off.feed_forward(x))
    (ya.sum() + yf.sum()).backward()
    for name in MULT:
        g = getattr(on.attention if name[0] in "vo" else on.feed_forward, name).grad
        assert (
            g is not None
            and g.shape == (1,)
            and torch.isfinite(g).all()
            and g.abs().sum() > 0
        ), name


@pytest.mark.parametrize("module", ["attention", "feed_forward"])
def test_scalar_without_multipliers_is_rejected(module):
    cfg = model_registry("gauge-d8-w256")
    cfg.n_layers = 2
    getattr(cfg.layer, module).scalar_multipliers = True
    cfg.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=64),
        )
    )
    with pytest.raises(ValueError, match="scalar_multipliers needs row_multipliers"):
        with torch.device("meta"):
            cfg.build()


def _qk_rows(attn):
    """wqkv row indices of q (in q_multiplier order: head h of KV group h // q_heads_per_kv)
    and of k (k_multiplier order), from the per-group layout [q_heads_per_kv, k, v]."""
    hd, n_kv = attn.head_dim, attn.n_kv_heads
    hpk = attn.n_heads // n_kv
    per_group = (hpk + 2) * hd
    q = torch.cat(
        [torch.arange(g * per_group, g * per_group + hpk * hd) for g in range(n_kv)]
    )
    k = torch.cat(
        [
            torch.arange(g * per_group + hpk * hd, g * per_group + (hpk + 1) * hd)
            for g in range(n_kv)
        ]
    )
    return q, k


def test_qk_rows_are_the_q_and_k_blocks_of_wqkv():
    a = _model(NONE).layers["0"].attention
    qr, kr = _qk_rows(a)
    x = torch.randn(5, 256, dtype=torch.float64)
    q0, k0, v0 = a.qkv_linear(x)
    with torch.no_grad():
        a.qkv_linear.wqkv.weight[qr] *= 3.0
        a.qkv_linear.wqkv.weight[kr] *= 5.0
    q1, k1, v1 = a.qkv_linear(x)
    assert torch.equal(v0, v1)
    torch.testing.assert_close(q1, 3.0 * q0)
    torch.testing.assert_close(k1, 5.0 * k0)


def test_qk_multipliers_equal_prescaled_rows():
    torch.manual_seed(6)
    a_on, a_off = (
        _model(SEVEN).layers["0"].attention,
        _model(NONE).layers["0"].attention,
    )
    for a in (a_on, a_off):
        a.inner_attention = _CausalSDPA()
    mq, mk = (
        torch.rand_like(a_on.q_multiplier) + 0.5,
        torch.rand_like(a_on.k_multiplier) + 0.5,
    )
    mv, mo = (
        torch.rand_like(a_on.v_multiplier) + 0.5,
        torch.rand_like(a_on.o_multiplier) + 0.5,
    )
    qr, kr = _qk_rows(a_off)
    with torch.no_grad():
        a_on.q_multiplier.copy_(mq), a_on.k_multiplier.copy_(mk)
        a_on.v_multiplier.copy_(mv), a_on.o_multiplier.copy_(mo)
        w = a_off.qkv_linear.wqkv.weight
        w[qr] *= mq[:, None]
        w[kr] *= mk[:, None]
        w[_v_rows(a_off)] *= mv[:, None]
        a_off.wo.weight *= mo[:, None]
    x, positions = torch.randn(12, 256, dtype=torch.float64), torch.arange(12)
    torch.testing.assert_close(
        a_on(x, None, positions), a_off(x, None, positions), rtol=1e-12, atol=1e-12
    )


def test_qk_multipliers_start_at_one_get_gradients_and_leave_other_weights_bitwise():
    off, on = dict(_model(NONE).named_parameters()), dict(
        _model(SEVEN).named_parameters()
    )
    extra = {n for n in on if n not in off}
    assert {n for n in extra if n.endswith(("q_multiplier", "k_multiplier"))} == {
        f"layers.{i}.attention.{m}_multiplier" for i in (0, 1) for m in "qk"
    }
    for n, p in on.items():
        assert (
            torch.equal(p, torch.ones_like(p)) if n in extra else torch.equal(p, off[n])
        ), n
    layer = _model(SEVEN).layers["0"]
    layer.attention.inner_attention = _CausalSDPA()
    x = torch.randn(8, 256, dtype=torch.float64)
    (
        layer.attention(x, None, torch.arange(8))
        * torch.randn(8, 256, dtype=torch.float64)
    ).sum().backward()
    for m in (layer.attention.q_multiplier, layer.attention.k_multiplier):
        assert (
            m.grad is not None
            and torch.isfinite(m.grad).all()
            and m.grad.abs().sum() > 0
        )


@pytest.mark.parametrize("which", ["q", "k"])
def test_scalar_qk_multipliers_are_refused(which):
    with pytest.raises(ValueError, match="qk-norm"):
        _model(((which,), ()), scalar=True)
