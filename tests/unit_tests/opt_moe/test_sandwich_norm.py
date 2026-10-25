# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe's sandwich norm (`post_norm_type`): each branch output is normalised
before the residual add, h = x + g * rms(attn(pre(x))), out = h + g * rms(ffn(pre(h))).
References are written from that equation, not from the module code."""

from types import SimpleNamespace

import pytest

import torch
import torch.nn.functional as F

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.opt_moe import model_registry

POST = ("post_attention_norm", "post_ffn_norm")
EPS = 1e-30  # gauge flavors' norm_eps


def _cfg(post: str, gains: bool = False):
    cfg = model_registry("gauge-d8-w256")
    cfg.n_layers = 2
    cfg.layer.post_norm_type = post
    if gains:
        cfg.layer.norm_type = "rmsnorm"
        cfg.layer.attention.norm_type = "rmsnorm"
    cfg.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=64),
        )
    )
    return cfg


def _model(post: str, gains: bool = False, seed: int = 0):
    with torch.device("meta"):
        model = _cfg(post, gains).build()
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


def _rms(y, g=None):
    out = y / torch.sqrt(y.pow(2).mean(-1, keepdim=True) + EPS)
    return out if g is None else out * g


def _block(model, i: int):
    layer = model.layers[str(i)]
    layer.attention.inner_attention = _CausalSDPA()
    return layer


def test_default_is_the_plain_pre_norm_block():
    assert _cfg("none").layer.post_norm_type == "none"
    layer = _model("none").layers["0"]
    assert layer.post_attention_norm is None and layer.post_ffn_norm is None


@pytest.mark.parametrize("gains", [False, True])
def test_parameter_free_version_adds_no_parameter_and_changes_no_weight(gains):
    off, on = dict(_model("none", gains).named_parameters()), dict(
        _model("np_rmsnorm", gains).named_parameters()
    )
    assert on.keys() == off.keys()
    for n, p in on.items():
        assert torch.equal(p, off[n]), n


def test_gains_version_adds_one_unit_gain_per_branch_and_changes_no_weight():
    off, on = dict(_model("none", True).named_parameters()), dict(
        _model("rmsnorm", True).named_parameters()
    )
    extra = {n: p for n, p in on.items() if n not in off}
    assert set(extra) == {f"layers.{i}.{p}.weight" for i in (0, 1) for p in POST}
    for n, p in extra.items():
        assert p.shape == (256,) and torch.equal(p, torch.ones_like(p)), n
    for n, p in off.items():
        assert torch.equal(on[n], p), n


@pytest.mark.parametrize("post", ["none", "np_rmsnorm", "rmsnorm"])
def test_block_equals_its_equation(post):
    torch.manual_seed(5)
    layer = _block(_model(post, gains=True), 0)
    if post == "rmsnorm":
        with torch.no_grad():
            for p in POST:
                getattr(layer, p).weight.copy_(
                    torch.rand(256, dtype=torch.float64) + 0.5
                )
    x, positions = torch.randn(12, 256, dtype=torch.float64), torch.arange(12)

    def write(branch, name):
        if post == "none":
            return branch
        g = getattr(layer, name).weight if post == "rmsnorm" else None
        return _rms(branch, g)

    h = x + write(
        layer.attention(layer.attention_norm(x), None, positions), "post_attention_norm"
    )
    ref = h + write(layer.feed_forward(layer.ffn_norm(h)), "post_ffn_norm")
    torch.testing.assert_close(layer(x, None, positions), ref, rtol=1e-12, atol=1e-12)


def test_parameter_free_branch_writes_have_unit_rms_per_token():
    layer = _block(_model("np_rmsnorm"), 1)
    x, positions = 7.0 * torch.randn(9, 256, dtype=torch.float64), torch.arange(9)
    h = layer(x, None, positions)
    # out - x = two unit-RMS writes; the attention write alone is recoverable from a hook.
    seen = {}
    layer.post_attention_norm.register_forward_hook(lambda m, i, o: seen.update(a=o))
    layer.post_ffn_norm.register_forward_hook(lambda m, i, o: seen.update(f=o))
    layer(x, None, positions)
    for w in (seen["a"], seen["f"]):
        torch.testing.assert_close(
            w.pow(2).mean(-1).sqrt(), torch.ones(9, dtype=torch.float64)
        )
    torch.testing.assert_close(h, x + seen["a"] + seen["f"], rtol=1e-12, atol=1e-12)


def test_every_sandwich_gain_gets_a_gradient():
    layer = _block(_model("rmsnorm", gains=True), 0)
    x = torch.randn(8, 256, dtype=torch.float64)
    (
        layer(x, None, torch.arange(8)) * torch.randn(8, 256, dtype=torch.float64)
    ).sum().backward()
    for p in POST:
        g = getattr(layer, p).weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, p


@pytest.mark.parametrize(
    "post, gains", [("rmsnorm", True), ("rmsnorm", False), ("np_rmsnorm", True)]
)
def test_every_parameter_has_a_sharding_placement(post, gains):
    from torchtitan.models.opt_moe.sharding import set_opt_moe_sharding_config

    cfg = _cfg(post, gains)
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


@pytest.mark.parametrize("post", ["np_rmsnorm", "rmsnorm"])
def test_hf_export_refuses_a_sandwich_block(post):
    from torchtitan.models.opt_moe.hf_assests.setup_hf import hf_config

    with pytest.raises(ValueError, match="sandwich"):
        hf_config(None, _cfg(post))


def test_run4_gains_pattern_does_not_catch_the_sandwich_gains():
    # run-4's gains group matches `\.(q_norm|k_norm|attention_norm|ffn_norm)\.weight$`;
    # a config that wants the sandwich gains trained must name them itself.
    import re

    pattern = re.compile(r"\.(q_norm|k_norm|attention_norm|ffn_norm)\.weight$")
    names = [n for n, _ in _model("rmsnorm", gains=True).named_parameters()]
    caught = [n for n in names if pattern.search(n)]
    assert not any("post_" in n for n in caught)
    assert any(n.endswith(".attention_norm.weight") for n in caught)
