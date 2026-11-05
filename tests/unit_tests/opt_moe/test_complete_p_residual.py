# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CompleteP residual, alpha = 1, base depth K (arXiv 2505.01618 Table 1):
h = x + (K / L) * f(x). References are written from that equation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import model_registry
from torchtitan.models.opt_moe.hf_assests.setup_hf import hf_config
from torchtitan.models.opt_moe.utils.inits import setup_residual_scale


def _cfg(n_layers: int, residual_scale: str):
    cfg = model_registry("gauge-d8-w256")
    cfg.n_layers = n_layers
    cfg.layer.residual_scale = residual_scale
    cfg.finalize(parallelism=ParallelismConfig(), max_context_length=64)
    return cfg


def _model(n_layers: int, residual_scale: str, seed: int = 0):
    with torch.device("meta"):
        model = _cfg(n_layers, residual_scale).build()
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    return model.double()


class _CausalSDPA(torch.nn.Module):
    def forward(self, q, k, v, **kwargs):
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True
        )
        return out.transpose(0, 1)


def _block(model, i: int = 0):
    layer = model.layers[str(i)]
    layer.attention.inner_attention = _CausalSDPA()
    return layer


@pytest.mark.parametrize("n_layers", [8, 16, 32, 64])
def test_block_scale_is_base_over_depth(n_layers):
    assert setup_residual_scale("complete_p_base16", n_layers) == (16 / n_layers, 1.0)


def test_base_depth_gives_exact_unit_scales():
    block, identity = setup_residual_scale("complete_p_base16", 16)
    assert block == 1.0 and identity == 1.0


@pytest.mark.parametrize(
    "bad",
    [
        "complete_p_base",
        "complete_p_base0",
        "complete_p_base-4",
        "complete_p_base1.5",
        "complete_p_base016",
        "complete_p_basex",
        "complete_p_base16 ",
    ],
)
def test_malformed_base_is_refused(bad):
    with pytest.raises(ValueError):
        setup_residual_scale(bad, 16)


def test_default_stays_identity():
    assert model_registry("gauge-d16-w256").layer.residual_scale == "identity"
    assert setup_residual_scale("identity", 64) == (1.0, 1.0)


def test_at_base_depth_the_model_is_plain_bitwise():
    plain, cp = _model(2, "identity"), _model(2, "complete_p_base2")
    for (n, p), (m, q) in zip(plain.named_parameters(), cp.named_parameters()):
        assert n == m and torch.equal(p, q), n
    x, pos = torch.randn(10, 256, dtype=torch.float64), torch.arange(10)
    for i in (0, 1):
        a, b = _block(plain, i), _block(cp, i)
        assert b._skip_block_mul and b._skip_identity_mul
        assert torch.equal(a(x, None, pos), b(x, None, pos))


@pytest.mark.parametrize("n_layers, base", [(2, 4), (4, 2)])
def test_block_equals_its_equation_off_base(n_layers, base):
    layer = _block(_model(n_layers, f"complete_p_base{base}"), 0)
    assert layer._skip_identity_mul and not layer._skip_block_mul
    s = base / n_layers
    x, pos = torch.randn(12, 256, dtype=torch.float64), torch.arange(12)
    h = x + s * layer.attention(layer.attention_norm(x), None, pos)
    ref = h + s * layer.feed_forward(layer.ffn_norm(h))
    torch.testing.assert_close(layer(x, None, pos), ref, rtol=1e-12, atol=1e-12)


def test_hf_export_carries_the_scale():
    c = hf_config(SimpleNamespace(config=_cfg(4, "complete_p_base16")))
    assert (c.residual_block_scale, c.residual_identity_scale) == (4.0, 1.0)
