# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe's HF format: every native tensor has an HF name and comes back
unchanged (every registered flavor, and a model with every feature: conv,
sinks, gate, dense-first layers, MTP), and the HF config of a model config
loads as ``OptMoeConfig``."""

import json

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config, MODEL_FLAVORS
from torchtitan.models.opt_moe.hf.configuration_opt_moe import OptMoeConfig
from torchtitan.models.opt_moe.state_dict_adapter import (
    hf_config,
    OPTMoEStateDictAdapter,
)

_ALL_FEATURES = dict(
    n_layers=3,
    swa_pattern="SSF",
    rope_pattern="RRN",
    n_dense_layers=1,
    dense_hidden_dim=256,
    conv_kernel_size=4,
    gated_attention_type="element-wise",
    num_mtp_layers=1,
    vocab_size=512,
)
# sinks in place of the gate (one or neither)
_ALL_FEATURES_SINKS = _ALL_FEATURES | dict(
    gated_attention_type=None, attention_sinks=True
)


def _config(flavor, **overrides):
    return build_model_config(flavor, parallelism=ParallelismConfig(), **overrides)


@pytest.mark.parametrize(
    "flavor, overrides",
    [(f, {}) for f in MODEL_FLAVORS]
    + [("mis-8b-proxy", _ALL_FEATURES), ("mis-8b-proxy", _ALL_FEATURES_SINKS)],
)
def test_every_tensor_maps_to_hf_and_back(flavor, overrides):
    cfg = _config(flavor, **overrides)
    with torch.device("meta"):
        model = cfg.build()
    native = model.state_dict()
    adapter = OPTMoEStateDictAdapter(cfg, None)
    hf = adapter.to_hf(native)
    assert all(k.startswith(("model.", "lm_head.")) for k in hf)
    back = adapter.from_hf(hf)
    assert back.keys() == native.keys()
    assert all(back[k].shape == native[k].shape for k in native)
    if cfg.mtp_layers:
        n = len(cfg.layers)
        assert f"model.layers.{n}.eh_proj.weight" in hf
        assert f"model.layers.{n}.shared_head.norm.weight" in hf
    config = OptMoeConfig(**hf_config(cfg))
    assert json.loads(config.to_json_string())["model_type"] == "opt_moe"


@pytest.mark.parametrize("features", [_ALL_FEATURES, _ALL_FEATURES_SINKS])
def test_values_come_back_unchanged(features):
    cfg = _config("mis-8b-proxy", **features)
    model = cfg.build()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p))
    native = {k: v.clone() for k, v in model.state_dict().items()}
    adapter = OPTMoEStateDictAdapter(cfg, None)
    back = adapter.from_hf(adapter.to_hf(native))
    for key, value in native.items():
        assert torch.equal(back[key], value), key


def test_hf_config_states_the_architecture():
    cfg = _config("mis-8b-proxy", **_ALL_FEATURES)
    c = OptMoeConfig(**hf_config(cfg))
    # with the conv: transformers' hybrid cache layers (K / V + the k / v conv states)
    assert c.layer_types == ["hybrid_sliding", "hybrid_sliding", "hybrid"]
    assert c.number_of_conv_states == 2
    assert c.rope_layers[2] is None and c.rope_layers[0]["dim"] == 128
    assert (c.first_k_dense_replace, c.intermediate_size) == (1, 256)
    assert c.conv_kernel_size == 4
    assert not c.attention_sinks and c.attention_gate == "element-wise"
    assert c.num_nextn_predict_layers == 1 and c.n_routed_experts == 64


def test_the_gate_and_sinks_together_are_refused():
    """One or neither: the HF and vLLM models are checked for each alone."""
    both = _ALL_FEATURES | dict(attention_sinks=True)
    with pytest.raises(ValueError, match="one or neither"):
        _config("mis-8b-proxy", **both)
    config = hf_config(_config("mis-8b-proxy", **_ALL_FEATURES))
    with pytest.raises(ValueError, match="one or neither"):
        OptMoeConfig(**config | {"attention_sinks": True})
