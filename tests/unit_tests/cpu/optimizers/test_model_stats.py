# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Residual-stream and RMSNorm-gain statistics (log_residual_stats, log_gain_stats).

The residual statistics are captured in the eager layer loop of
``OPTMoEModel._trunk`` on armed steps only; the blocks are untouched. References
are recomputed from the tensors themselves through test-only hooks, not from
the code under test.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.utils.checkpoint
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.nn.attention.flex_attention import flex_attention
from torchtitan.config.parallelism import ParallelismConfig

from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.opt_moe import build_model_config
from torchtitan.optimizers.container import DiSCOOptimizersContainer


@pytest.fixture(autouse=True)
def eager_flex(monkeypatch):
    # The compiled flex kernel has no CPU lowering; eager flex is exact (forward only).
    monkeypatch.setattr(FlexInnerAttention, "_compiled_flex_attn", flex_attention)


def _causal_sdpa(
    q, k, v, *, score_mod, block_mask, scale, enable_gqa, return_aux, kernel_options
):
    # One document per sequence: block-causal is causal, and SDPA has a CPU backward.
    assert score_mod is None and not return_aux.lse
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=scale, enable_gqa=enable_gqa
    )
    return out, SimpleNamespace(lse=None, max_scores=None)


def _tiny(*, gains: bool = True, seed: int = 0, n_layers: int = 2, sinks: bool = False):
    cfg = build_model_config(
        "gauge-d8-w256",
        parallelism=ParallelismConfig(),
        n_layers=n_layers,
        vocab_size=512,
        attention_sinks=sinks,
        # learnable gains (rmsnorm, the default) or none (the parameter-free np_rmsnorm)
        norm_type="rmsnorm" if gains else "np_rmsnorm",
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    return model


def _batch(seed: int = 1, docs=(60, 36)):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 512, (sum(docs),), generator=g)
    positions = torch.cat([torch.arange(n) for n in docs])
    return tokens, positions


def _forward(model, tokens, positions):
    return model(tokens, positions, model._get_attention_metadata(positions))


def _direct(model, tokens, positions):
    """{key: mean square} from hooks on every block's input and output."""
    seen: dict[str, float] = {}
    handles = []
    for i, block in model.layers.items():

        def pre(mod, args, kwargs, i=i):
            seen[f"in/{i}"] = args[0].detach().double()

        def post(mod, args, kwargs, out, i=i):
            seen[f"out/{i}"] = out.detach().double()

        handles += [
            block.register_forward_pre_hook(pre, with_kwargs=True),
            block.register_forward_hook(post, with_kwargs=True),
        ]
    try:
        with torch.no_grad():
            _forward(model, tokens, positions)
    finally:
        for h in handles:
            h.remove()
    last = max(model.layers.keys(), key=int)
    out = {}
    for i in model.layers.keys():
        x, y = seen[f"in/{i}"], seen[f"out/{i}"]
        out[f"residual_rms/in/L-{i}"] = float(x.square().mean())
        out[f"residual_rms/write/L-{i}"] = float((y - x).square().mean())
    out["residual_rms/final_norm_in"] = float(seen[f"out/{last}"].square().mean())
    return out


def test_capture_matches_direct_recomputation():
    """Armed: RMS of every block's input, of its total write (out - x) and of the
    final norm's input; the read disarms the capture."""
    from torchtitan.optimizers import model_stats

    model = _tiny()
    tokens, positions = _batch()
    want = _direct(model, tokens, positions)
    model_stats.arm_residual_capture([model])
    with torch.no_grad():
        _forward(model, tokens, positions)
    got = model_stats.residual_metrics([model], parallelism_context=None)
    assert set(got) == set(want)
    for k, ms in want.items():
        assert got[k] == pytest.approx(math.sqrt(ms), rel=1e-6), k
    assert model.residual_capture is None


def test_unarmed_by_default_and_the_forward_is_unchanged():
    """Unarmed is the default and adds nothing to the state dict; an armed
    forward returns bitwise the same output as an unarmed one."""
    from torchtitan.optimizers import model_stats

    model = _tiny()
    assert model.residual_capture is None
    keys = list(model.state_dict())
    tokens, positions = _batch()
    with torch.no_grad():
        off = _forward(model, tokens, positions)
        model_stats.arm_residual_capture([model])
        on = _forward(model, tokens, positions)
    assert len(model.residual_capture) == 1
    assert torch.equal(off, on)
    assert list(model.state_dict()) == keys


def test_capture_under_accumulation_and_ac(monkeypatch):
    """Two microbatches average; AC recomputes blocks, never the loop, so each
    forward adds exactly one entry, with the values of the run without AC."""
    from torchtitan.optimizers import model_stats

    monkeypatch.setattr(FlexInnerAttention, "_compiled_flex_attn", _causal_sdpa)
    model = _tiny()
    a, b = _batch(1, docs=(96,)), _batch(2, docs=(96,))
    da, db = _direct(model, *a), _direct(model, *b)
    model_stats.arm_residual_capture([model])
    with torch.no_grad():
        _forward(model, *a)
        _forward(model, *b)
    assert len(model.residual_capture) == 2
    got = model_stats.residual_metrics([model], parallelism_context=None)
    for k in da:
        assert got[k] == pytest.approx(math.sqrt((da[k] + db[k]) / 2), rel=1e-6), k

    model_stats.arm_residual_capture([model])
    _forward(model, *a).sum().backward()
    no_ac = model_stats.residual_metrics([model], parallelism_context=None)
    for name in list(model.layers.keys()):
        model.layers.register_module(name, checkpoint_wrapper(model.layers[name]))
    for early_stop in (True, False):
        model.zero_grad(set_to_none=True)
        model_stats.arm_residual_capture([model])
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            _forward(model, *a).sum().backward()
        assert len(model.residual_capture) == 1
        with_ac = model_stats.residual_metrics([model], parallelism_context=None)
        assert with_ac.keys() == no_ac.keys()
        for k in no_ac:
            assert with_ac[k] == pytest.approx(no_ac[k], rel=1e-7), (early_stop, k)


def test_arming_needs_a_model_that_captures():
    from torchtitan.optimizers import model_stats

    with pytest.raises(ValueError):
        model_stats.arm_residual_capture([torch.nn.Linear(2, 2)])


def test_gain_metrics_match_direct_formulas():
    """Mean and RMS of each RMSNorm gain after the step, and ||g - g_before|| /
    ||g_before||; names are module names, free of wrapper prefixes."""
    from torchtitan.optimizers import model_stats

    model = _tiny(gains=True)
    model.layers.register_module("0", checkpoint_wrapper(model.layers["0"]))
    before = model_stats.snapshot_gains([model])
    norms = {
        n.replace("._checkpoint_wrapped_module", ""): m.weight
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.RMSNorm) and m.weight is not None
    }
    assert len(before) == len(norms) == 2 * 4 + 1
    ref_before = {n: w.detach().clone().double() for n, w in norms.items()}
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for w in norms.values():
            w.add_(0.01 * torch.randn(w.shape, generator=g))
    got = model_stats.gain_metrics([model], before, parallelism_context=None)
    assert set(got) == {
        f"gain_{s}/{n}" for n in norms for s in ("mean", "rms", "rel_update")
    }
    for n, w in norms.items():
        a, b = w.detach().double(), ref_before[n]
        assert got[f"gain_mean/{n}"] == pytest.approx(float(a.mean()), rel=1e-6)
        assert got[f"gain_rms/{n}"] == pytest.approx(
            float(a.square().mean().sqrt()), rel=1e-6
        )
        assert got[f"gain_rel_update/{n}"] == pytest.approx(
            float((a - b).norm() / b.norm()), rel=1e-5
        )

    off = _tiny(gains=False)
    assert model_stats.snapshot_gains([off]) == []
    assert model_stats.gain_metrics([off], [], parallelism_context=None) == {}


def _container(
    model,
    *,
    log_residual_stats: bool,
    log_gain_stats: bool,
    log_sink_stats: bool = False,
):
    c = MagicMock(spec=DiSCOOptimizersContainer)
    c.model_parts = [model]
    c.parallelism_context = None
    c.log_residual_stats = log_residual_stats
    c.log_gain_stats = log_gain_stats
    c.log_sink_stats = log_sink_stats
    c.log_norm_freq = 10
    c.optimizers = []
    c._norms_armed = False
    c._model_stats_armed = False
    c._gains_before = None
    c.gnoise = None  # a container built with gnoise off
    c.collect_moe_metrics.return_value = {}
    for name in (
        "begin_step",
        "wants_norms_at",
        "_arm_model_stats",
        "collect_metrics",
        "_collect_model_stats",
    ):
        setattr(c, name, getattr(DiSCOOptimizersContainer, name).__get__(c))
    return c


def test_container_arms_on_the_per_unit_cadence_and_reads_once():
    model = _tiny(gains=True)
    c = _container(model, log_residual_stats=True, log_gain_stats=True)
    tokens, positions = _batch()

    c.begin_step(1, should_log=True, is_final_step=False)
    assert model.residual_capture == []
    with torch.no_grad():
        _forward(model, tokens, positions)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.RMSNorm) and m.weight is not None:
                m.weight.mul_(1.01)
    got = c.collect_metrics(1)
    assert sum(k.startswith("residual_rms/") for k in got) == 2 * 2 + 1
    assert sum(k.startswith("gain_") for k in got) == 3 * 9
    assert all(
        v == pytest.approx(0.01, rel=1e-4)
        for k, v in got.items()
        if k.startswith("gain_rel_update/")
    )
    assert model.residual_capture is None
    again = c.collect_metrics(1)  # one read per armed step
    assert not any(k.startswith(("residual_rms/", "gain_")) for k in again)

    c.begin_step(505, should_log=True, is_final_step=False)  # past dense, off cadence
    assert model.residual_capture is None
    with torch.no_grad():
        _forward(model, tokens, positions)
    assert not any(
        k.startswith(("residual_rms/", "gain_")) for k in c.collect_metrics(505)
    )
    c.begin_step(510, should_log=True, is_final_step=False)
    with torch.no_grad():
        _forward(model, tokens, positions)
    assert any(k.startswith("residual_rms/") for k in c.collect_metrics(510))


def test_container_without_the_flags_reports_nothing():
    """Defaults: never armed, no gains snapshot, no keys -- nothing to reduce."""
    model = _tiny(gains=True)
    c = _container(model, log_residual_stats=False, log_gain_stats=False)
    c.begin_step(1, should_log=True, is_final_step=False)
    assert model.residual_capture is None and c._gains_before is None
    with torch.no_grad():
        _forward(model, *_batch())
    got = c.collect_metrics(1)
    assert not any(k.startswith(("residual_rms/", "gain_")) for k in got)


def test_container_logs_every_sink_value_on_the_cadence_and_only_when_asked():
    model = _tiny(gains=False, sinks=True)
    n_heads = model.layers["0"].attention.n_heads
    with torch.no_grad():
        for i in range(2):
            model.layers[str(i)].attention.sinks.copy_(
                torch.arange(n_heads, dtype=torch.float32) / 8 + i
            )
    c = _container(
        model, log_residual_stats=False, log_gain_stats=False, log_sink_stats=True
    )
    c.begin_step(1, should_log=True, is_final_step=False)
    got = c.collect_metrics(1)
    sink = {k: v for k, v in got.items() if k.startswith("sink_value/")}
    assert sink == {
        f"sink_value/layers.{i}.attention/head-{h}": h / 8 + i
        for i in range(2)
        for h in range(n_heads)
    }
    assert not any(
        k.startswith("sink_value/") for k in c.collect_metrics(1)
    )  # one read per armed step
    c.begin_step(505, should_log=True, is_final_step=False)  # past dense, off cadence
    assert not any(k.startswith("sink_value/") for k in c.collect_metrics(505))

    off = _container(model, log_residual_stats=False, log_gain_stats=False)
    off.begin_step(1, should_log=True, is_final_step=False)
    assert not any(k.startswith("sink_value/") for k in off.collect_metrics(1))

    plain = _container(
        _tiny(gains=False),
        log_residual_stats=False,
        log_gain_stats=False,
        log_sink_stats=True,
    )
    plain.begin_step(1, should_log=True, is_final_step=False)
    with pytest.raises(ValueError, match="attention sinks"):
        plain.collect_metrics(1)
