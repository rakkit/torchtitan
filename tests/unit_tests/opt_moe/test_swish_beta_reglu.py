# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe's two activation arms (ps2r U-1): a ReGLU gated FFN, and a learned per-layer Swish beta.

From the definitions: ReGLU(gate, up) = relu(gate) * up. The per-layer beta (one scalar per layer,
shape [1], init 1) multiplies the gate input: act(beta * gate, up), i.e. silu(beta z) up with z = w1 x
for SwiGLU, so it equals the same FFN with w1 pre-scaled by beta (the function class is unchanged)."""

import re
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.common.activation import ReGLU, SwiGLU
from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.models.opt_moe import model_registry
from torchtitan.optimizers.disco.optimizer import DiSCO

D, F_ = 256, 1024
BETA_RE = r"(^|\.)feed_forward\.swish_beta$"  # the group pattern of cell.py (act.beta): wrapped names carry _checkpoint_wrapped_module.  # noqa: B950


def _cfg(beta=False, reglu=False):
    cfg = model_registry("gauge-d8-w256")
    cfg.n_layers = 2
    cfg.layer.feed_forward.swish_beta = beta
    if reglu:
        cfg.layer.feed_forward.activation_fn = ReGLU.Config()
    cfg.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(),
            training=TrainingConfig(max_context_length=64),
        )
    )
    return cfg


def _model(beta=False, reglu=False, seed=0):
    cfg = _cfg(beta, reglu)
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    return model.double()


def test_off_adds_nothing_and_on_adds_one_scalar_per_layer_initialised_to_one():
    off = {n: p for n, p in _model().named_parameters()}
    on = {n: p for n, p in _model(beta=True).named_parameters()}
    extra = {n: p for n, p in on.items() if n not in off}
    assert set(extra) == {
        "layers.0.feed_forward.swish_beta",
        "layers.1.feed_forward.swish_beta",
    }
    assert all(p.shape == (1,) and float(p) == 1.0 for p in extra.values())
    assert all(
        torch.equal(on[n], p) for n, p in off.items()
    )  # every other weight bitwise equal (the beta takes no RNG)


def test_beta_at_one_is_an_exact_noop_and_gets_a_gradient():
    on, off = (
        _model(beta=True).layers["0"].feed_forward,
        _model().layers["0"].feed_forward,
    )
    x = torch.randn(9, D, dtype=torch.float64)
    y = on(x)
    assert torch.equal(y, off(x))
    y.sum().backward()
    g = on.swish_beta.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs()) > 0


def test_beta_equals_the_same_ffn_with_w1_prescaled():
    torch.manual_seed(1)
    on, off = (
        _model(beta=True).layers["0"].feed_forward,
        _model().layers["0"].feed_forward,
    )
    beta = 0.37
    with torch.no_grad():
        on.swish_beta.fill_(beta)
        off.w13.weight[0] *= beta
    x = torch.randn(10, D, dtype=torch.float64)
    torch.testing.assert_close(on(x), off(x), rtol=1e-12, atol=1e-12)


def test_beta_is_the_definition_silu_of_beta_z_times_up():
    ffn = _model(beta=True).layers["0"].feed_forward
    with torch.no_grad():
        ffn.swish_beta.fill_(2.5)
    x = torch.randn(7, D, dtype=torch.float64)
    z, u = x @ ffn.w13.weight[0].T, x @ ffn.w13.weight[1].T
    want = (F.silu(2.5 * z) * u) @ ffn.w2.weight.T
    torch.testing.assert_close(ffn(x), want, rtol=1e-12, atol=1e-12)


def test_reglu_is_the_definition_and_a_gate_scale_is_a_gauge():
    ffn = _model(reglu=True).layers["0"].feed_forward
    swi = _model().layers["0"].feed_forward
    x = torch.randn(7, D, dtype=torch.float64)
    z, u = x @ ffn.w13.weight[0].T, x @ ffn.w13.weight[1].T
    torch.testing.assert_close(
        ffn(x), (F.relu(z) * u) @ ffn.w2.weight.T, rtol=1e-12, atol=1e-12
    )
    assert (
        float((ffn(x) - swi(x)).abs().max()) > 1e-3
    )  # a different function from SwiGLU
    base = ffn(x)
    with torch.no_grad():
        ffn.w13.weight[0] *= 3.0
        ffn.w13.weight[1] /= 3.0
    torch.testing.assert_close(
        ffn(x), base, rtol=1e-12, atol=1e-12
    )  # relu(3 z) (u / 3) = relu(z) u
    with torch.no_grad():
        swi.w13.weight[0] *= 3.0
        swi.w13.weight[1] /= 3.0
    assert (
        float((swi(x) - _model().layers["0"].feed_forward(x)).abs().max()) > 1e-3
    )  # SwiGLU's gate scale is not a gauge


def test_the_activation_class_is_the_swiglu_class_unless_asked():
    assert isinstance(_cfg().layers[0].feed_forward.activation_fn, SwiGLU.Config)
    assert isinstance(
        _cfg(reglu=True).layers[0].feed_forward.activation_fn, ReGLU.Config
    )


def test_the_job_config_dump_tells_reglu_from_swiglu_and_shows_the_beta_flag():
    """A config with no field dumps as {}: ReGLU carries a tag so that job_config.json (and the duplicate check on it) can tell the two apart."""  # noqa: B950
    import dataclasses

    assert dataclasses.asdict(_cfg().layer.feed_forward.activation_fn) == {}
    assert dataclasses.asdict(_cfg(reglu=True).layer.feed_forward.activation_fn) == {
        "kind": "reglu"
    }
    assert (
        _cfg(beta=True).layer.feed_forward.swish_beta is True
        and _cfg().layer.feed_forward.swish_beta is False
    )


def test_every_parameter_has_a_sharding_placement_with_the_beta():
    """fully_shard under DP needs every parameter to be a DTensor; a direct parameter without a placement crashes at startup."""
    from torchtitan.models.opt_moe.sharding import set_opt_moe_sharding_config

    for beta, reglu in ((True, False), (True, True), (False, True), (False, False)):
        cfg = _cfg(beta, reglu)
        set_opt_moe_sharding_config(
            cfg, enable_sp=False, enable_tp=False, enable_ep=False
        )
        with torch.device("meta"):
            model = cfg.build()
        uncovered = [
            f"{mn}.{pn}"
            for mn, mod in model.named_modules()
            for pn, _ in mod.named_parameters(recurse=False)
            if getattr(mod, "_sharding_config", None) is None
            or pn not in mod._sharding_config.state_shardings
        ]
        assert not uncovered, (beta, reglu, uncovered)


def test_the_group_pattern_matches_the_betas_only_also_under_activation_checkpointing_names():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    model = _model(beta=True)
    for k in list(model.layers):
        model.layers[k] = checkpoint_wrapper(model.layers[k])
    names = [n for n, _ in model.named_parameters()]
    hit = [n for n in names if re.search(BETA_RE, n)]
    assert hit == [
        "layers.0._checkpoint_wrapped_module.feed_forward.swish_beta",
        "layers.1._checkpoint_wrapped_module.feed_forward.swish_beta",
    ]
    assert not any(re.search(BETA_RE, n) for n in names if "swish_beta" not in n)


# ---------------------------------------------------------------- in DiSCO: the beta in a sign group at lr_t
@pytest.fixture(scope="module")
def one_rank():
    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


def test_the_betas_take_the_gains_sign_step(one_rank):
    """The gains rule (DiSCO sgd, sign, momentum 0.1): the first step is beta - lr sign(g) on every [1]-shaped beta."""
    model = _model(beta=True).float()
    betas = {n: p for n, p in model.named_parameters() if n.endswith("swish_beta")}
    lr = 0.01
    group = {
        "params": list(betas.values()),
        "param_names": list(betas),
        "lr": lr,
        "momentum": 0.1,
        "nesterov": False,
        "weight_decay": 0.0,
        "decoupled_wd": False,
        "rule": "sgd",
        "args": {"pre": "identity", "compute": "sign", "scale": "one"},
    }
    layouts = {n: model.param_layouts().get(n, ParamLayout("vector")) for n in betas}
    opt = DiSCO(
        [group],
        parallel_dims=_NoParallel(),
        lr=lr,
        rule="sgd",
        args={"pre": "identity", "compute": "sign", "scale": "one"},
        param_layouts=layouts,
    )
    grads = {n: torch.tensor([g]) for n, g in zip(betas, (0.7, -0.2))}
    for n, p in betas.items():
        p.grad = grads[n].clone()
    opt.step()
    for n, p in betas.items():
        assert float(p) == pytest.approx(
            1.0 - lr * float(torch.sign(grads[n])), abs=1e-7
        ), n
