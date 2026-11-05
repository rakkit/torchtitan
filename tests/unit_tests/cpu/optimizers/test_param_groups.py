# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Param groups: what a group inherits, how `args` merge, what is checkpointed."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
from torchtitan.models.common.param_layout import ParamLayout

from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.optimizers.disco.presets import FrobeniusAUS
from torchtitan.optimizers.disco.rules import rule_of


@pytest.fixture(autouse=True, scope="module")
def _process_group():
    """One gloo rank for this module; destroyed after it if this module made it
    (a group left behind breaks other modules that create their own)."""
    created = not dist.is_initialized()
    if created:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    if created:
        dist.destroy_process_group()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


def _groups():
    w = [torch.nn.Parameter(torch.randn(8, 4)) for _ in range(3)]
    return [
        {"params": w[:2], "param_names": ["a.weight", "b.weight"]},
        {
            "params": w[2:],
            "param_names": ["c.weight"],
            "lr": 0.5,
            "rule": "hyperball",
            "args": {"compute": "svd", "tangent": True},
        },
    ]


_LAYOUTS = {n: ParamLayout("matrix") for n in ("a.weight", "b.weight", "c.weight")}


def _opt(**kw):
    kw.setdefault("args", {"steps": 3})
    kw.setdefault("param_layouts", _LAYOUTS)
    return DiSCO(_groups(), parallelism_context=_NoParallel(), lr=0.1, **kw)


def test_every_parameter_must_be_declared():
    with pytest.raises(ValueError, match="not declared"):
        _opt(param_layouts={"a.weight": ParamLayout("matrix")})


def test_a_vocabulary_matrix_refuses_a_gathering_rule():
    """A group that misses the embedding by name used to send it to
    Newton-Schulz on one rank without a word."""
    layouts = {**_LAYOUTS, "a.weight": ParamLayout("embedding")}
    with pytest.raises(ValueError, match="embedding"):
        _opt(param_layouts=layouts, rule="muon")


def test_a_gram_compute_refuses_a_wide_unit():
    """A wide unit's [n, n] Gram is rank-deficient: in fp16 the Gram iteration
    returned NaN at every scale."""
    w = torch.nn.Parameter(torch.randn(4, 8))
    with pytest.raises(ValueError, match="tall"):
        DiSCO(
            [{"params": [w], "param_names": ["w.weight"]}],
            parallelism_context=_NoParallel(),
            lr=0.1,
            rule="muon",
            args={"compute": "gram_polar", "steps": 5},
            param_layouts={"w.weight": ParamLayout("matrix")},
        )


def test_groups_inherit_the_defaults_and_merge_args():
    opt = _opt()
    g0, g1 = opt.param_groups
    assert (g0["lr"], g0["momentum"], g0["rule"]) == (0.1, 0.95, "muon")
    assert g0["args"] == {"steps": 3}
    assert (g1["lr"], g1["rule"]) == (0.5, "hyperball")
    assert g1["args"] == {"steps": 3, "compute": "svd", "tangent": True}
    rules = {e.name: e.rule for e in opt.entries}
    assert rules["a.weight"] == rule_of({"rule": "muon", "args": {"steps": 3}})
    hb = rules["c.weight"].compute
    assert isinstance(hb, FrobeniusAUS) and hb.tangent and hb.inner.compute.steps == 3


def test_rule_config_is_not_checkpoint_state():
    sd = _opt().state_dict()
    for g in sd["param_groups"]:
        assert "rule" not in g and "args" not in g
        assert g["t"] == 0
    with pytest.raises(ValueError):
        _opt(rule="nope")
    with pytest.raises(ValueError):
        _opt(args={"backend_steps": 3})  # old spelling: an error, not silence


def test_a_container_checkpoint_loads_back():
    """The rule config stays out of checkpoints, but upstream's flat loader
    wants every live group key: resuming a DiSCO run raised a KeyError."""
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

    cfg = DiSCOOptimizersContainer.Config(
        groups={"default": {"lr": 0.1, "rule": "muon", "args": {"compute": "svd"}}}
    )

    def build():
        model = torch.nn.Sequential(torch.nn.Linear(4, 8, bias=False))
        model.param_layouts = lambda: {"0.weight": ParamLayout("matrix")}
        container = DiSCOOptimizersContainer(
            cfg, model_parts=[model], parallelism_context=_NoParallel()
        )
        return model, container

    model, container = build()
    model[0].weight.grad = torch.ones(8, 4)
    container.step()
    sd = container.state_dict()
    assert not any(k.endswith((".rule", ".args")) for k in sd)
    saved = container.optimizers[0].state[model[0].weight]["momentum_buffer"].clone()
    # A fresh container, as a resumed job has: nothing in it yet but zeros.
    model, container = build()
    container.load_state_dict(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in sd.items()}
    )
    (opt,) = container.optimizers
    assert saved.abs().sum() > 0
    assert torch.equal(opt.state[model[0].weight]["momentum_buffer"], saved)
    assert opt.param_groups[0]["args"] == {"compute": "svd"}


def test_group_keys_are_cli_flags_and_groups_become_upstream_configs():
    import tyro

    from torchtitan.optimizers.container import DiSCOOptimizersContainer

    cfg = DiSCOOptimizersContainer.Config()
    cfg.groups = {
        "default": {"lr": 0.1, "rule": "muon", "args": {"steps": 5}},
        "embedding": {
            "match": r"tok_embeddings",
            "lr": 0.2,
            "args": {"compute": "row-l2", "scale": "sqrt_n"},
        },
        "head": {"match": r"lm_head", "optimizer": "AdamW", "lr": 1e-3},
    }
    parsed = tyro.cli(
        DiSCOOptimizersContainer.Config,
        args=[
            "--groups.default.lr",
            "0.05",
            "--groups.embedding.args.compute",
            "svd",
            "--groups.default.args.steps",
            "8",
        ],
        default=cfg,
    )
    assert parsed.groups["default"]["lr"] == 0.05
    assert parsed.groups["default"]["args"]["steps"] == 8
    assert parsed.groups["embedding"]["args"]["compute"] == "svd"
    pgs = DiSCOOptimizersContainer._param_group_configs(parsed)
    assert [(p.pattern, p.optimizer_name) for p in pgs] == [
        ("tok_embeddings", "DiSCO"),
        ("lm_head", "AdamW"),
        (".*", "DiSCO"),
    ]
    assert pgs[0].optimizer_kwargs == {
        "lr": 0.2,
        "args": {"compute": "svd", "scale": "sqrt_n"},
    }
    assert pgs[1].optimizer_kwargs == {"lr": 1e-3}
    with pytest.raises(ValueError):
        DiSCOOptimizersContainer._param_group_configs(DiSCOOptimizersContainer.Config())


def test_groups_select_by_declared_kind():
    """`kind` resolves to exactly the parameters the model declares as that
    kind, per model part, so recipes need no FQN regex for the embedding, the
    head or the routers."""
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(16, 4)
            self.proj = torch.nn.Linear(4, 4, bias=False)
            self.head = torch.nn.Linear(4, 16, bias=False)

        def param_layouts(self):
            return {
                "emb.weight": ParamLayout("embedding"),
                "proj.weight": ParamLayout("matrix"),
                "head.weight": ParamLayout("unembedding"),
            }

    cfg = DiSCOOptimizersContainer.Config()
    cfg.groups = {
        "vocab_in": {"kind": "embedding", "args": {"scale": "sqrt_n"}},
        "vocab_out": {"kind": "unembedding", "args": {"scale": "inv_sqrt_n"}},
        "default": {"lr": 0.1, "rule": "muon", "args": {"compute": "svd"}},
    }
    cfg.groups["vocab_in"]["args"]["compute"] = "row-l2"
    cfg.groups["vocab_out"]["args"]["compute"] = "row-l2"
    container = DiSCOOptimizersContainer(
        cfg, model_parts=[Toy()], parallelism_context=_NoParallel()
    )
    (opt,) = container.optimizers
    got = {e.name: (e.kind, e.rule.kind) for e in opt.entries}
    assert got == {
        "emb.weight": ("embedding", "rows"),
        "head.weight": ("unembedding", "rows"),
        "proj.weight": ("matrix", "unit"),
    }
    with pytest.raises(ValueError, match="exactly one of"):
        DiSCOOptimizersContainer._param_group_configs(
            DiSCOOptimizersContainer.Config(
                groups={"default": {"lr": 0.1}, "x": {"kind": "router", "match": "r"}}
            )
        )
    cfg.groups = {"r": {"kind": "router"}, "default": {"lr": 0.1}}
    with pytest.raises(ValueError, match="router"):  # nothing is declared a router
        DiSCOOptimizersContainer(cfg, model_parts=[Toy()], parallelism_context=_NoParallel())


def test_the_transport_is_float32_by_default_and_float16_is_refused():
    """float16 flushes real momentum entries (~1e-7): on step-500 momenta the
    update was off by 29% (median) against 1.4% in float32."""
    from torchtitan.optimizers.container import DiSCOOptimizersContainer
    from torchtitan.optimizers.utils import disco_factory_kwargs

    assert DiSCOOptimizersContainer.Config().communication_dtype == "float32"
    kw = disco_factory_kwargs(DiSCOOptimizersContainer.Config(), _NoParallel())
    assert kw["communication_dtype"] == torch.float32
    cfg = DiSCOOptimizersContainer.Config(communication_dtype="float16")
    with pytest.raises(ValueError, match="float16"):
        disco_factory_kwargs(cfg, _NoParallel())


def _two_groups(second: dict, **kw):
    w, v = (torch.nn.Parameter(torch.randn(16, 4)) for _ in range(2))
    layouts = {n: ParamLayout("matrix") for n in ("w.weight", "h.weight")}
    return DiSCO(
        [
            {"params": [w], "param_names": ["w.weight"]},
            {"params": [v], "param_names": ["h.weight"], **second},
        ],
        parallelism_context=_NoParallel(),
        lr=0.1,
        param_layouts=layouts,
        **kw,
    )


def test_a_group_without_args_takes_only_what_its_rule_inherits():
    """torch filled a missing `args` with the default dict itself, so a group
    of another rule written without args ran with the default group's args
    (a head took muon's `scale`, 448x too large on the real head)."""
    default = {"compute": "svd", "steps": 3, "scale": "spectral"}
    opt = _two_groups({"rule": "hyperball"}, rule="muon", args=default)
    assert opt.param_groups[0]["args"] == default  # same rule: every default
    # hyperball has no `scale`: it takes only the args it has
    assert opt.param_groups[1]["args"] == {"compute": "svd", "steps": 3}
    hb = next(e.rule for e in opt.entries if e.name == "h.weight").compute
    assert isinstance(hb, FrobeniusAUS) and hb.inner.compute.steps == 3


def test_a_rule_setting_outside_args_is_an_error():
    """`scale` next to `rule` instead of inside `args` used to be ignored:
    the group ran the default rule."""
    with pytest.raises(ValueError, match="args"):
        _two_groups({"compute": "row-l2", "scale": "sqrt_n"})


def test_a_fused_parameter_logs_the_radial_metrics_of_its_split_twin():
    """The units of a fused w13 shared one radial state, so each unit's
    accumulators also summed the other unit's steps."""
    from torchtitan.models.common.param_layout import stacked_units

    torch.manual_seed(0)
    a, b = torch.randn(8, 6), torch.randn(8, 6)
    fused = torch.nn.Parameter(torch.stack([a, b]))
    split = [torch.nn.Parameter(a.clone()), torch.nn.Parameter(b.clone())]
    kw = dict(
        parallelism_context=_NoParallel(),
        lr=0.1,
        momentum=0.9,
        args={"compute": "svd", "steps": 0, "scale": "one"},
    )
    opts = [
        DiSCO(
            [{"params": [fused], "param_names": ["m.w13.weight"]}],
            param_layouts={
                "m.w13.weight": ParamLayout(
                    "matrix", units=stacked_units(("w1", "w3"), 8, 6)
                )
            },
            **kw,
        ),
        DiSCO(
            [{"params": split, "param_names": ["m.w1.weight", "m.w3.weight"]}],
            param_layouts={
                "m.w1.weight": ParamLayout("matrix"),
                "m.w3.weight": ParamLayout("matrix"),
            },
            **kw,
        ),
    ]
    for _ in range(3):
        g = torch.randn(16, 6)
        fused.grad = g.view(2, 8, 6).clone()
        split[0].grad, split[1].grad = g[:8].clone(), g[8:].clone()
        logged = []
        for opt in opts:
            opt.calculate_norm_at_next_step(
                norms_to_log=["frobenius_norm"], gram_level=0, track_spectrum=False
            )
            opt.step()
            logged.append(
                {
                    k: v
                    for k, v in opt.get_norms_at_current_step().items()
                    if k.startswith("track_radial_")
                }
            )
        assert logged[0] and logged[0].keys() == logged[1].keys()
        for k in logged[0]:
            assert logged[0][k] == pytest.approx(logged[1][k], rel=1e-5, abs=1e-7), k


def test_momentum_and_nesterov_come_from_the_config():
    """A resume runs the config's `momentum` and `nesterov` (as `weight_decay`):
    the checkpoint does not carry them, a value it still carries is replaced,
    and the momentum buffer itself is restored."""
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

    def build(beta, nesterov):
        cfg = DiSCOOptimizersContainer.Config(
            groups={
                "default": {
                    "lr": 0.1,
                    "momentum": beta,
                    "nesterov": nesterov,
                    "rule": "muon",
                    "args": {"compute": "svd"},
                }
            }
        )
        model = torch.nn.Sequential(torch.nn.Linear(4, 8, bias=False))
        model.param_layouts = lambda: {"0.weight": ParamLayout("matrix")}
        return model, DiSCOOptimizersContainer(
            cfg, model_parts=[model], parallelism_context=_NoParallel()
        )

    model, old = build(0.9, False)
    model[0].weight.grad = torch.ones_like(model[0].weight)
    old.step()
    sd = old.state_dict()
    assert not any(k.endswith((".momentum", ".nesterov")) for k in sd)
    fqn = old.optimizers[0].param_groups[0]["param_names"][0]
    sd[f"param_groups.{fqn}.momentum"] = 0.5  # a value a checkpoint still carries
    saved = old.optimizers[0].state[model[0].weight]["momentum_buffer"].clone()
    model2, new = build(0.95, True)
    new.load_state_dict(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in sd.items()}
    )
    g = new.optimizers[0].param_groups[0]
    assert (g["momentum"], g["nesterov"]) == (0.95, True)
    assert torch.equal(
        new.optimizers[0].state[model2[0].weight]["momentum_buffer"], saved
    )


def test_weight_decay_comes_from_the_config():
    """`weight_decay` is config-only: a resume never takes the checkpoint's.
    `decoupled_wd` is gone: a group that sets it is refused."""
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

    def build(wd, **extra):
        cfg = DiSCOOptimizersContainer.Config(
            groups={
                "default": {
                    "lr": 0.1,
                    "weight_decay": wd,
                    "rule": "muon",
                    "args": {"compute": "svd"},
                    **extra,
                }
            }
        )
        model = torch.nn.Sequential(torch.nn.Linear(4, 8, bias=False))
        model.param_layouts = lambda: {"0.weight": ParamLayout("matrix")}
        return DiSCOOptimizersContainer(
            cfg, model_parts=[model], parallelism_context=_NoParallel()
        )

    old = build(0.1)
    sd = old.state_dict()
    assert not any(k.endswith(".weight_decay") for k in sd)
    # An older checkpoint still carries weight_decay: the config wins.
    fqn = old.optimizers[0].param_groups[0]["param_names"][0]
    sd[f"param_groups.{fqn}.weight_decay"] = 0.1
    new = build(1e-4)
    new.load_state_dict(sd)
    g = new.optimizers[0].param_groups[0]
    assert g["weight_decay"] == 1e-4 and "decoupled_wd" not in g
    with pytest.raises(TypeError, match="decoupled_wd"):  # a DiSCO default kwarg
        build(0.1, decoupled_wd=True)


def test_another_rule_that_inherits_a_polar_express_compute_takes_its_safety():
    """A bare `{"rule": "muon"}` group under an amphione default inherits the
    default's gram_polar_express and must run it at the default's safety (it
    ran at the backend's 1.01)."""
    names = ["a.weight", "b.weight"]
    ws = [torch.nn.Parameter(torch.randn(8, 8)) for _ in names]
    opt = DiSCO(
        [
            {"params": [ws[0]], "param_names": [names[0]]},
            {"params": [ws[1]], "param_names": [names[1]], "rule": "muon", "args": {}},
        ],
        parallelism_context=_NoParallel(),
        lr=0.1,
        momentum=0.95,
        nesterov=True,
        rule="amphione",
        args={
            "compute": "gram_polar_express",
            "steps": 6,
            "polar_express_safety": 1.05,
        },
        param_layouts={n: ParamLayout("matrix") for n in names},
    )
    args = opt.param_groups[1]["args"]
    assert args["compute"] == "gram_polar_express"
    assert args["polar_express_safety"] == 1.05
    assert "1.05" in repr({e.name: e.rule for e in opt.entries}["b.weight"])


def test_the_safety_is_inherited_only_where_it_applies():
    """Under an amphione default with Gram Polar Express at safety 1.05, the
    conv group (amphione, row-l2 direction) and an sgd row group inherit no
    safety (their computes have no polar express iteration) and build without
    stating `polar_express_safety: None`; a group that sets it itself on such
    a compute is refused."""
    names = ["a.weight", "b.weight", "c.weight"]
    ws = [torch.nn.Parameter(torch.randn(8, 8)) for _ in names]

    def build(conv_args):
        return DiSCO(
            [
                {"params": [ws[0]], "param_names": [names[0]]},
                {
                    "params": [ws[1]],
                    "param_names": [names[1]],
                    "rule": "amphione",
                    "args": conv_args,
                },
                {
                    "params": [ws[2]],
                    "param_names": [names[2]],
                    "rule": "sgd",
                    "args": {"compute": "row-l2"},
                },
            ],
            parallelism_context=_NoParallel(),
            lr=0.1,
            momentum=0.95,
            nesterov=True,
            rule="amphione",
            args={
                "compute": "gram_polar_express",
                "steps": 6,
                "polar_express_safety": 1.05,
            },
            param_layouts={n: ParamLayout("matrix") for n in names},
        )

    opt = build({"compute": "row-l2"})
    rules = {e.name: e.rule for e in opt.entries}
    assert rules["a.weight"].compute.direction.compute.polar_express_safety == 1.05
    assert "polar_express_safety" not in opt.param_groups[1]["args"]
    assert "polar_express_safety" not in opt.param_groups[2]["args"]
    with pytest.raises(ValueError, match="polar_express_safety"):
        build({"compute": "row-l2", "polar_express_safety": 1.05})
