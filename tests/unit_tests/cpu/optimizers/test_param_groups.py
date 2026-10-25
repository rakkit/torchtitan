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
from torchtitan.optimizers.disco.presets import AUSMap, EtaSchedule
from torchtitan.optimizers.disco.rules import rule_of


@pytest.fixture(autouse=True, scope="module")
def _process_group():
    if not dist.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield


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
            "rule": "aus",
            "args": {
                "compute": "svd",
                "eta": {"shape": "inverse_poly", "start": 10, "c": 0.1},
            },
        },
    ]


_LAYOUTS = {n: ParamLayout("matrix") for n in ("a.weight", "b.weight", "c.weight")}


def _opt(**kw):
    kw.setdefault("args", {"steps": 3})
    kw.setdefault("param_layouts", _LAYOUTS)
    return DiSCO(_groups(), parallel_dims=_NoParallel(), lr=0.1, **kw)


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
            parallel_dims=_NoParallel(),
            lr=0.1,
            rule="right_polar",
            param_layouts={"w.weight": ParamLayout("matrix")},
        )


def test_groups_inherit_the_defaults_and_merge_args():
    opt = _opt()
    g0, g1 = opt.param_groups
    assert (g0["lr"], g0["momentum"], g0["rule"]) == (0.1, 0.95, "muon")
    assert g0["args"] == {"steps": 3}
    assert (g1["lr"], g1["rule"]) == (0.5, "aus")
    assert g1["args"] == {
        "steps": 3,
        "compute": "svd",
        "eta": {"shape": "inverse_poly", "start": 10, "c": 0.1},
    }
    rules = {e.name: e.rule for e in opt.entries}
    assert rules["a.weight"] == rule_of({"rule": "muon", "args": {"steps": 3}})
    aus = rules["c.weight"].compute
    assert isinstance(aus, AUSMap) and aus.eta == EtaSchedule("inverse_poly", 10, 0.1)


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
            cfg, model_parts=[model], parallel_dims=_NoParallel()
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
        cfg, model_parts=[Toy()], parallel_dims=_NoParallel()
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
        DiSCOOptimizersContainer(cfg, model_parts=[Toy()], parallel_dims=_NoParallel())


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
        parallel_dims=_NoParallel(),
        lr=0.1,
        param_layouts=layouts,
        **kw,
    )


def test_a_group_without_args_takes_only_what_its_rule_inherits():
    """torch filled a missing `args` with the default dict itself, so a head
    group written without args ran with the default group's `scale` (448x
    too large on the real head)."""
    default = {"compute": "svd", "steps": 3, "scale": "spectral"}
    opt = _two_groups({"rule": "lm_head"}, rule="muon", args=default)
    assert opt.param_groups[0]["args"] == default  # same rule: every default
    assert opt.param_groups[1]["args"] == {"steps": 3}  # lm_head owns `scale`
    head = next(e.rule for e in opt.entries if e.name == "h.weight")
    assert head.scale == "inv_sqrt_n"


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
        parallel_dims=_NoParallel(),
        lr=0.1,
        momentum=0.1,
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


def test_the_weight_decay_pair_comes_from_the_config():
    """`weight_decay` and `decoupled_wd` are both config-only: a resume never
    pairs the checkpoint's decay with the config's mode."""
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

    def build(wd, decoupled):
        cfg = DiSCOOptimizersContainer.Config(
            groups={
                "default": {
                    "lr": 0.1,
                    "weight_decay": wd,
                    "decoupled_wd": decoupled,
                    "rule": "muon",
                    "args": {"compute": "svd"},
                }
            }
        )
        model = torch.nn.Sequential(torch.nn.Linear(4, 8, bias=False))
        model.param_layouts = lambda: {"0.weight": ParamLayout("matrix")}
        return DiSCOOptimizersContainer(
            cfg, model_parts=[model], parallel_dims=_NoParallel()
        )

    old = build(0.1, False)
    sd = old.state_dict()
    assert not any(k.endswith((".weight_decay", ".decoupled_wd")) for k in sd)
    # An older checkpoint still carries weight_decay: the config wins.
    fqn = old.optimizers[0].param_groups[0]["param_names"][0]
    sd[f"param_groups.{fqn}.weight_decay"] = 0.1
    new = build(1e-4, True)
    new.load_state_dict(sd)
    g = new.optimizers[0].param_groups[0]
    assert (g["weight_decay"], g["decoupled_wd"]) == (1e-4, True)


def _table_container(table):
    """Two DiSCO groups; the second (not the default: its keys are DiSCO's
    defaults) carries the table."""
    from torchtitan.optimizers.container import DiSCOOptimizersContainer

    second = {
        "match": r"^1\.weight$",
        "lr": 0.2,
        "rule": "muon",
        "args": {"compute": "svd"},
    }
    if table is not None:
        second["lr_table"] = table
    cfg = DiSCOOptimizersContainer.Config(
        groups={
            "second": second,
            "default": {"lr": 0.1, "rule": "muon", "args": {"compute": "svd"}},
        }
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, bias=False), torch.nn.Linear(8, 4, bias=False)
    )
    model.param_layouts = lambda: {
        "0.weight": ParamLayout("matrix"),
        "1.weight": ParamLayout("matrix"),
    }
    return model, DiSCOOptimizersContainer(
        cfg, model_parts=[model], parallel_dims=_NoParallel()
    )


def test_an_lr_table_is_config_not_checkpoint_state():
    """A group's lr_table stays out of checkpoints; a run with a table loads a
    checkpoint saved with another table or with none (a branch), and keeps its own."""
    a, b = [[0, 1.0], [10, 0.5]], [[0, 1.0], [10, 0.0]]
    for saved_table, live_table in ((a, b), (None, b), (a, None)):
        model, container = _table_container(saved_table)
        for m in model:
            m.weight.grad = torch.ones_like(m.weight)
        container.step()
        sd = container.state_dict()
        assert not any(k.endswith(".lr_table") for k in sd)
        model, container = _table_container(live_table)
        container.load_state_dict(
            {k: v.clone() if torch.is_tensor(v) else v for k, v in sd.items()}
        )
        (g,) = [
            g
            for g in container.optimizers[0].param_groups
            if g["param_names"] == ["1.weight"]
        ]
        assert g.get("lr_table") == live_table
