# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""WeightAverages: power averages, per-unit renormalisation, row lengths,
resume, the validator swap, snapshots and score files. References are written
from the definitions, not from the code."""

import os

import pytest
import torch
import torch.nn as nn

from torchtitan.components.optimizer.averages import WeightAverages
from torchtitan.models.common.param_layout import ParamLayout, RunList, UnitSpec


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 6, bias=False)  # two declared units: rows 0-2, 3-5
        self.o = nn.Linear(6, 4, bias=False)
        self.g = nn.Parameter(torch.ones(4))
        self.w13 = nn.Parameter(
            torch.randn(2, 3, 4)
        )  # fused [2, 3, 4]: units = the two stacked [3, 4]

    def param_layouts(self):
        units = (
            UnitSpec("a", RunList(((0, 3),)), (3, 4)),
            UnitSpec("b", RunList(((3, 3),)), (3, 4)),
        )
        return {
            "qkv.weight": ParamLayout("matrix", units=units),
            "w13": ParamLayout("matrix", units=units),
        }


def _by_name(avg, model, i):
    (locs,) = avg._materialize(i)
    return {n: x for (n, _), x in zip(model.named_parameters(), locs)}


def _cfg(**kw):
    kw.setdefault("gammas", [0.0, 1.0, 3.0, 7.0])
    return WeightAverages.Config(**kw)


def _run(avg, model, steps, seed=0, start=1):
    """Random weights at every step (the 'trained' weights), returning them."""
    gen = torch.Generator().manual_seed(seed)
    seen = {}
    for s in range(start, start + steps):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(torch.randn(p.shape, generator=gen))
        seen[s] = {n: p.detach().clone() for n, p in model.named_parameters()}
        avg.step(s)
    return seen


def _reference(seen, steps, gamma):
    w = {s: float(s) ** gamma for s in steps}
    tot = sum(w.values())
    return {
        n: sum(seen[s][n].double() * w[s] for s in steps) / tot for n in seen[steps[0]]
    }


@pytest.mark.parametrize("every,start", [(1, 0), (2, 3)])
def test_power_averages_follow_the_definition(every, start):
    model = Toy()
    avg = WeightAverages(_cfg(every=every, start_step=start), model_parts=[model])
    seen = _run(avg, model, 20)
    fired = [s for s in range(1, 21) if s > start and (s - start) % every == 0]
    holder = avg.optimizers[0]
    for i, g in enumerate(avg.gammas):
        ref = _reference(
            {s - start: seen[s] for s in fired}, [s - start for s in fired], g
        )
        for n, p in model.named_parameters():
            got = holder.state[p][f"avg_{i}"].double()
            assert torch.allclose(got, ref[n], rtol=1e-5, atol=1e-6), (g, n)


def test_a_resumed_average_is_bitwise_the_uninterrupted_one():
    m1, m2 = Toy(), Toy()
    a1 = WeightAverages(_cfg(renorm=[["qkv", "rows"]]), model_parts=[m1])
    seen = _run(a1, m1, 12)
    a2 = WeightAverages(_cfg(renorm=[["qkv", "rows"]]), model_parts=[m2])
    _run(a2, m2, 5)
    sd = {k: v.clone() if torch.is_tensor(v) else v for k, v in a2.state_dict().items()}
    m3 = Toy()
    a3 = WeightAverages(_cfg(renorm=[["qkv", "rows"]]), model_parts=[m3])
    a3.load_state_dict(sd)
    gen_seen = {s: seen[s] for s in range(6, 13)}
    for s in range(6, 13):
        with torch.no_grad():
            for n, p in m3.named_parameters():
                p.copy_(gen_seen[s][n])
        a3.step(s)
    for (n, p1), (_, p3) in zip(m1.named_parameters(), m3.named_parameters()):
        for i in range(4):
            assert torch.equal(
                a1.optimizers[0].state[p1][f"avg_{i}"],
                a3.optimizers[0].state[p3][f"avg_{i}"],
            ), (n, i)


@pytest.mark.parametrize("mode", ["frobenius", "spectral"])
def test_each_declared_unit_takes_the_live_units_norm(mode):
    model = Toy()
    avg = WeightAverages(
        _cfg(gammas=[1.0], renorm=[["qkv", mode], [r"^o\.", mode], ["w13", mode]]),
        model_parts=[model],
    )
    _run(avg, model, 6)
    locs = _by_name(avg, model, 0)
    norm = (
        (lambda x: x.norm())
        if mode == "frobenius"
        else (lambda x: torch.linalg.matrix_norm(x, ord=2))
    )
    raw = avg.optimizers[0].state[model.qkv.weight]["avg_0"]
    for rows in (slice(0, 3), slice(3, 6)):  # the declared units
        got, live, a = (
            locs["qkv.weight"][rows],
            model.qkv.weight.detach()[rows],
            raw[rows],
        )
        assert torch.allclose(norm(got), norm(live), rtol=1e-5)
        assert torch.allclose(
            got / norm(got), a / norm(a), rtol=1e-5, atol=1e-6
        )  # the average's direction
    o = locs["o.weight"]
    assert torch.allclose(
        norm(o), norm(model.o.weight.detach()), rtol=1e-5
    )  # undeclared: one unit
    for k in (0, 1):  # a stacked [2, 3, 4] parameter: unit k = slice k
        assert torch.allclose(
            norm(locs["w13"][k]), norm(model.w13.detach()[k]), rtol=1e-5
        )
    assert torch.equal(
        locs["g"], avg.optimizers[0].state[model.g]["avg_0"]
    )  # unmatched: as averaged


def test_rows_take_the_averaged_row_lengths():
    model = Toy()
    avg = WeightAverages(
        _cfg(gammas=[3.0], renorm=[["qkv", "rows"], ["w13", "rows"]]),
        model_parts=[model],
    )
    seen = _run(avg, model, 7)
    w = {s: float(s) ** 3 for s in range(1, 8)}
    lengths = sum(
        seen[s]["qkv.weight"].double().norm(dim=1, keepdim=True) * w[s] for s in w
    ) / sum(w.values())
    q = _by_name(avg, model, 0)["qkv.weight"].double()
    assert torch.allclose(q.norm(dim=1, keepdim=True), lengths, rtol=1e-5)
    raw = avg.optimizers[0].state[model.qkv.weight]["avg_0"].double()
    assert torch.allclose(
        q / q.norm(dim=1, keepdim=True), raw / raw.norm(dim=1, keepdim=True), rtol=1e-5
    )
    lengths3 = sum(
        seen[s]["w13"].double().norm(dim=-1, keepdim=True) * w[s] for s in w
    ) / sum(w.values())
    q3 = _by_name(avg, model, 0)["w13"].double()
    assert torch.allclose(q3.norm(dim=-1, keepdim=True), lengths3, rtol=1e-5)


class _Validator:
    """Records the weights it is asked to score and the tag."""

    def __init__(self):
        self.calls = []

    def validate(self, model_parts, step, tag=None):
        self.calls.append(
            (
                step,
                tag,
                {n: p.detach().clone() for n, p in model_parts[0].named_parameters()},
            )
        )


def test_validation_scores_each_average_and_restores_the_weights_bitwise():
    model = Toy()
    avg = WeightAverages(
        _cfg(gammas=[0.0, 7.0], renorm=[["qkv", "frobenius"]]), model_parts=[model]
    )
    _run(avg, model, 9)
    live = {n: p.detach().clone() for n, p in model.named_parameters()}
    v = _Validator()
    avg.validate(v, 9)
    assert [c[1] for c in v.calls] == ["avg_g0", "avg_g7"]
    for i, (_, _, w) in enumerate(v.calls):
        (locs,) = avg._materialize(i)
        for (n, _), x in zip(model.named_parameters(), locs):
            assert torch.equal(w[n], x), n
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), live[n]), n
    avg2 = WeightAverages(_cfg(gammas=[0.0], evaluate=False), model_parts=[model])
    v2 = _Validator()
    avg2.validate(v2, 9)
    assert v2.calls == []


def test_snapshots_and_score_files(tmp_path):
    model = Toy()
    avg = WeightAverages(
        _cfg(gammas=[], snapshot_steps=[2, 4]),
        model_parts=[model],
        output_dir=str(tmp_path),
    )
    seen = _run(avg, model, 5)
    for s in (2, 4):
        blob = torch.load(tmp_path / "snapshots" / f"step-{s}.pt", weights_only=True)
        assert blob["step"] == s
        assert set(blob["weights"]) == {n for n, _ in model.named_parameters()}
        for n, t in blob["weights"].items():
            assert torch.equal(t, seen[s][n])
    assert not os.path.exists(tmp_path / "snapshots" / "step-3.pt")
    mix = {n: 0.5 * (seen[2][n] + seen[4][n]) for n in seen[2]}
    torch.save({"weights": mix}, tmp_path / "mix24.pt")
    live = {n: p.detach().clone() for n, p in model.named_parameters()}
    scorer = WeightAverages(
        _cfg(gammas=[], score_files=[str(tmp_path / "mix24.pt")]), model_parts=[model]
    )
    v = _Validator()
    scorer.score(v, 5)
    ((step, tag, w),) = v.calls
    assert (step, tag) == (5, "score/mix24")
    for n in mix:
        assert torch.equal(w[n], mix[n])
        assert torch.equal(dict(model.named_parameters())[n].detach(), live[n])


@pytest.mark.parametrize(
    "kw",
    [
        {"gammas": [-1.0]},
        {"gammas": [1.0, 1.0]},
        {"every": 0},
        {"renorm": [["qkv", "max"]]},
        {"renorm": [["qkv"]]},
        {"snapshot_steps": [0]},
    ],
)
def test_bad_configs_are_refused(kw):
    with pytest.raises(ValueError):
        WeightAverages.Config(**{"gammas": [0.0], **kw})


def test_a_vector_cannot_be_renormalised_as_a_matrix():
    with pytest.raises(ValueError, match="matrix"):
        WeightAverages(_cfg(renorm=[[r"^g$", "frobenius"]]), model_parts=[Toy()])


def test_dtensor_parameters_on_a_one_rank_mesh():
    """Sharded (Shard 0) parameters: the swap writes the parameter's own local
    storage, renorm and rows work on DTensors, and the state round-trips."""
    import socket

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard

    own = not dist.is_initialized()
    if own:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = Toy()
        for mod, name in (
            (model.qkv, "weight"),
            (model.o, "weight"),
            (model, "g"),
            (model, "w13"),
        ):
            p = getattr(mod, name)
            setattr(
                mod, name, nn.Parameter(distribute_tensor(p.detach(), mesh, [Shard(0)]))
            )
        cfg = _cfg(
            gammas=[1.0],
            renorm=[["qkv", "spectral"], [r"^o\.", "rows"], ["w13", "rows"]],
        )
        avg = WeightAverages(cfg, model_parts=[model])
        for s in range(1, 5):
            with torch.no_grad():
                for _, p in model.named_parameters():
                    p.to_local().copy_(torch.randn(p.to_local().shape))
            avg.step(s)
        live = {n: p.to_local().clone() for n, p in model.named_parameters()}
        v = _Validator()
        avg.validate(v, 4)
        ((_, tag, w),) = v.calls
        (locs,) = avg._materialize(0)
        for (n, _), x in zip(model.named_parameters(), locs):
            assert torch.equal(w[n].to_local(), x), n  # the validator saw the average
        for n, p in model.named_parameters():
            assert torch.equal(
                p.to_local(), live[n]
            ), n  # and the live weights are back
        sd = avg.state_dict()
        model2 = Toy()
        for mod, name in (
            (model2.qkv, "weight"),
            (model2.o, "weight"),
            (model2, "g"),
            (model2, "w13"),
        ):
            p = getattr(mod, name)
            setattr(
                mod, name, nn.Parameter(distribute_tensor(p.detach(), mesh, [Shard(0)]))
            )
        avg2 = WeightAverages(cfg, model_parts=[model2])
        avg2.load_state_dict(sd)
        (locs2,) = avg2._materialize(0)
        with torch.no_grad():
            for (_, p2), (_, p) in zip(
                model2.named_parameters(), model.named_parameters()
            ):
                p2.to_local().copy_(p.to_local())
        (locs2,) = avg2._materialize(0)
        for a, b in zip(locs, locs2):
            assert torch.equal(a, b)
    finally:
        if own:
            dist.destroy_process_group()


def test_a_snapshot_mixture_equals_the_online_average(tmp_path):
    """Snapshots at every step mixed with power weights give the online average,
    renormalised the same way (the post-hoc path is the online one)."""
    import importlib.util
    import pathlib

    model = Toy()
    cfg = _cfg(
        gammas=[3.0],
        renorm=[["qkv", "spectral"], [r"^o\.", "rows"]],
        snapshot_steps=list(range(1, 9)),
    )
    avg = WeightAverages(cfg, model_parts=[model], output_dir=str(tmp_path))
    _run(avg, model, 8)
    spec = importlib.util.spec_from_file_location(
        "mix", pathlib.Path(__file__).parents[3] / "scripts" / "mix_snapshots.py"
    )
    mix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mix)
    snaps = [str(tmp_path / "snapshots" / f"step-{s}.pt") for s in range(1, 9)]
    mix.main([str(tmp_path / "m.pt"), *snaps, "--gamma", "3"])
    got = torch.load(tmp_path / "m.pt", weights_only=True)["weights"]
    online = _by_name(avg, model, 0)
    for n, x in online.items():
        assert torch.allclose(got[n], x, rtol=1e-5, atol=1e-6), n


def test_exported_averages_are_the_scored_ones(tmp_path):
    model = Toy()
    cfg = _cfg(gammas=[0.0, 3.0], renorm=[["qkv", "frobenius"]], export_steps=[6])
    avg = WeightAverages(cfg, model_parts=[model], output_dir=str(tmp_path))
    _run(avg, model, 6)
    for i, g in enumerate(("0", "3")):
        got = torch.load(
            tmp_path / "snapshots" / f"avg_g{g}-step-6.pt", weights_only=True
        )["weights"]
        for n, x in _by_name(avg, model, i).items():
            assert torch.equal(got[n], x.float()), (g, n)
    with pytest.raises(ValueError, match="export_steps"):
        WeightAverages.Config(gammas=[], export_steps=[3])
