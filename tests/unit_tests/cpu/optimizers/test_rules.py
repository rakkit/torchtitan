# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rules: pre -> compute -> scale, and the properties the optimizer relies on."""

import pytest
import torch

from torchtitan.optimizers.disco.presets import (
    Centre,
    ColNorm,
    compute_of,
    EPS,
    GramPolar,
    Orthogonalise,
    RowL2,
    RowNorm,
    UnitNorm,
)
from torchtitan.optimizers.disco.rules import (
    Chain,
    Compute,
    lmo,
    register,
    Result,
    RowsIdentity,
    Rule,
    rule_of,
    RULES,
    Scale,
)
from torchtitan.optimizers.kernels.zeropower import zeropower_backends


def _g(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * 1e-2


def _rule(**args):
    return rule_of({"rule": "sgd", "args": args})


@pytest.mark.parametrize("backend", ["newtonschulz5", "polar_express", "svd"])
@pytest.mark.parametrize("shape", [(48, 32), (4, 24, 40)])
def test_rule_is_compute_then_scale(backend, shape):
    g = _g(*shape)
    want = Scale("spectral").whole(zeropower_backends[backend](g, steps=5, eps=EPS))
    rule = _rule(compute=backend, steps=5, scale="spectral")
    assert rule.kind == "unit" and rule.tail == (Scale("spectral"),)
    got = rule.run_whole(g).update
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert torch.equal(got, lmo(g, backend, "spectral", 5))


@pytest.mark.parametrize(
    "compute, scale",
    [
        ("row-l2", "sqrt_n"),
        ("row-l2", "inv_sqrt_n"),
        ("row-l2", "n"),
        ("sign", "inv_n"),
    ],
)
def test_row_rules_run_on_any_split_of_the_rows(compute, scale):
    g = _g(40, 16)
    rule = _rule(compute=compute, scale=scale)
    assert rule.kind == "rows"
    whole = rule.call_fused(g, 40)
    halves = torch.cat([rule.call_fused(g[:20], 40), rule.call_fused(g[20:], 40)])
    assert torch.equal(whole, halves)


@pytest.mark.parametrize(
    "old, compute, scale, formula",
    [
        (
            "embed_sqrt",
            "row-l2",
            "sqrt_n",
            lambda x, n: x / (_rows(x) + EPS) * n**0.5,
        ),
        (
            "unembed_sqrt",
            "row-l2",
            "inv_sqrt_n",
            lambda x, n: x / (_rows(x) + EPS) * n**-0.5,
        ),
        ("embed_linear", "row-l2", "n", lambda x, n: x / (_rows(x) + EPS) * n),
        (
            "unembed_linear",
            "row-l2",
            "inv_n",
            lambda x, n: x / (_rows(x) + EPS) * (1 / n),
        ),
        (
            "rmnp_row_norm_rms_rms",
            "row-l2",
            "spectral",
            lambda x, n: x / (_rows(x) + EPS) * (40 / n) ** 0.5,
        ),
    ],
)
def test_the_old_norm_factors_are_a_row_compute_and_a_scale(
    old, compute, scale, formula
):
    """`norm_factor: embed_sqrt` and friends were a row normalisation and a
    factor in one name; each is now `compute: row-l2` and a `scale`."""
    g = _g(40, 16)
    got = _rule(compute=compute, scale=scale).reference(g).update
    torch.testing.assert_close(got, formula(g, 16), rtol=1e-6, atol=0, msg=old)


def _rows(x):
    return x.pow(2).sum(-1, keepdim=True).sqrt()


def test_phases_see_the_units_row_count_not_the_shards():
    """A shard-local `m` used to mis-scale m-dependent scalings on row-sharded
    units."""
    g = _g(40, 16)
    ph = Scale("spectral")
    whole = ph(g, None, 40)
    halves = torch.cat([ph(g[:20], None, 40), ph(g[20:], None, 40)])
    assert torch.equal(whole, halves)
    ph = UnitNorm()  # a reducing phase: sum the halves' statistics
    assert ph.reduce
    stat = ph.stat(g[:20]) + ph.stat(g[20:])
    halves = torch.cat([ph(g[:20], stat, 40), ph(g[20:], stat, 40)])
    torch.testing.assert_close(halves, ph.whole(g), rtol=1e-6, atol=0)


def test_matrix_rule_needs_the_whole_unit():
    assert _rule(compute="newtonschulz5", steps=5, scale="spectral").kind == "unit"
    with pytest.raises(ValueError):  # a spectral scale on the raw moment
        _rule(scale="spectral")
    with pytest.raises(ValueError, match="steps"):  # an iteration needs steps
        _rule(compute="newtonschulz5")


def test_what_a_rule_reduces_comes_from_its_phases():
    r = _rule(pre="col-l2", compute="newtonschulz5", steps=5, scale="spectral")
    assert r.pre == (ColNorm(),) and r.pre[0].reduce
    assert (r.kind, r.needs_weight, r.has_state, r.output) == (
        "unit",
        False,
        False,
        "update",
    )
    r = _rule(pre="row-l2", compute="row-l2", scale="sqrt_n")
    assert r.pre == (RowNorm(),) and not r.pre[0].reduce and r.kind == "rows"


def test_a_chain_is_its_main_computes_kind_then_row_maps():
    c = compute_of("gram_polar+row-l2", 5)
    assert c == Chain(GramPolar(5), (RowL2(),)) and c.kind == "gram"
    assert compute_of("svd+row-l2").kind == "unit"
    with pytest.raises(ValueError, match="row maps"):
        compute_of("row-l2+svd")
    g = _g(48, 32)
    got = _rule(compute="svd+row-l2", steps=5, scale="sqrt_n").reference(g).update
    ortho = zeropower_backends["svd"](g, steps=5, eps=EPS)
    torch.testing.assert_close(got, RowL2().core(ortho) * 32**0.5, rtol=1e-6, atol=0)


def test_vector_is_one_row_under_the_row_rule():
    v = _g(64, seed=3)
    u = _rule(compute="row-l2", scale="sqrt_n").run_whole(v.view(1, -1)).update
    torch.testing.assert_close(
        u.pow(2).mean().sqrt(), torch.tensor(1.0), rtol=1e-5, atol=1e-5
    )
    sign = _rule(compute="sign", scale="inv_n").run_whole(v.view(1, -1)).update
    assert torch.equal(sign.view(-1), torch.sign(v) / 64)


def test_unknown_and_old_names_are_errors():
    with pytest.raises(ValueError):
        _rule(pre="diag-l2")
    with pytest.raises(ValueError):
        _rule(scale="nope")
    with pytest.raises(ValueError):
        _rule(compute="nope")
    with pytest.raises(ValueError):
        rule_of(dict(rule="nope"))
    for old in ("backend", "norm_factor", "pre_norm"):  # old spellings: unknown, no BC
        with pytest.raises(ValueError, match="unknown args"):
            _rule(**{old: "identity"})


def test_rule_of_reads_the_group():
    args = dict(pre="row-l2", compute="svd", steps=3, scale="sqrt_n")
    assert rule_of(dict(rule="muon", args=args)) == Rule(
        (RowNorm(),), Orthogonalise("svd", 3), scale="sqrt_n"
    )
    assert rule_of(dict(rule="sgd")) == Rule((), RowsIdentity())  # no comms
    with pytest.raises(ValueError):  # an old spelling is an error, not silence
        rule_of(dict(args={"backend_steps": 3}))


@pytest.mark.parametrize("rule", ["muon", "sgd"])
@pytest.mark.parametrize("compute", ["svd", "row-l2"])
def test_eps_is_a_constant_not_an_arg(rule, compute):
    """fp32 everywhere, so the division guard is `EPS`, not a hyper-parameter
    (the row-l2 compute's included: muon's `eps` arg is gone)."""
    assert EPS == 1e-30
    with pytest.raises(ValueError, match=r"unknown args \['eps'\]"):
        rule_of(dict(rule=rule, args=dict(compute=compute, eps=1e-8)))


def test_a_rule_that_breaks_its_contract_fails_at_build():
    """Declarations are checked when the rule is made, not at the first step."""

    class NoGram(Compute):
        kind = "gram"

    class WeightWithoutW(Compute):
        output = "weight"

        def run(self, x, w, state, ctx):
            return Result(weight=x)

    class Coeffs(Compute):
        output = "update_coeffs"
        needs_weight = True

        def run(self, x, w, state, ctx):
            return Result(update=x, alpha=torch.ones(x.shape[0]))

    with pytest.raises(ValueError, match="gram"):
        Rule(compute=NoGram())
    with pytest.raises(ValueError, match="needs the weight"):
        Rule(compute=WeightWithoutW())
    with pytest.raises(ValueError, match="post phases and scale"):
        Rule(compute=Coeffs(), scale="spectral")
    with pytest.raises(ValueError, match="post phases and scale"):
        Rule(compute=Coeffs(), post=(Centre(),))
    Rule(compute=Coeffs())  # fine with neither

    class StatefulUpdate(Compute):  # a plain update runs the compiled path,
        has_state = True  # which passes no state: it would crash at step 1

    with pytest.raises(ValueError, match="state"):
        Rule(compute=StatefulUpdate())


@pytest.mark.parametrize("name", ["amphione", "hyperball"])
def test_a_whole_unit_rule_refuses_a_gram_compute(name):
    """Their direction needs the unit whole on its owner, which a Gram compute is not."""
    with pytest.raises(ValueError, match="Gram"):
        rule_of({"rule": name, "args": {"compute": "gram_polar", "steps": 5}})


def test_register_refuses_a_taken_name_and_unknown_own_args():
    with pytest.raises(ValueError, match="already"):
        register("muon")(lambda steps=5: Rule())
    with pytest.raises(TypeError, match="own"):
        register("test_own_typo", own={"computee"})(lambda compute="svd": Rule())
    assert "test_own_typo" not in RULES


def test_tiled_orthogonalisers_refuse_units_their_kernels_cannot_take():
    """A short conv's [C, 4] unit crashes quack's GEMM and hangs the Triton
    kernels; the rule refuses it at build instead, chained or not."""
    from torchtitan.optimizers.disco.presets import Orthogonalise
    from torchtitan.optimizers.disco.rules import rule_of

    for backend, bad, good in (
        ("gram_polar_express", (1024, 4), (1024, 8)),
        ("polar_express_triton", (1024, 8), (1024, 16)),
        ("newtonschulz5_triton", (1024, 8), (1024, 16)),
    ):
        with pytest.raises(ValueError, match="tiled kernels"):
            Orthogonalise(backend, 5).check_unit(*bad)
        Orthogonalise(backend, 5).check_unit(*good)
    for backend in ("newtonschulz5", "polar_express", "svd"):
        Orthogonalise(backend, 5).check_unit(1024, 4)
    for group in (
        {"rule": "muon", "args": {"compute": "gram_polar_express", "steps": 5}},
        {"rule": "muon", "args": {"compute": "gram_polar_express+row-l2", "steps": 5}},
    ):
        with pytest.raises(ValueError, match="tiled kernels"):
            rule_of(group).compute.check_unit(1024, 4)
