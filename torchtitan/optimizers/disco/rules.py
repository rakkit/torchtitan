# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What an update rule is, and how presets are registered.

A rule turns one unit's momentum into the step its rows receive. A unit is a
matrix a rule works on: a whole weight, one expert, or q / k / v of a fused
projection (`models/common/param_layout.py`).

    momentum rows ----> pre ----> compute ----> post ----> x scale(m, n) ----> U
    (where FSDP                                                    apply:
     keeps them)                                     w <- decay w - lr U

    pre      Phases on the momentum rows. A phase works row by row, or declares
             `reduce` and gives a statistic the optimizer sums over every rank
             holding rows of the unit (one packed all-reduce per round).
    compute  The map, and the only place a direction is formed. Its `kind`
             says where the unit must be whole, and so what it costs:
               rows   the rows a rank holds are enough        no traffic
               gram   those rows plus the [n, n] Gram         one all-reduce
               unit   the whole unit on one owner rank        all-to-all out, back
             It may declare `needs_weight` (the owner receives W), `has_state`
             (kept per unit on the owner), `keeps_radius` (it holds each
             unit's norm, which cancels weight decay) and its `output`: an
             update, a per-unit coefficient `w <- alpha w - U`, or the new
             weight. A `Chain` is a main compute followed by row maps
             (`gram_polar+row-l2`).
    post     Phases a preset needs after the compute (the head's re-centring).
    scale    The unit's step size relative to lr, a factor of its shape: a
             constant, the last multiply in the compiled compute.

How the optimizer runs them each step (`optimizer.py`), per unit:

    pre   : on the rows each rank holds; reducing phases -> one packed all-reduce
    rows  : compute, post, scale on those rows
    gram  : partial X^T X -> one packed all-reduce -> finish on those rows
    unit  : rows --all-to-all--> owner: compute, post, scale --all-to-all--> rows
            (an expert batch is already whole on its rank: one batched call)
    apply : w <- decay w - lr U  |  w <- alpha w - U  |  w <- W_next
            (decay = 1 - lr wd, or 1 - wd with `decoupled_wd`)

A preset is a function registered by name whose keyword arguments are its
args; `optimizer.groups.<g>.rule` picks it, `.args` fills them. The built-in
phases, computes and presets are in `presets.py`: a new optimizer is a
`Compute` and a preset there, or in its own module, side by side with muon
and aus.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple

import torch


# ------------------------------------------------------------------ phases
class Phase:
    """A map on rows of one unit (`[r, n]`, or a stack `[B, m, n]` of whole
    units). `reduce`: it also needs `stat(rows)` summed over ALL rows of the
    unit; the optimizer sums the partial statistics over the ranks holding the
    rows. `m` is the unit's true row count; columns are never sharded."""

    reduce = False

    def stat(self, rows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, rows: torch.Tensor, stat, m: int) -> torch.Tensor:
        raise NotImplementedError

    def whole(self, x: torch.Tensor) -> torch.Tensor:
        """The phase on units that are whole here."""
        return self(x, self.stat(x) if self.reduce else None, x.size(-2))


# args.scale: (m, n) -> factor. A unit is an [m, n] matrix as it is stored:
# m rows, n columns. For a Linear weight [d_out, d_in]: m = d_out, n = d_in.
# For the embedding table and the LM head [vocab, dim]: m = vocab, n = dim.
SCALES: dict[str, Callable[[int, int], float]] = {
    "one": lambda m, n: 1.0,
    "spectral": lambda m, n: (m / n) ** 0.5,  # sqrt(d_out / d_in)
    "image_spectral": lambda m, n: max((m / n) ** 0.5, 1.0),  # at least 1
    "n": lambda m, n: n,  # d_in; per row of a table: dim
    "sqrt_n": lambda m, n: n**0.5,  # sqrt(d_in); embedding rows: sqrt(dim)
    "inv_n": lambda m, n: 1 / n,  # 1 / d_in
    "inv_sqrt_n": lambda m, n: n**-0.5,  # 1 / sqrt(d_in); head rows: 1 / sqrt(dim)
    "sqrt_mn": lambda m, n: (m * n) ** 0.5,  # sqrt(d_out d_in)
}


@dataclass(frozen=True)
class Scale(Phase):
    """`u * SCALES[name](m, n)`: the unit's step size relative to `lr`."""

    name: str

    def __post_init__(self):
        if self.name not in SCALES:
            raise ValueError(f"unknown scale {self.name!r}; known: {sorted(SCALES)}")

    def __call__(self, rows, stat, m):
        return rows * SCALES[self.name](m, rows.size(-1))


# ------------------------------------------------------------------ compute
class StepCtx(NamedTuple):
    lr: float
    wd: float
    t: int  # 1-based optimizer step
    decoupled: bool = False  # weight decay independent of lr

    @property
    def decay(self) -> float:
        """What weight decay scales W by this step: `1 - wd` when decoupled,
        else `1 - lr wd`."""
        return 1.0 - self.wd if self.decoupled else 1.0 - self.lr * self.wd


@dataclass
class Result:
    """What a compute returns for a stack of `B` units. `update` alone: the
    plain step. `alpha`, per unit: `w <- alpha w - update`, the step's size
    folded into the update. `weight`: the new weight. `state`: what the
    compute gets back next step for the same units, stacked on dim 0."""

    update: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    state: torch.Tensor | None = None


class Compute:
    """The map of a rule. `kind`: where the unit has to be whole ("rows",
    "gram", "unit"). `needs_weight`: the computing rank also receives the
    unit's pre-update weight. `has_state`: `run` takes and returns per-unit
    state of shape `state_shape(m, n)`, kept on the computing rank between
    steps and checkpointed; the first step sees zeros. `keeps_radius`: the
    step holds each unit's norm, so weight decay would be undone by it.
    `output`: "update", "update_coeffs" or "weight" -- see `Result`.

    Rows and unit kinds with output "update" implement `core`, which the rule
    traces together with its row-wise post phases into one compiled function.
    Gram kinds implement `gram` and `finish`; any other output implements
    `run`. None of them may modify their inputs in place."""

    kind = "unit"
    needs_weight = False
    has_state = False
    keeps_radius = False
    output = "update"

    def core(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gram(self, rows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def finish(self, rows: torch.Tensor, R: torch.Tensor, m: int) -> torch.Tensor:
        raise NotImplementedError

    def run(self, x, w, state, ctx: StepCtx) -> Result:
        raise NotImplementedError

    def state_shape(self, m: int, n: int) -> tuple[int, ...]:
        return ()


@dataclass(frozen=True)
class RowsIdentity(Compute):
    """No map: the rule is its phases, applied to the rows a rank holds."""

    kind = "rows"

    def core(self, x):
        return x


def _check_compute(c: Compute) -> None:
    """A compute implements the methods its kind and output are called by."""
    overrides = {
        "gram": ("gram", "finish"),
        "rows": ("core",),
        "unit": ("core",) if c.output == "update" else ("run",),
    }[c.kind]
    for name in overrides:
        if getattr(type(c), name) is getattr(Compute, name):
            raise ValueError(f"{type(c).__name__} must implement `{name}`")


@dataclass(frozen=True)
class Chain(Compute):
    """`main`, then row maps on its result: `gram_polar+row-l2` is the Gram
    polar factor with every row brought to unit norm. The chain is `main`'s
    kind; the row maps run where its result is (fused into the compiled
    compute, or on the local rows after a Gram `finish`)."""

    main: Compute
    after: tuple[Compute, ...]

    def __post_init__(self):
        _check_compute(self.main)
        if self.main.output != "update" or self.main.needs_weight:
            raise ValueError("a chain starts from a compute returning a plain update")
        for c in self.after:
            if c.kind != "rows" or c.output != "update":
                raise ValueError(f"only row maps can follow in a chain, not {c}")

    @property
    def kind(self) -> str:
        return self.main.kind

    def core(self, x):
        y = self.main.core(x)
        for c in self.after:
            y = c.core(y)
        return y

    def gram(self, rows):
        return self.main.gram(rows)

    def finish(self, rows, R, m):
        y = self.main.finish(rows, R, m)
        for c in self.after:
            y = c.core(y)
        return y


# ------------------------------------------------------------------ rule
_COMPILED: dict[tuple, Callable] = {}


@dataclass(frozen=True)
class Rule:
    """pre -> compute -> post -> x scale.

    `pre`: phases on the moment. `compute`: the map. `post`: projections a
    preset needs after the map (the head's re-centring), not a config key.
    `scale`: the unit's step size relative to `lr`, a factor of its shape.
    Hashable: units with equal rules batch together."""

    pre: tuple[Phase, ...] = ()
    compute: Compute = field(default_factory=RowsIdentity)
    post: tuple[Phase, ...] = ()
    scale: str = "one"

    def __post_init__(self):
        c = self.compute
        Scale(self.scale)  # validates the name
        if c.kind not in ("rows", "gram", "unit"):
            raise ValueError(f"unknown compute kind {c.kind!r}")
        if c.output not in ("update", "update_coeffs", "weight"):
            raise ValueError(f"unknown compute output {c.output!r}")
        if c.output != "update" and self.tail:
            raise ValueError(
                f"{type(c).__name__} returns {c.output!r}; post phases and scale "
                "apply to a plain update, so it has to fold them into its compute"
            )
        if c.output == "weight" and not c.needs_weight:
            raise ValueError("a compute returning the new weight needs the weight")
        if c.output == "update" and (c.needs_weight or c.has_state):
            raise ValueError(
                "a plain update runs the compiled path, which passes neither the "
                "weight nor state: return 'update_coeffs' or 'weight' instead"
            )
        if c.kind != "unit" and (c.needs_weight or c.has_state or c.output != "update"):
            raise ValueError(
                "weights, state and non-update outputs need the unit whole on "
                "one rank (compute kind 'unit')"
            )
        _check_compute(c)

    @property
    def kind(self) -> str:
        return self.compute.kind

    @property
    def needs_weight(self) -> bool:
        return self.compute.needs_weight

    @property
    def has_state(self) -> bool:
        return self.compute.has_state

    @property
    def output(self) -> str:
        return self.compute.output

    @cached_property
    def tail(self) -> tuple[Phase, ...]:
        """What follows the compute: `post`, then the scale."""
        return self.post + ((Scale(self.scale),) if self.scale != "one" else ())

    @cached_property
    def fused_tail(self) -> tuple[Phase, ...]:
        """The leading row-wise phases of `tail`, traced into `fused` with the
        compute (rows and unit kinds with output "update"); the rest run after
        it -- a scale is a constant in the compiled graph."""
        if self.kind == "gram" or self.output != "update":
            return ()
        n = 0
        while n < len(self.tail) and not self.tail[n].reduce:
            n += 1
        return self.tail[:n]

    @cached_property
    def rest_tail(self) -> tuple[Phase, ...]:
        return self.tail[len(self.fused_tail) :]

    @cached_property
    def fused(self) -> Callable:
        """`core` then `fused_tail`, compiled once per (compute, phases)."""
        key = (self.compute, self.fused_tail)
        fn = _COMPILED.get(key)
        if fn is None:
            core, phases = self.compute.core, self.fused_tail

            @torch.compile(dynamic=False, fullgraph=True)
            def fn(x, m):
                y = core(x)
                for ph in phases:
                    y = ph(y, None, m)
                return y

            _COMPILED[key] = fn
        return fn

    def call_fused(self, x: torch.Tensor, m: int) -> torch.Tensor:
        if x.ndim == 3 and x.shape[0] == 0:
            return x
        return call_compiled(self.fused, x, m)

    def run_whole(self, x, w=None, ctx: StepCtx | None = None, state=None) -> Result:
        """compute and tail on units that are whole here: `x` one unit
        `[m, n]` or a stack `[B, m, n]`; `w` the matching weights when
        `needs_weight`; `state` what the compute returned last step for them."""
        if self.output != "update":
            return self.compute.run(x, w, state, ctx)
        m = x.size(-2)
        if self.kind == "gram":
            y = self.compute.finish(x, self.compute.gram(x), m)
        else:
            y = self.call_fused(x, m)
        for ph in self.rest_tail:
            y = ph.whole(y)
        return Result(update=y)

    def reference(self, g, w=None, ctx: StepCtx | None = None, state=None) -> Result:
        """The whole rule on one unit on one rank (tests, and the single-rank
        meaning every distributed path must reproduce)."""
        for ph in self.pre:
            g = ph.whole(g)
        return self.run_whole(g, w, ctx, state)


_COMPILE_LIMIT = 128


def compile_limits():
    """Every compiled rule shares one code object, hence one recompile budget
    over all (compute, phases, shape): raise it around the calls. Entering it
    costs ~70 us, so a step enters it once, not per call."""
    return torch._dynamo.config.patch(
        recompile_limit=_COMPILE_LIMIT, cache_size_limit=_COMPILE_LIMIT
    )


def call_compiled(fn: Callable, *args):
    """A compiled rule function, within the rules' recompile budget."""
    if torch._dynamo.config.recompile_limit >= _COMPILE_LIMIT:
        return fn(*args)  # inside `compile_limits()` (a DiSCO step)
    with compile_limits():
        return fn(*args)


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class Preset:
    build: Callable[..., Rule]
    args: dict  # the preset's args and their defaults: `build`'s keywords
    own: frozenset  # args a group of it never takes from the default group

    @property
    def inherits(self) -> frozenset:
        """The default group's args a group of this preset takes."""
        return frozenset(self.args) - self.own


RULES: dict[str, Preset] = {}


def register(name: str, *, own=()):
    """Register the decorated function as the preset `name`. Its keyword
    arguments are the preset's args, with their defaults; `own` lists the args
    a group of it never takes from the default group."""

    def deco(build):
        params = inspect.signature(build).parameters.values()
        if any(p.default is inspect.Parameter.empty for p in params):
            raise TypeError(f"preset {name!r}: every arg needs a default")
        args = {p.name: p.default for p in params}
        if not set(own) <= set(args):
            raise TypeError(
                f"preset {name!r}: own {sorted(set(own) - set(args))} are not its args"
            )
        if name in RULES:
            raise ValueError(f"preset {name!r} is already registered")
        RULES[name] = Preset(build, args, frozenset(own))
        return build

    return deco


def pick(table: dict, key: str, what: str):
    if key not in table:
        raise ValueError(f"unknown {what} {key!r}; known: {sorted(table)}")
    return table[key]


_RENAMED = {"pre_norm": "pre", "backend": "compute", "norm_factor": "scale"}


def rule_of(group: dict) -> Rule:
    """The rule a param group configures: `group["rule"]` from `RULES`, called
    with `group["args"]` (the preset's defaults fill the rest)."""
    name = group.get("rule", "muon")
    preset = pick(RULES, name, "rule")
    args = dict(group.get("args") or {})
    old = sorted(set(args) & set(_RENAMED))
    if old:
        raise ValueError(
            f"{name}: args {old} were renamed: "
            + ", ".join(f"{k} -> {_RENAMED[k]}" for k in old)
            + " (a row normalisation such as embed_sqrt is now compute: row-l2"
            " with scale: sqrt_n; see CONFIG_MIGRATION section 69)"
        )
    unknown = set(args) - set(preset.args)
    if unknown:
        raise ValueError(
            f"{name}: unknown args {sorted(unknown)}; allowed {sorted(preset.args)}"
        )
    return preset.build(**args)


def lmo(g, compute, scale, steps):
    """Reference entry point for tests: one plain rule on one full unit."""
    args = {"compute": compute, "steps": steps, "scale": scale}
    return rule_of({"rule": "sgd", "args": args}).reference(g).update


__all__ = [
    "call_compiled",
    "Chain",
    "Compute",
    "lmo",
    "Phase",
    "pick",
    "Preset",
    "register",
    "Result",
    "RowsIdentity",
    "Rule",
    "rule_of",
    "RULES",
    "Scale",
    "SCALES",
    "StepCtx",
]

# The built-in presets register themselves on import; rule_of needs them.
from . import presets  # noqa: E402, F401
