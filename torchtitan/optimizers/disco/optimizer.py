# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DiSCO: distributed Muon / Scion.

Per step: momentum -> each rule's `pre` phases on the rows each rank holds ->
its compute, wherever the unit has to be whole -> its `post` phases -> one
apply. Where a unit is whole follows from its family (`layout.classify`) and
its rule's compute kind (`rules.py`):

    local       rows and Gram rules on the rows a rank holds (one packed
                all-reduce per reducing phase, one for the Gram); vectors and
                scalars whole
    expert      the rank holds whole experts: one batched call per shape
    sharded     rows over the FSDP axis: one packed all-to-all pair
    replicated  whole on every rank: owners compute, one all-gather

What a rule declares adds traffic only where it is needed: the pre-update
weight to the owner (`needs_weight`), per-unit coefficients (`update_coeffs`:
one small all-reduce). A rule returning a new weight sends back the step
`W - W_next` in the update's slot. On a logging step the owners also receive the
pre-update weights (and, for local units, the moment and the update) over the
same tables, and every metric is computed after the post phases, from what the
apply writes -- see `metrics.py`.
"""

import functools
import hashlib
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.profiler import record_function

from torchtitan.distributed.utils import fsdp_shard_mesh, metrics_shard_mesh
from torchtitan.observability import structured_logger as sl

from ..helpers import gram_helper
from ..helpers.head_geometry import head_geometry, HEAD_METRIC_NAMES
from ..helpers.norm_helper import NORM_FUNCTIONS
from ..helpers.radial_helper import new_radial_state, RADIAL_ACCUMULATOR_NAMES
from ..utils import remove_orig_mod_and_weight_for_p_name
from .layout import (
    _sharded_axes,
    classify,
    Family,
    local_rows,
    matrix_view,
    ParamLayout,
    resolve_units,
    rows_of_coordinate,
    RunList,
)
from .metrics import ExpertUnits, MetricSink, pseudo_post_update_weight, UnitValues
from .presets import Sign
from .rules import compile_limits, Result, Rule, rule_of, RULES, StepCtx
from .transport import Exchange, Replicated, Unit, unit_rows

logger = logging.getLogger(__name__)


def _momentum(bufs, grads, m: float) -> None:
    torch._foreach_mul_(bufs, 1.0 - m)
    torch._foreach_add_(bufs, grads, alpha=m)


# One pass over the moments instead of two; bitwise equal to `_momentum`.
_momentum_one_pass = torch.compile(_momentum, fullgraph=True, dynamic=False)


def _ignore(*a, **k) -> None:
    return None


@dataclass
class _Entry:
    i: int  # index into DiSCO.entries
    name: str
    clean: str
    param: torch.Tensor
    local: torch.Tensor  # the rank's rows, as stored
    group_idx: int
    rule: Rule
    family: Family
    view: tuple[int, int]  # [R, C]
    kind: str = "matrix"  # the model's declaration: embedding, router, ...
    units: list = field(default_factory=list)
    unit_local: list = field(default_factory=list)  # per unit: its rows within `local`
    local_units: list = field(default_factory=list)  # positions of units held here
    split: bool = False  # a 1-D parameter whose single row is spread over ranks
    sh: int = -1  # index into the sharded-exchange parameter list, or -1
    rep: int = -1  # index into the replicated parameter list, or -1

    def rows2d(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(1, -1) if t.ndim <= 1 else t.reshape(-1, self.view[1])


def _check_rule(name: str, p, layout: ParamLayout, fam: Family, rule: Rule) -> None:
    """What a parameter's layout allows its rule to be."""
    if layout.kind == "scalar" and (
        not isinstance(rule.compute, Sign) or rule.pre or rule.post
    ):
        raise ValueError(f"{name}: scalar parameters need compute=sign, pre=identity")
    if p.ndim < 2 and rule.pre:
        raise ValueError(
            f"{name}: pre is defined for matrices, not {p.ndim}-D parameters"
        )
    if p.ndim < 2 and (
        rule.kind == "gram" or rule.needs_weight or rule.output != "update"
    ):
        raise ValueError(
            f"{name}: a Gram rule, or one that needs the weight or returns more "
            "than an update, runs on matrices only"
        )
    if layout.kind in ("embedding", "unembedding") and rule.kind == "unit":
        raise ValueError(
            f"{name} is the model's {layout.kind}: it needs a row or Gram rule "
            "(compute=row-l2, right_polar, lm_head), not one that gathers all "
            f"its vocabulary rows on one rank ({rule}). A group matching it by "
            "name may have missed it; groups can match by kind instead."
        )


def _per_unit(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A per-unit coefficient broadcast against units `x` (`[m, n]` or `[B, m, n]`)."""
    return a.view(-1, *(1,) * (x.ndim - 1)) if a.ndim == 1 else a


class DiSCO(torch.optim.Optimizer):
    _CONFIG_ONLY_GROUP_KEYS = (
        "rule",
        "args",
    )  # the run's config, never the checkpoint's

    def __init__(
        self,
        params,
        *,
        parallel_dims,
        lr,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=False,
        rule="muon",
        args=None,
        communication_dtype=torch.float32,
        enable_step_tracing=False,
        param_layouts=None,
    ):
        """`lr, weight_decay, momentum, nesterov, rule, args` are the defaults a
        param group inherits (torch semantics); a group's own `args` merge over
        the default `args` its rule inherits. `rule` names a preset in
        `rules.RULES`. `momentum` is the weight of the NEW gradient:
        `buf <- (1 - momentum) buf + momentum g`."""
        self._span = sl.log_trace_span if enable_step_tracing else nullcontext
        self._trace_scalar = sl.log_trace_scalar if enable_step_tracing else _ignore
        self.communication_dtype = communication_dtype
        self._param_layouts = {
            remove_orig_mod_and_weight_for_p_name(k, remove_weight=False): v
            for k, v in (param_layouts or {}).items()
        }
        self.parallel_dims = parallel_dims
        if parallel_dims.tp_enabled:
            raise ValueError(
                "DiSCO does not support tensor parallelism; use tensor_parallel_degree=1"
            )
        if parallel_dims.dp_replicate_enabled or parallel_dims.fsdp_enabled:
            self.is_dp_rank_0 = (
                parallel_dims.get_optional_mesh("loss").get_local_rank() == 0
            )
        else:
            self.is_dp_rank_0 = dist.get_rank() == 0
        self._default_args = dict(args or {})
        super().__init__(
            params,
            dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                rule=rule,
                args=None,  # a group's own; the default's are merged below
                t=0,  # optimizer steps taken; rules with a schedule read it
            ),
        )
        for g in self.param_groups:
            unknown = sorted(set(g) - set(self.defaults) - {"params", "param_names"})
            if unknown:
                raise ValueError(
                    f"DiSCO group {g.get('param_names', [])[:2]}: unknown keys "
                    f"{unknown}; a rule's settings (compute, scale, pre, steps, "
                    "...) go inside `args`"
                )
            # The default rule takes every default arg; another rule takes the
            # ones it inherits. The group's own args win key by key.
            if g["rule"] == rule:
                base = self._default_args
            else:
                known = RULES[g["rule"]].inherits if g["rule"] in RULES else ()
                base = {k: v for k, v in self._default_args.items() if k in known}
            g["args"] = {**base, **(g["args"] or {})}
        # Metric arming, set per logging step by the container.
        self.need_to_calculate_norm = False
        self.norms_to_log: list[str] = list(NORM_FUNCTIONS.keys())
        self.gram_level = 0
        self.gram_scalar_names: list[str] = []
        self.gram_vector_names: list[str] = []
        self.track_spectrum = True
        self.log_metrics_locally = False
        self.norms_at_current_step: dict[str, torch.Tensor] = {}
        # LM-head geometry. `head_probe_h` is the head's input for a few probe
        # tokens; the container sets it for one norm step, step() drops it.
        self.head_probe_h: torch.Tensor | None = None
        self.head_diag_diameter_iters: int = 10
        self._head: dict[str, dict[str, torch.Tensor]] = {}
        self._zeros: dict[torch.device, torch.Tensor] = {}
        self._build()

    @property
    def has_head_geometry(self) -> bool:
        """Whether a unit is the LM head, whose geometry needs the probe."""
        return bool(self._head_keys)

    # ---------------------------------------------------------------- arming
    def calculate_norm_at_next_step(
        self,
        norms_to_log=None,
        gram_level=None,
        track_spectrum=None,
        log_metrics_locally=None,
    ):
        self.need_to_calculate_norm = True
        if norms_to_log is not None:
            self.norms_to_log = norms_to_log
        if gram_level is not None:
            self.gram_level = gram_level
            self.gram_scalar_names = gram_helper.gram_scalar_names(gram_level)
            self.gram_vector_names = gram_helper.gram_vector_names(gram_level)
        if track_spectrum is not None:
            self.track_spectrum = track_spectrum
        if log_metrics_locally is not None:
            self.log_metrics_locally = log_metrics_locally
        self.norms_at_current_step = {}

    def get_norms_at_current_step(self):
        """Hands the logging step's metrics over; nothing is kept after."""
        norms, self.norms_at_current_step = self.norms_at_current_step, {}
        return norms

    def _get_cached_zero_scalar(self, device):
        z = self._zeros.get(device)
        if z is None:
            z = self._zeros[device] = torch.zeros((), device=device)
        return z

    def state_dict(self):
        sd = super().state_dict()
        for g in sd["param_groups"]:
            for k in self._CONFIG_ONLY_GROUP_KEYS:
                g.pop(k, None)
        # Expert moments are views into their block's buffer, and DCP's staging
        # cannot copy a DTensor over a view at an offset: hand out copies.
        for i, st in sd["state"].items():
            buf = st.get("momentum_buffer")
            local = buf.to_local() if isinstance(buf, DTensor) else buf
            if local is not None and local.untyped_storage().nbytes() != local.nbytes:
                sd["state"][i] = {**st, "momentum_buffer": buf.clone()}
        # A non-expert unit's radial accumulators advance on its metric owner
        # only, and DCP saves one copy of a plain tensor: every rank saves the
        # owner's values.
        mesh = self._sink.mesh
        es = [e for e in self.entries if e.family is not Family.EXPERT]
        if mesh is not None and mesh.size() > 1 and es:
            names = list(RADIAL_ACCUMULATOR_NAMES)

            def owned(e):  # radial accumulators, then the rule state rows
                st = self.state[e.param]
                ts = [st["radial_state"][n] for n in names]
                return ts + ([st["rule_state"]] if "rule_state" in st else [])

            parts = []
            for e in es:
                for t in owned(e):
                    rows = t.reshape(t.shape[0] if t.ndim else 1, -1)  # a row per unit
                    mine = [(e.i, k) in self._radial_mine for k in range(len(rows))]
                    mine = torch.tensor(mine, device=t.device)[:, None]
                    parts.append(torch.where(mine, rows, 0).reshape(-1))
            flat = torch.cat(parts)
            dist.all_reduce(flat, group=mesh.get_group())
            index = {
                id(p): i
                for i, p in enumerate(p for g in self.param_groups for p in g["params"])
            }
            off = 0
            for e in es:
                got = []
                for t in owned(e):
                    got.append(flat[off : off + t.numel()].view(t.shape).clone())
                    off += t.numel()
                i = index[id(e.param)]
                sd["state"][i] = {
                    **sd["state"][i],
                    "radial_state": dict(zip(names, got[: len(names)])),
                    **(
                        {"rule_state": got[len(names)]} if len(got) > len(names) else {}
                    ),
                }
        return sd

    def load_state_dict(self, state_dict):
        """The run's config wins over the checkpoint for the rule keys."""
        keep = [
            {k: g[k] for k in self._CONFIG_ONLY_GROUP_KEYS} for g in self.param_groups
        ]
        super().load_state_dict(state_dict)
        for gi, (g, cfg) in enumerate(zip(self.param_groups, keep)):
            for k, v in cfg.items():
                if k in g and g[k] != v:
                    logger.warning(
                        f"[DiSCO] group {gi}: checkpoint {k}={g[k]!r} != config {v!r}; keeping config"
                    )
                g[k] = v
        self._init_state()
        self._plans()

    # ------------------------------------------------------------------ build
    def _init_state(self) -> None:
        for e in self.entries:
            p, st = e.param, self.state[e.param]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(p)
            if "radial_state" not in st:
                if e.family is Family.EXPERT and isinstance(p, DTensor):
                    st["radial_state"] = {
                        n: distribute_tensor(
                            torch.zeros(p.shape[0], device=p.device),
                            p.device_mesh,
                            p.placements,
                        )
                        for n in RADIAL_ACCUMULATOR_NAMES
                    }
                else:
                    # Experts: one slot per expert; fused params: per unit.
                    if e.family is Family.EXPERT:
                        shape = (p.shape[0],)
                    else:
                        shape = (len(e.units),) if len(e.units) > 1 else ()
                    st["radial_state"] = new_radial_state(p.device, shape=shape)
            if e.rule.has_state and "rule_state" not in st:
                # One row per unit, checkpointed; the first step sees zeros.
                shapes = {e.rule.compute.state_shape(*u.shape) for u in e.units}
                if len(shapes) > 1:
                    raise ValueError(
                        f"{e.clean}: its rule keeps state of shapes {sorted(shapes)}"
                        " across the units; one parameter's units need one"
                    )
                shape = e.rule.compute.state_shape(*e.units[0].shape)
                if e.family is Family.EXPERT and isinstance(p, DTensor):
                    st["rule_state"] = distribute_tensor(
                        torch.zeros(p.shape[0], *shape, device=p.device),
                        p.device_mesh,
                        p.placements,
                    )
                else:
                    st["rule_state"] = torch.zeros(
                        len(e.units), *shape, device=p.device
                    )

    def _build(self) -> None:
        pd = self.parallel_dims
        entries: list[_Entry] = []
        for gi, group in enumerate(self.param_groups):
            rule = rule_of(group)
            for name, p in zip(group["param_names"], group["params"]):
                if not p.requires_grad:
                    continue
                key = remove_orig_mod_and_weight_for_p_name(name, remove_weight=False)
                layout = self._param_layouts.get(key)
                if layout is None:
                    raise ValueError(
                        f"{name} is not declared: the model's param_layouts() must "
                        "state what every parameter is (models/common/param_layout.py)"
                    )
                fam = classify(name, p, layout, row_rule=rule.kind != "unit")
                _check_rule(name, p, layout, fam, rule)
                local = p.to_local() if isinstance(p, DTensor) else p
                e = _Entry(
                    len(entries),
                    name,
                    remove_orig_mod_and_weight_for_p_name(name),
                    p,
                    local,
                    gi,
                    rule,
                    fam,
                    matrix_view(p.shape),
                    layout.kind,
                )
                e.split = (
                    p.ndim == 1 and isinstance(p, DTensor) and bool(_sharded_axes(p))
                )
                held = (
                    RunList.contiguous(0, e.view[0])
                    if e.split or not isinstance(p, DTensor)
                    else local_rows(p, e.view)
                )
                for pos, spec in enumerate(resolve_units(name, p.shape, layout)):
                    e.units.append(
                        Unit(
                            -1,
                            -1,
                            p,
                            spec.name,
                            spec.rows,
                            tuple(spec.shape),
                            e.view,
                            (gi, rule),
                            pos,
                        )
                    )
                    e.unit_local.append(
                        held.intersect(spec.rows).positions_within(held)
                    )
                e.local_units = [k for k, r in enumerate(e.unit_local) if r.n_rows]
                if rule.kind == "gram" and any(
                    u.shape[0] < u.shape[-1] for u in e.units
                ):
                    raise ValueError(
                        f"{name}: a Gram compute needs tall units (rows >= columns); "
                        "a wide unit's [n, n] Gram is rank-deficient. Use muon or "
                        "left_polar for it."
                    )
                entries.append(e)
        self.entries = entries
        self._init_state()
        fam = {f: [e for e in entries if e.family is f] for f in Family}
        for f in (Family.SHARDED, Family.REPLICATED, Family.LOCAL):
            fam[f].sort(key=lambda e: e.param.numel(), reverse=True)
        fam[Family.EXPERT].sort(
            key=lambda e: (e.param.numel(), e.param.shape[1]), reverse=True
        )
        self.fam = fam
        cd = self.communication_dtype
        # -- sharded exchange: matrix units every step; local (row-rule) units of
        #    sharded parameters only on logging steps; weights for all of them.
        sh_entries = fam[Family.SHARDED] + [
            e
            for e in fam[Family.LOCAL]
            if e.param.ndim >= 2
            and isinstance(e.param, DTensor)
            and _sharded_axes(e.param)
        ]
        self._sh_entries, units = sh_entries, []
        for i, e in enumerate(sh_entries):
            e.sh = i
            for u in e.units:
                u.index, u.param_index = len(units), i
                units.append(u)
        self._ex_step = self._ex_row = self._ex_all = self._ex_w = None
        self._ex_vocab: list[Exchange] = []  # one per sharded embedding / head
        self._sh_units = units
        if units:
            axes = {
                (_sharded_axes(e.param)[0][0], len(_sharded_axes(e.param)))
                for e in sh_entries
            }
            if len(axes) != 1 or next(iter(axes))[1] != 1:
                raise ValueError(
                    f"sharded parameters must share exactly one sharded mesh axis, got {axes}"
                )
            p0 = sh_entries[0].param
            axis = _sharded_axes(p0)[0][0]
            mesh = fsdp_shard_mesh(pd)
            self._shard_group, rank, world = (
                p0.device_mesh.get_group(axis),
                p0.device_mesh.get_coordinate()[axis],
                p0.device_mesh.size(axis),
            )
            if mesh is not None and mesh.size() != world:
                raise ValueError(
                    f"sharded axis size {world} != fsdp mesh size {mesh.size()}"
                )
            owner = [u.index % world for u in units]
            kw = dict(
                axis=axis,
                group=self._shard_group,
                rank=rank,
                world=world,
                dtype=cd,
                owner=owner,
            )
            step_units = [u for u in units if self._is_matrix(u)]
            step_idx = {u.index for u in step_units}
            vocab = {
                u.index
                for u in units
                if sh_entries[u.param_index].kind in ("embedding", "unembedding")
            }
            row_units = [
                u for u in units if u.index not in step_idx and u.index not in vocab
            ]
            rest = [u for u in units if u.index not in vocab]
            w_units = [
                u for u in step_units if sh_entries[u.param_index].rule.needs_weight
            ]
            self._ex_step = (
                Exchange(step_units, **kw, returns=True) if step_units else None
            )
            self._ex_row = Exchange(row_units, **kw) if row_units else None
            self._ex_w = Exchange(w_units, **kw) if w_units else None
            self._ex_all = Exchange(rest, **kw) if rest else None
            # Gathered whole onto their owner on logging steps only, one at a
            # time, each straight into its arrival buffer.
            self._ex_vocab = [Exchange([u], **kw) for u in units if u.index in vocab]
            self._sh_owner, self._sh_world = owner, world
        else:
            self._shard_group = (
                fsdp_shard_mesh(pd).get_group()
                if fsdp_shard_mesh(pd) is not None
                else None
            )
        # -- replicated
        rep = fam[Family.REPLICATED]
        mesh = metrics_shard_mesh(pd)  # owners live in the metric sink's rank space
        w = mesh.size() if mesh is not None else 1
        if w > 1:  # snake over owner blocks: each rank gets big and small units
            blocks = [rep[i : i + w] for i in range(0, len(rep), w)]
            rep = [
                e for b, blk in enumerate(blocks) for e in (blk[::-1] if b % 2 else blk)
            ]
            fam[Family.REPLICATED] = rep
        units = []
        for i, e in enumerate(rep):
            e.rep = i
            for u in e.units:
                u.index, u.param_index = len(units), i
                units.append(u)
        self._rep = (
            Replicated(
                units,
                group=mesh.get_group() if mesh is not None else None,
                rank=mesh.get_local_rank() if mesh is not None else 0,
                world=w,
                dtype=cd,
            )
            if units
            else None
        )
        # -- experts: consecutive same-shape entries form one batched call
        self._blocks: list[list[_Entry]] = []
        for e in fam[Family.EXPERT]:
            if (
                self._blocks
                and tuple(self._blocks[-1][0].param.shape[1:])
                == tuple(e.param.shape[1:])
                and self._blocks[-1][0].group_idx == e.group_idx
            ):
                self._blocks[-1].append(e)
            else:
                self._blocks.append([e])
        self._big: dict[int, torch.Tensor] = {}
        self._u_sh: dict[int, torch.Tensor] = {}
        self._u_rep: dict[int, torch.Tensor] = {}
        # (entry, unit position) -> what the compute returned last step; on the
        # computing rank, not checkpointed.
        self._split_full: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._wsw = None
        # LM-head units in a fixed order, identical on every rank: the packed
        # collective in `_flush_head` needs all ranks to agree on the layout.
        self._head_keys = [
            _key(e, un)
            for e in self.entries
            if e.kind == "unembedding"
            for un in e.units
        ]
        self._sink = MetricSink(self)
        self._radial_mine = self._radial_owned_units()
        self._sink_ready = False
        self._plans()
        self._check_plan_agreement()
        logger.info(
            "[DiSCO] " + " | ".join(f"{f.value} {len(v)}" for f, v in fam.items())
        )

    def _radial_owned_units(self) -> set[tuple[int, int]]:
        """(entry, unit position) of the non-expert units whose radial
        accumulators this rank advances: the unit's metric owner (all in the
        metric sink's rank space)."""
        r, mine = self._sink.rank, set()
        for e in self._sh_entries:
            mine.update((e.i, u.pos) for u in e.units if self._sh_owner[u.index] == r)
        for li, e in enumerate(self.fam[Family.LOCAL]):
            if e.sh < 0 and li % self._sink.world == r:
                mine.update((e.i, u.pos) for u in e.units)
        if self._rep is not None:
            rep = self.fam[Family.REPLICATED]
            mine.update((rep[u.param_index].i, u.pos) for u in self._rep.owned)
        return mine

    def _release_logging_payloads(self) -> None:
        """The weights, moments and updates a logging step gathers on the owners
        are used once every `log_norm_freq` steps; do not keep their buffers
        (fp32 vocabulary matrices among them) between logging steps."""
        self._wsw = None
        if self._ex_all is not None:
            self._ex_all.release("w", torch.float32)
        if self._ex_row is not None:
            self._ex_row.release("graw")
            self._ex_row.release("u")

    def _check_plan_agreement(self) -> None:
        """Every rank of this model part must build the same plan, or the first
        exchange hangs instead of failing: compare a hash of it once, here."""
        mesh = self.parallel_dims.get_optional_mesh("loss")
        if mesh is None or mesh.size() == 1 or not self.entries:
            return
        plan = [
            (e.clean, tuple(e.param.shape), e.kind, e.family.value, e.rule)
            + tuple(u.rows.runs for u in e.units)
            for e in self.entries
        ]
        digest = hashlib.sha256(repr(plan).encode()).digest()
        mine = torch.tensor(
            [int.from_bytes(digest[:7], "little")],
            dtype=torch.int64,
            device=self.entries[0].param.device,
        )
        seen = [torch.empty_like(mine) for _ in range(mesh.size())]
        dist.all_gather(seen, mine, group=mesh.get_group())
        if any(int(x) != int(mine) for x in seen):
            raise RuntimeError(
                "DiSCO built different plans on the ranks of one model part "
                "(parameters, declared kinds, rules or units differ)"
            )

    def _is_matrix(self, u: Unit) -> bool:
        return self._sh_entries[u.param_index].family is Family.SHARDED

    def _plans(self) -> None:
        """Per-group lists for the foreach passes; rebuilt when state is replaced."""
        self._alias_expert_moments()
        # The travelling rows are the momentum itself (no pre phase, no
        # nesterov): the exchange may keep its views of them across steps.
        self._eff_is_moment = all(
            0.0 < self.param_groups[e.group_idx]["momentum"] < 1.0
            and not self.param_groups[e.group_idx]["nesterov"]
            and not e.rule.pre
            for e in self.fam[Family.SHARDED]
        )
        by_group: dict[int, list[_Entry]] = {}
        for e in self.entries:
            by_group.setdefault(e.group_idx, []).append(e)
        self._groups = [
            (gi, es, [self.state[e.param]["momentum_buffer"] for e in es])
            for gi, es in by_group.items()
        ]
        self._bufs_local = {
            e.i: (b.to_local() if isinstance(b, DTensor) else b)
            for _, es, bs in self._groups
            for e, b in zip(es, bs)
        }

    def _alias_expert_moments(self) -> None:
        """Each expert block's moments become views of one `[k * n_local, m, n]`
        buffer, which the block's batched rule call reads without a copy.
        Redone whenever the state was replaced (a checkpoint load)."""
        for bi, block in enumerate(self._blocks):
            bufs = [self.state[e.param]["momentum_buffer"] for e in block]
            locs = [b.to_local() if isinstance(b, DTensor) else b for b in bufs]
            n_local = locs[0].shape[0]
            big = self._big.get(bi)
            if n_local == 0 or (
                big is not None
                and all(
                    x.data_ptr() == big[j * n_local].data_ptr()
                    for j, x in enumerate(locs)
                )
            ):
                continue
            shape = (len(block) * n_local, *locs[0].shape[1:])
            if big is None or big.shape != shape:
                big = torch.empty(shape, dtype=locs[0].dtype, device=locs[0].device)
            for j, (e, b, x) in enumerate(zip(block, bufs, locs)):
                view = big[j * n_local : (j + 1) * n_local]
                view.copy_(x)
                self.state[e.param]["momentum_buffer"] = (
                    DTensor.from_local(
                        view,
                        b.device_mesh,
                        b.placements,
                        run_check=False,
                        shape=b.shape,
                        stride=b.stride(),
                    )
                    if isinstance(b, DTensor)
                    else view
                )
            self._big[bi] = big

    # ------------------------------------------------------------------- step
    @record_function("disco.step")
    @torch.no_grad()
    @compile_limits()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        logging_step = self.need_to_calculate_norm
        self._trace_scalar({"disco_logging_step": int(logging_step)})
        values: dict[str, UnitValues] | None = {} if logging_step else None
        self._head = {}
        self._wsw = None
        if logging_step and not self._sink_ready:
            self._register_metric_units()
        with self._span("disco_prepare_grads"):
            eff = self._effective_grads()
            self._phase_pass(eff, "pre")
        u: dict[int, torch.Tensor] = {}
        self._coef: dict[int, dict] = {}  # entry -> {unit position: (alpha, beta)}
        # The sharded units' exchanges run while the other families compute
        # and apply: out before the local and expert work, back before theirs.
        sent = self._send_sharded(eff, values)
        with self._span("disco_local"):
            self._step_local(eff, u)
        with self._span("disco_experts"):
            self._step_experts(eff, u, values)
        with self._span("disco_sharded"):
            back = self._step_sharded(eff, u, values, sent)
        with self._span("disco_replicated"):
            self._step_replicated(eff, u, values)
        self._phase_pass(u, "post")
        if values is not None:
            self._local_metrics(eff, u, values)
            self._flush_metrics(values)
            self._release_logging_payloads()
        self._apply(u, sharded=False)
        if back is not None:
            back()
        self._apply(u, sharded=True)
        for g in self.param_groups:
            g["t"] = int(g["t"]) + 1
        self.need_to_calculate_norm = False
        self.head_probe_h = None
        return loss

    def _ctx(self, e: _Entry) -> StepCtx:
        g = self.param_groups[e.group_idx]
        return StepCtx(float(g["lr"]), float(g["weight_decay"]), int(g["t"]) + 1)

    def _lr_wd(self, e: _Entry) -> tuple[float, float]:
        g = self.param_groups[e.group_idx]
        return float(g["lr"]), float(g["weight_decay"])

    def _effective_grads(self) -> dict[int, torch.Tensor]:
        """Momentum pass (one call per momentum value, all groups together),
        then each entry's effective gradient as fp32 rows `[nl, C]`."""
        groups, moments = [], {}
        for gi, es, _ in self._groups:
            g = self.param_groups[gi]
            m, nesterov = float(g["momentum"]), bool(g["nesterov"])
            grads = []
            for e in es:
                gr = e.param.grad
                if gr is None:
                    raise RuntimeError(f"[DiSCO] {e.name} has no gradient")
                grads.append(gr.to_local() if isinstance(gr, DTensor) else gr)
            groups.append((es, grads, m, nesterov))
            if 0.0 < m < 1.0:
                bufs, gs = moments.setdefault(m, ([], []))
                bufs += [self._bufs_local[e.i] for e in es]
                gs += grads
        for m, (bufs, gs) in moments.items():
            if bufs[0].is_cuda:
                with torch._dynamo.config.patch(
                    recompile_limit=128, cache_size_limit=128
                ):
                    _momentum_one_pass(bufs, gs, m)
            else:
                _momentum(bufs, gs, m)
        eff = {}
        for es, grads, m, nesterov in groups:
            out = grads
            if 0.0 < m < 1.0:
                out = [self._bufs_local[e.i] for e in es]
                if nesterov:
                    out = torch._foreach_lerp(out, grads, m)
            for e, t in zip(es, out):
                eff[e.i] = e.rows2d(t)
        return eff

    @staticmethod
    def _map_units(e: _Entry, x: torch.Tensor, fn) -> torch.Tensor:
        """`fn(rows, k, m)` on unit `k`'s rows of `x` (the rank's `[nl, C]` rows
        of `e`), `m` the unit's true row count. A new tensor: `x` may be the
        momentum buffer."""
        ul = e.unit_local
        if len(ul) == 1 and ul[0].runs == ((0, x.shape[0]),):
            return fn(x, 0, e.units[0].shape[0])
        out = torch.empty_like(x)
        for k, runs in enumerate(ul):
            if runs.n_rows:
                rows = unit_rows_local(x, runs)
                _write_rows(out, runs, fn(rows, k, e.units[k].shape[0]))
        return out

    def _phase_pass(self, t: dict[int, torch.Tensor], which: str) -> None:
        """Every rule's `pre` phases -- or, for the units computed on the rows a
        rank holds, the rest of its `post` -- on those rows, in order. A
        reducing phase takes its statistic over the whole unit: summed over the
        FSDP axis for units whose rows are spread (one packed all-reduce per
        round, for every unit that reached a reducing phase), locally otherwise."""
        todo = []
        for e in self.entries:
            if e.i not in t:
                continue
            if which == "pre":
                phases = e.rule.pre
            elif e.family is not Family.LOCAL or e.param.ndim <= 1:
                continue  # post ran with the compute, on the whole unit
            else:
                phases = e.rule.rest_tail  # the whole tail for Gram rules
            if phases:
                todo.append((e, list(phases)))
        while todo:
            pending = []  # (entry, unit position, partial statistic)
            for e, phases in todo:
                while phases and not (phases[0].reduce and e.sh >= 0):
                    ph = phases.pop(0)
                    t[e.i] = self._map_units(
                        e,
                        t[e.i],
                        lambda x, k, m, ph=ph: ph(
                            x, ph.stat(x) if ph.reduce else None, m
                        ),
                    )
                if phases:
                    ph = phases[0]
                    for k, runs in enumerate(e.unit_local):
                        rows = unit_rows_local(t[e.i], runs)  # [0, C] if none here
                        pending.append((e, k, ph.stat(rows)))  # fixed size per unit
            if pending:
                packed = torch.cat([s.reshape(-1) for _, _, s in pending])
                dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self._shard_group)
                reduced: dict[int, dict[int, torch.Tensor]] = {}
                off = 0
                for e, k, s in pending:
                    reduced.setdefault(e.i, {})[k] = packed[off : off + s.numel()].view(
                        s.shape
                    )
                    off += s.numel()
                for e, phases in todo:
                    if e.i in reduced:
                        ph, red = phases.pop(0), reduced[e.i]
                        t[e.i] = self._map_units(
                            e, t[e.i], lambda x, k, m, ph=ph, red=red: ph(x, red[k], m)
                        )
            todo = [(e, phases) for e, phases in todo if phases]

    def _run(self, rule: Rule, st, w, e: _Entry, pairs, *, lone_2d: bool) -> Result:
        """`rule` on the stack `st` of the units `pairs` [(entry, unit)], all
        whole here. The result's `update` is what the holders apply: the
        update, the coefficient update, or the step `W - W_next`. Keeps the
        coefficients in `self._coef` and the state in each unit's row of
        `state[p]["rule_state"]`."""
        state = None
        if rule.has_state:
            state = torch.stack([self._rule_row(ee, un) for ee, un in pairs])
        if lone_2d and rule.output == "update" and len(pairs) == 1:
            res = rule.run_whole(st[0], None, self._ctx(e))
            res.update = res.update.unsqueeze(0)
        else:
            res = rule.run_whole(st, w, self._ctx(e), state)
        if rule.has_state:
            for j, (ee, un) in enumerate(pairs):
                self._rule_row(ee, un).copy_(res.state[j])
        if rule.output == "update_coeffs":
            for j, (ee, un) in enumerate(pairs):
                self._coef.setdefault(ee.i, {})[un.pos] = _coefs_of(res, j)
        elif rule.output == "weight":
            res.update = w.float() - res.weight.float()  # rides the update's slot
        return res

    def _rule_row(self, e: _Entry, un: Unit) -> torch.Tensor:
        """`un`'s row of `e`'s rule state, as a view."""
        t = self.state[e.param]["rule_state"]
        if isinstance(t, DTensor):  # experts: the rows of the local experts
            return t.to_local()[e.local_units.index(un.pos)]
        return t[un.pos]

    # ---- families
    def _step_local(self, eff, u) -> None:
        """Rows and Gram rules on the rows each rank holds; vectors and scalars
        whole (a sharded vector's single row is assembled first)."""
        self._step_gram(eff, u)
        self._split_full = {}
        for e in self.fam[Family.LOCAL]:
            r = e.rule
            if r.kind == "gram":
                continue  # done above
            if (
                e.split
            ):  # one row over ranks: assemble, everyone applies, keep own slice
                p = e.param
                g_full = DTensor.from_local(
                    eff[e.i].reshape(-1),
                    p.device_mesh,
                    p.placements,
                    shape=p.shape,
                    stride=p.stride(),
                ).full_tensor()
                u_full = r.run_whole(g_full.view(1, -1)).update.view(-1)
                _, off = compute_local_shape_and_global_offset(
                    p.shape, p.device_mesh, p.placements
                )
                u[e.i] = u_full[off[0] : off[0] + e.local.numel()]
                self._split_full[e.i] = (g_full, u_full)
            elif e.param.ndim <= 1:
                u[e.i] = r.run_whole(eff[e.i]).update
            else:
                u[e.i] = self._map_units(
                    e, eff[e.i], lambda x, k, m, r=r: r.call_fused(x, m)
                )

    def _step_gram(self, eff, u) -> None:
        """Gram rules: partial `X^T X` per unit, one all-reduce for the units
        whose rows are spread over the FSDP axis, then `rows @ Q` locally."""
        pending = []  # (entry, unit position, partial Gram)
        for e in self.fam[Family.LOCAL]:
            if e.rule.kind != "gram":
                continue
            c = e.rule.compute
            out = torch.empty_like(eff[e.i])
            for k, runs in enumerate(e.unit_local):
                rows = unit_rows_local(eff[e.i], runs)
                if e.sh >= 0:
                    pending.append((e, k, c.gram(rows)))
                elif runs.n_rows:
                    m = e.units[k].shape[0]
                    _write_rows(out, runs, c.finish(rows, c.gram(rows), m))
            u[e.i] = out
        if not pending:
            return
        packed = torch.cat([part.reshape(-1) for _, _, part in pending])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self._shard_group)
        off = 0
        for e, k, part in pending:
            R = packed[off : off + part.numel()].view(part.shape)
            off += part.numel()
            runs = e.unit_local[k]
            if runs.n_rows:
                rows = unit_rows_local(eff[e.i], runs)
                _write_rows(
                    u[e.i], runs, e.rule.compute.finish(rows, R, e.units[k].shape[0])
                )

    def _step_experts(self, eff, u, values) -> None:
        metric = []
        for bi, block in enumerate(self._blocks):
            e0 = block[0]
            n_local = e0.local.shape[0]
            if n_local == 0:
                continue
            r = e0.rule
            k, m, n = len(block), e0.local.shape[1], e0.local.shape[2]
            big = self._big[bi]  # the block's moments
            if any(
                eff[e.i].data_ptr() != big[j * n_local].data_ptr()
                for j, e in enumerate(block)
            ):  # pre phases or nesterov made new tensors
                big = torch.cat([eff[e.i].view(n_local, m, n) for e in block])
            w = torch.cat([e.local for e in block]) if r.needs_weight else None
            pairs = [(e, e.units[pos]) for e in block for pos in e.local_units]
            res = self._run(r, big, w, e0, pairs, lone_2d=False)
            out = res.update.view(k, n_local, m, n)
            for j, e in enumerate(block):
                u[e.i] = out[j]
                if values is not None:
                    a, b = (
                        None if x is None else x.view(k, n_local)[j]
                        for x in (res.alpha, res.beta)
                    )
                    metric.append(
                        ExpertUnits(
                            e.i,
                            e.local,
                            big.view(k, n_local, m, n)[j],
                            out[j],
                            functools.partial(self._terms, e, e.local, out[j], a, b),
                            self._exact(e),
                            self._radial(e),
                        )
                    )
        if metric:
            per = self._sink.expert_block_values(metric)
            for x in metric:
                e = self.entries[x.key]
                for local_e, global_e in enumerate(self._expert_indices(e)):
                    values[f"ep_{global_e}/{e.clean}"] = per[x.key][local_e]

    def _send_sharded(self, eff, values):
        """Issue the sharded units' exchanges to their owners without waiting:
        the pre-update weights (fp32; every unit on a logging step, else the
        units whose rule needs them), then the effective gradients."""
        ex = self._ex_step
        if ex is None and (values is None or self._ex_all is None):
            return None
        es = self._sh_entries
        device = es[0].param.device
        w = None
        if values is not None and self._ex_all is not None:
            w = self._ex_all.to_owner(
                {e.sh: e.rows2d(e.local) for e in es},
                "w",
                device,
                torch.float32,
                async_op=True,
            )
        elif self._ex_w is not None:
            w = self._ex_w.to_owner(
                {e.sh: e.rows2d(e.local) for e in es if e.rule.needs_weight},
                "w",
                device,
                torch.float32,
                async_op=True,
            )
        g = None
        if ex is not None:
            g = ex.to_owner(
                {e.sh: eff[e.i] for e in es},
                "g",
                device,
                async_op=True,
                keep_views=self._eff_is_moment,
            )
        return w, g

    def _step_sharded(self, eff, u, values, sent):
        """The owners' compute on what `_send_sharded` sent. Returns the wait
        for the updates' way back to the holders (None if nothing travels)."""
        if sent is None:
            return None
        ex, es = self._ex_step, self._sh_entries
        wsw = sent[0]() if sent[0] is not None else None
        if values is not None:
            self._wsw = wsw
        if ex is None:
            return None
        ws = sent[1]()
        for us, st in zip(ex.batches, ws["stacks"]):
            e = es[us[0].param_index]
            pairs = [(es[un.param_index], un) for un in us]
            w = None
            if e.rule.needs_weight:
                w = torch.stack([wsw["full"][un.index] for un in us])
            res = self._run(e.rule, st, w, e, pairs, lone_2d=True)
            if values is not None:
                for j, (ee, un) in enumerate(pairs):
                    W = wsw["full"][un.index]
                    self._unit_metrics(
                        values, ee, un, W, st[j], res.update[j], *_coefs_of(res, j)
                    )
            st.copy_(res.update)
        if not self._u_sh:
            self._u_sh = self._holder_updates(ws["hold"], es, eff)
        back = ex.to_holders(
            "g",
            {e.sh: self._u_sh[e.i] for e in es if e.family is Family.SHARDED},
            async_op=True,
            keep_views=True,  # the holders' updates live across steps
        )
        for e in es:
            if e.family is Family.SHARDED:
                u[e.i] = self._u_sh[e.i]
        self._share_coefs(es, ex.units, self._shard_group)
        return back

    @staticmethod
    def _holder_updates(hold, es, eff) -> dict[int, torch.Tensor]:
        """Each sharded parameter's update rows: views of the step exchange's
        receive buffer when it is fp32 -- spent once the return all-to-all is
        waited, which is when the updates are written."""
        sh = [e for e in es if e.family is Family.SHARDED]
        if hold.dtype != torch.float32:
            return {e.i: torch.empty_like(eff[e.i]) for e in sh}
        assert sum(eff[e.i].numel() for e in sh) <= hold.numel()
        out, off = {}, 0
        for e in sh:
            n = eff[e.i].numel()
            out[e.i] = hold[off : off + n].view(eff[e.i].shape)
            off += n
        return out

    def _step_replicated(self, eff, u, values) -> None:
        rep = self._rep
        if rep is None:
            return
        es = self.fam[Family.REPLICATED]
        device = es[0].param.device
        ws = rep.inputs({e.rep: eff[e.i] for e in es}, device)
        for us, st in zip(rep.batches, ws["stacks"]):
            e = es[us[0].param_index]
            pairs = [(es[un.param_index], un) for un in us]
            w = None
            if e.rule.needs_weight or values is not None:
                w = torch.stack(
                    [unit_rows(ee.rows2d(ee.local), un) for ee, un in pairs]
                )
            res = self._run(
                e.rule, st, w if e.rule.needs_weight else None, e, pairs, lone_2d=True
            )
            if values is not None:
                for j, (ee, un) in enumerate(pairs):
                    self._unit_metrics(
                        values, ee, un, w[j], st[j], res.update[j], *_coefs_of(res, j)
                    )
            st.copy_(res.update)
        # A whole single-unit param reads its update straight from the gathered
        # buffer when that is fp32; the rest are copied out of it.
        copied = {}
        for e in es:
            un = e.units[0]
            if (
                ws["recv"].dtype == torch.float32
                and len(e.units) == 1
                and un.rows.runs == ((0, e.view[0]),)
            ):
                u[e.i] = ws["gathered"][un.index]
                continue
            if e.i not in self._u_rep:
                self._u_rep[e.i] = torch.empty_like(eff[e.i])
            u[e.i] = copied[e.rep] = self._u_rep[e.i]
        rep.all_gather(copied)
        self._share_coefs(es, rep.units, rep.group)

    def _share_coefs(self, es, units, group) -> None:
        """Owners computed each coefficient unit's alpha and beta; every holder
        of its rows needs them: one all-reduce, zeros from the other ranks."""
        cu = [un for un in units if es[un.param_index].rule.output == "update_coeffs"]
        if not cu:
            return
        buf = torch.zeros(2, len(cu), device=es[0].param.device)
        for i, un in enumerate(cu):
            got = self._coef.get(es[un.param_index].i, {}).get(un.pos)
            if got is not None:
                buf[0, i] = got[0]
                buf[1, i] = 1.0 if got[1] is None else got[1]
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(buf, group=group)
        for i, un in enumerate(cu):
            self._coef.setdefault(es[un.param_index].i, {})[un.pos] = (
                buf[0, i],
                buf[1, i],
            )

    def _apply(self, u: dict[int, torch.Tensor], *, sharded: bool) -> None:
        """`w <- (1 - lr wd) w - lr U`; per unit `w <- alpha w - beta U` for
        coefficient rules; `w <- w - (W - W_next)` for rules returning weights.
        For the sharded family, or for every other one."""
        for gi, es, _ in self._groups:
            es = [  # expert params with no local experts are not in u
                e for e in es if e.i in u and (e.family is Family.SHARDED) == sharded
            ]
            if not es:
                continue
            g = self.param_groups[gi]
            lr, wd = float(g["lr"]), float(g["weight_decay"])
            plain = [e for e in es if e.rule.output == "update"]
            if plain:
                locals_ = [e.local for e in plain]
                if wd != 0.0:
                    torch._foreach_mul_(locals_, 1.0 - wd * lr)
                torch._foreach_add_(
                    locals_, [u[e.i].view_as(e.local) for e in plain], alpha=-lr
                )
            for e in es:
                if e.rule.output != "update_coeffs":
                    continue
                rows, urows = e.rows2d(e.local), e.rows2d(u[e.i])
                for pos, (a, b) in self._coef.get(e.i, {}).items():
                    for st, n in e.unit_local[pos].runs:
                        rows[st : st + n].mul_(a)
                        step = urows[st : st + n]
                        rows[st : st + n].sub_(step if b is None else b * step)
            delta = [e for e in es if e.rule.output == "weight"]
            if delta:
                torch._foreach_sub_(
                    [e.local for e in delta], [u[e.i].view_as(e.local) for e in delta]
                )

    # ---------------------------------------------------------------- metrics
    def _radial(self, e: _Entry, un: Unit | None = None):
        """The accumulators `e`'s metrics advance; a fused param's unit gets
        views of its own slot."""
        st = self.state[e.param]["radial_state"]
        if e.family is Family.EXPERT or len(e.units) == 1:
            return st
        return {n: t[un.pos] for n, t in st.items()}

    def _terms(self, e: _Entry, W, U, alpha=None, beta=None):
        """For units `W` (one `[m, n]` or a stack) and what the holders apply,
        `U`: the step logged as the update, the weight the apply writes, and
        whether their difference is exactly that step."""
        lr, wd = self._lr_wd(e)
        out, exact = e.rule.output, self._exact(e)
        if out == "update":
            return -lr * U, pseudo_post_update_weight(W, U, lr, wd), exact
        U = U.float()
        if out == "weight":  # U is W - W_next
            return -U, W.float() - U, exact
        b = 1.0 if beta is None else _per_unit(beta, U)
        return -b * U, _per_unit(alpha, U) * W.float() - b * U, exact

    def _exact(self, e: _Entry) -> bool:
        """Whether the weight the apply writes minus `W` is the logged step."""
        out = e.rule.output
        return self._lr_wd(e)[1] == 0.0 if out == "update" else out == "weight"

    def _unit_metrics(self, values, e: _Entry, un: Unit, W, G, U, alpha, beta) -> None:
        """One unit's metrics on the rank that has it whole, after its post phases."""
        upd, w_after, exact = self._terms(e, W, U, alpha, beta)
        values[_key(e, un)] = self._sink.unit_values(
            W,
            G,
            upd,
            w_after,
            self._radial(e, un),
            exact=exact,
            transpose=e.kind == "embedding",
        )
        self._head_geometry(e, un, W, U)

    def _vector_metrics(self, values, e: _Entry, W, G, U) -> None:
        upd, w_after, exact = self._terms(e, W, U)
        values[e.clean] = self._sink.unit_values(
            W, G, upd, w_after, self._radial(e), exact=exact, gram=False
        )

    def _local_metrics(self, eff, u, values) -> None:
        """Metrics of the local family, after its post phases: units of sharded
        parameters on their owners (over the exchange), the rest whole here."""
        ex = self._ex_row
        if ex is not None:
            es = self._sh_entries
            device = es[0].param.device
            gs = ex.to_owner({e.sh: eff[e.i] for e in es}, "graw", device)
            us_ = ex.to_owner(
                {e.sh: u[e.i] for e in es if e.family is Family.LOCAL}, "u", device
            )
            for un in ex.owned:
                e = es[un.param_index]
                self._unit_metrics(
                    values,
                    e,
                    un,
                    self._wsw["full"][un.index],
                    gs["full"][un.index],
                    us_["full"][un.index],
                    None,
                    None,
                )
        for vx in self._ex_vocab:
            e = self._sh_entries[vx.units[0].param_index]
            device = e.param.device
            w = vx.to_owner({e.sh: e.rows2d(e.local)}, "w", device, torch.float32)
            g = vx.to_owner({e.sh: eff[e.i]}, "graw", device)
            u_ = vx.to_owner({e.sh: u[e.i]}, "u", device)
            for un in vx.owned:
                self._unit_metrics(
                    values,
                    e,
                    un,
                    w["full"][un.index],
                    g["full"][un.index],
                    u_["full"][un.index],
                    None,
                    None,
                )
            vx.release("w", torch.float32)
            vx.release("graw")
            vx.release("u")
        for li, e in enumerate(self.fam[Family.LOCAL]):
            if e.sh >= 0:
                continue
            mine = li % self._sink.world == self._sink.rank
            if e.split:
                W = e.param.full_tensor()  # a collective: every rank takes it
                if mine:
                    g_full, u_full = self._split_full[e.i]
                    self._vector_metrics(values, e, W, g_full, u_full)
            elif not mine:
                continue
            elif e.param.ndim <= 1:
                self._vector_metrics(
                    values, e, e.local.view(-1), eff[e.i].view(-1), u[e.i].view(-1)
                )
            else:
                w2d = e.rows2d(e.local)
                for un, runs in zip(e.units, e.unit_local):
                    self._unit_metrics(
                        values,
                        e,
                        un,
                        unit_rows_local(w2d, runs),
                        unit_rows_local(eff[e.i], runs),
                        unit_rows_local(u[e.i], runs),
                        None,
                        None,
                    )

    def _head_geometry(self, e: _Entry, un: Unit, W, U) -> None:
        """LM-head geometry for one unit, on the rank that has it whole.

        `W` is the whole PRE-update weight and `U` the whole direction the
        holders apply, after every phase of the rule. `h` is the one input the
        optimizer cannot produce itself.
        """
        if self.head_probe_h is None or e.kind != "unembedding":
            return
        if W.ndim != 2 or W.shape != U.shape:
            return
        self._head[_key(e, un)] = head_geometry(
            self.head_probe_h,
            U,
            W,
            self._lr_wd(e)[0],
            diameter_iters=self.head_diag_diameter_iters,
        )

    def _flush_head(self, final: dict, device: torch.device) -> None:
        """Move the head metrics from the unit's owner to the storing rank.

        They are plain scalars, not part of the sink's buffer, so they ride
        their own packed `[len(keys) * k]` all-reduce -- taken by every rank,
        on the same branch the sink's own gather uses.
        """
        keys = self._head_keys
        if not keys or self.head_probe_h is None:
            return
        sink, n = self._sink, len(HEAD_METRIC_NAMES)
        buf = torch.zeros(len(keys) * n, dtype=torch.float32, device=device)
        for i, k in enumerate(keys):
            vals = self._head.get(k)
            if vals is not None:
                buf[i * n : (i + 1) * n].copy_(
                    torch.stack([vals[m].float() for m in HEAD_METRIC_NAMES])
                )
        local = self.log_metrics_locally or sink.mesh is None or sink.world == 1
        if not local:
            dist.all_reduce(buf, group=sink.mesh.get_group())
        if not sink.stores_norms:
            return
        for i, k in enumerate(keys):
            # In local mode each storing rank keeps only its own units, exactly
            # as `MetricSink.flush` does.
            if local and k not in self._head:
                continue
            for j, m in enumerate(HEAD_METRIC_NAMES):
                final[f"head_{m}/{k}"] = buf[i * n + j]

    def _expert_indices(self, e: _Entry) -> list[int]:
        m = int(e.param.shape[1])
        return [
            s // m + i
            for s, n in local_rows(e.param, e.view).runs
            for i in range(n // m)
        ]

    def _register_metric_units(self) -> None:
        sink = self._sink
        if self._sh_units:
            if self._sh_world != sink.world:
                raise ValueError("sharded axis and metrics mesh disagree")
            for un in self._sh_units:
                sink.add(
                    _key(self._sh_entries[un.param_index], un),
                    un.shape,
                    self._sh_owner[un.index],
                )
        if self._rep is not None:
            for un in self._rep.units:
                sink.add(
                    _key(self.fam[Family.REPLICATED][un.param_index], un),
                    un.shape,
                    self._rep.owner[un.index],
                )
        for li, e in enumerate(self.fam[Family.LOCAL]):
            if e.sh >= 0:
                continue
            if e.param.ndim <= 1:
                sink.add(e.clean, (e.param.numel(),), li % sink.world)
            else:
                for un in e.units:
                    sink.add(_key(e, un), un.shape, li % sink.world)
        es = self.fam[Family.EXPERT]
        if es:
            mesh = es[0].param.device_mesh
            ranks = (
                [dist.get_rank()]
                if sink.mesh is None
                else [int(x) for x in sink.mesh.mesh.flatten().tolist()]
            )
            for r, g in enumerate(ranks):
                coord = [int(c) for c in (mesh.mesh == g).nonzero()[0].tolist()]
                for e in es:
                    m = int(e.param.shape[1])
                    for s0, n in rows_of_coordinate(
                        e.param.shape, e.view, e.param, coord
                    ).runs:
                        for i in range(n // m):
                            sink.add(
                                f"ep_{s0 // m + i}/{e.clean}", tuple(e.param.shape), r
                            )
        sink.finalize()
        self._sink_ready = True

    def _flush_metrics(self, values) -> None:
        sink = self._sink
        device = self.entries[0].param.device if self.entries else torch.device("cpu")
        final = sink.flush(values, device)
        if sink.stores_replicated_norms:
            for e in self.fam[Family.LOCAL]:
                if e.param.numel() == 1:
                    final[f"scalar_param_supremum/{e.clean}"] = e.local.abs()
        self._flush_head(final, device)
        if sink.stores_norms:
            self.norms_at_current_step.update(final)


def _coefs_of(res: Result, j: int):
    """Unit `j`'s (alpha, beta) from a coefficient result, else (None, None)."""
    return (
        None if res.alpha is None else res.alpha[j],
        None if res.beta is None else res.beta[j],
    )


def _key(e: _Entry, u: Unit) -> str:
    """A named unit logs under the name a split parameter would have had:
    `...qkv_linear.wqkv` + `wq` -> `...qkv_linear.wq`, `...w13` + `w1` -> `...w1`."""
    if not u.name:
        return e.clean
    head = e.clean.rsplit(".", 1)[0]
    return f"{head}.{u.name}" if head != e.clean else u.name


def unit_rows_local(rows2d: torch.Tensor, runs: RunList) -> torch.Tensor:
    r = runs.runs
    if not r:
        return rows2d[:0]  # the rank holds none of this unit's rows
    if len(r) == 1:
        a, n = r[0]
        return rows2d if (a, n) == (0, rows2d.shape[0]) else rows2d[a : a + n]
    return torch.cat([rows2d[a : a + n] for a, n in r], dim=0)


def _write_rows(dst: torch.Tensor, runs: RunList, src: torch.Tensor) -> None:
    off = 0
    for a, n in runs.runs:
        dst[a : a + n].copy_(src[off : off + n])
        off += n


__all__ = ["DiSCO", "ParamLayout"]
