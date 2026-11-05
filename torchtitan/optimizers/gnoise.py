# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradient-noise logging (`optimizer.gnoise`), off by default.

A small batch b is one (rank, microbatch); for a routed expert one (expert copy,
microbatch). Each FSDP param group's `post_backward` is wrapped per instance: at
its entry the group still holds the unsharded bf16 weight W the forward used and
this small batch's gradient g_b, before FSDP reshards and reduces. The reduced
gradient G is read before the clip. Formulas: gnoise_math.py; research mode:
gnoise_research.py; notes: optimizers/readme.md.
"""

import logging
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.utils import metrics_shard_rank, rank_owns_metrics_shard
from torchtitan.models.common.param_layout import by_rank
from torchtitan.optimizers import gnoise_math as gm
from torchtitan.optimizers.disco.layout import resolve_units
from torchtitan.optimizers.utils import remove_orig_mod_and_weight_for_p_name

logger = logging.getLogger(__name__)

_LAGS = (1, 2, 4, 8)  # lag correlations <O_t, O_{t-h}>
_LAYER = re.compile(r"layers\.\d+\.")
_EXPERT = re.compile(r"\.e\d+(?=\.|$)")
_RESIDUAL_WRITERS = re.compile(
    r"(^tok_embeddings\.weight$|\.attention\.wo\.weight$|\.feed_forward\.w2\.weight$"
    r"|\.moe\.(routed|shared)_experts\.w2\.weight$)"
)


@dataclass(kw_only=True, slots=True)
class GradNoiseConfig:
    enable: bool = False
    """Gradient-noise statistics on DiSCO's norm-logging steps (`log_norm_freq`)."""

    e1_freq: int = 0
    """0: on every norm-logging step. N: on steps divisible by N (a multiple of log_norm_freq)."""

    e1_first_and_last_steps: bool = True
    """Also on step 1 and on the last step."""

    ema_steps: int = 200
    """Time constant, in training steps, of the per-row / head / gain / type EMAs."""

    research_freq: int = 0
    """0: off. N: the research statistics on norm-logging steps divisible by N."""

    research_sizes: list[int] = field(default_factory=list)
    """Subset sizes in small batches; each must divide K and be <= K / 2. Empty: all."""

    research_matrices: str = ""
    """Regex (search) on the unit keys the research mode covers; empty: every unit."""

    research_topk: list[int] = field(default_factory=lambda: [1, 4, 16])
    """k of the top-k noise share (W_t's top-k singular directions)."""

    curvature_model_freq: int = 0
    """0: off. N: the same-batch model-level curvature every N steps (research runs)."""

    curvature_matrix_freq: int = 0
    """0: off. N: the same-batch per-matrix curvature every N steps (research runs)."""

    def validate(self, *, log_norm_freq: int) -> None:
        if not self.enable:
            if self != GradNoiseConfig():
                raise ValueError(
                    "optimizer.gnoise: knobs are set but gnoise.enable is False"
                )
            return
        if log_norm_freq <= 0:
            raise ValueError("optimizer.gnoise needs optimizer.log_norm_freq > 0")
        if self.ema_steps <= 0:
            raise ValueError("optimizer.gnoise.ema_steps must be > 0")
        if self.e1_freq < 0 or self.e1_freq % log_norm_freq:
            raise ValueError(
                f"optimizer.gnoise.e1_freq ({self.e1_freq}) must be 0 or a multiple of "
                f"log_norm_freq ({log_norm_freq})"
            )
        every = self.e1_freq or log_norm_freq
        d = GradNoiseConfig()
        knobs = (
            "research_sizes",
            "research_matrices",
            "research_topk",
            "curvature_model_freq",
            "curvature_matrix_freq",
        )
        if self.research_freq == 0 and any(
            getattr(self, k) != getattr(d, k) for k in knobs
        ):
            raise ValueError(
                "optimizer.gnoise: research knobs set with research_freq 0"
            )
        for name in ("research_freq", "curvature_model_freq", "curvature_matrix_freq"):
            v = getattr(self, name)
            if v < 0 or v % every:
                raise ValueError(
                    f"optimizer.gnoise.{name} ({v}) must be a multiple of the E1 interval "
                    f"({every}: e1_freq, or log_norm_freq when e1_freq is 0)"
                )
        if any(s < 1 for s in self.research_sizes):
            raise ValueError("optimizer.gnoise.research_sizes: every size >= 1")
        if any(k < 1 for k in self.research_topk):
            raise ValueError("optimizer.gnoise.research_topk: every k >= 1")
        if self.research_matrices:
            re.compile(self.research_matrices)


def refuse_unsupported(*, parallelism, training) -> None:
    """The setups gnoise does not cover; called at config time, only when enabled."""
    bad = []
    if parallelism.tensor_parallel_degree > 1:
        bad.append("tensor parallel > 1 (partial gradients are not small batches)")
    if parallelism.context_parallel_degree > 1:
        bad.append("context parallel > 1 (CP shards are correlated sequences)")
    if parallelism.pipeline_parallel_degree > 1:
        bad.append("pipeline parallel > 1 (one reduce per PP step)")
    if not training.disable_cuda_graphs:
        bad.append("CUDA graphs (a logging-step branch does not replay)")
    if training.enable_cpu_offload:
        bad.append("CPU offload (the unsharded weight is not on the device)")
    if torch._dynamo.config.compiled_autograd:
        bad.append("compiled autograd (FSDP's post_backward must run eagerly)")
    if bad:
        raise ValueError("optimizer.gnoise is not supported with: " + "; ".join(bad))


def _local(t: torch.Tensor) -> torch.Tensor:
    return t._local_tensor if isinstance(t, DTensor) else t


_CHUNK = 1 << 24  # elements per row chunk: fp64 temporaries stay near 128 MB


@torch.no_grad()
def row_stats(
    g: torch.Tensor, w: torch.Tensor, *, col_heads: int = 0, cols: bool = False
):
    """Per row of g, w [R, C] (any float dtype), in fp64 and row chunks: ||g_i||^2,
    <g_i, w_i>, ||w_i||^2; sum |g * w|; optionally the column sums of g * w and, per
    column head block, the sums of g^2 and g * w. fp64 because tr = (A - F) / (H - 1/N)
    cancels: where the signal dominates, fp32 sums leave percent-level noise in tr."""
    R, C = g.shape
    f32 = torch.float64
    q, dot, wn2 = (torch.empty(R, dtype=f32, device=g.device) for _ in range(3))
    absum = torch.zeros((), dtype=f32, device=g.device)
    colsum = torch.zeros(C, dtype=f32, device=g.device) if cols else None
    hb = torch.zeros(col_heads, dtype=f32, device=g.device) if col_heads else None
    hd = torch.zeros(col_heads, dtype=f32, device=g.device) if col_heads else None
    step = max(1, _CHUNK // max(C, 1))
    for a in range(0, R, step):
        gc, wc = g[a : a + step].to(f32), w[a : a + step].to(f32)
        gw = gc * wc
        q[a : a + step] = gc.square().sum(1)
        dot[a : a + step] = gw.sum(1)
        wn2[a : a + step] = wc.square().sum(1)
        absum += gw.abs().sum()
        if cols:
            colsum += gw.sum(0)
        if col_heads:
            hb += gc.view(gc.shape[0], col_heads, -1).square().sum((0, 2))
            hd += gw.view(gw.shape[0], col_heads, -1).sum((0, 2))
    return q, dot, wn2, absum, colsum, hb, hd


def type_key(key: str) -> str:
    return _EXPERT.sub(".e*", _LAYER.sub("layers.*.", key))


def _rows(runs) -> torch.Tensor:
    return torch.cat([torch.arange(a, a + n) for a, n in runs])


@dataclass
class _Param:
    i: int
    name: str
    kind: str
    fp: Any  # FSDPParam
    p: nn.Parameter
    R: int  # rows of the global [R, C] view
    C: int
    keys: list[str]  # unit keys (experts: global expert units, e-major)
    shapes: list[tuple[int, int]]
    row_unit: torch.Tensor  # [R] global unit of every global row
    head_rows: torch.Tensor | None = None  # rows inside a row head block
    head_ids: torch.Tensor | None = None  # their head-block id
    unit_rows: list[torch.Tensor] = field(default_factory=list)  # global rows per unit
    num_heads: int = 0  # row head blocks (q / k / v) or column head blocks (wo)
    col_heads: bool = False
    qk_units: list[tuple[str, int, int]] = field(
        default_factory=list
    )  # key, head lo, hi
    residual: bool = False
    ffn: tuple[str, str] | None = None  # (module prefix, "w3" | "w2"): channel gauge
    w3_rows: torch.Tensor | None = None
    # routed experts: the hook holds E_local experts, rows of `per` units each
    expert: bool = False
    moe: str = ""
    num_experts: int = 0
    local_experts: int = 0
    per: int = 1
    local_row_unit: torch.Tensor | None = None
    slots: dict[str, slice] = field(default_factory=dict)
    row_slice: slice | None = None

    @property
    def U(self) -> int:  # noqa: N802
        return len(self.keys)


class _Metrics(Mapping):
    """One step's metrics: scalars and vector families (keys[i] -> values[i]), each value
    with the param index that owns it (local logging). Read as a Mapping by tests."""

    def __init__(self) -> None:
        self.scalars: dict[str, tuple[Any, int]] = {}
        self.families: list[tuple[list[str], list[int] | int, torch.Tensor]] = []
        self._map: dict[str, tuple[int, Any]] | None = None

    def put(self, key: str, value, owner: int) -> None:
        self.scalars[key] = (value, owner)
        self._map = None

    def put_vec(
        self, keys: list[str], values: torch.Tensor, owners: list[int] | int
    ) -> None:
        assert len(keys) == values.numel(), (len(keys), values.shape)
        self.families.append((keys, owners, values.reshape(-1)))
        self._map = None

    def _index(self) -> dict[str, tuple[int, Any]]:
        if self._map is None:
            m = {}
            for fi, (keys, _, _) in enumerate(self.families):
                for i, k in enumerate(keys):
                    m[k] = (fi, i)
            for k in self.scalars:
                m[k] = (-1, k)
            self._map = m
        return self._map

    def __getitem__(self, key: str):
        fi, i = self._index()[key]
        return self.scalars[key][0] if fi < 0 else self.families[fi][2][i]

    def __iter__(self):
        return iter(self._index())

    def __len__(self) -> int:
        return len(self._index())

    def owners(self) -> dict[str, int]:
        out = {}
        for keys, owners, _ in self.families:
            for i, k in enumerate(keys):
                out[k] = owners if isinstance(owners, int) else owners[i]
        for k, (_, o) in self.scalars.items():
            out[k] = o
        return out


class GradNoise:
    """Per-step state of the gradient-noise statistics; built only when enabled."""

    def __init__(
        self, config: GradNoiseConfig, *, model_parts, parallel_dims, container
    ):
        self.config = config
        self.model_parts = model_parts
        self.pd = parallel_dims
        self.container = container
        mesh = (
            parallel_dims.get_optional_mesh("loss")
            if parallel_dims is not None
            else None
        )
        self.group = mesh.get_group() if mesh is not None else None
        self.world = mesh.size() if mesh is not None else 1
        self.rank = mesh.get_local_rank() if mesh is not None else 0
        self.dp_ranks = mesh.mesh.flatten().tolist() if mesh is not None else [0]
        self.armed = False
        self.research = False
        self.step = 0
        self._last_update = None
        self.ema: dict[str, torch.Tensor] = {}
        self.ema_reset_step = (
            None  # the first EMA update of this process (resume resets)
        )
        self.metrics = _Metrics()
        self.interval = config.e1_freq or max(
            1, int(getattr(container, "log_norm_freq", 1) or 1)
        )
        self._sel_cache: dict[tuple, tuple[list[int], torch.Tensor]] = {}
        self._saved = None  # W_t and G_t of a curvature step
        self._build()
        self._experts_topology()
        self._flat_layout()
        self._hidden_layout()
        from torchtitan.optimizers import gnoise_research

        self.research_engine = (
            gnoise_research.Research(self) if config.research_freq > 0 else None
        )
        self._install()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        from torch.distributed.fsdp import FSDPModule

        from torchtitan.models.opt_moe.norm_moe import NormMoE

        self.params: list[_Param] = []
        self.groups = []
        names, layouts, heads = {}, {}, {}
        self.moes: dict[str, nn.Module] = {}
        for part in self.model_parts:
            for n, p in part.named_parameters():
                names[id(p)] = remove_orig_mod_and_weight_for_p_name(
                    n, remove_weight=False
                )
            declare = getattr(part, "param_layouts", None)
            for k, v in (declare() if declare is not None else {}).items():
                layouts[
                    remove_orig_mod_and_weight_for_p_name(k, remove_weight=False)
                ] = v
            for mn, mod in part.named_modules():
                clean = remove_orig_mod_and_weight_for_p_name(mn, remove_weight=False)
                hd = getattr(mod, "head_dim", None)
                if isinstance(hd, int) and hd > 0:
                    heads[clean] = hd
                if isinstance(mod, NormMoE):
                    self.moes[clean] = mod
            for mod in part.modules():
                if isinstance(mod, FSDPModule):
                    for g in mod._get_fsdp_state()._fsdp_param_groups:
                        if all(g is not h for h in self.groups):
                            self.groups.append(g)
        if not self.groups:
            raise ValueError(
                "optimizer.gnoise needs the model wrapped by FSDP2 (fully_shard)"
            )
        # A batch-wise balance loss couples every token of a microbatch: the router's
        # gradient is no per-token sum there, so its noise is not estimated.
        skip = tuple(
            f"{k}.router."
            for k, mod in self.moes.items()
            if mod.aux_loss is not None and mod.aux_loss.loss_type == "batch_wise"
        )
        if skip:
            logger.info(
                f"[gnoise] routers left out (batch-wise balance loss): {list(skip)}"
            )
        for g in self.groups:
            for fp in g.fsdp_params:
                p = fp.sharded_param
                if p.requires_grad and not names[id(p)].startswith(skip or ("\x00",)):
                    self.params.append(
                        self._param(
                            len(self.params), names[id(p)], fp, p, layouts, heads
                        )
                    )
        self.by_fp = {id(pp.fp): pp for pp in self.params}
        self.moe_list = sorted(self.moes)
        self.moe_off, off = {}, 0
        for k in self.moe_list:
            self.moe_off[k] = off
            off += self.moes[k].num_experts
        self.n_moe_e = off
        self._layout_slots()

    def _param(self, i, name, fp, p, layouts, heads) -> _Param:
        shape = tuple(p.shape)
        layout = layouts.get(name) or by_rank(torch.Size(shape))
        kind = layout.kind
        dev = _local(p).device
        base = name.removesuffix(".weight")
        if kind in ("vector", "scalar"):
            R = max(1, math.prod(shape))
            pp = _Param(
                i,
                name,
                kind,
                fp,
                p,
                R,
                1,
                [base],
                [(R, 1)],
                torch.zeros(R, dtype=torch.long, device=dev),
            )
            pp.unit_rows = [torch.arange(R, device=dev)]
            return pp
        specs = resolve_units(name, torch.Size(shape), layout)
        R, C = math.prod(shape[:-1]), shape[-1]
        owner_mod = name.rsplit(".", 2)[0]  # e.g. layers.0.attention.qkv_linear
        hd = heads.get(owner_mod) or heads.get(owner_mod.rsplit(".", 1)[0])
        row_unit = torch.empty(R, dtype=torch.long)
        head_of_row = torch.full((R,), -1, dtype=torch.long)
        keys, shapes, qk, nheads = [], [], [], 0
        expert = kind == "expert_batch"
        for k, s in enumerate(specs):
            rows = _rows(s.rows.runs)
            row_unit[rows] = k
            if expert:
                e, _, sub = s.name.partition(".")
                keys.append(f"{base}.e{e}" + (f".{sub}" if sub else ""))
            else:
                keys.append(f"{base}.{s.name}" if s.name else base)
            shapes.append(tuple(s.shape))
            if (
                hd
                and not expert
                and s.name in ("wq", "wk", "wv")
                and s.shape[0] % hd == 0
            ):
                n = s.shape[0] // hd
                head_of_row[rows] = nheads + torch.arange(len(rows)) // hd
                if s.name in ("wq", "wk"):
                    qk.append((keys[-1], nheads, nheads + n))
                nheads += n
        pp = _Param(i, name, kind, fp, p, R, C, keys, shapes, row_unit.to(dev))
        pp.unit_rows = [
            (row_unit == k).nonzero().flatten().to(dev) for k in range(len(keys))
        ]
        if nheads:
            hr = (head_of_row >= 0).nonzero().flatten()
            pp.head_rows, pp.head_ids = hr.to(dev), head_of_row[hr].to(dev)
            pp.num_heads, pp.qk_units = nheads, qk
        elif hd and name.endswith(".wo.weight") and C % hd == 0:
            pp.num_heads, pp.col_heads = C // hd, True
        pp.residual = bool(_RESIDUAL_WRITERS.search(name))
        if not expert and (name.endswith(".w13.weight") or name.endswith(".w2.weight")):
            pp.ffn = (
                name.rsplit(".", 2)[0],
                "w3" if name.endswith(".w13.weight") else "w2",
            )
            if pp.ffn[1] == "w3":
                pp.w3_rows = _rows(specs[-1].rows.runs).to(dev)
        if expert:
            pp.expert = True
            pp.moe = name.split(".routed_experts.")[0]
            pp.num_experts = shape[0]
            pp.per = len(specs) // shape[0]
        return pp

    def _layout_slots(self) -> None:
        """Fixed offsets in the per-small-batch scalar vector and the per-row vectors."""
        off, roff = 0, 0

        def take(pp, key, n):
            nonlocal off
            pp.slots[key] = slice(off, off + n)
            off += n

        for pp in self.params:
            if pp.expert:
                continue  # sized once the local expert count is known
            take(pp, "t", pp.U)  # ||g_u||^2
            take(pp, "s", pp.U)  # <g_u, W_u>
            take(pp, "rr", pp.U)  # sum_i d_i^2, d_i = <g_i, w_i> / ||w_i||
            take(pp, "loo", 1)
            pp.row_slice = slice(roff, roff + pp.R)
            roff += pp.R
            if pp.num_heads:
                take(pp, "hb", pp.num_heads)
                take(pp, "hdot", pp.num_heads)
            if pp.ffn is not None:
                take(pp, "ffn", len(pp.w3_rows) if pp.ffn[1] == "w3" else pp.C)
        self._off = off
        self.n_rows = roff

    def _experts_topology(self) -> None:
        """Which dp positions hold a copy of each expert, and each position's EP group."""
        pd = self.pd
        ep = pd.ep if pd is not None and pd.ep_enabled else 1
        if ep > 1:
            sparse = pd.get_activated_mesh(["dp_replicate", "efsdp", "ep"])
            names = list(sparse.mesh_dim_names)
            coords = {}
            for idx in torch.cartesian_prod(
                *[torch.arange(s) for s in sparse.mesh.shape]
            ).reshape(-1, sparse.mesh.ndim):
                r = int(sparse.mesh[tuple(idx.tolist())])
                coords[r] = dict(zip(names, idx.tolist()))
            ep_of = [coords[r]["ep"] for r in self.dp_ranks]
            gid = [
                tuple(v for k, v in coords[r].items() if k != "ep")
                for r in self.dp_ranks
            ]
        else:
            ep_of = [0] * self.world
            gid = list(range(self.world))
        self.ep = ep
        self.ep_of = ep_of
        # M[p, q] = 1 if dp position q is in p's EP group (sums routed counts)
        self.ep_group = torch.tensor(
            [
                [1.0 if gid[q] == gid[p_] else 0.0 for q in range(self.world)]
                for p_ in range(self.world)
            ]
        )
        off = self._off
        for pp in self.params:
            if not pp.expert:
                continue
            pp.local_experts = pp.num_experts // ep
            m = pp.R // pp.num_experts
            e = torch.arange(pp.local_experts * m) // m
            sub_rows = m // pp.per
            pp.local_row_unit = (
                e * pp.per + (torch.arange(pp.local_experts * m) % m) // sub_rows
            ).to(pp.row_unit.device)
            UL = pp.local_experts * pp.per
            pp.slots["t"] = slice(off, off + UL)
            pp.slots["s"] = slice(off + UL, off + 2 * UL)
            off += 2 * UL
        self.slot_res = slice(off, off + 2)  # residual gauge: sum <g, W>, sum |g * W|
        self.n_slots = off + 2

    def _flat_layout(self) -> None:
        """Static index maps of the vectorised finalisation: the dense params' units, rows,
        heads and types laid end to end (param order), and the metric keys per family."""
        dev = self._device()
        D = [pp for pp in self.params if not pp.expert]
        self.dense = D
        keys, owners, ti, si, ri, ru, rp = [], [], [], [], [], [], []
        uo = 0
        for j, pp in enumerate(D):
            keys += pp.keys
            owners += [pp.i] * pp.U
            for name, acc in (("t", ti), ("s", si), ("rr", ri)):
                acc.append(torch.arange(pp.slots[name].start, pp.slots[name].stop))
            ru.append(pp.row_unit.cpu() + uo)
            rp.append(torch.full((pp.R,), j, dtype=torch.long))
            uo += pp.U
        cat = lambda xs: (torch.cat(xs) if xs else torch.zeros(0, dtype=torch.long)).to(
            dev
        )  # noqa: E731
        self.n_units = uo
        self.unit_owner = owners
        self.idx_t, self.idx_s, self.idx_rr = cat(ti), cat(si), cat(ri)
        self.row_unit_flat, self.row_dparam = cat(ru), cat(rp)
        self.unit_nrows = torch.bincount(self.row_unit_flat, minlength=uo).double()
        self.dense_pi = torch.tensor([pp.i for pp in D], dtype=torch.long, device=dev)
        self.dense_owner = [pp.i for pp in D]
        self.unit_keys = {
            c: [f"gnoise_{c}/{k}" for k in keys]
            for c in (
                "tr",
                "sig",
                "rad_tr",
                "rad_sig",
                "se_rad_tr",
                "se_rad_sig",
                "tan_tr",
                "tan_sig",
                "rrad_tr",
                "rrad_sig",
                "rtan_tr",
                "rtan_sig",
            )
        }
        for comp in ("rowbs", "rowradbs"):
            self.unit_keys[f"{comp}_unres"] = [f"gnoise_{comp}_unres/{k}" for k in keys]
            for q in ("q10", "q50", "q90"):
                self.unit_keys[f"{comp}_{q}"] = [f"gnoise_{comp}_{q}/{k}" for k in keys]
        self.param_keys = {
            c: [f"gnoise_{c}/{pp.keys[0]}" for pp in D]
            for c in ("sign_loo", "sign_gauss")
        }
        # types: dense units by index, routed-expert types (nansum per expert family) after them
        tid: dict[str, int] = {}
        self.unit_type = torch.tensor(
            [tid.setdefault(type_key(k), len(tid)) for k in keys],
            dtype=torch.long,
            device=dev,
        )
        self.n_dense_types = len(tid)
        for pp in self.params:
            if pp.expert:
                pp.etype = [
                    tid.setdefault(type_key(pp.keys[sub]), len(tid))
                    for sub in range(pp.per)
                ]
        self.type_keys = list(tid)
        # heads: q / k / v row blocks and wo column blocks, end to end
        hp = [pp for pp in D if pp.num_heads]
        self.head_params = hp
        hb, hd, hkeys, howner, seg, rh_rows, rh_heads = [], [], [], [], [], [], []
        qk_seg, self.qk_keys, self.qk_owner = [], [], []
        ho = 0
        for j, pp in enumerate(hp):
            pp.head_off = ho
            nh = pp.num_heads
            hb.append(torch.arange(pp.slots["hb"].start, pp.slots["hb"].stop))
            hd.append(torch.arange(pp.slots["hdot"].start, pp.slots["hdot"].stop))
            base = pp.name.removesuffix(".weight")
            hkeys += [f"{base}.h{h}" for h in range(nh)]
            howner += [pp.i] * nh
            seg.append(torch.full((nh,), j, dtype=torch.long))
            q = torch.full((nh,), -1, dtype=torch.long)
            if not pp.col_heads:
                rh_rows.append(pp.head_rows.cpu() + pp.row_slice.start)
                rh_heads.append(pp.head_ids.cpu() + ho)
                for key, lo, hi in pp.qk_units:
                    q[lo:hi] = len(self.qk_keys)
                    self.qk_keys.append(f"gnoise_gauge_qk/{key}")
                    self.qk_owner.append(pp.i)
            qk_seg.append(q)
            ho += nh
        self.n_heads = ho
        self.idx_hb, self.idx_hdot = cat(hb), cat(hd)
        self.head_seg, self.qk_seg = cat(seg), cat(qk_seg)
        self.rh_rows, self.rh_heads = cat(rh_rows), cat(rh_heads)
        self.head_owner = howner
        self.head_keys = {
            c: [f"gnoise_{c}/{k}" for k in hkeys] for c in ("hd_tr", "hd_sig")
        }
        self.hp_keys = {
            q: [f"gnoise_hdbs_{q}/{pp.name.removesuffix('.weight')}" for pp in hp]
            for q in ("q10", "q50", "q90")
        }
        self.hp_owner = [pp.i for pp in hp]
        nd = self.n_dense_types
        self.type_put_keys = {
            c: [f"gnoise_type_{c}/{k}" for k in self.type_keys] for c in ("tr", "sig")
        }
        self.type_put_keys.update(
            {
                c: [f"gnoise_type_{c}/{k}" for k in self.type_keys[:nd]]
                for c in ("rad_tr", "rad_sig")
            }
        )
        self.type_put_keys["bs"] = [f"gnoise_type_bs/{k}" for k in self.type_keys]
        self.type_put_keys["bs_rad"] = [
            f"gnoise_type_bs_rad/{k}" for k in self.type_keys[:nd]
        ]

    def _hidden_layout(self) -> None:
        """The hidden matrices (layout kind matrix, no experts) of the update metrics:
        Psi on every E1 step, the lag correlations on the E1 steps divisible by
        `lag_every`, the smallest multiple of the E1 interval >= 8 (so at most four
        stored update copies)."""
        H = [pp for pp in self.params if not pp.expert and pp.kind == "matrix"]
        self.hidden = {id(pp.p): pp for pp in H}
        self.h_off, off = {}, 0
        keys, owners = [], []
        for pp in H:
            self.h_off[pp.i] = off
            off += pp.U
            keys += pp.keys
            owners += [pp.i] * pp.U
        self.n_hidden = off
        self.h_keys, self.h_owner = keys, owners
        self.lag_every = self.interval * math.ceil(8 / self.interval)
        self.dirs: dict[
            int, dict[int, torch.Tensor]
        ] = {}  # param -> step -> local rows of O
        self._observed: dict[int, tuple[list, list[int]]] = {}
        self.psi_due = False
        self.cur_step = 0
        logger.info(
            f"[gnoise] Psi on every E1 step, lag correlations h in {list(_LAGS)} on steps "
            f"divisible by {self.lag_every} ({len(H)} hidden matrices)"
        )

    def note_step(self, step: int) -> None:
        """Every training step, before the forward (the lag copies need every step)."""
        self.cur_step, self.psi_due = step, False

    def _stores(self, step: int) -> bool:
        return any((step + h) % self.lag_every == 0 for h in _LAGS)

    @torch.no_grad()
    def on_update(self, disco, u) -> None:
        """DiSCO's `update_observer`: u[e.i] is the update it applied to entry e this step
        (W <- decay W - lr u, alpha W - u, or W - u: the rule's own step). Psi = <O, G> /
        (|O| |G|) per hidden unit with G the reduced gradient DiSCO read (after the clip:
        a common scalar, so Psi is the pre-clip one), and <O_t, O_{t-h}> / (|O_t| |O_{t-h}|);
        sums of local rows, one all_reduce (replicated rows count rep times in every sum,
        which the ratios cancel)."""
        step = self.cur_step
        mine = self._observed.get(id(disco))
        if mine is None:  # static: this instance's hidden params and their unit indices
            pps = [
                self.hidden[id(e.param)]
                for e in disco.entries
                if id(e.param) in self.hidden
            ]
            idx = [self.h_off[pp.i] + k for pp in pps for k in range(pp.U)]
            mine = self._observed[id(disco)] = (pps, idx)
        pps, idx = mine
        if self.research_engine is not None and self.research_engine.psi_pending:
            self.research_engine.psi(self.metrics, disco, u)
        if not pps:
            return
        lag = self.psi_due and step % self.lag_every == 0
        ents = [
            (e, self.hidden[id(e.param)])
            for e in disco.entries
            if e.i in u and id(e.param) in self.hidden
        ]
        if self.psi_due:
            dev = self._device()
            sums = torch.zeros(
                self.n_hidden, 3 + 3 * len(_LAGS), dtype=torch.float64, device=dev
            )
            for e, pp in ents:
                O = e.rows2d(u[e.i]).double()
                G = e.rows2d(_local(e.param.grad)).double()
                f = [(O * G).sum(1), (O * O).sum(1), (G * G).sum(1)]
                for h in _LAGS:
                    P = self.dirs.get(pp.i, {}).get(step - h) if lag else None
                    if P is None:
                        f += [torch.zeros_like(f[0])] * 3
                    else:
                        P = P.double()
                        f += [(O * P).sum(1), (P * P).sum(1), torch.ones_like(f[0])]
                ru = disco._row_unit(e) + self.h_off[pp.i]
                sums.index_add_(0, ru, torch.stack(f, dim=1))
            if self.group is not None:
                dist.all_reduce(sums, group=self.group)
            it = torch.tensor(idx, dtype=torch.long, device=dev)
            sums = sums[it]
            keys = [self.h_keys[i] for i in idx]
            owners = [self.h_owner[i] for i in idx]
            psi = sums[:, 0] / (sums[:, 1] * sums[:, 2]).sqrt()
            self.metrics.put_vec([f"gnoise_psi/{k}" for k in keys], psi, owners)
            if lag:
                for j, h in enumerate(_LAGS):
                    num, p2, seen = (
                        sums[:, 3 + 3 * j],
                        sums[:, 4 + 3 * j],
                        sums[:, 5 + 3 * j],
                    )
                    c = num / (sums[:, 1] * p2).sqrt()
                    self.metrics.put_vec(
                        [f"gnoise_lag{h}/{k}" for k in keys],
                        torch.where(seen > 0, c, math.nan),
                        owners,
                    )
        store = self._stores(step)
        for (
            e,
            pp,
        ) in ents:  # at most four copies per param: drop what no later lag step reads
            d = self.dirs.setdefault(pp.i, {})
            for s0 in [
                s0
                for s0 in d
                if all(s0 + h <= step for h in _LAGS if (s0 + h) % self.lag_every == 0)
            ]:
                del d[s0]
            if store:
                d[step] = e.rows2d(u[e.i]).detach().clone()

    def due(self, step: int, *, first_or_last: bool) -> bool:
        """E1 on this norm-logging step: every `e1_freq`-th (0: every one), plus
        step 1 and the last step (`first_or_last`) unless `e1_first_and_last_steps`
        is off."""
        return step % self.interval == 0 or (
            self.config.e1_first_and_last_steps and first_or_last
        )

    def _install(self) -> None:
        run = torch._dynamo.disable(self._on_post_backward)
        for g in self.groups:
            orig = g.post_backward

            def wrapped(*args, _g=g, _orig=orig, **kwargs):
                run(_g)
                return _orig(*args, **kwargs)

            g.post_backward = wrapped
        for o in getattr(self.container, "optimizers", ()):
            if hasattr(o, "update_observer"):  # DiSCO
                o.update_observer = self.on_update
        self.head_groups = {
            id(g)
            for g in self.groups
            for fp in g.fsdp_params
            if id(fp) in self.by_fp and self.by_fp[id(fp)].kind == "unembedding"
        }
        logger.info(
            f"[gnoise] on: {len(self.params)} params in {len(self.groups)} FSDP groups, "
            f"dp world {self.world}, ep {self.ep}"
        )

    def validate_layout(self, *, accum: int) -> None:
        """K = dp world x accumulation small batches per step; needs K >= 2."""
        if self.world * accum < 2:
            raise ValueError(
                "optimizer.gnoise needs at least 2 small batches per step (dp ranks "
                f"x gradient accumulation), got {self.world} x {accum}"
            )
        if self.research_engine is not None:
            self.research_engine.set_layout(accum)

    # ------------------------------------------------------------- per step
    def arm(self, step: int, *, research: bool) -> None:
        self.armed, self.research, self.step = True, research, step
        self.psi_due = True
        cfg = self.config
        self.curv_model = (
            cfg.curvature_model_freq > 0 and step % cfg.curvature_model_freq == 0
        )
        self.curv_matrix = (
            cfg.curvature_matrix_freq > 0 and step % cfg.curvature_matrix_freq == 0
        )
        self._saved = None
        self.mb = -1
        self.n_tokens: list[int] = []
        self.n_den = None
        self.scal: list[torch.Tensor] = []
        self.d_rows: list[torch.Tensor] = []
        self.a_rows = None
        self.records: list[dict[int, int]] = []
        self.moe_snap: dict[str, torch.Tensor] = {}
        self.moe_delta: list[torch.Tensor] = []
        self.w_rownorm: dict[int, torch.Tensor] = {}
        if self.research_engine is not None and research:
            self.research_engine.begin()

    def set_tokens(self, global_valid_tokens: torch.Tensor) -> None:
        if self.armed:
            self.n_den = global_valid_tokens

    def begin_microbatch(self, num_valid_tokens: int) -> None:
        if not self.armed:
            return
        self._close_moe_delta()
        self.mb += 1
        self.n_tokens.append(int(num_valid_tokens))
        dev = self._device()
        self.scal.append(torch.zeros(self.n_slots, dtype=torch.float64, device=dev))
        self.d_rows.append(torch.zeros(self.n_rows, dtype=torch.float64, device=dev))
        if self.a_rows is None:
            self.a_rows = torch.zeros(2, self.n_rows, dtype=torch.float64, device=dev)
        self.records.append(defaultdict(int))
        for k in self.moe_list:
            self.moe_snap[k] = self.moes[k].tokens_per_expert_E.detach().clone()

    def _close_moe_delta(self) -> None:
        if self.mb < 0 or not self.moe_list:
            return
        self.moe_delta.append(
            torch.cat(
                [
                    self.moes[k].tokens_per_expert_E.detach() - self.moe_snap[k]
                    for k in self.moe_list
                ]
            ).double()
        )

    def _device(self) -> torch.device:
        return _local(self.params[0].p).device

    # ------------------------------------------------------------------ hook
    def _on_post_backward(self, g) -> None:
        if not self.armed:
            return
        todo = []
        for fp in g.fsdp_params:
            pp = self.by_fp.get(id(fp))
            if pp is None or getattr(fp, "_unsharded_param", None) is None:
                continue
            if (
                fp.unsharded_accumulated_grad is not None
                or fp.unsharded_param.grad is not None
            ):
                todo.append((pp, fp))
        if not todo:
            return
        if not g.reduce_grads:
            if id(g) in self.head_groups:
                return  # the chunked loss's chunks 0..N-2: read at the reducing call
            raise RuntimeError(
                "[gnoise] a FSDP group skipped its gradient reduce (no-sync "
                "accumulation): gnoise needs one reduce per microbatch"
            )
        if not g.is_unsharded:
            raise RuntimeError(
                f"[gnoise] {g._module_fqn}: post_backward entered with the group "
                "sharded (unshard_in_backward off?); the unsharded weight is needed"
            )
        if self.n_den is None or self.mb < 0:
            raise RuntimeError(
                "[gnoise] an armed step without set_tokens / begin_microbatch"
            )
        n_mb = self.n_tokens[self.mb]  # a microbatch with no valid token is left out
        c = (self.n_den.double() / n_mb) ** 2 if n_mb > 0 else 0.0
        rec = self.records[self.mb]
        for pp, fp in todo:
            fp.accumulate_unsharded_grad_if_needed()  # FSDP's next op; idempotent
            grad = fp.unsharded_accumulated_grad
            if grad is None:
                grad = fp.unsharded_param.grad
            grad, W = _local(grad).detach(), _local(fp.unsharded_param).detach()
            self._accumulate(pp, grad, W, c)
            rec[pp.i] += 1
            if self.research_engine is not None and self.research:
                self.research_engine.snapshot(pp, grad, W, self.mb)

    @torch.no_grad()
    def _accumulate(self, pp: _Param, grad: torch.Tensor, W: torch.Tensor, c) -> None:
        s = self.scal[self.mb]
        rows = grad.numel() // pp.C
        q, dot, wn2, absum, colsum, hb, hd = row_stats(
            grad.reshape(rows, pp.C),
            W.reshape(rows, pp.C),
            col_heads=pp.num_heads if pp.col_heads else 0,
            cols=pp.ffn is not None and pp.ffn[1] == "w2",
        )
        wn = self.w_rownorm.get(pp.i)
        if wn is None:
            wn = self.w_rownorm[pp.i] = wn2.sqrt()
        if pp.expert:
            UL = pp.local_experts * pp.per
            z = torch.zeros(UL, dtype=s.dtype, device=s.device)
            s[pp.slots["t"]] = z.index_add(0, pp.local_row_unit, q)
            s[pp.slots["s"]] = z.index_add(0, pp.local_row_unit, dot)
        else:
            z = torch.zeros(pp.U, dtype=s.dtype, device=s.device)
            s[pp.slots["t"]] = z.index_add(0, pp.row_unit, q)
            s[pp.slots["s"]] = z.index_add(0, pp.row_unit, dot)
            d = torch.where(wn > 0, dot / wn.clamp_min(1e-300), 0.0)
            s[pp.slots["rr"]] = z.index_add(0, pp.row_unit, d * d)
            self.d_rows[self.mb][pp.row_slice] = d
            self.a_rows[0, pp.row_slice] += c * q
            self.a_rows[1, pp.row_slice] += c * (d * d)
        if pp.num_heads and not pp.col_heads:
            z = torch.zeros(pp.num_heads, dtype=s.dtype, device=s.device)
            s[pp.slots["hb"]] = z.index_add(0, pp.head_ids, q[pp.head_rows])
            s[pp.slots["hdot"]] = z.index_add(0, pp.head_ids, dot[pp.head_rows])
        elif pp.col_heads:
            s[pp.slots["hb"]] = hb
            s[pp.slots["hdot"]] = hd
        if pp.ffn is not None:
            s[pp.slots["ffn"]] = dot[pp.w3_rows] if pp.ffn[1] == "w3" else colsum
        if pp.residual:
            s[self.slot_res] += torch.stack([dot.sum(), absum])

    # -------------------------------------------------------------- finalize
    @torch.no_grad()
    def before_clip(self) -> None:
        """Read G (before the clip), reduce, estimate; disarms the step."""
        if not self.armed:
            return
        self._close_moe_delta()
        accum = self.mb + 1
        for k, rec in enumerate(self.records):
            bad = [
                (pp.name, rec.get(pp.i, 0))
                for pp in self.params
                if rec.get(pp.i, 0) != 1
            ]
            if bad:
                raise RuntimeError(
                    f"[gnoise] microbatch {k}: {len(bad)} params not recorded exactly "
                    f"once: {bad[:4]}"
                )
        dev = self._device()
        if self.curv_model or self.curv_matrix:  # W_t and the pre-clip G_t, sharded
            self._saved = [
                (_local(pp.p.detach()).clone(), _local(pp.p.grad).detach().clone())
                for pp in self.params
            ]
        feats = self._read_reduced()
        # every dense param's global rows end to end: ||G_i||^2, <G_i, w_i>, ||w_i||^2
        Fr = (
            torch.cat([feats[pp.i][:, :3] for pp in self.dense])
            if self.dense
            else (torch.zeros(0, 3, dtype=torch.float64, device=dev))
        )
        Dr = torch.where(
            Fr[:, 2] > 0, Fr[:, 1] / Fr[:, 2].clamp_min(1e-300).sqrt(), 0.0
        )
        if self.group is not None:
            dist.all_reduce(self.a_rows, group=self.group)
        loo = self._loo(Dr)
        n_local = torch.tensor(self.n_tokens, dtype=torch.float64, device=dev)
        moe = (
            torch.stack(self.moe_delta)
            if self.moe_list
            else torch.zeros(accum, 0, device=dev)
        )
        packed = torch.cat([torch.stack(self.scal), n_local[:, None], moe, loo], dim=1)
        if self.group is not None:
            out = torch.empty(
                self.world * accum, packed.shape[1], dtype=packed.dtype, device=dev
            )
            dist.all_gather_into_tensor(out, packed.contiguous(), group=self.group)
            packed = out
        o = self.n_slots
        scal, n_b = packed[:, :o].double(), packed[:, o].double()
        moe_d = packed[:, o + 1 : o + 1 + self.n_moe_e].double()
        loo_b = packed[:, o + 1 + self.n_moe_e :].double()
        res = _Metrics()
        self._estimate(res, scal, n_b, moe_d, loo_b, feats, Fr, Dr, accum)
        if self.research_engine is not None and self.research:
            self.research_engine.finish(res, n_b, moe_d, accum)
        self.metrics = res
        self.armed = False
        self.scal, self.d_rows, self.a_rows = [], [], None

    def _read_reduced(self) -> dict[int, torch.Tensor]:
        """Per global row of every param, on every rank: ||G_i||^2, <G_i, w_i>, ||w_i||^2
        (+ per column head: ||G_i,h||^2, <G_i,h, w_i,h>), w = the master cast to FSDP's
        param_dtype, i.e. the compute copy the hook's projections used."""
        out = {}
        for pp in self.params:
            G = pp.p.grad
            if G is None:
                raise RuntimeError(f"[gnoise] {pp.name} has no reduced gradient")
            Wl = _local(pp.p.detach())
            if (
                pp.fp.param_dtype is not None
            ):  # FSDP2's cast: the forward's values exactly
                Wl = Wl.to(pp.fp.param_dtype)
            Gl = _local(G)
            rows = Gl.reshape(-1, 1) if pp.p.ndim <= 1 else Gl.reshape(-1, pp.C)
            q, dot, wn2, *_ = row_stats(rows, Wl.reshape(rows.shape))
            f = [q, dot, wn2]
            if pp.col_heads:  # per row and column head: small matrices, kept whole
                gh = rows.double().view(rows.shape[0], pp.num_heads, -1)
                w = (
                    Wl.double()
                    .reshape(rows.shape)
                    .view(rows.shape[0], pp.num_heads, -1)
                )
                f += list(gh.square().sum(2).unbind(1))
                f += list((gh * w).sum(2).unbind(1))
            out[pp.i] = self._full_rows(pp, G, torch.stack(f, dim=-1))
        return out

    def _loo(self, Dr) -> torch.Tensor:
        """[accum, n_params]: the leave-one-out sign agreement of each dense param's row
        radials (0 for experts), from the rows of every param at once."""
        dev = self._device()
        out = torch.zeros(
            self.mb + 1, len(self.params), dtype=torch.float64, device=dev
        )
        if not self.dense:
            return out
        z = torch.zeros(len(self.dense), dtype=torch.float64, device=dev)
        for k in range(self.mb + 1):
            agree, used = gm.loo_sign_terms(self.d_rows[k].double(), Dr)
            cnt = z.index_add(0, self.row_dparam, used.double())
            hit = z.index_add(0, self.row_dparam, agree.double())
            out[k, self.dense_pi] = torch.where(
                cnt > 0, hit / cnt.clamp_min(1.0), math.nan
            )
        return out

    # -------------------------------------------------------------- estimate
    def _put(self, res, key: str, value, owner: int) -> None:
        res.put(key, value, owner)

    def _estimate(self, res, scal, n_b, moe_d, loo_b, feats, Fr, Dr, accum) -> None:
        n_den = self.n_den.double()
        valid = n_b > 0
        K = (
            valid.sum().double()
        )  # small batches with tokens (per-row sums leave the rest out)
        N = n_b.sum()
        H = torch.where(valid, 1.0 / n_b.clamp_min(1.0), 0.0).sum() / K.clamp_min(1.0)
        res.put("gnoise/K", K, 0)
        res.put("gnoise/N", N, 0)
        res.put("gnoise/N_check", N - n_den, 0)
        res.put("gnoise/n_spread", n_b.max() - n_b.min(), 0)
        dev = scal.device
        uk, uo, ru = self.unit_keys, self.unit_owner, self.row_unit_flat
        # every dense unit at once: [units, K] small-batch sums against [units] reduced sums
        z = torch.zeros(self.n_units, dtype=torch.float64, device=dev)
        Fq = z.index_add(0, ru, Fr[:, 0])
        Fs = z.index_add(0, ru, Fr[:, 1])
        fro = z.index_add(0, ru, Fr[:, 2]).sqrt()
        tot = gm.estimate(scal[:, self.idx_t].T, Fq, n_b, n_den)
        r_b = scal[:, self.idx_s].T / fro.clamp_min(1e-300)[:, None]
        Rg = Fs / fro.clamp_min(1e-300)
        rad = gm.estimate(r_b * r_b, Rg * Rg, n_b, n_den, cross=Rg[:, None] * r_b)
        RR = z.index_add(0, ru, Dr * Dr)
        rrad = gm.estimate(scal[:, self.idx_rr].T, RR, n_b, n_den)
        comps = {
            "tr": tot["tr"],
            "sig": tot["sig"],
            "rad_tr": rad["tr"],
            "rad_sig": rad["sig"],
            "se_rad_tr": rad["se_tr"],
            "se_rad_sig": rad["se_sig"],
            "tan_tr": tot["tr"] - rad["tr"],
            "tan_sig": tot["sig"] - rad["sig"],
            "rrad_tr": rrad["tr"],
            "rrad_sig": rrad["sig"],
            "rtan_tr": tot["tr"] - rrad["tr"],
            "rtan_sig": tot["sig"] - rrad["sig"],
        }
        for comp, v in comps.items():
            res.put_vec(uk[comp], v, uo)
        nd, nt = self.n_dense_types, len(self.type_keys)
        zt = torch.zeros(nd, dtype=torch.float64, device=dev)
        t_tr, t_sig = zt.index_add(0, self.unit_type, tot["tr"]), zt.index_add(
            0, self.unit_type, tot["sig"]
        )
        t_rtr, t_rsig = zt.index_add(0, self.unit_type, rad["tr"]), zt.index_add(
            0, self.unit_type, rad["sig"]
        )
        e_tr = torch.zeros(nt - nd, dtype=torch.float64, device=dev)
        e_sig = torch.zeros(nt - nd, dtype=torch.float64, device=dev)
        for pp in self.params:
            if pp.expert:
                f = feats[pp.i]
                ze = torch.zeros(pp.U, dtype=torch.float64, device=dev)
                fq, fs = ze.index_add(0, pp.row_unit, f[:, 0]), ze.index_add(
                    0, pp.row_unit, f[:, 1]
                )
                fr = ze.index_add(0, pp.row_unit, f[:, 2]).sqrt()
                self._expert(
                    res, pp, scal, n_b, moe_d, accum, fq, fs, fr, n_den, e_tr, e_sig
                )
        res.put_vec(
            self.param_keys["sign_loo"],
            loo_b[:, self.dense_pi].nanmean(0),
            self.dense_owner,
        )
        rows = gm.from_sums(self.a_rows[0], Fr[:, 0], K, H, N, n_den)
        rows += gm.from_sums(self.a_rows[1], Dr * Dr, K, H, N, n_den)
        self._row_flat = (
            rows  # tr, sig, radial tr, radial sig per dense row (EMA inputs)
        )
        ema_in = dict(zip(("row/tr", "row/sig", "row/rtr", "row/rsig"), rows))
        self._heads(res, scal, n_b, n_den, feats, Fr, ema_in)
        self._gauges(res, scal)
        tk = self.type_put_keys
        ema_in["type/tr"], ema_in["type/sig"] = torch.cat([t_tr, e_tr]), torch.cat(
            [t_sig, e_sig]
        )
        ema_in["type/rad_tr"], ema_in["type/rad_sig"] = t_rtr, t_rsig
        for comp in ("tr", "sig", "rad_tr", "rad_sig"):
            res.put_vec(tk[comp], ema_in[f"type/{comp}"], 0)
        self._emas(res, ema_in, n_b)

    @property
    def rows(self) -> dict[str, torch.Tensor]:
        """The last step's per-row estimates of each dense param (tests)."""
        out = {}
        for pp in self.dense:
            for name, v in zip(("tr", "sig", "rtr", "rsig"), self._row_flat):
                out[f"row/{pp.i}/{name}"] = v[pp.row_slice]
        return out

    def _heads(self, res, scal, n_b, n_den, feats, Fr, ema_in) -> None:
        if not self.n_heads:
            return
        dev = scal.device
        zh = torch.zeros(self.n_heads, dtype=torch.float64, device=dev)
        Fh = zh.index_add(0, self.rh_heads, Fr[self.rh_rows, 0])
        for pp in self.head_params:
            if pp.col_heads:
                nh = pp.num_heads
                Fh[pp.head_off : pp.head_off + nh] = feats[pp.i][:, 3 : 3 + nh].sum(0)
        hb = scal[:, self.idx_hb]  # [K, heads]
        est = gm.estimate(hb.T, Fh, n_b, n_den)
        res.put_vec(self.head_keys["hd_tr"], est["tr"], self.head_owner)
        res.put_vec(self.head_keys["hd_sig"], est["sig"], self.head_owner)
        ema_in["head/tr"], ema_in["head/sig"] = est["tr"], est["sig"]
        if self.qk_keys:  # head-block radial of q / k: 0 under QK-norm (a check)
            wh = zh.index_add(0, self.rh_heads, Fr[self.rh_rows, 2]).sqrt()
            hdot = scal[:, self.idx_hdot]
            rel = (
                hdot.abs() / (hb.clamp_min(0).sqrt() * wh[None]).clamp_min(1e-300)
            ).amax(0)
            sel = self.qk_seg >= 0
            seg, v = self.qk_seg[sel], rel[sel]
            nq = len(self.qk_keys)
            mx = torch.full(
                (nq,), -math.inf, dtype=torch.float64, device=dev
            ).scatter_reduce(0, seg, v, "amax")
            nan = (
                torch.zeros(nq, dtype=torch.float64, device=dev).index_add(
                    0, seg, v.isnan().double()
                )
                > 0
            )
            res.put_vec(self.qk_keys, torch.where(nan, math.nan, mx), self.qk_owner)

    def _gauges(self, res, scal) -> None:
        pairs = defaultdict(dict)
        for pp in self.params:
            if pp.ffn is not None:
                pairs[pp.ffn[0]][pp.ffn[1]] = (scal[:, pp.slots["ffn"]], pp.i)
        for prefix, d in pairs.items():
            if "w3" in d and "w2" in d:
                a, b = d["w3"][0], d["w2"][0]
                rel = (a - b).abs().sum(1) / (a.abs() + b.abs()).sum(1).clamp_min(
                    1e-300
                )
                res.put(f"gnoise_gauge_ffn/{prefix}", rel.max(), d["w2"][1])
        rs = scal[:, self.slot_res]
        res.put(
            "gnoise_gauge_resid/max",
            (rs[:, 0].abs() / rs[:, 1].clamp_min(1e-300)).max(),
            0,
        )

    def expert_counts(self, moe_d, accum) -> torch.Tensor:
        """[world, accum, sum_E]: tokens each position's expert copies processed per
        microbatch (the routed counts summed over its EP group)."""
        d = moe_d.reshape(self.world, accum, -1)
        return torch.einsum("pq,qak->pak", self.ep_group.to(d), d)

    def expert_rows(self, pp: _Param, accum: int) -> tuple[torch.Tensor, torch.Tensor]:
        """For each global expert e: the gather rows of its copies' small batches and the
        local expert index there. Returns (rows [E, Kc * accum], local [E, Kc * accum])."""
        rows, local = [], []
        for e in range(pp.num_experts):
            c = e // pp.local_experts
            pos = [q for q in range(self.world) if self.ep_of[q] == c]
            rows.append([q * accum + k for q in pos for k in range(accum)])
            local.append([e - c * pp.local_experts] * (len(pos) * accum))
        return torch.tensor(rows), torch.tensor(local)

    def _expert(
        self, res, pp, scal, n_b, moe_d, accum, Fq, Fs, fro, n_den, e_tr, e_sig
    ) -> None:
        counts = self.expert_counts(moe_d, accum).reshape(self.world * accum, -1)
        rows, local = self.expert_rows(pp, accum)  # [E, Ke]
        rows, local = rows.to(scal.device), local.to(scal.device)
        off = self.moe_off[pp.moe]
        e_idx = torch.arange(pp.num_experts, device=scal.device)[:, None].expand_as(
            rows
        )
        n_e = counts[rows, off + e_idx]  # [E, Ke]
        top_k = self.moes[pp.moe].router.top_k
        if not hasattr(pp, "ex_keys"):
            pp.ex_keys = [
                {
                    c: [
                        f"gnoise_{c}/{pp.keys[e * pp.per + sub]}"
                        for e in range(pp.num_experts)
                    ]
                    for c in ("tr", "sig", "rad_tr", "rad_sig", "ntok", "nb")
                }
                for sub in range(pp.per)
            ]
        for sub in range(pp.per):
            u_host = [e * pp.per + sub for e in range(pp.num_experts)]
            u = torch.tensor(u_host, device=scal.device)  # global units of this sub
            cols = pp.slots["t"].start + local * pp.per + sub
            t_b = scal[rows, cols]
            s_b = scal[rows, cols + (pp.slots["s"].start - pp.slots["t"].start)]
            tot = gm.estimate(t_b, Fq[u], n_e, n_den)
            r_b = s_b / fro[u].clamp_min(1e-300)[:, None]
            Rg = Fs[u] / fro[u].clamp_min(1e-300)
            rad = gm.estimate(r_b * r_b, Rg * Rg, n_e, n_den, cross=Rg[:, None] * r_b)
            ek = pp.ex_keys[sub]
            for comp, v in (
                ("tr", tot["tr"]),
                ("sig", tot["sig"]),
                ("rad_tr", rad["tr"]),
                ("rad_sig", rad["sig"]),
                ("ntok", tot["N"]),
                ("nb", tot["K"]),
            ):
                res.put_vec(ek[comp], v, pp.i)
            # B_simple per expert in routed tokens, and in total tokens (x N / N_e)
            ok = torch.isfinite(tot["tr"]) & (tot["sig"] > 0)
            bs = torch.where(ok, tot["tr"] / tot["sig"].clamp_min(1e-300), math.nan)
            lkey = _EXPERT.sub("", pp.keys[u_host[0]])
            res.put(
                f"gnoise_ex_unest/{lkey}", 1.0 - (tot["K"] >= 2).double().mean(), pp.i
            )
            res.put(f"gnoise_ex_bs_q50/{lkey}", bs.nanmedian(), pp.i)
            res.put(
                f"gnoise_ex_bs_total_q50/{lkey}",
                (bs * n_b.sum() / tot["N"]).nanmedian(),
                pp.i,
            )
            # rank-deficient small batches: fewer routed tokens than min(m, n)
            m_, n_ = pp.shapes[u_host[0]]
            res.put(
                f"gnoise_ex_lowtok/{lkey}",
                ((n_e > 0) & (n_e < min(m_, n_))).double().mean(),
                pp.i,
            )
            t = pp.etype[sub] - self.n_dense_types
            e_tr[t] += tot["tr"].nansum()
            e_sig[t] += tot["sig"].nansum()
        if pp.name.endswith("w2.weight"):  # one count check per MoE layer
            d = moe_d[:, off : off + pp.num_experts].sum(1)
            ok = n_b > 0
            ratio = torch.where(
                ok, d / (top_k * n_b.clamp_min(1.0)), 0.0
            ).sum() / ok.sum().clamp_min(1)
            res.put(f"gnoise_moe_count_check/{pp.moe}", ratio, pp.i)

    def _emas(self, res, ema_in, n_b) -> None:
        step = self.step
        if self._last_update is None:
            self.ema_reset_step = step
            dt = float(self.config.ema_steps)
        else:
            dt = float(step - self._last_update)
        self._last_update = step
        res.put("gnoise/ema_reset_step", torch.tensor(float(self.ema_reset_step)), 0)
        out = gm.ema_update(self.ema, ema_in, dt, float(self.config.ema_steps))
        n_mean, N = n_b.mean(), n_b.sum()
        dev = n_b.device
        q3 = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=dev)
        uk, uo, ru = self.unit_keys, self.unit_owner, self.row_unit_flat
        if self.dense:
            z = torch.zeros(self.n_units, dtype=torch.float64, device=dev)
            for comp, a, b in (("rowbs", "tr", "sig"), ("rowradbs", "rtr", "rsig")):
                tr, sig, se = out[f"row/{a}"], out[f"row/{b}"], out[f"row/{b}/se"]
                ok = torch.isfinite(sig) & (sig > 2 * se)
                ratio = torch.where(ok, tr / sig.clamp_min(1e-300), math.nan)
                res.put_vec(
                    uk[f"{comp}_unres"],
                    1.0 - z.index_add(0, ru, ok.double()) / self.unit_nrows,
                    uo,
                )
                q = gm.segment_nanquantile(ratio, ru, self.n_units, q3)
                for i, name in enumerate(("q10", "q50", "q90")):
                    res.put_vec(uk[f"{comp}_{name}"], q[i], uo)
            g = gm.gaussian_sign_agreement(out["row/rsig"], out["row/rtr"], n_mean, N)
            good = ~torch.isnan(g)
            zp = torch.zeros(len(self.dense), dtype=torch.float64, device=dev)
            num = zp.index_add(0, self.row_dparam, torch.where(good, g, 0.0))
            res.put_vec(
                self.param_keys["sign_gauss"],
                num / zp.index_add(0, self.row_dparam, good.double()),
                self.dense_owner,
            )
        if self.n_heads:
            q = gm.segment_nanquantile(
                out["head/tr"] / out["head/sig"],
                self.head_seg,
                len(self.head_params),
                q3,
            )
            for i, name in enumerate(("q10", "q50", "q90")):
                res.put_vec(self.hp_keys[name], q[i], self.hp_owner)
        tk = self.type_put_keys
        res.put_vec(tk["bs"], out["type/tr"] / out["type/sig"], 0)
        res.put_vec(tk["bs_rad"], out["type/rad_tr"] / out["type/rad_sig"], 0)

    # ------------------------------------------------------------- curvature
    @property
    def curvature_due(self) -> bool:
        return self._saved is not None

    @torch.no_grad()
    def curvature(self, engine, microbatch_groups, n_den, local_loss) -> None:
        """Same-batch curvature along the step Delta = W_{t+1} - W_t, after the optimizer
        step: model level c = 2 (L(W_{t+1}) - L(W_t) - <G_t, Delta>) / ||Delta||^2 from one
        forward; per unit c_k = <Delta_k, G'_k - G_k> / ||Delta_k||^2 from a second forward
        and backward (G' at W_{t+1}). Buffers (MoE counts, aux-loss accumulators) and RNG
        are restored and the gradients cleared, so training is unchanged."""
        from torch.distributed.fsdp import FSDPModule

        saved, self._saved = self._saved, None
        model = engine.model_parts[0]
        buffers = [(b, b.detach().clone()) for b in model.buffers()]
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        dev = self._device()
        rep = (
            self.pd.dp_replicate
            if self.pd is not None and self.pd.dp_replicate_enabled
            else 1
        )
        dots = torch.zeros(
            2, dtype=torch.float64, device=dev
        )  # <G, Delta>, ||Delta||^2
        for pp, (W0, G0) in zip(self.params, saved):
            delta = _local(pp.p.detach()).double() - W0.double()
            dots += torch.stack([(G0.double() * delta).sum(), delta.square().sum()])
        res = self.metrics
        if self.curv_model:
            L1 = self._batch_loss(engine, microbatch_groups, n_den, backward=False)
            for m in model.modules():
                if isinstance(m, FSDPModule):
                    m.reshard()  # a no-grad forward leaves [norm, lm_head] gathered
            L0 = local_loss.detach().double().reshape(1)
            packed = torch.cat([dots, L0, L1.reshape(1)])
            if self.group is not None:
                dist.all_reduce(packed, group=self.group)
            gd, dd = packed[0] / rep, packed[1] / rep
            res.put("gnoise_curv/model", 2.0 * (packed[3] - packed[2] - gd) / dd, 0)
            res.put("gnoise_curv/step_norm", dd.sqrt(), 0)
        if self.curv_matrix:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                self._batch_loss(engine, microbatch_groups, n_den, backward=True)
            for pp, (W0, G0) in zip(self.params, saved):
                if pp.expert:
                    continue
                delta = _local(pp.p.detach()).double() - W0.double()
                g1 = _local(pp.p.grad).double() - G0.double()
                vec = pp.p.ndim <= 1
                d2 = delta.reshape(-1, 1) if vec else delta.reshape(-1, pp.C)
                h2 = g1.reshape(d2.shape)
                loc = torch.stack([(d2 * h2).sum(1), d2.square().sum(1)], dim=-1)
                full = self._full_rows(pp, pp.p.grad, loc)
                z = torch.zeros(pp.U, dtype=torch.float64, device=dev)
                num = z.index_add(0, pp.row_unit, full[:, 0])
                den = z.index_add(0, pp.row_unit, full[:, 1])
                res.put_vec([f"gnoise_curv/{k}" for k in pp.keys], num / den, pp.i)
            model.zero_grad(set_to_none=True)
        for b, v in buffers:
            b.copy_(v)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)

    def _batch_loss(
        self, engine, microbatch_groups, n_den, *, backward: bool
    ) -> torch.Tensor:
        """The step's batch through the engine's own preprocessing and loss, bypassing
        its counters; returns this rank's sum of loss / N."""
        import spmd_types as spmd

        from torchtitan.distributed import utils as dist_utils

        model = engine.model_parts[0]
        total = torch.zeros((), dtype=torch.float64, device=self._device())
        for group in microbatch_groups:
            for mb in group:
                d = mb.to_input_dict(engine.device, non_blocking=True)
                inputs, labels, extra = model.preprocess_inputs(
                    d,
                    parallel_dims=engine.parallel_dims,
                    parallelism=engine.config.parallelism,
                    max_num_documents=engine.max_num_documents,
                    max_context_length=engine.config.training.max_context_length,
                    **engine.preprocess_inputs_kwargs,
                )
                with dist_utils.get_spmd_context(
                    parallel_dims=engine.parallel_dims,
                    spmd_typechecking=engine.config.debug.spmd_typechecking,
                ):
                    pred = model(inputs, **extra)
                    loss, _ = engine.loss_fn(
                        pred,
                        labels,
                        **mb.to_loss_kwargs(engine.device, non_blocking=True),
                        global_valid_tokens=n_den,
                    )
                    if backward:
                        with spmd.no_typecheck():
                            loss.backward()
                total += _local(loss.detach()).double().sum()
        return total

    def _full_rows(self, pp, like, loc) -> torch.Tensor:
        """Per-row features `loc` of a param's local rows, gathered to every global row."""
        nf = loc.shape[-1]
        if not isinstance(like, DTensor):
            return loc.reshape(pp.R, nf).double()
        Gl = _local(like)
        vec = pp.p.ndim <= 1
        lshape = (Gl.shape[0], nf) if vec else tuple(Gl.shape[:-1]) + (nf,)
        gshape = (pp.p.shape[0], nf) if vec else tuple(pp.p.shape[:-1]) + (nf,)
        full = DTensor.from_local(
            loc.reshape(lshape),
            like.device_mesh,
            like.placements,
            run_check=False,
            shape=torch.Size(gshape),
            stride=torch.empty(gshape, device="meta").stride(),
        ).full_tensor()
        return full.reshape(pp.R, nf).double()

    # --------------------------------------------------------------- collect
    def collect(self) -> dict[str, float]:
        """This rank's share of the last armed step's metrics, as floats (one sync)."""
        m, self.metrics = self.metrics, _Metrics()
        if not m.scalars and not m.families:
            return {}
        if self.container.log_metrics_locally:
            if not rank_owns_metrics_shard(self.pd):
                return {}
            mrank, mworld = metrics_shard_rank(self.pd)
        elif self.rank == 0:
            mrank, mworld = 0, 1
        else:
            return {}
        dev = self._device()
        keys, parts = [], []
        sk = [k for k, (_, o) in m.scalars.items() if o % mworld == mrank]
        if sk:
            keys += sk
            parts.append(
                torch.stack(
                    [
                        torch.as_tensor(
                            m.scalars[k][0], dtype=torch.float64, device=dev
                        ).reshape(())
                        for k in sk
                    ]
                )
            )
        for fkeys, owners, vals in m.families:
            if isinstance(owners, int):
                if owners % mworld == mrank:
                    keys += fkeys
                    parts.append(vals.double())
                continue
            hit = self._sel_cache.get((id(owners), mworld, mrank))
            if (
                hit is None or hit[2] is not owners
            ):  # owner lists are static: one build each
                idx = [i for i, o in enumerate(owners) if o % mworld == mrank]
                hit = (idx, torch.tensor(idx, dtype=torch.long, device=dev), owners)
                self._sel_cache[(id(owners), mworld, mrank)] = hit
            idx, it, _ = hit
            if len(idx) == len(owners):
                keys += fkeys
                parts.append(vals.double())
            elif idx:
                keys += [fkeys[i] for i in idx]
                parts.append(vals.double()[it])
        if not keys:
            return {}
        return dict(zip(keys, torch.cat(parts).tolist()))
