# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Online power averages of the model weights, renormalised per unit in the
norm a rule holds, scored by the built-in validator; plus weight snapshots
and the scoring of saved weight files (post-hoc mixtures)."""

import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.optim import Optimizer

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable

from .optimizer import OptimizersContainer

__all__ = [
    "WeightAverages",
    "mix_snapshots",
    "power_weights",
    "renorm_rows",
    "RENORM_MODES",
]

logger = logging.getLogger(__name__)

RENORM_MODES = ("none", "frobenius", "spectral", "rows")


def power_weights(steps: Sequence[int], gamma: float) -> list[float]:
    """The power average's weights over the weights after `steps`: s^gamma / sum."""
    w = [float(s) ** gamma for s in steps]
    tot = math.fsum(w)
    return [x / tot for x in w]


def _unit_norm(x: torch.Tensor, mode: str) -> torch.Tensor:
    """In float64: CUDA's float32 spectral norm is off by ~1e-4 (measured)."""
    x = x.double()
    if mode == "frobenius":
        return x.norm()
    return torch.linalg.matrix_norm(x, ord=2)  # spectral


def renorm_rows(
    full_avg: torch.Tensor,
    full_ref: torch.Tensor,
    units: Sequence[torch.Tensor],
    mode: str,
) -> torch.Tensor:
    """`full_avg` ([R, C]) with each unit (a set of row indices) scaled to the
    norm of the same unit of `full_ref`, Frobenius or spectral. A zero unit is
    left as it is."""
    out = full_avg.clone()
    for idx in units:
        a, r = full_avg[idx], full_ref[idx]
        na, nr = _unit_norm(a, mode), _unit_norm(r, mode)
        out[idx] = a * torch.where(na > 0, nr / na, torch.ones_like(na)).to(a.dtype)
    return out


def _rows_to_lengths(avg: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Each row (last dim) of `avg` scaled to its length in `lengths` (..., 1)."""
    n = avg.norm(dim=-1, keepdim=True)
    return avg * torch.where(n > 0, lengths / n, torch.ones_like(n))


def mix_snapshots(
    blobs: Sequence[dict], weights: Sequence[float]
) -> dict[str, torch.Tensor]:
    """sum_k weights[k] * snapshot k, renormalised as the run's averages are:
    frobenius / spectral units to the last snapshot's unit norms (the held
    radius), rows to the same mixture of the row lengths."""
    if len(blobs) != len(weights) or not blobs:
        raise ValueError("mix_snapshots: one weight per snapshot")
    last = blobs[-1]
    out = {}
    for name, ref in last["weights"].items():
        xs = [b["weights"][name].double() for b in blobs]
        mix = sum(w * x for w, x in zip(weights, xs))
        mode = last["modes"].get(name, "none")
        if mode in ("frobenius", "spectral"):
            units = [torch.tensor(u) for u in last["units"][name]]
            c = mix.shape[-1]
            mix = renorm_rows(
                mix.reshape(-1, c), ref.double().reshape(-1, c), units, mode
            ).reshape(mix.shape)
        elif mode == "rows":
            lengths = sum(w * x.norm(dim=-1, keepdim=True) for w, x in zip(weights, xs))
            mix = _rows_to_lengths(mix, lengths)
        out[name] = mix.to(ref.dtype)
    return out


class _AverageHolder(Optimizer):
    """Per-tensor state (`avg_<i>`, `rows_<i>`) for one model part; never
    stepped. An Optimizer only to reuse the FQN-keyed DCP state machinery."""

    def __init__(self, named_params: Sequence[tuple[str, torch.Tensor]]) -> None:
        params = [p for _, p in named_params]
        names = [canonical_fqn(n) for n, _ in named_params]
        super().__init__([{"params": params, "param_names": names}], {})

    def step(self, closure=None) -> None:  # pyrefly: ignore[bad-override]
        raise RuntimeError("_AverageHolder is never stepped; call WeightAverages.step")


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _row_lengths(p: torch.Tensor) -> torch.Tensor:
    """The lengths of `p`'s rows (over the last dim), placed like `p`."""
    rn = _local(p).detach().float().norm(dim=-1, keepdim=True)
    if not isinstance(p, DTensor):
        return rn
    shape = torch.Size([*p.shape[:-1], 1])
    return DTensor.from_local(
        rn,
        device_mesh=p.device_mesh,
        placements=p.placements,
        run_check=False,
        shape=shape,
        stride=torch.empty(shape, device="meta").stride(),
    )


class WeightAverages(OptimizersContainer):
    """Power averages of every trainable parameter, sibling of `EMA`: after
    step `s` (counted from `start_step`, every `every` steps) the average with
    exponent gamma is avg <- avg + (s^gamma / S) (w - avg), S the sum of
    s^gamma over the updates so far (gamma 0: the plain mean). A parameter
    whose renorm mode is `rows` also averages its row lengths."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore[bad-override]
        gammas: list[float] = field(default_factory=list)
        """Exponents of the power averages (e.g. [0, 1, 3, 7]); empty: none."""

        start_step: int = 0
        """The first update fires at start_step + every."""

        every: int = 1
        """Update every n optimizer steps."""

        renorm: list[list[str]] = field(default_factory=list)
        """[regex, mode] pairs on canonical FQNs, the first match wins, an
        unmatched parameter is `none`: frobenius / spectral scale each unit of
        the average to the norm of the live unit (the radius the rule holds);
        rows scales each row to its averaged length (learned row lengths)."""

        evaluate: bool = True
        """Score each average with the validator whenever the trainer validates."""

        snapshot_steps: list[int] = field(default_factory=list)
        """Write the full weights (fp32, canonical FQNs) after these steps to
        <dump>/<snapshot_folder>/step-<n>.pt (rank 0)."""

        snapshot_folder: str = "snapshots"

        export_steps: list[int] = field(default_factory=list)
        """Write each renormalised average after these steps to
        <dump>/<snapshot_folder>/avg_g<gamma>-step-<n>.pt (rank 0), scoreable
        later through score_files."""

        score_files: list[str] = field(default_factory=list)
        """Score these weight files (canonical FQN -> full tensor, e.g. a
        mixture of snapshots) with the validator at start, then stop."""

        start_from_gamma: float | None = None
        """Angular WSD: right after the run's FIRST load of `initial_load_path`
        (not after a restart from its own checkpoint), replace the model
        weights by this gamma's renormalised average (what the validation
        scores as avg_g<gamma>) and stop averaging. The optimizer state is the
        checkpoint's. The gamma must be one of `gammas`."""

        def __post_init__(self) -> None:
            if self.every < 1 or self.start_step < 0:
                raise ValueError("averages: every >= 1 and start_step >= 0")
            if any(not math.isfinite(g) or g < 0 for g in self.gammas):
                raise ValueError(
                    f"averages: gammas must be finite and >= 0, got {self.gammas}"
                )
            if len(set(self.gammas)) != len(self.gammas):
                raise ValueError(f"averages: gammas repeat: {self.gammas}")
            for pair in self.renorm:
                if len(pair) != 2 or pair[1] not in RENORM_MODES:
                    raise ValueError(
                        f"averages.renorm: [regex, mode] with mode in {RENORM_MODES}, got {pair}"
                    )
                re.compile(pair[0])
            if any(s < 1 for s in self.snapshot_steps):
                raise ValueError("averages.snapshot_steps must be >= 1")
            if self.export_steps and not self.gammas:
                raise ValueError("averages.export_steps needs gammas")
            if self.start_from_gamma is not None and float(
                self.start_from_gamma
            ) not in [float(g) for g in self.gammas]:
                raise ValueError(
                    f"averages.start_from_gamma {self.start_from_gamma} must be one of gammas {self.gammas}"
                )

    def __init__(
        self, config: Config, *, model_parts: list[nn.Module], output_dir: str = ""
    ) -> None:
        self.config = config
        self.model_parts = model_parts
        self.output_dir = output_dir
        self.gammas = [float(g) for g in config.gammas]
        self._renorm = [(re.compile(r), m) for r, m in config.renorm]
        self._snapshots = set(config.snapshot_steps)
        self._sums: list[tuple[int, float]] = [(0, 0.0) for _ in self.gammas]
        self._named: list[list[tuple[str, torch.Tensor]]] = []
        self._modes: dict[int, str] = {}
        self._units: dict[int, list[torch.Tensor]] = {}
        self.optimizers: list[_AverageHolder] = []
        all_params: list[nn.Parameter] = []
        for model in model_parts:
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            layouts = getattr(model, "param_layouts", None)
            layouts = {
                canonical_fqn(k): v for k, v in (layouts() if layouts else {}).items()
            }
            holder = _AverageHolder(named)
            for n, p in named:
                mode = self.mode_of(canonical_fqn(n))
                self._modes[id(p)] = mode
                if mode in ("frobenius", "spectral"):
                    self._units[id(p)] = self._unit_rows(canonical_fqn(n), p, layouts)
                if mode == "rows" and p.ndim < 2:
                    raise ValueError(
                        f"averages.renorm rows needs a matrix, {n} is {tuple(p.shape)}"
                    )
                if isinstance(p, DTensor) and mode == "rows":
                    if any(
                        not isinstance(pl, (Shard, Replicate))
                        or (isinstance(pl, Shard) and pl.dim >= p.ndim - 1)
                        for pl in p.placements
                    ):
                        raise ValueError(
                            f"averages.renorm rows needs whole rows on each rank, {n}: {p.placements}"
                        )
                st = holder.state[p]
                for i in range(len(self.gammas)):
                    st[f"avg_{i}"] = p.detach().clone().float()
                    if mode == "rows":
                        st[f"rows_{i}"] = _row_lengths(p)
            self._named.append(named)
            self.optimizers.append(holder)
            all_params.extend(p for _, p in named)
        self._post_init(all_params)

    def mode_of(self, fqn: str) -> str:
        return next((m for r, m in self._renorm if r.search(fqn)), "none")

    @staticmethod
    def _unit_rows(fqn: str, p: torch.Tensor, layouts: dict) -> list[torch.Tensor]:
        if p.ndim < 2:
            raise ValueError(
                f"averages.renorm frobenius / spectral needs a matrix, {fqn} is {tuple(p.shape)}"
            )
        layout = layouts.get(fqn)
        if layout is None or not layout.units:
            return [torch.arange(math.prod(p.shape[:-1]))]
        out = []
        for u in layout.units:
            out.append(torch.cat([torch.arange(s, s + n) for s, n in u.rows.runs]))
        return out

    def zero_grad(self, *args, **kwargs) -> None:
        pass

    # ---- update
    def _sum_at(self, i: int, k: int) -> float:
        """S after k updates for gamma i, as the same running sum on every
        path (fresh, resumed)."""
        k0, s = self._sums[i]
        if k < k0:
            k0, s = 0, 0.0
        g, e = self.gammas[i], self.config.every
        for j in range(k0 + 1, k + 1):
            s += float(j * e) ** g
        self._sums[i] = (k, s)
        return s

    @torch.no_grad()
    def step(self, current_step: int) -> None:  # pyrefly: ignore[bad-override]
        if self.gammas:
            s = current_step - self.config.start_step
            if s > 0 and s % self.config.every == 0:
                k = s // self.config.every
                for i, g in enumerate(self.gammas):
                    c = float(s) ** g / self._sum_at(i, k)
                    for holder, named in zip(self.optimizers, self._named):
                        params = [p for _, p in named]
                        avgs = [holder.state[p][f"avg_{i}"] for p in params]
                        torch._foreach_lerp_(avgs, [p.detach() for p in params], c)
                        for p in params:
                            if self._modes[id(p)] == "rows":
                                rows = holder.state[p][f"rows_{i}"]
                                rows.lerp_(_row_lengths(p), c)
        if current_step in self._snapshots:
            self.save_snapshot(current_step)
        if current_step in self.config.export_steps:
            for i, g in enumerate(self.gammas):
                self._save_full(
                    self._materialize(i),
                    current_step,
                    f"avg_g{g:g}-step-{current_step}.pt",
                )

    @torch.no_grad()
    def start_from_average(self) -> None:
        """The weights become the renormalised average of `config.start_from_gamma`,
        bitwise what the validation scores as that average (the live weights of the
        loaded checkpoint are the reference of the renormalisation); the averages
        stop updating."""
        i = [float(g) for g in self.gammas].index(float(self.config.start_from_gamma))
        locs = self._materialize(i)
        self._reshard()
        self._load_locals(locs)
        self._reshard()
        self.gammas = []

    # ---- the renormalised average as each rank's local tensors
    def _materialize(self, i: int) -> list[list[torch.Tensor]]:
        out = []
        for holder, named in zip(self.optimizers, self._named):
            locs = []
            for _, p in named:
                avg = holder.state[p][f"avg_{i}"]
                mode = self._modes[id(p)]
                if mode == "none":
                    locs.append(_local(avg).to(p.dtype))
                elif mode == "rows":
                    rows = _local(holder.state[p][f"rows_{i}"])
                    locs.append(_rows_to_lengths(_local(avg), rows).to(p.dtype))
                else:
                    full = avg.full_tensor() if isinstance(avg, DTensor) else avg
                    ref = (
                        p.detach().full_tensor()
                        if isinstance(p, DTensor)
                        else p.detach()
                    )
                    units = [u.to(full.device) for u in self._units[id(p)]]
                    shape = full.shape
                    full = renorm_rows(
                        full.float().reshape(-1, shape[-1]),
                        ref.float().reshape(-1, shape[-1]),
                        units,
                        mode,
                    ).reshape(shape)
                    locs.append(self._local_slice(p, full).to(p.dtype))
            out.append(locs)
        return out

    @staticmethod
    def _local_slice(p: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
        if not isinstance(p, DTensor):
            return full
        shape, off = compute_local_shape_and_global_offset(
            p.shape, p.device_mesh, p.placements
        )
        idx = tuple(slice(o, o + n) for o, n in zip(off, shape))
        return full[idx]

    def _load_locals(self, locs: list[list[torch.Tensor]]) -> None:
        for named, ls in zip(self._named, locs):
            for (_, p), x in zip(named, ls):
                _local(p.data).copy_(x)

    def _backup(self) -> list[list[torch.Tensor]]:
        return [[_local(p.data).clone() for _, p in named] for named in self._named]

    def _reshard(self) -> None:
        from torch.distributed.fsdp import FSDPModule

        for model in self.model_parts:
            for m in model.modules():
                if isinstance(m, FSDPModule):
                    m.reshard()

    def _validate_with(self, validator, locs, step: int, tag: str) -> None:
        backup = self._backup()
        try:
            self._reshard()
            self._load_locals(locs)
            validator.validate(self.model_parts, step, tag=tag)
        finally:
            self._load_locals(backup)
            self._reshard()

    def validate(self, validator, step: int) -> None:
        """Score every average (after the trainer's own validation)."""
        if not (self.gammas and self.config.evaluate):
            return
        for i, g in enumerate(self.gammas):
            self._validate_with(
                validator, self._materialize(i), step, tag=f"avg_g{g:g}"
            )

    # ---- snapshots and saved files
    def save_snapshot(self, step: int) -> None:
        sd = {}
        rank0 = not dist.is_initialized() or dist.get_rank() == 0
        for named in self._named:
            for n, p in named:
                t = p.detach().full_tensor() if isinstance(p, DTensor) else p.detach()
                if rank0:
                    sd[canonical_fqn(n)] = t.float().cpu()
        if rank0:
            modes, units = {}, {}
            for named in self._named:
                for n, p in named:
                    modes[canonical_fqn(n)] = self._modes[id(p)]
                    if id(p) in self._units:
                        units[canonical_fqn(n)] = [
                            u.tolist() for u in self._units[id(p)]
                        ]
            folder = os.path.join(self.output_dir, self.config.snapshot_folder)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"step-{step}.pt")
            blob = {"step": step, "weights": sd, "modes": modes, "units": units}
            torch.save(blob, path + ".tmp")
            os.replace(path + ".tmp", path)
            logger.info(f"averages: snapshot of {len(sd)} tensors -> {path}")

    def _save_full(self, locs, step: int, fname: str) -> None:
        """The parameters' local tensors `locs`, gathered whole (rank 0 writes)."""
        sd = {}
        rank0 = not dist.is_initialized() or dist.get_rank() == 0
        for named, ls in zip(self._named, locs):
            for (n, p), x in zip(named, ls):
                if isinstance(p, DTensor):
                    x = DTensor.from_local(
                        x,
                        p.device_mesh,
                        p.placements,
                        run_check=False,
                        shape=p.shape,
                        stride=p.stride(),
                    ).full_tensor()
                if rank0:
                    sd[canonical_fqn(n)] = x.float().cpu()
        if rank0:
            folder = os.path.join(self.output_dir, self.config.snapshot_folder)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, fname)
            torch.save({"step": step, "weights": sd}, path + ".tmp")
            os.replace(path + ".tmp", path)
            logger.info(f"averages: {fname} -> {path}")

    def score(self, validator, step: int) -> None:
        """Validate each of `score_files` (canonical FQN -> full tensor, the
        parameters' local slices loaded), tagged by its file name."""
        for path in self.config.score_files:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            weights = blob["weights"] if "weights" in blob else blob
            locs = []
            for named in self._named:
                ls = []
                for n, p in named:
                    key = canonical_fqn(n)
                    if key not in weights:
                        raise KeyError(f"averages.score_files: {path} has no {key}")
                    full = weights[key].to(_local(p).device)
                    if tuple(full.shape) != tuple(p.shape):
                        raise ValueError(
                            f"{path}: {key} is {tuple(full.shape)}, the model's {tuple(p.shape)}"
                        )
                    ls.append(self._local_slice(p, full).to(p.dtype))
                locs.append(ls)
            tag = os.path.splitext(os.path.basename(path))[0]
            self._validate_with(validator, locs, step, tag=f"score/{tag}")
