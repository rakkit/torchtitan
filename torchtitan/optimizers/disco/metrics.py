# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-unit optimizer metrics: computed on the unit's owner, packed into one
flat float32 buffer per rank with a fixed arity, gathered (or kept local) once,
and unpacked into `track_*` keys on the storing ranks."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor import DTensor

from torchtitan.distributed.utils import (
    metrics_shard_mesh,
    metrics_shard_rank,
    rank_owns_metrics_shard,
)

from ..helpers import gram_helper, norm_helper, power_iteration
from ..helpers.gram_helper import calculate_gram_metrics
from ..helpers.norm_helper import calculate_norm, calculate_norm_batched
from ..helpers.radial_helper import (
    calculate_radial_metrics,
    RADIAL_METRIC_NAMES,
    SpectralInputs,
)


def pseudo_post_update_weight(w, u, lr, decay):
    pseudo_w = w * decay if decay != 1.0 else w
    return pseudo_w - lr * u


def pop_spectrum(norms: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    spectrum = norms.pop("spectrum", None)
    sigma_max = norms.pop("sigma_max", None)
    if sigma_max is None and spectrum is not None and spectrum.numel():
        sigma_max = spectrum[0]
    return spectrum, sigma_max


def gram_log_param_name(cleaned_name: str, shape: tuple[int, ...]) -> str:
    if gram_helper.gram_matrix_is_transposed(shape):
        return f"{cleaned_name}.T"
    return cleaned_name


def _materialize(t: torch.Tensor) -> torch.Tensor:
    wait = getattr(t, "wait", None)
    return wait() if callable(wait) else t


def _leading(x):
    if x is None or x.ndim == 0:
        return x
    return x[0] if x.numel() else None


def _write(buf: torch.Tensor, off: int, t: torch.Tensor) -> int:
    """Copy `t` into `buf` at `off` as float32; return the offset past it."""
    buf[off : off + t.numel()].copy_(t.float())
    return off + t.numel()


def _split_norms(out: "UnitValues", upd_norms: dict, w_norms: dict):
    """Move the two norm dicts into `out` in buffer order; return their sigmas."""
    out.upd_spec, upd_sigma = pop_spectrum(upd_norms)
    out.w_spec, w_sigma = pop_spectrum(w_norms)
    out.upd, out.w = list(upd_norms.values()), list(w_norms.values())
    return w_sigma, upd_sigma


def radial_spectral_inputs(W_before, W_after, w_spectrum, upd_spectrum, exact):
    """SpectralInputs for one unit from what the norm pass already produced.
    `exact`: `W_after - W_before` is the logged update itself, so its spectrum
    already holds the step's sigma."""
    sigma_after = _leading(w_spectrum)
    sigma_update = _leading(upd_spectrum) if exact else None
    sigma_before = u1 = v1 = aus_sigma = None
    if W_before.ndim == 1:
        sigma_before = W_before.detach().abs().max().float()
    elif W_before.ndim == 2:
        sigma_before, u1, v1 = power_iteration.top_singular_pair(W_before)
        if sigma_update is None:
            sigma_update = power_iteration.spectral_norm(W_after - W_before)
        if sigma_after is not None:  # aus_rms_to_rms: the angle between the weights
            tiny = torch.finfo(torch.float32).tiny
            diff = W_before.float() / sigma_before.clamp_min(
                tiny
            ) - W_after.float() / sigma_after.clamp_min(tiny)
            aus_sigma = power_iteration.spectral_norm(diff)
            del diff
    if sigma_after is None and sigma_update is None and sigma_before is None:
        return None
    return SpectralInputs(sigma_before, u1, v1, sigma_after, sigma_update, aus_sigma)


def _stack_if_all(xs: list):
    return torch.stack(xs) if xs and all(x is not None for x in xs) else None


def _stack_spectral(per_unit: list) -> SpectralInputs | None:
    """Stack per-expert SpectralInputs field-wise; a field any unit is missing
    stays None, and one missing unit drops the whole batch."""
    if any(si is None for si in per_unit):
        return None
    fields = [
        None
        if any(si[f] is None for si in per_unit)
        else torch.stack([si[f] for si in per_unit])
        for f in range(len(SpectralInputs._fields))
    ]
    return SpectralInputs(*fields)


class ExpertUnits(NamedTuple):
    """One expert parameter's local experts on a logging step, `[E, m, n]` each."""

    key: int  # the caller's handle for the result
    W: torch.Tensor  # pre-update weights
    G: torch.Tensor  # effective gradients
    U: torch.Tensor  # what the holders apply
    # -> (the step logged as the update, the weights the apply writes), formed
    # when needed so a logging step never holds them for every expert at once
    terms: Callable[[], tuple]
    exact: bool  # w_after - W is the update itself
    radial_state: dict


@dataclass
class MetricUnit:
    """One logged matrix: who computes it and where its values sit in the buffer."""

    key: str  # `clean` or `ep_{e}/clean`
    shape: tuple[int, ...]  # the matrix the gram orientation is decided on
    owner: int
    slot: int = -1
    spec_len: int = 0
    gvec_len: int = 0


@dataclass
class UnitValues:
    """What the owner computed for one unit, in buffer order."""

    upd: list = field(default_factory=list)
    w: list = field(default_factory=list)
    gram: list = field(default_factory=list)
    radial: list = field(default_factory=list)
    gram_vecs: list = field(default_factory=list)
    upd_spec: torch.Tensor | None = None
    w_spec: torch.Tensor | None = None


class _Layout:
    """One step's buffer layout: per-unit scalars by slot, then the gram
    vectors, then the update and param spectra. Offsets are relative to
    `rank_base`, the start of the owning rank's slice of the gathered buffer."""

    def __init__(self, sink: "MetricSink"):
        opt = sink.opt
        n_vec = len(opt.gram_vector_names)
        self.n_gram = len(opt.gram_scalar_names)
        # The scalar order of one slot; the flag picks the gram-oriented name.
        self.row_names = (
            [(f"track_update_{n}/", False) for n in opt.norms_to_log]
            + [(f"track_param_{n}/", False) for n in opt.norms_to_log]
            + [(f"track_gram_{n}/", True) for n in opt.gram_scalar_names]
            + [(f"track_radial_{n}/", False) for n in RADIAL_METRIC_NAMES]
        )
        self.per_unit = len(self.row_names)
        scalars = sink.max_slots * self.per_unit
        # Vector sections: per rank, its owned units in slot order, padded to the max.
        self.gvec_off, self.spec_off = [], []
        gvec_max = spec_max = 0
        for units in sink.units_by_owner:
            g_off, s_off, g, s = {}, {}, 0, 0
            for mu in units:
                g_off[mu.key], s_off[mu.key] = g, s
                g += n_vec * mu.gvec_len
                s += mu.spec_len
            self.gvec_off.append(g_off)
            self.spec_off.append(s_off)
            gvec_max, spec_max = max(gvec_max, g), max(spec_max, s)
        self.want_gvec, self.want_spec = n_vec > 0, bool(opt.track_spectrum)
        self.spec_max = spec_max if self.want_spec else 0
        self.gvec_base = scalars
        self.spec_base = scalars + (gvec_max if self.want_gvec else 0)
        self.total = self.spec_base + 2 * self.spec_max

    def gvec_at(self, rank_base: int, owner: int, key: str) -> int:
        return rank_base + self.gvec_base + self.gvec_off[owner][key]

    def spec_at(self, rank_base: int, owner: int, key: str, param: bool) -> int:
        shift = self.spec_max if param else 0
        return rank_base + self.spec_base + shift + self.spec_off[owner][key]


class MetricSink:
    """Owns the unit registry, the per-rank buffer layout and the unpack."""

    def __init__(self, opt):
        self.opt = opt
        pd = opt.parallel_dims
        self.mesh = metrics_shard_mesh(pd)
        self.rank, self.world = metrics_shard_rank(pd)
        self.units: list[MetricUnit] = []

    # ---- registry (built once, identical on every rank) ----
    def add(self, key: str, shape, owner: int) -> MetricUnit:
        mu = MetricUnit(key, tuple(int(s) for s in shape), owner)
        self.units.append(mu)
        return mu

    def finalize(self) -> None:
        slots = [0] * self.world
        self.units_by_owner = [[] for _ in range(self.world)]
        for mu in self.units:
            mu.slot = slots[mu.owner]
            slots[mu.owner] += 1
            self.units_by_owner[mu.owner].append(mu)
            s = mu.shape
            mu.spec_len = max(s[0], 1) if len(s) == 1 else min(s[-2], s[-1])
            mu.gvec_len = gram_helper.gram_vector_len(
                tuple(s[-2:]) if len(s) >= 2 else (s[0], 1)
            )
        self.max_slots = max(slots) if slots else 0

    # ---- computations ----
    def unit_values(
        self,
        W,
        G_raw,
        update,
        w_after,
        radial_state,
        *,
        exact,
        transpose=False,
        gram=True,
    ) -> UnitValues:
        """`update`: the step term logged as the update (`-lr U` for a plain
        rule); `w_after`: the weight the apply writes; `exact`: `w_after - W`
        is `update` itself (no decay or rescaling of W in the step)."""
        opt = self.opt
        out = UnitValues()
        norm_kw = dict(transpose=transpose, want_spectrum=opt.track_spectrum)
        upd_norms = calculate_norm(update, opt.norms_to_log, **norm_kw)
        w_norms = calculate_norm(w_after, opt.norms_to_log, **norm_kw)
        w_sigma, upd_sigma = _split_norms(out, upd_norms, w_norms)
        spectral = radial_spectral_inputs(W, w_after, w_sigma, upd_sigma, exact)
        radial = calculate_radial_metrics(
            W, w_after, radial_state, spectral=spectral, transpose=transpose
        )
        out.radial = [radial[n] for n in RADIAL_METRIC_NAMES]
        if gram and opt.gram_level > 0:
            self._fill_gram(out, W, G_raw, w_after)
        elif gram:
            zero = opt._get_cached_zero_scalar(W.device)
            out.gram = [zero] * len(opt.gram_scalar_names)
        return out

    def _fill_gram(self, out: UnitValues, W, G_raw, pseudo_w) -> None:
        opt = self.opt
        g = calculate_gram_metrics(W, G_raw, pseudo_w, level=opt.gram_level)
        if not g:  # degenerate unit (fewer than two rows): zeros, fixed arity
            zero = opt._get_cached_zero_scalar(W.device)
            out.gram = [zero] * len(opt.gram_scalar_names)
            return
        out.gram = [g[n] for n in opt.gram_scalar_names]
        out.gram_vecs = [g[n] for n in opt.gram_vector_names]

    def expert_block_values(self, entries: list["ExpertUnits"]) -> dict:
        """{key: list[UnitValues] per local expert}. The norm and singular-pair
        passes are batched across the entries of one shape."""
        opt = self.opt
        batched = self._expert_batched_norms(entries)
        results: dict = {}
        for key, p_local, g_raw, _, terms, exact, radial_state in entries:
            b_upd, b_w, b_spec = batched[key]
            n_ep = p_local.shape[0]
            pw_all = terms()[1]
            vals = [UnitValues() for _ in range(n_ep)]
            sigmas = [
                _split_norms(vals[e], dict(b_upd[e]), dict(b_w[e])) for e in range(n_ep)
            ]
            w_sig, u_sig = [s[0] for s in sigmas], [s[1] for s in sigmas]
            if b_spec is None:
                spectral = _stack_spectral(
                    [
                        radial_spectral_inputs(p, q, ws, us, exact)
                        for p, q, ws, us in zip(p_local, pw_all, w_sig, u_sig)
                    ]
                )
            else:
                spectral = SpectralInputs(
                    *b_spec[:3], _stack_if_all(w_sig), *b_spec[3:]
                )
            state_local = {
                name: (t.to_local() if isinstance(t, DTensor) else t)[:n_ep]
                for name, t in radial_state.items()
            }
            radial = calculate_radial_metrics(
                p_local, pw_all, state_local, spectral=spectral, batch_ndim=1
            )
            for e, out in enumerate(vals):
                out.radial = [radial[n][e] for n in RADIAL_METRIC_NAMES]
                if opt.gram_level > 0:
                    self._fill_gram(out, p_local[e], g_raw[e], pw_all[e])
            results[key] = vals
        return results

    def _expert_batched_norms(self, entries) -> dict:
        """{key: (per-expert update norms, param norms, batched spectral triple)}.
        Entries of one shape are batched, in chunks of `_cap_for` rows."""
        opt = self.opt
        want_spectrum = opt.track_spectrum
        full_spectrum = want_spectrum or not set(opt.norms_to_log).issubset(
            norm_helper._SVD_FREE_NORMS | norm_helper._SIGMA_ONLY_NORMS
        )
        batched: dict = {}

        def _cap_for(shape) -> int:
            if not full_spectrum:
                return max(int(norm_helper._GESVDA_MAX_BATCH), 1)
            return norm_helper.gram_batch_capacity(shape[0], shape[1])

        def _norms(t):
            return calculate_norm_batched(
                t, opt.norms_to_log, want_spectrum=want_spectrum
            )

        def _run(group):
            if not group:
                return
            pieces = [x.terms()[:2] for x in group]
            upd = torch.cat([t for t, _ in pieces], dim=0)
            pw = torch.cat([w for _, w in pieces], dim=0)
            del pieces
            un, wn = _norms(upd), _norms(pw)
            wb = torch.cat([x.W for x in group], dim=0)
            sig_b, u1_b, v1_b = norm_helper.gram_top_singular_pair(wb)
            sig_u_b = norm_helper.gram_top_singular_pair(pw.float() - wb.float())[0]
            # aus_rms_to_rms: the angle between the weights, with the
            # post-update sigma the norm pass already produced (as for
            # dense units); skipped when no spectral norm is logged.
            sig_a_b = wn.get("sigma_max")
            if sig_a_b is None and "spectrum" in wn:
                sig_a_b = wn["spectrum"][..., 0]
            aus_b = None
            if sig_a_b is not None:
                tiny = torch.finfo(torch.float32).tiny
                diff = wb.float() / sig_b.clamp_min(tiny)[
                    :, None, None
                ] - pw.float() / (sig_a_b.float().clamp_min(tiny)[:, None, None])
                aus_b = norm_helper.gram_top_singular_pair(diff)[0]
                del diff
            del wb
            spec = (sig_b, u1_b, v1_b, sig_u_b, aus_b)
            del upd, pw
            off = 0
            for x in group:
                k, e = x.key, x.U.shape[0]
                rows = slice(off, off + e)
                batched[k] = (
                    [{n: un[n][off + i] for n in un} for i in range(e)],
                    [{n: wn[n][off + i] for n in wn} for i in range(e)],
                    tuple(None if t is None else t[rows] for t in spec),
                )
                off += e

        by_shape: dict = {}
        for ent in entries:
            key = (tuple(ent.U.shape[1:]), ent.U.dtype, ent.W.dtype)
            by_shape.setdefault(key, []).append(ent)
        for (shape, _udt, _pdt), group in by_shape.items():
            cap = _cap_for(shape)
            pending, count = [], 0
            for ent in group:
                e = ent.U.shape[0]
                if pending and count + e > cap:
                    _run(pending)
                    pending, count = [], 0
                pending.append(ent)
                count += e
            _run(pending)
        return batched

    # ---- pack / gather / unpack ----
    def flush(self, values: dict[str, UnitValues], device: torch.device) -> dict:
        """`values`: key -> UnitValues for every unit this rank owns. Returns the
        `track_*` dict on ranks that store metrics (empty elsewhere)."""
        opt = self.opt
        lay = _Layout(self)
        buf = torch.zeros(lay.total, dtype=torch.float32, device=device)
        zero = opt._get_cached_zero_scalar(device)
        for mu in self.units_by_owner[self.rank]:
            v = values.get(mu.key)
            if v is None:
                continue
            row = v.upd + v.w + (v.gram if v.gram else [zero] * lay.n_gram) + v.radial
            _write(buf, mu.slot * lay.per_unit, torch.stack([x.float() for x in row]))
            if lay.want_gvec and v.gram_vecs:
                off = lay.gvec_at(0, self.rank, mu.key)
                for vec in v.gram_vecs:
                    off = _write(buf, off, vec)
            if lay.want_spec and v.upd_spec is not None:
                _write(buf, lay.spec_at(0, self.rank, mu.key, False), v.upd_spec)
                if v.w_spec is not None:
                    _write(buf, lay.spec_at(0, self.rank, mu.key, True), v.w_spec)
        local = opt.log_metrics_locally or self.mesh is None or self.world == 1
        gbuf = buf
        if not local:
            gbuf = funcol.all_gather_tensor(buf, gather_dim=0, group=self.mesh)
            gbuf = _materialize(gbuf)
        if not self.stores_norms:
            return {}
        final = {}
        owned = [
            (r, mu)
            for r in ([self.rank] if local else range(self.world))
            for mu in self.units_by_owner[r]
        ]
        # Every scalar in one device-to-host copy: 0-d device views would cost
        # the logger one sync each.
        n_rows = len(lay.row_names)
        bases = torch.tensor(
            [
                (0 if local else r * lay.total) + mu.slot * lay.per_unit
                for r, mu in owned
            ],
            dtype=torch.long,
            device=gbuf.device,
        )
        rows = torch.arange(n_rows, device=gbuf.device)
        scalars = gbuf[(bases[:, None] + rows).reshape(-1)].tolist()
        for j, (r, mu) in enumerate(owned):
            rank_base = 0 if local else r * lay.total
            gname = gram_log_param_name(mu.key, mu.shape)
            for i, (prefix, gram_named) in enumerate(lay.row_names):
                final[prefix + (gname if gram_named else mu.key)] = scalars[
                    j * n_rows + i
                ]
            if lay.want_gvec:
                off, n_g = lay.gvec_at(rank_base, r, mu.key), mu.gvec_len
                for n in opt.gram_vector_names:
                    final[f"track_gram_{n}/{gname}"] = gbuf[off : off + n_g]
                    off += n_g
            if lay.want_spec:
                n_s = mu.spec_len
                off = lay.spec_at(rank_base, r, mu.key, False)
                final[f"track_spectrum_update/{mu.key}"] = gbuf[off : off + n_s]
                off = lay.spec_at(rank_base, r, mu.key, True)
                final[f"track_spectrum_param/{mu.key}"] = gbuf[off : off + n_s]
        return final

    @property
    def stores_norms(self) -> bool:
        if self.opt.log_metrics_locally:
            return rank_owns_metrics_shard(self.opt.parallel_dims)
        return self.opt.is_dp_rank_0

    @property
    def stores_replicated_norms(self) -> bool:
        if not self.opt.log_metrics_locally:
            return self.stores_norms
        return self.stores_norms and self.rank == 0
