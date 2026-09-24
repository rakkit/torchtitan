# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers for visualizing and exporting a parameter's singular-value spectrum
(see helpers/norm_helper.py and disco/metrics.py, which compute and
gather the raw values into `track_spectrum_update/...` and
`track_spectrum_param/...` metric entries) via Weights & Biases.

Controlled by `SpectrumLoggingConfig`, built from
`DiSCOOptimizersContainer.Config`'s `enable_spectrum_export` field:

- `enable_export`: write every tracked parameter's spectrum for the event
  into a Parquet file and upload it as a versioned W&B Artifact, for
  offline/programmatic analysis. Spectra longer than
  `_MAX_RESAMPLE_POINTS` are linearly resampled down to that many points
  along the rank-index axis purely to bound per-event file size;
  `param_name`'s original length is kept in the `num_singular_values`
  column, so shorter spectra (the common case) stay exact and the tradeoff
  is always visible. Nothing is expert-capped: this is the lossless record.
  (An in-UI "atlas" plot mode lived here too, and went with its config
  field.)

`process_norms_for_logging` is the entry point: given the raw norms dict
produced by `DiSCOOptimizersContainer.get_parameter_norms()`
(components/optimizer.py), it replaces every raw spectrum tensor with
whatever's enabled, so components/metrics.py never needs any spectrum-
specific logic -- it just forwards whatever's left in the dict to
`wandb.log()`.
"""

import functools
import os
from typing import Any, NamedTuple

import torch

__all__ = [
    "SpectrumLoggingConfig",
    "process_norms_for_logging",
]


class SpectrumLoggingConfig(NamedTuple):
    """
    Bundles the settings `process_norms_for_logging` needs, so callers pass
    one object instead of a growing list of loose bool/str arguments.
    All three come from `DiSCOOptimizersContainer.Config` / its `output_dir`
    build kwarg and are set in its `__init__`. Stored as
    `DiSCOOptimizersContainer.spectrum_logging_config`.
    """

    enable_export: bool = False
    export_dir: str | None = None


# Bounds per-event Parquet size (see `_export_spectrum`); storage only, since
# `num_singular_values` in the exported row keeps the true pre-resample
# length. Spectra at or below this length are exported exact.
_MAX_RESAMPLE_POINTS = 512


@functools.lru_cache(maxsize=128)
def _resample_plan(n: int, num_points: int, device: torch.device):
    """Gather indices + interpolation weights for an `n -> num_points` resample.

    Depends only on the two lengths, never on the values, and a logging step
    resamples hundreds of tensors drawn from a handful of distinct lengths --
    so building `linspace` twice and re-running `searchsorted` per call was
    ~98% of the export cost (measured: 577 ms -> 9.4 ms for one rank's
    spectra, 61x, bit-identical output).

    NOTE the twin of this function in `gram_vector_logging.py`. The two modules
    are near-duplicates; a fix applied to one does not reach the other (that
    is exactly how the batched device->host transfer came to exist in only
    one of them). Change both, or unify them.
    """
    src_x = torch.linspace(0, 1, n, device=device)
    dst_x = torch.linspace(0, 1, num_points, device=device)
    idx = torch.searchsorted(src_x, dst_x).clamp(1, n - 1)
    x0, x1 = src_x[idx - 1], src_x[idx]
    w = ((dst_x - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
    return idx, w


def _resample_1d(s: torch.Tensor, num_points: int) -> torch.Tensor:
    """Linearly interpolate a 1-D sequence onto `num_points` evenly spaced
    positions along its index axis (preserves overall shape/endpoints)."""
    idx, w = _resample_plan(s.numel(), num_points, s.device)
    y0, y1 = s[idx - 1], s[idx]
    return y0 + w * (y1 - y0)


# Distribution columns ("W1*", "W2*", "W3*") summarize *all* gathered
# experts for a (layer, weight_type) via an overlay -- see
# `_draw_distribution_cell` -- distinct from the individual "E{idx}{type}"
# columns, which are capped at `_MAX_EXPERTS_TO_PLOT`. Both curves
# (spectrum + cumulative energy) share one cell, same as every other cell
# in the grid; the "*" suffix is just to avoid colliding with the dense
# grid's own literal "W1"/"W2"/"W3" columns (different grid, but same
# `_render_grid` code path inspects column labels generically).


def _is_spectrum_entry(key: str, v: Any) -> bool:
    return isinstance(v, torch.Tensor) and v.numel() > 1 and "track_spectrum_" in key


def _export_spectrum(
    spectrum_tensors: dict[str, torch.Tensor], step: int, export_dir: str
) -> None:
    """
    Write every spectrum tracked this logging event into a single Parquet
    file (one row per (kind, param_name) pair; see `_MAX_RESAMPLE_POINTS` for
    the only lossy step involved), then upload it into a single, persistent,
    run-scoped W&B Artifact ("spectrum-{run.id}"), auto-versioned by wandb
    (a new version each time this runs). No-ops if no W&B run is active
    (e.g. `enable_wandb=False`), same "give up gracefully" policy as the
    rest of this module.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import wandb

    rows = []
    for key, tensor in spectrum_tensors.items():
        kind, _, param_name = key.replace("track_spectrum_", "").partition("/")
        n = tensor.numel()
        values = (
            tensor
            if n <= _MAX_RESAMPLE_POINTS
            else _resample_1d(tensor, _MAX_RESAMPLE_POINTS)
        )
        rows.append(
            {
                "step": step,
                "param_name": param_name,
                "kind": kind,
                "num_singular_values": n,
                "singular_values": values.tolist(),
            }
        )

    os.makedirs(export_dir, exist_ok=True)
    path = export_path(export_dir, step)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")

    if wandb.run is None:
        return
    artifact = wandb.Artifact(name=f"spectrum-{wandb.run.id}", type="spectrum")
    artifact.add_file(path, name=os.path.basename(path))
    wandb.run.log_artifact(artifact, aliases=[f"step_{step}"])


def export_path(export_dir: str, step: int) -> str:
    """One file per writing rank: with per-rank logging or PP, several ranks
    export at the same step into the same folder."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return os.path.join(export_dir, f"step_{step}_rank{rank}.parquet")


def process_norms_for_logging(
    all_norms: dict[str, Any],
    step: int,
    config: SpectrumLoggingConfig,
) -> dict[str, Any]:
    """
    Pull every raw singular-value spectrum tensor out of `all_norms` (keys
    containing "track_spectrum_", produced by DiSCO's norm tracking -- see
    disco/metrics.py). Raw spectrum tensors are never valid values for a scalar
    logger, so they're always popped out; what (if anything) then happens to
    them is controlled by `config` (see `SpectrumLoggingConfig`):
    - `config.enable_export`: write+upload the full spectrum for every
      tracked parameter this event (see `_export_spectrum`) — nothing is
      added back into the dict for this, since it's not a per-step scalar
      metric, it's a side-channel artifact upload.

    Mutates and returns `all_norms`. This is the seam that keeps
    components/metrics.py free of any spectrum-specific logic — by the time
    a norms dict reaches a logger, spectrum entries are already the exact
    objects `wandb.log()` expects, same as every other metric value.
    """
    spectrum_keys = [k for k, v in all_norms.items() if _is_spectrum_entry(k, v)]
    if not spectrum_keys:
        return all_norms

    # One batched device->host transfer, not one `.cpu()` per spectrum: there
    # are 2 per tracked parameter (update + weight), ~672 per rank per logging
    # step for qwen30b-a3b at dp_shard=64, and each individual `.cpu()` is its
    # own CUDA sync + copy. Concat on-device (one kernel), one `.cpu()`, slice
    # back on the host. Measured 10.0 ms -> 5.3 ms, values bit-identical.
    #
    # Spectra have different lengths (min(m,n) per parameter), hence the
    # explicit offsets rather than a reshape. Each slice is `.clone()`d for the
    # same reason gram_vector_logging.py clones: torch pickles the FULL
    # underlying storage a view points into, so an un-cloned narrow view would
    # drag the whole concatenated buffer across the process boundary on every
    # grid submission below. `.clone()` on an already-CPU tensor is a plain
    # memcpy, so this does not reintroduce the cost just batched away.
    _raw = [all_norms.pop(key) for key in spectrum_keys]
    _flat = torch.cat([t.detach().reshape(-1) for t in _raw]).float().cpu()
    _offsets = [0]
    for _t in _raw:
        _offsets.append(_offsets[-1] + _t.numel())
    spectrum_tensors = {
        key: _flat[_offsets[i] : _offsets[i + 1]].clone()
        for i, key in enumerate(spectrum_keys)
    }

    if config.enable_export and config.export_dir is not None:
        _export_spectrum(spectrum_tensors, step, config.export_dir)

    return all_norms
