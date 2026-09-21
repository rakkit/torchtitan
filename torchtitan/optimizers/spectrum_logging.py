# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers for visualizing and exporting a parameter's singular-value spectrum
(see optimizers/norm_helper.py / optimizers/disco.py, which compute and
gather the raw values into `track_spectrum_update/...` and
`track_spectrum_param/...` metric entries) via Weights & Biases.

Two things can happen to each spectrum, independently controlled by
`SpectrumLoggingConfig` (built from `DiSCOOptimizersContainer.Config`'s
`enable_spectrum_plot`/`enable_spectrum_export` fields -- i.e.
`config.optimizer.enable_spectrum_plot`/`enable_spectrum_export` in the job
config -- see `components/optimizer.py`'s
`DiSCOOptimizersContainer.spectrum_logging_config`):

- `enable_plot`: instead of one image per parameter (illegible/unusable at
  ~31k tracked parameters), render a small number of "atlas" grid images —
  one small-multiples cell per parameter, arranged so the whole model's
  structure is visible at a glance. Up to two grids, each split by kind
  (`plot_dense_update`/`plot_dense_param`/`plot_moe_update`/
  `plot_moe_param`):
  - dense grid: rows = layers, columns = attention/FFN matrix types (WQ,
    WK, WV, WO, W1, W2, W3, plus any other per-layer matrix that's tracked,
    e.g. the MoE router's gate), discovered dynamically from the tracked
    keys rather than hardcoded. Embedding/lm-head (and anything else
    outside a "layers.N." prefix) have no meaningful layer index or matrix
    type of their own, so rather than wasting a row/column across the whole
    grid for them, they get one dedicated top row with their own per-cell
    titles instead of the shared column header — still one image, see
    `_render_grid`.
  - MoE grid: rows = MoE-enabled layers, columns = one distribution column
    per weight type (`W1*`/`W2*`/`W3*` -- both curves overlaid across
    *every* gathered expert, see `_draw_distribution_cell`) followed by
    `E{expert_idx}{type}` (e.g. "E0W1", "E1W2", ...) for individual
    experts, capped at `_MAX_EXPERTS_TO_PLOT` experts — a plotting-only cap
    on the individual columns only: it thins out what's *drawn* there, not
    what disco.py computes/gathers upstream (see `_MAX_EXPERTS_TO_PLOT`),
    and the distribution columns always reflect every expert regardless.
  Each cell is the same normalized-spectrum + cumulative-energy curve as
  before, minus per-cell axis ticks (illegible at grid-cell size); row/
  column headers plus one figure-level legend carry the labeling instead.
- `enable_export`: write every tracked parameter's spectrum for the event
  into a Parquet file and upload it as a versioned W&B Artifact -- a
  "database" companion meant for offline/programmatic analysis, since the
  plot already covers in-UI visualization. Spectra longer than
  `_MAX_RESAMPLE_POINTS` are linearly resampled down to that many points
  along the rank-index axis purely to bound per-event file size;
  `param_name`'s original length is kept in the `num_singular_values`
  column, so shorter spectra (the common case) stay exact and the tradeoff
  is always visible. Unlike the plot, this is never expert-capped -- it's
  meant to be the lossless record.

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


# Rendering is CPU-bound (matplotlib draw calls + PNG encode) — parallelize
# the (small, fixed) number of grid images across a small process pool
# rather than doing it serially in the training process. Threads were tried
# first in an earlier per-parameter version of this module and measured to
# give only ~7% speedup: matplotlib's cost is dominated by pure-Python
# layout/text work that holds the GIL, so threads mostly serialize anyway.
# Processes side-step the GIL entirely. Sized to the max number of grids
# `_build_grids` can ever produce in one event (dense/moe x update/param) --
# an earlier per-parameter version of this module needed far more workers
# (hundreds of small renders per event); consolidating into a handful of
# "atlas" images made that unnecessary, but the pool itself (built once,
# persistent) is still worth keeping for the ~4x wall-clock win on what's
# now a few hundred-subplot renders instead of thousands of tiny ones.


# Shared resolution cap, for two unrelated reasons:
# - plotting: grid cells are tiny (a few tens of px at most), so no more
#   than a few hundred distinct x-positions are ever visually
#   distinguishable — drawing every point of a multi-thousand-length
#   spectrum costs render time for detail nobody can see.
# - export: bounds per-event Parquet size (see `_export_spectrum`) —
#   storage-only, `num_singular_values` in the exported row always records
#   the true, pre-resample length.
# Either way, spectra at or below this length are handled exact/unsampled.
_MAX_RESAMPLE_POINTS = 512

# Plotting-only cap on how many experts' columns appear in the MoE grid —
# thins out what's *drawn*; disco.py's step_experts still computes/gathers
# every expert's spectrum regardless (see module docstring). Bump this when
# you want to look at more experts; it's just a rendering knob.

# Fixed preferred column order for the dense grid — anything else tracked
# per-layer (e.g. the MoE router's "gate") is appended after these, sorted.

# Sentinel rows for the dense grid's top, one each for embedding/lm-head/
# anything else that isn't inside a "layers.N." prefix. These aren't real
# matrix "types" or layers, so `_build_grids` routes them (any row < 0) into
# their own small standalone grid instead of the per-layer dense one — see
# `_build_grids`, `_order_embed_columns`.


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
    path = os.path.join(export_dir, f"step_{step}.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")

    if wandb.run is None:
        return
    artifact = wandb.Artifact(name=f"spectrum-{wandb.run.id}", type="spectrum")
    artifact.add_file(path, name=os.path.basename(path))
    wandb.run.log_artifact(artifact, aliases=[f"step_{step}"])


def process_norms_for_logging(
    all_norms: dict[str, Any],
    step: int,
    config: SpectrumLoggingConfig,
) -> dict[str, Any]:
    """
    Pull every raw singular-value spectrum tensor out of `all_norms` (keys
    containing "track_spectrum_", produced by DiSCO's norm tracking — see
    disco.py). Raw spectrum tensors are never valid values for a scalar
    logger, so they're always popped out; what (if anything) then happens to
    them is controlled by `config` (see `SpectrumLoggingConfig`):
    - `config.enable_plot`: bucket every tensor into up to 4 "atlas" grids
      (see `_build_grids`) and render each in parallel across a small
      process pool (`_get_executor`, `_render_grid`), adding
      `plot_dense_update`/`plot_dense_param`/`plot_moe_update`/
      `plot_moe_param` back into the dict as `wandb.Image` objects.
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

    # Move to CPU here, in the main process, before crossing the process
    # boundary for rendering — see _get_executor's "spawn" note. Both the
    # export and the plot branches need CPU tensors, so do this once
    # regardless of which flags are on.
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
