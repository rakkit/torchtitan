# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Post-processing for DiSCO's vector-valued gram metrics (`track_gram_*` keys
whose value is a >1-element tensor -- see gram_helper.py). Raw multi-element
tensors are never valid values for a scalar logger (WandBLogger/
TensorBoardLogger), so they're always popped out of the norms dict and
rendered as "index vs. value" atlas-grid images instead -- one small-
multiples image per distinct gram metric name (rows = layers, columns =
weight-type, same idea as spectrum_logging.py's atlas grids for singular-
value spectra), rather than a per-parameter image (illegible/unusable at
tens of thousands of tracked parameters) or a histogram (loses the row-index
axis, which is exactly what most gram row-metrics -- dominance ratios,
cross-Gram diagonals -- are about).

Scalars need no handling at all here: they're already valid logger values,
so they simply aren't touched by this module.

This is a **copy-and-adapt** of spectrum_logging.py's generic atlas-grid
machinery (row/column parsing from tracked key names, grid layout, MoE
distribution-overlay handling), not a shared extraction -- deliberately, so
spectrum_logging.py's working, already-shipped plotting stays completely
untouched. What's copied is genuinely generic (parsing/layout has no
"spectrum" semantics baked in); what's different is the per-cell curve:
spectrum draws two curves (normalized spectrum + cumulative energy, forced
into `[0, 1]`, since singular values are non-negative and sortable) where
gram vectors get **one** raw-value line, y-axis auto-ranged per cell -- most
gram vectors aren't non-negative/sortable (cosines in `[-1, 1]`, unbounded
ratios, signed eigenvalues), so forcing spectrum's normalization wouldn't
make sense, and using one uniform simple treatment across all ~39 vector
metrics is simpler than special-casing the eigenvalue-like ones.

With ~39 distinct vector metric names at level 3 (dense + MoE each), this
can render up to ~78 images per logging event -- much more than spectrum's
4. Every vector-valued metric is plotted (no default subset/opt-in list);
narrow this later if the render-time/storage cost proves too high in
practice.

Also supports a `spectrum_logging._export_spectrum`-style Parquet export
(`GramVectorLoggingConfig.enable_export`/`export_dir`, see
`_export_gram_vectors`) -- every tracked gram vector for the event (full or
downsampled, see `_MAX_RESAMPLE_POINTS`) written to one file per step,
uploaded as a versioned W&B Artifact, independent of whether plotting is
also enabled.
"""

import functools
import os
from typing import Any, NamedTuple

import torch

__all__ = [
    "GramVectorLoggingConfig",
    "process_gram_vectors_for_logging",
]


class GramVectorLoggingConfig(NamedTuple):
    """
    Bundles the settings `process_gram_vectors_for_logging` needs -- mirrors
    `spectrum_logging.SpectrumLoggingConfig` exactly (same two independent
    toggles, same `export_dir`). All three come from
    `DiSCOOptimizersContainer.Config` / its `output_dir` build kwarg and are
    set in its `__init__`. Stored as
    `DiSCOOptimizersContainer.gram_vector_logging_config`.
    """

    enable_export: bool = False
    export_dir: str | None = None


# Own pool, separate from spectrum_logging.py's -- keeps the two modules
# fully decoupled per the copy-and-adapt decision. Same size as spectrum's;
# revisit once the "plot everything" cost is observed in practice.


_GRAM_PREFIX = "track_gram_"


# Same two reasons as spectrum_logging.py's identical constant: grid cells
# are tiny (a few tens of px), so no more than a few hundred distinct
# x-positions are ever visually distinguishable.
_MAX_RESAMPLE_POINTS = 512

# Plotting-only cap on how many experts' columns appear in the MoE grid --
# disco.py's step_experts still computes/gathers every expert's gram vector
# regardless (see module docstring).


# Sentinel rows for the dense grid's top -- see _build_grids/_order_embed_columns.


@functools.lru_cache(maxsize=128)
def _resample_plan(n: int, num_points: int, device: torch.device):
    """Gather indices + interpolation weights for an `n -> num_points` resample.

    Depends only on the two lengths, never on the values, and a logging step
    resamples hundreds of tensors drawn from a handful of distinct lengths --
    so building `linspace` twice and re-running `searchsorted` per call was
    ~98% of the export cost (measured: 577 ms -> 9.4 ms for one rank's
    spectra, 61x, bit-identical output).

    NOTE the twin of this function in `spectrum_logging.py`. The two modules
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


def _is_gram_vector_entry(key: str, v: Any) -> bool:
    return isinstance(v, torch.Tensor) and v.numel() > 1 and "track_gram_" in key


# Which metric names also get cheap derived mean/min/max scalars alongside
# their atlas plot -- opt-in, not every vector (most gram vectors don't need
# this; add more names here if you want the same treatment elsewhere).
_MEAN_MIN_MAX_METRIC_NAMES = frozenset({"V_R_raw"})


def _metric_name(key: str) -> str:
    return key[len(_GRAM_PREFIX) :].partition("/")[0]


def _mean_min_max_keys(key: str) -> tuple[str, str, str]:
    """
    `track_gram_V_R_raw/layers.0.attention.wq` ->
    (`track_gram_V_R_raw_mean/layers.0.attention.wq`,
     `track_gram_V_R_raw_min/layers.0.attention.wq`,
     `track_gram_V_R_raw_max/layers.0.attention.wq`) -- suffixes the metric
    name (the path component before the first "/", same one `_parse_gram_
    short_name` reads as "kind") so these land as ordinary scalar keys, not
    inside any atlas grid.
    """
    short_name = key[len(_GRAM_PREFIX) :]
    metric_name, sep, rest = short_name.partition("/")
    base = f"{_GRAM_PREFIX}{metric_name}"
    suffix = f"{sep}{rest}"
    return f"{base}_mean{suffix}", f"{base}_min{suffix}", f"{base}_max{suffix}"


def _export_gram_vectors(
    gram_tensors: dict[str, torch.Tensor], step: int, export_dir: str
) -> None:
    """
    Write every gram vector tracked this logging event into a single
    Parquet file (one row per (metric_name, param_name) pair -- `param_name`
    carries the "ep_N/..." prefix as-is for expert params, same one-split
    convention `spectrum_logging._export_spectrum` uses; see
    `_MAX_RESAMPLE_POINTS` for the only lossy step involved), then upload it
    into a single, persistent, run-scoped W&B Artifact
    ("gram-vectors-{run.id}"), auto-versioned by wandb (a new version each
    time this runs). No-ops if no W&B run is active (e.g. `enable_wandb=
    False`), same "give up gracefully" policy as
    `spectrum_logging._export_spectrum`, which this mirrors directly.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import wandb

    rows = []
    for key, tensor in gram_tensors.items():
        metric_name, _, param_name = key[len(_GRAM_PREFIX) :].partition("/")
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
                "metric_name": metric_name,
                "num_values": n,
                "values": values.tolist(),
            }
        )

    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"step_{step}.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")

    if wandb.run is None:
        return
    artifact = wandb.Artifact(name=f"gram-vectors-{wandb.run.id}", type="gram-vectors")
    artifact.add_file(path, name=os.path.basename(path))
    wandb.run.log_artifact(artifact, aliases=[f"step_{step}"])


def process_gram_vectors_for_logging(
    all_norms: dict[str, Any],
    step: int,
    config: GramVectorLoggingConfig,
) -> dict[str, Any]:
    """
    Pull every vector-valued gram metric out of `all_norms` (keys containing
    "track_gram_" whose value has more than one element -- produced by
    DiSCO's gram tracking at gram_level >= 1, see disco.py/gram_helper.py).
    Raw multi-element tensors are never valid values for a scalar logger, so
    they're always popped out; what (if anything) then happens to them is
    controlled by `config` (see `GramVectorLoggingConfig`, mirrors
    `spectrum_logging.SpectrumLoggingConfig` exactly):
    - `config.enable_plot`: bucket into one atlas grid per (metric_name,
      dense|moe) pair and render each in parallel across a small process
      pool, adding `plot_gram_{metric_name}_dense`/`plot_gram_{metric_name}
      _moe` back into the dict as `wandb.Image` objects.
    - `config.enable_export`: write+upload every tracked gram vector for
      this event (see `_export_gram_vectors`) -- nothing added back into the
      dict for this, it's a side-channel artifact upload, not a per-step
      scalar metric.
    Scalars are untouched -- they're already valid logger values.

    Also always derives 3 cheap scalar summaries (`..._mean`/`..._min`/
    `..._max` -- see `_mean_min_max_keys`) for the small set of metric names
    in `_MEAN_MIN_MAX_METRIC_NAMES` (currently just `V_R_raw`), independent
    of `config` -- computed here at the logging boundary rather than in
    gram_helper.py's output, cheap to reconstruct post-hoc from the
    already-logged vector.

    Mutates and returns `all_norms`. Same seam spectrum_logging.py's
    `process_norms_for_logging` provides: by the time a norms dict reaches
    a logger, every gram-vector entry is already the exact object
    `wandb.log()` expects.
    """
    gram_vector_keys = [k for k, v in all_norms.items() if _is_gram_vector_entry(k, v)]
    if not gram_vector_keys:
        return all_norms

    # Single batched device->host transfer instead of one `.cpu()` call per
    # vector -- at gram_level=3 there can be thousands (~39 metric names x
    # every tracked param), and each individual `.cpu()` is its own
    # CUDA sync + copy. Concat on-device (cheap, one kernel), one `.cpu()`,
    # then slice back into per-key tensors on CPU.
    #
    # Each slice must be `.clone()`d before being hearded off to a worker
    # process below (via `executor.submit`): torch's default tensor
    # pickling serializes the FULL underlying storage a view points into,
    # not just the logical slice -- an un-cloned narrow view into the
    # shared `flat` buffer would drag the entire batched buffer across the
    # process boundary on every single grid submission (measured: pickling
    # a 10-element view into an 8MB tensor serializes all ~8MB; a `.clone()`
    # of the same view serializes ~430 bytes). `.clone()` on an already-CPU
    # tensor is a cheap plain memcpy, no device sync -- this is not
    # reintroducing the cost we just batched away.
    raw_tensors = [all_norms.pop(key) for key in gram_vector_keys]
    lengths = [t.numel() for t in raw_tensors]
    flat = torch.cat([t.detach().reshape(-1) for t in raw_tensors]).float().cpu()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    gram_tensors = {
        key: flat[offsets[i] : offsets[i + 1]].clone()
        for i, key in enumerate(gram_vector_keys)
    }

    for key, tensor in gram_tensors.items():
        if _metric_name(key) not in _MEAN_MIN_MAX_METRIC_NAMES:
            continue
        mean_key, min_key, max_key = _mean_min_max_keys(key)
        all_norms[mean_key] = tensor.mean()
        all_norms[min_key] = tensor.min()
        all_norms[max_key] = tensor.max()

    if config.enable_export and config.export_dir is not None:
        _export_gram_vectors(gram_tensors, step, config.export_dir)

    return all_norms
