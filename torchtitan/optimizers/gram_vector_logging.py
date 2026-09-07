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
import multiprocessing
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, NamedTuple

import numpy as np
import torch

__all__ = [
    "GramVectorLoggingConfig",
    "process_gram_vectors_for_logging",
]


class GramVectorLoggingConfig(NamedTuple):
    """
    Bundles the settings `process_gram_vectors_for_logging` needs -- mirrors
    `spectrum_logging.SpectrumLoggingConfig` exactly (same two independent
    toggles, same deferred `export_dir`). `enable_plot`/`enable_export` are
    set from `OptimizersContainer.Config` directly in its `__init__`;
    `export_dir` depends on the top-level `config.dump_folder`, so
    trainer.py patches that in afterward. Stored as
    `OptimizersContainer.gram_vector_logging_config`.
    """

    enable_plot: bool = True
    enable_export: bool = False
    export_dir: str | None = None


# Own pool, separate from spectrum_logging.py's -- keeps the two modules
# fully decoupled per the copy-and-adapt decision. Same size as spectrum's;
# revisit once the "plot everything" cost is observed in practice.
_MAX_WORKERS = 4

_PYPLOT = None


def _get_pyplot():
    """Import matplotlib.pyplot exactly once, forcing the non-interactive
    Agg backend before pyplot is ever imported -- see spectrum_logging.py's
    identical helper for the full rationale (must happen before any other
    import of pyplot, or `matplotlib.use()` can silently no-op)."""
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


_GRAM_PREFIX = "track_gram_"

_LINE_COLOR = "tab:blue"

# Same two reasons as spectrum_logging.py's identical constant: grid cells
# are tiny (a few tens of px), so no more than a few hundred distinct
# x-positions are ever visually distinguishable.
_MAX_RESAMPLE_POINTS = 512

# Plotting-only cap on how many experts' columns appear in the MoE grid --
# disco.py's step_experts still computes/gathers every expert's gram vector
# regardless (see module docstring).
_MAX_EXPERTS_TO_PLOT = 4

_PREFERRED_DENSE_COLUMNS = ["WQ", "WK", "WV", "WO", "W1", "W2", "W3"]

# Sentinel rows for the dense grid's top -- see _build_grids/_order_embed_columns.
_OTHER_ROW = -3
_EMBED_ROW = -2
_LM_HEAD_ROW = -1


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


def _parse_gram_short_name(short_name: str) -> tuple[str, int | None, int, str]:
    """
    Parse a `short_name` (a `track_gram_` key with that prefix already
    stripped) into `(metric_name, expert_idx, row, weight_type)` -- same
    parsing as spectrum_logging.py's `_parse_spectrum_short_name`, just
    generalized: the leading path component is a gram metric name (e.g.
    "V_R_raw", "WU_diagonal") instead of "update"/"param", giving one grid
    per metric name:
    - "V_R_raw/layers.9.attention.wq"        -> ("V_R_raw", None, 9, "WQ")
    - "V_R_raw/layers.9.feed_forward.w1.T"   -> ("V_R_raw", None, 9, "W1.T")
    - "V_R_raw/ep_3/layers.9.moe.experts.w1" -> ("V_R_raw", 3, 9, "W1")
    - "V_R_raw/tok_embeddings"                -> ("V_R_raw", None, _EMBED_ROW, "EMBED")
    - "V_R_raw/output.T"                      -> ("V_R_raw", None, _LM_HEAD_ROW, "LM_HEAD.T")

    A trailing ".T" on the param name (disco.py's `_gram_log_param_name`
    marks parameters gram_helper transposed this way -- see
    `gram_helper.gram_matrix_is_transposed`) is stripped BEFORE row/embed
    routing, so e.g. "output.T" still correctly routes to the LM_HEAD row
    instead of falling through to `_OTHER_ROW`, then re-appended to the
    weight_type label. Re-appending (not discarding) matters: without it,
    every transposed weight type collapses to the literal string "T"
    (`"w1.T".split(".")[-1] == "T"`, same for w2/w3/wq/etc.), so ALL
    transposed weight types for a given row silently collide in the same
    `(row, "T")` grid cell and overwrite each other -- a transposed matrix
    measures a genuinely different orientation/dynamic (see
    gram_helper.py's orientation-policy docstring) and must not be pooled
    with non-transposed weights of a different type just because they
    share the "T" label.
    """
    parts = short_name.split("/")
    metric_name = parts[0]
    rest = parts[1:]

    expert_idx = None
    if rest and rest[0].startswith("ep_"):
        expert_idx = int(rest[0][len("ep_") :])
        rest = rest[1:]

    name = ".".join(rest)
    transposed_suffix = ""
    if name.endswith(".T"):
        name = name[: -len(".T")]
        transposed_suffix = ".T"

    if name == "tok_embeddings":
        return metric_name, expert_idx, _EMBED_ROW, "EMBED" + transposed_suffix
    if name == "output":
        return metric_name, expert_idx, _LM_HEAD_ROW, "LM_HEAD" + transposed_suffix

    dotted = name.split(".")
    row = int(dotted[1]) if dotted[0] == "layers" else _OTHER_ROW
    weight_type = dotted[-1].upper() + transposed_suffix
    return metric_name, expert_idx, row, weight_type


def _order_dense_columns(present: set[str]) -> list[str]:
    ordered = [c for c in _PREFERRED_DENSE_COLUMNS if c in present]
    extra = sorted(present - set(_PREFERRED_DENSE_COLUMNS))
    return ordered + extra


_MOE_COLUMN_RE = re.compile(r"^E(\d+)(.+)$")
_MOE_TYPE_ORDER = {"W1": 0, "W2": 1, "W3": 2}
_DIST_COLUMN_RE = re.compile(r"^(W\d+)\*$")


def _order_moe_columns(
    present: set[str], dist_weight_types: set[str] = frozenset()
) -> list[str]:
    dist_cols = [
        f"{wt}*"
        for wt in sorted(dist_weight_types, key=lambda w: _MOE_TYPE_ORDER.get(w, 99))
    ]

    def sort_key(col: str):
        m = _MOE_COLUMN_RE.match(col)
        expert_idx, weight_type = int(m.group(1)), m.group(2)
        return (expert_idx, _MOE_TYPE_ORDER.get(weight_type, 99), weight_type)

    return dist_cols + sorted(present, key=sort_key)


_EMBED_GRID_COLUMNS = ["EMBED", "LM_HEAD"]


def _order_embed_columns(present: set[str]) -> list[str]:
    ordered = [c for c in _EMBED_GRID_COLUMNS if c in present]
    extra = sorted(present - set(_EMBED_GRID_COLUMNS))
    return ordered + extra


class _Grid(NamedTuple):
    row_labels: list[int]
    col_labels: list[str]
    cells: dict[tuple[int, str], torch.Tensor]
    embed_cells: dict[str, torch.Tensor]
    dist_cells: dict[tuple[int, str], list[torch.Tensor]]


def _build_grids(gram_tensors: dict[str, torch.Tensor]) -> dict[str, _Grid]:
    """
    Bucket every tracked gram vector into one grid per (metric_name, dense|moe)
    pair, discovered dynamically from the tracked key names (unlike
    spectrum_logging.py's `_build_grids`, which can hardcode exactly 4 grids
    since "kind" is always update/param -- here "kind" is one of up to ~39
    metric names, so every dict is built with `defaultdict` instead of a
    fixed set of keys).
    """
    cells: dict[str, dict[tuple[int, str], torch.Tensor]] = defaultdict(dict)
    embed_cells: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    dist_cells: dict[str, dict[tuple[int, str], list[torch.Tensor]]] = defaultdict(dict)
    dist_types_seen: dict[str, set[str]] = defaultdict(set)
    cols_seen: dict[str, set[str]] = defaultdict(set)
    rows_seen: dict[str, set[int]] = defaultdict(set)

    for key, tensor in gram_tensors.items():
        short_name = key[len(_GRAM_PREFIX) :]
        metric_name, expert_idx, row, weight_type = _parse_gram_short_name(short_name)

        if expert_idx is not None:
            grid_name = f"moe_{metric_name}"
            dist_cells[grid_name].setdefault((row, weight_type), []).append(tensor)
            dist_types_seen[grid_name].add(weight_type)
            rows_seen[grid_name].add(row)
            if expert_idx >= _MAX_EXPERTS_TO_PLOT:
                continue
            col = f"E{expert_idx}{weight_type}"
            cells[grid_name][(row, col)] = tensor
            cols_seen[grid_name].add(col)
            continue

        grid_name = f"dense_{metric_name}"
        if row < 0:
            embed_cells[grid_name][weight_type] = tensor
            continue
        cells[grid_name][(row, weight_type)] = tensor
        cols_seen[grid_name].add(weight_type)
        rows_seen[grid_name].add(row)

    grids: dict[str, _Grid] = {}
    all_grid_names = set(cells) | set(embed_cells) | set(dist_cells)
    for grid_name in all_grid_names:
        grid_cells = cells.get(grid_name, {})
        grid_embed_cells = embed_cells.get(grid_name, {})
        grid_dist_cells = dist_cells.get(grid_name, {})
        if grid_name.startswith("dense_"):
            col_labels = _order_dense_columns(cols_seen[grid_name])
        else:
            col_labels = _order_moe_columns(
                cols_seen[grid_name], dist_types_seen[grid_name]
            )
        row_labels = sorted(rows_seen[grid_name])
        grids[grid_name] = _Grid(
            row_labels, col_labels, grid_cells, grid_embed_cells, grid_dist_cells
        )
    return grids


def _value_curve(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared by both drawing functions: (x, raw value) for one gram-vector
    tensor, downsampled to `_MAX_RESAMPLE_POINTS` if longer. Unlike
    spectrum_logging.py's `_spectrum_curves`, no normalization -- most gram
    vectors aren't non-negative/sortable, so there's no natural `s[0]` or
    cumulative-energy interpretation."""
    n = s.numel()
    if n > _MAX_RESAMPLE_POINTS:
        x = torch.linspace(0, n - 1, _MAX_RESAMPLE_POINTS)
        y = _resample_1d(s, _MAX_RESAMPLE_POINTS)
    else:
        x = torch.arange(n, dtype=torch.float32)
        y = s
    return x, y


def _y_range_with_padding(y_min: float, y_max: float) -> tuple[float, float]:
    if y_min == y_max:
        pad = 1.0 if y_min == 0 else abs(y_min) * 0.1
        return y_min - pad, y_max + pad
    pad = (y_max - y_min) * 0.05
    return y_min - pad, y_max + pad


def _draw_value_cell(ax, tensor: torch.Tensor) -> None:
    """Draw one cell's raw index-vs-value line, no tick labels -- illegible
    at grid-cell size; row/column headers carry the labeling. y-axis is
    auto-ranged per cell from the tensor's own min/max (no forced [0, 1]
    range like spectrum's normalized curves)."""
    x, y = _value_curve(tensor)
    y_np = y.numpy()
    ax.plot(x, y_np, color=_LINE_COLOR, linewidth=0.6)
    ax.set_xlim(0, max(tensor.numel() - 1, 1))
    ax.set_ylim(*_y_range_with_padding(float(y_np.min()), float(y_np.max())))


def _draw_value_distribution_cell(ax, tensors: list[torch.Tensor]) -> None:
    """
    Overlay every gathered expert's raw value curve (low alpha) in one
    (layer, weight_type) distribution cell -- same "spaghetti plot" idea as
    spectrum_logging.py's `_draw_distribution_cell`, adapted for a single
    raw-value series per expert instead of two normalized ones.
    """
    from matplotlib.collections import LineCollection

    alpha = max(0.03, min(0.3, 8.0 / len(tensors)))
    n_max = 1
    segments = []
    y_min, y_max = float("inf"), float("-inf")
    for s in tensors:
        n_max = max(n_max, s.numel())
        x, y = _value_curve(s)
        x_np, y_np = x.numpy(), y.numpy()
        segments.append(np.column_stack([x_np, y_np]))
        y_min = min(y_min, float(y_np.min()))
        y_max = max(y_max, float(y_np.max()))
    ax.add_collection(
        LineCollection(segments, colors=_LINE_COLOR, linewidths=0.4, alpha=alpha)
    )
    ax.set_xlim(0, n_max - 1)
    ax.set_ylim(*_y_range_with_padding(y_min, y_max))


def _render_grid(grid_name: str, grid: "_Grid"):
    """
    Runs in a worker PROCESS (see `_get_executor`). Builds one "atlas" image
    for a single gram metric name and wraps it as `wandb.Image` -- layout
    logic (figure sizing, row/column headers, embed-row handling, subplot
    geometry) copied from spectrum_logging.py's `_render_grid`; only the
    per-cell drawer and legend differ (one raw-value line instead of two
    normalized curves).
    """
    import wandb
    from matplotlib.lines import Line2D

    plt = _get_pyplot()
    embed_cols = _order_embed_columns(set(grid.embed_cells))
    has_embed_row = bool(embed_cols)
    nrows = len(grid.row_labels) + (2 if has_embed_row else 0)
    ncols = max(len(grid.col_labels), len(embed_cols))
    fig_width = max(ncols * 0.75, 3.0)
    spacer_units = 0.4
    content_units = len(grid.row_labels) + (
        (1.0 + spacer_units) if has_embed_row else 0.0
    )
    fig_height = max(content_units * 0.5, 3.0)
    height_ratios = (
        [1.0, spacer_units] + [1.0] * len(grid.row_labels)
        if has_embed_row
        else [1.0] * nrows
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        dpi=260,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    header_row = 2 if has_embed_row else 0
    row_index = {row: header_row + i for i, row in enumerate(grid.row_labels)}
    col_index = {col: j for j, col in enumerate(grid.col_labels)}

    for ax_row in axes:
        for ax in ax_row:
            ax.set_box_aspect(1)

    if has_embed_row:
        for j in range(ncols):
            ax = axes[0][j]
            if j < len(embed_cols):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(embed_cols[j], fontsize=5, pad=3)
            else:
                ax.axis("off")
            axes[1][j].axis("off")

    for i in range(header_row, nrows):
        for j in range(ncols):
            ax = axes[i][j]
            if j >= len(grid.col_labels):
                ax.axis("off")
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            if i == header_row:
                ax.set_title(grid.col_labels[j], fontsize=5, pad=3)
            if j == 0:
                ax.set_ylabel(
                    str(grid.row_labels[i - header_row]),
                    fontsize=5,
                    rotation=0,
                    ha="right",
                    va="center",
                )

    for j, col in enumerate(embed_cols):
        _draw_value_cell(axes[0][j], grid.embed_cells[col])

    for (row, col), tensor in grid.cells.items():
        _draw_value_cell(axes[row_index[row]][col_index[col]], tensor)

    for col in grid.col_labels:
        m = _DIST_COLUMN_RE.match(col)
        if not m:
            continue
        weight_type = m.group(1)
        j = col_index[col]
        for row in grid.row_labels:
            tensors = grid.dist_cells.get((row, weight_type))
            if not tensors:
                continue
            _draw_value_distribution_cell(axes[row_index[row]][j], tensors)

    fig.suptitle(grid_name, fontsize=9, y=0.995)
    legend_handles = [Line2D([0], [0], color=_LINE_COLOR, lw=1.5, label="value")]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.995),
        fontsize=7,
        ncol=1,
        framealpha=0.8,
    )
    header_in = 0.7
    top = 1 - header_in / fig_height
    fig.subplots_adjust(
        left=0.12, right=0.97, top=top, bottom=0.02, wspace=0.15, hspace=0.15
    )
    img = wandb.Image(fig)
    plt.close(fig)
    return img


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


_EXECUTOR = None


def _get_executor() -> ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        # "spawn" for the same reason as spectrum_logging.py's identical
        # helper -- a clean interpreter with no inherited CUDA context.
        ctx = multiprocessing.get_context("spawn")
        _EXECUTOR = ProcessPoolExecutor(max_workers=_MAX_WORKERS, mp_context=ctx)
    return _EXECUTOR


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

    if not config.enable_plot:
        return all_norms

    grids = _build_grids(gram_tensors)
    executor = _get_executor()
    futures = {
        grid_name: executor.submit(_render_grid, grid_name, grid)
        for grid_name, grid in grids.items()
    }
    for grid_name, future in futures.items():
        all_norms[f"plot_gram_{grid_name}"] = future.result()
    return all_norms
