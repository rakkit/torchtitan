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
`SpectrumLoggingConfig` (built from `OptimizersContainer.Config`'s
`enable_spectrum_plot`/`enable_spectrum_export` fields -- i.e.
`config.optimizer.enable_spectrum_plot`/`enable_spectrum_export` in the job
config -- see `components/optimizer.py`'s
`OptimizersContainer.spectrum_logging_config`):

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
produced by `OptimizersContainer.get_parameter_norms()`
(components/optimizer.py), it replaces every raw spectrum tensor with
whatever's enabled, so components/metrics.py never needs any spectrum-
specific logic -- it just forwards whatever's left in the dict to
`wandb.log()`.
"""

import functools
import multiprocessing
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any, NamedTuple

import numpy as np
import torch

__all__ = [
    "SpectrumLoggingConfig",
    "process_norms_for_logging",
]


class SpectrumLoggingConfig(NamedTuple):
    """
    Bundles the settings `process_norms_for_logging` needs, so callers pass
    one object instead of a growing list of loose bool/str arguments.
    `enable_plot`/`enable_export` are set from `OptimizersContainer.Config`
    directly in its `__init__`; `export_dir` depends on the top-level
    `config.dump_folder`, so trainer.py patches that in afterward. Stored as
    `OptimizersContainer.spectrum_logging_config`.
    """

    enable_plot: bool = True
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
_MAX_WORKERS = 4


_PYPLOT = None


def _get_pyplot():
    """
    Import matplotlib.pyplot exactly once, forcing the non-interactive Agg
    backend *before* pyplot is ever imported anywhere. Calling
    `matplotlib.use()` after pyplot has already been imported (e.g. by some
    other part of the training pipeline, with whatever backend it picked by
    default) silently no-ops in some matplotlib versions instead of actually
    switching backends — on a headless node that's a classic cause of
    figures that "exist" but never render/upload correctly.
    """
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


_SPEC_COLOR = "tab:blue"
_ENERGY_COLOR = "tab:orange"


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
_MAX_EXPERTS_TO_PLOT = 4

# Fixed preferred column order for the dense grid — anything else tracked
# per-layer (e.g. the MoE router's "gate") is appended after these, sorted.
_PREFERRED_DENSE_COLUMNS = ["WQ", "WK", "WV", "WO", "W1", "W2", "W3"]

# Sentinel rows for the dense grid's top, one each for embedding/lm-head/
# anything else that isn't inside a "layers.N." prefix. These aren't real
# matrix "types" or layers, so `_build_grids` routes them (any row < 0) into
# their own small standalone grid instead of the per-layer dense one — see
# `_build_grids`, `_order_embed_columns`.
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


def _parse_spectrum_short_name(short_name: str) -> tuple[str, int | None, int, str]:
    """
    Parse a `short_name` (a `track_spectrum_` key with that prefix already
    stripped) into `(kind, expert_idx, row, weight_type)`:
    - "update/layers.9.attention.wq"        -> ("update", None, 9, "WQ")
    - "update/ep_3/layers.9.moe.experts.w1" -> ("update", 3, 9, "W1")
    - "update/tok_embeddings"                -> ("update", None, _EMBED_ROW, "EMBED")
    - "update/output"                        -> ("update", None, _LM_HEAD_ROW, "LM_HEAD")

    `weight_type` is just the last dot-separated component of the cleaned
    parameter name, uppercased — this is deliberately generic (not a
    hardcoded list of known module paths) so any per-layer matrix type
    (attention, FFN, MoE shared-experts, MoE router, or something added
    later) lands in the dense grid automatically; only presence of an
    "ep_<N>" segment (added by disco.py's step_experts for routed MoE
    experts, see disco.py) routes an entry to the MoE grid instead. For the
    special rows (row < 0), `weight_type` is informational only — see
    `_build_grids`, which places them in the first real column instead.
    """
    parts = short_name.split("/")
    kind = parts[0]
    rest = parts[1:]

    expert_idx = None
    if rest and rest[0].startswith("ep_"):
        expert_idx = int(rest[0][len("ep_") :])
        rest = rest[1:]

    name = ".".join(rest)
    if name == "tok_embeddings":
        return kind, expert_idx, _EMBED_ROW, "EMBED"
    if name == "output":
        return kind, expert_idx, _LM_HEAD_ROW, "LM_HEAD"

    dotted = name.split(".")
    row = int(dotted[1]) if dotted[0] == "layers" else _OTHER_ROW
    weight_type = dotted[-1].upper()
    return kind, expert_idx, row, weight_type


def _order_dense_columns(present: set[str]) -> list[str]:
    ordered = [c for c in _PREFERRED_DENSE_COLUMNS if c in present]
    extra = sorted(present - set(_PREFERRED_DENSE_COLUMNS))
    return ordered + extra


_MOE_COLUMN_RE = re.compile(r"^E(\d+)(.+)$")
_MOE_TYPE_ORDER = {"W1": 0, "W2": 1, "W3": 2}

# Distribution columns ("W1*", "W2*", "W3*") summarize *all* gathered
# experts for a (layer, weight_type) via an overlay -- see
# `_draw_distribution_cell` -- distinct from the individual "E{idx}{type}"
# columns, which are capped at `_MAX_EXPERTS_TO_PLOT`. Both curves
# (spectrum + cumulative energy) share one cell, same as every other cell
# in the grid; the "*" suffix is just to avoid colliding with the dense
# grid's own literal "W1"/"W2"/"W3" columns (different grid, but same
# `_render_grid` code path inspects column labels generically).
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
    # Embedding/lm-head (and anything else not inside a "layers.N." prefix)
    # have no meaningful layer index or matrix "type" they share with the
    # rest of the grid, so they're kept separate here ({col_label: tensor})
    # rather than forced into a (row, col) cell of the main grid -- see
    # `_render_grid`, which draws them as their own dedicated top row with
    # their own per-cell titles instead of wasting a row/column in the main
    # grid, but still folds them into the same image. Always empty for the
    # MoE grids.
    embed_cells: dict[str, torch.Tensor]
    # MoE grids only: {(row, weight_type): [every gathered expert's raw
    # tensor]} -- unlike `cells`, this is NEVER filtered by
    # `_MAX_EXPERTS_TO_PLOT`, since it feeds the "W1*"/"W2*"/"W3*"
    # distribution columns (see `_draw_distribution_cell`), which are meant
    # to show the pattern across *all* experts even when only a few get
    # their own column. Always empty for the dense grids.
    dist_cells: dict[tuple[int, str], list[torch.Tensor]]


def _build_grids(spectrum_tensors: dict[str, torch.Tensor]) -> dict[str, _Grid]:
    """
    Bucket every tracked spectrum tensor into up to 4 grids ("dense_update",
    "dense_param", "moe_update", "moe_param"), each a `_Grid`. See module
    docstring for what rows/columns mean in each grid.
    """
    cells: dict[str, dict[tuple[int, str], torch.Tensor]] = {
        "dense_update": {},
        "dense_param": {},
        "moe_update": {},
        "moe_param": {},
    }
    embed_cells: dict[str, dict[str, torch.Tensor]] = {
        "dense_update": {},
        "dense_param": {},
    }
    dist_cells: dict[str, dict[tuple[int, str], list[torch.Tensor]]] = {
        "moe_update": {},
        "moe_param": {},
    }
    dist_types_seen: dict[str, set[str]] = {"moe_update": set(), "moe_param": set()}
    cols_seen: dict[str, set[str]] = {k: set() for k in cells}
    rows_seen: dict[str, set[int]] = {k: set() for k in cells}

    for key, tensor in spectrum_tensors.items():
        short_name = key.replace("track_spectrum_", "")
        kind, expert_idx, row, weight_type = _parse_spectrum_short_name(short_name)

        if expert_idx is not None:
            grid_name = f"moe_{kind}"
            # Unconditional: every gathered expert feeds the distribution
            # columns, regardless of the per-column cap below.
            dist_cells[grid_name].setdefault((row, weight_type), []).append(tensor)
            dist_types_seen[grid_name].add(weight_type)
            rows_seen[grid_name].add(row)
            if expert_idx >= _MAX_EXPERTS_TO_PLOT:
                continue
            col = f"E{expert_idx}{weight_type}"
            cells[grid_name][(row, col)] = tensor
            cols_seen[grid_name].add(col)
            continue

        grid_name = f"dense_{kind}"
        if row < 0:
            embed_cells[grid_name][weight_type] = tensor
            continue
        cells[grid_name][(row, weight_type)] = tensor
        cols_seen[grid_name].add(weight_type)
        rows_seen[grid_name].add(row)

    grids = {}
    for grid_name, grid_cells in cells.items():
        grid_embed_cells = embed_cells.get(grid_name, {})
        grid_dist_cells = dist_cells.get(grid_name, {})
        if not grid_cells and not grid_embed_cells and not grid_dist_cells:
            continue
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


def _spectrum_curves(
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared by both drawing functions: (x, normalized spectrum, cumulative
    energy) for one raw singular-value tensor, downsampled to
    `_MAX_RESAMPLE_POINTS` if longer. Both curves land in `[0, 1]`, which is
    exactly why no twin axis is needed to draw them together -- see
    `_draw_spectrum_cell`."""
    n = s.numel()
    s_norm = s / s[0].clamp_min(1e-12)
    energy = torch.cumsum(s * s, dim=0)
    energy = energy / energy[-1].clamp_min(1e-12)
    if n > _MAX_RESAMPLE_POINTS:
        x = torch.linspace(0, n - 1, _MAX_RESAMPLE_POINTS)
        s_norm = _resample_1d(s_norm, _MAX_RESAMPLE_POINTS)
        energy = _resample_1d(energy, _MAX_RESAMPLE_POINTS)
    else:
        x = torch.arange(n, dtype=torch.float32)
    return x, s_norm, energy


def _draw_spectrum_cell(ax, tensor: torch.Tensor) -> None:
    """Draw one cell's normalized-spectrum + cumulative-energy curves (the
    same two curves as the old per-parameter plot), with no tick labels —
    illegible at grid-cell size; row/column headers carry the labeling.
    Both curves share one Axes (no `twinx()`): they're both already in
    `[0, 1]`, and a second Axes was only ever needed for a differently
    colored/labeled right-hand tick axis, which grid cells don't draw at
    all -- `twinx()` itself has real per-call overhead (a whole second
    Axes, transform sharing, figure bookkeeping) that adds up when called
    for every cell in a large grid."""
    x, s_norm, energy = _spectrum_curves(tensor)
    ax.plot(x, s_norm.numpy(), color=_SPEC_COLOR, linewidth=0.6)
    ax.plot(x, energy.numpy(), color=_ENERGY_COLOR, linewidth=0.6)
    ax.set_xlim(0, max(tensor.numel() - 1, 1))
    ax.set_ylim(0, 1.05)


def _draw_distribution_cell(ax, tensors: list[torch.Tensor]) -> None:
    """
    Overlay every gathered expert's spectrum (blue) and cumulative energy
    (orange) curves (low alpha) in one (layer, weight_type) distribution
    cell -- "spaghetti plot": darker/denser regions are where experts
    agree, giving a sense of the spread across *all* experts even though
    only `_MAX_EXPERTS_TO_PLOT` get their own individual column. Both curve
    families share one Axes, same as `_draw_spectrum_cell`.
    """
    from matplotlib.collections import LineCollection

    # Alpha tuned so a handful of experts are each individually visible but
    # a couple hundred overlapping ones read as a density gradient rather
    # than a single opaque blob.
    alpha = max(0.03, min(0.3, 8.0 / len(tensors)))
    n_max = 1
    spec_segments = []
    energy_segments = []
    for s in tensors:
        n_max = max(n_max, s.numel())
        x, s_norm, energy = _spectrum_curves(s)
        # One artist for all ~256 lines instead of 256 separate ax.plot()
        # calls -- each Line2D carries real per-call bookkeeping overhead
        # (style/transform/clip state, bbox updates) that dominates at this
        # scale; a LineCollection amortizes that to a single call. Same
        # rendered pixels either way, ~34x less overhead measured in
        # testing (30s -> <1s for one cell's worth of 256 lines).
        x_np = x.numpy()
        spec_segments.append(np.column_stack([x_np, s_norm.numpy()]))
        energy_segments.append(np.column_stack([x_np, energy.numpy()]))
    ax.add_collection(
        LineCollection(spec_segments, colors=_SPEC_COLOR, linewidths=0.4, alpha=alpha)
    )
    ax.add_collection(
        LineCollection(
            energy_segments, colors=_ENERGY_COLOR, linewidths=0.4, alpha=alpha
        )
    )
    ax.set_xlim(0, n_max - 1)
    ax.set_ylim(0, 1.05)


def _render_grid(grid_name: str, grid: "_Grid"):
    """
    Runs in a worker PROCESS (see `_get_executor`). Builds one "atlas" image
    and wraps it as `wandb.Image` directly here, same rationale as the old
    per-parameter version: avoids routing PNG bytes back through the main
    process, and `wandb.Image`'s internal `savefig` then runs in parallel
    across workers instead of serially.

    Layout: `len(grid.row_labels)` x `len(grid.col_labels)` small-multiples
    cells, one column header row on top. If `grid.embed_cells` is non-empty
    (the dense grids only), two extra rows are prepended above that: one for
    embedding/lm-head — each with its own individual title instead of the
    shared column header (they aren't WQ/WK/../W3-typed), any unused columns
    in that row are hidden outright rather than left as empty boxes — and
    one short, fully blank spacer row right below it, so the embed row
    doesn't crowd the shared column-header row (which shifts down to sit
    above the first real layer row instead).
    """
    import wandb
    from matplotlib.lines import Line2D

    plt = _get_pyplot()
    embed_cols = _order_embed_columns(set(grid.embed_cells))
    has_embed_row = bool(embed_cols)
    nrows = len(grid.row_labels) + (2 if has_embed_row else 0)
    ncols = max(len(grid.col_labels), len(embed_cols))
    # Small per-cell footprint — these are meant to be scanned for gross
    # structure, not read precisely; zoom in the W&B image viewer for that.
    fig_width = max(ncols * 0.75, 3.0)
    # Row-height "units": 1.0 per real row, plus (for the embed row) 1.0 for
    # itself and 0.4 for the short blank spacer below it that keeps it from
    # crowding the shared column-header row.
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

    # fig_width/fig_height above use different per-unit cell sizes (0.75 per
    # column vs. 0.5 per row), so the nominal cell box is ~1.5:1 (wider than
    # tall) -- force every cell's actual drawn box back to square, so curve
    # slopes/shapes are visually comparable across cells regardless of grid
    # shape (many rows x few columns, or vice versa).
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
                ax.axis("off")  # unused slot in the embed row -- no empty box
            axes[1][j].axis("off")  # blank spacer row -- no box, no ticks

    for i in range(header_row, nrows):
        for j in range(ncols):
            ax = axes[i][j]
            if j >= len(grid.col_labels):
                ax.axis("off")  # embed row is wider than the real grid -- rare
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
        _draw_spectrum_cell(axes[0][j], grid.embed_cells[col])

    for (row, col), tensor in grid.cells.items():
        _draw_spectrum_cell(axes[row_index[row]][col_index[col]], tensor)

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
            _draw_distribution_cell(axes[row_index[row]][j], tensors)

    fig.suptitle(grid_name, fontsize=9, y=0.995)
    legend_handles = [
        Line2D([0], [0], color=_SPEC_COLOR, lw=1.5, label=r"$\sigma_i/\sigma_1$"),
        Line2D([0], [0], color=_ENERGY_COLOR, lw=1.5, label="E(k)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.995),
        fontsize=7,
        ncol=2,
        framealpha=0.8,
    )
    # Reserve a fixed *absolute* height for the title/legend header, rather
    # than a fixed fraction of the figure — with fig_height scaling up to
    # 20+in for a 40-layer grid, a fixed fraction (e.g. top=0.92) leaves a
    # huge, pointless gap above the actual cells.
    header_in = 0.7
    top = 1 - header_in / fig_height
    fig.subplots_adjust(
        left=0.12, right=0.97, top=top, bottom=0.02, wspace=0.15, hspace=0.15
    )
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _is_spectrum_entry(key: str, v: Any) -> bool:
    return isinstance(v, torch.Tensor) and v.numel() > 1 and "track_spectrum_" in key


_EXECUTOR = None


def _get_executor() -> ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        # "spawn" (not the Linux default "fork") gives each worker a clean,
        # fresh interpreter with no inherited CUDA context. The parent is a
        # training process that almost certainly has CUDA initialized, and
        # forking a CUDA-initialized process is a well-known source of
        # hangs/crashes if the child ever touches CUDA — these workers never
        # do (inputs are moved to CPU before being sent, see below), but
        # "spawn" removes the risk entirely rather than relying on that
        # staying true. The one-time slower worker startup is amortized: the
        # pool is persistent, built once and reused for the rest of the run.
        ctx = multiprocessing.get_context("spawn")
        _EXECUTOR = ProcessPoolExecutor(max_workers=_MAX_WORKERS, mp_context=ctx)
    return _EXECUTOR


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

    if not config.enable_plot:
        return all_norms

    grids = _build_grids(spectrum_tensors)
    executor = _get_executor()
    futures = {
        grid_name: executor.submit(_render_grid, grid_name, grid)
        for grid_name, grid in grids.items()
    }
    for grid_name, future in futures.items():
        all_norms[f"plot_{grid_name}"] = future.result()
    return all_norms
