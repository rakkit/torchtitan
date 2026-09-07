# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
VERBATIM COPY of radial_helper.py taken immediately BEFORE the batched-expert
refactor, kept so the unit suite can assert the 2-D path still matches the
code that was already validated end-to-end. Do not edit by hand; if
radial_helper changes intentionally, re-snapshot and say why.

Whole-tensor "radial dynamics" metrics -- how a weight's norm and direction
evolve under training. Unlike gram_helper.py's row-wise Gram-matrix
framework (which needs the raw momentum/gradient `V_raw` and is gated
behind `gram_level`), these only need the weight before/after this step's
update (`W_before`, `W_after` -- disco.py passes the same `pseudo_w`
already used as gram's `W_after`) and are all whole-tensor Frobenius-norm
-scale scalars, never a row-wise vector -- cheap enough to always compute
whenever any per-param logging fires at all, independent of `gram_level`
and `norms_to_log`.

Notation:
  W_t    = W_before, the weight before this step's update.
  W_t+1  = W_after, the weight after (disco.py's `pseudo_w` approximation).
  dW_t   = W_after - W_before, the realised displacement.
  r_t    = ||W_t||  (Frobenius norm -- "radius").
  a_t    = ||dW_t|| (Frobenius norm -- "raw step").
  q_t    = W_t / r_t,  v_t = dW_t / a_t  (unit directions).
  c_t    = <q_t, v_t>  ("radial_cosine" -- is the step outward-radial or
           tangential relative to the weight's own direction).

Four running accumulators (`raw_A2`, `angular_A1`, `angular_A2`, `R1`)
persist across steps in the caller-supplied `state` dict, mutated in
place -- disco.py stores the canonical values in
`self.state[p]["radial_state"]` (the same place `momentum_buffer` lives),
so they survive checkpoint save/restore via the optimizer's default
`state_dict()`/`load_state_dict()` with no extra plumbing. `R2(t)` from
the "radial error" formula is exactly the same running sum as `raw_A2` --
one accumulator serves both, so `R2` is not separately stored.
`relative_step` here is the same formula as gram_helper.py's
`U_relative_step_fro` -- expected, not an accidental duplicate: this one
is unconditional, that one is gated behind `gram_level`.

`alpha_fit`/`tau_fit` (fitting the angle-decay power law
`theta_t = C * (t + tau)^-alpha`) is a deliberately deferred follow-up --
it needs bounded/subsampled history storage and periodic (not per-step)
refitting to actually stay cheap at scale, unlike everything here, which
is a genuine O(1)-per-call update.
"""

from typing import NamedTuple

import torch

RADIAL_METRIC_NAMES: list[str] = [
    "radius",
    "raw_step",
    "relative_step",
    "radial_cosine",
    "tangent_fraction",
    "angle",
    "angle_from_cos",
    "radial_first_order",
    "radial_second_order",
    "radial_ratio",
    "raw_A2",
    "angular_A1",
    "angular_A2",
    "R1",
    "E_radial",
    # --- spectral extensions (see SpectralInputs / the module docstring) ---
    # sigma_max of the pre-/post-update weight and derived per-step ratios.
    "spectral_radius",
    "spectral_radius_next",
    "spectral_growth",
    "spectral_relative_step",
    # Alignment of the update with a norming covector of the weight, under
    # three induced operator norms.
    "radiality_rms_to_rms",
    "radiality_rms_to_inf",
    "radiality_l1_to_rms",
    # Distance between the normalised weight and the normalised update,
    # N(W/N(W) - U/N(U)), under the same four norms.
    "aus_frobenius",
    "aus_rms_to_rms",
    "aus_rms_to_inf",
    "aus_l1_to_rms",
]

# Metrics that cannot be computed without `SpectralInputs` (they need
# sigma_max, or the leading singular vectors, of the pre-update weight). They
# are still always present in the output -- filled with the degenerate 0
# sentinel -- because disco.py sizes its flat logging buffers from
# len(RADIAL_METRIC_NAMES) and needs the arity fixed.
_SPECTRAL_METRIC_NAMES: tuple[str, ...] = (
    "spectral_radius",
    "spectral_radius_next",
    "spectral_growth",
    "spectral_relative_step",
    "radiality_rms_to_rms",
    "aus_rms_to_rms",
)

# Metrics that need a genuine matrix (row/column structure). Undefined, and so
# sentinel-filled, for 1-D parameters.
_MATRIX_METRIC_NAMES: tuple[str, ...] = (
    "radiality_rms_to_inf",
    "radiality_l1_to_rms",
    "aus_rms_to_inf",
    "aus_l1_to_rms",
)

_ACCUMULATOR_NAMES: tuple[str, ...] = ("raw_A2", "angular_A1", "angular_A2", "R1")

# a_t below this fraction of r_t is treated as a degenerate (no real update)
# step -- see the valid_wu comment in calculate_radial_metrics for why a
# relative floor is needed instead of a bare a_t > 0 check.
_REL_DEGENERACY_EPS = 1e-6


class SpectralInputs(NamedTuple):
    """Spectral quantities `calculate_radial_metrics` cannot get on its own.

    Two of these are genuinely free at every disco.py call site and one is not:

      * `sigma_after` -- `calculate_norm(pseudo_w)["spectrum"][0]`.
      * `sigma_update` -- `calculate_norm(-lr*u)["spectrum"][0]`.
      * `sigma_before`, `u1_before`, `v1_before` -- NOT free. `calculate_norm`
        is never called on the pre-update weight in any of the four `step_*`
        paths, so these come from optimizers/power_iteration.py, warm-started
        across logging events by the caller.

    Any field may be None; every metric that depends on a missing field falls
    back to the degenerate 0 sentinel, so the output arity never changes.
    """

    sigma_before: torch.Tensor | None = None
    u1_before: torch.Tensor | None = None
    v1_before: torch.Tensor | None = None
    sigma_after: torch.Tensor | None = None
    sigma_update: torch.Tensor | None = None
    # sigma_max(W/sigma_max(W) - U/sigma_max(U)), for `aus_rms_to_rms`. Needs a
    # decomposition of a matrix this function does not otherwise form, so the
    # caller computes it (it already owns the power-iteration machinery) rather
    # than this module allocating an extra m-by-n temporary per call.
    aus_sigma: torch.Tensor | None = None


def _guarded_radiality(
    raw: torch.Tensor,
    weight_norm_unscaled: torch.Tensor,
    update_norm_unscaled: torch.Tensor,
) -> torch.Tensor:
    """Deterministic 0 sentinel on a degenerate step.

    Same convention and the same relative floor as `radial_cosine` /
    `radial_ratio` below: an update whose magnitude is negligible against the
    weight's own scale is dominated by floating-point reduction-order noise,
    which dividing by it amplifies into large run-to-run-inconsistent swings.
    Reusing `_REL_DEGENERACY_EPS` rather than introducing a second epsilon
    keeps every degeneracy decision in this module on one threshold.
    """
    valid = (weight_norm_unscaled > 0) & (
        update_norm_unscaled > _REL_DEGENERACY_EPS * weight_norm_unscaled
    )
    return torch.where(valid, raw, torch.zeros_like(raw))


def _axis_reductions(
    W: torch.Tensor, U: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row (`dim=1`) or per-column (`dim=0`) squared norms and cross dots.

    Returns `(sq_W, sq_U, dot_WU)`. These three feed BOTH the radiality and the
    aus metric for that axis, which is why they are computed once here instead
    of separately: the aus norm expands without ever forming the difference
    matrix,

        ||axis_i(W/a - U/b)||^2 = sq_W[i]/a^2 + sq_U[i]/b^2 - 2*dot[i]/(a*b),

    so a full m-by-n temporary is never allocated for it.
    """
    return (W * W).sum(dim), (U * U).sum(dim), (W * U).sum(dim)


def _axis_radiality(
    sq_W: torch.Tensor, sq_U: torch.Tensor, dot: torch.Tensor
) -> torch.Tensor:
    """Radiality along one axis: the update's alignment with the weight's
    largest row/column, which is where that induced norm's norming covector
    concentrates.

    Returns `(radiality, max_norm_W, max_norm_U)` -- the two maxima come back
    because `_axis_aus` needs exactly the same pair, and recomputing them would
    repeat a reduction over the whole axis.

    The induced norms' dimension factors (`sqrt(d_in)` for rms->inf,
    `1/sqrt(d_out)` for l1->rms) appear in both `N(W)` and `N(U)` and cancel in
    the ratio, so only the unscaled L2 norms appear here. `argmax` picks the
    first maximal row/column at a tie -- one valid subgradient selection at a
    non-smooth point.
    """
    idx = torch.argmax(sq_W)
    max_W = sq_W[idx].clamp_min(0).sqrt()
    max_U = sq_U.max().clamp_min(0).sqrt()
    tiny = torch.finfo(sq_W.dtype).tiny
    raw = dot[idx] / (max_W * max_U).clamp_min(tiny)
    return _guarded_radiality(raw, max_W, max_U), max_W, max_U


def _axis_aus(
    sq_W: torch.Tensor,
    sq_U: torch.Tensor,
    dot: torch.Tensor,
    max_W: torch.Tensor,
    max_U: torch.Tensor,
) -> torch.Tensor:
    """`N(W/N(W) - U/N(U))` along one axis, computed from the reductions alone.

    As in `_axis_radiality`, the dimension factor cancels: it scales `N(W)`,
    `N(U)` and `N(difference)` identically, so what is left is the max over the
    axis of the normalised difference's L2 norm.
    """
    tiny = torch.finfo(sq_W.dtype).tiny
    a = max_W.clamp_min(tiny)
    b = max_U.clamp_min(tiny)
    per_axis = sq_W / (a * a) + sq_U / (b * b) - 2.0 * dot / (a * b)
    raw = per_axis.clamp_min(0.0).max().sqrt()
    return _guarded_radiality(raw, max_W, max_U)


def new_radial_state(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    shape: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    """Fresh, zero-initialized accumulator state. `shape=()` (the default)
    for a single tracked tensor; `shape=(num_local_experts,)` for expert
    params, where each expert index needs its own independent accumulators
    (each has its own `W_before`/`W_after` pair) -- index into the result
    per-expert (e.g. `state["raw_A2"][ep_idx]`) when calling
    `calculate_radial_metrics`, which mutates whatever 0-d view it's given
    in place."""
    return {
        name: torch.zeros(shape, device=device, dtype=dtype)
        for name in _ACCUMULATOR_NAMES
    }


@torch.no_grad()
def calculate_radial_metrics(
    W_before: torch.Tensor,
    W_after: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    spectral: "SpectralInputs | None" = None,
    transpose: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Returns all of `RADIAL_METRIC_NAMES` as a flat dict of 0-d tensors.

    `state` holds the 4 running accumulators (`raw_A2`, `angular_A1`,
    `angular_A2`, `R1`), mutated in place: this call's `raw_A2`/
    `angular_A1`/`angular_A2`/`R1`/`E_radial` outputs reflect `Sum_{i<t}`
    (i.e. NOT including this step's own contribution -- the correct
    semantics for these "history so far" metrics), then `state` is updated
    afterward so the NEXT call sees this step's contribution included.

    `spectral` supplies the sigma_max / leading-singular-vector quantities this
    function cannot derive itself (see `SpectralInputs`). `transpose` matches
    norm_helper.calculate_norm's flag and must be passed the same value at a
    given call site, so that `radiality_rms_to_inf` / `aus_rms_to_inf` refer to
    the same axis as the `rms_to_inf` norm logged alongside them.

    The returned key set is exactly `RADIAL_METRIC_NAMES`, always, for every
    input shape and regardless of whether `spectral` was given -- unavailable
    metrics take the same deterministic 0 sentinel as the other degenerate
    cases here. disco.py sizes its flat logging buffers from
    `len(RADIAL_METRIC_NAMES)`, so the arity must not depend on the input.
    """
    if isinstance(W_before, torch.nn.Parameter):
        W_before = W_before.data
    if isinstance(W_after, torch.nn.Parameter):
        W_after = W_after.data
    if W_before.dtype in (torch.float16, torch.bfloat16):
        W_before = W_before.float()
    if W_after.dtype in (torch.float16, torch.bfloat16):
        W_after = W_after.float()

    dtype = W_before.dtype
    tiny = torch.finfo(dtype).tiny

    U = W_after - W_before
    r_t = W_before.norm()
    r_next = W_after.norm()
    a_t = U.norm()

    # Relative (not absolute/exact-zero) floor on a_t: an update whose
    # magnitude is numerically negligible compared to the weight's own
    # scale (e.g. a near-zero-lr step at the tail of a decay schedule)
    # should report the same deterministic degenerate sentinel regardless
    # of which parallelism strategy computed it. `a_t > 0` alone only
    # catches the literal-zero case -- below this relative threshold, a_t
    # is dominated by ordinary floating-point reduction-order noise (e.g.
    # DDP's all-reduce vs FSDP's all-gather summing gradients in a
    # different order), which radial_cosine/radial_ratio (both divide by
    # a_t or a_t^2) amplify into large, run-to-run-inconsistent swings
    # even though the underlying update is physically negligible.
    valid_wu = (r_t > 0) & (a_t > _REL_DEGENERACY_EPS * r_t)
    valid_ww = (r_t > 0) & (r_next > 0)

    relative_step = a_t / r_t.clamp_min(tiny)

    dot_wu = (W_before * U).sum()
    radial_cosine_raw = (dot_wu / (r_t * a_t).clamp_min(tiny)).clamp(-1.0, 1.0)
    # Explicit degenerate-case sentinel (0, an undefined angle) rather than
    # letting the tiny floor alone produce an arbitrary non-zero value --
    # same convention as _row_normalise/_effective_rank's torch.where
    # guards in gram_helper.py.
    radial_cosine = torch.where(
        valid_wu, radial_cosine_raw, torch.zeros_like(radial_cosine_raw)
    )
    tangent_fraction = (1.0 - radial_cosine * radial_cosine).clamp_min(0.0).sqrt()

    # Canonical angle via atan2, not acos: acos's derivative blows up near
    # cos=1, so small angles (the common case most training steps) lose
    # precision in fp32 -- nearby small angles round to indistinguishable
    # cosine values. atan2 doesn't have this issue. Mathematically the same
    # quantity as arccos(<q_t, q_t+1>) -- verified via the 2D-trigonometry
    # identity: place q_t at (r_t, 0); the step lands W_t+1 at
    # (r_t + a_t*c_t, a_t*sqrt(1-c_t^2)), whose angle from the x-axis is
    # exactly this atan2 expression -- just computed via a more numerically
    # robust path. Gated on `valid_ww` (matching angle_from_cos below), not
    # `valid_wu` -- deliberately NOT the relative-degeneracy floor above:
    # unlike radial_cosine/radial_ratio, atan2's inputs (a_t*tangent_fraction,
    # r_t + a_t*radial_cosine) stay well-conditioned as a_t -> 0 (numerator
    # -> 0, denominator -> r_t > 0), so `angle` doesn't inherit the
    # near-zero-a_t noise-amplification these other metrics have, and
    # doesn't need the same protection.
    angle_raw = torch.atan2(a_t * tangent_fraction, r_t + a_t * radial_cosine)
    angle = torch.where(valid_ww, angle_raw, torch.zeros_like(angle_raw))

    # Independent sanity check against `angle` above (same quantity, via
    # the acos formula instead of atan2) -- not fed into the accumulators,
    # kept purely to catch a geometry/implementation bug if it ever
    # meaningfully diverges from `angle`.
    dot_ww = (W_before * W_after).sum()
    cos_angle = (dot_ww / (r_t * r_next).clamp_min(tiny)).clamp(-1.0, 1.0)
    angle_from_cos = torch.where(
        valid_ww, torch.arccos(cos_angle), torch.zeros_like(cos_angle)
    )

    # Algebraically == 2*a_t*r_t*radial_cosine (c_t = dot_wu/(r_t*a_t)),
    # but computed directly from the dot product already at hand: avoids a
    # divide-then-remultiply round trip, and stays well-defined even when
    # r_t or a_t is exactly 0 (dot_wu is 0 there too, no separate
    # degenerate-case guard needed, unlike radial_cosine).
    radial_first_order = 2.0 * dot_wu
    radial_second_order = a_t * a_t
    # Away from the degenerate regime, radial_ratio is deliberately NOT
    # given a relative-floor treatment the way genuinely-unbounded ratios
    # elsewhere in gram_helper.py are: it's *supposed* to swing large or
    # small depending on which regime dominates (that's its whole
    # diagnostic purpose). But when the step itself is degenerate (a_t
    # negligible vs r_t -- see valid_wu above), both radial_first_order and
    # radial_second_order are individually noise-dominated, and dividing
    # noise by noise-squared is pure amplification, not signal -- gate it
    # to the same deterministic 0 sentinel as radial_cosine so it doesn't
    # report large, run-to-run-inconsistent swings on a step where nothing
    # meaningful happened.
    radial_ratio_raw = radial_first_order.abs() / radial_second_order.clamp_min(tiny)
    radial_ratio = torch.where(
        valid_wu, radial_ratio_raw, torch.zeros_like(radial_ratio_raw)
    )

    raw_A2 = state["raw_A2"].clone()
    angular_A1 = state["angular_A1"].clone()
    angular_A2 = state["angular_A2"].clone()
    R1 = state["R1"].clone()
    # Same "meant to swing large" reasoning as radial_ratio -- additive
    # +tiny (not a relative floor) is intentional.
    E_radial = R1 / (raw_A2 + tiny)

    state["raw_A2"].add_(radial_second_order)
    state["angular_A1"].add_(angle)
    state["angular_A2"].add_(angle * angle)
    state["R1"].add_(radial_first_order)

    # ---------------- spectral / radiality extensions ----------------
    zero = torch.zeros_like(r_t)
    extra: dict[str, torch.Tensor] = {name: zero for name in RADIAL_METRIC_NAMES[15:]}

    # ||W/||W||_F - U/||U||_F||_F == sqrt(2 - 2*cos) with cos == radial_cosine,
    # so this needs no second pass over the data and no difference tensor. It
    # is defined for any shape, unlike the operator-norm variants below.
    extra["aus_frobenius"] = _guarded_radiality(
        (2.0 - 2.0 * radial_cosine).clamp_min(0.0).sqrt(), r_t, a_t
    )

    if W_before.ndim == 2:
        Wm, Um = (W_before, U) if not transpose else (W_before.T, U.T)
        # rows of the (possibly transposed) matrix -> the rms->inf geometry;
        # columns -> the l1->rms geometry. Both sets of reductions are shared
        # between that axis's radiality and its aus norm.
        row_sq_W, row_sq_U, row_dot = _axis_reductions(Wm, Um, 1)
        col_sq_W, col_sq_U, col_dot = _axis_reductions(Wm, Um, 0)

        rad_row, row_max_W, row_max_U = _axis_radiality(row_sq_W, row_sq_U, row_dot)
        rad_col, col_max_W, col_max_U = _axis_radiality(col_sq_W, col_sq_U, col_dot)
        extra["radiality_rms_to_inf"] = rad_row
        extra["radiality_l1_to_rms"] = rad_col
        extra["aus_rms_to_inf"] = _axis_aus(
            row_sq_W, row_sq_U, row_dot, row_max_W, row_max_U
        )
        extra["aus_l1_to_rms"] = _axis_aus(
            col_sq_W, col_sq_U, col_dot, col_max_W, col_max_U
        )

    if spectral is not None:
        sig_w = spectral.sigma_before
        sig_next = spectral.sigma_after
        sig_u = spectral.sigma_update
        u1, v1 = spectral.u1_before, spectral.v1_before

        if sig_w is not None:
            sig_w = sig_w.to(dtype)
            extra["spectral_radius"] = sig_w
            if sig_next is not None:
                sig_next = sig_next.to(dtype)
                extra["spectral_radius_next"] = sig_next
                extra["spectral_growth"] = torch.where(
                    sig_w > 0, sig_next / sig_w.clamp_min(tiny), torch.zeros_like(sig_w)
                )
            if sig_u is not None:
                sig_u = sig_u.to(dtype)
                # Operator-norm analogue of `relative_step` (which is the same
                # ratio in Frobenius norm).
                extra["spectral_relative_step"] = torch.where(
                    sig_w > 0, sig_u / sig_w.clamp_min(tiny), torch.zeros_like(sig_w)
                )
                if u1 is not None and v1 is not None and W_before.ndim == 2:
                    # rms->rms radiality. A norming covector of
                    # N(W) = sqrt(d_in/d_out)*||W||_op is
                    # sqrt(d_in/d_out)*u1 v1^T; the dimension factor cancels
                    # against N(U), leaving u1^T U v1 / ||U||_op.
                    u1 = u1.to(dtype)
                    v1 = v1.to(dtype)
                    extra["radiality_rms_to_rms"] = _guarded_radiality(
                        (u1 @ (U @ v1)) / sig_u.clamp_min(tiny), sig_w, sig_u
                    )
        elif sig_next is not None:
            extra["spectral_radius_next"] = sig_next.to(dtype)

        if spectral.aus_sigma is not None and sig_w is not None and sig_u is not None:
            # N(W/N(W) - U/N(U)) for the rms->rms norm. The dimension factor
            # cancels exactly as above, so this is sigma_max of the normalised
            # difference -- which needs a second decomposition and therefore
            # has to be supplied by the caller rather than computed here.
            extra["aus_rms_to_rms"] = _guarded_radiality(
                spectral.aus_sigma.to(dtype), sig_w, sig_u
            )

    return {
        "radius": r_t,
        "raw_step": a_t,
        "relative_step": relative_step,
        "radial_cosine": radial_cosine,
        "tangent_fraction": tangent_fraction,
        "angle": angle,
        "angle_from_cos": angle_from_cos,
        "radial_first_order": radial_first_order,
        "radial_second_order": radial_second_order,
        "radial_ratio": radial_ratio,
        "raw_A2": raw_A2,
        "angular_A1": angular_A1,
        "angular_A2": angular_A2,
        "R1": R1,
        "E_radial": E_radial,
        **extra,
    }
