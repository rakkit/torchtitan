# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""failfirst.py's runner over the p4 mutants (column scales, spectral cadence, gnoise update
metrics). Run from the tree root:
    python tests/unit_tests/gpu/gnoise/failfirst_p4.py [mutant ids] > result.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import failfirst  # noqa: E402

P = "torchtitan/optimizers/disco/presets.py"
CO = "torchtitan/optimizers/container.py"
ME = "torchtitan/optimizers/disco/metrics.py"
RH = "torchtitan/optimizers/helpers/radial_helper.py"
G = "torchtitan/optimizers/gnoise.py"
GR = "torchtitan/optimizers/gnoise_research.py"
CS = "tests/unit_tests/cpu/optimizers/test_angular_muown_col_scale.py"
SN = "tests/unit_tests/cpu/optimizers/test_spectral_norm_freq.py"
MU = "tests/unit_tests/gpu/disco/test_metrics_union.py"
UM = "tests/unit_tests/gpu/gnoise/test_gnoise_update_metrics.py"
NS = "tests/unit_tests/cpu/test_gnoise_noise_shape.py"
DEF = f"{CS}::test_it_is_its_definition_step_after_step"
UM8 = f"{UM}::TestUpdateMetrics::test_fsdp4_e1_freq4_no_dense_k8"
UM4 = f"{UM}::TestUpdateMetrics::test_ddp4_every_step_k4"

failfirst.MUTANTS = [
    # A: column scales
    (
        "A-dc-without-g",
        P,
        "gamma_c = (G * (g[..., None] * U)).sum(dim=-2)",
        "gamma_c = (G * U).sum(dim=-2)",
        [DEF],
    ),
    (
        "A-dU-without-c",
        P,
        "            G = G * c[..., None, :]\n",
        "            pass\n",
        [DEF],
    ),
    (
        "A-U-not-over-c",
        P,
        "            U = U / c[..., None, :]\n",
        "            pass\n",
        [DEF],
    ),
    (
        "A-W-without-c",
        P,
        "weight=(g_new[..., None] * U_new * c_new[..., None, :]).to(w.dtype),",
        "weight=(g_new[..., None] * U_new).to(w.dtype),",
        [DEF],
    ),
    (
        "A-c-rate-no-mag-lr",
        P,
        "c_new = c - (eta * self.magnitude_lr) * step",
        "c_new = c - eta * step",
        [DEF],
    ),
    (
        "A-c-starts-at-col-norms",
        P,
        "c = torch.where(Sc[..., 0, :] == 0, 1.0, Sc[..., 0, :])",
        "c = torch.where(Sc[..., 0, :] == 0, W.norm(dim=-2), Sc[..., 0, :])",
        [DEF],
    ),
    (
        "A-c-adam-no-bias-corr",
        P,
        "step = (mc / (1.0 - b1**t)) / ((vc / (1.0 - b2**t)).sqrt() + self.eps)",
        "step = mc / (vc.sqrt() + self.eps)",
        [DEF],
    ),
    (
        "A-col-state-dropped",
        P,
        "state=torch.cat([new, rows], dim=-2).to(state.dtype),",
        "state=torch.cat([new, rows * 0], dim=-2).to(state.dtype),",
        [f"{CS}::test_a_disco_step_is_the_definition_and_a_resume_continues_bitwise"],
    ),
    (
        "A-state-shape-no-rows",
        P,
        "            return (m + 3, n + 3)\n",
        "            return (m, n + 3)\n",
        [f"{CS}::test_a_disco_step_is_the_definition_and_a_resume_continues_bitwise"],
    ),
    (
        "A-off-path-moved",
        P,
        "        new = torch.cat([M, *(v[..., None] for v in cols)], dim=-1)\n",
        "        new = torch.cat([M, *(v[..., None] for v in cols)], dim=-1) * (1 + 2**-20)\n",
        [f"{CS}::test_off_is_the_class_before_the_change_bitwise"],
    ),
    (
        "A-refusal-not-gated",
        P,
        "        if self.col_scale and (\n",
        "        if (\n",
        [f"{CS}::test_unsupported_settings_are_refused_only_with_col_scale"],
    ),
    (
        "A-corner-ignores-it",
        P,
        "            col_scale=(bool(col_scale), False),\n",
        "",
        [f"{CS}::test_the_preset_passes_it_and_the_hyperball_corner_refuses_it"],
    ),
    # C: spectral cadence
    (
        "C-cadence-ignored",
        CO,
        "        return n <= 0 or is_final_step or step == 1 or step % n == 0\n",
        "        return True\n",
        [f"{SN}::test_the_spectral_part_runs_on_its_cadence_only"],
    ),
    (
        "C-step1-not-spectral",
        CO,
        "        return n <= 0 or is_final_step or step == 1 or step % n == 0\n",
        "        return n <= 0 or is_final_step or step % n == 0\n",
        [f"{SN}::test_the_spectral_part_runs_on_its_cadence_only"],
    ),
    (
        "C-norms-not-filtered",
        CO,
        "            return [x for x in self.norms_to_log if x in norm_helper._SVD_FREE_NORMS]\n",
        "            return self.norms_to_log\n",
        [f"{SN}::test_the_spectral_part_runs_on_its_cadence_only"],
    ),
    (
        "C-spectra-exported",
        CO,
        "track_spectrum=self.spectrum_logging_config.enable_export and spectral,",
        "track_spectrum=self.spectrum_logging_config.enable_export,",
        [f"{SN}::test_the_spectral_part_runs_on_its_cadence_only"],
    ),
    (
        "C-no-validation",
        CO,
        "        if n < 0 or (n and (every <= 0 or n % every)):\n",
        "        if False:\n",
        [f"{SN}::test_the_knob_must_be_a_multiple_of_log_norm_freq"],
    ),
    (
        "C-radial-not-filtered",
        RH,
        "    return [n for n in RADIAL_METRIC_NAMES if n not in _SPECTRAL_METRIC_NAMES]\n",
        "    return list(RADIAL_METRIC_NAMES)\n",
        [f"{SN}::test_keys_appear_exactly_on_their_steps_and_training_is_unchanged"],
    ),
    (
        "C-spectral-inputs-anyway",
        ME,
        "            if opt.spectral_metrics\n            else None\n",
        "            if True\n            else None\n",
        [f"{SN}::test_keys_appear_exactly_on_their_steps_and_training_is_unchanged"],
    ),
    (
        "C-experts-radial-unfiltered",
        ME,
        "                out.radial = [radial[n][e] for n in opt.radial_names]",
        "                out.radial = [radial[n][e] for n in list(radial)[: len(opt.radial_names)]]",
        [f"{MU}::TestMetricsUnion::test_fsdp4_local_spectral_off"],
    ),
    # B: gnoise update metrics
    (
        "B1-psi-over-G2",
        G,
        "            psi = sums[:, 0] / (sums[:, 1] * sums[:, 2]).sqrt()\n",
        "            psi = sums[:, 0] / sums[:, 2]\n",
        [UM4],
    ),
    (
        "B1-psi-units-shuffled",
        G,
        "                ru = disco._row_unit(e) + self.h_off[pp.i]\n",
        "                ru = disco._row_unit(e).flip(0) + self.h_off[pp.i]\n",
        [UM8],
    ),
    (
        "B1-psi-O-is-grad",
        G,
        "                O = e.rows2d(u[e.i]).double()\n",
        "                O = e.rows2d(_local(e.param.grad)).double()\n",
        [UM4],
    ),
    (
        "B3-lag-wrong-step",
        G,
        ".get(step - h) if lag else None",
        ".get(step - h + 1) if lag else None",
        [UM4],
    ),
    (
        "B3-store-wrong-steps",
        G,
        "        return any((step + h) % self.lag_every == 0 for h in _LAGS)\n",
        "        return any((step - h) % self.lag_every == 0 for h in _LAGS)\n",
        [UM8],
    ),
    (
        "B3-never-pruned",
        G,
        "                del d[s0]\n",
        "                pass\n",
        [UM4],
    ),
    (
        "B3-lag-every-interval",
        G,
        "        self.lag_every = self.interval * math.ceil(8 / self.interval)\n",
        "        self.lag_every = self.interval\n",
        [UM8],
    ),
    (
        "B1r-groups-contiguous",
        GR,
        "if q % groups == a], device=dev)",
        "if q * groups // k == a], device=dev)",
        [UM8],
    ),
    (
        "B1r-subset-norm-full",
        GR,
        "v.append(dots[gi].sum() / (oo * R[gi][:, gi].sum()).sqrt())",
        "v.append(dots[gi].sum() / (oo * R.sum()).sqrt())",
        [UM8],
    ),
    (
        "B2-ntop-unweighted",
        GR,
        "wD = nb.sqrt()[:, None] * ((n_den / nb)[:, None] * fv - Gh)",
        "wD = (n_den / nb)[:, None] * fv - Gh",
        [NS],
    ),
    (
        "B2-ntop-smallest",
        GR,
        "torch.linalg.eigvalsh(wD @ wD.T).flip(0).clamp_min(0)",
        "torch.linalg.eigvalsh(wD @ wD.T).clamp_min(0)",
        [NS],
    ),
    (
        "B4-bs1-no-sqrt",
        GR,
        "noise = torch.linalg.eigvalsh(C).clamp_min(0).sqrt().sum().square()",
        "noise = torch.linalg.eigvalsh(C).clamp_min(0).sum()",
        [NS],
    ),
    (
        "B4-bs1-grad-frobenius",
        GR,
        "grad = torch.linalg.svdvals(Gh.view(m, c)).sum().square()",
        "grad = Gh.square().sum()",
        [NS],
    ),
    (
        "B4-bs1-dof-K",
        GR,
        "D = (wD / math.sqrt(k - 1)).view(k, m, c)",
        "D = (wD / math.sqrt(k)).view(k, m, c)",
        [NS],
    ),
    (
        "B4-bs1-tall-reshape",
        GR,
        "M2 = D.permute(1, 0, 2).reshape(m, k * c)",
        "M2 = D.reshape(m, k * c)",
        [NS],
    ),
    (
        "B4-bs1-not-gated",
        GR,
        "        self.bs1 = K >= _BS1_MIN_K\n",
        "        self.bs1 = True\n",
        [UM4],
    ),
    (
        "B4-bs1-research-path",
        GR,
        "        vals += self._noise_shape(it, flat, n, n_den, valid)\n",
        "        vals += self._noise_shape(it, flat, n.flip(0), n_den, valid)\n",
        [UM8],
    ),
]

if __name__ == "__main__":
    sys.exit(failfirst.main())
