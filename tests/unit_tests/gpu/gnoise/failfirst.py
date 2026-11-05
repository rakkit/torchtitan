# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Show each gnoise test failing on a deliberately wrong version first.

Each mutant is one exact edit of a source file (the old text must occur once); the
file is restored and its md5 re-checked after every run. Run from the tree root:
    python tests/unit_tests/gpu/gnoise/failfirst.py [mutant ids] > result.txt
"""

import hashlib
import os
import subprocess
import sys
import time

G = "torchtitan/optimizers/gnoise.py"
GM = "torchtitan/optimizers/gnoise_math.py"
GR = "torchtitan/optimizers/gnoise_research.py"
TR = "torchtitan/trainer.py"
CO = "torchtitan/optimizers/container.py"
R = "tests/unit_tests/gpu/gnoise/test_gnoise_reference.py"
RS = "tests/unit_tests/gpu/gnoise/test_gnoise_research.py"
S = "tests/unit_tests/gpu/gnoise/test_gnoise_setup.py"
M = "tests/unit_tests/cpu/test_gnoise_math.py"
V = "tests/unit_tests/gpu/gnoise/test_gnoise_vectorized.py"
C = "tests/unit_tests/gpu/gnoise/test_gnoise_curvature.py"
HA = "tests/unit_tests/gpu/gnoise/gn_harness.py"  # T5: stands in for the trainer's valid-token count

MUTANTS = [
    (
        "T2-H-from-mean-n",
        GM,
        "    H = inv.sum(-1) / k.clamp_min(1.0)\n",
        "    H = 1.0 / (n.sum(-1) / k.clamp_min(1.0))\n",
        [f"{M}::test_unbiased_uneven_and_equal_to_reference"],
    ),
    (
        "T2-no-1/N",
        GM,
        "    den = H - 1.0 / N.clamp_min(1.0)\n",
        "    den = H\n",
        [f"{M}::test_unbiased_uneven_and_equal_to_reference"],
    ),
    (
        "T4-no-N/n_b-rescale",
        GM,
        "    w = torch.where(valid, (n_den * inv) ** 2, 0.0)",
        "    w = torch.where(valid, (n_den / n.sum(-1, keepdim=True) * k[..., None]) ** 2, 0.0)",
        [f"{R}::TestKnownAnswers::test_duplicate_uneven_fsdp4"],
    ),
    (
        "T3-radial-not-from-W",
        G,
        "        gw = gc * wc\n",
        "        gw = gc * gc.roll(1, 0)\n",
        [f"{R}::TestReference::test_fsdp4_bf16_uneven_masked_accum2_chunked"],
    ),
    (
        "T7-ffn-channel-pairing",
        G,
        'dot[pp.w3_rows] if pp.ffn[1] == "w3"',
        'dot[pp.w3_rows].flip(0) if pp.ffn[1] == "w3"',
        [f"{R}::TestReference::test_fsdp4_fp32_gauges"],
    ),
    (
        "T7-residual-sign",
        G,
        "s[self.slot_res] += torch.stack([dot.sum(), absum])",
        "s[self.slot_res] += torch.stack([dot.abs().sum(), absum])",
        [f"{R}::TestReference::test_fsdp4_fp32_gauges"],
    ),
    (
        "T7-wo-heads-interleaved",
        G,
        "            hb += gc.view(gc.shape[0], col_heads, -1).square().sum((0, 2))\n",
        "            hb += gc.view(gc.shape[0], -1, col_heads).transpose(1, 2).square().sum((0, 2))\n",
        [f"{R}::TestReference::test_fsdp4_bf16_uneven_masked_accum2_chunked"],
    ),
    (
        "T8-local-shard-as-full",
        G,
        "        if not isinstance(like, DTensor):\n",
        "        if True:\n",
        [f"{R}::TestReference::test_hsdp22_bf16_uneven_masked_accum2_chunked"],
    ),
    (
        "T9-microbatches-merged",
        G,
        "        self.mb += 1\n",
        "        self.mb = 0\n",
        [f"{R}::TestReference::test_fsdp4_bf16_uneven_masked_accum2_chunked"],
    ),
    (
        "T22-every-entry",
        G,
        "            if fp.unsharded_param.grad is not None:\n",
        "            if True:\n",
        [f"{S}::TestSetup::test_exactly_once_chunked"],
    ),
    (
        "T11-T22-first-chunk",
        G,
        "                return  # the chunked loss's chunks 0..N-2: read at the reducing call\n",
        "                pass\n",
        [f"{S}::TestSetup::test_exactly_once_chunked"],
    ),
    (
        "T23-no-entry-guard",
        G,
        "        if not g.is_unsharded:\n",
        "        if False:\n",
        [f"{S}::TestSetup::test_entry_guard"],
    ),
    (
        "T24-F-with-fp32-master",
        G,
        "            if pp.fp.param_dtype is not None:  # FSDP2's cast: the forward's values exactly\n",
        "            if False:\n",
        [f"{R}::TestReference::test_fsdp4_bf16_uneven_masked_accum2_chunked"],
    ),
    (
        "T14-plain-sign",
        G,
        "            agree, used = gm.loo_sign_terms(self.d_rows[k].double(), Dr)\n",
        "            agree, used = gm.loo_sign_terms(self.d_rows[k].double(), Dr + self.d_rows[k].double())\n",
        [f"{R}::TestReference::test_fsdp4_bf16_uneven_masked_accum2_chunked"],
    ),
    (
        "T15-topk-from-G",
        GR,
        "        Wt = self.w_own[it.j].float()\n",
        "        Wt = g.sum(0)\n",
        [f"{RS}::TestResearch::test_research_fsdp4"],
    ),
    (
        "T15-self-pairs",
        GM,
        "    iu = torch.triu_indices(g, g, 1)\n",
        "    iu = torch.triu_indices(g, g, 0)\n",
        [f"{M}::test_agreement_known_answers_and_isotropic_topk"],
    ),
    (
        "T16-TP",
        G,
        "    if parallelism.tensor_parallel_degree > 1:\n",
        "    if False:\n",
        [f"{S}::test_refusals_only_when_enabled"],
    ),
    (
        "T16-CP",
        G,
        "    if parallelism.context_parallel_degree > 1:\n",
        "    if False:\n",
        [f"{S}::test_refusals_only_when_enabled"],
    ),
    (
        "T16-PP",
        G,
        "    if parallelism.pipeline_parallel_degree > 1:\n",
        "    if False:\n",
        [f"{S}::test_refusals_only_when_enabled"],
    ),
    (
        "T16-accelerator-graphs",
        G,
        "    if not training.disable_accelerator_graphs:\n",
        "    if False:\n",
        [f"{S}::test_refusals_only_when_enabled"],
    ),
    (
        "T16-cpu-offload",
        G,
        "    if parallelism.fsdp_enable_cpu_offloading:\n",
        "    if False:\n",
        [f"{S}::test_refusals_only_when_enabled"],
    ),
    (
        "T16-compiled-autograd",
        G,
        "    if torch._dynamo.config.compiled_autograd:\n",
        "    if False:\n",
        [f"{S}::test_refuses_compiled_autograd"],
    ),
    (
        "T21-refusals-not-gated",
        TR,
        "            if gnoise is not None and gnoise.enable:\n",
        "            if gnoise is not None:\n",
        [f"{S}::test_refusals_only_when_enabled"],
    ),
    (
        "T17-ema-in-state-dict",
        G,
        "        out = gm.ema_update(self.ema, ema_in, dt, float(self.config.ema_steps))\n",
        "        out = gm.ema_update(self.ema, ema_in, dt, float(self.config.ema_steps))\n"
        "        self.container.optimizers[0].state[self.params[0].p]['gnoise_ema'] = next(iter(self.ema.values()))\n",
        [f"{S}::TestSetup::test_no_checkpoint_state"],
    ),
    (
        "T20-built-when-off",
        CO,
        "            if config.gnoise.enable\n            else None\n",
        "            if True\n            else None\n",
        [f"{S}::TestSetup::test_off_installs_nothing"],
    ),
    (
        "T25-microbatch-tokens",
        G,
        "        n_e = counts[rows, off + e_idx]  # [E, Ke]\n",
        "        n_e = n_b[rows]  # [E, Ke]\n",
        [f"{R}::TestReference::test_moe_fsdp4_ep1"],
    ),
    (
        "T25-N-not-N_e",
        GM,
        "    Fh = (n_den / N.clamp_min(1.0)) ** 2 * F\n",
        "    Fh = F\n",
        [f"{R}::TestReference::test_moe_fsdp4_ep1"],
    ),
    (
        "T25-zero-count-kept",
        GM,
        "    valid = n > 0\n    k = valid.sum(-1).double()\n",
        "    valid = n >= 0\n    k = valid.sum(-1).double()\n",
        [f"{R}::TestReference::test_moe_fsdp4_ep1"],
    ),
    (
        "T26-router-with-a-balance-loss",
        G,
        "if mod.router.aux_loss is not None",
        "if False",
        [f"{S}::TestSetup::test_router_left_out_with_a_balance_loss"],
    ),
    (
        "T13-every-rank-stores-all",
        G,
        "        sk = [k for k, (_, o) in m.scalars.items() if o % mworld == mrank]\n",
        "        sk = list(m.scalars)\n",
        [f"{S}::TestSetup::test_one_owner_per_key"],
    ),
    (
        "T5-count-masked-tokens",
        HA,
        "            gn.begin_microbatch(int((lab != -100).sum()))\n",
        "            gn.begin_microbatch(lab.numel())\n",
        [f"{R}::TestKnownAnswers::test_masked_extra_document_hsdp22"],
    ),
    (
        "T27-wrong-expert-rows",
        GR,
        "                    rows = (pp.local_row_unit == loc).nonzero().flatten()\n",
        "                    rows = (pp.local_row_unit == (loc + pp.per) % (pp.local_experts * pp.per)).nonzero().flatten()\n",
        [f"{RS}::TestResearchExperts::test_research_experts_ep1"],
    ),
    (
        "Tc-no-G0-subtracted",
        G,
        "                g1 = _local(pp.p.grad).double() - G0.double()\n",
        "                g1 = _local(pp.p.grad).double()\n",
        [f"{C}::TestCurvature::test_curvature_fsdp4"],
    ),
    (
        "Tc-model-sign",
        G,
        '            res.put("gnoise_curv/model", 2.0 * (packed[3] - packed[2] - gd) / dd, 0)\n',
        '            res.put("gnoise_curv/model", 2.0 * (packed[3] - packed[2] + gd) / dd, 0)\n',
        [f"{C}::TestCurvature::test_curvature_fsdp4"],
    ),
    (
        "Tc-no-rep-divide",
        G,
        "            gd, dd = packed[0] / rep, packed[1] / rep\n",
        "            gd, dd = packed[0], packed[1]\n",
        [f"{C}::TestCurvature::test_curvature_ddp4"],
    ),
    (
        "Tc-buffers-not-restored",
        G,
        "        for b, v in buffers:\n            b.copy_(v)\n",
        "        for b, v in buffers:\n            pass\n",
        [f"{C}::TestCurvature::test_curvature_moe_fsdp4"],
    ),
    (
        "Tc-grads-left",
        G,
        "            model.zero_grad(set_to_none=True)\n        for b, v in buffers:\n",
        "        for b, v in buffers:\n",
        [f"{C}::TestCurvature::test_curvature_fsdp4"],
    ),
    (
        "Tc-no-reshard",
        G,
        "                    m.reshard()  # a no-grad forward leaves [norm, lm_head] gathered\n",
        "                    pass\n",
        [
            f"{C}::TestCurvature::test_curvature_fsdp4",
            f"{C}::TestCurvature::test_curvature_hsdp22_bf16_chunked",
        ],
    ),
    (
        "T25-ep-counts-own",
        G,
        '        return torch.einsum("pq,qak->pak", self.ep_group.to(d), d)\n',
        "        return d\n",
        [f"{R}::TestReference::test_moe_fsdp4_ep2_balanced"],
    ),
    (
        "T25-ep-other-copies",
        G,
        "            pos = [q for q in range(self.world) if self.ep_of[q] == c]\n",
        "            pos = [q for q in range(self.world) if self.ep_of[q] != c]\n",
        [f"{R}::TestReference::test_moe_fsdp4_ep2_balanced"],
    ),
    (
        "T25-ep-groups-by-edp_shard",
        G,
        '                tuple(v for k, v in coords[r].items() if k != "ep")\n',
        '                tuple(v for k, v in coords[r].items() if k != "edp_shard")\n',
        [f"{R}::TestReference::test_moe_fsdp4_ep2_balanced"],
    ),
    (
        "T27-ep-wrong-holders",
        GR,
        "                        holders = [q for q in range(W) if gn.ep_of[q] == c]\n",
        "                        holders = [q for q in range(W) if gn.ep_of[q] != c]\n",
        [f"{RS}::TestResearchExperts::test_research_experts_ep2"],
    ),
    (
        "T15-sizes-ignored",
        GR,
        "        self.sizes = sorted(cfg.research_sizes) if cfg.research_sizes else _divisors(K)\n",
        "        self.sizes = _divisors(K)\n",
        [f"{RS}::TestResearch::test_research_sizes_subset_fsdp4"],
    ),
    (
        "V-quantile-no-interpolation",
        GM,
        "    hi = xs[start[None, :] + ranks.ceil().long()]\n",
        "    hi = lo\n",
        [f"{V}::TestVectorized::test_equal_to_perkey_hsdp22_dense_e1"],
    ),
    (
        "V-loo-counts-every-row",
        G,
        "            cnt = z.index_add(0, self.row_dparam, used.double())\n",
        "            cnt = z.index_add(0, self.row_dparam, torch.ones_like(Dr))\n",
        [f"{V}::TestVectorized::test_equal_to_perkey_hsdp22_dense_e1"],
    ),
    (
        "V-unres-over-all-rows",
        G,
        '                res.put_vec(uk[f"{comp}_unres"], 1.0 - z.index_add(0, ru, ok.double()) / self.unit_nrows, uo)\n',
        '                res.put_vec(uk[f"{comp}_unres"], 1.0 - z.index_add(0, ru, ok.double()) / self.unit_nrows.sum(), uo)\n',
        [f"{V}::TestVectorized::test_equal_to_perkey_hsdp22_dense_e1"],
    ),
    (
        "V-heads-col-sum-wrong-axis",
        G,
        "                Fh[pp.head_off : pp.head_off + nh] = feats[pp.i][:, 3 : 3 + nh].sum(0)\n",
        "                Fh[pp.head_off : pp.head_off + nh] = feats[pp.i][:, 3 + nh : 3 + 2 * nh].sum(0)\n",
        [f"{V}::TestVectorized::test_equal_to_perkey_hsdp22_dense_e1"],
    ),
    (
        "V-expert-type-nansum-dropped",
        G,
        '            e_tr[t] += tot["tr"].nansum()\n',
        '            e_tr[t] += tot["tr"].sum()\n',
        [f"{V}::TestVectorized::test_equal_to_perkey_fsdp4_moe_sparse_e1"],
    ),
    (
        "E1-due-ignores-dense-flag",
        G,
        "        return step % self.interval == 0 or (self.config.e1_first_and_last_steps and dense)\n",
        "        return step % self.interval == 0 or dense\n",
        [f"{V}::TestVectorized::test_e1_schedule"],
    ),
    (
        "E1-research-vs-log-norm-freq",
        G,
        "        every = self.e1_freq or log_norm_freq\n",
        "        every = log_norm_freq\n",
        [f"{V}::test_e1_knob_rules"],
    ),
]


def md5(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main() -> int:
    want = set(sys.argv[1:])
    env = dict(
        os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.getcwd()
    )
    survivors = 0
    for mid, path, old, new, tests in MUTANTS:
        if want and mid not in want:
            continue
        src = open(path).read()
        before = md5(path)
        assert (
            src.count(old) == 1
        ), f"{mid}: the old text occurs {src.count(old)} times in {path}"
        t0 = time.time()
        try:
            open(path, "w").write(src.replace(old, new))
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-x",
                    "-p",
                    "no:randomly",
                    "-p",
                    "no:cacheprovider",
                    *tests,
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=3000,
            )
        finally:
            open(path, "w").write(src)
        assert md5(path) == before, f"{mid}: {path} not restored"
        tail = [
            ln
            for ln in r.stdout.splitlines()
            if " passed" in ln or " failed" in ln or "error" in ln.lower()
        ][-1:]
        verdict = "FAILED as required" if r.returncode != 0 else "SURVIVED"
        survivors += r.returncode == 0
        print(
            f"{mid:28s} {verdict:18s} rc={r.returncode} {time.time() - t0:5.0f}s  {tail[0] if tail else ''}  [{' '.join(tests)}]",
            flush=True,
        )
    print(f"survivors: {survivors}")
    return survivors


if __name__ == "__main__":
    sys.exit(main())
