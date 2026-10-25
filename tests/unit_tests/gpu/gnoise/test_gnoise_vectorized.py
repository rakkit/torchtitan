# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vectorised E1 finalisation against the per-key one, and the E1 interval knob.

`data/perkey_ref.<scenario>.pt` holds every metric (value and owner) of two armed steps
(1 and 11, so the EMAs take dt = 10) recorded on CPU with the per-key implementation
(tree 5e289b632e89 a75e742d13276e7e, `.cache/gnoise/record/test_record_perkey.py`). The
same scenarios must give the same keys, owners and values to fp64 round-off."""

import math
import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gn_harness as H  # noqa: E402

from torchtitan.optimizers.gnoise import GradNoiseConfig  # noqa: E402

pytestmark = pytest.mark.multi_gpu

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RTOL = 1e-12
SCEN = {
    "fsdp4_dense_research_chunked": dict(
        topo=dict(dp_replicate=1, dp_shard=4),
        moe=False,
        accum=2,
        chunks=4,
        knobs=dict(enable=True, research_freq=1),
    ),
    "hsdp22_dense_e1": dict(
        topo=dict(dp_replicate=2, dp_shard=2),
        moe=False,
        accum=1,
        chunks=0,
        knobs=dict(enable=True),
    ),
    "fsdp4_moe_sparse_e1": dict(
        topo=dict(dp_replicate=1, dp_shard=4),
        moe=True,
        accum=1,
        chunks=0,
        ntok=4,
        knobs=dict(enable=True),
    ),  # 4 tokens a rank: some experts not estimable
    "fsdp4_moe_research": dict(
        topo=dict(dp_replicate=1, dp_shard=4),
        moe=True,
        accum=1,
        chunks=0,
        knobs=dict(
            enable=True,
            research_freq=1,
            research_matrices=r"^(?!.*routed_experts)|routed_experts\..*\.e[01]\b",
        ),
    ),
}


# families added after the per-key fixtures were recorded (their own tests: test_gnoise_update_metrics.py)
_LATER = ("gnoise_ntop", "gnoise_bs1")


class TestVectorized(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _scenario(self, name):
        s = SCEN[name]
        H.patch_dense_attention()
        dev = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                s["topo"],
                dev,
                tmp,
                gnoise=dict(s["knobs"]),
                chunks=s["chunks"],
                model_cfg=H.gauge_config(moe=s["moe"], num_experts=8)
                if s["moe"]
                else None,
            )
            gn = cont.gnoise
            gn.validate_layout(accum=s["accum"])
            got = {}
            for step in (1, 11):
                mbs = []
                for k in range(s["accum"]):
                    n = H.SEQ * (1 + self.rank % 3)
                    inp, lab = H.batch(
                        1000 * step + 100 * k + self.rank,
                        n,
                        dev,
                        mask_frac=0.25 if self.rank == 2 else 0.0,
                    )
                    if s["moe"]:
                        nt = s.get("ntok", H.SEQ // 2)
                        inp, lab = inp[:nt], lab[:nt]
                    mbs.append((inp, lab))
                nl = torch.tensor(
                    float(sum(int((lab != -100).sum()) for _, lab in mbs))
                )
                dist.all_reduce(nl)
                m = H.run_step(model, pd, cfg, cont, step, mbs, int(nl))
                owners = gn.metrics.owners()
                m = {k: v for k, v in m.items() if not k.startswith(_LATER)}
                got[step] = (
                    {k: float(v) for k, v in m.items()},
                    {k: owners[k] for k in m},
                )
        ref = torch.load(os.path.join(DATA, f"perkey_ref.{name}.pt"))  # noqa: TOR102
        errors, worst = [], 0.0
        for step in (1, 11):
            (gv, go), (rv, ro) = got[step], ref[step]
            if set(gv) != set(rv):
                errors.append(
                    f"step {step}: keys differ: missing {sorted(set(rv) - set(gv))[:5]}, "
                    f"extra {sorted(set(gv) - set(rv))[:5]}"
                )
                continue
            for k in rv:
                a, b = gv[k], rv[k]
                if go[k] != ro[k]:
                    errors.append(f"step {step} {k}: owner {go[k]} vs {ro[k]}")
                if math.isnan(a) or math.isnan(b):
                    if not (math.isnan(a) and math.isnan(b)):
                        errors.append(f"step {step} {k}: {a} vs {b}")
                    continue
                d = abs(a - b) / max(abs(a), abs(b), 1e-300)
                worst = max(worst, d if a != b else 0.0)
                if d > RTOL:
                    errors.append(f"step {step} {k}: {a!r} vs {b!r} (rel {d:.2e})")
        if self.rank == 0:
            print(
                f"VECTORIZED {name}: {len(ref[1][0])} keys x 2 steps, worst rel {worst:.2e}",
                flush=True,
            )
        assert not errors, f"{len(errors)} mismatches:\n" + "\n".join(errors[:25])

    @with_comms
    def test_equal_to_perkey_fsdp4_dense_research_chunked(self):
        self._scenario("fsdp4_dense_research_chunked")

    @with_comms
    def test_equal_to_perkey_hsdp22_dense_e1(self):
        self._scenario("hsdp22_dense_e1")

    @with_comms
    def test_equal_to_perkey_fsdp4_moe_sparse_e1(self):
        self._scenario("fsdp4_moe_sparse_e1")

    @with_comms
    def test_equal_to_perkey_fsdp4_moe_research(self):
        self._scenario("fsdp4_moe_research")

    @with_comms
    def test_e1_schedule(self):
        """E1 arms on step % e1_freq == 0, plus step 1 / the dense phase / the last step
        unless e1_dense_phase is off; the default arms on every norm-logging step."""
        H.patch_dense_attention()
        with tempfile.TemporaryDirectory() as tmp:
            pd, model, cont, cfg = H.build(
                dict(dp_replicate=1, dp_shard=4),
                torch.device("cpu"),
                tmp,
                gnoise=dict(enable=True),
            )
            gn = cont.gnoise
            cont.log_norm_freq, cont.dense_norm_steps = 2, 5

            def armed(e1_freq, dense_phase, last=16):
                gn.config.e1_freq, gn.config.e1_dense_phase = e1_freq, dense_phase
                gn.interval = e1_freq or cont.log_norm_freq
                out = []
                for step in range(1, last + 1):
                    gn.armed = False
                    cont.begin_step(step, should_log=True, is_final_step=step == last)
                    if gn.armed:
                        out.append(step)
                return out

            assert armed(0, True) == [
                1,
                2,
                3,
                4,
                5,
                6,
                8,
                10,
                12,
                14,
                16,
            ]  # every norm-logging step
            assert armed(4, True, last=14) == [1, 2, 3, 4, 5, 8, 12, 14]
            assert armed(4, False, last=14) == [4, 8, 12]
            assert armed(0, False) == [2, 4, 6, 8, 10, 12, 14, 16]


def test_e1_knob_rules():
    with pytest.raises(ValueError, match="knobs are set"):
        GradNoiseConfig(e1_freq=100).validate(log_norm_freq=10)
    with pytest.raises(ValueError, match="knobs are set"):
        GradNoiseConfig(e1_dense_phase=False).validate(log_norm_freq=10)
    with pytest.raises(ValueError, match="e1_freq"):
        GradNoiseConfig(enable=True, e1_freq=15).validate(log_norm_freq=10)
    with pytest.raises(ValueError, match="e1_freq"):
        GradNoiseConfig(enable=True, e1_freq=-10).validate(log_norm_freq=10)
    with pytest.raises(ValueError, match="E1 interval"):
        GradNoiseConfig(enable=True, e1_freq=100, research_freq=50).validate(
            log_norm_freq=10
        )
    with pytest.raises(ValueError, match="E1 interval"):
        GradNoiseConfig(
            enable=True, e1_freq=100, research_freq=100, curvature_model_freq=50
        ).validate(log_norm_freq=10)
    GradNoiseConfig(enable=True, e1_freq=100, research_freq=200).validate(
        log_norm_freq=10
    )
    GradNoiseConfig(
        enable=True, e1_freq=50, e1_dense_phase=False, research_freq=100
    ).validate(log_norm_freq=10)
    GradNoiseConfig(enable=True, research_freq=20).validate(log_norm_freq=10)
