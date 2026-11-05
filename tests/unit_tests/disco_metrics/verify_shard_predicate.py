# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Truth-table the shard predicate without needing a real process group."""
from torchtitan.distributed.utils import metrics_shard_mesh, rank_owns_metrics_shard


class M:
    def __init__(self, r, n):
        self.r, self.n = r, n

    def get_local_rank(self):
        return self.r

    def size(self):
        return self.n

    def _flatten(self, name):
        # `fsdp_shard_mesh` flattens ["dp_shard", "cp"] into one axis; without
        # this the CP path could not be exercised at all.
        return self


class PD:
    def __init__(
        self,
        fsdp,
        rep,
        tp,
        fsdp_r=0,
        rep_r=0,
        tp_r=0,
        fsdp_n=1,
        rep_n=1,
        tp_n=1,
        cp=False,
    ):
        self.fsdp_enabled, self.dp_replicate_enabled, self.tp_enabled = fsdp, rep, tp
        # FSDP shards over "dp_shard" (+ "cp" when cp > 1, flattened).
        self.cp_enabled = cp
        self._m = {
            "dp_shard": M(fsdp_r, fsdp_n),
            "dp_replicate": M(rep_r, rep_n),
            "tp": M(tp_r, tp_n),
        }
        # Under CP the shard mesh is dp_shard x cp FLATTENED, so it
        # is strictly larger than dp_shard alone. Modelling them the same size
        # made this test blind to the cp axis being dropped.
        self._m["dp_shard"] = (
            M(fsdp_r, max(1, fsdp_n // 2)) if cp else M(fsdp_r, fsdp_n)
        )
        self._m["dp_shard, cp"] = M(fsdp_r, fsdp_n)

    def get_optional_mesh(self, n, **kw):
        if isinstance(n, list):
            # Multi-axis lookup: the entry for the exact axis list requested.
            key = ", ".join(n)
            if key in self._m:
                return self._m[key]
            # No silent fallback: an unexpected axis list must fail loudly, or a
            # dropped axis looks identical to the correct lookup.
            raise KeyError(f"stub has no mesh for axes {n!r}")
        return self._m[n]


def owners(pd_factory, ranks):
    return [r for r in ranks if rank_owns_metrics_shard(pd_factory(r))]


print("=== HSDP: dp_shard=4 x dp_replicate=2 (production shape) ===")
got = []
for rep in range(2):
    for f in range(4):
        pd = PD(True, True, False, fsdp_r=f, rep_r=rep, fsdp_n=4, rep_n=2)
        if rank_owns_metrics_shard(pd):
            got.append((rep, f))
print("  loggers at (dp_replicate, fsdp):", got)
assert got == [(0, 0), (0, 1), (0, 2), (0, 3)], got
print("  -> 4 loggers, one per fsdp rank, replica 0 only. Covers all params once. OK")

print("\n=== pure DDP: dp_replicate=4, no fsdp (the bug) ===")
got = [
    r
    for r in range(4)
    if rank_owns_metrics_shard(PD(False, True, False, rep_r=r, rep_n=4))
]
print("  loggers at dp_replicate:", got)
assert got == [0, 1, 2, 3], got
print("  -> all 4 log. Before the fix this was [0] and 3/4 of metrics vanished. OK")

print("\n=== TP is excluded in both ===")
assert not rank_owns_metrics_shard(
    PD(True, True, True, fsdp_r=1, tp_r=1, fsdp_n=4, tp_n=2)
)
assert rank_owns_metrics_shard(PD(True, True, True, fsdp_r=1, tp_r=0, fsdp_n=4, tp_n=2))
assert not rank_owns_metrics_shard(
    PD(False, True, True, rep_r=2, tp_r=1, rep_n=4, tp_n=2)
)
print("  tp rank != 0 never logs; tp rank 0 does. OK")

print("\n=== single rank / no DP ===")
assert rank_owns_metrics_shard(PD(False, False, False))
assert metrics_shard_mesh(PD(False, False, False)) is None
print("  logs, shard mesh None. OK")

print("\n=== shard mesh selection ===")
assert (
    metrics_shard_mesh(PD(True, True, False, fsdp_n=4, rep_n=2)).size() == 4
)  # fsdp wins
assert metrics_shard_mesh(PD(False, True, False, rep_n=4)).size() == 4  # dp_replicate
print("  fsdp when fsdp_enabled, else dp_replicate. OK")
print("\n=== CP: the shard mesh is dp_shard x cp flattened ===")
got = [
    f
    for f in range(4)
    if rank_owns_metrics_shard(PD(True, False, False, fsdp_r=f, fsdp_n=4, cp=True))
]
assert got == [0, 1, 2, 3], got
assert metrics_shard_mesh(PD(True, False, False, fsdp_n=4, cp=True)).size() == 4
print("  cp=2: all 4 fsdp ranks log, via the flattened dp_shard+cp axis. OK")

print("\nALL PREDICATE CHECKS PASSED")
