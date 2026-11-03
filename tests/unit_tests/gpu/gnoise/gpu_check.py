# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read gnoise GPU test runs from their W&B offline datastores (login node, llm_env).

  bitwise A B               rank-0 train loss and grad_norm bitwise at every common step (T1, T20)
  known D [tol]             identical small batches: |tr| <= tol * n * |S_unit| for every component
                            at every logged step; gauges < 2e-2; N_check == 0 (known answer, T7)
  keys D [G]                one owner per key: union of gnoise keys over ranks, no duplicates; with
                            G (a gather-mode run) the union equals G's rank-0 keys, values equal (T13)
  cost A B [lo hi]          wall time A (off) vs B (on) between steps lo and hi from _timestamp; the
                            mean 10-step interval with and without a research step; peak memory (T19)
  show D key-prefix [step]  print matching keys of rank 0
  compare A B prefix rel    every key with `prefix` (all ranks) at common steps within relative `rel`
                            of max(|a|, |b|, the unit's |S|) (T10 bitwise with rel 0, T11, T12)
  gauges D [tol]            every gauge check (q/k head radial, FFN channel, residual) < tol (2e-2)
  t9 A B                    per type, the mean over logged steps of gnoise_type_tr: |A - B| <= 2 SE (T9)
A, B, D, G: dump folders (storage/gnoise-tests/<job>-<arm>).
"""

import glob
import json
import math
import re
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

LOSS, GN = "loss_metrics/global_avg_loss", "grad_norm"


def stores(dump: str) -> dict[int, list[str]]:
    out = {}
    for p in sorted(glob.glob(f"{dump}/tb/*/rank_*/wandb/offline-run-*/*.wandb")):
        r = int(re.search(r"/rank_(\d+)/", p).group(1))
        out.setdefault(r, []).append(p)
    return out


def history(paths: list[str], want=None) -> dict[int, dict[str, float]]:
    """{step: {key: value}} of one rank (all its datastores)."""
    out: dict[int, dict[str, float]] = {}
    for p in paths:
        ds = datastore.DataStore()
        ds.open_for_scan(p)
        while True:
            try:
                x = ds.scan_data()
            except AssertionError:
                break
            if x is None:
                break
            rec = pb.Record()
            rec.ParseFromString(x)
            if rec.WhichOneof("record_type") != "history":
                continue
            it = {
                ("/".join(i.nested_key) or i.key): i.value_json
                for i in rec.history.item
            }
            if "_step" not in it:
                continue
            st = int(json.loads(it["_step"]))
            row = out.setdefault(st, {})
            for k, v in it.items():
                if want is None or want(k):
                    try:
                        row[k] = float(json.loads(v))
                    except (TypeError, ValueError):
                        pass
    return out


def bitwise(a: str, b: str) -> int:
    ha = history(stores(a)[0], lambda k: k in (LOSS, GN))
    hb = history(stores(b)[0], lambda k: k in (LOSS, GN))
    steps = sorted(s for s in set(ha) & set(hb) if LOSS in ha[s] and LOSS in hb[s])
    bad = [s for s in steps if ha[s][LOSS] != hb[s][LOSS]]
    gbad = [s for s in steps if ha[s].get(GN) != hb[s].get(GN)]
    print(
        f"{len(steps)} common steps ({steps[0] if steps else '-'}..{steps[-1] if steps else '-'}); "
        f"loss differs at {len(bad)}, grad_norm at {len(gbad)}"
    )
    if bad:
        s = bad[0]
        print(f"FAIL first differing step {s}: {ha[s][LOSS]!r} vs {hb[s][LOSS]!r}")
        return 1
    if not steps:
        print("FAIL no common steps")
        return 1
    print("PASS-BITWISE")
    return 0


def known(d: str, tol: float = 1e-3) -> int:
    all_k: dict[int, dict[str, float]] = {}
    for r, paths in stores(d).items():
        for st, row in history(paths, lambda k: k.startswith("gnoise")).items():
            all_k.setdefault(st, {}).update(row)
    bad, n_checked, worst = [], 0, 0.0
    for st, m in sorted(all_k.items()):
        K = m.get("gnoise/K")
        if K is None:
            continue
        for k, v in m.items():
            if not re.match(r"gnoise_(tr|rad_tr|rrad_tr|hd_tr|rs_tr|srad_tr)/", k):
                continue
            unit = k.split("/", 1)[1]
            base = re.sub(r"\.h\d+$", "", unit)
            sig = [m[x] for x in (f"gnoise_sig/{unit}",) if x in m] or [
                m[x] for x in m if x.startswith(f"gnoise_sig/{base}")
            ]
            if not sig or not math.isfinite(v):
                continue
            s_unit = max(abs(x) for x in sig)
            n_b = (
                4096 * 32
            )  # tokens of a small batch (upper bound over the layouts used)
            r = abs(v) / max(s_unit * n_b, 1e-300)
            worst = max(worst, r)
            n_checked += 1
            if r > tol:
                bad.append(
                    f"step {st} {k} = {v:.3e} (unit S {s_unit:.3e}, ratio {r:.2e})"
                )
        for k, v in m.items():
            if k.startswith("gnoise_gauge") and not (v < 2e-2):
                bad.append(f"step {st} {k} = {v}")
            ma = re.match(r"gnoise_agree_s(\d+)/(.*)$", k)
            if ma and math.isfinite(v):  # identical small batches: every LMO equal
                s_, unit = ma.groups()
                want = m.get(
                    f"gnoise_polar_rank/{unit}",
                    m.get(f"gnoise_agree_present_s{s_}/{unit}", 1.0),
                )
                n_agree = globals().setdefault("_n_agree", [0])
                n_agree[0] += 1
                if abs(v - want) > 2e-3:
                    bad.append(f"step {st} {k} = {v:.6f}, want {want:.6f}")
        if m.get("gnoise/N_check", 0.0) != 0.0:
            bad.append(f"step {st} N_check {m['gnoise/N_check']}")
    print(
        f"checked {n_checked} components over {len(all_k)} steps; worst |tr| / (n S) = {worst:.2e}; "
        f"{globals().get('_n_agree', [0])[0]} agreement values"
    )
    for line in bad[:20]:
        print("  ", line)
    print("PASS" if not bad and n_checked else "FAIL")
    return 1 if bad or not n_checked else 0


def keys(d: str, gather: str | None = None) -> int:
    per_rank = {}
    for r, paths in stores(d).items():
        h = history(paths, lambda k: k.startswith("gnoise"))
        per_rank[r] = {(st, k): v for st, row in h.items() for k, v in row.items()}
    seen, dup = {}, []
    for r, kv in per_rank.items():
        for key in kv:
            if key in seen:
                dup.append((key, seen[key], r))
            seen[key] = r
    print(
        f"ranks {sorted(per_rank)}; {len(seen)} (step, key) pairs; duplicates {len(dup)}"
    )
    rc = 1 if dup else 0
    if gather:
        g = history(stores(gather)[0], lambda k: k.startswith("gnoise"))
        gk = {(st, k): v for st, row in g.items() for k, v in row.items()}
        steps = {st for st, _ in gk} & {st for st, _ in seen}
        a = {x for x in seen if x[0] in steps}
        b = {x for x in gk if x[0] in steps}
        print(
            f"local union {len(a)} vs gather {len(b)} over {len(steps)} common steps; "
            f"missing {len(b - a)}, extra {len(a - b)}"
        )
        rc |= int(a != b)
    print("PASS" if rc == 0 else "FAIL")
    return rc


def cost(a: str, b: str, lo: int = 21, hi: int = 2000) -> int:
    out = {}
    for name, d in (("A", a), ("B", b)):
        h = history(
            stores(d)[0],
            lambda k: k
            in ("_timestamp", "memory/max_reserved(GiB)", "memory/max_active(GiB)"),
        )
        ts = {st: row["_timestamp"] for st, row in h.items() if "_timestamp" in row}
        steps = sorted(s for s in ts if lo <= s <= hi)
        mem = max(
            (row.get("memory/max_reserved(GiB)", 0) for row in h.values()), default=0
        )
        act = max(
            (row.get("memory/max_active(GiB)", 0) for row in h.values()), default=0
        )
        iv = {
            s: ts[s] - ts[s - 10]
            for s in steps
            if s - 10 in ts and s > 500 and s % 10 == 0
        }
        out[name] = (ts, steps, mem, act, iv)
        print(
            f"{name} {d}: steps {steps[0]}..{steps[-1]}, wall {ts[steps[-1]] - ts[steps[0]]:.1f} s, "
            f"peak reserved {mem:.2f} GiB, active {act:.2f} GiB"
        )
    (ta, sa, *_), (tb, sb, *_) = out["A"], out["B"]
    s0, s1 = max(sa[0], sb[0]), min(sa[-1], sb[-1])
    wa, wb = ta[s1] - ta[s0], tb[s1] - tb[s0]
    print(
        f"steps {s0}..{s1}: A {wa:.1f} s, B {wb:.1f} s, overhead {(wb - wa) / wa * 100:+.2f}%"
    )
    for name, (ts, steps, _, _, iv) in out.items():
        if iv:
            vals = sorted(iv.values())
            print(
                f"  {name}: 10-step intervals past step 500: n {len(vals)}, median {vals[len(vals) // 2]:.3f} s"
            )
    return 0


def show(d: str, prefix: str, step: str | None = None) -> int:
    h = history(stores(d)[0], lambda k: k.startswith(prefix))
    for st in sorted(h):
        if step is not None and st != int(step):
            continue
        for k in sorted(h[st]):
            print(st, k, h[st][k])
    return 0


def union(d: str, prefix: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for r, paths in stores(d).items():
        for st, row in history(paths, lambda k: k.startswith(prefix)).items():
            out.setdefault(st, {}).update(row)
    return out


def compare(a: str, b: str, prefix: str, rel: str) -> int:
    rel = float(rel)
    ua, ub = union(a, prefix), union(b, prefix)
    steps = sorted(set(ua) & set(ub))
    n, bad, worst, errs = 0, [], 0.0, []
    for st in steps:
        for k in set(ua[st]) | set(ub[st]):
            x, y = ua[st].get(k), ub[st].get(k)
            if x is None or y is None:
                bad.append(f"step {st} {k}: only in {'A' if y is None else 'B'}")
                continue
            if math.isnan(x) and math.isnan(y):
                continue
            unit = k.split("/", 1)[1] if "/" in k else k
            comp = k.split("/", 1)[0]
            ref_key = "gnoise_tr" if comp.endswith("_tr") else "gnoise_sig"
            sc = max(abs(x), abs(y), abs(ua[st].get(f"{ref_key}/{unit}", 0.0)), 1e-300)
            if comp.startswith("gnoise_gauge"):
                sc = 1.0  # gauge residuals are relative already; compare absolutely
            e = abs(x - y) / sc
            errs.append(e)
            worst = max(worst, e)
            n += 1
            if e > rel or (rel == 0 and x != y):
                bad.append(f"step {st} {k}: {x!r} vs {y!r}")
    print(
        f"{n} values over {len(steps)} common steps; worst relative difference {worst:.2e}; "
        + ", ".join(
            f">{t:g}: {sum(e > t for e in errs)}" for t in (1e-7, 1e-6, 1e-5, 1e-4)
        )
    )
    for line in bad[:15]:
        print("  ", line)
    ok = n > 0 and not bad
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def t9(a: str, b: str) -> int:
    ua, ub = union(a, "gnoise_type_tr/"), union(b, "gnoise_type_tr/")
    rc = 0
    for k in sorted({k for row in ua.values() for k in row}):  # noqa: B020
        va = [row[k] for row in ua.values() if k in row and math.isfinite(row[k])]
        vb = [row[k] for row in ub.values() if k in row and math.isfinite(row[k])]
        if len(va) < 10 or len(vb) < 10:
            continue
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        sa = (sum((x - ma) ** 2 for x in va) / (len(va) - 1) / len(va)) ** 0.5
        sb = (sum((x - mb) ** 2 for x in vb) / (len(vb) - 1) / len(vb)) ** 0.5
        se = (sa * sa + sb * sb) ** 0.5
        z = (ma - mb) / se if se > 0 else float("inf")
        flag = "ok" if abs(z) <= 2 else "OUTSIDE 2 SE"
        rc |= abs(z) > 2
        print(f"{k}: A {ma:.4e} B {mb:.4e} (n {len(va)}/{len(vb)}), z {z:+.2f} {flag}")
    print(
        "PASS"
        if rc == 0
        else "REPORT (outside 2 SE: within-rank correlation, not fixed)"
    )
    return int(rc)


def gauges(d: str, tol: str = "2e-2") -> int:
    u = union(d, "gnoise_gauge")
    vals = [(st, k, v) for st, row in u.items() for k, v in row.items()]
    worst = max(vals, key=lambda x: x[2]) if vals else None
    bad = [x for x in vals if not x[2] < float(tol)]
    print(f"{len(vals)} gauge values over {len(u)} steps; worst {worst}")
    for x in bad[:10]:
        print("  ", x)
    print("PASS" if vals and not bad else "FAIL")
    return 0 if vals and not bad else 1


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    fn = {
        "bitwise": bitwise,
        "known": known,
        "keys": keys,
        "cost": cost,
        "show": show,
        "compare": compare,
        "t9": t9,
        "gauges": gauges,
    }[cmd]
    conv = [
        float(a)
        if cmd == "known" and i == 1
        else (int(a) if cmd == "cost" and i >= 2 else a)
        for i, a in enumerate(args)
    ]
    sys.exit(fn(*conv))
