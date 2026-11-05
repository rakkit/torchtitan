# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read an offline W&B run and report the DiSCO metric keys it actually logged."""
import collections, glob, json, sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

pat = sys.argv[1]
f = sorted(glob.glob(pat))[0]
ds = datastore.DataStore()
ds.open_for_scan(f)
best = {}
while True:
    data = ds.scan_data()
    if data is None:
        break
    rec = pb.Record()
    rec.ParseFromString(data)
    t = rec.WhichOneof("record_type")
    items = (
        rec.history.item
        if t == "history"
        else (rec.summary.update if t == "summary" else None)
    )
    if items:
        d = {(i.key or "/".join(i.nested_key)): i.value_json for i in items}
        if len(d) > len(best):
            best = d
keys = sorted(k for k in best if k and not k.startswith("_"))
print(f"TOTAL logged keys: {len(keys)}")
for pref in [
    "track_radial_",
    "track_param_",
    "track_update_",
    "track_spectrum",
    "track_gram_",
    "plot_",
    "scalar_param_",
]:
    print(f"  {pref:<18} {len([k for k in keys if k.startswith(pref)])}")
rad = sorted({k.split("/")[0] for k in keys if k.startswith("track_radial_")})
print(f"\nRADIAL METRICS: {len(rad)}")
bad = []
for m in rad:
    ex = [k for k in keys if k.startswith(m + "/")]
    vals = []
    for k in ex:
        try:
            vals.append(float(json.loads(best[k])))
        except Exception:
            pass
    nz = sum(1 for v in vals if v != 0.0)
    fin = all(v == v and abs(v) != float("inf") for v in vals)
    if not fin:
        bad.append(m)
    print(
        f"  {m[13:]:<26} n={len(ex):<4} nonzero={nz:<4} finite={fin}  sample={vals[0]:.6g}"
        if vals
        else f"  {m[13:]:<26} n={len(ex)}"
    )
print()
if bad:
    print("NON-FINITE METRICS:", bad)
    sys.exit(1)
print("all logged radial values are finite")
