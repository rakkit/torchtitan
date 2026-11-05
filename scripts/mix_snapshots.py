# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mix weight snapshots (WeightAverages.snapshot_steps) into one weight file
for `averages.score_files`.

  python scripts/mix_snapshots.py OUT.pt SNAP.pt [SNAP.pt ...] --gamma 3 [--start 0]
  python scripts/mix_snapshots.py OUT.pt SNAP.pt [SNAP.pt ...] --weights 0.2 0.8

--gamma: the power average over the snapshots' steps s - start (weight (s - start)^gamma);
--weights: explicit mixture weights (normalised to sum 1)."""

import argparse
import math

import torch

from torchtitan.components.optim.averages import mix_snapshots, power_weights


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("snapshots", nargs="+")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gamma", type=float)
    g.add_argument("--weights", type=float, nargs="+")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args(argv)
    blobs = [torch.load(p, map_location="cpu", weights_only=True) for p in a.snapshots]
    steps = [b["step"] for b in blobs]
    if steps != sorted(steps):
        raise ValueError(f"snapshots must be in step order: {steps}")
    if a.gamma is not None:
        w = power_weights([s - a.start for s in steps], a.gamma)
    else:
        tot = math.fsum(a.weights)
        w = [x / tot for x in a.weights]
    torch.save(
        {"weights": mix_snapshots(blobs, w), "mixed": list(zip(steps, w))}, a.out
    )
    print(f"{a.out}: {len(blobs)} snapshots, weights {[round(x, 6) for x in w]}")


if __name__ == "__main__":
    main()
