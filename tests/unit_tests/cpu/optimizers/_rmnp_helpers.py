# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The shared constants and data of the RMNP row-direction / deflate tests, copied
verbatim from wt-s11's test_angular_muown_batch6.py (that module stays on the side
branch: it tests the side line's length rules)."""

import torch

from torchtitan.optimizers.disco.presets import EtaSchedule

LR, M_START, B1, BETA = 0.05, 4.0, 0.9, 0.95
F_ETA = EtaSchedule("inverse_poly", start=1, c=0.5)
F_ETA_DICT = {"shape": "inverse_poly", "start": 1, "c": 0.5, "p": 1.0}


def _polar(X):
    u, _, vh = torch.linalg.svd(X, full_matrices=False)
    return u @ vh


def _data(m, n, seed, steps=6):
    """Rows of unequal norm, and gradients."""
    gen = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(
        torch.randn(max(m, n), max(m, n), generator=gen, dtype=torch.float64)
    )[0][:m, :n]
    W = W * torch.logspace(-0.5, 0.5, m, dtype=torch.float64)[:, None]
    return W, [
        0.1 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        for _ in range(steps)
    ]
