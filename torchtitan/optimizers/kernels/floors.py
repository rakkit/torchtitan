# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Below these a normalisation returns 0 (it divides by `inf`) instead of
dividing a tiny input by the bare `eps` guard.

The cut is a margin above where fp32 fails, not the failure point: a row norm
from subnormal squares is still accurate to 1e-5 at entries of 1e-21 and 0.1%
at 1e-22, degrades from ~5e-23, and reads 0 below 2**-75 (~2.6e-23), where
`x / (0 + 1e-30)` blew up. A moment that small has decayed for hundreds of
steps without a gradient, so stopping its update early is the intent."""

import torch

SUMSQ_FLOOR = torch.finfo(torch.float32).tiny  # ~1.2e-38
NORM_FLOOR = SUMSQ_FLOOR**0.5  # ~1.1e-19
