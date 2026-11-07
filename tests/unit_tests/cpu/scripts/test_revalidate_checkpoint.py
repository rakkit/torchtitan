# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect

from scripts.revalidate_checkpoint import _LossCapture
from torchtitan.components.validator import Validator


def test_loss_capture_takes_the_validators_arguments():
    """Validator.validate calls log_validation(loss=, step=, tag=); the
    script's stand-in for the metrics processor must accept the same call."""
    assert "tag=tag" in inspect.getsource(Validator.validate)
    capture = _LossCapture()
    capture.log_validation(loss=2.5, step=7, tag=None)
    assert capture.loss == 2.5
