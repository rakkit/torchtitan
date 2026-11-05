# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""With per-rank logging or PP several ranks export at the same step into the
same folder: each needs its own file, or they overwrite each other."""

import torch.distributed as dist

from torchtitan.optimizers.export.spectrum_logging import export_path


def test_each_rank_writes_its_own_file(monkeypatch, tmp_path):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    paths = set()
    for rank in range(3):
        monkeypatch.setattr(dist, "get_rank", lambda rank=rank: rank)
        paths.add(export_path(str(tmp_path), 5))
    assert len(paths) == 3
