# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every norm parameter of a built opt_moe model has a declared placement (the
q / k, block and final norms, with and without gains): `fully_shard` rejects a
plain-tensor parameter under spmd_types. And a model config built for one
layout refuses another."""

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.opt_moe import build_model_config


@pytest.mark.parametrize("flavor", ["gauge-d16-w512", "mis-8b-proxy"])  # dense, MoE
@pytest.mark.parametrize("norm_type", ["rmsnorm", "np_rmsnorm"])
def test_every_norm_parameter_has_a_placement(flavor, norm_type):
    cfg = build_model_config(
        flavor,
        parallelism=ParallelismConfig(),
        seq_len=256,
        n_layers=2,
        swa_pattern=None,
        rope_pattern=None,
        norm_type=norm_type,
        qk_norm=True,
    )
    with torch.device("meta"):
        model = cfg.build()
    norms = [(n, m) for n, m in model.named_modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 2 * 2 + 2 * 2 + 1  # q / k, attention / ffn norms, final
    uncovered = [
        f"{name}.{p}"
        for name, mod in norms
        for p, _ in mod.named_parameters(recurse=False)
        if mod._sharding_config is None or p not in mod._sharding_config.state_shardings
    ]
    assert not uncovered
    assert (norm_type == "rmsnorm") == any(m.weight is not None for _, m in norms)


def test_a_model_built_for_one_layout_refuses_another():
    """build_model_config shards for a parallelism; overriding EP / CP / TP / SP
    afterwards (a command-line override) would otherwise be ignored by the model."""
    cfg = build_model_config(
        "gauge-d8-w256", parallelism=ParallelismConfig(), seq_len=256
    )
    with torch.device("meta"):
        model = cfg.build()
    with pytest.raises(ValueError, match="built for"):
        model.parallelize(parallelism=ParallelismConfig(expert_parallel_degree=2))
