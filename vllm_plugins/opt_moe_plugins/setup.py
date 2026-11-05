# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup


setup(
    name="opt_moe_plugins",
    version="0.1.0",
    description="Out-of-tree vLLM plugins: StagingMoEllama and OptMoE",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins": [
            "register_staging_moellama = vllm_staging_moellama:register",
            "register_opt_moe = vllm_opt_moe:register",
        ]
    },
)
