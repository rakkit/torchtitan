# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from PIL import Image

from torchtitan.components.data.multimodal import internvl_tiling
from torchtitan.components.data.multimodal.internvl_tiling import (
    InternVLTilingVisionProcessor,
)
from torchtitan.components.data.multimodal.vision_processor import PixelBudgetResize
from torchtitan.models.opt_moe.hf import modeling_opt_moe_vl as hf

# The exported HF folder carries its own copy of the tiling (it must work without
# torchtitan): the two must stay the same, or serving sees other tiles than training.
TILE = 64
SIZES = [(64, 64), (91, 91), (300, 100), (100, 300), (640, 480), (64, 1000), (500, 500)]


@pytest.mark.parametrize("max_tiles", [1, 2, 5, 12])
def test_hf_tile_grid_is_the_training_grid(max_tiles):
    for width, height in SIZES:
        kwargs = dict(max_tiles=max_tiles, tile_size=TILE)
        assert hf.tile_grid(width, height, **kwargs) == internvl_tiling.tile_grid(
            width, height, **kwargs
        )


def test_hf_tiles_are_the_training_tiles():
    generator = torch.Generator().manual_seed(0)
    images = [
        Image.fromarray(
            torch.randint(
                0, 256, (h, w, 3), generator=generator, dtype=torch.uint8
            ).numpy()
        )
        for w, h in SIZES[:4]
    ]
    processor = InternVLTilingVisionProcessor.Config(
        patch_size=16,
        temporal_patch_size=1,
        spatial_merge_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize=PixelBudgetResize.Config(min_pixels=TILE**2, max_pixels=TILE**2),
        patch_order="block",
        image_interpolation_mode="bicubic",
        tile_size=TILE,
        max_tiles=12,
    ).build()
    expected = [
        processor.resize_and_normalize(image, num_images_in_sample=len(images))
        for image in images
    ]
    pixels, tokens = hf.tile_images(images, tile_size=TILE, max_tiles=12)
    torch.testing.assert_close(
        pixels, torch.cat([p for p, _ in expected]).permute(0, 3, 1, 2)
    )
    assert tokens == [processor.num_placeholder_tokens(grid) for _, grid in expected]
