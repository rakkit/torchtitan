# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""InternVL's dynamic tiling, the image preprocessing of DataComp-VLM, for an
encoder trained at one square size (SigLIP 2 So400m/16 at 512 px).

An image is resized to the grid of ``tile_size`` squares whose aspect ratio is
closest to the image's (at most ``max_tiles`` squares), cut into those squares
row by row, and, when there is more than one square, followed by a thumbnail of
the whole image at ``tile_size``. The squares are stacked as frames of one item
(``temporal_patch_size`` 1), so the encoder attends within each square and the
prompt gets ``num_squares * tile_size**2 / (patch_size * merge_size)**2``
placeholders (256 per 512-px square with patch 16 and a 2x2 merge)::

    1024 x 768 image (4:3), max 12 squares:
        grid 4 x 3 = 12 squares of 512 x 512, then a 512 x 512 thumbnail
        -> grid_thw [13, 32, 32], 13 * 256 = 3,328 placeholders

The choice of grid follows ``internvl_chat/internvl/train/dataset.py``
(``find_closest_aspect_ratio`` / ``dynamic_preprocess``): among the grids of
``1 .. max_tiles`` squares sorted by square count, the closest aspect ratio
wins; on a tie the larger grid wins when the image's area exceeds half of the
grid's. A sample with several images splits the budget:
``max(1, max_tiles // num_images)`` squares each (InternVL's multi-image rule).
"""

from dataclasses import dataclass

import torch
from PIL import Image

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor


def tile_grid(
    width: int, height: int, *, max_tiles: int, tile_size: int
) -> tuple[int, int]:
    """InternVL's ``find_closest_aspect_ratio``: the ``(columns, rows)`` grid of
    at most ``max_tiles`` squares for a ``width x height`` image."""
    aspect_ratio = width / height
    grids = sorted(
        {
            (i, j)
            for n in range(1, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_tiles
        },
        key=lambda grid: grid[0] * grid[1],
    )
    best_diff, best = float("inf"), (1, 1)
    for columns, rows in grids:
        diff = abs(aspect_ratio - columns / rows)
        if diff < best_diff:
            best_diff, best = diff, (columns, rows)
        elif (
            diff == best_diff and width * height > 0.5 * tile_size**2 * columns * rows
        ):
            best = (columns, rows)
    return best


class InternVLTilingVisionProcessor(VisionProcessor):
    """``VisionProcessor`` whose images are InternVL tiles (module docstring).
    ``Config.resize`` is not used for images; videos are not supported."""

    @dataclass(kw_only=True, slots=True)
    class Config(VisionProcessor.Config):
        tile_size: int
        """Side of one square in pixels: the encoder's training resolution."""

        max_tiles: int
        """Squares per sample (InternVL's ``max_dynamic_patch``; DataComp-VLM
        uses 12)."""

        def __post_init__(self) -> None:
            VisionProcessor.Config.__post_init__(self)
            if self.temporal_patch_size != 1:
                raise ValueError(
                    "Tiles are frames of one item: temporal_patch_size must be 1."
                )
            if self.tile_size % (self.patch_size * self.spatial_merge_size):
                raise ValueError(
                    "tile_size must be a multiple of patch_size * spatial_merge_size."
                )
            if self.video_resize is not None:
                raise ValueError("InternVLTilingVisionProcessor takes no video.")

    def resize_and_normalize(
        self, image: str | bytes | Image.Image, *, num_images_in_sample: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one image and cut it into tiles.

        Returns float32 ``(N, tile, tile, C)`` pixels (N squares, then the
        thumbnail when N > 1) and the patch grid ``[N(+1), tile / patch,
        tile / patch]``; ``num_images_in_sample`` splits the sample's
        ``max_tiles`` budget across its images."""
        import torchvision.transforms.v2.functional as TVF

        config = self.config
        size = config.tile_size
        image_CHW = self._decode_image(image)
        if (
            image_CHW.ndim == 4
        ):  # an animated image: its first frame, as InternVL (PIL opens frame 0)
            image_CHW = image_CHW[0]
        _, height, width = image_CHW.shape
        columns, rows = tile_grid(
            width,
            height,
            max_tiles=max(1, config.max_tiles // num_images_in_sample),
            tile_size=size,
        )
        mode = TVF.InterpolationMode(config.image_interpolation_mode)
        resized_CHW = TVF.resize(
            image_CHW, [rows * size, columns * size], interpolation=mode, antialias=True
        )
        # row by row, left to right: (rows, columns) squares of size x size
        tiles_NCHW = (
            resized_CHW.unfold(1, size, size)
            .unfold(2, size, size)
            .permute(1, 2, 0, 3, 4)
            .reshape(rows * columns, -1, size, size)
        )
        if rows * columns > 1:
            thumbnail_CHW = TVF.resize(
                image_CHW, [size, size], interpolation=mode, antialias=True
            )
            tiles_NCHW = torch.cat([tiles_NCHW, thumbnail_CHW.unsqueeze(0)])
        return self._normalize(tiles_NCHW)
