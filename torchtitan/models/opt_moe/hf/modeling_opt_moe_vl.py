# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face format of torchtitan's ``OPTMoEVLModel`` (SigLIP 2 encoder):
``OptMoeForCausalLM`` + SigLIP 2's vision tower (transformers'
``SiglipVisionModel``, no pooling head) + the Qwen3.5-style connector, for
``generate`` and eval harnesses.

Images are InternVL tiles (``tile_images``): each 512 x 512 square gives 32 x 32
patches, the connector merges 2 x 2 neighbours into one LLM token (256 per
square), and the prompt holds one ``<|image_pad|>`` per token between
``<|vision_start|>`` and ``<|vision_end|>`` (``image_prompt``). The prefill
replaces the pads' embeddings by the image features in order; later steps
reuse the cache.

``config.vision_config``: SigLIP 2's vision config plus ``spatial_merge_size``,
``tile_size``, ``max_tiles`` and ``image_token_id``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import SiglipVisionConfig, SiglipVisionModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_opt_moe import OptMoeForCausalLM

_VISION_KEYS = ("spatial_merge_size", "image_token_id")  # the model's, not SigLIP 2's


class OptMoeVisionMerger(nn.Module):
    """LayerNorm per patch, 2 x 2 neighbours concatenated (row, then column
    within the pair: torchtitan's block order), biased 2-layer MLP."""

    def __init__(self, dim: int, out_dim: int, merge: int, eps: float):
        super().__init__()
        self.merge = merge
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.linear_fc1 = nn.Linear(dim * merge * merge, dim * merge * merge)
        self.act_fn = nn.GELU(approximate="tanh")
        self.linear_fc2 = nn.Linear(dim * merge * merge, out_dim)

    def forward(self, patches_BPC: torch.Tensor) -> torch.Tensor:
        b, p, c = patches_BPC.shape
        side, m = int(math.isqrt(p)), self.merge
        x = self.norm(patches_BPC).view(b, side // m, m, side // m, m, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            b * (side // m) ** 2, m * m * c
        )  # (bh, bw) tokens of (m, n, c)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class OptMoeForConditionalGeneration(OptMoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        vision = dict(config.vision_config)
        self.vision_extra = {k: vision.pop(k) for k in _VISION_KEYS}
        vision_config = SiglipVisionConfig(**vision, vision_use_head=False)
        self.visual = SiglipVisionModel(vision_config)
        self.merger = OptMoeVisionMerger(
            vision_config.hidden_size,
            config.hidden_size,
            self.vision_extra["spatial_merge_size"],
            vision_config.layer_norm_eps,
        )
        self.post_init()

    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``pixel_values`` ``[N tiles, 3, 512, 512]`` (normalized) -> ``[N * 256, D]``."""
        hidden = self.visual(
            pixel_values=pixel_values
        ).last_hidden_state  # after post_layernorm
        return self.merger(hidden)

    def forward(
        self, input_ids=None, pixel_values=None, inputs_embeds=None, **kwargs
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pads = input_ids == self.vision_extra["image_token_id"]
                features = self.image_features(pixel_values.to(inputs_embeds.dtype))
                if int(pads.sum()) != features.shape[0]:
                    raise ValueError(
                        f"{int(pads.sum())} <|image_pad|> tokens, {features.shape[0]} image features"
                    )
                inputs_embeds = inputs_embeds.masked_scatter(
                    pads.unsqueeze(-1), features.to(inputs_embeds.dtype)
                )
        return super().forward(inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, pixel_values=None, **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )
        first = past_key_values is None or past_key_values.get_seq_length() == 0
        inputs["pixel_values"] = (
            pixel_values if first else None
        )  # images enter in the prefill only
        return inputs


def tile_grid(
    width: int, height: int, *, max_tiles: int, tile_size: int
) -> tuple[int, int]:
    """InternVL's ``find_closest_aspect_ratio`` (a copy of
    ``torchtitan/components/data/multimodal/internvl_tiling.py``'s, so the exported folder
    needs no torchtitan): the ``(columns, rows)`` grid of at most ``max_tiles``."""
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


def tile_images(images, *, tile_size: int = 512, max_tiles: int = 12):
    """PIL images of one sample -> normalized tiles ``[N, 3, tile, tile]`` and
    the LLM tokens per image (256 per tile with patch 16 and a 2 x 2 merge):
    InternVL's dynamic tiling with the budget split across the sample's images,
    as in training (``torchtitan/components/data/multimodal/internvl_tiling.py``)."""
    import torchvision.transforms.v2.functional as TVF

    tiles, tokens = [], []
    budget = max(1, max_tiles // max(1, len(images)))
    for image in images:
        x = TVF.pil_to_tensor(image.convert("RGB"))
        _, h, w = x.shape
        columns, rows = tile_grid(w, h, max_tiles=budget, tile_size=tile_size)
        resized = TVF.resize(
            x,
            [rows * tile_size, columns * tile_size],
            interpolation=TVF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        t = (
            resized.unfold(1, tile_size, tile_size)
            .unfold(2, tile_size, tile_size)
            .permute(1, 2, 0, 3, 4)
        )
        t = t.reshape(rows * columns, 3, tile_size, tile_size)
        if rows * columns > 1:
            thumb = TVF.resize(
                x,
                [tile_size, tile_size],
                interpolation=TVF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            t = torch.cat([t, thumb.unsqueeze(0)])
        tiles.append(t)
        tokens.append(t.shape[0] * (tile_size // 32) ** 2)
    pixels = (torch.cat(tiles).float() / 255.0 - 0.5) / 0.5
    return pixels, tokens


def image_prompt(num_tokens: int) -> str:
    """The text that stands for one image in a prompt."""
    return "<|vision_start|>" + "<|image_pad|>" * num_tokens + "<|vision_end|>"


__all__ = [
    "OptMoeForConditionalGeneration",
    "OptMoeVisionMerger",
    "image_prompt",
    "tile_images",
]
