# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe's VLM in vLLM: ``OptMoeForConditionalGeneration``, the architecture
of an HF folder written by torchtitan's ``convert_opt_moe_vl_to_hf.py`` (its
reference is that folder's ``modeling_opt_moe_vl.py``): SigLIP 2's vision tower
(vLLM's ``SiglipVisionTransformer``, whose weights map 1:1 onto the folder's
``visual.*``) + the connector (``merger.*``) + the text model of ``model.py``,
unchanged.

An image is InternVL's dynamic tiling (``tile_image``): ``tile_size`` squares
on the grid closest to the aspect ratio (at most ``max_tiles``, the budget split
across the prompt's images), plus a thumbnail when there is more than one
square; each square is 32 x 32 patches, the connector merges 2 x 2 neighbours,
so 256 LLM tokens per square. The prompt holds ONE ``<|image_pad|>`` per image
(``<|vision_start|><|image_pad|><|vision_end|>``); the processor expands it to
its tiles' tokens, and the pads' embeddings are replaced by the connector's
output. ``--limit-mm-per-prompt '{"image": 0}'`` does not build the tower and
the connector.

Tensor suffixes: ``B`` tiles, ``P`` patches of a tile, ``C`` vision width."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace

import torch
from torch import nn
from transformers import BatchFeature, SiglipVisionConfig

from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder,
    support_torch_compile,
)
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.siglip import SiglipVisionTransformer
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.transformers_utils.processors.internvl import (
    find_closest_aspect_ratio,
    get_internvl_target_ratios,
)

from .model import OptMoeForCausalLM

IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def tile_image(image, *, tile_size: int, max_tiles: int) -> torch.Tensor:
    """A PIL image -> its uint8 tiles ``[N, 3, tile, tile]``: the grid's squares
    row by row, then the thumbnail when N > 1 (the HF reference's
    ``tile_images`` for one image, its grid rule is vLLM's InternVL one; the
    model normalizes to [-1, 1] on the GPU)."""
    import torchvision.transforms.v2.functional as TVF

    x = TVF.pil_to_tensor(image.convert("RGB"))
    _, h, w = x.shape
    columns, rows = find_closest_aspect_ratio(
        w / h,
        get_internvl_target_ratios(1, max_tiles),
        width=w,
        height=h,
        image_size=tile_size,
    )
    bicubic = TVF.InterpolationMode.BICUBIC
    resized = TVF.resize(
        x,
        [rows * tile_size, columns * tile_size],
        interpolation=bicubic,
        antialias=True,
    )
    tiles = (
        resized.unfold(1, tile_size, tile_size)
        .unfold(2, tile_size, tile_size)
        .permute(1, 2, 0, 3, 4)
        .reshape(rows * columns, 3, tile_size, tile_size)
    )
    if rows * columns > 1:
        thumb = TVF.resize(
            x, [tile_size, tile_size], interpolation=bicubic, antialias=True
        )
        tiles = torch.cat([tiles, thumb.unsqueeze(0)])
    return tiles


class OptMoeProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class OptMoeDummyInputsBuilder(BaseDummyInputsBuilder[OptMoeProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_BLOCK * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        v = self.info.get_hf_config().vision_config
        size = v["tile_size"]
        # the largest grid (the last, they are sorted by squares): an image of its shape
        columns, rows = get_internvl_target_ratios(1, v["max_tiles"])[-1]
        return {
            "image": self._get_dummy_images(
                width=columns * size,
                height=rows * size,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class OptMoeMultiModalProcessor(BaseMultiModalProcessor[OptMoeProcessingInfo]):
    """No HF processor: the tiling above runs here, on the frontend's CPU."""

    def apply(self, inputs, timing_ctx):
        # InternVL's multi-image rule as in training: the prompt's max_tiles
        # budget is split across its images (in the kwargs, so in the cache key)
        budget = self.info.get_hf_config().vision_config["max_tiles"]
        count = max(1, inputs.mm_data_items.get_all_counts().get("image", 0))
        kwargs = {**inputs.hf_processor_mm_kwargs, "max_tiles": max(1, budget // count)}
        return super().apply(replace(inputs, hf_processor_mm_kwargs=kwargs), timing_ctx)

    def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs):
        images = self._get_hf_mm_data(mm_items)[0].get("images", [])
        if not images:
            return BatchFeature()
        size = self.info.get_hf_config().vision_config["tile_size"]
        tiles = [
            tile_image(
                image, tile_size=size, max_tiles=hf_processor_mm_kwargs["max_tiles"]
            )
            for image in images
        ]
        return BatchFeature(
            {
                "pixel_values": torch.cat(tiles),
                "image_num_tiles": torch.tensor([len(t) for t in tiles]),
            }
        )

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        sizes = hf_inputs.get("image_num_tiles", torch.empty(0))
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", sizes),
            "image_num_tiles": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        v = self.info.get_hf_config().vision_config
        pad = v["image_token_id"]
        # 32 x 32 patches of 16 px per 512-px tile, 2 x 2 merged: 256
        per_tile = (v["tile_size"] // (v["patch_size"] * v["spatial_merge_size"])) ** 2
        tiles = out_mm_kwargs.get_data().get("image_num_tiles", torch.empty(0)).tolist()
        return [
            PromptReplacement(
                modality="image",
                target=[pad],
                replacement=lambda i: [pad] * (tiles[i] * per_tile),
            )
        ]


@support_torch_compile(
    dynamic_arg_dims={"pixel_values": 0},
    enable_if=should_torch_compile_mm_encoder,
    is_encoder=True,
)
class OptMoeVisionTower(SiglipVisionTransformer):
    """vLLM's SigLIP tower, compiled as a whole with ``compile_mm_encoder`` (as
    its InternVL / Qwen-VL encoders are); the number of tiles is dynamic."""


class OptMoeVisionMerger(nn.Module):
    """The HF reference's connector: LayerNorm per patch, 2 x 2 neighbours
    concatenated (row, then column within the pair: torchtitan's block order),
    biased 2-layer MLP with tanh GELU."""

    def __init__(self, dim: int, out_dim: int, merge: int, eps: float):
        super().__init__()
        self.merge = merge
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.linear_fc1 = nn.Linear(dim * merge * merge, dim * merge * merge)
        self.act_fn = nn.GELU(approximate="tanh")
        self.linear_fc2 = nn.Linear(dim * merge * merge, out_dim)

    def forward(self, patches_BPC: torch.Tensor) -> torch.Tensor:
        b, p, c = patches_BPC.shape
        side, m = math.isqrt(p), self.merge
        x = self.norm(patches_BPC).view(b, side // m, m, side // m, m, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * (side // m) ** 2, m * m * c)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


@MULTIMODAL_REGISTRY.register_processor(
    OptMoeMultiModalProcessor,
    info=OptMoeProcessingInfo,
    dummy_inputs=OptMoeDummyInputsBuilder,
)
class OptMoeForConditionalGeneration(nn.Module, SupportsMultiModal):
    packed_modules_mapping = OptMoeForCausalLM.packed_modules_mapping
    # the checkpoint's text names (model.*, lm_head.*) -> the language_model child
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_BLOCK
        raise ValueError("Only the image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        vision = config.vision_config
        siglip = SiglipVisionConfig(**vision)
        # not built with --limit-mm-per-prompt '{"image": 0}'; the tower is not quantized
        # (keyword arguments: support_torch_compile type-checks positional ones)
        with self._mark_tower_model(vllm_config, "image"):
            self.visual = OptMoeVisionTower(
                config=siglip,
                quant_config=None,
                prefix=f"{prefix}visual",
                use_head=False,
            )
            self.merger = OptMoeVisionMerger(
                siglip.hidden_size,
                config.hidden_size,
                vision["spatial_merge_size"],
                siglip.layer_norm_eps,
            )
        with self._mark_language_model(vllm_config):
            self.language_model = OptMoeForCausalLM(
                vllm_config=vllm_config, prefix=f"{prefix}language_model."
            )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return []
        tiles = ((pixel_values.float() / 255.0 - 0.5) / 0.5).to(self.visual.dtype)
        features = self.merger(self.visual(tiles))
        per_tile = features.shape[0] // tiles.shape[0]
        return features.split(
            [n * per_tile for n in kwargs["image_num_tiles"].tolist()]
        )

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return self.language_model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
