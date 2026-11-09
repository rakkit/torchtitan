# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TitanRL renderer for opt_moe chat models: the renderers library's Qwen3.8 renderer, text only,
reasoning effort medium.

Our chat template follows Qwen3.8's conventions (ChatML turns, ``<think>``, XML tool calls), so
the library's ``Qwen38Renderer`` gives the same prompts, except that

- it looks up Qwen's four vision tokens at construction, which our text-only tokenizer does
  not have (``AssertionError: Special token '<|vision_start|>' not found``);
- it defaults to reasoning effort ``xhigh``, which writes an effort line into every prompt that
  our template (default ``medium``, no line) never writes.

Select it with
``CustomRendererConfig(import_path="torchtitan.models.opt_moe.renderer.OptMoeChatRenderer")``.
"""

from __future__ import annotations

from typing import Any, Literal

from renderers.configs import Qwen38RendererConfig
from renderers.qwen38 import Qwen38Renderer

_VISION_TOKENS = frozenset(
    {"<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"}
)


class OptMoeChatRendererConfig(Qwen38RendererConfig):
    """Qwen3.8 renderer config with our template's default reasoning effort."""

    # pyrefly: ignore [bad-override]
    name: Literal["opt_moe_chat"] = "opt_moe_chat"
    reasoning_effort: Literal["xhigh", "medium", "low"] = "medium"


class OptMoeChatRenderer(Qwen38Renderer):
    """``Qwen38Renderer`` for a tokenizer without vision tokens; image parts are refused."""

    config_class = OptMoeChatRendererConfig

    def _token_id(self, token: str) -> int:
        if token in _VISION_TOKENS:
            return -1  # never emitted: text only
        return super()._token_id(token)

    # pyrefly: ignore [bad-override]
    def _process_image(self, part: dict[str, Any]):
        raise ValueError(f"{type(self).__name__} is text only; got an image part")
