# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-turn SFT: train on every assistant turn, mask everything else.

WHY THIS EXISTS
---------------
Upstream's ``ChatProcessor`` (``hf_datasets/text_datasets.py:82-98``) rejects multi-turn
conversations outright::

    # TODO(data-sft-multiturn): Extend validation and loss masking before
    # accepting multi-turn conversations.
    if len(messages) != 2:
        raise ValueError(f"Expected single-turn [user, assistant], got ...")

and masks by prompt PREFIX LENGTH, which only works because there is exactly one
assistant turn. A real conversation has assistant turns interleaved with user turns, so
a prefix rule would either train on user text or train on only the last answer.

This processor emits a ``TextSequence`` so upstream's packing and ``TextCollator``
produce token-flat ``[T]`` batches with ``num_valid_tokens`` -- unlike the fork's
``experiments/sft/``, which yields ``{"input", "positions"}`` and therefore cannot run on
0.5.0 at all (``trainer.py`` unconditionally pops ``num_valid_tokens``).

HOW THE SPANS ARE FOUND
-----------------------
Tokenize the whole conversation ONCE -- those are the authoritative ids -- then locate
each assistant turn by re-rendering prefixes::

    start = len(tokenize(messages[:i], add_generation_prompt=True))
    stop  = len(tokenize(messages[:i+1]))
    train on [start, stop)

``add_generation_prompt=True`` renders the prompt THROUGH the assistant header
(``<|assistant|>`` or equivalent), so the header is excluded from the supervised span by
construction rather than by a hand-set token count. A template that ignores the flag
degrades to training on the header, which is what a hand-set count of 0 did anyway.

That is ``1 + 2k`` template renders for k assistant turns, versus O(n^2) for a
pairwise prefix diff.

COST
----
Chat templates render in **Python** Jinja, so unlike pretrain tokenization (Rust, GIL
released) this does not scale across threads. It is the one genuinely GIL-bound part of
the data path.
"""

from __future__ import annotations
import logging

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np

import tyro

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import SampleProcessor, TextSequence
from torchtitan.components.data.keyed_loader import KeyedMixDataLoader
from torchtitan.components.data.mix import build_mix, DatasetSpec, replace_columns
from torchtitan.components.data.packing import FirstFitPackingConfig
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX


def _as_list(value: Any) -> list:
    """Accept a column that holds either a real list or a JSON string.

    The production SFT corpus stores `messages` and `tools` as JSON STRINGS
    (verified on sft-v2-64k: dtype `str` holding `[{"role": ...}]`), because parquet
    cannot express a ragged struct list cheaply. Assuming a list here would fail with an
    unhelpful TypeError deep inside the template render.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        if not text.strip():
            return []
        return json.loads(text)
    return list(value)


def _render(tokenizer: Any, messages: list[dict], **kwargs: Any) -> list[int]:
    """Tokenize a message list through the chat template.

    Accepts either a tokenizer that returns ids directly or one that returns a string,
    so this works with both upstream's Jinja tokenizer and an HF AutoTokenizer.
    """
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(rendered, str):
        return list(tokenizer.encode(rendered, add_bos=False, add_eos=False))
    return list(rendered)


# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
# Row adapters: one per corpus LAYOUT, not per corpus.
#
# This is the one thing the old 937-line SFT loader was right about. Corpora
# genuinely disagree about how a conversation is stored, and no single column
# convention covers them: `DATASET_MESSAGE_BUILDERS` there had five entries.
# Everything else it carried -- the packing pool, best-fit state, shared-memory
# counters, a thread pool -- is grain's job now and is gone.
#
# An adapter maps one row to `(messages, tools, enable_thinking)`. Attach one per
# dataset via `DatasetSpec.row_adapter`; the default handles the common layout.
# ---------------------------------------------------------------------------


def normalize_tool_calls(messages: list[dict]) -> list[dict]:
    """Coerce assistant `tool_calls[*].function.arguments` to a dict.

    Chat templates pipe `arguments` through Jinja's `|items`, which raises on a string.
    Corpora store it as a JSON string, an already-parsed dict, a list, or None depending
    on how they were written.
    """
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        for call in message["tool_calls"]:
            function = call.get("function") or call
            if "arguments" not in function:
                continue
            arguments = function["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(
                        "SFT: unparsable tool_call arguments %.80s", str(arguments)
                    )
                    arguments = {}
            function["arguments"] = arguments if isinstance(arguments, dict) else {}
    return messages


def conversation_rows(
    messages_key: str = "messages",
    tools_key: str | None = "tools",
    thinking_key: str | None = "enable_thinking",
):
    """The common layout: a conversation column, optional tools and thinking columns.

    Tolerates JSON-string columns (parquet cannot express a ragged struct list cheaply,
    so production corpora store them as strings) and `"on"`/`"off"` thinking flags.
    """

    def adapt(row: dict[str, Any]):
        messages = normalize_tool_calls(_as_list(row[messages_key]))
        tools = _as_list(row.get(tools_key)) if tools_key else None
        thinking = row.get(thinking_key) if thinking_key else None
        if isinstance(thinking, str):
            thinking = thinking == "on"
        return messages, tools or None, thinking

    return adapt


def prompt_response_rows(prompt_key: str = "prompt", response_key: str = "response"):
    """A column PAIR rather than a conversation, e.g. prompt/response or question/answer.

    Yields a two-turn conversation, so the same per-assistant-turn masking applies
    unchanged.
    """

    def adapt(row: dict[str, Any]):
        return (
            [
                {"role": "user", "content": row[prompt_key]},
                {"role": "assistant", "content": row[response_key]},
            ],
            None,
            None,
        )

    return adapt


def nested_tools_rows(messages_key: str = "messages", tools_key: str = "functions"):
    """Tools carried INSIDE the first message, and `None` content that must become "".

    A `None` content field renders as the string "None" in most templates, silently
    training the model to emit it.
    """

    def adapt(row: dict[str, Any]):
        messages = _as_list(row[messages_key])
        for message in messages:
            if message.get("content") is None:
                message["content"] = ""
        tools = messages[0].get(tools_key) if messages else None
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                logger.warning("SFT: unparsable tools JSON %.80s", tools)
                tools = None
        return normalize_tool_calls(messages), tools or None, False

    return adapt


class MultiTurnChatProcessor(SampleProcessor):
    """Tokenize a conversation, training only on assistant turns."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        row_adapter: Annotated[
            Callable[[dict[str, Any]], tuple[list[dict], Any, Any]], tyro.conf.Suppress
        ] = None
        """Maps one row to `(messages, tools, enable_thinking)`. Defaults to
        `conversation_rows()`; pick another when the corpus stores its conversations
        differently."""
        chat_template_kwargs: dict[str, Any] | None = None
        """Extra kwargs for every render. The production config sets
        `truncate_history_thinking=False` so PREFIX renders match the full-conversation
        render -- without it the per-turn spans are computed against different text and
        the loss mask silently lands in the wrong place."""
        drop_if_longer_than: int | None = None
        """Skip conversations longer than this many tokens. They cannot be truncated
        without corrupting the turn structure. Set it to the model's `max_context_length`:
        that is what `FirstFitPackingConfig` filters on (`packing.py:77-79`), so anything
        longer is discarded by the packer regardless -- doing it here makes the loss
        countable instead of invisible."""

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._tokenizer = context.tokenizer
        self._adapt = config.row_adapter or conversation_rows()
        self._template_kwargs = dict(config.chat_template_kwargs or {})
        self._drop_if_longer_than = config.drop_if_longer_than
        self._num_dropped = 0
        self._num_seen = 0

    @property
    def num_dropped(self) -> int:
        """Conversations skipped for length."""
        return self._num_dropped

    def _tokenize(self, messages: list[dict], extra: dict[str, Any]) -> list[int]:
        return _render(self._tokenizer, messages, **{**self._template_kwargs, **extra})

    def _assistant_spans(
        self, messages: list[dict], total: int, extra: dict[str, Any]
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            # The generation prompt is the assistant header. Rendering the prefix WITH it
            # puts `start` after the header, so the header stays masked without anyone
            # having to count its tokens.
            start = len(
                self._tokenize(
                    messages[:index], {**extra, "add_generation_prompt": True}
                )
            )
            stop = len(self._tokenize(messages[: index + 1], extra))
            start, stop = min(start, total), min(stop, total)
            if stop > start:
                spans.append((start, stop))
        return spans

    def _record_drop(self, num_tokens: int) -> None:
        """Report dropped conversations. They are otherwise invisible: the packer would
        discard them with no counter and no log, which is the failure this guards."""
        self._num_dropped += 1
        if self._num_dropped == 1 or self._num_dropped % 1000 == 0:
            logger.warning(
                "SFT: dropped %d of %d conversations so far for exceeding %d tokens "
                "(latest was %d). They cannot be truncated without corrupting the turn "
                "structure; raise the model's max_context_length to keep them.",
                self._num_dropped,
                self._num_seen,
                self._drop_if_longer_than,
                num_tokens,
            )

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        self._num_seen += 1
        messages, tools, enable_thinking = self._adapt(sample)
        if not messages:
            return None

        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
        if enable_thinking is not None:
            extra["enable_thinking"] = bool(enable_thinking)
        if not any(m.get("role") == "assistant" for m in messages):
            # Nothing to learn from: every token would be masked, so the sample would
            # contribute a zero-token row and skew packing.
            return None

        token_ids = self._tokenize(messages, extra)
        if len(token_ids) < 2:
            return None
        if (
            self._drop_if_longer_than is not None
            and len(token_ids) > self._drop_if_longer_than
        ):
            self._record_drop(len(token_ids))
            return None

        input_ids = np.asarray(token_ids, dtype=np.int64)
        trainable = np.zeros(len(input_ids), dtype=bool)
        for start, stop in self._assistant_spans(messages, len(input_ids), extra):
            trainable[start:stop] = True

        if not trainable.any():
            return None

        # Next-token alignment, exactly as TextProcessor does it: predict token t+1 from
        # token t. A position is supervised when its TARGET is inside an assistant span.
        labels = np.where(trainable[1:], input_ids[1:], IGNORE_INDEX)
        return TextSequence(input_ids=input_ids[:-1], labels=labels)


def make_sft_dataloader_config(
    datasets: list[DatasetSpec],
    *,
    max_context_length: int,
    chat_template_kwargs: dict[str, Any] | None = None,
    num_packing_bins: int = 32,
    messages_key: str = "messages",
    **loader_kwargs: Any,
) -> KeyedMixDataLoader.Config:
    """Dataloader for multi-turn SFT.

    Differs from pretrain in four ways, each forced rather than chosen:

    1. **first_fit packing, not concat_then_split.** A conversation must not be split
       across rows -- half a dialogue is not a training example. That means padding, and
       therefore `num_packing_bins` well above upstream's default of 8 (measured: 8 bins
       7.7% padding, 32 bins 2.4%).
    2. **Each dataset may carry its own `row_adapter`.** Corpora disagree about how a
       conversation is stored; `DatasetSpec.row_adapter` selects the layout per dataset
       and everything after it is shared. See the adapters above.
    3. **Oversized conversations are dropped, and logged.** The threshold MUST be
       the model's `max_context_length`, not `num_tokens_per_microbatch_per_dp_rank`:
       `FirstFitPackingConfig.build` filters on `context.max_context_length`
       (`packing.py:77-79`), so a conversation longer than that is discarded by the
       packer even when it would fit the wider row. Counting against the row width would
       undercount the drops whenever the two differ.
    4. **The model spec's `enable_token_mask_for_moe` must be True.** first_fit
       pads, and padding otherwise pollutes the MoE load-balance statistics AND
       `tokens_per_expert_E`, which drives the expert-bias update
       (`norm_moe.py:783-793`). Pretrain with concat_then_split has no padding and does
       not need it. This helper cannot set it -- it lives on the model -- so the config
       must, and `check_py_config_data_path.py` should verify it.
    """
    # Columns come from the SPEC when it sets them, because the row adapter decides
    # which columns exist: `prompt_response_rows` reads a prompt/response PAIR and has no
    # `messages` column at all, so forcing one here made two of the three shipped
    # adapters fail startup validation.
    datasets = [
        dataset
        if dataset.columns and dataset.columns != (dataset.text_key,)
        else replace_columns(dataset, (messages_key,), ("tools", "enable_thinking"))
        for dataset in datasets
    ]
    mix = build_mix(
        datasets,
        # Per-dataset, so a mixture can span corpus layouts. `row_adapter` is the only
        # thing that varies; everything downstream is shared.
        processor_for=lambda dataset: MultiTurnChatProcessor.Config(
            row_adapter=dataset.row_adapter or conversation_rows(messages_key),
            chat_template_kwargs=chat_template_kwargs,
            drop_if_longer_than=max_context_length,
        ),
    )
    # shuffle defaults to False here, NOT to GrainDataLoader.Config's True. True inserts
    # WindowShuffleIterDataset, which (a) is the weak shuffle this design rejects -- a
    # window can only move a row `window` positions -- and (b) renames the state key
    # `parent` to `parent_window_start_state`, which silently zeroed every
    # `data_docs/{alias}` series on this path.
    loader_kwargs.setdefault("shuffle", False)
    return KeyedMixDataLoader.Config(
        dataset=FirstFitPackingConfig(dataset=mix, num_packing_bins=num_packing_bins),
        collator=TextCollator.Config(),
        dataset_ids=tuple(dataset.alias for dataset in datasets),
        dataset_weights=tuple(float(dataset.weight) for dataset in datasets),
        **loader_kwargs,
    )
