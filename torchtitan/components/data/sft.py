# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-turn SFT: train on every assistant turn, mask everything else.

Upstream's ``ChatProcessor`` takes single-turn conversations only and masks by the
prompt's prefix length; a multi-turn conversation interleaves assistant turns with user
turns, so every assistant span is found here. The processor emits a ``TextSequence``, so
upstream's packing and ``TextCollator`` produce token-flat ``[T]`` batches with
``num_valid_tokens``.

HOW THE SPANS ARE FOUND
-----------------------
Tokenize the whole conversation ONCE -- those are the authoritative ids -- then locate
each assistant turn by re-rendering prefixes::

    full  = tokenize(messages)
    start = common_prefix(tokenize(messages[:i], add_generation_prompt=True), full)
    stop  = common_prefix(tokenize(messages[:i+1]), full)
    train on [start, stop)

``add_generation_prompt=True`` renders the prompt THROUGH the assistant header
(``<|assistant|>`` or equivalent), so the header is excluded from the supervised span by
construction rather than by a hand-set token count. A template that ignores the flag
trains on the header.

The span starts where the generation prompt stops agreeing with the full render, not at
the generation prompt's length: the two differ whenever the template renders a new turn's
header differently from the same turn in history. Qwen3-style thinking templates do: the
generation prompt ends ``<think>\\n`` (open, for the model to fill) while a history turn
without reasoning renders ``<think></think>\\n`` (measured on sft-v2-64k: 168 of 1,476
assistant turns in 300 conversations). The prompt's length would start the span one token
late and mask ``</think>``; the common prefix trains the model to close the block. Such
turns are counted (``num_prompt_mismatches``) and logged.

That is ``1 + 2k`` template renders for k assistant turns, versus O(n^2) for a
pairwise prefix diff.

COST
----
Chat templates render in **Python** Jinja, so unlike pretrain tokenization (Rust, GIL
released) this does not scale across threads. It is the one genuinely GIL-bound part of
the data path.
"""

from __future__ import annotations

import json
import logging
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


def _common_prefix_length(prefix: list[int], full: np.ndarray) -> int:
    """The number of leading tokens ``prefix`` and ``full`` share."""
    n = min(len(prefix), len(full))
    differ = np.flatnonzero(np.asarray(prefix[:n], dtype=full.dtype) != full[:n])
    return int(differ[0]) if differ.size else n


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
# Row adapters: one per corpus LAYOUT, not per corpus. Corpora disagree about
# how a conversation is stored, and no single column convention covers them.
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


def _has_reasoning(message: dict[str, Any]) -> bool:
    """An assistant turn carries reasoning: a ``reasoning_content``, or content that opens
    with a ``<think>`` block holding text. An empty block (``<think>\n\n</think>``, as
    non-thinking rows write it) is no reasoning."""
    if str(message.get("reasoning_content") or "").strip():
        return True
    content = message.get("content")
    if not isinstance(content, str) or not content.lstrip().startswith("<think>"):
        return False
    inner = content.lstrip()[len("<think>") :].split("</think>", 1)[0]
    return bool(inner.strip())


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
        """Skip conversations whose training sequence (the rendered tokens minus one:
        inputs predict the next token) is longer than this. They cannot be truncated
        without corrupting the turn structure. Set it to the model's
        `max_context_length`: `FirstFitPackingConfig` SPLITS longer sequences into
        `max_context_length` pieces (`_SplitTextSequenceDocuments`), so without this
        drop a long conversation would be trained as unrelated fragments."""

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._tokenizer = context.tokenizer
        self._adapt = config.row_adapter or conversation_rows()
        self._template_kwargs = dict(config.chat_template_kwargs or {})
        self._drop_if_longer_than = config.drop_if_longer_than
        self._num_dropped = 0
        self._num_seen = 0
        self._num_prompt_mismatches = 0
        self._num_turn_mismatches = 0
        self._num_turns_off_flag = 0  # assistant turns left untrained: reasoning presence != the row's thinking flag
        self._num_untrained = 0  # conversations skipped: no assistant turn to train on

    @property
    def num_dropped(self) -> int:
        """Conversations skipped for length."""
        return self._num_dropped

    @property
    def num_turns_off_flag(self) -> int:
        """Assistant turns left as context: reasoning presence differs from the row's flag."""
        return self._num_turns_off_flag

    @property
    def num_untrained(self) -> int:
        """Conversations skipped with no assistant turn to train on."""
        return self._num_untrained

    @property
    def num_turn_mismatches(self) -> int:
        """Assistant turns whose own render is not a token prefix of the full
        render (the span stops at their common prefix)."""
        return self._num_turn_mismatches

    @property
    def num_prompt_mismatches(self) -> int:
        """Assistant turns whose generation-prompt render is not a token prefix of the
        full render (the span starts at their common prefix)."""
        return self._num_prompt_mismatches

    def _tokenize(self, messages: list[dict], extra: dict[str, Any]) -> list[int]:
        return _render(self._tokenizer, messages, **{**self._template_kwargs, **extra})

    def _assistant_spans(
        self, messages: list[dict], input_ids: np.ndarray, extra: dict[str, Any]
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            if _has_reasoning(message) != extra["enable_thinking"]:
                # A turn that does not match the row's thinking flag stays as context: its
                # span under this flag's generation prompt would teach the wrong form.
                self._record_turn_off_flag()
                continue
            # The generation prompt is the assistant header. Rendering the prefix WITH it
            # puts `start` after the header, so the header stays masked without anyone
            # having to count its tokens.
            prompt = self._tokenize(
                messages[:index], {**extra, "add_generation_prompt": True}
            )
            start = _common_prefix_length(prompt, input_ids)
            if start < len(prompt):
                self._record_prompt_mismatch()
            turn = self._tokenize(messages[: index + 1], extra)
            stop = _common_prefix_length(turn, input_ids)
            if stop <= start:
                # The turn's own render and the full conversation stop agreeing
                # before its content: the template renders an earlier message
                # differently once the conversation goes on.
                raise ValueError(
                    f"SFT: assistant turn {index} has no tokens to train on: the "
                    "chat template renders an earlier message differently once "
                    "the conversation continues (e.g. template kwargs that change "
                    "messages before the last user turn, such as low_effort or "
                    "truncate_history_thinking=True), so the span cannot be found"
                )
            if stop < len(turn):
                self._record_turn_mismatch()
            spans.append((start, stop))
        return spans

    def _record_turn_off_flag(self) -> None:
        self._num_turns_off_flag += 1
        if self._num_turns_off_flag == 1 or self._num_turns_off_flag % 1000 == 0:
            logger.warning(
                "SFT: %d assistant turns so far left untrained: their reasoning does not "
                "match the row's thinking flag (context only).",
                self._num_turns_off_flag,
            )

    def _record_untrained(self) -> None:
        self._num_untrained += 1
        if self._num_untrained == 1 or self._num_untrained % 1000 == 0:
            logger.warning(
                "SFT: skipped %d of %d conversations so far with no assistant turn to "
                "train on (none matches the row's thinking flag).",
                self._num_untrained,
                self._num_seen,
            )

    def _record_turn_mismatch(self) -> None:
        self._num_turn_mismatches += 1
        if self._num_turn_mismatches == 1 or self._num_turn_mismatches % 1000 == 0:
            logger.warning(
                "SFT: %d assistant turns so far whose own render is not a prefix of "
                "the full conversation; their loss stops where the two renders stop "
                "agreeing.",
                self._num_turn_mismatches,
            )

    def _record_prompt_mismatch(self) -> None:
        self._num_prompt_mismatches += 1
        if self._num_prompt_mismatches == 1 or self._num_prompt_mismatches % 1000 == 0:
            logger.warning(
                "SFT: %d assistant turns so far whose generation prompt the chat template "
                "renders differently from the same turn in the full conversation (e.g. "
                "'<think>\\n' vs '<think></think>'); their loss starts where the two "
                "renders stop agreeing.",
                self._num_prompt_mismatches,
            )

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

        if enable_thinking is None:
            # A row without reasoning renders with thinking OFF: under the thinking prompt
            # ("<think>\n") its answers would teach an empty think block whenever thinking is on.
            enable_thinking = any(
                _has_reasoning(m) for m in messages if m.get("role") == "assistant"
            )
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
        extra["enable_thinking"] = bool(enable_thinking)
        if not any(m.get("role") == "assistant" for m in messages):
            # Nothing to learn from: every token would be masked, so the sample would
            # contribute a zero-token row and skew packing.
            return None

        token_ids = self._tokenize(messages, extra)
        if len(token_ids) < 2:
            return None
        # The training sequence is one token shorter than the render (next-token shift).
        if (
            self._drop_if_longer_than is not None
            and len(token_ids) - 1 > self._drop_if_longer_than
        ):
            self._record_drop(len(token_ids) - 1)
            return None

        input_ids = np.asarray(token_ids, dtype=np.int64)
        trainable = np.zeros(len(input_ids), dtype=bool)
        for start, stop in self._assistant_spans(messages, input_ids, extra):
            trainable[start:stop] = True

        if not trainable.any():
            self._record_untrained()
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

    Differs from pretrain in three ways, each forced rather than chosen:

    1. **first_fit packing, not concat_then_split.** A conversation must not be split
       across rows -- half a dialogue is not a training example. That means padding, and
       therefore `num_packing_bins` well above upstream's default of 8 (measured: 8 bins
       7.7% padding, 32 bins 2.4%).
    2. **Each dataset may carry its own `row_adapter`.** Corpora disagree about how a
       conversation is stored; `DatasetSpec.row_adapter` selects the layout per dataset
       and everything after it is shared. See the adapters above.
    3. **Oversized conversations are dropped, and logged.** The threshold MUST be
       the model's `max_context_length`, not `num_tokens_per_microbatch_per_dp_rank`:
       `FirstFitPackingConfig.build` splits every sequence longer than
       `context.max_context_length` into pieces (`_SplitTextSequenceDocuments`), even
       when it would fit the wider row, so only this drop keeps a conversation whole.

    The padding first_fit adds is marked in upstream's padding mask, which keeps it
    out of the MoE load-balance statistics and ``tokens_per_expert_E``.
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
