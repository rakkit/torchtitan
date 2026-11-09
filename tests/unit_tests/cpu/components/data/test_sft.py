# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-turn SFT masking: loss on every assistant turn, nothing else.

Upstream refuses multi-turn entirely and masks by prompt prefix length. The property
that matters here is which TOKENS are supervised, so the tests decode the supervised
positions back to text rather than comparing mask shapes.
"""

from __future__ import annotations

import grain.python as grain
import numpy as np
import pytest

from torchtitan.components.data.sft import MultiTurnChatProcessor
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX

# One character -> one token keeps the mapping between text and tokens exact, so a test
# failure points at the masking rather than at tokenizer quirks.
VOCAB_OFFSET = 100


class CharTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        return (
            [self.bos_id] * add_bos
            + [VOCAB_OFFSET + ord(c) for c in text]
            + [self.eos_id] * add_eos
        )

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        # "U:hi|A:yo|" -- one char per token, turn-delimited.
        text = "".join(f"{m['role'][0].upper()}:{m['content']}|" for m in messages)
        if add_generation_prompt:
            text += "A:"
        return text


CONTEXT = DatasetBuildContext(
    tokenizer=CharTokenizer(),
    max_context_length=256,
    num_tokens_per_microbatch=256,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


def _processor(**config_kwargs):
    return MultiTurnChatProcessor(
        MultiTurnChatProcessor.Config(**config_kwargs), context=CONTEXT
    )


def _supervised_text(sequence) -> str:
    """The characters the model is actually trained to produce."""
    return "".join(
        chr(int(t) - VOCAB_OFFSET)
        for t in sequence.labels
        if int(t) != IGNORE_INDEX and int(t) >= VOCAB_OFFSET
    )


def _conversation(*turns):
    roles = ("user", "assistant")
    return {
        "messages": [{"role": roles[i % 2], "content": t} for i, t in enumerate(turns)]
    }


def test_every_assistant_turn_is_supervised_not_just_the_last():
    """The core difference from upstream's single-turn prefix masking."""
    sample = _conversation("q1", "ANSWER1", "q2", "ANSWER2", "q3", "ANSWER3")

    sequence = _processor()(sample, np.random.default_rng(0))

    supervised = _supervised_text(sequence)
    for expected in ("ANSWER1", "ANSWER2", "ANSWER3"):
        assert expected in supervised, f"{expected} must contribute loss"


def test_user_turns_are_never_supervised():
    sample = _conversation("SECRETQUESTION", "reply")

    sequence = _processor()(sample, np.random.default_rng(0))

    supervised = _supervised_text(sequence)
    assert "SECRETQUESTION" not in supervised
    assert "reply" in supervised


def test_generation_prompt_header_is_masked():
    """The assistant header is prompt, not something to learn to emit.

    It is masked by DERIVATION, not by a configured token count: the prefix is rendered
    with `add_generation_prompt=True`, which ends exactly where the assistant content
    begins. An earlier version had a `generation_prompt_len` config field defaulting to
    0, so the default silently trained on the header.
    """
    sequence = _processor()(_conversation("q", "ANSWER"), np.random.default_rng(0))

    supervised = _supervised_text(sequence)
    # CharTokenizer renders the header as "A:". Supervision must start at the answer.
    assert supervised.startswith("ANSWER"), supervised
    assert not supervised.startswith(":"), "the header colon leaked into the loss"


def test_labels_are_next_token_aligned():
    """Same contract as TextProcessor: predict t+1 from t, so lengths match."""
    sample = _conversation("q", "abc")

    sequence = _processor()(sample, np.random.default_rng(0))

    assert len(sequence.input_ids) == len(sequence.labels)
    full = CONTEXT.tokenizer.apply_chat_template(sample["messages"])
    assert len(sequence.input_ids) == len(full) - 1


def test_conversation_without_an_assistant_turn_is_skipped():
    """Every token would be masked; emitting it would skew packing with a dead row."""
    sample = {"messages": [{"role": "user", "content": "hello"}]}
    assert _processor()(sample, np.random.default_rng(0)) is None


def test_empty_conversation_is_skipped():
    assert _processor()({"messages": []}, np.random.default_rng(0)) is None


def test_oversized_conversations_are_dropped_and_counted():
    """Conversations cannot be truncated without corrupting turn structure, so the loss
    is made visible rather than silent."""
    processor = _processor(drop_if_longer_than=8)
    long_sample = _conversation("q" * 50, "a" * 50)

    assert processor(long_sample, np.random.default_rng(0)) is None
    assert processor.num_dropped == 1

    short = _conversation("q", "a")
    assert processor(short, np.random.default_rng(0)) is not None
    assert processor.num_dropped == 1, "a kept sample must not increment the counter"


def test_the_length_limit_is_on_the_training_sequence():
    """The training sequence is the render minus one token (next-token shift), so a
    render of max + 1 tokens fits a max-token context and must be kept."""
    sample = _conversation("q", "a")  # "U:q|A:a|": 8 tokens, a 7-token sequence
    assert (
        _processor(drop_if_longer_than=7)(sample, np.random.default_rng(0)) is not None
    )
    assert _processor(drop_if_longer_than=6)(sample, np.random.default_rng(0)) is None


def test_a_generation_prompt_unlike_the_history_turn_starts_the_loss_where_they_differ():
    """Qwen3-style thinking templates end the generation prompt with an open block
    ('<think>\\n') but render a history turn without reasoning as a closed one
    ('<think></think>'). Starting the span at the generation prompt's LENGTH masks the
    closing tag and starts one token late; the span must start where the two renders
    stop agreeing, so the model learns to close the block."""

    class OpenThinkTemplate(CharTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            text = "".join(
                f"A:{{}}{m['content']}|"
                if m["role"] == "assistant"
                else f"U:{m['content']}|"
                for m in messages
            )
            # "{~" as '<think>\n': open, and not a prefix of the history's "{}".
            return text + ("A:{~" if add_generation_prompt else "")

    processor = MultiTurnChatProcessor(
        MultiTurnChatProcessor.Config(),
        context=DatasetBuildContext(
            tokenizer=OpenThinkTemplate(),
            max_context_length=256,
            num_tokens_per_microbatch=256,
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        ),
    )
    sequence = processor(
        _conversation("q1", "ONE", "q2", "TWO"), np.random.default_rng(0)
    )

    assert _supervised_text(sequence) == "}ONE|}TWO|"
    assert processor.num_prompt_mismatches == 2
    assert processor.num_turn_mismatches == 0


def test_a_template_that_rewrites_earlier_messages_is_refused():
    """A template that renders a message differently depending on where the
    conversation ends (e.g. the user's low_effort suffix on the last user message,
    or reasoning truncated in turns before it) makes a turn's own render and the
    full conversation diverge before that turn's content: its span would be empty
    and the turn silently left out of the loss. The processor raises instead."""

    class LastUserSuffixTemplate(CharTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
            text = "".join(
                f"{m['role'][0].upper()}:{m['content']}{'!' * (i == last_user)}|"
                for i, m in enumerate(messages)
            )
            return text + ("A:" if add_generation_prompt else "")

    processor = MultiTurnChatProcessor(
        MultiTurnChatProcessor.Config(),
        context=DatasetBuildContext(
            tokenizer=LastUserSuffixTemplate(),
            max_context_length=256,
            num_tokens_per_microbatch=256,
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        ),
    )
    with pytest.raises(ValueError, match="no tokens to train on"):
        processor(_conversation("q1", "ONE", "q2", "TWO"), np.random.default_rng(0))


def test_output_feeds_upstream_packing_and_collation():
    """The reason experiments/sft cannot run on 0.5.0: it never produced a TextSequence."""
    from torchtitan.components.data.collators import TextCollator

    sequences = [
        _processor()(_conversation("q", "answer"), np.random.default_rng(0))
        for _ in range(3)
    ]
    collator = TextCollator.Config().build(context=CONTEXT)
    # Upstream #4572 merged labels into the batch dict; the collator no longer
    # returns a (batch, labels) tuple.
    batch = collator(sequences)
    labels = batch.labels

    assert batch.input.numel() == CONTEXT.num_tokens_per_microbatch
    n_loss = int(
        batch.loss_token_counts.sum()
    )  # upstream's field (was num_valid_tokens)
    assert n_loss > 0
    assert (labels != IGNORE_INDEX).sum() == n_loss


# ---------------------------------------------- real production corpus schema


def test_messages_may_be_a_json_string():
    """The production corpus (sft-v2-64k) stores `messages` as a JSON STRING.

    Parquet cannot express a ragged struct list cheaply, so the column is `str`. Assuming
    a Python list fails deep inside the template render with an unhelpful TypeError.
    """
    import json

    turns = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ANSWER"},
    ]
    as_string = {"messages": json.dumps(turns)}
    as_list = {"messages": turns}

    from_string = _processor()(as_string, np.random.default_rng(0))
    from_list = _processor()(as_list, np.random.default_rng(0))

    assert _supervised_text(from_string) == _supervised_text(from_list)
    assert "ANSWER" in _supervised_text(from_string)


def test_empty_json_columns_are_tolerated():
    """`tools` is "[]" for most rows in the production corpus."""
    import json

    sample = {
        "messages": json.dumps(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "r"},
            ]
        ),
        "tools": "[]",
        "enable_thinking": True,
    }
    assert _processor()(sample, np.random.default_rng(0)) is not None


def test_a_row_without_reasoning_renders_with_thinking_off():
    """Under the thinking prompt ("<think>\\n") an answer without reasoning would train an
    empty think block, teaching the model to skip thinking while thinking is on."""
    seen: list[dict] = []

    class RecordingTokenizer(CharTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            seen.append(dict(kwargs))
            return super().apply_chat_template(messages, add_generation_prompt)

    processor = MultiTurnChatProcessor(
        MultiTurnChatProcessor.Config(),
        context=DatasetBuildContext(
            tokenizer=RecordingTokenizer(),
            max_context_length=256,
            num_tokens_per_microbatch=256,
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        ),
    )
    assert processor(_conversation("q", "a"), np.random.default_rng(0)) is not None
    assert seen and all(kw.get("enable_thinking") is False for kw in seen)
    seen.clear()
    with_reasoning = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning_content": "r"},
        ]
    }
    processor(with_reasoning, np.random.default_rng(0))
    assert seen and all(kw.get("enable_thinking") is True for kw in seen)


def test_turns_that_do_not_match_the_thinking_flag_are_not_trained():
    row = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "FIRST", "reasoning_content": "r"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "SECOND"},
        ]
    }
    supervised = _supervised_text(_processor()(row, np.random.default_rng(0)))
    # the row has reasoning -> thinking on: the turn without reasoning stays context only
    assert "FIRST" in supervised and "SECOND" not in supervised
    # an explicit flag that no turn matches leaves nothing to train: the row is skipped
    row_off = dict(
        row, enable_thinking="on", messages=[row["messages"][0], row["messages"][3]]
    )
    assert _processor()(row_off, np.random.default_rng(0)) is None


def test_an_empty_think_block_is_no_reasoning():
    """Non-thinking rows written as "<think>\\n\\n</think>\\n\\nA" (Qwen style) render with
    thinking off; under the thinking prompt they would train an empty think block."""
    from torchtitan.components.data.sft import _has_reasoning

    assert not _has_reasoning({"content": "<think>\n\n</think>\n\nA"})
    assert not _has_reasoning({"content": "  <think></think>A"})
    assert _has_reasoning({"content": "<think>plan</think>A"})
    assert _has_reasoning({"content": "<think>cut off inside the thinking"})
    assert not _has_reasoning({"content": "A <think>x</think>"})
    processor = _processor()
    row = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\nANSWER"},
        ]
    }
    # thinking off (no reasoning): the turn trains
    assert "ANSWER" in _supervised_text(processor(row, np.random.default_rng(0)))
    assert processor.num_turns_off_flag == 0


def test_untrained_turns_and_rows_are_counted():
    processor = _processor()
    row = {
        "enable_thinking": "on",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "NO REASONING"},
        ],
    }
    assert processor(row, np.random.default_rng(0)) is None
    assert processor.num_turns_off_flag == 1
    assert processor.num_untrained == 1


def test_template_kwargs_are_applied_to_every_render():
    """`truncate_history_thinking=False` must reach the PREFIX renders too.

    If prefix renders differ from the full-conversation render, the per-turn spans are
    computed against different text and the loss mask lands in the wrong place --
    silently, because the ids still tokenize fine.
    """
    seen: list[dict] = []

    class RecordingTokenizer(CharTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            seen.append(dict(kwargs))
            return super().apply_chat_template(messages, add_generation_prompt)

    processor = MultiTurnChatProcessor(
        MultiTurnChatProcessor.Config(
            chat_template_kwargs={"truncate_history_thinking": False}
        ),
        context=DatasetBuildContext(
            tokenizer=RecordingTokenizer(),
            max_context_length=256,
            num_tokens_per_microbatch=256,
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        ),
    )
    processor(_conversation("q1", "a1", "q2", "a2"), np.random.default_rng(0))

    assert len(seen) >= 3, "expected the full render plus prefix renders"
    assert all(
        kw.get("truncate_history_thinking") is False for kw in seen
    ), "every render, including prefixes, must carry the template kwargs"


def test_a_template_that_ignores_add_generation_prompt_still_masks_correctly():
    """The header derivation degrades safely.

    `BaseTokenizer.apply_chat_template` forwards **kwargs into Jinja, where an unused
    variable is simply undefined, so a template with no generation prompt ignores the
    flag rather than failing. The answer must still be supervised and the user turn must
    still be masked; the only loss is that a header, if any, stays in the span -- which
    is what a hand-set length of 0 did anyway.
    """

    class NoGenerationPromptTokenizer(CharTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            del add_generation_prompt, kwargs
            return "".join(f"{m['role'][0].upper()}:{m['content']}|" for m in messages)

    processor = MultiTurnChatProcessor(
        MultiTurnChatProcessor.Config(),
        context=DatasetBuildContext(
            tokenizer=NoGenerationPromptTokenizer(),
            max_context_length=256,
            num_tokens_per_microbatch=256,
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        ),
    )
    result = processor(_conversation("SECRET", "ANSWER"), np.random.default_rng(0))

    supervised = _supervised_text(result)
    assert "ANSWER" in supervised
    assert "SECRET" not in supervised


def test_a_rejected_conversation_does_not_kill_the_run(tmp_path):
    """The processor returns None to REJECT a sample, on five separate paths.

    Grain's `map` passes None straight through, so without a post-filter the first
    rejected conversation reaches the packer as `None.input_ids` and the job dies. The
    `drop_if_longer_than` feature makes this certain rather than unlikely: the SFT
    factory always sets it, and the drop log tells the operator to raise
    max_context_length -- advice that presumes the run survived the drop.
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from torchtitan.components.data.keyed_loader import KeyedMixDataLoader
    from torchtitan.components.data.mix import DatasetSpec
    from torchtitan.components.data.parquet_manifest import (
        build_manifest,
        MANIFEST_FILENAME,
        write_manifest,
    )
    from torchtitan.components.data.sft import make_sft_dataloader_config

    good = json.dumps(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}]
    )
    # Rejected: no assistant turn at all -> processor returns None.
    bad = json.dumps([{"role": "user", "content": "only a question"}])
    rows = ([good] * 4 + [bad]) * 12

    root = tmp_path / "sft"
    root.mkdir()
    pq.write_table(pa.table({"messages": rows}), root / "p.parquet")
    write_manifest(build_manifest(root), root / MANIFEST_FILENAME)

    config = make_sft_dataloader_config(
        [DatasetSpec(alias="sft", path=str(root))],
        max_context_length=64,
        num_packing_bins=2,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        num_prefetch_microbatches=1,
        repeat=True,
    )
    loader = KeyedMixDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=CharTokenizer(),
        max_context_length=64,
        num_tokens_per_microbatch=64,
    )
    try:
        iterator = iter(loader)
        for _ in range(5):
            batch = next(iterator)  # #4572: one dict, labels included
            assert batch.input.numel() == 64
    finally:
        loader.close()


def test_sft_does_not_inherit_the_window_shuffle(tmp_path):
    """`GrainDataLoader.Config.shuffle` defaults to True; SFT must not take it.

    True inserts WindowShuffleIterDataset, which renames the state key `parent` to
    `parent_window_start_state`. That broke the cursor walk and silently zeroed every
    `data_docs/{alias}` series on this path.
    """
    from torchtitan.components.data.mix import DatasetSpec
    from torchtitan.components.data.sft import make_sft_dataloader_config

    config = make_sft_dataloader_config(
        [DatasetSpec(alias="sft", path="/nonexistent")],
        max_context_length=64,
    )
    assert config.shuffle is False


def test_row_adapters_cover_the_layouts_the_old_loader_had_builders_for():
    """One adapter per corpus LAYOUT was the one thing the 937-line loader got right."""
    import json

    from torchtitan.components.data.sft import (
        conversation_rows,
        nested_tools_rows,
        normalize_tool_calls,
        prompt_response_rows,
    )

    # 1. The common layout, with JSON-string columns (how the corpus stores them)
    #    and an "on"/"off" thinking flag.
    msgs, tools, thinking = conversation_rows()(
        {
            "messages": json.dumps([{"role": "user", "content": "q"}]),
            "tools": json.dumps([{"name": "t"}]),
            "enable_thinking": "on",
        }
    )
    assert msgs == [{"role": "user", "content": "q"}]
    assert tools == [{"name": "t"}]
    assert thinking is True

    # 2. A column PAIR becomes a two-turn conversation.
    msgs, tools, thinking = prompt_response_rows("question", "answer")(
        {"question": "2+2?", "answer": "4"}
    )
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "4"
    assert tools is None and thinking is None

    # 3. Tools nested in the first message, and None content that must become "".
    msgs, tools, _ = nested_tools_rows()(
        {
            "messages": [
                {"role": "user", "content": None, "functions": json.dumps([{"n": 1}])},
                {"role": "assistant", "content": "hi"},
            ]
        }
    )
    assert msgs[0]["content"] == "", "None content renders as the string 'None'"
    assert tools == [{"n": 1}]

    # 4. tool_call arguments must reach the template as a dict whatever they arrived as.
    for raw, want in (
        ('{"a": 1}', {"a": 1}),
        ({"a": 1}, {"a": 1}),
        ([1], {}),
        ("nope", {}),
    ):
        out = normalize_tool_calls(
            [{"role": "assistant", "tool_calls": [{"function": {"arguments": raw}}]}]
        )
        assert out[0]["tool_calls"][0]["function"]["arguments"] == want, raw
