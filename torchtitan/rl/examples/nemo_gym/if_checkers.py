# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""IFEval-style instruction checkers for the NeMo Gym instruction_following rows.

The registry of ``verifiable_instructions`` (the commit NeMo Gym pins) with the overrides
NeMo Gym's instruction_following server applies: its fixed ``count:count_unique`` and
``count:counting_composition`` (the originals count punctuation as words; NeMo Gym issues
4613 / 5067; copied below under their Apache-2.0 license), and, stricter than NeMo Gym, a
``language:response_language`` that fails when no language can be detected (the original
counts that as followed, so "12345 !!!" passes "answer in Korean").
"""

import re

import langdetect
from verifiable_instructions import (
    instructions,
    instructions_registry,
    instructions_util,
)

# ---- copied from NVIDIA NeMo Gym resources_servers/instruction_following/count_unique.py (abde55c3) ----
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class CountUniqueChecker(instructions.CountUniqueChecker):
    """`count:count_unique` scored on words only, not on punctuation.

    The upstream checker in verifiable-instructions compares the raw `nltk.word_tokenize` tokens with
    their set, and punctuation marks are tokens, so two sentences that both end in a period repeat the
    `.` token and the response fails however well it follows the instruction. This subclass keeps the
    upstream instruction text and only replaces `check_following`, with the same rule as the fix in
    allenai/open-instruct#1718: a word is a token with at least one letter or digit. Case and contraction
    splitting are unchanged on purpose, so `Dogs` and `dogs` stay different words and `Don't` still
    tokenizes to `Do` and `n't`.
    """

    def check_following(self, value: str) -> bool:
        # Same tokenizer as upstream, minus tokens that are only punctuation.
        words = [
            token
            for token in instructions_util.nltk.word_tokenize(value)
            if any(char.isalnum() for char in token)
        ]
        return len(words) == len(set(words))


# ---- copied from NVIDIA NeMo Gym resources_servers/instruction_following/counting_composition.py (abde55c3; same
# NVIDIA copyright and Apache-2.0 license as above) ----


# The spaced divider the instruction asks for, on a line of its own. Inline asterisks (for example
# emphasis inside a paragraph) are not dividers. Spaces or tabs around the stars and CRLF are allowed.
_DIVIDER_LINE = re.compile(r"^[ \t]*\*[ \t]+\*[ \t]+\*[ \t]*\r?$", re.MULTILINE)


class CountingCompositionChecker(instructions.CountingCompositionChecker):
    """`count:counting_composition` scored the way its instruction is worded.

    The upstream checker in verifiable-instructions splits paragraphs on `***` although the instruction
    asks for `* * *`, and counts the punctuation that ends each sentence as a word, so a response that
    follows the instruction always fails. This subclass keeps the upstream instruction text and arguments
    and only replaces `check_following`, with the same rules as the fix in allenai/open-instruct#1718.
    """

    def check_following(self, value: str) -> bool:
        args = self.get_instruction_args()
        paragraphs = _DIVIDER_LINE.split(value)
        # A divider before the first or after the last paragraph is tolerated; an empty paragraph
        # between two dividers is not.
        if paragraphs and not paragraphs[0].strip():
            paragraphs = paragraphs[1:]
        if paragraphs and not paragraphs[-1].strip():
            paragraphs = paragraphs[:-1]
        if len(paragraphs) != 3:
            return False

        for paragraph in paragraphs:
            sentences = instructions_util.split_into_sentences(paragraph)
            if len(sentences) != args["n_sent"]:
                return False
            for sentence in sentences:
                # Same tokenizer as upstream, minus tokens that are only punctuation.
                words = [
                    token
                    for token in instructions_util.nltk.word_tokenize(sentence)
                    if any(char.isalnum() for char in token)
                ]
                if len(words) != args["n_words"]:
                    return False
        return True


class StrictResponseLanguageChecker(instructions.ResponseLanguageChecker):
    """`language:response_language`, failing when the language cannot be detected."""

    def check_following(self, value):
        try:
            return langdetect.detect(value) == self._language
        except langdetect.LangDetectException:
            return False


INSTRUCTION_DICT = {
    **instructions_registry.INSTRUCTION_DICT,
    "count:counting_composition": CountingCompositionChecker,
    "count:count_unique": CountUniqueChecker,
    "language:response_language": StrictResponseLanguageChecker,
}
