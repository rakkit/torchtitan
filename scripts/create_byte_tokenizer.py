# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


class ByteTokenizer(BaseTokenizer):
    """
    Tokenize and encode/decode text using a byte-level tokenizer.

    Args:
        bos_token (str): String to use for representing the
            beginning-of-sequence token.
        eos_token (str): String to use for representing the
            end-of-sequence token.
        special_tokens (list[str] | None): optional list of
            additional tokens to add
    """

    NUM_BYTE_VALUES = 256

    def __init__(
        self,
        bos_token: str = "〈BOS〉",
        eos_token: str = "〈EOS〉",
        special_tokens: list[str] | None = None,
    ):
        super().__init__()

        if special_tokens is None:
            special_tokens = []
        if bos_token not in special_tokens:
            special_tokens.insert(0, bos_token)
        if eos_token not in special_tokens:
            special_tokens.insert(1, eos_token)

        self.special_tokens = {
            tok: i + ByteTokenizer.NUM_BYTE_VALUES
            for (i, tok) in enumerate(special_tokens)
        }
        self.bos_id = self.special_tokens[bos_token]
        self.eos_id = self.special_tokens[eos_token]
        self.pad_id = -1

        self._vocab_size = ByteTokenizer.NUM_BYTE_VALUES + len(self.special_tokens)

        self._vocab = {chr(i): i for i in range(ByteTokenizer.NUM_BYTE_VALUES)}
        self._vocab.update(self.special_tokens)

        assert len(self._vocab) == self._vocab_size, (
            f"unexpected vocabulary size; make sure none of the specified "
            f"special tokens collide with the original "
            f"{ByteTokenizer.NUM_BYTE_VALUES} ASCII symbols"
        )

        self._inv_vocab = {v: k for (k, v) in self._vocab.items()}
        logger.info(
            f"ByteTokenizer built: #words {self.get_vocab_size()}, BOS ID {self.bos_id}, "
            f"EOS ID {self.eos_id}"
        )

    def encode(self, text: str, *, bos: bool, eos: bool) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            text (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            list[int]: A list of token IDs.
        """
        # This always byte-tokenizes even sequences that would result in
        # special tokens. This means it's impossible to obtain special
        # tokens from tokenization.
        tokens = list(text.encode(errors="replace"))
        # This version would encode arbitrary binary strings.
        # tokens = []
        # for token in text:
        #     if ord(token) > 255:
        #         # Handle multi-byte characters using UTF-8.
        #         # TODO Ideally we would do some math to split the
        #         #      chars arbitrarily, either only if UTF-8 doesn't
        #         #      work or always.
        #         tokens.append(token.encode(errors="replace"))
        #     else:
        #         tokens.append(ord(token))

        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (list[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # This is a bit awkward, but we try to prevent UTF-8 shenanigans.
        curr_bytestr = []
        text_parts = []
        for val in tokens:
            if val >= ByteTokenizer.NUM_BYTE_VALUES:
                if curr_bytestr:
                    text_part = bytes(curr_bytestr).decode(errors="replace")
                    # This version would decode arbitrary tokens.
                    # text_part = bytes(curr_bytestr).decode("latin-1", errors="replace")

                    text_parts.append(text_part)
                    curr_bytestr.clear()
                text_parts.append(self._inv_vocab[val])
            else:
                curr_bytestr.append(val)
        text_parts.append(bytes(curr_bytestr).decode(errors="replace"))
        # This version would decode arbitrary tokens.
        # text_parts.append(bytes(curr_bytestr).decode("latin-1", errors="replace"))
        return "".join(text_parts)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary as a dictionary."""
        return self._vocab

    def token_to_id(self, token: str) -> int | None:
        """Convert token to ID."""
        return self._vocab[token]

    def id_to_token(self, token_id: int) -> str | None:
        """Convert ID to token."""
        return self._inv_vocab[token_id]


def build_byte_tokenizer(job_config: JobConfig) -> ByteTokenizer:
    return ByteTokenizer()


if __name__ == "__main__":
    import json
    import os
    from argparse import ArgumentParser

    import tokenizers

    parser = ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help=(
            "Directory to save the HuggingFace-converted byte-level tokenizer in. "
            "Will create `tokenizer.json` and `tokenizer_config.json` in this directory."
        ),
        required=True,
    )
    args = parser.parse_args()

    hf_tok_save_dir = args.save_dir
    hf_tok_save_json = os.path.join(hf_tok_save_dir, "tokenizer.json")
    hf_tok_save_config_json = os.path.join(hf_tok_save_dir, "tokenizer_config.json")

    tt_tok = ByteTokenizer()

    hf_tok = tokenizers.Tokenizer(tokenizers.models.BPE())
    hf_tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=False,
    )
    hf_tok.decoder = tokenizers.decoders.ByteLevel(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=False,
    )
    hf_tok.post_processor = tokenizers.processors.ByteLevel(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=False,
    )

    # Split special tokens when encoding.
    hf_tok.encode_special_tokens = True

    hf_tok_trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=256,
        min_frequency=2,
        show_progress=False,
        # initial_alphabet=list(map(chr, range(256))),
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
    )
    hf_tok.train([], trainer=hf_tok_trainer)

    # Create a string that contains all valid UTF-8 bytes.
    utf_8_multibytes = (
        list(range(256))
        + [256 + 64 * i for i in range(28)]
        + [255]
        + [2048]
        + [4096 * i for i in range(1, 16)]
        + [65_536, 262_144, 524_288, 786_432, 1_048_576]
    )
    byte_alphabet = "".join(map(chr, utf_8_multibytes))
    # Total number of bytes: 256
    # Total number of illegal bytes in UTF-8: 13 (0xc0, 0xc1, 0xf5 to 0ff)
    assert len(set(byte_alphabet.encode())) == 256 - 13

    alphabet_tt = tt_tok.encode(byte_alphabet, bos=False, eos=False)
    alphabet_hf = hf_tok.encode(byte_alphabet).ids

    # Save the tokenizer to get access to its vocabulary.
    os.makedirs(hf_tok_save_dir, exist_ok=True)
    hf_tok.save(hf_tok_save_json, pretty=True)
    del hf_tok

    with open(hf_tok_save_json, "r") as f:
        hf_tok_conf = json.load(f)

    old_hf_tok_vocab = hf_tok_conf["model"]["vocab"]
    old_hf_tok_inv_vocab = {v: k for (k, v) in old_hf_tok_vocab.items()}

    new_hf_tok_inv_vocab = {}
    for (alphabet_tt_id, alphabet_hf_id) in zip(alphabet_tt, alphabet_hf):
        tok = old_hf_tok_inv_vocab[alphabet_hf_id]
        new_hf_tok_inv_vocab[tok] = alphabet_tt_id

    # Add illegal bytes into vocab, following the pattern in the patched
    # vocabulary.
    new_hf_tok_inv_vocab["\u00c0"] = 192
    new_hf_tok_inv_vocab["\u00c1"] = 193
    new_hf_tok_inv_vocab["\u00f5"] = 245
    new_hf_tok_inv_vocab["\u00f6"] = 246
    new_hf_tok_inv_vocab["\u00f7"] = 247
    new_hf_tok_inv_vocab["\u00f8"] = 248
    new_hf_tok_inv_vocab["\u00f9"] = 249
    new_hf_tok_inv_vocab["\u00fa"] = 250
    new_hf_tok_inv_vocab["\u00fb"] = 251
    new_hf_tok_inv_vocab["\u00fc"] = 252
    new_hf_tok_inv_vocab["\u00fd"] = 253
    new_hf_tok_inv_vocab["\u00fe"] = 254
    new_hf_tok_inv_vocab["\u00ff"] = 255
    new_hf_tok_inv_vocab = {
        k: v for (k, v) in sorted(new_hf_tok_inv_vocab.items(), key=lambda kv: kv[1])
    }

    hf_tok_conf["model"]["vocab"] = new_hf_tok_inv_vocab

    with open(hf_tok_save_json, "w") as f:
        json.dump(hf_tok_conf, f, indent=2)

    # Load tokenizer with adjusted vocabulary.
    hf_tok = tokenizers.Tokenizer.from_file(hf_tok_save_json)

    # Need to re-set this attribute after loading to split special
    # tokens when encoding.
    hf_tok.encode_special_tokens = True

    alphabet_hf = hf_tok.encode(byte_alphabet).ids
    assert alphabet_hf == alphabet_tt
