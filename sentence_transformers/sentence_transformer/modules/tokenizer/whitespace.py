# [MODULE DOC]
# Fast whitespace tokenizer that splits text on whitespace, strips punctuation, removes stop words
# and maps tokens to vocabulary indices.
from __future__ import annotations

import collections
import json
import os
import string
from collections.abc import Iterable

from .word import ENGLISH_STOP_WORDS, WordTokenizer


class WhitespaceTokenizer(WordTokenizer):
    # [CLASS DOC]
    # WordTokenizer subclass with a vocab, a stop-word set and an optional lowercasing flag.
    """
    Simple and fast white-space tokenizer. Splits sentence based on white spaces.
    Punctuation are stripped from tokens.
    """

    def __init__(
        self, vocab: Iterable[str] = [], stop_words: Iterable[str] = ENGLISH_STOP_WORDS, do_lower_case: bool = False
    ):
        # [FUNCTION DOC]
        # Stores the stop words as a set and the lowercase flag, then sets the vocabulary.
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def get_vocab(self):
        # [FUNCTION DOC]
        # Returns the stored vocabulary.
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]):
        # [FUNCTION DOC]
        # Stores the vocab and builds an ordered word-to-index map (a later duplicate overwrites the index).
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def tokenize(self, text: str, **kwargs) -> list[int]:
        # [FUNCTION DOC]
        # Tokenizes text into vocab indices. Lowercases first if do_lower_case. Splits on whitespace; for
        # each token drops it if a stop word, keeps its index if in vocab, otherwise retries with
        # surrounding punctuation stripped, then that stripped form lowercased, applying the same checks.
        # Unknown tokens are dropped. Returns the list of indices.
        if self.do_lower_case:
            text = text.lower()

        tokens = text.split()

        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.strip(string.punctuation)
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

        return tokens_filtered

    def save(self, output_path: str):
        # [FUNCTION DOC]
        # Writes whitespacetokenizer_config.json in output_path with vocab, stop words and lowercase flag.
        with open(os.path.join(output_path, "whitespacetokenizer_config.json"), "w") as fOut:
            json.dump(
                {
                    "vocab": list(self.word2idx.keys()),
                    "stop_words": list(self.stop_words),
                    "do_lower_case": self.do_lower_case,
                },
                fOut,
            )

    @staticmethod
    def load(input_path: str):
        # [FUNCTION DOC]
        # Static method that reads whitespacetokenizer_config.json from input_path and returns a new
        # WhitespaceTokenizer built from it.
        with open(os.path.join(input_path, "whitespacetokenizer_config.json")) as fIn:
            config = json.load(fIn)

        return WhitespaceTokenizer(**config)
