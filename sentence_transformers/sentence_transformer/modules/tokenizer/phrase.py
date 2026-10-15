# [MODULE DOC]
# Phrase-aware word tokenizer that uses NLTK word tokenization and merges consecutive words into
# a single token when the joined n-gram (e.g. New_York) exists in the vocabulary.
from __future__ import annotations

import collections
import json
import logging
import os
import string
from collections.abc import Iterable

from transformers.utils.import_utils import NLTK_IMPORT_ERROR, is_nltk_available

from .word import ENGLISH_STOP_WORDS, WordTokenizer

logger = logging.getLogger(__name__)


class PhraseTokenizer(WordTokenizer):
    # [CLASS DOC]
    # WordTokenizer subclass that detects vocabulary phrases joined by ngram_separator (default
    # underscore) up to max_ngram_length words, removes stop words and maps tokens to vocab indices.
    # Requires NLTK.
    """Tokenizes the text with respect to existent phrases in the vocab.

    This tokenizers respects phrases that are in the vocab. Phrases are separated with 'ngram_separator', for example,
    in Google News word2vec file, ngrams are separated with a _ like New_York. These phrases are detected in text and merged as one special token. (New York is the ... => [New_York, is, the])
    """

    def __init__(
        self,
        vocab: Iterable[str] = [],
        stop_words: Iterable[str] = ENGLISH_STOP_WORDS,
        do_lower_case: bool = False,
        ngram_separator: str = "_",
        max_ngram_length: int = 5,
    ):
        # [FUNCTION DOC]
        # Raises ImportError if NLTK is not installed. Stores the stop words as a set, the lowercase flag,
        # separator and maximum n-gram length, then sets the vocabulary.
        # KNOWN ISSUE: do_lower_case is stored and saved but never used by tokenize, which always tries
        # the original and then the lowercased form.
        if not is_nltk_available():
            raise ImportError(NLTK_IMPORT_ERROR.format(self.__class__.__name__))

        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.ngram_separator = ngram_separator
        self.max_ngram_length = max_ngram_length
        self.set_vocab(vocab)

    def get_vocab(self):
        # [FUNCTION DOC]
        # Returns the stored vocabulary.
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]):
        # [FUNCTION DOC]
        # Stores the vocab, builds an ordered word-to-index map (a later duplicate overwrites the index),
        # and collects phrase entries: words containing the separator, without a doubled separator, and
        # with at most max_ngram_length parts, recording each phrase's word count. Logs at info level the
        # phrase lengths and number of phrases when the vocab is non-empty.
        # KNOWN ISSUE: vocab is iterated twice, so a one-shot iterator would leave the phrase lookup empty.
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

        # Check for ngram in vocab
        self.ngram_lookup = set()
        self.ngram_lengths = set()
        for word in vocab:
            if self.ngram_separator is not None and self.ngram_separator in word:
                # Sum words might me malformed in e.g. google news word2vec, containing two or more _ after each other
                ngram_count = word.count(self.ngram_separator) + 1
                if self.ngram_separator + self.ngram_separator not in word and ngram_count <= self.max_ngram_length:
                    self.ngram_lookup.add(word)
                    self.ngram_lengths.add(ngram_count)

        if len(vocab) > 0:
            logger.info(f"PhraseTokenizer - Phrase ngram lengths: {self.ngram_lengths}")
            logger.info(f"PhraseTokenizer - Num phrases: {len(self.ngram_lookup)}")

    def tokenize(self, text: str, **kwargs) -> list[int]:
        # [FUNCTION DOC]
        # Tokenizes text into vocab indices. Splits with NLTK word_tokenize (preserving lines). For each
        # phrase length from longest to shortest, slides over the tokens and replaces any window whose
        # separator-joined form (or its lowercase) is a known phrase with that single phrase token. Then for
        # each token: drops it if it is a stop word; keeps its index if in vocab; otherwise retries with
        # the lowercased form, then the lowercased form with surrounding punctuation stripped, applying the
        # same stop-word and vocab checks. Tokens not found are dropped. Returns the list of indices.
        from nltk import word_tokenize

        tokens = word_tokenize(text, preserve_line=True)

        # phrase detection
        for ngram_len in sorted(self.ngram_lengths, reverse=True):
            idx = 0
            while idx <= len(tokens) - ngram_len:
                ngram = self.ngram_separator.join(tokens[idx : idx + ngram_len])
                if ngram in self.ngram_lookup:
                    tokens[idx : idx + ngram_len] = [ngram]
                elif ngram.lower() in self.ngram_lookup:
                    tokens[idx : idx + ngram_len] = [ngram.lower()]
                idx += 1

        # Map tokens to idx, filter stop words
        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
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

        return tokens_filtered

    def save(self, output_path: str):
        # [FUNCTION DOC]
        # Writes phrasetokenizer_config.json in output_path with the vocab, stop words, lowercase flag,
        # separator and maximum n-gram length.
        with open(os.path.join(output_path, "phrasetokenizer_config.json"), "w") as fOut:
            json.dump(
                {
                    "vocab": list(self.word2idx.keys()),
                    "stop_words": list(self.stop_words),
                    "do_lower_case": self.do_lower_case,
                    "ngram_separator": self.ngram_separator,
                    "max_ngram_length": self.max_ngram_length,
                },
                fOut,
            )

    @staticmethod
    def load(input_path: str):
        # [FUNCTION DOC]
        # Static method that reads phrasetokenizer_config.json from input_path and returns a new
        # PhraseTokenizer built from it.
        with open(os.path.join(input_path, "phrasetokenizer_config.json")) as fIn:
            config = json.load(fIn)

        return PhraseTokenizer(**config)
