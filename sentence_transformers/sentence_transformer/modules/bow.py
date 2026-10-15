# [MODULE DOC]
# Bag-of-Words input module that turns text into a vocabulary-sized (optionally tf-idf weighted)
# sentence embedding vector, without any neural network.
from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import Tensor

from sentence_transformers.base.modules.input_module import InputModule

from .tokenizer import WhitespaceTokenizer

logger = logging.getLogger(__name__)


class BoW(InputModule):
    # [CLASS DOC]
    # Input module producing a bag-of-words vector of length len(vocab) per text. Each vocabulary
    # index has a weight; a token occurrence adds (cumulative mode) or sets (non-cumulative mode) that
    # weight at its index. Config keys saved: vocab, word_weights, unknown_word_weight,
    # cumulative_term_frequency. Accepts only the text modality and saves its config in a subfolder.
    """Implements a Bag-of-Words (BoW) model to derive sentence embeddings.

    A weighting can be added to allow the generation of tf-idf vectors. The output vector has the size of the vocab.
    """

    save_in_root: bool = False
    config_keys: list[str] = ["vocab", "word_weights", "unknown_word_weight", "cumulative_term_frequency"]
    modalities: list[str] = ["text"]

    def __init__(
        self,
        vocab: list[str],
        word_weights: dict[str, float] = {},
        unknown_word_weight: float = 1,
        cumulative_term_frequency: bool = True,
    ):
        # [FUNCTION DOC]
        # Initializes the module. Deduplicates vocab preserving order and stores it plus the weighting
        # options. Builds a per-index weight list: weight from word_weights for the exact word, else for
        # its lowercased form, else unknown_word_weight (counting such unknown words). Logs at info level
        # how many words had no weight. Creates a WhitespaceTokenizer over the vocab with no stop words
        # and no lowercasing, and sets the embedding dimension to the vocab size.
        super().__init__()
        vocab = list(dict.fromkeys(vocab))  # Ensure vocab is unique
        self.vocab = vocab
        self.word_weights = word_weights
        self.unknown_word_weight = unknown_word_weight
        self.cumulative_term_frequency = cumulative_term_frequency

        # Maps wordIdx -> word weight
        self.weights = []
        num_unknown_words = 0
        for word in vocab:
            weight = unknown_word_weight
            if word in word_weights:
                weight = word_weights[word]
            elif word.lower() in word_weights:
                weight = word_weights[word.lower()]
            else:
                num_unknown_words += 1
            self.weights.append(weight)

        logger.info(
            f"{num_unknown_words} out of {len(vocab)} words without a weighting value. Set weight to {unknown_word_weight}"
        )

        self.tokenizer = WhitespaceTokenizer(vocab, stop_words=set(), do_lower_case=False)
        self.sentence_embedding_dimension = len(vocab)

    def forward(self, features: dict[str, Tensor]):
        # [FUNCTION DOC]
        # Identity forward pass: the embedding is already built during preprocessing, so the features
        # dict is returned unchanged.
        # Nothing to do, everything is done in get_sentence_features
        return features

    def preprocess(self, inputs: list[str], prompt: str | None = None, **kwargs) -> list[int]:
        # [FUNCTION DOC]
        # Tokenizes input texts (after prepending the prompt if one is given) with the whitespace
        # tokenizer, passing extra kwargs to it, and returns the result of get_sentence_features.
        # KNOWN ISSUE: The return annotation says a list of ints, but it actually returns a dict with a
        # "sentence_embedding" tensor.
        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)
        tokenized = [self.tokenizer.tokenize(text, **kwargs) for text in inputs]
        return self.get_sentence_features(tokenized)

    def get_embedding_dimension(self):
        # [FUNCTION DOC]
        # Returns the embedding dimension, i.e. the vocabulary size.
        return self.sentence_embedding_dimension

    def get_sentence_features(
        self, tokenized_texts: list[list[int]], pad_seq_length: int = 0
    ) -> dict[Literal["sentence_embedding"], torch.Tensor]:
        # [FUNCTION DOC]
        # Builds bag-of-words vectors. For each token-id list, creates a float32 zero vector of vocab
        # size and, for every token id, adds its weight (cumulative) or sets it to its weight
        # (non-cumulative). Stacks the vectors and returns a dict with key "sentence_embedding".
        # pad_seq_length is ignored.
        vectors = []

        for tokens in tokenized_texts:
            vector = torch.zeros(self.get_embedding_dimension(), dtype=torch.float32)
            for token in tokens:
                if self.cumulative_term_frequency:
                    vector[token] += self.weights[token]
                else:
                    vector[token] = self.weights[token]
            vectors.append(vector)

        return {"sentence_embedding": torch.stack(vectors)}

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        # [FUNCTION DOC]
        # Saves only the config JSON to output_path; there are no weights to save.
        self.save_config(output_path)
