# [MODULE DOC]
# WordWeights module that scales each token embedding by a per-word weight (for example idf).
from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from sentence_transformers.base.modules.module import Module

logger = logging.getLogger(__name__)


class WordWeights(Module):
    # [CLASS DOC]
    # Module storing a weight per vocabulary index in a one-dimensional embedding layer. Config keys:
    # vocab, word_weights, unknown_word_weight.
    """This model can weight word embeddings, for example, with idf-values."""

    config_keys: list[str] = ["vocab", "word_weights", "unknown_word_weight"]

    def __init__(self, vocab: list[str], word_weights: dict[str, float], unknown_word_weight: float = 1):
        # [FUNCTION DOC]
        # Builds the per-index weights: the weight for the exact word, else for its lowercase form, else
        # unknown_word_weight (counted). Logs at info level the number of unknown words and loads the
        # weights into an embedding layer of size vocab length by 1.
        """
        Initializes the WordWeights class.

        Args:
            vocab (List[str]): Vocabulary of the tokenizer.
            word_weights (Dict[str, float]): Mapping of tokens to a float weight value. Word embeddings are multiplied
                by this float value. Tokens in word_weights must not be equal to the vocab (can contain more or less values).
            unknown_word_weight (float, optional): Weight for words in vocab that do not appear in the word_weights lookup.
                These can be, for example, rare words in the vocab where no weight exists. Defaults to 1.
        """
        super().__init__()
        self.vocab = vocab
        self.word_weights = word_weights
        self.unknown_word_weight = unknown_word_weight

        weights = []
        num_unknown_words = 0
        for word in vocab:
            weight = unknown_word_weight
            if word in word_weights:
                weight = word_weights[word]
            elif word.lower() in word_weights:
                weight = word_weights[word.lower()]
            else:
                num_unknown_words += 1
            weights.append(weight)

        logger.info(
            f"{num_unknown_words} of {len(vocab)} words without a weighting value. Set weight to {unknown_word_weight}"
        )

        self.emb_layer = nn.Embedding(len(vocab), 1)
        self.emb_layer.load_state_dict({"weight": torch.FloatTensor(weights).unsqueeze(1)})

    def forward(self, features: dict[str, Tensor]):
        # [FUNCTION DOC]
        # Looks up the weight of each "input_ids" token, zeroes padding positions using "attention_mask",
        # multiplies "token_embeddings" by the weights, and stores the weighted "token_embeddings" and
        # the per-text weight sum "token_weights_sum" (used by Pooling) in the mutated features dict,
        # which it returns.
        attention_mask = features["attention_mask"]
        token_embeddings = features["token_embeddings"]

        # Compute a weight value for each token
        token_weights_raw = self.emb_layer(features["input_ids"]).squeeze(-1)
        token_weights = token_weights_raw * attention_mask.float()
        token_weights_sum = torch.sum(token_weights, 1)

        # Multiply embedding by token weight value
        token_weights_expanded = token_weights.unsqueeze(-1).expand(token_embeddings.size())
        token_embeddings = token_embeddings * token_weights_expanded

        features.update({"token_embeddings": token_embeddings, "token_weights_sum": token_weights_sum})
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        # [FUNCTION DOC]
        # Saves the config JSON and the torch weights to output_path.
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)
