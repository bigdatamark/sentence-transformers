# [MODULE DOC]
# WordEmbeddings input module: looks up static word vectors (e.g. GloVe, word2vec) for tokens
# produced by a word-level tokenizer, outputting padded token embeddings. Can be built from a
# plain-text or gzipped embeddings file, which is downloaded from the UKP server when missing.
from __future__ import annotations

import gzip
import logging
import os

from transformers import PreTrainedTokenizerBase

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from sentence_transformers.base.modules.input_module import InputModule
from sentence_transformers.util import fullname, http_get, import_module_class

from .tokenizer import TransformersTokenizerWrapper, WhitespaceTokenizer, WordTokenizer

logger = logging.getLogger(__name__)


class WordEmbeddings(InputModule):
    # [CLASS DOC]
    # Text-only input module with a word tokenizer and a torch embedding layer. Config keys
    # (wordembedding_config.json): tokenizer_class (dotted path of the tokenizer class),
    # update_embeddings and max_seq_length.
    config_keys: list[str] = ["tokenizer_class", "update_embeddings", "max_seq_length"]
    config_file_name: str = "wordembedding_config.json"
    modalities: list[str] = ["text"]

    def __init__(
        self,
        tokenizer: WordTokenizer | PreTrainedTokenizerBase,
        embedding_weights,
        update_embeddings: bool = False,
        max_seq_length: int = 1000000,
    ):
        # [FUNCTION DOC]
        # Initializes the module. Wraps a Hugging Face tokenizer in TransformersTokenizerWrapper; raises
        # ValueError if the tokenizer is neither that nor a WordTokenizer. Converts list or numpy weights to
        # a tensor, creates an embedding layer of the same shape and loads the weights into it, and makes
        # it trainable only if update_embeddings. Stores tokenizer, update_embeddings and max_seq_length.
        nn.Module.__init__(self)
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            tokenizer = TransformersTokenizerWrapper(tokenizer)
        elif not isinstance(tokenizer, WordTokenizer):
            raise ValueError("tokenizer must be a WordTokenizer or a HuggingFace tokenizer. ")
        if isinstance(embedding_weights, list):
            embedding_weights = np.asarray(embedding_weights)

        if isinstance(embedding_weights, np.ndarray):
            embedding_weights = torch.from_numpy(embedding_weights)

        num_embeddings, embeddings_dimension = embedding_weights.size()
        self.embeddings_dimension = embeddings_dimension
        self.emb_layer = nn.Embedding(num_embeddings, embeddings_dimension)
        self.emb_layer.load_state_dict({"weight": embedding_weights})
        self.emb_layer.weight.requires_grad = update_embeddings
        self.tokenizer = tokenizer
        self.update_embeddings = update_embeddings
        self.max_seq_length = max_seq_length

    def forward(self, features, **kwargs):
        # [FUNCTION DOC]
        # Looks up embeddings for "input_ids" and sets "token_embeddings", sets "cls_token_embeddings" to
        # None and keeps "attention_mask". Returns the mutated features dict.
        token_embeddings = self.emb_layer(features["input_ids"])
        cls_tokens = None
        features.update(
            {
                "token_embeddings": token_embeddings,
                "cls_token_embeddings": cls_tokens,
                "attention_mask": features["attention_mask"],
            }
        )
        return features

    def preprocess(self, inputs: list[str], prompt: str | None = None, **kwargs):
        # [FUNCTION DOC]
        # Tokenizes inputs (after prepending the prompt if given), right-pads all token id lists with 0 to
        # the longest length, and returns tensors "input_ids", "attention_mask" (1 for real tokens) and
        # "sentence_lengths".
        # KNOWN ISSUE: max_seq_length is never applied, so texts are not truncated; and an empty input list
        # makes the maximum length computation raise ValueError.
        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)
        tokenized_texts = [self.tokenizer.tokenize(text, **kwargs) for text in inputs]
        sentence_lengths = [len(tokens) for tokens in tokenized_texts]
        max_len = max(sentence_lengths)

        input_ids = []
        attention_masks = []
        for tokens in tokenized_texts:
            padding = [0] * (max_len - len(tokens))
            input_ids.append(tokens + padding)
            attention_masks.append([1] * len(tokens) + padding)

        output = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "sentence_lengths": torch.tensor(sentence_lengths, dtype=torch.long),
        }

        return output

    def get_embedding_dimension(self) -> int:
        # [FUNCTION DOC]
        # Returns the embedding dimension.
        return self.embeddings_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs):
        # [FUNCTION DOC]
        # Saves config JSON, torch weights and the tokenizer files to output_path.
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save(output_path)

    def get_config_dict(self):
        # [FUNCTION DOC]
        # Returns the config dict: the tokenizer's fully qualified class name, update_embeddings and
        # max_seq_length.
        return {
            "tokenizer_class": fullname(self.tokenizer),
            "update_embeddings": self.update_embeddings,
            "max_seq_length": self.max_seq_length,
        }

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Class method that loads the module. Loads the config, resolves the tokenizer class from its
        # dotted path through the trust_remote_code gate (import_module_class), loads the tokenizer from the
        # module directory, loads the weights and constructs the module with the "emb_layer.weight" tensor
        # and the remaining config.
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        # `tokenizer_class` is a config-controlled dotted path, so resolve it through the same trust gate as
        # the module `type` field
        tokenizer_class = import_module_class(
            config.pop("tokenizer_class"),
            model_name_or_path=model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", False),
            revision=revision,
            token=token,
            cache_folder=cache_folder,
            local_files_only=local_files_only,
        )
        tokenizer_local_path = cls.load_dir_path(model_name_or_path=model_name_or_path, **hub_kwargs)
        tokenizer = tokenizer_class.load(tokenizer_local_path)

        weights = cls.load_torch_weights(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(tokenizer=tokenizer, embedding_weights=weights["emb_layer.weight"], **config)
        return model

    @classmethod
    def from_text_file(
        cls,
        embeddings_file_path: str,
        update_embeddings: bool = False,
        item_separator: str = " ",
        tokenizer=WhitespaceTokenizer(),
        max_vocab_size: int | None = None,
    ):
        # [FUNCTION DOC]
        # Class method that builds the module from an embeddings text file (gzip if the name ends with .gz),
        # one word per line followed by its vector values separated by item_separator. If the file does
        # not exist and the name has no path separators, downloads it from the UKP embeddings server
        # (network and file write); a missing path with separators raises ValueError. Skips a word2vec
        # header line (two fields before any vocab), adds a zero PADDING_TOKEN vector at index 0, skips
        # lines whose dimension differs (logging an error), and stops after max_vocab_size words if set.
        # Sets the vocabulary on the tokenizer and returns a new module. Logs progress with tqdm.
        # KNOWN ISSUE: The default tokenizer is a single WhitespaceTokenizer instance created at definition
        # time and mutated by set_vocab, so it is shared between all calls that rely on the default.
        logger.info(f"Read in embeddings file {embeddings_file_path}")

        if not os.path.exists(embeddings_file_path):
            logger.info(f"{embeddings_file_path} does not exist, try to download from server")

            if "/" in embeddings_file_path or "\\" in embeddings_file_path:
                raise ValueError(f"Embeddings file not found: {embeddings_file_path}")

            url = "https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/" + embeddings_file_path
            http_get(url, embeddings_file_path)

        embeddings_dimension = None
        vocab = []
        embeddings = []

        with (
            gzip.open(embeddings_file_path, "rt", encoding="utf8")
            if embeddings_file_path.endswith(".gz")
            else open(embeddings_file_path, encoding="utf8") as fIn
        ):
            iterator = tqdm(fIn, desc="Load Word Embeddings", unit="Embeddings")
            for line in iterator:
                split = line.rstrip().split(item_separator)

                if not vocab and len(split) == 2:  # Handle Word2vec format
                    continue

                word = split[0]

                if embeddings_dimension is None:
                    embeddings_dimension = len(split) - 1
                    vocab.append("PADDING_TOKEN")
                    embeddings.append(np.zeros(embeddings_dimension))

                if (
                    len(split) - 1
                ) != embeddings_dimension:  # Assure that all lines in the embeddings file are of the same length
                    logger.error(
                        "ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token."
                    )
                    continue

                vector = np.array([float(num) for num in split[1:]])
                embeddings.append(vector)
                vocab.append(word)

                if max_vocab_size is not None and max_vocab_size > 0 and len(vocab) > max_vocab_size:
                    break

            embeddings = np.asarray(embeddings)

            tokenizer.set_vocab(vocab)
            return cls(tokenizer=tokenizer, embedding_weights=embeddings, update_embeddings=update_embeddings)
