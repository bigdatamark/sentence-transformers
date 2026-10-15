# [MODULE DOC]
# LSTM module that runs a (by default bidirectional) LSTM over token embeddings and replaces them
# with the LSTM hidden states.
from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from torch import nn

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util.decorators import deprecated_kwargs


class LSTM(Module):
    # [CLASS DOC]
    # Module wrapping a batch-first torch LSTM. Config keys (stored in lstm_config.json):
    # embedding_dimension, hidden_dim, num_layers, dropout, bidirectional; the old key
    # word_embedding_dimension is renamed. Output size is hidden_dim, doubled when bidirectional.
    """Bidirectional LSTM running over word embeddings."""

    config_keys: list[str] = ["embedding_dimension", "hidden_dim", "num_layers", "dropout", "bidirectional"]
    config_file_name: str = "lstm_config.json"
    config_key_renames = {"word_embedding_dimension": "embedding_dimension"}

    @deprecated_kwargs(**config_key_renames)
    def __init__(
        self,
        embedding_dimension: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = True,
    ):
        # [FUNCTION DOC]
        # Stores the config, computes the output dimension and creates the LSTM. Deprecated keyword names
        # are accepted through the deprecated_kwargs decorator.
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embeddings_dimension = hidden_dim
        if self.bidirectional:
            self.embeddings_dimension *= 2

        self.encoder = nn.LSTM(
            embedding_dimension,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Reads "token_embeddings" and "sentence_lengths" (lengths clamped to at least 1), packs the
        # padded sequences (unsorted allowed, lengths moved to CPU), runs the LSTM, unpacks back to a
        # padded batch-first tensor and stores it as "token_embeddings" in the mutated features dict.
        # Note the unpacked length is the longest sentence length in the batch, which can be shorter than
        # the input padding.
        token_embeddings = features["token_embeddings"]
        sentence_lengths = torch.clamp(features["sentence_lengths"], min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            token_embeddings, sentence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed = self.encoder(packed)
        unpack = nn.utils.rnn.pad_packed_sequence(packed[0], batch_first=True)[0]
        features.update({"token_embeddings": unpack})
        return features

    def get_embedding_dimension(self) -> int:
        # [FUNCTION DOC]
        # Returns the output dimension (hidden_dim, doubled when bidirectional).
        return self.embeddings_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        # [FUNCTION DOC]
        # Saves the config JSON, then temporarily moves the module to CPU (needed for safetensors), saves
        # the weights, and moves it back to its original device.
        self.save_config(output_path)

        # Saving LSTM models with Safetensors does not work unless the weights are on CPU
        # See https://github.com/huggingface/sentence-transformers/pull/2722
        device = next(self.parameters()).device
        self.cpu()
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)
        self.to(device)

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
        # Class method that loads config and weights from a local path or Hub repo, constructs the module
        # and returns it with weights loaded.
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model
