# [MODULE DOC]
# Dropout module applied to the sentence embedding.
from __future__ import annotations

from torch import Tensor, nn

from sentence_transformers.base.modules.module import Module


class Dropout(Module):
    # [CLASS DOC]
    # Module that applies dropout with probability dropout (config key) to "sentence_embedding".
    """Dropout layer.

    Args:
        dropout: Sets a dropout value for dense layer.
    """

    config_keys: list[str] = ["dropout"]

    def __init__(self, dropout: float = 0.2):
        # [FUNCTION DOC]
        # Stores the dropout probability (default 0.2) and creates the torch dropout layer.
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, features: dict[str, Tensor]):
        # [FUNCTION DOC]
        # Replaces "sentence_embedding" in the features dict with its dropped-out version (active only in
        # training mode) and returns the mutated dict.
        features.update({"sentence_embedding": self.dropout_layer(features["sentence_embedding"])})
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        # [FUNCTION DOC]
        # Saves only the config JSON to output_path.
        self.save_config(output_path)
