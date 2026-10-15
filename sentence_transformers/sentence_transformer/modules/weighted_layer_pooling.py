# [MODULE DOC]
# Weighted layer pooling module: replaces token embeddings with a learned weighted average of the
# hidden states of several transformer layers.
from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from torch import Tensor, nn

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util.decorators import deprecated_kwargs


class WeightedLayerPooling(Module):
    # [CLASS DOC]
    # Module that averages per-layer token embeddings from layer_start onward using trainable layer
    # weights. Config keys: embedding_dimension, layer_start, num_hidden_layers (the old key
    # word_embedding_dimension is renamed). The weights are saved as torch weights.
    """Token embeddings are weighted mean of their different hidden layer representations"""

    config_keys: list[str] = ["embedding_dimension", "layer_start", "num_hidden_layers"]
    config_key_renames = {"word_embedding_dimension": "embedding_dimension"}

    @deprecated_kwargs(**config_key_renames)
    def __init__(self, embedding_dimension, num_hidden_layers: int = 12, layer_start: int = 4, layer_weights=None):
        # [FUNCTION DOC]
        # Stores the config. Uses the given layer_weights or creates a trainable parameter of ones, one
        # per layer from layer_start to num_hidden_layers inclusive (the embedding layer counts as layer 0).
        # Deprecated keyword names are accepted through the deprecated_kwargs decorator.
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float))
        )

    def forward(self, features: dict[str, Tensor]):
        # [FUNCTION DOC]
        # If "all_layer_embeddings" is missing, returns features unchanged. Otherwise stacks the layer
        # outputs, keeps layers from layer_start onward, multiplies each by its weight, sums them and
        # divides by the weight total. Stores the result as "token_embeddings" in the mutated features
        # dict and returns it.
        ft_all_layers = features.get("all_layer_embeddings")
        if ft_all_layers is None:
            return features

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start :, :, :, :]  # Start from 4th layers output

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({"token_embeddings": weighted_average})
        return features

    def get_embedding_dimension(self):
        # [FUNCTION DOC]
        # Returns the embedding dimension.
        return self.embedding_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        # [FUNCTION DOC]
        # Saves the config JSON and the torch weights to output_path.
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

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
