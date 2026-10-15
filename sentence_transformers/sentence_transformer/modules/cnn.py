# [MODULE DOC]
# 1D convolution module that applies several parallel convolutions with different kernel sizes over
# token embeddings and concatenates their outputs.
from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from torch import nn

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util.decorators import deprecated_kwargs


class CNN(Module):
    # [CLASS DOC]
    # Module applying one Conv1d per kernel size to token embeddings. Output token embedding size is
    # out_channels times the number of kernels. Config keys: in_embedding_dimension, out_channels,
    # kernel_sizes (stored in cnn_config.json); the old key in_word_embedding_dimension is renamed.
    """CNN-layer with multiple kernel-sizes over the word embeddings"""

    config_keys: list[str] = ["in_embedding_dimension", "out_channels", "kernel_sizes"]
    config_file_name: str = "cnn_config.json"
    config_key_renames = {"in_word_embedding_dimension": "in_embedding_dimension"}

    @deprecated_kwargs(**config_key_renames)
    def __init__(
        self,
        in_embedding_dimension: int,
        out_channels: int = 256,
        kernel_sizes: list[int] = [1, 3, 5],
        stride_sizes: list[int] | None = None,
    ):
        # [FUNCTION DOC]
        # Creates the convolutions. Inputs: input embedding dimension, out_channels per conv (default 256),
        # kernel_sizes (default 1, 3, 5) and optional stride_sizes (default all 1). Each conv uses padding
        # of half of (kernel size minus 1), rounded down, so odd kernels keep sequence length.
        # Deprecated keyword names are accepted through the deprecated_kwargs decorator.
        # KNOWN ISSUE: stride_sizes is not in config_keys, so non-default strides are lost when saving
        # and reloading. Even kernel sizes (or strides above 1) change the sequence length, so the outputs of
        # different convs cannot be concatenated.
        nn.Module.__init__(self)
        self.in_embedding_dimension = in_embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embeddings_dimension = out_channels * len(kernel_sizes)
        self.convs = nn.ModuleList()

        in_channels = in_embedding_dimension
        if stride_sizes is None:
            stride_sizes = [1] * len(kernel_sizes)

        for kernel_size, stride in zip(kernel_sizes, stride_sizes):
            padding_size = int((kernel_size - 1) / 2)
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_size,
            )
            self.convs.append(conv)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Reads "token_embeddings" (batch, seq, dim), transposes to channel-first, applies each conv,
        # concatenates along channels, transposes back, and writes the result into the features dict
        # under "token_embeddings" (mutating it). Returns the dict.
        token_embeddings = features["token_embeddings"]

        token_embeddings = token_embeddings.transpose(1, -1)
        vectors = [conv(token_embeddings) for conv in self.convs]
        out = torch.cat(vectors, 1).transpose(1, -1)

        features.update({"token_embeddings": out})
        return features

    def get_embedding_dimension(self) -> int:
        # [FUNCTION DOC]
        # Returns the output embedding dimension (out_channels times number of kernels).
        return self.embeddings_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        # [FUNCTION DOC]
        # Saves the config JSON and the torch weights (safetensors when safe_serialization) to output_path.
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
        # Class method that loads the module from a local path or Hub repo: loads the config using the
        # hub options (subfolder, token, cache folder, revision, local_files_only), constructs the module,
        # loads its weights into it and returns it.
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
