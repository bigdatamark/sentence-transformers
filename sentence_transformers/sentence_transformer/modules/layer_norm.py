# [MODULE DOC]
# Layer normalization module applied to the sentence embedding.
from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from torch import Tensor, nn

from sentence_transformers.base.modules.module import Module


class LayerNorm(Module):
    # [CLASS DOC]
    # Module wrapping a torch LayerNorm over vectors of size dimension (the only config key). Has
    # trainable affine weights that are saved and loaded.
    config_keys: list[str] = ["dimension"]

    def __init__(self, dimension: int):
        # [FUNCTION DOC]
        # Stores the dimension and creates the LayerNorm layer.
        super().__init__()
        self.dimension = dimension
        self.norm = nn.LayerNorm(dimension)

    def forward(self, features: dict[str, Tensor]):
        # [FUNCTION DOC]
        # Replaces "sentence_embedding" in the features dict with its layer-normalized value and returns
        # the mutated dict.
        features["sentence_embedding"] = self.norm(features["sentence_embedding"])
        return features

    def get_embedding_dimension(self):
        # [FUNCTION DOC]
        # Returns the configured dimension.
        return self.dimension

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
        # Class method that loads config and weights from a local path or Hub repo (subfolder, token,
        # cache folder, revision, local_files_only), constructs the module and returns it with weights loaded.
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
