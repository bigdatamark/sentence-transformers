# [MODULE DOC]
# Deprecated CLIPModel module, kept so older saved models that reference it still load. It is a
# thin subclass of the generic Transformer module configured for CLIP text and image features.
from __future__ import annotations

import logging
from typing import Any

from sentence_transformers.base.modules.transformer import ModalityConfig, Transformer

logger = logging.getLogger(__name__)


class CLIPModel(Transformer):
    # [CLASS DOC]
    # Legacy CLIP module subclassing Transformer. Maps text inputs to the CLIP text features method and
    # image inputs to the image features method, outputting a "sentence_embedding" feature.
    """Legacy module kept only for backward compatibility with older saved models.

    New multimodal models should use :class:`~sentence_transformers.base.modules.Transformer` directly,
    which handles CLIP-style models and other multimodal architectures natively via modality auto-detection.

    .. deprecated:: 5.4
        Use :class:`~sentence_transformers.base.modules.Transformer` instead.
    """

    def __init__(
        self, model_name_or_path: str = "openai/clip-vit-base-patch32", _from_auto_load: bool = False, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Constructs the module. Logs a deprecation warning unless created via load (_from_auto_load).
        # Renames a legacy processor_name keyword to tokenizer_name_or_path, then delegates to the
        # Transformer constructor with the model name (default openai/clip-vit-base-patch32) and kwargs.
        if not _from_auto_load:
            logger.warning("The CLIPModel module is deprecated. Please use the Transformer module instead.")
        if "processor_name" in kwargs:
            kwargs["tokenizer_name_or_path"] = kwargs.pop("processor_name")
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs) -> CLIPModel:
        # [FUNCTION DOC]
        # Class method that loads the module: reads constructor kwargs from the saved config via
        # _load_init_kwargs and constructs the class with _from_auto_load set so no warning is logged.
        init_kwargs = cls._load_init_kwargs(model_name_or_path=model_name_or_path, **kwargs)
        return cls(model_name_or_path=model_name_or_path, _from_auto_load=True, **init_kwargs)

    @staticmethod
    def _get_default_modality_config(config: dict[str, Any]) -> tuple[ModalityConfig, str]:
        # [FUNCTION DOC]
        # HELPER: Returns the default modality config and output name. Text maps to get_text_features and
        # image to get_image_features of the transformers CLIP model, each with an output name inferred
        # (pooler output vs plain tensor) from that method; the module output name is
        # "sentence_embedding". The config argument is unused. Imports transformers lazily.
        """Get the default modality configuration for the current transformer task.

        Returns:
            tuple[MODALITY_CONFIG, str]: A tuple of (modality_config, module_output_name).
                The modality_config maps modality keys to dicts with 'method' and 'method_output_name'.
                The module_output_name is the name of the output feature this module creates.
        """
        from transformers import CLIPModel as TransformersCLIPModel

        # Use Transformer._infer_method_output_name to check whether the method outputs a BaseModelOutputWithPooling
        # with a pooler_output, or just a Tensor output
        modality_config: ModalityConfig = {
            "text": {
                "method": "get_text_features",
                "method_output_name": Transformer._infer_method_output_name(
                    "pooler_output", TransformersCLIPModel.get_text_features
                ),
            },
            "image": {
                "method": "get_image_features",
                "method_output_name": Transformer._infer_method_output_name(
                    "pooler_output", TransformersCLIPModel.get_image_features
                ),
            },
        }
        module_output_name = "sentence_embedding"
        return modality_config, module_output_name
