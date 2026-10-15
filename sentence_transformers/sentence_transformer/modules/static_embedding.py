# [MODULE DOC]
# StaticEmbedding input module: a very fast embedding-bag model that tokenizes text with a fast
# (Rust) tokenizer and averages trainable per-token embeddings to get a text embedding. Provides
# constructors from model2vec distillation and from pre-trained model2vec models.
from __future__ import annotations

import inspect
import logging
import math
import os
from pathlib import Path
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors_file
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast

from sentence_transformers.base.modules.input_module import InputModule
from sentence_transformers.util import get_device_name

logger = logging.getLogger(__name__)


class StaticEmbedding(InputModule):
    # [CLASS DOC]
    # Text-only input module holding a tokenizers Tokenizer and a torch EmbeddingBag (mean mode).
    # Attributes: embedding, num_embeddings, embedding_dim, tokenizer (padding disabled), base_model
    # (used for the model card).
    modalities: list[str] = ["text"]

    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Initializes the module. A transformers fast tokenizer is unwrapped to its backend Tokenizer;
        # any other non-Tokenizer type raises ValueError. If embedding_weights are given (numpy arrays are
        # converted to tensors) creates a trainable EmbeddingBag from them; else if embedding_dim is
        # given creates a randomly initialized EmbeddingBag of vocab size by embedding_dim; else raises
        # ValueError. Disables padding on the tokenizer (mutating it) and stores an optional base_model
        # kwarg.
        """
        Initializes the StaticEmbedding model given a tokenizer. The model is a simple embedding bag model that
        takes the mean of trained per-token embeddings to compute text embeddings.

        Args:
            tokenizer (Tokenizer | PreTrainedTokenizerFast): The tokenizer to be used. Must be a fast tokenizer
                from ``transformers`` or ``tokenizers``.
            embedding_weights (np.ndarray | torch.Tensor | None, optional): Pre-trained embedding weights.
                Defaults to None.
            embedding_dim (int | None, optional): Dimension of the embeddings. Required if embedding_weights
                is not provided. Defaults to None.

        .. tip::

            Due to the extremely efficient nature of this module architecture, the overhead for moving inputs to the
            GPU can be larger than the actual computation time. Therefore, consider using a CPU device for inference
            and training.

        Example::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.modules import StaticEmbedding
            from tokenizers import Tokenizer

            # Pre-distilled embeddings:
            static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
            # or distill your own embeddings:
            static_embedding = StaticEmbedding.from_distillation("BAAI/bge-base-en-v1.5", device="cuda")
            # or start with randomized embeddings:
            tokenizer = Tokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
            static_embedding = StaticEmbedding(tokenizer, embedding_dim=512)

            model = SentenceTransformer(modules=[static_embedding])

            embeddings = model.encode(["What are Pandas?", "The giant panda, also known as the panda bear or simply the panda, is a bear native to south central China."])
            similarity = model.similarity(embeddings[0], embeddings[1])
            # tensor([[0.8093]]) (If you use potion-base-8M)
            # tensor([[0.6234]]) (If you use the distillation method)
            # tensor([[-0.0693]]) (For example, if you use randomized embeddings)

        Raises:
            ValueError: If the tokenizer is not a fast tokenizer.
            ValueError: If neither `embedding_weights` nor `embedding_dim` is provided.
        """
        super().__init__()

        if isinstance(tokenizer, PreTrainedTokenizerFast):
            tokenizer = tokenizer._tokenizer
        elif not isinstance(tokenizer, Tokenizer):
            raise ValueError(
                "The tokenizer must be fast (i.e. Rust-backed) to use this class. "
                "Use Tokenizer.from_pretrained() from `tokenizers` to load a fast tokenizer."
            )

        if embedding_weights is not None:
            if isinstance(embedding_weights, np.ndarray):
                embedding_weights = torch.from_numpy(embedding_weights)

            self.embedding = nn.EmbeddingBag.from_pretrained(embedding_weights, freeze=False)
        elif embedding_dim is not None:
            self.embedding = nn.EmbeddingBag(tokenizer.get_vocab_size(), embedding_dim)
        else:
            raise ValueError("Either `embedding_weights` or `embedding_dim` must be provided.")

        self.num_embeddings = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

        self.tokenizer: Tokenizer = tokenizer
        self.tokenizer.no_padding()

        # For the model card
        self.base_model = kwargs.get("base_model", None)

    def preprocess(self, inputs: list[str], prompt: str | None = None, **kwargs) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Tokenizes inputs (after prepending prompt if given) in batch without special tokens. Flattens
        # all token ids into one long tensor and computes offsets, the start index of each text in that
        # flat tensor (cumulative lengths). Returns a dict with "input_ids" and "offsets".
        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)
        encodings = self.tokenizer.encode_batch(inputs, add_special_tokens=False)
        encodings_ids = [encoding.ids for encoding in encodings]

        offsets = torch.from_numpy(np.cumsum([0] + [len(token_ids) for token_ids in encodings_ids[:-1]]))
        input_ids = torch.tensor([token_id for token_ids in encodings_ids for token_id in token_ids], dtype=torch.long)
        return {"input_ids": input_ids, "offsets": offsets}

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Computes the mean of token embeddings per text with the EmbeddingBag using input_ids and
        # offsets, stores it as "sentence_embedding" in the features dict and returns the mutated dict.
        features["sentence_embedding"] = self.embedding(features["input_ids"], features["offsets"])
        return features

    @property
    def max_seq_length(self) -> int:
        # [FUNCTION DOC]
        # Property returning infinity, since there is no sequence length limit.
        # KNOWN ISSUE: Annotated to return an int but returns the float infinity.
        return math.inf

    def get_embedding_dimension(self) -> int:
        # [FUNCTION DOC]
        # Returns the embedding dimension.
        return self.embedding_dim

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        # [FUNCTION DOC]
        # Saves the state dict as model.safetensors (or pytorch_model.bin when not safe_serialization)
        # and the tokenizer as tokenizer.json in output_path.
        # KNOWN ISSUE: base_model is not saved, so it is lost when the module is reloaded.
        if safe_serialization:
            save_safetensors_file(self.state_dict(), os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        self.tokenizer.save(str(Path(output_path) / "tokenizer.json"))

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
        # Class method that loads tokenizer.json and the weights from a local path or Hub repo, takes the
        # "embedding.weight" tensor (falling back to the model2vec key "embeddings"), and returns a new
        # module built from them. Raises KeyError if neither key exists.
        # KNOWN ISSUE: Always constructs StaticEmbedding instead of cls, so subclasses get the wrong type.
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        tokenizer_path = cls.load_file_path(model_name_or_path, filename="tokenizer.json", **hub_kwargs)
        tokenizer = Tokenizer.from_file(tokenizer_path)

        weights = cls.load_torch_weights(model_name_or_path=model_name_or_path, **hub_kwargs)
        try:
            weights = weights["embedding.weight"]
        except KeyError:
            # For compatibility with model2vec models, which are saved with just an "embeddings" key
            weights = weights["embeddings"]
        return StaticEmbedding(tokenizer, embedding_weights=weights)

    @classmethod
    def from_distillation(
        cls,
        model_name: str,
        vocabulary: list[str] | None = None,
        device: str | None = None,
        pca_dims: int | None = 256,
        apply_zipf: bool = True,
        sif_coefficient: float | None = 1e-4,
        token_remove_pattern: str | None = r"\[unused\d+\]",
        quantize_to: str = "float32",
        use_subword: bool = True,
        **kwargs: Any,
    ) -> StaticEmbedding:
        # [FUNCTION DOC]
        # Class method that distills a static embedding model from a sentence-transformer/transformer model
        # with model2vec. Raises ImportError if model2vec's distill is unavailable. Builds the distill
        # arguments (vocabulary, device, pca_dims, apply_zipf, use_subword, quantize_to, sif_coefficient,
        # token_remove_pattern, plus extra kwargs); drops any the installed distill function does not
        # accept, logging a warning naming them. Runs distillation, converts the resulting embedding
        # (numpy array or embedding layer) to a tensor, and returns a new instance with the distilled
        # tokenizer, weights and base_model set to model_name.
        # KNOWN ISSUE: The detected device from get_device_name is assigned after the kwargs are built and
        # never used, so device stays None unless given by the caller (auto-detection is left to model2vec).
        r"""
        Creates a StaticEmbedding instance from a distillation process using the `model2vec` package.

        Args:
            model_name (str): The name of the model to distill.
            vocabulary (list[str] | None, optional): A list of vocabulary words to use. Defaults to None.
            device (str): The device to run the distillation on (e.g., 'cpu', 'cuda'). If not specified,
                the strongest device is automatically detected. Defaults to None.
            pca_dims (int | None, optional): The number of dimensions for PCA reduction. Defaults to 256.
            apply_zipf (bool): Whether to apply Zipf's law during distillation. Defaults to True.
            sif_coefficient (float | None, optional): The coefficient for SIF weighting. Defaults to 1e-4.
            token_remove_pattern (str | None, optional): A regex pattern to remove tokens from the vocabulary.
                Defaults to r"\[unused\d+\]".
            quantize_to (str): The data type to quantize the weights to. Defaults to 'float32'.
            use_subword (bool): Whether to use subword tokenization. Defaults to True.

        Returns:
            StaticEmbedding: An instance of StaticEmbedding initialized with the distilled model's
                tokenizer and embedding weights.

        Raises:
            ImportError: If the `model2vec` package is not installed.
        """

        try:
            from model2vec.distill import distill
        except ImportError:
            raise ImportError(
                "To use this method, please install the `model2vec` package: `pip install model2vec[distill]`"
            )

        distill_signature = inspect.signature(distill)
        distill_kwargs = set(distill_signature.parameters.keys()) - {"model_name"}
        kwargs = {
            "vocabulary": vocabulary,
            "device": device,
            "pca_dims": pca_dims,
            "apply_zipf": apply_zipf,
            "use_subword": use_subword,
            "quantize_to": quantize_to,
            "sif_coefficient": sif_coefficient,
            "token_remove_pattern": token_remove_pattern,
            **kwargs,
        }
        if leftovers := set(kwargs.keys()) - distill_kwargs:
            logger.warning(
                f"Your version of `model2vec` does not support the {', '.join(map(repr, leftovers))} arguments for the `distill` method. "
                "Consider updating `model2vec` to take advantage of these arguments."
            )
            kwargs = {key: value for key, value in kwargs.items() if key in distill_kwargs}

        device = get_device_name()
        static_model = distill(model_name, **kwargs)
        if isinstance(static_model.embedding, np.ndarray):
            embedding_weights = torch.from_numpy(static_model.embedding).contiguous()
        else:
            embedding_weights = static_model.embedding.weight
        tokenizer: Tokenizer = static_model.tokenizer

        return cls(tokenizer, embedding_weights=embedding_weights, base_model=model_name)

    @classmethod
    def from_model2vec(cls, model_id_or_path: str) -> StaticEmbedding:
        # [FUNCTION DOC]
        # Class method that loads a pre-trained model2vec StaticModel by id or path (raises ImportError if
        # model2vec is missing), converts its embedding to a tensor, and returns a new instance with its
        # tokenizer, weights and base_model set to the id or path.
        """
        Create a StaticEmbedding instance from a model2vec model. This method loads a pre-trained model2vec model
        and extracts the embedding weights and tokenizer to create a StaticEmbedding instance.

        Args:
            model_id_or_path (str): The identifier or path to the pre-trained model2vec model.

        Returns:
            StaticEmbedding: An instance of StaticEmbedding initialized with the tokenizer and embedding weights
                 the model2vec model.

        Raises:
            ImportError: If the `model2vec` package is not installed.
        """

        try:
            from model2vec import StaticModel
        except ImportError:
            raise ImportError("To use this method, please install the `model2vec` package: `pip install model2vec`")

        static_model = StaticModel.from_pretrained(model_id_or_path)
        if isinstance(static_model.embedding, np.ndarray):
            embedding_weights = torch.from_numpy(static_model.embedding).contiguous()
        else:
            embedding_weights = static_model.embedding.weight
        tokenizer: Tokenizer = static_model.tokenizer

        return cls(tokenizer, embedding_weights=embedding_weights, base_model=model_id_or_path)
