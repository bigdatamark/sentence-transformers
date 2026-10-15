# [MODULE DOC] The SentenceTransformer model class: a sequential stack of modules (typically a Transformer followed by Pooling
# and optional Dense / Normalize modules) that maps texts or other inputs to fixed-size dense embeddings. Builds on
# BaseModel (loading, saving, prompts, preprocessing, multi-process pools, Hub integration) and FitMixin (legacy
# training). Adds encode / encode_query / encode_document, similarity functions, embedding truncation and default
# module construction for plain Hugging Face checkpoints. ALLOWED_PRECISIONS lists the supported output precisions.
from __future__ import annotations

import copy
import itertools
import logging
import math
import queue
import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from multiprocessing import Queue
from typing import Any, ClassVar, Literal, overload

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.autonotebook import trange
from transformers.utils import logging as transformers_logging
from typing_extensions import TypeIs, deprecated

from sentence_transformers.base.modality_types import SingleInput
from sentence_transformers.base.model import BaseModel
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sentence_transformer.modules import Pooling
from sentence_transformers.util import batch_to_device, truncate_embeddings
from sentence_transformers.util.decorators import deprecated_kwargs
from sentence_transformers.util.quantization import quantize_embeddings
from sentence_transformers.util.similarity import SimilarityFunction

from .fit_mixin import FitMixin
from .model_card import SentenceTransformerModelCardData

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)

ALLOWED_PRECISIONS = {"float32", "int8", "uint8", "binary", "ubinary"}


class SentenceTransformer(BaseModel, FitMixin):
    # [CLASS DOC] Dense embedding model. Class attributes: model card data class, default Hub organization, default prompts (query
    # and document set to None), model type name, and the tuple of supported similarity function names (cosine, dot,
    # euclidean, manhattan). Instance state added on top of BaseModel: similarity_fn_name (lazily defaulting to cosine,
    # with cached matrix and pairwise similarity callables) and truncate_dim (optional output dimension truncation).
    """
    Loads or creates a SentenceTransformer model that can be used to map text and other inputs to dense embeddings.

    Args:
        model_name_or_path (str, optional): If a filepath on disk, loads the model from that path. Otherwise, tries
            to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from
            the Hugging Face Hub with that name. Defaults to None.
        modules (list[nn.Module], optional): A list of torch modules that are called sequentially. Can be used to
            create custom SentenceTransformer models from scratch. Defaults to None.
        device (str, optional): Device (like ``"cuda"``, ``"cpu"``, ``"mps"``, ``"npu"``) that should be used for
            computation. If None, checks if a GPU can be used. If a ``device_map`` is provided via
            ``model_kwargs``, that controls device placement and this argument is ignored. Defaults to None.
        prompts (dict[str, str], optional): A dictionary with prompts for the model. The key is the prompt name,
            the value is the prompt text. The prompt text will be prepended before any text to encode. For example:
            ``{"query": "query: ", "passage": "passage: "}``. If a model has saved prompts, you can override
            them by passing your own, or pass ``{"query": "", "document": ""}`` to disable them.
            Defaults to None.
        default_prompt_name (str, optional): The name of the prompt that should be used by default. If not set,
            no prompt will be applied. Defaults to None.
        cache_folder (str, optional): Path to store models. Can also be set by the ``SENTENCE_TRANSFORMERS_HOME``
            environment variable. Defaults to None.
        trust_remote_code (bool, optional): Whether to allow for custom models defined on the Hub in their own
            modeling files. Only set to ``True`` for repositories you trust and in which you have read the code,
            as it will execute code present on the Hub on your local machine. Defaults to False.
        revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a
            commit id, for a stored model on Hugging Face. Defaults to None.
        local_files_only (bool, optional): Whether to only look at local files (i.e., do not try to download
            the model). Defaults to False.
        token (bool or str, optional): Hugging Face authentication token to download private models.
            Defaults to None.
        use_auth_token (bool or str, optional): Deprecated. Use ``token`` instead.
        model_kwargs (dict[str, Any], optional): Keyword arguments passed to the underlying Hugging Face
            Transformers model via ``AutoModel.from_pretrained``. Particularly useful options include:

            - ``torch_dtype``: Override the default ``torch.dtype`` and load the model under a specific
              dtype. Can be ``torch.float16``, ``torch.bfloat16``, ``torch.float32``, or ``"auto"`` to
              use the dtype from the model's ``config.json``.
            - ``attn_implementation``: The attention implementation to use. For example ``"eager"``,
              ``"sdpa"``, or ``"flash_attention_2"``. If you ``pip install kernels``, then
              ``"flash_attention_2"`` should work without having to install ``flash_attn``. It is
              frequently the fastest option. Defaults to ``"sdpa"`` when available (torch>=2.1.1).
            - ``device_map``: Controls how the model is placed across devices, e.g. ``"auto"`` for
              model parallelism, or a single device like ``"cuda:1"`` to load the backbone directly
              onto one GPU (useful when serving multiple models in one process). When set, it takes
              precedence over the ``device`` argument.
            - ``provider``: For ``backend="onnx"``, the ONNX execution provider
              (e.g. ``"CUDAExecutionProvider"``).
            - ``file_name``: For ``backend="onnx"`` or ``"openvino"``, the filename to load
              (e.g. for optimized or quantized models).
            - ``export``: For ``backend="onnx"`` or ``"openvino"``, whether to export the model to the
              backend format. Also set automatically if the exported file doesn't exist.

            See the `PreTrainedModel.from_pretrained
            <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
            documentation for more details. Defaults to None.
        processor_kwargs (dict[str, Any], optional): Keyword arguments passed to the Hugging Face Transformers
            processor/tokenizer via ``AutoProcessor.from_pretrained``. See the `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details. Defaults to None.
        config_kwargs (dict[str, Any], optional): Keyword arguments passed to the Hugging Face Transformers
            config via ``AutoConfig.from_pretrained``. See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details. Defaults to None.
        model_card_data (:class:`~sentence_transformers.sentence_transformer.model_card.SentenceTransformerModelCardData`, optional):
            A model card data object that contains information about the model. Used to generate a model card
            when saving the model. If not set, a default model card data object is created. Defaults to None.
        backend (str, optional): The backend to use for inference. Can be ``"torch"`` (default), ``"onnx"``,
            or ``"openvino"``. Defaults to ``"torch"``.
        similarity_fn_name (str or SimilarityFunction, optional): The name of the similarity function to use.
            Valid options are ``"cosine"``, ``"dot"``, ``"euclidean"``, and ``"manhattan"``. If not set, it is
            automatically set to ``"cosine"`` when :meth:`similarity` or :meth:`similarity_pairwise` are first
            called. Defaults to None.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. ``None`` means no
            truncation. Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer

            # Load a pre-trained SentenceTransformer model
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

            # Encode some texts
            sentences = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 768)

            # Get the similarity scores between all sentences
            similarities = model.similarity(embeddings, embeddings)
            print(similarities)
            # tensor([[1.0000, 0.6817, 0.0492],
            #         [0.6817, 1.0000, 0.0421],
            #         [0.0492, 0.0421, 1.0000]])
    """

    model_card_data_class = SentenceTransformerModelCardData
    default_huggingface_organization: str | None = "sentence-transformers"
    _default_prompts: dict[str, str | None] = {"query": None, "document": None}
    model_type: str = "SentenceTransformer"
    # Supported single-vector similarities. Multi-vector names (maxsim) would produce wrong shapes on 2D embeddings.
    SUPPORTED_SIMILARITY_FN_NAMES: ClassVar[tuple[str, ...]] = (
        SimilarityFunction.COSINE.value,
        SimilarityFunction.DOT_PRODUCT.value,
        SimilarityFunction.EUCLIDEAN.value,
        SimilarityFunction.MANHATTAN.value,
    )

    @deprecated_kwargs(tokenizer_kwargs="processor_kwargs")
    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        modules: list[nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: SentenceTransformerModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        # SentenceTransformer-specific args
        similarity_fn_name: Literal["cosine", "dot", "euclidean", "manhattan"] | SimilarityFunction | None = None,
        truncate_dim: int | None = None,
    ) -> None:
        # [FUNCTION DOC] Set similarity_fn_name (validated by its setter; ValueError for unsupported names) and truncate_dim before calling
        # the base constructor, which loads or assembles modules and parses the saved config (which may fill these if unset).
        # The deprecated use_auth_token is mapped to token with a FutureWarning; passing both raises ValueError. The
        # deprecated keyword tokenizer_kwargs is remapped to processor_kwargs by a decorator. For the known INSTRUCTOR
        # checkpoints, disables prompt inclusion in pooling; for other repo names containing "instructor", logs a warning
        # when any Pooling module still includes the prompt. Side effects: downloads/loads model files.
        # Set before super().__init__() so _parse_model_config can check these
        self.similarity_fn_name = similarity_fn_name
        self.truncate_dim = truncate_dim

        # Handle deprecated use_auth_token
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in a future release of SentenceTransformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "Both `token` and `use_auth_token` are specified. Please only specify the `token` argument."
                )
            token = use_auth_token

        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
        )
        self.model_card_data: SentenceTransformerModelCardData

        # Handle INSTRUCTOR models
        if model_name_or_path in ("hkunlp/instructor-base", "hkunlp/instructor-large", "hkunlp/instructor-xl"):
            self.set_pooling_include_prompt(include_prompt=False)
        elif (
            model_name_or_path
            and "/" in model_name_or_path
            and "instructor" in model_name_or_path.split("/")[1].lower()
        ):
            if any(module.include_prompt for module in self if isinstance(module, Pooling)):
                logger.warning(
                    "Instructor models require `include_prompt=False` in the pooling configuration. "
                    "Either update the model configuration or call `model.set_pooling_include_prompt(False)` after loading the model."
                )

    @overload
    def encode_query(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: Literal[True] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode_query(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        *,
        convert_to_tensor: Literal[True],
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode_query(
        self,
        inputs: SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        *,
        convert_to_numpy: Literal[False],
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode_query(
        self,
        inputs: Sequence[SingleInput],
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        *,
        convert_to_numpy: Literal[False],
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[Tensor]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    # Catch-all so calls with union-typed arguments or output_value overrides still resolve.
    @overload
    def encode_query(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @deprecated_kwargs(sentences="inputs")
    def encode_query(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        # [FUNCTION DOC] Encode inputs as queries: when neither prompt_name nor prompt is given and a "query" prompt exists, use it; then
        # delegate to encode with every argument plus task "query" (which routers use to pick query-specific modules).
        # Returns whatever encode returns. The deprecated keyword sentences is remapped to inputs.
        """
        Computes embeddings specifically optimized for query representation.

        This method is a specialized version of :meth:`encode` that differs in exactly two ways:

        1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "query" prompt,
           if available in the model's ``prompts`` dictionary.
        2. It sets the ``task`` to "query". If the model has a :class:`~sentence_transformers.base.modules.Router`
           module, it will use the "query" task type to route the input through the appropriate submodules.

        .. tip::

            Adjusting ``batch_size`` can significantly improve processing speed. The optimal value depends on your
            hardware, model size, precision, and input length. Benchmark a few batch sizes on a small subset of your
            data to find the best value.

        All other parameters are identical to :meth:`encode`. See :meth:`encode` for the full parameter documentation.
        """
        if prompt_name is None and prompt is None and "query" in self.prompts:
            prompt_name = "query"

        return self.encode(
            inputs=inputs,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=output_value,
            precision=precision,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            pool=pool,
            chunk_size=chunk_size,
            task="query",
            **kwargs,
        )

    @overload
    def encode_document(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: Literal[True] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode_document(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        *,
        convert_to_tensor: Literal[True],
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode_document(
        self,
        inputs: SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        *,
        convert_to_numpy: Literal[False],
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode_document(
        self,
        inputs: Sequence[SingleInput],
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        *,
        convert_to_numpy: Literal[False],
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[Tensor]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    # Catch-all so calls with union-typed arguments or output_value overrides still resolve.
    @overload
    def encode_document(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @deprecated_kwargs(sentences="inputs")
    def encode_document(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        # [FUNCTION DOC] Encode inputs as documents: when neither prompt_name nor prompt is given, use the first existing prompt named
        # "document", "passage" or "corpus"; then delegate to encode with task "document". Returns what encode returns.
        """
        Computes embeddings specifically optimized for document/passage representation.

        This method is a specialized version of :meth:`encode` that differs in exactly two ways:

        1. If no ``prompt_name`` or ``prompt`` is provided, it uses the first available prompt from the following
           candidates: ``"document"``, ``"passage"``, ``"corpus"`` (checked in that order).
        2. It sets the ``task`` to "document". If the model has a :class:`~sentence_transformers.base.modules.Router`
           module, it will use the "document" task type to route the input through the appropriate submodules.

        .. tip::

            Adjusting ``batch_size`` can significantly improve processing speed. The optimal value depends on your
            hardware, model size, precision, and input length. Benchmark a few batch sizes on a small subset of your
            data to find the best value.

        All other parameters are identical to :meth:`encode`. See :meth:`encode` for the full parameter documentation.
        """
        if prompt_name is None and prompt is None:
            for candidate_prompt_name in ["document", "passage", "corpus"]:
                if candidate_prompt_name in self.prompts:
                    prompt_name = candidate_prompt_name
                    break

        return self.encode(
            inputs=inputs,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=output_value,
            precision=precision,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            pool=pool,
            chunk_size=chunk_size,
            task="document",
            **kwargs,
        )

    def is_singular_input(self, inputs: Any) -> TypeIs[SingleInput]:
        # [FUNCTION DOC] Return whether the input is one example rather than a batch, delegating to the base implementation (redeclared
        # only for type narrowing).
        """Check if the input is a single example rather than a batch. Redeclared with
        :class:`~typing_extensions.TypeIs` so type checkers narrow the branches in :meth:`encode`."""
        return super().is_singular_input(inputs)

    # Ordered overloads: the first match wins, so the all-defaults signatures come first and each
    # deviation pins its deviating flag as a required keyword. Singular inputs precede Sequence ones
    # so plain strings and bare conversations resolve as one input (both also match Sequence).
    @overload
    def encode(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: Literal[True] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        *,
        convert_to_tensor: Literal[True],
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode(
        self,
        inputs: SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        *,
        convert_to_numpy: Literal[False],
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode(
        self,
        inputs: SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        *,
        output_value: Literal["token_embeddings"],
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode(
        self,
        inputs: SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        *,
        output_value: None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> dict[str, Tensor]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode(
        self,
        inputs: Sequence[SingleInput],
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        *,
        convert_to_numpy: Literal[False],
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[Tensor]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode(
        self,
        inputs: Sequence[SingleInput],
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        *,
        output_value: Literal["token_embeddings"],
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[Tensor]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @overload
    def encode(
        self,
        inputs: Sequence[SingleInput],
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        *,
        output_value: None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[dict[str, Tensor]]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    # Catch-all so forwarding calls with union-typed arguments (e.g. from encode_query) still resolve.
    @overload
    def encode(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]: ...
        # [FUNCTION DOC] Typing-only overload signature (no runtime behavior); the implementation below handles the call.

    @torch.inference_mode()
    @deprecated_kwargs(sentences="inputs")
    def encode(
        self,
        inputs: Sequence[SingleInput] | SingleInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        # [FUNCTION DOC] Compute embeddings. Runs in inference mode (no autograd). On HPU devices, wraps the model in an HPU graph once.
        # Progress bar defaults to on at INFO/DEBUG log level. convert_to_tensor disables numpy conversion; any output_value
        # other than "sentence_embedding" disables both conversions. Raises ValueError for non-positive batch size, for extra
        # keyword arguments not accepted by the model (besides task and processing_kwargs), or for an unsupported precision.
        # A singular input is wrapped in a one-item list; other non-list iterables are materialized to a list. If a pool or a
        # non-empty device list is given, encodes via multiple processes in float32, then quantizes the merged result once
        # (restoring tensor or list format) and unwraps singular input. Otherwise resolves the prompt, moves the model to the
        # device and sets eval mode, uses truncate_dim or the model default, sorts inputs by descending length (interleaved
        # when inputs can be flattened) and processes them in batches: preprocess with the prompt and kwargs, pad on HPU,
        # move to device, run the model, truncate sentence embeddings if requested. For "token_embeddings", each row is
        # trimmed after its last non-padded token; for None, each item becomes a dict of that row's values for every output
        # key (non-indexable values copied as is); for sentence embeddings, optionally L2-normalized and moved to CPU when
        # converting to numpy. Results are restored to input order, quantized when precision is not float32, then returned
        # as one stacked tensor (empty tensor if no inputs), a 2-D numpy array (bfloat16 upcast to float32), or a list of
        # tensors; singular input returns its single element.
        # KNOWN ISSUE: The docstring says convert_to_numpy returns a list of numpy vectors, but the code returns one 2-D numpy array.
        """
        Computes embeddings for the given inputs.

        .. tip::

            If you are unsure whether you should use :meth:`encode`, :meth:`encode_query`, or :meth:`encode_document`,
            your best bet is to use :meth:`encode_query` and :meth:`encode_document` for Information Retrieval tasks
            with clear query and document/passage distinction, and use :meth:`encode` for all other tasks.

            Note that :meth:`encode` is the most general method and can be used for any task, including Information
            Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three
            methods will return identical embeddings.

        .. tip::

            Adjusting ``batch_size`` can significantly improve processing speed. The optimal value depends on your
            hardware, model size, precision, and input length. Benchmark a few batch sizes on a small subset of your
            data to find the best value.

        Args:
            inputs: The inputs to embed. Can be a string, a list of strings, or multimodal inputs
                (dicts, images, arrays).
            prompt_name (str, optional): The name of the prompt to use for encoding. Must be a key in the ``prompts``
                dictionary, which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (str, optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored.
                Defaults to None.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encoding. Defaults to None,
                in which case the progress bar will be shown if the logger's effective level is INFO or DEBUG.
            output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return.
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor.
            device (str, torch.device, list, or None, optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding

                If a list is provided, multi-process encoding will be used. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by
                ``start_multi_process_pool()``.
            chunk_size (int, optional): Size of chunks for multi-process encoding.
            **kwargs: Additional keyword arguments to pass to the model's ``preprocess`` and ``forward`` methods.
                A notable one is ``processing_kwargs``, which lets you override the processor kwargs configured on
                the underlying :class:`~sentence_transformers.base.modules.Transformer` for this call only (e.g.
                ``processing_kwargs={"text": {"max_length": 256, "truncation": True}}``). See
                :meth:`Transformer.preprocess <sentence_transformers.base.modules.Transformer.preprocess>` for the
                accepted structure.

        Returns:
            Union[List[Tensor], ndarray, Tensor, dict[str, Tensor], list[dict[str, Tensor]]]: By default, a 2d numpy
                array with shape [num_inputs, output_dimension] is returned. If ``output_value`` is ``None``, a list
                of dicts (or a single dict for singular input) is returned.
        """
        if self.device.type == "hpu" and not self.is_hpu_graph_enabled:
            import habana_frameworks.torch as ht

            if hasattr(ht, "hpu") and hasattr(ht.hpu, "wrap_in_hpu_graph"):
                ht.hpu.wrap_in_hpu_graph(self, disable_tensor_cache=True)
                self.is_hpu_graph_enabled = True

        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        # Cast an individual input to a list with length 1
        is_singular_input = self.is_singular_input(inputs)
        if is_singular_input:
            inputs = [inputs]
        elif not isinstance(inputs, list):
            # Materialize e.g. datasets.Column to avoid slow Arrow deserialization on each index.
            # The annotation pins ndarray.tolist()'s Any so checkers keep the narrowed input type.
            materialized: list[SingleInput] = inputs.tolist() if isinstance(inputs, np.ndarray) else list(inputs)
            inputs = materialized

        # Validate kwargs
        model_kwargs = self.get_model_kwargs()
        if unused_kwargs := set(kwargs) - set(model_kwargs) - {"task", "processing_kwargs"}:
            raise ValueError(
                f"{self.__class__.__name__}.encode() has been called with additional keyword arguments that this model does not use: {list(unused_kwargs)}. "
                + (
                    f"As per {self.__class__.__name__}.get_model_kwargs(), the valid additional keyword arguments are: {model_kwargs}."
                    if model_kwargs
                    else f"As per {self.__class__.__name__}.get_model_kwargs(), this model does not accept any additional keyword arguments."
                )
            )

        # Validate precision
        if precision is not None and precision not in ALLOWED_PRECISIONS:
            raise ValueError(f"Precision {precision!r} is not supported, must be one of {ALLOWED_PRECISIONS}")

        # If pool or a list of devices is provided, use multi-process encoding
        if pool is not None or (isinstance(device, list) and len(device) > 0):
            embeddings = self._multi_process(
                inputs,
                # Utility and post-processing parameters
                show_progress_bar=show_progress_bar,
                # Multi-process encoding parameters
                pool=pool,
                device=device,
                chunk_size=chunk_size,
                # Encoding parameters
                prompt_name=prompt_name,
                prompt=prompt,
                batch_size=batch_size,
                output_value=output_value,
                # Quantize once after merging, not per-worker: int8/uint8 calibration ranges would differ per chunk.
                precision="float32",
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings,
                truncate_dim=truncate_dim,
                **kwargs,
            )
            if len(embeddings) and precision and precision != "float32":
                embeddings = quantize_embeddings(embeddings, precision=precision)
                # quantize_embeddings returns a numpy matrix: restore the requested output format.
                if convert_to_tensor:
                    embeddings = torch.from_numpy(embeddings)
                elif not convert_to_numpy and isinstance(embeddings, np.ndarray):
                    embeddings = [torch.from_numpy(embedding) for embedding in embeddings]
            if is_singular_input:
                embeddings = embeddings[0]
            return embeddings

        prompt = self._resolve_prompt(prompt, prompt_name)

        # Set device
        if device is None:
            device = self.device
        self.to(device)
        self.eval()

        truncate_dim = truncate_dim if truncate_dim is not None else self.truncate_dim
        all_embeddings = []
        length_sorted_idx = np.argsort([-self._input_length(sen) for sen in inputs])
        if self._can_flatten_inputs():
            length_sorted_idx = self._interleave_sorted_indices(length_sorted_idx)
        inputs_sorted = [inputs[idx] for idx in length_sorted_idx]

        is_hpu = self.device.type == "hpu"
        for start_index in trange(0, len(inputs_sorted), batch_size, desc="Batches", disable=not show_progress_bar):
            inputs_batch = inputs_sorted[start_index : start_index + batch_size]
            features = self.preprocess(inputs_batch, prompt=prompt, **kwargs)

            if is_hpu:
                features = self._pad_features_for_hpu(features)

            features = batch_to_device(features, device)

            # Route through __call__ so that model.compile() applies to the forward pass.
            out_features = self(features, **kwargs)
            if is_hpu:
                out_features = copy.deepcopy(out_features)

            if truncate_dim is not None:
                out_features["sentence_embedding"] = truncate_embeddings(
                    out_features["sentence_embedding"], truncate_dim
                )

            if output_value == "token_embeddings":
                embeddings = []
                for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1
                    embeddings.append(token_emb[: last_mask_id + 1])
            elif output_value is None:
                embeddings = []
                for idx in range(len(out_features["sentence_embedding"])):
                    batch_item = {}
                    for name, value in out_features.items():
                        try:
                            batch_item[name] = value[idx]
                        except TypeError:
                            batch_item[name] = value
                    embeddings.append(batch_item)
            else:
                embeddings = out_features[output_value]
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if all_embeddings and precision and precision != "float32":
            all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

        if convert_to_tensor:
            if len(all_embeddings):
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = torch.from_numpy(all_embeddings)
                else:
                    all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.tensor([], device=self.device)
        elif convert_to_numpy:
            if not isinstance(all_embeddings, np.ndarray):
                if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
                    all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if is_singular_input:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @staticmethod
    def _pad_features_for_hpu(features: dict[str, Tensor]) -> dict[str, Tensor]:
        # [FUNCTION DOC] HELPER: For HPU graphs, right-pad input_ids (with ones), attention_mask and token_type_ids (with zeros) as int8 to
        # the next power-of-two sequence length. Returns the features unchanged when there are no input_ids or no padding
        # is needed; otherwise mutates and returns the dict.
        """Pad input features to the next power of 2 for HPU graph compatibility."""
        if "input_ids" not in features:
            return features

        batch_size, seq_len = features["input_ids"].shape
        padded_len = 2 ** math.ceil(math.log2(seq_len)) if seq_len > 0 else 0
        pad_len = padded_len - seq_len
        if pad_len == 0:
            return features

        features["input_ids"] = torch.cat(
            (features["input_ids"], torch.ones((batch_size, pad_len), dtype=torch.int8)), -1
        )
        features["attention_mask"] = torch.cat(
            (features["attention_mask"], torch.zeros((batch_size, pad_len), dtype=torch.int8)), -1
        )
        if "token_type_ids" in features:
            features["token_type_ids"] = torch.cat(
                (features["token_type_ids"], torch.zeros((batch_size, pad_len), dtype=torch.int8)), -1
            )
        return features

    @property
    def similarity_fn_name(self) -> Literal["cosine", "dot", "euclidean", "manhattan"]:
        # [FUNCTION DOC] Return the similarity function name; if unset, set it to cosine first (which also builds the similarity callables).
        """Return the name of the similarity function.

        If not previously set, accessing this property defaults it to ``"cosine"``.
        """
        if self._similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE
        return self._similarity_fn_name

    @similarity_fn_name.setter
    def similarity_fn_name(
        self, value: Literal["cosine", "dot", "euclidean", "manhattan"] | SimilarityFunction | None
    ) -> None:
        # [FUNCTION DOC] Accept a name or SimilarityFunction member (converted to its string value) or None. Raises ValueError for names not
        # in the supported tuple (with a hint to use MultiVectorEncoder for "maxsim"). For non-None values, build the matrix
        # and pairwise similarity callables. Stores the name.
        if isinstance(value, SimilarityFunction):
            value = value.value
        if value is not None and value not in self.SUPPORTED_SIMILARITY_FN_NAMES:
            hint = " Use MultiVectorEncoder for MaxSim late-interaction scoring." if value == "maxsim" else ""
            raise ValueError(
                f"{type(self).__name__} only supports similarity_fn_name in "
                f"{self.SUPPORTED_SIMILARITY_FN_NAMES}, got {value!r}.{hint}"
            )
        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(value)
        self._similarity_fn_name = value

    def similarity(
        self,
        embeddings1: Tensor | np.ndarray | list[Tensor] | list[np.ndarray],
        embeddings2: Tensor | np.ndarray | list[Tensor] | list[np.ndarray],
    ) -> Tensor:
        # [FUNCTION DOC] Return the matrix of similarities between every embedding in the first and second input using the configured
        # function (initializing the default first). Inputs may be tensors, arrays or lists of them.
        """
        Compute the similarity between two collections of embeddings. The output is a matrix with the
        similarity scores between all embeddings from the first parameter and all embeddings from the
        second parameter, computed with the similarity function set via ``similarity_fn_name``.

        Args:
            embeddings1 (Union[Tensor, ndarray]): [num_embeddings_1, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2 (Union[Tensor, ndarray]): [num_embeddings_2, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            Tensor: A [num_embeddings_1, num_embeddings_2]-shaped torch tensor with similarity scores.
        """
        # Access similarity_fn_name to trigger lazy initialization of _similarity
        self.similarity_fn_name  # noqa: B018
        return self._similarity(embeddings1, embeddings2)

    def similarity_pairwise(
        self,
        embeddings1: Tensor | np.ndarray | list[Tensor] | list[np.ndarray],
        embeddings2: Tensor | np.ndarray | list[Tensor] | list[np.ndarray],
    ) -> Tensor:
        # [FUNCTION DOC] Return the similarity between aligned pairs of embeddings (row i with row i) using the configured pairwise function.
        """
        Compute the pairwise similarity between two collections of embeddings, i.e. the similarity
        between ``embeddings1[i]`` and ``embeddings2[i]`` for each ``i``.

        Args:
            embeddings1 (Union[Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2 (Union[Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            Tensor: A [num_embeddings]-shaped torch tensor with pairwise similarity scores.
        """
        # Access similarity_fn_name to trigger lazy initialization of _similarity_pairwise
        self.similarity_fn_name  # noqa: B018
        return self._similarity_pairwise(embeddings1, embeddings2)

    @deprecated(
        "The `encode_multi_process` method has been deprecated, and its functionality has been integrated into `encode`. "
        "You can now call `encode` with the same parameters to achieve multi-process encoding.",
    )
    def encode_multi_process(
        self,
        sentences: list[str],
        pool: dict[Literal["input", "output", "processes"], Any],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        chunk_size: int | None = None,
        show_progress_bar: bool | None = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
    ) -> np.ndarray:
        # [FUNCTION DOC] Deprecated wrapper: encode sentences with the given pool and settings via encode, returning a numpy array of
        # sentence embeddings. Emits a deprecation warning.
        """
        .. warning::
            This method is deprecated. You can now call :meth:`SentenceTransformer.encode`
            with the same parameters instead, which will automatically handle multi-process encoding using the provided ``pool``.
        """
        return self.encode(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value="sentence_embedding",
            precision=precision,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            pool=pool,
            chunk_size=chunk_size,
        )

    def _multi_process(
        self,
        inputs: Sequence[SingleInput],
        show_progress_bar: bool | None = True,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        device: str | list[str | torch.device] | None = None,
        chunk_size: int | None = None,
        **encode_kwargs,
    ) -> list | Tensor | np.ndarray:
        # [FUNCTION DOC] HELPER: Multi-process encoding. Disables per-worker progress bars. Creates a pool from the device list when none is
        # given (and stops it afterwards). Chunk size defaults to about a tenth of the per-process share, capped at 5000 and at
        # least 1. Puts (chunk id, chunk, kwargs) jobs on the input queue, collects as many results from the output queue
        # (optional progress bar), sorts them by chunk id, and concatenates: lists are chained, tensors and arrays are
        # concatenated; with no inputs returns an empty tensor, empty array or empty list according to the conversion flags.
        """Internal method for multi-process encoding.

        Either ``pool`` or ``device`` (as a list) must be provided. If ``pool`` is ``None`` and ``device``
        is a list, a temporary pool is created and cleaned up after encoding.
        """
        convert_to_tensor = encode_kwargs.get("convert_to_tensor", False)
        convert_to_numpy = encode_kwargs.get("convert_to_numpy", False)
        encode_kwargs["show_progress_bar"] = False

        # Create a pool if not provided, but a list of devices is
        created_pool = False
        if pool is None and isinstance(device, list):
            pool = self.start_multi_process_pool(device)
            created_pool = True

        try:
            # Determine chunk size
            if chunk_size is None:
                chunk_size = min(math.ceil(len(inputs) / len(pool["processes"]) / 10), 5000)
                chunk_size = max(chunk_size, 1)

            input_queue: torch.multiprocessing.Queue = pool["input"]
            output_queue: torch.multiprocessing.Queue = pool["output"]

            # Send inputs to the input queue in chunks
            chunk_id = -1
            for chunk_id, chunk_start in enumerate(range(0, len(inputs), chunk_size)):
                chunk = inputs[chunk_start : chunk_start + chunk_size]
                input_queue.put([chunk_id, chunk, encode_kwargs])

            # Collect results from the output queue
            output_list = sorted(
                [output_queue.get() for _ in trange(chunk_id + 1, desc="Chunks", disable=not show_progress_bar)],
                key=lambda x: x[0],
            )

            # Handle the various output formats
            embeddings = [output[1] for output in output_list]
            if embeddings:
                if isinstance(embeddings[0], list):
                    embeddings = list(itertools.chain.from_iterable(embeddings))
                elif isinstance(embeddings[0], torch.Tensor):
                    embeddings = torch.cat(embeddings)
                elif isinstance(embeddings[0], np.ndarray):
                    embeddings = np.concatenate(embeddings, axis=0)
            elif convert_to_tensor:
                embeddings = torch.tensor([])
            elif convert_to_numpy:
                embeddings = np.array([])
            return embeddings

        finally:
            if created_pool:
                self.stop_multi_process_pool(pool)

    @staticmethod
    def _multi_process_worker(
        target_device: str, model: SentenceTransformer, input_queue: Queue, results_queue: Queue
    ) -> None:
        # [FUNCTION DOC] HELPER: Worker process loop: repeatedly take a job from the input queue, encode it on the target device with the
        # given kwargs, move tensor (or dict of tensor) results to CPU, and put (chunk id, result) on the results queue.
        # KNOWN ISSUE: The queue get call blocks without a timeout, so the queue.Empty handler that should end the loop is never reached; the worker only stops when its process is terminated.
        """Internal working process to encode inputs in multi-process setup.

        Workers are terminated externally via ``stop_multi_process_pool``.
        """
        while True:
            try:
                chunk_id, inputs, kwargs = input_queue.get()
                embeddings = model.encode(inputs, device=target_device, **kwargs)
                # Move embeddings to CPU if needed
                if isinstance(embeddings, torch.Tensor) and embeddings.device.type != "cpu":
                    embeddings = embeddings.cpu()
                elif isinstance(embeddings, dict):
                    embeddings = {
                        key: value.cpu() if isinstance(value, torch.Tensor) and value.device.type != "cpu" else value
                        for key, value in embeddings.items()
                    }
                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break

    def set_pooling_include_prompt(self, include_prompt: bool) -> None:
        # [FUNCTION DOC] Set include_prompt on the first Pooling module found in the model (no-op if none).
        """
        Sets the `include_prompt` attribute in the pooling layer in the model, if there is one.

        This is useful for INSTRUCTOR models, as the prompt should be excluded from the pooling strategy
        for these models.
        """
        for module in self:
            if isinstance(module, Pooling):
                module.include_prompt = include_prompt
                break

    @deprecated("The `get_sentence_features` method is deprecated and will be removed in a future version.")
    def get_sentence_features(self, *features) -> dict[str, Tensor]:
        # [FUNCTION DOC] Deprecated: delegate to the first module's get_sentence_features with the given arguments.
        return self[0].get_sentence_features(*features)

    def get_embedding_dimension(self) -> int | None:
        # [FUNCTION DOC] Return the output embedding dimension: scanning modules from last to first, use the first non-None value of an
        # embedding-dimension getter (new name, then legacy sentence or word embedding names). If truncate_dim is set, return
        # the minimum of the two (or truncate_dim if unknown). Returns None if nothing is known.
        """
        Returns the number of dimensions in the output of :meth:`SentenceTransformer.encode`.

        Returns:
            Optional[int]: The number of dimensions in the output of `encode`. If it's not known, it's `None`.
        """
        output_dim = None
        for module in reversed(self._modules.values()):
            for name in (
                "get_embedding_dimension",
                "get_sentence_embedding_dimension",
                "get_word_embedding_dimension",
            ):
                method = getattr(module, name, None)
                if callable(method):
                    output_dim = method()
                    break
            if output_dim is not None:
                break
        if self.truncate_dim is not None:
            if output_dim is None:
                return self.truncate_dim
            return min(output_dim, self.truncate_dim)
        return output_dim

    @deprecated(
        "The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.",
        category=FutureWarning,
    )
    def get_sentence_embedding_dimension(self) -> int | None:
        # [FUNCTION DOC] Deprecated alias (FutureWarning) for get_embedding_dimension.
        return self.get_embedding_dimension()

    @contextmanager
    def truncate_embeddings(self, truncate_dim: int | None) -> Iterator[None]:
        # [FUNCTION DOC] Context manager that temporarily sets truncate_dim to the given value and restores the previous value on exit,
        # even on errors.
        """
        In this context, :meth:`SentenceTransformer.encode` outputs
        embeddings truncated at dimension ``truncate_dim``.

        This may be useful when you are using the same model for different applications where different dimensions
        are needed.

        Args:
            truncate_dim (int, optional): The dimension to truncate embeddings to. ``None`` does no truncation.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

                with model.truncate_embeddings(truncate_dim=16):
                    embeddings_truncated = model.encode(["hello there", "hiya"])
                assert embeddings_truncated.shape[-1] == 16
        """
        original_output_dim = self.truncate_dim
        try:
            self.truncate_dim = truncate_dim
            yield
        finally:
            self.truncate_dim = original_output_dim

    @contextmanager
    @deprecated(
        "The `truncate_sentence_embeddings` method has been renamed to `truncate_embeddings`.",
        category=FutureWarning,
    )
    def truncate_sentence_embeddings(self, truncate_dim: int | None) -> Iterator[None]:
        # [FUNCTION DOC] Deprecated alias (FutureWarning) for the truncate_embeddings context manager.
        with self.truncate_embeddings(truncate_dim):
            yield

    @staticmethod
    @deprecated("SentenceTransformer.load(...) is deprecated, use SentenceTransformer(...) instead.")
    def load(input_path: str) -> SentenceTransformer:
        # [FUNCTION DOC] Deprecated: construct and return a SentenceTransformer from the given path.
        """Deprecated: Use SentenceTransformer(input_path) instead."""
        return SentenceTransformer(input_path)

    def _load_default_modules(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[nn.Module] | OrderedDict[str, nn.Module], dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Build default modules for a plain Hugging Face checkpoint. Merges token, trust_remote_code, revision and
        # local_files_only into model, processor and config kwargs (user values win), creates a Transformer module (with the
        # cache dir and backend), and if it outputs token embeddings appends a Pooling module: last-token pooling for causal
        # LM architectures (unless the config says is_causal is false), mean pooling otherwise. Unless loading local files
        # only, records the base model in the model card data. Returns the module list and an empty extra-config dict.
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules, except for
        CausalLM-based models which use Last Token pooling instead.

        This is used as a fallback when no pre-trained SentenceTransformer model is found.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the model. Defaults to None.
            processor_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the processor/tokenizer. Defaults to None.
            config_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the config. Defaults to None.

        Returns:
            tuple[list[nn.Module] | OrderedDict[str, nn.Module], dict[str, Any]]: A tuple of (modules, config).
        """
        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = {**shared_kwargs} if model_kwargs is None else {**shared_kwargs, **model_kwargs}
        processor_kwargs = {**shared_kwargs} if processor_kwargs is None else {**shared_kwargs, **processor_kwargs}
        config_kwargs = {**shared_kwargs} if config_kwargs is None else {**shared_kwargs, **config_kwargs}

        transformer_model = Transformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
            backend=self.backend,
        )
        modules = [transformer_model]
        if transformer_model.module_output_name == "token_embeddings":
            config = transformer_model.config
            # If a model was originally designed for causal language modeling, then we use last token pooling,
            # except if is_causal=False, then it's still bidirectional and we default to mean pooling.
            is_causal_lm = (
                getattr(config, "architectures", None)
                and config.architectures[0].endswith("ForCausalLM")
                and getattr(config, "is_causal", True)
            )
            pooling_mode = "lasttoken" if is_causal_lm else "mean"
            modules.append(Pooling(transformer_model.get_embedding_dimension(), pooling_mode))
        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules, {}

    def _parse_model_config(self, model_config: dict[str, Any]) -> None:
        # [FUNCTION DOC] Parse the saved model config via the base class; adopt the saved similarity_fn_name if none was set and the saved
        # name is supported; adopt the saved truncate_dim if none was set.
        super()._parse_model_config(model_config)
        # Inherit a supported saved similarity_fn_name unless the user overrode it (multi-vector
        # names like "maxsim", e.g. from converted MultiVectorEncoder saves, fall through).
        saved_similarity = model_config.get("similarity_fn_name")
        if self._similarity_fn_name is None and saved_similarity in self.SUPPORTED_SIMILARITY_FN_NAMES:
            self.similarity_fn_name = saved_similarity
        if self.truncate_dim is None:
            self.truncate_dim = model_config.get("truncate_dim", None)

    def _get_model_config(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the base config extended with similarity_fn_name and, when set, truncate_dim.
        config = super()._get_model_config() | {
            "similarity_fn_name": self.similarity_fn_name,
        }
        if self.truncate_dim is not None:
            config["truncate_dim"] = self.truncate_dim
        return config

    def _push_to_hub_usage_tip(self, repo_id: str) -> str:
        # [FUNCTION DOC] Return a markdown usage snippet for the Hub push message showing how to load the pushed repo with this class and
        # the current backend.
        class_name = self.__class__.__name__
        backend = self.get_backend()
        return f"""\
## Testing this pull request
You can test this pull request before merging by loading the model from this PR with the `revision` argument:
```python
from sentence_transformers import {class_name}

# NOTE: Update this to the number of your pull request
pr_number = 2
model = {class_name}(
    "{repo_id}",
    revision=f"refs/pr/{{pr_number}}",
    backend="{backend}",
)

# Verify that everything works as expected
embeddings = model.encode(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```

---
*This PR was auto-generated with \
[`push_to_hub`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.push_to_hub).*
"""
