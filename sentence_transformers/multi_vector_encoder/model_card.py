# [MODULE DOC]
# Model card support for MultiVectorEncoder: a training callback alias and a dataclass that fills in
# multi-vector-specific metadata and the usage code snippet for the generated README.
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.base.modules import Transformer

if TYPE_CHECKING:
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

logger = logging.getLogger(__name__)


class MultiVectorEncoderModelCardCallback(BaseModelCardCallback):
    # [CLASS DOC]
    # Model card training callback; identical to the base callback (which records training info into the model
    # card data), existing only as a dedicated type.
    pass


@dataclass
class MultiVectorEncoderModelCardData(BaseModelCardData):
    # [CLASS DOC]
    # Dataclass of model card data extending BaseModelCardData. Fields: task name (defaulted at registration),
    # tags (sentence-transformers, multi-vector, colbert, late-interaction), non-init usage examples and the
    # recorded query/document embedding shapes from running the snippet, pipeline tag, template path (the
    # model_card_template.md next to this module), model type "Multi-Vector Encoder", and the registered model.
    # Class-level snippet settings name the model class and a placeholder model id.
    """A dataclass storing data used in the model card for :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` models.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g.
            "MultiVectorEncoder based on answerdotai/ModernBERT-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/mve-modernbert-base-ms-marco".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training
            datasets, e.g. ``[{"name": "MS MARCO", "id": "microsoft/ms_marco"}]``.
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation
            datasets.
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic search with late interaction".
        tags (`Optional[List[str]]`): A list of tags for the model, e.g.
            ``["sentence-transformers", "multi-vector", "colbert", "late-interaction"]``.
        local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.
        generate_widget_examples (`bool`): If True, generate widget examples from the evaluation or training dataset.

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.
    """

    _snippet_model_class = "MultiVectorEncoder"
    _snippet_default_model_id = "multi_vector_encoder_model_id"

    task_name: str | None = None
    tags: list[str] = field(
        default_factory=lambda: [
            "sentence-transformers",
            "multi-vector",
            "colbert",
            "late-interaction",
        ]
    )

    usage_examples: list[list[str]] | None = field(default=None, init=False)
    usage_query_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    usage_document_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)

    pipeline_tag: str = field(default=None, init=False)
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False, repr=False)
    model_type: str = field(default="Multi-Vector Encoder", init=False, repr=False)

    model: MultiVectorEncoder | None = field(default=None, init=False, repr=False)

    def register_model(self, model: MultiVectorEncoder) -> None:
        # [FUNCTION DOC]
        # Register the model with the base class, then default the task name to "semantic search with late
        # interaction", the pipeline tag to "feature-extraction", and mark the model as an IR model if unset.
        # Mutates the dataclass.
        super().register_model(model)

        if self.task_name is None:
            self.task_name = "semantic search with late interaction"
        if self.pipeline_tag is None:
            self.pipeline_tag = "feature-extraction"
        # Late interaction is always retrieval: the base then sources widget examples positionally,
        # query from the first dataset column and documents from the second.
        if self.ir_model is None:
            self.ir_model = True

    def get_model_specific_metadata(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Return the base model-specific metadata updated with the embedding dimension and a human-readable
        # similarity name (MaxSim, MeanMaxSim, XTR, otherwise underscores to spaces in title case). If the first
        # Transformer module has a "fixed" query expansion strategy, the query length entry is overridden by the
        # expansion length. Requires a registered model.
        metadata = super().get_model_specific_metadata()
        # Humanize like the dense and sparse cards. The generic .title() fallback would render
        # "Maxsim" / "Xtr".
        similarity_fn_name = {
            "maxsim": "MaxSim",
            "meanmaxsim": "MeanMaxSim",
            "xtr": "XTR",
        }.get(self.model.similarity_fn_name, self.model.similarity_fn_name.replace("_", " ").title())
        metadata.update(
            {
                "output_dimensionality": self.model.get_embedding_dimension(),
                "similarity_fn_name": similarity_fn_name,
            }
        )
        # A "fixed" expansion pins every query to its own length, overriding the base's query_length.
        # "min" only sets a floor, so there query_length remains the cap.
        transformer = next((module for module in self.model if isinstance(module, Transformer)), None)
        query_expansion = getattr(transformer, "query_expansion", None) or {}
        if query_expansion.get("strategy") == "fixed":
            metadata["query_length"] = query_expansion["length"]
        return metadata

    def run_usage_snippet(self) -> None:
        # [FUNCTION DOC]
        # Actually run the usage example to capture outputs. Defaults usage examples to four planet sentences if
        # none are set. If widget generation is disabled, returns early. Otherwise keeps at most four examples,
        # prepares each for inference, splits into one query and the rest as documents (or uses the same single
        # example for both), encodes queries and documents (model forward passes), records the first query and
        # document embedding shapes, computes the similarity matrix, and stores it as comment-prefixed text lines
        # formatted to four decimals. Mutates the dataclass.
        if self.usage_examples is None:
            self.usage_examples = [
                "Which planet is known as the Red Planet?",
                "Venus is often called Earth's twin because of its similar size and proximity.",
                "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
            ]

        if not self.generate_widget_examples:
            return

        self.usage_examples = self.usage_examples[:4]
        prepared_examples = [self._prepare_for_inference(item) for item in self.usage_examples]
        # Mirror the positional queries / documents split of the generated snippet (first example is
        # the query, the rest are documents) so the printed similarity matrix matches what a user
        # running the snippet sees.
        if len(prepared_examples) >= 2:
            queries = prepared_examples[:1]
            documents = prepared_examples[1:]
        else:
            queries = documents = prepared_examples
        query_embeddings = self.model.encode_query(queries, show_progress_bar=False)
        doc_embeddings = self.model.encode_document(documents, show_progress_bar=False)
        self.usage_query_shape = tuple(query_embeddings[0].shape)
        self.usage_document_shape = tuple(doc_embeddings[0].shape)
        similarity = self.model.similarity(query_embeddings, doc_embeddings)

        with torch._tensor_str.printoptions(precision=4, sci_mode=False):
            self.similarities = "\n".join(f"# {line}" for line in str(similarity.cpu()).splitlines())

    def generate_usage_snippet(self) -> str:
        # [FUNCTION DOC]
        # Build the markdown usage snippet: a fenced Python code block that imports the model class, loads the
        # model id (or placeholder), defines the first example as the query list and the rest as documents (same
        # example for both if only one), encodes both, prints the recorded shapes (or symbolic shapes with the
        # output dimension), computes similarity, and prints either the recorded similarity text or the expected
        # matrix shape. Uses display examples when present, else usage examples, else the planet defaults.
        # Returns the snippet string.
        # MV encode() returns variable-length per-token lists (no stacked `.shape`), and late
        # interaction is always retrieval, so the snippet splits positionally into queries /
        # documents, matching how the base sources IR usage examples (query from the first dataset
        # column, documents from the second). Text and image documents render identically.
        display = self.usage_examples_display or self.usage_examples
        examples = display or [
            "Which planet is known as the Red Planet?",
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        ]
        model_class = self._snippet_model_class
        model_id = self.model_id or self._snippet_default_model_id
        output_dim = self._get_snippet_output_dimensionality()

        if len(examples) >= 2:
            queries = examples[:1]
            documents = examples[1:]
        else:
            queries = documents = examples
        query_shape = self.usage_query_shape or f"(num_query_tokens, {output_dim})"
        document_shape = self.usage_document_shape or f"(num_document_tokens, {output_dim})"

        lines = [
            f"from sentence_transformers import {model_class}",
            "",
            "# Download from the 🤗 Hub",
            f'model = {model_class}("{model_id}")',
            "# Run inference: each input becomes a sequence of per-token vectors (variable length).",
            "queries = [",
            *(f"    {self._format_snippet_value(item)}," for item in queries),
            "]",
            "documents = [",
            *(f"    {self._format_snippet_value(item)}," for item in documents),
            "]",
            "query_embeddings = model.encode_query(queries)",
            "document_embeddings = model.encode_document(documents)",
            "print(query_embeddings[0].shape, document_embeddings[0].shape)",
            f"# {query_shape} {document_shape}",
            "",
            "# Get the MaxSim similarity scores",
            "similarities = model.similarity(query_embeddings, document_embeddings)",
        ]
        if self.similarities:
            lines.append("print(similarities)")
            lines.append(self.similarities)
        else:
            lines.extend(["print(similarities.shape)", f"# [{len(queries)}, {len(documents)}]"])

        return "```python\n" + "\n".join(lines) + "\n```"

    def get_default_model_name(self) -> str:
        # [FUNCTION DOC]
        # Return the default model name, which is the model type string.
        return self.model_type
