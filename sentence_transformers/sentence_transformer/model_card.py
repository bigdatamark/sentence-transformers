# [MODULE DOC]
# Model card support for dense SentenceTransformer models. Defines the dataclass that collects
# metadata (languages, license, datasets, tags, widget examples, similarity scores) used to render
# the Hugging Face model card README from a template, plus the model-card trainer callback and a
# deprecated alias for that callback. Builds on the generic BaseModelCardData / BaseModelCardCallback.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import deprecated

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.sentence_transformer.modules import StaticEmbedding

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer


class SentenceTransformerModelCardCallback(BaseModelCardCallback):
    # [CLASS DOC]
    # Trainer callback that updates the model card data during training. Adds nothing beyond the
    # generic BaseModelCardCallback; it exists so dense models have a type-specific callback name.
    pass


@dataclass
class SentenceTransformerModelCardData(BaseModelCardData):
    # [CLASS DOC]
    # Dataclass holding model card metadata for a SentenceTransformer. Extends BaseModelCardData with
    # dense-specific defaults: a task_name describing similarity/search/clustering tasks, default tags
    # (sentence-transformers, sentence-similarity, feature-extraction, dense), a fixed template_path
    # pointing at model_card_template.md next to this file (not an init argument), and a model
    # reference (not an init argument) that is set later when the card is registered with a model.
    # Overrides base-model detection, usage-example generation, model-specific metadata and the usage
    # code snippet.
    """A dataclass storing data used in the model card.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g. "SentenceTransformer based on microsoft/mpnet-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/sbert-mpnet-base-allnli".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more".
        tags (`Optional[List[str]]`): A list of tags for the model,
            e.g. ["sentence-transformers", "sentence-similarity", "feature-extraction"].
        local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.
            Defaults to False.
        generate_widget_examples (`bool`): If True, generate widget examples from the evaluation or training dataset,
            and compute their similarities. Defaults to True.

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.

    Example::

        >>> model = SentenceTransformer(
        ...     "microsoft/mpnet-base",
        ...     model_card_data=SentenceTransformerModelCardData(
        ...         model_id="tomaarsen/sbert-mpnet-base-allnli",
        ...         train_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         eval_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         license="apache-2.0",
        ...         language="en",
        ...     ),
        ... )
    """

    # Potentially provided by the user
    task_name: str = (
        "semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more"
    )
    tags: list[str] = field(
        default_factory=lambda: [
            "sentence-transformers",
            "sentence-similarity",
            "feature-extraction",
            "dense",
        ]
    )

    # Computed once, always unchanged
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False, repr=False)

    # Passed via `register_model` only
    model: SentenceTransformer | None = field(default=None, init=False, repr=False)

    def try_to_set_base_model(self):
        # [FUNCTION DOC]
        # Determines the base model of the card. Runs the generic detection first; then, if the first
        # module of the model is a StaticEmbedding and no base model was found, uses that module's
        # base_model attribute (when set) as the base model via set_base_model. Mutates the card data.
        super().try_to_set_base_model()
        if isinstance(self.model[0], StaticEmbedding) and self.base_model is None:
            if self.model[0].base_model:
                self.set_base_model(self.model[0].base_model)

    def run_usage_snippet(self) -> None:
        # [FUNCTION DOC]
        # Fills in usage examples and computes example similarity scores for the card.
        # If no usage examples are set, picks defaults: for an information-retrieval model (ir_model) one
        # query plus three documents about the Red Planet, otherwise three generic sentences.
        # Returns early if generate_widget_examples is false. Otherwise prepares each example for
        # inference (e.g. converts video decoders), then: for an IR model encodes the first example as a
        # query and the rest as documents and computes the query-document similarity matrix; for other
        # models truncates examples to the first three and computes the pairwise similarity matrix of
        # their embeddings. Stores the matrix as a string (4-decimal precision, no scientific notation),
        # each line prefixed with a hash and space, in self.similarities. Runs model inference and
        # mutates usage_examples and similarities; returns nothing.
        if self.usage_examples is None:
            if self.ir_model:
                self.usage_examples = [
                    "Which planet is known as the Red Planet?",
                    "Venus is often called Earth's twin because of its similar size and proximity.",
                    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
                ]
            else:
                self.usage_examples = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "He drove to the stadium.",
                ]

        if not self.generate_widget_examples:
            return

        # Convert VideoDecoder objects to VideoDict so they can be processed
        prepared_examples = [self._prepare_for_inference(item) for item in self.usage_examples]

        if self.ir_model:
            query_embeddings = self.model.encode_query(
                prepared_examples[0], convert_to_tensor=True, show_progress_bar=False
            )
            document_embeddings = self.model.encode_document(
                prepared_examples[1:], convert_to_tensor=True, show_progress_bar=False
            )
            similarity = self.model.similarity(query_embeddings, document_embeddings)
        else:
            self.usage_examples = self.usage_examples[:3]  # Limit to 3 examples for standard similarity
            prepared_examples = prepared_examples[:3]
            embeddings = self.model.encode(prepared_examples, convert_to_tensor=True, show_progress_bar=False)
            similarity = self.model.similarity(embeddings, embeddings)

        with torch._tensor_str.printoptions(precision=4, sci_mode=False):
            self.similarities = "\n".join(f"# {line}" for line in str(similarity.cpu()).splitlines())

    def get_model_specific_metadata(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns the metadata dict from the base class extended with output_dimensionality (the model's
        # embedding dimension) and similarity_fn_name, a human-readable name: cosine, dot, euclidean,
        # manhattan map to fixed names; any other name has underscores replaced by spaces and is title
        # cased; defaults to "Cosine Similarity" when the model has no similarity function name.
        metadata = super().get_model_specific_metadata()
        similarity_fn_name = "Cosine Similarity"
        if self.model.similarity_fn_name:
            similarity_fn_name = {
                "cosine": "Cosine Similarity",
                "dot": "Dot Product",
                "euclidean": "Euclidean Distance",
                "manhattan": "Manhattan Distance",
            }.get(self.model.similarity_fn_name, self.model.similarity_fn_name.replace("_", " ").title())
        metadata.update(
            {
                "output_dimensionality": self.model.get_embedding_dimension(),
                "similarity_fn_name": similarity_fn_name,
            }
        )
        return metadata

    def generate_usage_snippet(self) -> str:
        # [FUNCTION DOC]
        # Builds the markdown Python code block showing how to use the model. For non-IR models defers to
        # the base implementation. For IR models uses usage_examples_display, else usage_examples, else
        # the default Red Planet query/documents; the first item is the query and the rest are documents.
        # Emits code that loads the model by model_id (or a placeholder id), encodes queries and
        # documents with encode_query / encode_document, prints their shapes (with the snippet output
        # dimensionality), computes similarity, and prints either the stored similarities or the expected
        # shape. Returns the fenced code block string.
        if not self.ir_model:
            return super().generate_usage_snippet()

        display = self.usage_examples_display or self.usage_examples
        examples = display or [
            "Which planet is known as the Red Planet?",
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        ]

        model_id = self.model_id or "sentence_transformers_model_id"
        output_dim = self._get_snippet_output_dimensionality()
        num_docs = len(examples) - 1

        lines = [
            "from sentence_transformers import SentenceTransformer",
            "",
            "# Download from the 🤗 Hub",
            f'model = SentenceTransformer("{model_id}")',
            "# Run inference",
            "queries = [",
            f"    {self._format_snippet_value(examples[0])},",
            "]",
            "documents = [",
        ]
        for item in examples[1:]:
            lines.append(f"    {self._format_snippet_value(item)},")
        lines.extend(
            [
                "]",
                "query_embeddings = model.encode_query(queries)",
                "document_embeddings = model.encode_document(documents)",
                "print(query_embeddings.shape, document_embeddings.shape)",
                f"# [1, {output_dim}] [{num_docs}, {output_dim}]",
                "",
                "# Get the similarity scores for the embeddings",
                "similarities = model.similarity(query_embeddings, document_embeddings)",
            ]
        )
        if self.similarities:
            lines.append("print(similarities)")
            lines.append(self.similarities)
        else:
            lines.extend(
                [
                    "print(similarities.shape)",
                    f"# [1, {num_docs}]",
                ]
            )

        return "```python\n" + "\n".join(lines) + "\n```"


@deprecated(
    "The `ModelCardCallback` has been renamed to `SentenceTransformerModelCardCallback` and the former is now deprecated. Please use `SentenceTransformerModelCardCallback` instead."
)
class ModelCardCallback(SentenceTransformerModelCardCallback):
    # [CLASS DOC]
    # Deprecated alias of SentenceTransformerModelCardCallback; using it emits a deprecation warning.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Forwards all arguments to the parent callback constructor.
        super().__init__(*args, **kwargs)
