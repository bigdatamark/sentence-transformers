# [MODULE DOC]
# Triplet accuracy evaluator adapted for MultiVectorEncoder models, reusing the single-vector
# TripletEvaluator's loop and metrics but swapping in late-interaction similarity functions and the
# query/document encode methods.
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from sentence_transformers.sentence_transformer.evaluation.triplet import TripletEvaluator
from sentence_transformers.util.similarity import SimilarityFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


class MultiVectorTripletEvaluator(TripletEvaluator):
    # [CLASS DOC]
    # Subclass of the sentence-transformer TripletEvaluator. For each (anchor, positive, negative) triplet,
    # counts it as correct when the anchor-positive late-interaction score exceeds the anchor-negative score
    # plus a margin. Anchors are encoded as queries, positives/negatives as documents. Similarity functions
    # offered are the model's supported multi-vector names (maxsim, meanmaxsim); margins may be per-function.
    # Embedding truncation is rejected.
    """Triplet evaluator for :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` models.

    Given ``(anchor, positive, negative)`` triplets, checks how often
    ``MaxSim(anchor, positive) > MaxSim(anchor, negative) + margin``. The anchors are encoded via
    :meth:`encode_query` (with the query prefix and length). Positives and negatives are encoded via
    :meth:`encode_document`. ``truncate_dim`` is not supported (multi-vector token embeddings have no
    Matryoshka-style truncation) and raises a ValueError.

    A ``margin`` dictionary must be keyed by a supported similarity, ``maxsim`` or ``meanmaxsim`` (a
    float applies to both). The two need different margins: MaxSim sums a per-token maximum over the
    query tokens, so scores grow with the query length (roughly one point per query token for
    normalized embeddings), while MeanMaxSim divides that by the token count and sits in cosine's
    ``[-1, 1]`` range.

    Example:
        ::

            from datasets import load_dataset

            from sentence_transformers import MultiVectorEncoder
            from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorTripletEvaluator

            model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

            dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

            evaluator = MultiVectorTripletEvaluator(
                anchors=dataset[:1000]["anchor"],
                positives=dataset[:1000]["positive"],
                negatives=dataset[:1000]["negative"],
                name="all-nli-dev",
            )
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Accept the same arguments as TripletEvaluator. Raises ValueError if a non-None truncation dimension is
        # passed as a keyword; otherwise delegates entirely to the parent constructor.
        if kwargs.get("truncate_dim") is not None:
            raise ValueError(
                "truncate_dim is not supported by MultiVectorEncoder evaluators: multi-vector token "
                "embeddings have no Matryoshka-style truncation. Remove truncate_dim to evaluate at "
                "the full dimension."
            )
        super().__init__(*args, **kwargs)

    def _get_similarity_functions(self) -> dict:
        # [FUNCTION DOC]
        # HELPER: Build the mapping from similarity name to a scoring callable. For each name in the model class's
        # supported similarity list, looks up the pairwise similarity function and wraps it into a callable that
        # takes anchor, positive and negative embeddings and returns the pair of anchor-positive and
        # anchor-negative score vectors. Imports the model class lazily to avoid an import cycle.
        # All supported multi-vector similarities. The parent picks via similarity_fn_names (default: the model's).
        from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

        functions = {}
        for name in MultiVectorEncoder.SUPPORTED_SIMILARITY_FN_NAMES:
            pairwise = SimilarityFunction.to_similarity_pairwise_fn(name)
            functions[name] = lambda a, p, n, pairwise=pairwise: (pairwise(a, p), pairwise(a, n))
        return functions

    def embed_inputs(
        self,
        model: MultiVectorEncoder,
        sentences: SingleInput | Sequence[SingleInput] | np.ndarray,
        encode_fn_name: str | None = None,
        **kwargs,
    ) -> list[Tensor]:
        # [FUNCTION DOC]
        # Encode inputs with the model. The encode function name selects the query encoder ("query"), the document
        # encoder ("document"), or the generic encoder (anything else). Passes the evaluator's batch size and
        # progress-bar setting plus extra kwargs. Returns the per-input token embedding tensors.
        if encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        else:
            encode_fn = model.encode
        return encode_fn(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            **kwargs,
        )
