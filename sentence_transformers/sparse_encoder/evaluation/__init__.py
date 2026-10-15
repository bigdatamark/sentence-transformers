# [PACKAGE DOC]
# Evaluators for sparse encoders. Re-exports sparse variants of the embedding similarity,
# information retrieval, binary classification, MSE, NanoBEIR, triplet, translation and reranking
# evaluators, plus the ReciprocalRankFusionEvaluator for hybrid ranking. The sparse variants add
# sparsity statistics (active dimensions, sparsity ratio) to the dense evaluator metrics.
from __future__ import annotations

from sentence_transformers.sparse_encoder.evaluation.reciprocal_rank_fusion import ReciprocalRankFusionEvaluator
from sentence_transformers.sparse_encoder.evaluation.sparse_binary_classification import (
    SparseBinaryClassificationEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.sparse_embedding_similarity import (
    SparseEmbeddingSimilarityEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.sparse_information_retrieval import (
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.sparse_mse import SparseMSEEvaluator
from sentence_transformers.sparse_encoder.evaluation.sparse_nano_beir import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.evaluation.sparse_reranking import SparseRerankingEvaluator
from sentence_transformers.sparse_encoder.evaluation.sparse_translation import SparseTranslationEvaluator
from sentence_transformers.sparse_encoder.evaluation.sparse_triplet import SparseTripletEvaluator

__all__ = [
    "SparseEmbeddingSimilarityEvaluator",
    "SparseInformationRetrievalEvaluator",
    "SparseBinaryClassificationEvaluator",
    "SparseMSEEvaluator",
    "SparseNanoBEIREvaluator",
    "SparseTripletEvaluator",
    "SparseTranslationEvaluator",
    "SparseRerankingEvaluator",
    "ReciprocalRankFusionEvaluator",
]
