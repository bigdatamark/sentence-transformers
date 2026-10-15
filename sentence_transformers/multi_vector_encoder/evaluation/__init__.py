# [PACKAGE DOC]
# Evaluators for MultiVectorEncoder models: information retrieval over a corpus, the NanoBEIR benchmark
# collection, triplet accuracy, distillation agreement against teacher scores, and reranking of candidate
# lists. Each scores queries against documents with the model's late-interaction similarity.
# Only re-exports; no side effects.
from __future__ import annotations

from .distillation import MultiVectorDistillationEvaluator
from .information_retrieval import MultiVectorInformationRetrievalEvaluator
from .nano_beir import MultiVectorNanoBEIREvaluator
from .reranking import MultiVectorRerankingEvaluator
from .triplet import MultiVectorTripletEvaluator

__all__ = [
    "MultiVectorInformationRetrievalEvaluator",
    "MultiVectorNanoBEIREvaluator",
    "MultiVectorTripletEvaluator",
    "MultiVectorDistillationEvaluator",
    "MultiVectorRerankingEvaluator",
]
