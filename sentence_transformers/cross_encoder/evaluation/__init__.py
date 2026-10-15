# [PACKAGE DOC]
# Evaluators for CrossEncoder models: classification (accuracy/F1/AP style metrics), correlation with gold
# similarity scores (Pearson/Spearman), reranking of candidate documents (MRR/NDCG/MAP), and the NanoBEIR
# reranking benchmark. Also re-exports deprecated legacy evaluator names (CE* classes) kept for backwards
# compatibility, which forward to the new evaluators. Only re-exports; no side effects.
from __future__ import annotations

from .classification import CrossEncoderClassificationEvaluator
from .correlation import CrossEncoderCorrelationEvaluator
from .deprecated import (
    CEBinaryAccuracyEvaluator,
    CEBinaryClassificationEvaluator,
    CECorrelationEvaluator,
    CEF1Evaluator,
    CERerankingEvaluator,
    CESoftmaxAccuracyEvaluator,
)
from .nano_beir import CrossEncoderNanoBEIREvaluator
from .reranking import CrossEncoderRerankingEvaluator

__all__ = [
    "CrossEncoderClassificationEvaluator",
    "CrossEncoderCorrelationEvaluator",
    "CrossEncoderRerankingEvaluator",
    "CrossEncoderNanoBEIREvaluator",
    # Deprecated:
    "CERerankingEvaluator",
    "CECorrelationEvaluator",
    "CEBinaryAccuracyEvaluator",
    "CEBinaryClassificationEvaluator",
    "CEF1Evaluator",
    "CESoftmaxAccuracyEvaluator",
]
