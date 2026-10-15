# [PACKAGE DOC]
# Late-interaction scoring functions for multi-vector embeddings. Re-exports ColBERT scoring (MaxSim summed
# over query tokens) in batched-all-pairs, pairwise and knowledge-distillation variants, mean-normalized
# versions of each, and XTR scoring (top-k retrieval-restricted MaxSim) as both classes and functions,
# including pairwise and distillation forms. Only re-exports; no side effects.
from __future__ import annotations

from .colbert import (
    colbert_kd_scores,
    colbert_scores,
    colbert_scores_pairwise,
    mean_colbert_kd_scores,
    mean_colbert_scores,
    mean_colbert_scores_pairwise,
)
from .xtr import XTRKDScores, XTRScores, xtr_kd_scores, xtr_scores, xtr_scores_pairwise

__all__ = [
    "colbert_scores",
    "colbert_scores_pairwise",
    "colbert_kd_scores",
    "mean_colbert_scores",
    "mean_colbert_scores_pairwise",
    "mean_colbert_kd_scores",
    "XTRScores",
    "XTRKDScores",
    "xtr_scores",
    "xtr_scores_pairwise",
    "xtr_kd_scores",
]
