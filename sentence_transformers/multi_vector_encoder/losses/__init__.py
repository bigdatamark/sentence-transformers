# [PACKAGE DOC]
# Training losses for MultiVectorEncoder models, all of which score query/document pairs with a
# late-interaction (MaxSim-style) scoring function over per-token embeddings.
# Re-exports: in-batch negatives ranking loss, its gradient-cached variant for large batches, a
# KL-divergence distillation loss against teacher scores, and a margin-MSE distillation loss.
# Only re-exports; no side effects.
from __future__ import annotations

from .cached_multiple_negatives_ranking import CachedMultiVectorMultipleNegativesRankingLoss
from .distill_kl_div import MultiVectorDistillKLDivLoss
from .margin_mse import MultiVectorMarginMSELoss
from .multiple_negatives_ranking import MultiVectorMultipleNegativesRankingLoss

__all__ = [
    "MultiVectorMultipleNegativesRankingLoss",
    "CachedMultiVectorMultipleNegativesRankingLoss",
    "MultiVectorDistillKLDivLoss",
    "MultiVectorMarginMSELoss",
]
