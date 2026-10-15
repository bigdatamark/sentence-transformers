# [PACKAGE DOC]
# Training losses for CrossEncoder models. Pointwise losses: binary cross entropy, multi-class cross
# entropy, MSE (distillation). Pairwise: margin MSE, RankNet. Contrastive: multiple negatives ranking and
# its cached variant. Listwise: ListNet, ListMLE, position-aware PListMLE (with its lambda weight helper),
# LambdaLoss with its weighting schemes (none, NDCG-Loss1, NDCG-Loss2, LambdaRank, NDCG-Loss2++), and
# ADR-MSE. Only re-exports; no side effects.
from __future__ import annotations

from .adr_mse import ADRMSELoss
from .binary_cross_entropy import BinaryCrossEntropyLoss
from .cached_multiple_negatives_ranking import CachedMultipleNegativesRankingLoss
from .cross_entropy import CrossEntropyLoss
from .lambda_loss import (
    LambdaLoss,
    LambdaRankScheme,
    NDCGLoss1Scheme,
    NDCGLoss2PPScheme,
    NDCGLoss2Scheme,
    NoWeightingScheme,
)
from .list_mle import ListMLELoss
from .list_net import ListNetLoss
from .margin_mse import MarginMSELoss
from .mse import MSELoss
from .multiple_negatives_ranking import MultipleNegativesRankingLoss
from .plist_mle import PListMLELambdaWeight, PListMLELoss
from .rank_net import RankNetLoss

__all__ = [
    "ADRMSELoss",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "MultipleNegativesRankingLoss",
    "CachedMultipleNegativesRankingLoss",
    "MarginMSELoss",
    "MSELoss",
    "ListNetLoss",
    "ListMLELoss",
    "PListMLELoss",
    "PListMLELambdaWeight",
    "LambdaLoss",
    "NoWeightingScheme",
    "NDCGLoss1Scheme",
    "NDCGLoss2Scheme",
    "LambdaRankScheme",
    "NDCGLoss2PPScheme",
    "RankNetLoss",
]
