# [PACKAGE DOC] Loss functions for training SentenceTransformer (dense embedding) models. Imports CoSENTLoss first (AnglELoss
# depends on it), then re-exports all loss classes plus related helpers: distance metric enums for contrastive,
# triplet and batch-hard triplet losses, and the contrastive tension data loader. Each loss is a torch module
# called with a list of tokenized feature dicts (one per input column) and a labels tensor, returning a scalar loss.
# CoSENTLoss must be imported before AnglELoss
from __future__ import annotations

from .cosent import CoSENTLoss  # isort: skip

from .adaptive_layer import AdaptiveLayerLoss
from .angle import AnglELoss
from .batch_all_triplet import BatchAllTripletLoss
from .batch_hard_soft_margin_triplet import BatchHardSoftMarginTripletLoss
from .batch_hard_triplet import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from .batch_semi_hard_triplet import BatchSemiHardTripletLoss
from .cached_gist_embed import CachedGISTEmbedLoss
from .cached_multiple_negatives_ranking import CachedMultipleNegativesRankingLoss
from .cached_multiple_negatives_symmetric_ranking import CachedMultipleNegativesSymmetricRankingLoss
from .contrastive import ContrastiveLoss, SiameseDistanceMetric
from .contrastive_tension import (
    ContrastiveTensionDataLoader,
    ContrastiveTensionLoss,
    ContrastiveTensionLossInBatchNegatives,
)
from .cosine_similarity import CosineSimilarityLoss
from .denoising_auto_encoder import DenoisingAutoEncoderLoss
from .distill_kl_div import DistillKLDivLoss
from .embed_distill import EmbedDistillLoss
from .gist_embed import GISTEmbedLoss
from .global_orthogonal_regularization import GlobalOrthogonalRegularizationLoss
from .margin_mse import MarginMSELoss
from .matryoshka import MatryoshkaLoss
from .matryoshka_2d import Matryoshka2dLoss
from .mega_batch_margin import MegaBatchMarginLoss
from .mse import MSELoss
from .multiple_negatives_ranking import MultipleNegativesRankingLoss
from .multiple_negatives_symmetric_ranking import MultipleNegativesSymmetricRankingLoss
from .online_contrastive import OnlineContrastiveLoss
from .softmax import SoftmaxLoss
from .triplet import TripletDistanceMetric, TripletLoss

__all__ = [
    "AdaptiveLayerLoss",
    "CosineSimilarityLoss",
    "SoftmaxLoss",
    "MultipleNegativesRankingLoss",
    "MultipleNegativesSymmetricRankingLoss",
    "TripletLoss",
    "TripletDistanceMetric",
    "MarginMSELoss",
    "MatryoshkaLoss",
    "Matryoshka2dLoss",
    "MSELoss",
    "ContrastiveLoss",
    "SiameseDistanceMetric",
    "CachedGISTEmbedLoss",
    "CachedMultipleNegativesRankingLoss",
    "CachedMultipleNegativesSymmetricRankingLoss",
    "ContrastiveTensionLoss",
    "ContrastiveTensionLossInBatchNegatives",
    "ContrastiveTensionDataLoader",
    "CoSENTLoss",
    "AnglELoss",
    "OnlineContrastiveLoss",
    "MegaBatchMarginLoss",
    "DenoisingAutoEncoderLoss",
    "GISTEmbedLoss",
    "GlobalOrthogonalRegularizationLoss",
    "BatchHardTripletLoss",
    "BatchHardTripletLossDistanceFunction",
    "BatchHardSoftMarginTripletLoss",
    "BatchSemiHardTripletLoss",
    "BatchAllTripletLoss",
    "DistillKLDivLoss",
    "EmbedDistillLoss",
]
