# [PACKAGE DOC]
# sentence_transformers.base: model-type-agnostic building blocks shared by the SentenceTransformer, SparseEncoder,
# CrossEncoder and MultiVectorEncoder families: BaseModel, BaseDataCollator, BaseTrainer, BaseTrainingArguments with the
# BatchSamplers/MultiDatasetBatchSamplers enums, PeftAdapterMixin, and the batch samplers (default, group-by-label,
# no-duplicates) and multi-dataset samplers (default, round-robin, proportional). It only re-exports these names (__all__).
from __future__ import annotations

from .data_collator import BaseDataCollator
from .model import BaseModel
from .peft_mixin import PeftAdapterMixin
from .sampler import (
    DefaultBatchSampler,
    GroupByLabelBatchSampler,
    MultiDatasetDefaultBatchSampler,
    NoDuplicatesBatchSampler,
    ProportionalBatchSampler,
    RoundRobinBatchSampler,
)
from .trainer import BaseTrainer
from .training_args import BaseTrainingArguments, BatchSamplers, MultiDatasetBatchSamplers

__all__ = [
    "BaseModel",
    "BaseDataCollator",
    "BaseTrainer",
    "BaseTrainingArguments",
    "BatchSamplers",
    "MultiDatasetBatchSamplers",
    "PeftAdapterMixin",
    "DefaultBatchSampler",
    "GroupByLabelBatchSampler",
    "NoDuplicatesBatchSampler",
    "MultiDatasetDefaultBatchSampler",
    "RoundRobinBatchSampler",
    "ProportionalBatchSampler",
]
