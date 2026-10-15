# [PACKAGE DOC]
# Sparse encoder package: models that map inputs to high-dimensional sparse vectors (typically
# vocabulary-sized, as in SPLADE or CSR) for lexical-style retrieval. Re-exports the SparseEncoder
# model, its data collator, model card data, trainer and training arguments. Subpackages hold
# callbacks, evaluators, losses and modules.
from __future__ import annotations

from .data_collator import SparseEncoderDataCollator
from .model import SparseEncoder
from .model_card import SparseEncoderModelCardData
from .trainer import SparseEncoderTrainer
from .training_args import SparseEncoderTrainingArguments

__all__ = [
    "SparseEncoder",
    "SparseEncoderDataCollator",
    "SparseEncoderModelCardData",
    "SparseEncoderTrainer",
    "SparseEncoderTrainingArguments",
]
