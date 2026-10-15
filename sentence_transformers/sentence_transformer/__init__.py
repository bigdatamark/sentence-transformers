# [PACKAGE DOC] Sentence Transformer package: the dense single-vector embedding model and its training stack.
# Re-exports the SentenceTransformer model, its model card data class, data collator, trainer and training
# arguments, and lists them in the public names list. Importing has no side effects beyond loading these modules.
from __future__ import annotations

from .data_collator import SentenceTransformerDataCollator
from .model import SentenceTransformer
from .model_card import SentenceTransformerModelCardData
from .trainer import SentenceTransformerTrainer
from .training_args import SentenceTransformerTrainingArguments

__all__ = [
    "SentenceTransformer",
    "SentenceTransformerModelCardData",
    "SentenceTransformerDataCollator",
    "SentenceTransformerTrainer",
    "SentenceTransformerTrainingArguments",
]
