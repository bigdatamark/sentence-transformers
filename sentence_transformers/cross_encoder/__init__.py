# [PACKAGE DOC]
# The cross encoder package: models that take a pair (or list) of texts jointly through a transformer
# and output one or more relevance/classification scores per pair, instead of producing independent
# embeddings. Re-exports the public entry points: the CrossEncoder model class, its training data
# collator, model card data holder, the Trainer subclass, and the training arguments dataclass.
# Subpackages provide evaluators, losses and modules. Only re-exports; no side effects.
from __future__ import annotations

from .data_collator import CrossEncoderDataCollator
from .model import CrossEncoder
from .model_card import CrossEncoderModelCardData
from .trainer import CrossEncoderTrainer
from .training_args import CrossEncoderTrainingArguments

__all__ = [
    "CrossEncoder",
    "CrossEncoderDataCollator",
    "CrossEncoderModelCardData",
    "CrossEncoderTrainer",
    "CrossEncoderTrainingArguments",
]
