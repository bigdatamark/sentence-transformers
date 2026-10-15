# [PACKAGE DOC]
# The multi-vector encoder package: models that encode each text into one embedding per token (rather
# than a single pooled vector) and compare texts with late interaction, typically MaxSim (for each query
# token take the max similarity over document tokens, then sum). Re-exports the MultiVectorEncoder model,
# its data collator, model card data, trainer, training arguments, and the maxsim / maxsim_pairwise
# similarity functions from the shared util.similarity module. Only re-exports; no side effects.
from __future__ import annotations

from sentence_transformers.util.similarity import maxsim, maxsim_pairwise

from .data_collator import MultiVectorEncoderDataCollator
from .model import MultiVectorEncoder
from .model_card import MultiVectorEncoderModelCardData
from .trainer import MultiVectorEncoderTrainer
from .training_args import MultiVectorEncoderTrainingArguments

__all__ = [
    "MultiVectorEncoder",
    "MultiVectorEncoderDataCollator",
    "MultiVectorEncoderModelCardData",
    "MultiVectorEncoderTrainer",
    "MultiVectorEncoderTrainingArguments",
    "maxsim",
    "maxsim_pairwise",
]
