# [MODULE DOC]
# Data collator for sparse encoder training.
from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.base.data_collator import BaseDataCollator


@dataclass
class SparseEncoderDataCollator(BaseDataCollator):
    # [CLASS DOC]
    # Dataclass collator for SparseEncoder training. Adds nothing to BaseDataCollator (which turns
    # dataset rows into model features and labels); it exists to give sparse training its own type.
    pass
