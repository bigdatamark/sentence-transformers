# [MODULE DOC] HELPER: Defines the data collator class for SentenceTransformer training.
from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.base.data_collator import BaseDataCollator


@dataclass
class SentenceTransformerDataCollator(BaseDataCollator):
    # [CLASS DOC] HELPER: Dataclass that is an empty subclass of BaseDataCollator; all tokenization/collation behavior
    # is inherited unchanged. Exists so the trainer has a model-specific collator type.
    pass
