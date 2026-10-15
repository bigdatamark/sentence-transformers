# [PACKAGE DOC] Deprecated dataset helpers for the legacy fit-style (v2.x) training loop. Re-exports the denoising
# auto-encoder dataset, the no-duplicates data loader, the parallel sentences dataset, the sentence-label
# dataset and the plain sentences dataset.
"""
This directory contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.
"""

from __future__ import annotations

from .denoising_auto_encoder import DenoisingAutoEncoderDataset
from .no_duplicates_dataloader import NoDuplicatesDataLoader
from .parallel_sentences import ParallelSentencesDataset
from .sentence_label import SentenceLabelDataset
from .sentences import SentencesDataset

__all__ = [
    "DenoisingAutoEncoderDataset",
    "NoDuplicatesDataLoader",
    "ParallelSentencesDataset",
    "SentencesDataset",
    "SentenceLabelDataset",
]
