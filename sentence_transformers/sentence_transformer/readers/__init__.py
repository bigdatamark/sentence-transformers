# [PACKAGE DOC]
# Deprecated data readers for the legacy fit-style (v2) training loop. Re-exports InputExample and
# the file readers (LabelSentenceReader, NLIDataReader, PairedFilesReader, STSDataReader,
# STSBenchmarkDataReader, TripletReader), each of which reads a dataset from disk and returns a
# list of InputExample objects. Kept only for backwards compatibility.
"""
This directory contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.
"""

from __future__ import annotations

from .input_example import InputExample
from .label_sentence import LabelSentenceReader
from .nli_data import NLIDataReader
from .paired_files import PairedFilesReader
from .sts_data import STSBenchmarkDataReader, STSDataReader
from .triplet import TripletReader

__all__ = [
    "InputExample",
    "LabelSentenceReader",
    "NLIDataReader",
    "PairedFilesReader",
    "STSDataReader",
    "STSBenchmarkDataReader",
    "TripletReader",
]
