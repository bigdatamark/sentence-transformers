# [MODULE DOC] Deprecated legacy dataset that simply wraps a list of InputExamples.
"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from sentence_transformers.sentence_transformer.readers.input_example import InputExample

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer


class SentencesDataset(Dataset):
    # [CLASS DOC] HELPER: Thin map-style dataset over a list of examples. Attribute: examples.
    """
    DEPRECATED: This class is no longer used. Instead of wrapping your List of InputExamples in a SentencesDataset
    and then passing it to the DataLoader, you can pass the list of InputExamples directly to the dataset loader.
    """

    def __init__(self, examples: list[InputExample], model: SentenceTransformer):
        # [FUNCTION DOC] Store the examples; the model argument is accepted for backwards compatibility and ignored.
        self.examples = examples

    def __getitem__(self, item):
        # [FUNCTION DOC] Return the example at the given index.
        return self.examples[item]

    def __len__(self):
        # [FUNCTION DOC] Return the number of examples.
        return len(self.examples)
