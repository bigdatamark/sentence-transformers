# [MODULE DOC] Deprecated legacy data loader that builds batches without duplicate texts, intended for in-batch
# negatives losses such as MultipleNegativesRankingLoss.
"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

In particular, you can pass "no_duplicates" to `batch_sampler` in the `SentenceTransformerTrainingArguments` class.
"""

from __future__ import annotations

import math
import random


class NoDuplicatesDataLoader:
    # [CLASS DOC] Iterable data loader over a list of InputExample-like objects (each with a texts list). Attributes:
    # batch_size, data_pointer (current position in the example list), collate_fn (set externally by the
    # legacy fit loop; None means yield raw lists) and train_examples (the list, shuffled in place).
    def __init__(self, train_examples, batch_size):
        # [FUNCTION DOC] Store the batch size and examples, reset the pointer to 0, set collate_fn to None, and shuffle the
        # example list in place (mutates the caller's list, uses the global random module).
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        """
        self.batch_size = batch_size
        self.data_pointer = 0
        self.collate_fn = None
        self.train_examples = train_examples
        random.shuffle(self.train_examples)

    def __iter__(self):
        # [FUNCTION DOC] Yield len(self) batches. For each batch, walk examples from the pointer, skipping any example whose
        # texts (stringified, stripped, lowercased) collide with a text already in the batch; accepted examples add
        # their texts to the seen set. The pointer advances for each examined example and wraps to 0 with a
        # reshuffle at the end of the list. Yields collate_fn(batch) if collate_fn is set, else the raw list.
        # KNOWN ISSUE: If fewer than batch_size mutually non-duplicate examples exist, the inner loop never terminates.
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                example = self.train_examples[self.data_pointer]

                valid_example = True
                for text in example.texts:
                    if not isinstance(text, str):
                        text = str(text)
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch.append(example)
                    for text in example.texts:
                        if not isinstance(text, str):
                            text = str(text)
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.train_examples):
                    self.data_pointer = 0
                    random.shuffle(self.train_examples)

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        # [FUNCTION DOC] Return the number of full batches: floor of example count divided by batch size.
        return math.floor(len(self.train_examples) / self.batch_size)
