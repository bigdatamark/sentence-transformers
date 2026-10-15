# [MODULE DOC]
# Deprecated reader for the SNLI / MultiNLI datasets stored as gzipped parallel line files.
"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

Instead, you should create a `datasets` `Dataset` for training: https://huggingface.co/docs/datasets/create_dataset
"""

from __future__ import annotations

import gzip
import os

from . import InputExample


class NLIDataReader:
    # [CLASS DOC]
    # Reader for NLI data in dataset_folder with files s1.<split>, s2.<split> and labels.<split>.
    # Labels are contradiction (0), entailment (1) and neutral (2).
    """Reads in the Stanford NLI dataset and the MultiGenre NLI dataset"""

    def __init__(self, dataset_folder):
        # [FUNCTION DOC]
        # Stores the dataset folder.
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0):
        # [FUNCTION DOC]
        # Reads the three gzipped files named by prefixing s1., s2. and labels. to filename, pairs lines
        # in order and creates an InputExample with guid "filename-index", the two sentences and the
        # mapped label. Stops after max_examples if positive. Returns the list. Raises KeyError on an
        # unknown label.
        # KNOWN ISSUE: Sentences keep their trailing newline characters, and the gzip files are never closed.
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        s1 = gzip.open(os.path.join(self.dataset_folder, "s1." + filename), mode="rt", encoding="utf-8").readlines()
        s2 = gzip.open(os.path.join(self.dataset_folder, "s2." + filename), mode="rt", encoding="utf-8").readlines()
        labels = gzip.open(
            os.path.join(self.dataset_folder, "labels." + filename), mode="rt", encoding="utf-8"
        ).readlines()

        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            guid = f"{filename}-{id}"
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        # [FUNCTION DOC]
        # Static method returning the label-name-to-id mapping.
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        # [FUNCTION DOC]
        # Returns the number of labels (3).
        return len(self.get_labels())

    def map_label(self, label):
        # [FUNCTION DOC]
        # Maps a label string (stripped and lowercased) to its integer id; raises KeyError if unknown.
        return self.get_labels()[label.strip().lower()]
