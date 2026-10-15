# [MODULE DOC] Deprecated legacy iterable dataset that yields examples grouped by label, for batch triplet losses that
# need several examples with the same label in each batch.
"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

In particular, you can pass "group_by_label" to `batch_sampler` in the `SentenceTransformerTrainingArguments` class.
"""

from __future__ import annotations

import logging

import numpy as np
from torch.utils.data import IterableDataset

from sentence_transformers.sentence_transformer.readers import InputExample

logger = logging.getLogger(__name__)


class SentenceLabelDataset(IterableDataset):
    # [CLASS DOC] Iterable dataset over InputExamples with labels. Attributes: samples_per_label (how many examples are
    # yielded consecutively per label draw), grouped_inputs (examples of usable labels concatenated, grouped
    # by label), groups_right_border (end index of each label group in grouped_inputs), label_range (shuffled
    # array of label group indices), with_replacement flag.
    """
    This dataset can be used for some specific Triplet Losses like BATCH_HARD_TRIPLET_LOSS which requires
    multiple examples with the same label in a batch.

    It draws n consecutive, random and unique samples from one label at a time. This is repeated for each label.

    Labels with fewer than n unique samples are ignored.
    This also applied to drawing without replacement, once less than n samples remain for a label, it is skipped.

    This *DOES NOT* check if there are more labels than the batch is large or if the batch size is divisible
    by the samples drawn per label.
    """

    def __init__(self, examples: list[InputExample], samples_per_label: int = 2, with_replacement: bool = False):
        # [FUNCTION DOC] Group examples by label, keep only labels with at least samples_per_label examples, concatenating their
        # examples and recording each group's end index. Create and shuffle (numpy global RNG) the label-group index
        # array, store with_replacement, and log how many examples and labels were usable.
        # KNOWN ISSUE: The docstring describes with_replacement backwards: in the code, True allows an example to be drawn again in later draws, and False draws each example at most once per pass over the labels.
        """
        Creates a LabelSampler for a SentenceLabelDataset.

        Args:
            examples (List[InputExample]): A list of InputExamples.
            samples_per_label (int, optional): The number of consecutive, random, and unique samples drawn per label.
                The batch size should be a multiple of samples_per_label. Defaults to 2.
            with_replacement (bool, optional): If True, each sample is drawn at most once (depending on the total number
                of samples per label). If False, one sample can be drawn in multiple draws, but not multiple times in
                the same drawing. Defaults to False.
        """
        super().__init__()

        self.samples_per_label = samples_per_label

        # Group examples by label
        label2ex = {}
        for example in examples:
            if example.label not in label2ex:
                label2ex[example.label] = []
            label2ex[example.label].append(example)

        # Include only labels with at least 2 examples
        self.grouped_inputs = []
        self.groups_right_border = []
        num_labels = 0

        for label, label_examples in label2ex.items():
            if len(label_examples) >= self.samples_per_label:
                self.grouped_inputs.extend(label_examples)
                self.groups_right_border.append(
                    len(self.grouped_inputs)
                )  # At which position does this label group / bucket end?
                num_labels += 1

        self.label_range = np.arange(num_labels)
        self.with_replacement = with_replacement
        np.random.shuffle(self.label_range)

        logger.info(
            f"SentenceLabelDataset: {len(examples)} examples, from which {len(self.grouped_inputs)} examples could be used (those labels appeared at least {self.samples_per_label} times). {num_labels} different labels found."
        )

    def __iter__(self):
        # [FUNCTION DOC] Yield examples until as many as grouped_inputs have been yielded. Cycle through the shuffled label groups;
        # for each, the candidate pool is the whole group (with replacement) or the group's examples not yet yielded
        # in this pass (without replacement). If the pool has at least samples_per_label items, pick that many
        # distinct ones at random and yield them, marking them seen. After the last label, reset seen sets and
        # reshuffle the labels.
        label_idx = 0
        count = 0
        already_seen = {}
        while count < len(self.grouped_inputs):
            label = self.label_range[label_idx]
            if label not in already_seen:
                already_seen[label] = set()

            left_border = 0 if label == 0 else self.groups_right_border[label - 1]
            right_border = self.groups_right_border[label]

            if self.with_replacement:
                selection = np.arange(left_border, right_border)
            else:
                selection = [i for i in np.arange(left_border, right_border) if i not in already_seen[label]]

            if len(selection) >= self.samples_per_label:
                for element_idx in np.random.choice(selection, self.samples_per_label, replace=False):
                    count += 1
                    already_seen[label].add(element_idx)
                    yield self.grouped_inputs[element_idx]

            label_idx += 1
            if label_idx >= len(self.label_range):
                label_idx = 0
                already_seen = {}
                np.random.shuffle(self.label_range)

    def __len__(self):
        # [FUNCTION DOC] Return the number of usable examples (length of grouped_inputs).
        return len(self.grouped_inputs)
