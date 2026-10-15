# [MODULE DOC] Deprecated dataset for TSDAE-style denoising auto-encoder training in the legacy fit loop. Produces
# InputExample pairs of (noised sentence, original sentence). Requires NLTK.
"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

See this script for more details on how to use the new training API:
https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/unsupervised_learning/TSDAE/train_stsb_tsdae.py
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset
from transformers.utils.import_utils import NLTK_IMPORT_ERROR, is_nltk_available

from sentence_transformers.sentence_transformer.readers.input_example import InputExample


class DenoisingAutoEncoderDataset(Dataset):
    # [CLASS DOC] Torch dataset over a list of sentences. Attributes: sentences (list of str) and noise_fn (callable from
    # string to noised string; defaults to the class's word-deletion function).
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    Args:
        sentences: A list of sentences
        noise_fn: A noise function: Given a string, it returns a string
            with noise, e.g. deleted words
    """

    def __init__(self, sentences: list[str], noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s)):
        # [FUNCTION DOC] Store sentences and noise function. Raises ImportError (with a formatted NLTK message) if NLTK is not installed.
        if not is_nltk_available():
            raise ImportError(NLTK_IMPORT_ERROR.format(self.__class__.__name__))

        self.sentences = sentences
        self.noise_fn = noise_fn

    def __getitem__(self, item):
        # [FUNCTION DOC] Return an InputExample whose texts are the noised sentence followed by the original sentence at the given index.
        sent = self.sentences[item]
        return InputExample(texts=[self.noise_fn(sent), sent])

    def __len__(self):
        # [FUNCTION DOC] Return the number of sentences.
        return len(self.sentences)

    # Deletion noise.
    @staticmethod
    def delete(text, del_ratio=0.6):
        # [FUNCTION DOC] Deletion noise: tokenize the text into words with NLTK, return the text unchanged if there are no words.
        # Otherwise keep each word independently with probability 1 minus del_ratio (default 0.6) using numpy's
        # global random state; if none survive, keep one random word. Return the kept words detokenized with the
        # Treebank detokenizer.
        from nltk import word_tokenize
        from nltk.tokenize.treebank import TreebankWordDetokenizer

        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed
