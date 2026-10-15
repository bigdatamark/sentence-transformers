# [MODULE DOC] Deprecated legacy dataset for multilingual knowledge distillation: reads parallel (translated) sentences
# and pairs every sentence with the teacher model's embedding of the source (first-column) sentence, so a
# student can be trained with an MSE loss to map translations to the same vector.
"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

Instead, you should create a `datasets` `Dataset` for training: https://huggingface.co/docs/datasets/create_dataset
"""

from __future__ import annotations

import gzip
import logging
import random
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from sentence_transformers.sentence_transformer.readers import InputExample

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer


class ParallelSentencesDataset(Dataset):
    # [CLASS DOC] Torch map-style dataset that lazily generates InputExamples (texts = one sentence, label = teacher
    # embedding of its source sentence). Attributes: student_model and teacher_model (teacher must have an encode
    # method), batch_size for teacher encoding, use_embedding_cache flag, embedding_cache (source text to
    # embedding), datasets (list of lists of (source, set of sentences) items), datasets_iterator (per-dataset
    # cursor), dataset_indices (dataset ids repeated by weight, defines one generation round), cache (pending
    # examples), num_sentences (reported length). datasets_tokenized and copy_dataset_indices are unused.
    """
    This dataset reader can be used to read-in parallel sentences, i.e., it reads in a file with tab-separated sentences with the same
    sentence in different languages. For example, the file can look like this (EN\tDE\tES):
    hello world     hallo welt  hola mundo
    second sentence zweiter satz    segunda oración

    The sentence in the first column will be mapped to a sentence embedding using the given the embedder. For example,
    embedder is a mono-lingual sentence embedding method for English. The sentences in the other languages will also be
    mapped to this English sentence embedding.

    When getting a sample from the dataset, we get one sentence with the according sentence embedding for this sentence.

    teacher_model can be any class that implement an encode function. The encode function gets a list of sentences and
    returns a list of sentence embeddings
    """

    def __init__(
        self,
        student_model: SentenceTransformer,
        teacher_model: SentenceTransformer,
        batch_size: int = 8,
        use_embedding_cache: bool = True,
    ):
        # [FUNCTION DOC] Store the models, batch size and cache flag; initialize all bookkeeping lists/dicts empty and the
        # sentence counter to 0.
        """
        Parallel sentences dataset reader to train student model given a teacher model

        Args:
            student_model (SentenceTransformer): The student sentence embedding model that should be trained.
            teacher_model (SentenceTransformer): The teacher model that provides the sentence embeddings for the first column in the dataset file.
            batch_size (int, optional): The batch size for training. Defaults to 8.
            use_embedding_cache (bool, optional): Whether to use an embedding cache. Defaults to True.
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.datasets = []
        self.datasets_iterator = []
        self.datasets_tokenized = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
        self.cache = []
        self.batch_size = batch_size
        self.use_embedding_cache = use_embedding_cache
        self.embedding_cache = {}
        self.num_sentences = 0

    def load_data(
        self, filepath: str, weight: int = 100, max_sentences: int | None = None, max_sentence_length: int = 128
    ) -> None:
        # [FUNCTION DOC] Read a tab-separated text file (gzip-decompressed if the path ends with ".gz", UTF-8) where each line holds
        # a source sentence followed by its translations. Lines with any column longer than max_sentence_length
        # characters (when positive) are skipped; reading stops after max_sentences accepted lines (when positive).
        # Logs the file being loaded, then passes the rows to add_dataset with the same weight and limits.
        # Returns None. Raises on file I/O errors.
        """
        Reads in a tab-separated .txt/.csv/.tsv or .gz file. The different columns contain the different translations of the sentence in the first column

        Args:
            filepath (str): Filepath to the file.
            weight (int, optional): If more than one dataset is loaded with load_data, specifies the frequency at which data should be sampled from this dataset. Defaults to 100.
            max_sentences (int, optional): Maximum number of lines to be read from the filepath. Defaults to None.
            max_sentence_length (int, optional): Skip the example if one of the sentences has more characters than max_sentence_length. Defaults to 128.

        Returns:
            None
        """

        logger.info(f"Loading {filepath}")
        parallel_sentences = []

        with (
            gzip.open(filepath, "rt", encoding="utf8")
            if filepath.endswith(".gz")
            else open(filepath, encoding="utf8") as fIn
        ):
            count = 0
            for line in fIn:
                sentences = line.strip().split("\t")
                if (
                    max_sentence_length is not None
                    and max_sentence_length > 0
                    and max([len(sent) for sent in sentences]) > max_sentence_length
                ):
                    continue

                parallel_sentences.append(sentences)
                count += 1
                if max_sentences is not None and max_sentences > 0 and count >= max_sentences:
                    break
        self.add_dataset(
            parallel_sentences, weight=weight, max_sentences=max_sentences, max_sentence_length=max_sentence_length
        )

    def add_dataset(
        self,
        parallel_sentences: list[list[str]],
        weight: int = 100,
        max_sentences: int | None = None,
        max_sentence_length: int = 128,
    ):
        # [FUNCTION DOC] Register a list of parallel sentence rows as a new dataset. Rows exceeding the length limit are skipped.
        # Rows are grouped by their first sentence into a map from source to the set of all sentences in the row
        # (including the source itself); stops once the map has max_sentences entries (when positive). If nothing
        # remains, returns without changes. Otherwise adds the total number of sentences in all sets to
        # num_sentences, appends the map items as a new dataset with cursor 0, and appends the new dataset id
        # weight times to dataset_indices. Returns None.
        sentences_map = {}
        for sentences in parallel_sentences:
            if (
                max_sentence_length is not None
                and max_sentence_length > 0
                and max([len(sent) for sent in sentences]) > max_sentence_length
            ):
                continue

            source_sentence = sentences[0]
            if source_sentence not in sentences_map:
                sentences_map[source_sentence] = set()

            for sent in sentences:
                sentences_map[source_sentence].add(sent)

            if max_sentences is not None and max_sentences > 0 and len(sentences_map) >= max_sentences:
                break

        if len(sentences_map) == 0:
            return

        self.num_sentences += sum([len(sentences_map[sent]) for sent in sentences_map])

        dataset_id = len(self.datasets)
        self.datasets.append(list(sentences_map.items()))
        self.datasets_iterator.append(0)
        self.dataset_indices.extend([dataset_id] * weight)

    def generate_data(self):
        # [FUNCTION DOC] Fill the example cache: for every entry in dataset_indices, take the next (source, sentences) entry of that
        # dataset, encode all sources with the teacher (via get_embeddings), then append one InputExample per target
        # sentence labeled with its source's embedding, and shuffle the cache. Mutates cache and dataset cursors.
        source_sentences_list = []
        target_sentences_list = []
        for data_idx in self.dataset_indices:
            src_sentence, trg_sentences = self.next_entry(data_idx)
            source_sentences_list.append(src_sentence)
            target_sentences_list.append(trg_sentences)

        # Generate embeddings
        src_embeddings = self.get_embeddings(source_sentences_list)

        for src_embedding, trg_sentences in zip(src_embeddings, target_sentences_list):
            for trg_sentence in trg_sentences:
                self.cache.append(InputExample(texts=[trg_sentence], label=src_embedding))

        random.shuffle(self.cache)

    def next_entry(self, data_idx):
        # [FUNCTION DOC] Return the dataset entry at the current cursor of the given dataset and advance the cursor; when it
        # passes the end, reset it to 0 and shuffle that dataset's entries in place.
        source, target_sentences = self.datasets[data_idx][self.datasets_iterator[data_idx]]

        self.datasets_iterator[data_idx] += 1
        if self.datasets_iterator[data_idx] >= len(self.datasets[data_idx]):  # Restart iterator
            self.datasets_iterator[data_idx] = 0
            random.shuffle(self.datasets[data_idx])

        return source, target_sentences

    def get_embeddings(self, sentences):
        # [FUNCTION DOC] Return teacher embeddings (numpy) for a list of sentences. Without caching, encode them all directly with
        # the configured batch size and no progress bar. With caching, encode only sentences not yet cached, store
        # their embeddings, and return the cached embeddings in input order (as a list).
        if not self.use_embedding_cache:
            return self.teacher_model.encode(
                sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
            )

        # Use caching
        new_sentences = []
        for sent in sentences:
            if sent not in self.embedding_cache:
                new_sentences.append(sent)

        if len(new_sentences) > 0:
            new_embeddings = self.teacher_model.encode(
                new_sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
            )
            for sent, embedding in zip(new_sentences, new_embeddings):
                self.embedding_cache[sent] = embedding

        return [self.embedding_cache[sent] for sent in sentences]

    def __len__(self):
        # [FUNCTION DOC] Return num_sentences (total sentences across all registered datasets).
        return self.num_sentences

    def __getitem__(self, idx):
        # [FUNCTION DOC] Return one example, regenerating the cache first if it is empty. The index is ignored; examples are
        # popped from the end of the shuffled cache, so access is effectively random and consuming.
        if len(self.cache) == 0:
            self.generate_data()

        return self.cache.pop()
