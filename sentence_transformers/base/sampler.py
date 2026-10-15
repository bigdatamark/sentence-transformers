# [MODULE DOC]
# Batch samplers used by the trainers. Defines string enums naming the single-dataset samplers (default,
# no-duplicates, no-duplicates with precomputed hashes, group-by-label) and multi-dataset samplers (round robin,
# proportional); a mixin storing the epoch so samplers can reseed their generator each epoch; DefaultBatchSampler (the
# plain torch batch sampler plus label columns, generator and seed); GroupByLabelBatchSampler (batches mixing several
# labels with pairs per label, for triplet mining); NoDuplicatesBatchSampler (batches with no shared values across rows,
# for in-batch negatives), with optional xxhash-based precomputed row hashes; and multi-dataset samplers that combine
# per-dataset batch samplers over a ConcatDataset. Module constants: int64/uint64 bounds for hash conversion and the
# set of columns excluded from duplicate checks ("dataset_name"). Optionally imports xxhash and datasets.
from __future__ import annotations

import hashlib
import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator
from itertools import accumulate, cycle
from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler, ConcatDataset, SubsetRandomSampler
from transformers.utils import ExplicitEnum

try:
    import xxhash
except ImportError:  # pragma: no cover - optional dependency
    xxhash = None

from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset

logger = logging.getLogger(__name__)

_XXHASH_INT64_MAX = 1 << 63
_XXHASH_UINT64_MAX = 1 << 64
_EXCLUDE_DATASET_COLUMNS = {"dataset_name"}


class BatchSamplers(ExplicitEnum):
    # [CLASS DOC]
    # String enum of batch sampler identifiers: "batch_sampler" (default), "no_duplicates", "no_duplicates_hashed" and
    # "group_by_label".
    """
    Stores the acceptable string identifiers for batch samplers.

    The batch sampler is responsible for determining how samples are grouped into batches during training.
    Valid options are:

    - ``BatchSamplers.BATCH_SAMPLER``: **[default]** Uses :class:`~sentence_transformers.base.sampler.DefaultBatchSampler`, the default
      PyTorch batch sampler.
    - ``BatchSamplers.NO_DUPLICATES``: Uses :class:`~sentence_transformers.base.sampler.NoDuplicatesBatchSampler`,
      ensuring no duplicate samples in a batch.
    - ``BatchSamplers.NO_DUPLICATES_HASHED``: Uses :class:`~sentence_transformers.base.sampler.NoDuplicatesBatchSampler`
      with ``precompute_hashes=True``, a variant that precomputes hashes for faster duplicate checks at a small memory cost.
      Requires the ``xxhash`` library to be installed.

      Both are recommended for losses that use in-batch negatives, such as:

        - :class:`~sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.MultipleNegativesSymmetricRankingLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesSymmetricRankingLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.MegaBatchMarginLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.GISTEmbedLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.CachedGISTEmbedLoss`
    - ``BatchSamplers.GROUP_BY_LABEL``: Uses :class:`~sentence_transformers.base.sampler.GroupByLabelBatchSampler`,
      which constructs each batch by drawing at least 2 samples from each of at least 2 distinct labels.
      This guarantees every batch contains multiple classes, which is required for in-batch triplet mining.
      Recommended for:

        - :class:`~sentence_transformers.sentence_transformer.losses.BatchAllTripletLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.BatchHardSoftMarginTripletLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.BatchHardTripletLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.BatchSemiHardTripletLoss`

    If you want to use a custom batch sampler, then you can subclass
    :class:`~sentence_transformers.base.sampler.DefaultBatchSampler` and pass the class (not an instance) to the
    ``batch_sampler`` argument in :class:`~sentence_transformers.sentence_transformer.training_args.SentenceTransformerTrainingArguments`
    (or :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments`, etc.).
    Alternatively, you can pass a function that accepts ``dataset``, ``batch_size``, ``drop_last``,
    ``valid_label_columns``, ``generator``, and ``seed`` and returns a
    :class:`~sentence_transformers.base.sampler.DefaultBatchSampler` instance.

    Usage:
        ::

            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
            from sentence_transformers.sentence_transformer.training_args import BatchSamplers
            from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
            from datasets import Dataset

            model = SentenceTransformer("microsoft/mpnet-base")
            train_dataset = Dataset.from_dict({
                "anchor": ["It's nice weather outside today.", "He drove to work."],
                "positive": ["It's so sunny.", "He took the car to the office."],
            })
            loss = MultipleNegativesRankingLoss(model)
            args = SentenceTransformerTrainingArguments(
                output_dir="checkpoints",
                batch_sampler=BatchSamplers.NO_DUPLICATES,
            )
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                loss=loss,
            )
            trainer.train()
    """

    BATCH_SAMPLER = "batch_sampler"
    NO_DUPLICATES = "no_duplicates"
    NO_DUPLICATES_HASHED = "no_duplicates_hashed"
    GROUP_BY_LABEL = "group_by_label"


class MultiDatasetBatchSamplers(ExplicitEnum):
    # [CLASS DOC]
    # String enum of multi-dataset sampler identifiers: "round_robin" and "proportional" (default).
    """
    Stores the acceptable string identifiers for multi-dataset batch samplers.

    The multi-dataset batch sampler is responsible for determining in what order batches are sampled from multiple
    datasets during training. Valid options are:

    - ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Uses :class:`~sentence_transformers.base.sampler.RoundRobinBatchSampler`,
      which uses round-robin sampling from each dataset until one is exhausted.
      With this strategy, it's likely that not all samples from each dataset are used, but each dataset is sampled
      from equally.
    - ``MultiDatasetBatchSamplers.PROPORTIONAL``: **[default]** Uses :class:`~sentence_transformers.base.sampler.ProportionalBatchSampler`,
      which samples from each dataset in proportion to its size.
      With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.

    If you want to use a custom multi-dataset batch sampler, then you can subclass
    :class:`~sentence_transformers.base.sampler.MultiDatasetDefaultBatchSampler` and pass the class (not an instance) to the
    ``multi_dataset_batch_sampler`` argument in :class:`~sentence_transformers.sentence_transformer.training_args.SentenceTransformerTrainingArguments`.
    (or :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments`, etc.). Alternatively,
    you can pass a function that accepts ``dataset`` (a :class:`~torch.utils.data.ConcatDataset`), ``batch_samplers``
    (i.e. a list of batch sampler for each of the datasets in the :class:`~torch.utils.data.ConcatDataset`), ``generator``,
    and ``seed`` and returns a :class:`~sentence_transformers.base.sampler.MultiDatasetDefaultBatchSampler` instance.

    Usage:
        ::

            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
            from sentence_transformers.sentence_transformer.training_args import MultiDatasetBatchSamplers
            from sentence_transformers.sentence_transformer.losses import CoSENTLoss
            from datasets import Dataset, DatasetDict

            model = SentenceTransformer("microsoft/mpnet-base")
            train_general = Dataset.from_dict({
                "sentence_A": ["It's nice weather outside today.", "He drove to work."],
                "sentence_B": ["It's so sunny.", "He took the car to the bank."],
                "score": [0.9, 0.4],
            })
            train_medical = Dataset.from_dict({
                "sentence_A": ["The patient has a fever.", "The doctor prescribed medication.", "The patient is sweating."],
                "sentence_B": ["The patient feels hot.", "The medication was given to the patient.", "The patient is perspiring."],
                "score": [0.8, 0.6, 0.7],
            })
            train_legal = Dataset.from_dict({
                "sentence_A": ["This contract is legally binding.", "The parties agree to the terms and conditions."],
                "sentence_B": ["Both parties acknowledge their obligations.", "By signing this agreement, the parties enter into a legal relationship."],
                "score": [0.7, 0.8],
            })
            train_dataset = DatasetDict({
                "general": train_general,
                "medical": train_medical,
                "legal": train_legal,
            })

            loss = CoSENTLoss(model)
            args = SentenceTransformerTrainingArguments(
                output_dir="checkpoints",
                multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
            )
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                loss=loss,
            )
            trainer.train()
    """

    ROUND_ROBIN = "round_robin"  # Round-robin sampling from each dataset
    PROPORTIONAL = "proportional"  # Sample from each dataset in proportion to its size [default]


class SetEpochMixin:
    # [CLASS DOC]
    # HELPER: mixin adding an epoch attribute (starts at 0) that the trainer updates each epoch so samplers can reseed.
    """
    Required for a BatchSampler as the Trainer will call set_epoch on the BatchSampler at the beginning of each epoch.
    The BatchSampler can then set the generator seed accordingly.
    """

    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # HELPER: passes all arguments to the next initialiser and sets epoch to 0.
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the epoch number.
        self.epoch = epoch


class DefaultBatchSampler(SetEpochMixin, BatchSampler):
    # [CLASS DOC]
    # Default sampler: identical to the torch batch sampler over the dataset (used as the sampler, so it yields index
    # lists in dataset order in consecutive batch_size chunks), plus attributes valid_label_columns, generator and seed for
    # use by subclasses, and the epoch from the mixin.
    """
    This sampler is the default batch sampler used in the SentenceTransformer library.
    It is equivalent to the PyTorch BatchSampler.

    Args:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): Number of samples per batch.
        drop_last (bool): If True, drop the last incomplete batch if the dataset size
            is not divisible by the batch size.
        valid_label_columns (List[str], optional): List of column names to check for labels.
            The first column name from ``valid_label_columns`` found in the dataset will
            be used as the label column.
        generator (torch.Generator, optional): Optional random number generator for shuffling
            the indices.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Initialises the torch batch sampler with the dataset as the underlying sampler, batch size and drop_last, and stores
        # the label columns, generator and seed.
        super().__init__(dataset, batch_size=batch_size, drop_last=drop_last)
        self.valid_label_columns = valid_label_columns
        self.generator = generator
        self.seed = seed


class GroupByLabelBatchSampler(DefaultBatchSampler):
    # [CLASS DOC]
    # Sampler for in-batch triplet mining. Attributes: dataset, groups (label to sample indices, keeping only labels with
    # at least 2 samples, each list trimmed to an even length) and the precomputed stream length. Each epoch, indices are
    # shuffled within labels and labels take turns (in a freshly shuffled order every round) emitting 2 samples each
    # until fewer than 2 labels have samples left; the resulting stream is cut into batches of exactly batch_size.
    """
    Batch sampler that groups samples by label for in-batch triplet mining.

    Samples are shuffled within each label, then interleaved in round-robin
    fashion to produce a stream where labels are well-mixed. This stream is
    chunked into batches of exactly ``batch_size``. Every batch is guaranteed
    to contain multiple distinct labels, each with at least 2 samples.

    Labels take turns emitting 2 samples each. The stream stops when fewer
    than 2 labels remain, so the dominant label's tail ends up in the
    remainder. Produces excellent per-batch balance.

    Recommended for:
        - :class:`~sentence_transformers.sentence_transformer.losses.BatchAllTripletLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.BatchHardSoftMarginTripletLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.BatchHardTripletLoss`
        - :class:`~sentence_transformers.sentence_transformer.losses.BatchSemiHardTripletLoss`

    Args:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): Number of samples per batch. Must be an even number >= 4.
        drop_last (bool): If True, drop the last incomplete batch if the dataset size
            is not divisible by the batch size.
        valid_label_columns (List[str], optional): List of column names to check for labels.
            The first column name from ``valid_label_columns`` found in the dataset will
            be used as the label column.
        generator (torch.Generator, optional): Optional random number generator for shuffling
            the indices.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Initialises the base sampler, stores the dataset, raises ValueError unless batch_size is even and at least 4, reads
        # the labels from the first valid label column, groups sample indices by label, keeps qualifying labels (raising
        # ValueError if fewer than 2 qualify), and computes the stream length as twice the sum over labels of the minimum of
        # that label's pair count and the second-largest pair count.
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        self.dataset = dataset

        if self.batch_size < 4 or self.batch_size % 2 == 1:
            raise ValueError(f"batch_size must be an even number >= 4, but got {self.batch_size}.")

        labels = self._determine_labels_to_use(dataset, self.valid_label_columns)
        groups: dict[Any, list[int]] = defaultdict(list)
        for sample_idx, label in enumerate(labels):
            groups[label].append(sample_idx)

        # Keep labels with >= 2 samples. Trim each to an even count so every
        # label contributes complete pairs in the interleaving.
        self.groups = {
            label: indices[: len(indices) // 2 * 2] for label, indices in groups.items() if len(indices) >= 2
        }
        if len(self.groups) < 2:
            raise ValueError(
                "GroupByLabelBatchSampler requires at least 2 distinct labels with >= 2 samples each, "
                f"but only {len(self.groups)} label(s) qualified."
            )

        # Pre-compute stream length: round-robin stops when only 1 label remains
        pairs = sorted((len(idx) // 2 for idx in self.groups.values()), reverse=True)
        cap = pairs[1]  # second-largest: the round when we'd drop to 1 label
        self._stream_length = 2 * sum(min(p, cap) for p in pairs)

    @staticmethod
    def _determine_labels_to_use(dataset: Dataset, valid_label_columns: list[str] | None) -> list[Any]:
        # [FUNCTION DOC]
        # HELPER: static. Returns the values of the first valid label column present in the dataset; raises ValueError if none
        # is present.
        for column_name in valid_label_columns or []:
            if column_name in dataset.column_names:
                return dataset[column_name]
        raise ValueError(
            f"None of the valid_label_columns {valid_label_columns} are in the dataset, "
            f"which only has these columns: {dataset.column_names}."
        )

    def __iter__(self) -> Iterator[list[int]]:
        # [FUNCTION DOC]
        # Generator of batches. Reseeds the generator with seed plus epoch if a generator is set. Shuffles each label's indices
        # into a queue, then repeatedly shuffles the remaining labels and pops 2 indices per label into a buffer, yielding a
        # batch whenever the buffer reaches batch_size (keeping the overflow), and dropping exhausted labels after each round.
        # At the end, a leftover buffer of at least 4 indices is yielded unless drop_last is set.
        # KNOWN ISSUE: the claimed guarantee that every batch holds at least 2 distinct labels is not strict: because the label
        # order is reshuffled each round, the last pair of one round and the first pair of the next can come from the same
        # label, so e.g. with batch_size 4 a batch (or the final leftover) may contain a single label.
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        # Shuffle samples within each label
        queues: dict[Any, deque[int]] = {}
        for label, indices in self.groups.items():
            perm = torch.randperm(len(indices), generator=self.generator)
            queues[label] = deque(indices[i] for i in perm)

        # Round-robin: each label emits 2 samples per round. Stop when < 2 labels remain.
        # The label visit order is reshuffled every round for diverse batches.
        remaining_labels = list(queues)
        batch: list[int] = []
        while len(remaining_labels) >= 2:
            remaining_labels = [
                remaining_labels[i] for i in torch.randperm(len(remaining_labels), generator=self.generator)
            ]
            for label in remaining_labels:
                batch.append(queues[label].popleft())
                batch.append(queues[label].popleft())
                if len(batch) >= self.batch_size:
                    yield batch[: self.batch_size]
                    batch = batch[self.batch_size :]
            remaining_labels = [label for label in remaining_labels if queues[label]]

        # Due to the round-robin loading, at least 4 elements ensures >= 2 distinct labels, each with >= 2 samples.
        if not self.drop_last and len(batch) >= 4:
            yield batch

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of full batches in the stream, plus one when drop_last is false and the remainder has at least 4
        # indices.
        n = self._stream_length // self.batch_size
        if not self.drop_last and self._stream_length % self.batch_size >= 4:
            n += 1
        return n


_PLAIN_VALUE_TYPES = (str, int, float, bool, bytes, type(None), np.integer, np.floating, np.bool_)


def _sample_value_str(value: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: returns a string key used to compare sample values for duplicates. Plain scalars (str, numbers, bools,
    # bytes, None, numpy scalars) and lists/tuples of them use their string form; other lists/tuples and dicts are rendered
    # recursively; other objects (media) use the "_hf_encoded" source if present and are digested as the SHA-256 of their
    # pickled form, falling back to their string form if pickling fails.
    # Plain python values keep their exact str(), the historical dedup key. Other objects (PIL
    # images, audio arrays, media decoders) often embed a memory address in str(), fresh on every
    # dataset access, so their pickled content is digested instead. Values that fail to pickle
    # fall back to str() and simply never match, as before.
    if isinstance(value, _PLAIN_VALUE_TYPES):
        return str(value)
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, _PLAIN_VALUE_TYPES) for item in value):
            return str(value)
        return "[" + ", ".join(_sample_value_str(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key!r}: {_sample_value_str(item)}" for key, item in value.items()) + "}"
    # Audio / Video decoders pickle a torch storage key taken from the object address, fresh on
    # every access. They carry the encoded source datasets itself re-encodes from, so digest that.
    encoded = getattr(value, "_hf_encoded", None)
    if encoded is not None:
        value = encoded
    try:
        return hashlib.sha256(pickle.dumps(value)).hexdigest()
    except Exception:
        return str(value)


def _xxhash_int64(value: str) -> int:
    # [FUNCTION DOC]
    # HELPER: returns the 64-bit xxhash of the UTF-8 string, converted from unsigned to signed int64 range.
    # Convert uint64 -> int64 to keep values compatible with Arrow int64 storage.
    hashed = xxhash.xxh64_intdigest(value.encode("utf-8"))
    if hashed >= _XXHASH_INT64_MAX:
        hashed -= _XXHASH_UINT64_MAX
    return hashed


def _hash_batch(
    batch: dict[str, list[Any]], columns: list[str], exclude_columns: set[str]
) -> dict[str, list[list[int]]]:
    # [FUNCTION DOC]
    # HELPER: batched mapping function (module level so it can be pickled for multiprocessing). For a batch dict of column
    # lists, returns {"__hashes": list per row of the signed xxhash of each non-excluded column's value key}; rows get empty
    # lists when no columns remain.
    # Must be defined at module scope because datasets.map with num_proc pickles this function.
    # Build per-row hash lists so we can later do fast overlap checks without re-reading the dataset.
    active_columns = [column for column in columns if column not in exclude_columns]
    batch_size = len(batch[active_columns[0]]) if active_columns else len(next(iter(batch.values()), []))
    if not active_columns:
        return {"__hashes": [[] for _ in range(batch_size)]}
    hashes: list[list[int]] = []
    for row_idx in range(batch_size):
        row_hashes: list[int] = []
        for column in active_columns:
            value = batch[column][row_idx]
            # Keep semantics aligned with the non-hash path, which compares
            # stringified per-column values (including list values as a whole).
            row_hashes.append(_xxhash_int64(_sample_value_str(value)))
        hashes.append(row_hashes)
    return {"__hashes": hashes}


class NoDuplicatesBatchSampler(DefaultBatchSampler):
    # [CLASS DOC]
    # Sampler ensuring no value appears twice within a batch, across all columns (label columns are removed from the
    # dataset used for comparison, and "dataset_name" is ignored). Attributes: dataset (without label columns),
    # precompute_hashes, precompute_num_proc, precompute_batch_size and the lazily built dense row hash matrix.
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
        precompute_hashes: bool = False,
        precompute_num_proc: int | None = None,
        precompute_batch_size: int = 1000,
    ) -> None:
        # [FUNCTION DOC]
        # Initialises the base sampler, removes the valid label columns from the dataset, and stores the hashing options. With
        # precompute_hashes, raises ImportError if xxhash is missing and defaults the worker count to the CPU count minus one,
        # clamped between 1 and 8.
        """
        This sampler creates batches such that each batch contains samples where the values are unique,
        even across columns. This is useful when losses consider other samples in a batch to be in-batch
        negatives, and you want to ensure that the negatives are not duplicates of the anchor/positive sample.

        Recommended for:
            - :class:`~sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss`
            - :class:`~sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss`
            - :class:`~sentence_transformers.sentence_transformer.losses.MultipleNegativesSymmetricRankingLoss`
            - :class:`~sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesSymmetricRankingLoss`
            - :class:`~sentence_transformers.sentence_transformer.losses.MegaBatchMarginLoss`
            - :class:`~sentence_transformers.sentence_transformer.losses.GISTEmbedLoss`
            - :class:`~sentence_transformers.sentence_transformer.losses.CachedGISTEmbedLoss`

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch if the dataset size
                is not divisible by the batch size.
            valid_label_columns (List[str], optional): List of column names to check for labels.
                The first column name from ``valid_label_columns`` found in the dataset will
                be used as the label column.
            generator (torch.Generator, optional): Optional random number generator for shuffling
                the indices.
            seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
            precompute_hashes (bool, optional): If True, precompute xxhash 64-bit values for dataset
                fields using ``datasets.map`` to speed up duplicate checks. Requires ``xxhash`` to
                be installed and uses additional memory: in theory roughly
                ``len(dataset) * num_columns * 8`` bytes for the dense int64 hash matrix,
                although actual memory usage may therefore differ in practice. Especially
                recommended for media (image / audio) datasets: without precomputation, every
                duplicate check re-reads its row from the dataset, which re-decodes the media.
                Defaults to False.
            precompute_num_proc (int, optional): Number of processes for hashing with ``datasets.map``.
                If set to ``None``, defaults to ``min(8, cpu_count - 1)`` when ``precompute_hashes``
                is True.
            precompute_batch_size (int, optional): Batch size for ``datasets.map`` hashing.
                Defaults to 1000.
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        if label_columns := set(dataset.column_names) & set(self.valid_label_columns or []):
            dataset = dataset.remove_columns(list(label_columns))
        self.dataset = dataset
        self.precompute_hashes = precompute_hashes
        self.precompute_num_proc = precompute_num_proc
        self.precompute_batch_size = precompute_batch_size
        self._row_hashes: np.ndarray | None = None
        if self.precompute_hashes:
            if xxhash is None:
                raise ImportError(
                    "NoDuplicatesBatchSampler with precompute_hashes=True requires `xxhash`. "
                    "Install `xxhash` to use this option."
                )
            if self.precompute_num_proc is None:
                cpu_count = os.cpu_count() or 1
                # Leave one core free to avoid saturating the system when hashing.
                default_workers = max(1, min(8, cpu_count - 1))
                self.precompute_num_proc = default_workers

    def _build_hashes(self) -> None:
        # [FUNCTION DOC]
        # HELPER: builds the row hash matrix once (no-op if disabled or already built). Maps the dataset in batches (parallel
        # processes, progress description) to per-row hash lists, reads the resulting Arrow list column, and stores it as a
        # dense int64 matrix of rows by columns (an empty 0 by 0 matrix for an empty dataset). Raises ValueError if the column is
        # not a list column, rows have different lengths, or the value buffer size is inconsistent. Releases the temporary
        # dataset afterwards.
        # Build once lazily on first iteration, then reuse across epochs.
        # Hashes depend on dataset content, not epoch seed/order.
        if not self.precompute_hashes or self._row_hashes is not None:
            return
        columns = list(self.dataset.column_names)
        # Precompute hash values once to avoid repeated string processing per batch.
        # Use num_proc to parallelize hashing across CPU cores.
        hash_ds = self.dataset.map(
            _hash_batch,
            batched=True,
            batch_size=self.precompute_batch_size,
            num_proc=self.precompute_num_proc,
            remove_columns=columns,
            fn_kwargs={"columns": columns, "exclude_columns": _EXCLUDE_DATASET_COLUMNS},
            desc="Hashing dataset values",
        )
        import pyarrow as pa

        try:
            column = hash_ds.data.column("__hashes")
            if isinstance(column, pa.ChunkedArray):
                column = column.combine_chunks()
            if not isinstance(column, (pa.ListArray, pa.LargeListArray)):
                raise ValueError("Expected a list column for hashed values.")

            row_count = len(column)
            if row_count == 0:
                self._row_hashes = np.zeros((0, 0), dtype=np.int64)
                return

            offsets = column.offsets.to_numpy(zero_copy_only=False)
            row_size = int(offsets[1] - offsets[0])
            # Dense ndarray storage below requires a fixed number of hashed
            # values per row to allow safe reshape(row_count, row_size).
            if row_size < 0 or not np.all(np.diff(offsets) == row_size):
                raise ValueError("Hashed rows have varying lengths.")
            # If every row has the same length, store as a dense ndarray to reduce overhead.
            values = column.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            if values.size != row_count * row_size:
                raise ValueError("Unexpected hashed value buffer size.")
            self._row_hashes = values.reshape((row_count, row_size))
        finally:
            # Drop the temporary dataset to release Arrow buffers promptly.
            del hash_ds

    def __iter__(self) -> Iterator[list[int]]:
        # [FUNCTION DOC]
        # Generator of duplicate-free batches. Reseeds the generator with seed plus epoch if set. Each row's values are its hash
        # row (hash mode, after building hashes) or the set of value keys of its non-excluded columns (reading the row). Takes a
        # random permutation of row indices and keeps the not-yet-used positions in a singly linked list. Each pass walks the
        # list from its head: a row whose values overlap the batch's values is skipped (kept for later passes); otherwise it is
        # added to the batch and unlinked, and the batch is yielded once it reaches batch_size. A pass that ends without filling
        # the batch yields the partial batch unless drop_last is set. Passes repeat until every row has been used. Yields
        # nothing for an empty dataset.
        """
        Iterate over the remaining non-yielded indices. For each index, check if the sample values are already in the
        batch. If not, add the sample values to the batch keep going until the batch is full. If the batch is full, yield
        the batch indices and continue with the next batch.
        """
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        if self.precompute_hashes:
            self._build_hashes()
            row_hashes: np.ndarray = self._row_hashes

            def get_sample_values(index: int) -> set[str] | np.ndarray:
                # [FUNCTION DOC]
                # HELPER: returns the precomputed hash row of a sample.
                return row_hashes[index]

        else:

            def get_sample_values(index: int) -> set[str] | np.ndarray:
                # [FUNCTION DOC]
                # HELPER: returns the set of value keys of a sample's columns, excluding "dataset_name".
                return {
                    _sample_value_str(value)
                    for key, value in self.dataset[index].items()
                    if key not in _EXCLUDE_DATASET_COLUMNS
                }

        def _has_overlap(sample_values: set[str] | np.ndarray, batch_values: set[str | np.int64]) -> bool:
            # [FUNCTION DOC]
            # HELPER: returns True if any of the sample's values (set or hash row) is already among the batch's values.
            # Non-hash path with set[str] allows for disjoint overlap checks
            if isinstance(sample_values, set):
                return not sample_values.isdisjoint(batch_values)
            # Hash path with ndarray does set instance checks
            return any(value in batch_values for value in sample_values)

        num_rows = len(self.dataset)
        if num_rows == 0:
            return

        # Create a random numpy permutation using int32 (or int64 if necessary)
        index_dtype = torch.int32 if num_rows <= np.iinfo(np.int32).max else torch.int64
        remaining_indices = torch.randperm(num_rows, generator=self.generator, dtype=index_dtype).numpy()

        # Plus a singly linked list over shuffled positions, where the last position is marked with -1
        # for simple termination
        position_dtype = np.int32 if num_rows + 1 <= np.iinfo(np.int32).max else np.int64
        next_positions = np.arange(1, num_rows + 1, dtype=position_dtype)
        next_positions[-1] = -1
        head_position = 0

        while head_position != -1:
            batch_values: set[str | np.int64] = set()
            batch_indices: list[int] = []
            current_position = head_position
            previous_position = -1
            full_batch = False
            while current_position != -1:
                next_position = int(next_positions[current_position])
                index = int(remaining_indices[current_position])
                sample_values = get_sample_values(index)
                if _has_overlap(sample_values, batch_values):
                    # Defer conflicting samples to later batches instead of reordering them.
                    previous_position = current_position
                    current_position = next_position
                    continue

                batch_indices.append(index)
                if previous_position == -1:
                    head_position = next_position
                else:
                    next_positions[previous_position] = next_position

                if len(batch_indices) == self.batch_size:
                    full_batch = True
                    yield batch_indices
                    break

                batch_values.update(sample_values)
                current_position = next_position

            if not full_batch:
                # NOTE: some indices might still have been ignored here
                if not self.drop_last:
                    yield batch_indices

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of batches implied by the dataset size and batch size (floor with drop_last, ceiling otherwise).
        # KNOWN ISSUE: the docstring calls this an upper bound, but when duplicates are deferred, later passes can each yield a
        # small partial batch (when drop_last is false), so the iterator can yield more batches than this count.
        """Return the approximate number of batches.

        .. note::

            This is an upper-bound estimate. The actual number of batches
            yielded by :meth:`__iter__` may be smaller when the dataset
            contains many duplicate values, because those samples are
            deferred or skipped rather than placed into a batch.
        """
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MultiDatasetDefaultBatchSampler(SetEpochMixin, BatchSampler, ABC):
    # [CLASS DOC]
    # Abstract base for samplers over a ConcatDataset that combine one batch sampler per sub-dataset. Attributes: dataset,
    # batch_samplers, generator, seed, and batch_size / drop_last taken from the first sub-sampler. Subclasses implement
    # iteration and length; yielded indices are offset into the concatenated dataset.
    """
    Abstract base batch sampler that yields batches from multiple batch samplers.
    This class must be subclassed to implement specific sampling strategies, and
    cannot be used directly.

    Args:
        dataset (ConcatDataset): A concatenation of multiple datasets.
        batch_samplers (List[BatchSampler]): A list of batch samplers, one for each dataset in the ConcatDataset.
        generator (torch.Generator, optional): A generator for reproducible sampling. Defaults to None.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_samplers: list[BatchSampler],
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Raises ValueError if the number of batch samplers differs from the number of sub-datasets; initialises the torch batch
        # sampler with the first sub-sampler's batch size and drop_last, and stores dataset, samplers, generator and seed.
        if len(dataset.datasets) != len(batch_samplers):
            raise ValueError("The number of batch samplers must match the number of datasets in the ConcatDataset.")
        super().__init__(dataset, batch_size=batch_samplers[0].batch_size, drop_last=batch_samplers[0].drop_last)
        self.dataset = dataset
        self.batch_samplers = batch_samplers
        self.generator = generator
        self.seed = seed

    @abstractmethod
    def __iter__(self) -> Iterator[list[int]]:
        # [FUNCTION DOC]
        # Abstract: must yield batches of global indices.
        """Yield batches from the underlying datasets in a specific order."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Abstract: must return the number of batches.
        """Return the number of batches in the sampler."""
        pass


class RoundRobinBatchSampler(MultiDatasetDefaultBatchSampler):
    # [CLASS DOC]
    # Multi-dataset sampler alternating between sub-datasets in fixed cyclic order, one batch each, stopping as soon as the
    # sub-sampler whose turn it is is exhausted.
    """
    Batch sampler that yields batches in a round-robin fashion from multiple batch samplers, until one is exhausted.
    With this sampler, it's unlikely that all samples from each dataset are used, but we do ensure that each dataset
    is sampled from equally.

    Args:
        dataset (ConcatDataset): A concatenation of multiple datasets.
        batch_samplers (List[BatchSampler]): A list of batch samplers, one for each dataset in the ConcatDataset.
        generator (torch.Generator, optional): A generator for reproducible sampling. Defaults to None.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __iter__(self) -> Iterator[list[int]]:
        # [FUNCTION DOC]
        # Reseeds the generator with seed plus epoch if set, computes each sub-dataset's start offset in the concatenation,
        # and cycles over the sub-samplers yielding each next batch shifted by its offset; stops at the first exhausted
        # sub-sampler.
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        num_samples = [len(dataset) for dataset in self.dataset.datasets]
        sample_offsets = [0] + list(accumulate(num_samples))

        batch_samplers = [iter(sampler) for sampler in self.batch_samplers]
        for dataset_idx in cycle(range(len(batch_samplers))):
            sample_offset = sample_offsets[dataset_idx]
            try:
                yield [idx + sample_offset for idx in next(batch_samplers[dataset_idx])]
            except StopIteration:
                # current iterator is apparently exhausted
                break

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the smallest sub-sampler length times the number of sub-samplers.
        # KNOWN ISSUE: this undercounts when the shortest sub-sampler is not the first one: e.g. with lengths 3 and 2 the
        # iterator yields 5 batches (it stops only when the second sampler is asked for a third batch) while this returns 4.
        return min(len(sampler) for sampler in self.batch_samplers) * len(self.batch_samplers)


class ProportionalBatchSampler(MultiDatasetDefaultBatchSampler):
    # [CLASS DOC]
    # Multi-dataset sampler that uses every batch of every sub-sampler, visiting sub-datasets in a random order where each
    # appears as many times as it has batches.
    """
    Batch sampler that samples from each dataset in proportion to its size, until all are exhausted simultaneously.
    With this sampler, all samples from each dataset are used and larger datasets are sampled from more frequently.

    Args:
        dataset (ConcatDataset): A concatenation of multiple datasets.
        batch_samplers (List[BatchSampler]): A list of batch samplers, one for each dataset in the ConcatDataset.
        generator (torch.Generator, optional): A generator for reproducible sampling. Defaults to None.
        seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
    """

    def __iter__(self) -> Iterator[list[int]]:
        # [FUNCTION DOC]
        # Reseeds the generator with seed plus epoch if set, computes sub-dataset offsets, builds a list repeating each
        # sub-dataset index once per batch of its sampler, draws it in random order (using the generator), and yields the next
        # batch of the drawn sub-sampler shifted by its offset (skipping exhausted ones).
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        num_samples = [len(dataset) for dataset in self.dataset.datasets]
        sample_offsets = [0] + list(accumulate(num_samples))

        num_batches = [len(sampler) for sampler in self.batch_samplers]
        dataset_indices = [idx for idx, length in enumerate(num_batches) for _ in range(length)]
        dataset_idx_sampler = SubsetRandomSampler(dataset_indices, generator=self.generator)

        batch_samplers = [iter(sampler) for sampler in self.batch_samplers]
        for dataset_idx in dataset_idx_sampler:
            sample_offset = sample_offsets[dataset_idx]
            try:
                yield [idx + sample_offset for idx in next(batch_samplers[dataset_idx])]
            except StopIteration:
                continue

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the total number of batches over all sub-samplers.
        return sum([len(sampler) for sampler in self.batch_samplers])


__all__ = [
    "DefaultBatchSampler",
    "GroupByLabelBatchSampler",
    "NoDuplicatesBatchSampler",
    "MultiDatasetDefaultBatchSampler",
    "RoundRobinBatchSampler",
    "ProportionalBatchSampler",
    "BatchSamplers",
    "MultiDatasetBatchSamplers",
]
