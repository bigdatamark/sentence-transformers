# [MODULE DOC]
# Pytest tests for RoundRobinBatchSampler, which takes a ConcatDataset and one batch sampler per sub-dataset and
# alternates yielding batches from each sub-dataset (offsetting indices into the concatenated index space), stopping
# as soon as any sub-sampler is exhausted. Skips the module if the datasets library is missing. Defines a module
# constant for the base sub-dataset length (25).
from __future__ import annotations

import pytest
from torch.utils.data import BatchSampler, ConcatDataset, SequentialSampler

from sentence_transformers.base.sampler import RoundRobinBatchSampler
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )

DATASET_LENGTH = 25


@pytest.fixture
def dummy_concat_dataset() -> ConcatDataset:
    # [FUNCTION DOC]
    # HELPER: Fixture building a ConcatDataset of two Datasets: the first has data 0 to 24 with alternating labels
    # 0/1; the second has 50 rows with data 100 to 124 followed by 200 to 224 and the same alternating labels
    # repeated.
    # KNOWN ISSUE: The docstring describes the second part as only 100 to 124, omitting the 200 to 224 rows (it is
    # twice as long).
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 1, 2, ... , 23, 24, 100, 101, ..., 123, 124],
        "label": [0, 1, 0, 1, ..., 0, 1],
    }
    """
    values_1 = list(range(DATASET_LENGTH))
    labels = [x % 2 for x in values_1]
    dataset_1 = Dataset.from_dict({"data": values_1, "label": labels})

    values_2 = [x + 100 for x in values_1] + [x + 200 for x in values_1]
    dataset_2 = Dataset.from_dict({"data": values_2, "label": labels + labels})

    return ConcatDataset([dataset_1, dataset_2])


def test_round_robin_batch_sampler(dummy_concat_dataset: ConcatDataset) -> None:
    # [FUNCTION DOC]
    # Creates sequential batch samplers of size 4 with drop_last for each sub-dataset, wraps them in the round-robin
    # sampler, and asserts: the total number of batches is 12 (6 from each, as sampling stops when the shorter first
    # sub-dataset runs out), the sampler length equals that count, and batches alternate, even positions holding only
    # first-dataset rows (data below 100) and odd positions only second-dataset rows (data at least 100).
    batch_size = 4
    batch_sampler_1 = BatchSampler(
        SequentialSampler(range(len(dummy_concat_dataset.datasets[0]))), batch_size=batch_size, drop_last=True
    )
    batch_sampler_2 = BatchSampler(
        SequentialSampler(range(len(dummy_concat_dataset.datasets[1]))), batch_size=batch_size, drop_last=True
    )

    sampler = RoundRobinBatchSampler(dataset=dummy_concat_dataset, batch_samplers=[batch_sampler_1, batch_sampler_2])
    batches = list(iter(sampler))

    # Despite the second dataset being larger (2 * DATASET_LENGTH), we still only sample DATASET_LENGTH // batch_size batches from each dataset
    # because the RoundRobinBatchSampler should stop sampling once it has sampled all elements from one dataset
    assert len(batches) == 2 * DATASET_LENGTH // batch_size
    assert len(sampler) == len(batches)

    # Assert that batches are produced in a round-robin fashion
    for i in range(0, len(batches), 2):
        # Batch from the first part of the dataset
        batch_1 = batches[i]
        assert all(dummy_concat_dataset[idx]["data"] < 100 for idx in batch_1), (
            f"Batch {i} contains data from the second part of the dataset: {[dummy_concat_dataset[idx]['data'] for idx in batch_1]}"
        )

        # Batch from the second part of the dataset
        batch_2 = batches[i + 1]
        assert all(dummy_concat_dataset[idx]["data"] >= 100 for idx in batch_2), (
            f"Batch {i + 1} contains data from the first part of the dataset: {[dummy_concat_dataset[idx]['data'] for idx in batch_2]}"
        )


def test_round_robin_batch_sampler_value_error(dummy_concat_dataset: ConcatDataset) -> None:
    # [FUNCTION DOC]
    # Asserts that passing three batch samplers for a two-part ConcatDataset raises ValueError stating that the number
    # of batch samplers must match the number of datasets.
    batch_size = 4
    batch_sampler_1 = BatchSampler(SequentialSampler(range(DATASET_LENGTH)), batch_size=batch_size, drop_last=True)
    batch_sampler_2 = BatchSampler(SequentialSampler(range(DATASET_LENGTH)), batch_size=batch_size, drop_last=True)
    batch_sampler_3 = BatchSampler(SequentialSampler(range(DATASET_LENGTH)), batch_size=batch_size, drop_last=True)

    with pytest.raises(
        ValueError, match="The number of batch samplers must match the number of datasets in the ConcatDataset"
    ):
        RoundRobinBatchSampler(
            dataset=dummy_concat_dataset, batch_samplers=[batch_sampler_1, batch_sampler_2, batch_sampler_3]
        )
