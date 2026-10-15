# [MODULE DOC]
# Pytest tests for NoDuplicatesBatchSampler (yields batches of dataset indices such that no stringified column value
# is shared between any two rows of a batch, ignoring configured label columns), its interaction with
# ProportionalBatchSampler, and the private value-to-string key helper used for deduplication. Skips the module if the
# datasets library is missing. Defines a module constant for the hash precomputation worker count: 0 on datasets 4.1.0
# or newer, else 1 (older versions reject 0). Several tests run in two modes: plain value comparison, or precomputed
# xxhash row hashes (skipped if xxhash is not installed).
from __future__ import annotations

import importlib.util
import io
import random
import wave

import numpy as np
import pytest
import torch
from packaging.version import parse as parse_version
from torch.utils.data import ConcatDataset

import sentence_transformers.base.sampler as sampler_module
from sentence_transformers.base.sampler import NoDuplicatesBatchSampler, ProportionalBatchSampler, _sample_value_str
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    import datasets
    from datasets import Audio, Dataset, Features, Value

    # datasets < 4.1.0 does not support num_proc=0
    PRECOMPUTE_NUM_PROC = 0 if parse_version(datasets.__version__) >= parse_version("4.1.0") else 1
else:
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def _wav_bytes(frequency: int) -> bytes:
    # [FUNCTION DOC]
    # HELPER: Returns the bytes of an in-memory mono 16-bit 16 kHz WAV file containing a 0.1-second (1600-sample) sine
    # wave of the given frequency at amplitude 10000. No file I/O.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        samples = np.sin(2 * np.pi * frequency * np.linspace(0, 0.1, 1600, endpoint=False))
        writer.writeframes((samples * 10_000).astype("<i2").tobytes())
    return buffer.getvalue()


@pytest.fixture
def dummy_dataset() -> Dataset:
    # [FUNCTION DOC]
    # HELPER: Fixture: Dataset of 100 rows whose "data" column holds each integer 0 to 49 exactly twice in random
    # order (uses the global Python RNG, so order differs per run) and whose "label" alternates 0/1.
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 47, 3, 30, 3, ... 2],
        "label": [0, 1, 0, 1, ..., 0, 1],
    }
    """

    # Create a list of two 0's, two 1's, two 2's, ... two 49's. Then shuffle.
    values = [j for i in range(50) for j in (i, i)]
    random.shuffle(values)
    data = {"data": values, "label": [i % 2 for i in range(100)]}
    return Dataset.from_dict(data)


@pytest.fixture
def dummy_duplicates_dataset() -> Dataset:
    # [FUNCTION DOC]
    # HELPER: Fixture: Dataset of 18 rows, 10 identical rows with anchor "anchor_1" and positive "positive_1", then 8
    # identical rows with "anchor_2" and "positive_2".
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "anchor": ["anchor_1", "anchor_1", "anchor_1", ... "anchor_2", "anchor_2"],
        "positive": ["positive_1", "positive_1", "positive_1", ... "positive_2", "positive_2"],
    }
    """
    values = [{"anchor": "anchor_1", "positive": "positive_1"}] * 10 + [
        {"anchor": "anchor_2", "positive": "positive_2"}
    ] * 8
    return Dataset.from_list(values)


def is_xxhash_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns whether the xxhash package can be imported (checks the import spec only).
    return importlib.util.find_spec("xxhash") is not None


def _reference_no_duplicates_batches(
    dataset: Dataset, batch_size: int, drop_last: bool, seed: int, valid_label_columns: list[str] | None = None
) -> list[list[int]]:
    # [FUNCTION DOC]
    # HELPER: Reference reimplementation of the original dict-based no-duplicates batching, used to check the
    # optimised sampler. Inputs: dataset, batch size, drop_last, seed and optional label columns. Removes any label
    # columns present. Seeds a torch generator and creates an insertion-ordered set of indices from a random
    # permutation of the row count. Repeatedly builds a batch by scanning the remaining indices in order: a row's
    # values are the strings of all its column values except a "dataset_name" column; skip it if any value is already
    # in the batch's value set, otherwise append it; if the batch reached batch size, keep it and stop scanning;
    # otherwise add its values to the set. If the scan finishes without filling the batch, keep the partial batch only
    # when drop_last is off. Then remove the batch's indices from the remaining set and repeat until none remain.
    # Returns the list of batches.
    """Reference implementation of the historical dict-based iteration logic."""
    if label_columns := set(dataset.column_names) & set(valid_label_columns or []):
        dataset = dataset.remove_columns(list(label_columns))

    generator = torch.Generator()
    generator.manual_seed(seed)

    def get_sample_values(index: int) -> set[str]:
        # [FUNCTION DOC]
        # HELPER: Returns the set of string forms of a row's column values, excluding a "dataset_name" column.
        return {str(value) for key, value in dataset[index].items() if key != "dataset_name"}

    def has_overlap(sample_values: set[str], batch_values: set[str]) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns True if the two string sets share any element.
        return not sample_values.isdisjoint(batch_values)

    remaining_indices = dict.fromkeys(torch.randperm(len(dataset), generator=generator).tolist())
    batches: list[list[int]] = []
    while remaining_indices:
        batch_values: set[str] = set()
        batch_indices: list[int] = []
        for index in remaining_indices:
            sample_values = get_sample_values(index)
            if has_overlap(sample_values, batch_values):
                continue
            batch_indices.append(index)
            if len(batch_indices) == batch_size:
                batches.append(batch_indices)
                break
            batch_values.update(sample_values)
        else:
            if not drop_last:
                batches.append(batch_indices)

        for index in batch_indices:
            del remaining_indices[index]
    return batches


@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_group_by_label_batch_sampler_label_a(dummy_dataset: Dataset, precompute_hashes: bool) -> None:
    # [FUNCTION DOC]
    # Parametrised over hash precomputation. With batch size 10, drop_last on and "label" ignored, asserts every
    # yielded batch has 10 rows and no repeated "data" values.
    # KNOWN ISSUE: The test name refers to the group-by-label sampler, but it exercises NoDuplicatesBatchSampler.
    batch_size = 10

    sampler_kwargs = {}
    if precompute_hashes:
        if not is_xxhash_available():
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": PRECOMPUTE_NUM_PROC,
            "precompute_batch_size": 10,
        }

    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset,
        batch_size=batch_size,
        drop_last=True,
        valid_label_columns=["label"],
        **sampler_kwargs,
    )

    batches = list(iter(sampler))

    # Assert all batch sizes are correct
    assert all(len(batch) == batch_size for batch in batches)

    # Assert batches contain no duplicate values
    for batch in batches:
        batch_values = [dummy_dataset[i]["data"] for i in batch]
        assert len(batch_values) == len(set(batch_values)), f"Batch {batch} contains duplicate values: {batch_values}"


@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_proportional_no_duplicates(
    dummy_duplicates_dataset: Dataset, drop_last: bool, precompute_hashes: bool
) -> None:
    # [FUNCTION DOC]
    # Parametrised over drop_last and hash precomputation. Builds two no-duplicates samplers of batch size 2 on the
    # duplicates dataset (one ignoring "anchor", one ignoring "positive"), combines them over a ConcatDataset of the
    # dataset twice with ProportionalBatchSampler (seed 12), and asserts: with drop_last, 16 batches all of size 2
    # covering 32 indices (regression for a crash in issue 2816); without, 18 batches covering 34 indices.
    batch_size = 2
    sampler_kwargs = {}
    if precompute_hashes:
        if not is_xxhash_available():
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": PRECOMPUTE_NUM_PROC,
            "precompute_batch_size": 10,
        }
    sampler_1 = NoDuplicatesBatchSampler(
        dataset=dummy_duplicates_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        valid_label_columns=["anchor"],
        **sampler_kwargs,
    )
    sampler_2 = NoDuplicatesBatchSampler(
        dataset=dummy_duplicates_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        valid_label_columns=["positive"],
        **sampler_kwargs,
    )

    concat_dataset = ConcatDataset([dummy_duplicates_dataset, dummy_duplicates_dataset])

    batch_sampler = ProportionalBatchSampler(
        concat_dataset, [sampler_1, sampler_2], generator=torch.Generator(), seed=12
    )
    batches = list(iter(batch_sampler))

    if drop_last:
        # If we drop the last batch (i.e. incomplete batches), we should have 16 batches out of the 18 possible,
        # because of the duplicates being skipped by the NoDuplicatesBatchSampler.
        # Notably, we should not crash like reported in #2816.
        assert len(batches) == 16
        # All batches are the same size: 2
        assert all(len(batch) == batch_size for batch in batches)
        assert len(sum(batches, [])) == 32
    else:
        # If we don't drop incomplete batches, we should be able to do 18 batches, and get more data.
        # Note: we don't get all data, because the NoDuplicatesBatchSampler will estimate the number of batches
        # and it would require more (non-complete) batches to get all data.
        assert len(batches) == 18
        assert len(sum(batches, [])) == 34


@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_no_duplicates_batch_sampler_matches_reference_algorithm(
    dummy_dataset: Dataset, drop_last: bool, precompute_hashes: bool
) -> None:
    # [FUNCTION DOC]
    # Parametrised over drop_last and hash precomputation. Asserts the sampler (batch size 10, seed 1337, "label"
    # ignored) yields exactly the same batches as the reference implementation, and the same total number of indices.
    batch_size = 10
    seed = 1337
    valid_label_columns = ["label"]
    sampler_kwargs = {}
    if precompute_hashes:
        if not is_xxhash_available():
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": PRECOMPUTE_NUM_PROC,
            "precompute_batch_size": 10,
        }

    expected_batches = _reference_no_duplicates_batches(
        dataset=dummy_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        seed=seed,
        valid_label_columns=valid_label_columns,
    )

    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        valid_label_columns=valid_label_columns,
        generator=torch.Generator(),
        seed=seed,
        **sampler_kwargs,
    )

    actual_batches = list(iter(sampler))
    assert actual_batches == expected_batches
    assert len(sum(actual_batches, [])) == len(sum(expected_batches, []))


@pytest.mark.skipif(not is_xxhash_available(), reason="xxhash not installed")
def test_no_duplicates_batch_sampler_precomputed_hashes_are_int64(dummy_dataset: Dataset) -> None:
    # [FUNCTION DOC]
    # Requires xxhash. Builds a sampler with hash precomputation, calls its private hash-building method directly, and
    # asserts the private row hash array exists and has dtype int64.
    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset,
        batch_size=10,
        drop_last=True,
        valid_label_columns=["label"],
        precompute_hashes=True,
        precompute_num_proc=PRECOMPUTE_NUM_PROC,
        precompute_batch_size=10,
    )
    sampler._build_hashes()

    assert sampler._row_hashes is not None
    assert sampler._row_hashes.dtype == np.int64


@pytest.mark.skipif(not is_xxhash_available(), reason="xxhash not installed")
def test_no_duplicates_batch_sampler_list_values_match_between_hash_and_non_hash() -> None:
    # [FUNCTION DOC]
    # Requires xxhash. On a two-row dataset where a list value in one row ("a","b") contains a string equal to a
    # scalar in another column of the other row ("a"), asserts the default and hashed samplers (batch size 2, seed
    # 1337, no drop_last) yield identical batches, and that this is one batch with both rows (a list value is compared
    # as its whole string form, not per element).
    # This shape intentionally creates cross-column overlap:
    # row0 has list value ["a", "b"], row1 has scalar "a" in another column.
    # Hash/non-hash modes must still produce identical batching decisions.
    dataset = Dataset.from_list(
        [
            {"anchor": ["a", "b"], "positive": "p0"},
            {"anchor": ["c", "d"], "positive": "a"},
        ]
    )

    sampler_kwargs = {
        "dataset": dataset,
        "batch_size": 2,
        "drop_last": False,
        "generator": torch.Generator(),
        "seed": 1337,
    }
    default_sampler = NoDuplicatesBatchSampler(**sampler_kwargs)
    hashed_sampler = NoDuplicatesBatchSampler(
        **sampler_kwargs,
        precompute_hashes=True,
        precompute_num_proc=PRECOMPUTE_NUM_PROC,
        precompute_batch_size=2,
    )

    default_batches = list(iter(default_sampler))
    hashed_batches = list(iter(hashed_sampler))

    assert hashed_batches == default_batches
    assert len(default_batches) == 1
    assert len(default_batches[0]) == 2


@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_no_duplicates_batch_sampler_deduplicates_images(precompute_hashes: bool) -> None:
    # [FUNCTION DOC]
    # Parametrised over hash precomputation; skips if PIL is missing. Builds 16 rows with distinct query strings and
    # an image column cycling over four solid-colour 4x4 images, samples with batch size 8 and seed 0, and asserts all
    # 16 rows are yielded, every batch has exactly 4 rows, and the images in each batch have distinct pixel bytes
    # (images must be deduplicated by content, not object identity).
    # PIL images stringify to a fresh object address on every dataset access, so without
    # content digests every image row looks unique and the dedup silently no-ops.
    pil_image = pytest.importorskip("PIL.Image")
    sampler_kwargs = {}
    if precompute_hashes:
        if not is_xxhash_available():
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": PRECOMPUTE_NUM_PROC,
            "precompute_batch_size": 10,
        }

    images = [pil_image.new("RGB", (4, 4), color) for color in ("red", "green", "blue", "yellow")]
    dataset = Dataset.from_dict(
        {"query": [f"query_{i}" for i in range(16)], "image": [images[i % 4] for i in range(16)]}
    )
    sampler = NoDuplicatesBatchSampler(
        dataset=dataset,
        batch_size=8,
        drop_last=False,
        generator=torch.Generator(),
        seed=0,
        **sampler_kwargs,
    )

    batches = list(iter(sampler))

    assert sum(len(batch) for batch in batches) == 16
    for batch in batches:
        # Only 4 unique images exist, so a batch of 8 must stop at 4 distinct rows.
        assert len(batch) == 4
        pixels = {dataset[i]["image"].tobytes() for i in batch}
        assert len(pixels) == len(batch)


@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_no_duplicates_batch_sampler_deduplicates_audio(precompute_hashes: bool) -> None:
    # [FUNCTION DOC]
    # Parametrised over hash precomputation; skips if torchcodec is missing. Builds 16 rows with distinct queries and
    # an Audio feature column cycling over four WAV clips of different frequencies, samples with batch size 8 and seed
    # 0, and asserts all 16 rows are yielded and each batch has exactly 4 rows (decoded audio must be deduplicated by
    # content).
    # Audio / Video features return torchcodec decoders whose pickle embeds a torch storage key
    # taken from the object address, so digesting the decoder itself is fresh on every access.
    pytest.importorskip("torchcodec")
    sampler_kwargs = {}
    if precompute_hashes:
        if not is_xxhash_available():
            pytest.skip("xxhash not installed")
        sampler_kwargs = {
            "precompute_hashes": True,
            "precompute_num_proc": PRECOMPUTE_NUM_PROC,
            "precompute_batch_size": 10,
        }

    clips = [_wav_bytes(frequency) for frequency in (440, 660, 880, 1100)]
    dataset = Dataset.from_dict(
        {
            "query": [f"query_{i}" for i in range(16)],
            "audio": [{"bytes": clips[i % 4], "path": None} for i in range(16)],
        },
        features=Features({"query": Value("string"), "audio": Audio()}),
    )
    sampler = NoDuplicatesBatchSampler(
        dataset=dataset,
        batch_size=8,
        drop_last=False,
        generator=torch.Generator(),
        seed=0,
        **sampler_kwargs,
    )

    batches = list(iter(sampler))

    assert sum(len(batch) for batch in batches) == 16
    for batch in batches:
        # Only 4 unique clips exist, so a batch of 8 must stop at 4 distinct rows.
        assert len(batch) == 4


def test_sample_value_str_decoder_digest_is_stable_across_accesses() -> None:
    # [FUNCTION DOC]
    # Skips if torchcodec is missing. Asserts the value-to-string helper returns the same key for two separate
    # accesses of the same decoded audio cell and different keys for different clips.
    # The decoder is rebuilt per access, so its digest must come from the encoded source it wraps.
    pytest.importorskip("torchcodec")
    dataset = Dataset.from_dict(
        {"audio": [{"bytes": _wav_bytes(440), "path": None}, {"bytes": _wav_bytes(880), "path": None}]},
        features=Features({"audio": Audio()}),
    )
    assert _sample_value_str(dataset[0]["audio"]) == _sample_value_str(dataset[0]["audio"])
    assert _sample_value_str(dataset[0]["audio"]) != _sample_value_str(dataset[1]["audio"])


def test_sample_value_str_plain_values_keep_str() -> None:
    # [FUNCTION DOC]
    # Asserts the helper returns plain string conversion for strings, integers, and lists of strings or floats
    # (backward-compatible keys).
    # The historical text/number dedup keys must stay byte-identical.
    assert _sample_value_str("a sentence") == "a sentence"
    assert _sample_value_str(5) == "5"
    assert _sample_value_str(["a", "b"]) == str(["a", "b"])
    assert _sample_value_str([1.0, 0.5]) == str([1.0, 0.5])


def test_sample_value_str_numpy_scalars_match_python_scalars() -> None:
    # [FUNCTION DOC]
    # Asserts NumPy integer, float and boolean scalars map to the same key as the equivalent Python scalars, while a
    # NumPy array does not map to its plain string form (it uses a content digest).
    # Every value went through str() before this function existed, so numpy scalars shared a key
    # with their python counterparts. Arrays stay on the digest path.
    assert _sample_value_str(np.int64(5)) == _sample_value_str(5)
    assert _sample_value_str(np.float64(0.5)) == _sample_value_str(0.5)
    assert _sample_value_str(np.bool_(True)) == _sample_value_str(True)
    assert _sample_value_str(np.arange(3)) != str(np.arange(3))


def test_sample_value_str_digests_object_content() -> None:
    # [FUNCTION DOC]
    # Asserts that for 10000-element arrays whose truncated string forms are identical, the helper gives equal keys
    # for equal content and different keys when one element differs.
    # str() of a large array is truncated to its head and tail, so distinct arrays could collide.
    # Content digests keep equal content equal and distinct content distinct.
    same_a = np.arange(10_000)
    same_b = np.arange(10_000)
    different = np.arange(10_000)
    different[5_000] = -1
    assert str(same_a) == str(different)
    assert _sample_value_str(same_a) == _sample_value_str(same_b)
    assert _sample_value_str(same_a) != _sample_value_str(different)


def test_sample_value_str_unpicklable_falls_back_to_str() -> None:
    # [FUNCTION DOC]
    # Asserts the helper falls back to plain string conversion for an unpicklable object (a lambda).
    unpicklable = lambda: None  # noqa: E731
    assert _sample_value_str(unpicklable) == str(unpicklable)


def test_no_duplicates_batch_sampler_uses_int64_indices_when_int32_range_is_exceeded(
    dummy_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC]
    # Simulates a dataset too large for int32 indices: monkeypatches the sampler module's NumPy integer-info lookup so
    # int32 reports a maximum of 8, and wraps its torch random permutation function to record the requested dtype.
    # Iterates a sampler on the 100-row dataset and asserts the permutation was requested with int64. The patches
    # apply to the global numpy and torch modules for the duration of the test (restored by monkeypatch).
    # Simulate "too many rows for int32" without allocating a huge dataset.
    real_iinfo = sampler_module.np.iinfo

    class _FakeIInfo:
        # [CLASS DOC]
        # HELPER: Minimal stand-in for NumPy's integer info object exposing only a max attribute.
        def __init__(self, max_value: int) -> None:
            # [FUNCTION DOC]
            # HELPER: Stores the given maximum value as the max attribute.
            self.max = max_value

    def fake_iinfo(dtype):
        # [FUNCTION DOC]
        # HELPER: Fake integer-info function: returns the stand-in with max 8 for int32, otherwise delegates to the
        # real function.
        if dtype is np.int32:
            return _FakeIInfo(max_value=8)
        return real_iinfo(dtype)

    captured_dtypes: dict[str, object] = {}
    real_randperm = sampler_module.torch.randperm

    def fake_randperm(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the dtype keyword argument (None if absent) in the captured dict, then delegates to the real
        # random permutation and returns its result.
        # Capture dtype selected for shuffled row indices.
        captured_dtypes["index_dtype"] = kwargs.get("dtype")
        return real_randperm(*args, **kwargs)

    monkeypatch.setattr(sampler_module.np, "iinfo", fake_iinfo)
    monkeypatch.setattr(sampler_module.torch, "randperm", fake_randperm)

    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset,
        batch_size=10,
        drop_last=True,
        valid_label_columns=["label"],
        generator=torch.Generator(),
        seed=123,
    )
    _ = list(iter(sampler))

    assert captured_dtypes["index_dtype"] == torch.int64
