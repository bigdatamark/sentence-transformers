# [MODULE DOC]
# Tests for sentence_transformers.util.quantization: quantize_embeddings for binary/ubinary bit packing (any
# dimension), int8/uint8 calibrated scaling with clipping, empty inputs and multi-vector inputs, and
# semantic_search_faiss precision validation and padding handling (skipped when faiss is not installed).
# Defines fixed seeded random queries (2x16) and calibration embeddings (100x16).
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sentence_transformers.util.quantization import quantize_embeddings, semantic_search_faiss


@pytest.mark.parametrize("precision", ["binary", "ubinary"])
@pytest.mark.parametrize(
    ("n_samples", "embedding_dim"),
    [
        (4, 10),  # 40 total bits -> 5 bytes. 5 % 4 != 0  -- old code raises ValueError
        (3, 10),  # 30 bits -> ceil = 4 bytes. 4 % 3 != 0 -- old code raises ValueError
        (5, 12),  # 60 bits -> 8 bytes. 8 % 5 != 0        -- old code raises ValueError
    ],
)
def test_binary_quantize_non_multiple_of_8_does_not_raise(precision: str, n_samples: int, embedding_dim: int) -> None:
    # [FUNCTION DOC]
    # Parametrized over binary/ubinary and dimensions not divisible by 8; quantizes seeded random embeddings and
    # asserts the packed shape is samples by ceiling of dimension over 8, with int8 dtype for binary and uint8 for
    # ubinary.
    """quantize_embeddings must not crash for embedding dimensions not divisible by 8."""
    rng = np.random.default_rng(seed=0)
    embeddings = rng.standard_normal((n_samples, embedding_dim)).astype(np.float32)

    result = quantize_embeddings(embeddings, precision)

    expected_packed_dim = -(-embedding_dim // 8)  # ceil(embedding_dim / 8)
    assert result.shape == (n_samples, expected_packed_dim), (
        f"Expected shape ({n_samples}, {expected_packed_dim}), got {result.shape}"
    )
    if precision == "binary":
        assert result.dtype == np.int8
    else:
        assert result.dtype == np.uint8


@pytest.mark.parametrize("precision", ["binary", "ubinary"])
@pytest.mark.parametrize(
    ("n_samples", "embedding_dim"),
    [
        (4, 8),
        (2, 16),
        (3, 24),
    ],
)
def test_binary_quantize_multiple_of_8_shape(precision: str, n_samples: int, embedding_dim: int) -> None:
    # [FUNCTION DOC]
    # Same as above for dimensions divisible by 8, expecting dimension divided by 8 packed columns.
    """For dims divisible by 8, the packed shape must equal (n, dim // 8)."""
    rng = np.random.default_rng(seed=1)
    embeddings = rng.standard_normal((n_samples, embedding_dim)).astype(np.float32)

    result = quantize_embeddings(embeddings, precision)

    assert result.shape == (n_samples, embedding_dim // 8)
    if precision == "binary":
        assert result.dtype == np.int8
    else:
        assert result.dtype == np.uint8


@pytest.mark.parametrize("precision", ["binary", "ubinary"])
def test_binary_quantize_row_independence(precision: str) -> None:
    # [FUNCTION DOC]
    # Asserts an all-positive row packs to 127 (binary, offset by 128) or 255 (ubinary) and an all-negative row to
    # -128 or 0, showing rows are packed independently.
    """Each row is packed independently. An all-positive row must not bleed into an all-negative one."""
    embedding_dim = 8
    embeddings = np.array(
        [[1.0] * embedding_dim, [-1.0] * embedding_dim],
        dtype=np.float32,
    )
    result = quantize_embeddings(embeddings, precision)

    if precision == "binary":
        # all-one bits packed = 0xFF = 255. 255 - 128 = 127 (max int8)
        assert result[0, 0] == 127, f"All-positive row: expected 127, got {result[0, 0]}"
        # all-zero bits packed = 0x00 = 0. 0 - 128 = -128 (min int8)
        assert result[1, 0] == -128, f"All-negative row: expected -128, got {result[1, 0]}"
    else:  # ubinary
        assert result[0, 0] == 0xFF, f"All-positive row: expected 255, got {result[0, 0]}"
        assert result[1, 0] == 0x00, f"All-negative row: expected 0, got {result[1, 0]}"


@pytest.mark.parametrize("precision", ["int8", "uint8"])
def test_quantize_clips_out_of_range_values(precision: str) -> None:
    # [FUNCTION DOC]
    # Quantizes a row with values outside the calibration ranges and asserts results saturate at the dtype
    # limits instead of wrapping (exact expected int8 and uint8 rows).
    """Values outside the calibration range must saturate, not wrap around, on cast.

    Without clipping, a value that should quantize above 255 silently wraps
    (e.g. 300 -> 44 in uint8) instead of saturating at the dtype's min/max.
    See issue #3159 / PR #2865.
    """
    calibration_embeddings = np.array([[1, 20, -3], [4, 5, -60]], dtype=np.float32)
    dataset = np.array([[-1, 15, 1]], dtype=np.float32)  # -1 and 1 fall outside calibration range

    result = quantize_embeddings(dataset, precision, calibration_embeddings=calibration_embeddings)

    if precision == "int8":
        expected = np.array([[-128, 42, 127]], dtype=np.int8)
    else:
        expected = np.array([[0, 170, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_quantize_multi_vector_handles_empty_matrices() -> None:
    # [FUNCTION DOC]
    # Quantizes a list of a 10x16 matrix and an empty 0x16 matrix in int8, uint8, binary and ubinary, asserting
    # the empty matrix yields an empty output with the right width and dtype; also asserts a list of only empty
    # matrices quantizes to int8 without error.
    """A (0, dim) matrix (e.g. a fully-masked multi-vector document) must quantize to the
    correctly-shaped empty output instead of crashing packbits / calibration."""
    rng = np.random.RandomState(0)
    matrices = [rng.randn(10, 16).astype(np.float32), np.zeros((0, 16), dtype=np.float32)]
    for precision, dtype, out_dim in [
        ("int8", np.int8, 16),
        ("uint8", np.uint8, 16),
        ("binary", np.int8, 2),
        ("ubinary", np.uint8, 2),
    ]:
        quantized = quantize_embeddings(matrices, precision=precision)
        assert quantized[0].shape[0] == 10
        assert quantized[1].shape == (0, out_dim)
        assert quantized[1].dtype == dtype

    # A corpus of only empty matrices must not crash int8 calibration either.
    all_empty = quantize_embeddings([np.zeros((0, 16), dtype=np.float32)], precision="int8")
    assert all_empty[0].shape == (0, 16)


def test_quantize_empty_list_returns_empty_list() -> None:
    # [FUNCTION DOC]
    # Asserts an empty list input returns an empty list for every precision.
    """An empty input list (e.g. encode() of zero texts) must return an empty list, not IndexError."""
    for precision in ("int8", "uint8", "binary", "ubinary", "float32"):
        assert quantize_embeddings([], precision=precision) == []


skip_without_faiss = pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")

QUERIES = np.random.default_rng(seed=1).standard_normal((2, 16), dtype=np.float32)
CALIBRATION = np.random.default_rng(seed=2).standard_normal((100, 16), dtype=np.float32)


def _corpus(n_docs: int, precision: str, seed: int = 0) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: Generates seeded random n_docs x 16 embeddings and quantizes them to the given precision using the
    # shared calibration embeddings.
    embeddings = np.random.default_rng(seed=seed).standard_normal((n_docs, 16), dtype=np.float32)
    return quantize_embeddings(embeddings, precision=precision, calibration_embeddings=CALIBRATION)


@skip_without_faiss
@pytest.mark.parametrize("corpus_precision", ["int8", "binary", "float16"])
def test_semantic_search_faiss_rejects_unsupported_precision(corpus_precision: str) -> None:
    # [FUNCTION DOC]
    # Parametrized over int8, binary and float16; asserts semantic_search_faiss raises a ValueError mentioning
    # corpus_precision for these unsupported corpus precisions.
    """An unsupported ``corpus_precision`` must say so, not fall through to an unrelated error.

    Without validation, no index construction branch matches and the call dies on
    ``AttributeError: 'NoneType' object has no attribute 'add'``.
    """
    with pytest.raises(ValueError, match="corpus_precision"):
        semantic_search_faiss(
            QUERIES,
            corpus_embeddings=_corpus(4, "uint8"),
            corpus_precision=corpus_precision,
            top_k=2,
        )


@skip_without_faiss
@pytest.mark.parametrize("corpus_precision", ["ubinary", "uint8"])
@pytest.mark.parametrize("rescore", [True, False])
def test_semantic_search_faiss_drops_padded_indices(corpus_precision: str, rescore: bool) -> None:
    # [FUNCTION DOC]
    # Parametrized over ubinary/uint8 and rescoring; searches a 3-document corpus with top 5 and asserts no -1
    # corpus ids appear and each query returns exactly documents 0, 1 and 2.
    """A corpus smaller than ``top_k`` must not produce ``corpus_id: -1`` entries.

    FAISS pads short result sets with index -1, which would resolve to the last corpus
    entry when used to index the corpus.
    """
    results, _ = semantic_search_faiss(
        QUERIES,
        corpus_embeddings=_corpus(3, corpus_precision),
        corpus_precision=corpus_precision,
        top_k=5,  # deliberately larger than the 3-document corpus
        rescore=rescore,
        rescore_multiplier=2,
        calibration_embeddings=CALIBRATION,
    )

    for query_results in results:
        corpus_ids = [entry["corpus_id"] for entry in query_results]
        assert -1 not in corpus_ids
        # Every existing document is returned exactly once, and nothing else is.
        assert sorted(corpus_ids) == [0, 1, 2]


@skip_without_faiss
@pytest.mark.parametrize("rescore", [True, False])
def test_semantic_search_faiss_empty_corpus(rescore: bool) -> None:
    # [FUNCTION DOC]
    # Asserts an empty ubinary corpus gives an empty result list per query, with or without rescoring.
    """An empty corpus yields empty result lists rather than phantom hits."""
    results, _ = semantic_search_faiss(
        QUERIES,
        corpus_embeddings=np.empty((0, 2), dtype=np.uint8),
        corpus_precision="ubinary",
        top_k=5,
        rescore=rescore,
        rescore_multiplier=2,
    )

    assert results == [[], []]


@skip_without_faiss
@pytest.mark.parametrize("corpus_precision", ["ubinary", "uint8"])
def test_semantic_search_faiss_returns_top_k_when_corpus_is_large_enough(corpus_precision: str) -> None:
    # [FUNCTION DOC]
    # Asserts a 50-document corpus returns exactly 5 valid hits per query with rescoring.
    """The padding check must not shorten results for a corpus larger than ``top_k``."""
    results, _ = semantic_search_faiss(
        QUERIES,
        corpus_embeddings=_corpus(50, corpus_precision, seed=3),
        corpus_precision=corpus_precision,
        top_k=5,
        rescore=True,
        rescore_multiplier=2,
        calibration_embeddings=CALIBRATION,
    )

    for query_results in results:
        assert len(query_results) == 5
        assert all(0 <= entry["corpus_id"] < 50 for entry in query_results)
