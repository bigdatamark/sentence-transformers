# [MODULE DOC]
# Tests for sentence_transformers.util.mine_hard_negatives, which, for each (anchor, positive) row of a dataset,
# retrieves candidate passages from a corpus (the dataset's positives or a separate corpus) with an embedding
# model (optionally FAISS, a cross-encoder rescorer, multi-processing and an embedding cache), filters them by
# rank window (range_min/range_max), score bounds and absolute/relative margins against the positive, samples
# negatives ("top" or "random"), and emits them as triplet, n-tuple, labeled-pair or labeled-list datasets with
# optional scores. Covers column naming, prompts, scores consistency, multiple positives, parameter validation,
# FAISS padding edge cases and deprecated parameters. Uses eight fixed queries with matching passages plus eight
# non-matching passages; skipped at module level when datasets is unavailable.
from __future__ import annotations

import importlib.util
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from sentence_transformers import CrossEncoder
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.util import is_datasets_available, mine_hard_negatives

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip("The datasets library is not available.", allow_module_level=True)

# Test data
QUERIES = [
    "What is the capital of France?",
    "Who wrote the novel 1984?",
    "What is the largest planet in our solar system?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "What is the chemical symbol for gold?",
    "Who developed the theory of relativity?",
    "What is the tallest mountain in the world?",
]

PASSAGES = [
    "Paris is the capital of France and one of the most populated cities in Europe.",
    "George Orwell wrote the dystopian novel Nineteen Eighty-Four (1984), which was published in 1949.",
    "Jupiter is the largest planet in our solar system and the fifth planet from the sun.",
    "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "The chemical symbol for gold is Au, derived from the Latin word 'Aurum'.",
    "Albert Einstein developed the theory of relativity, which revolutionized our understanding of space, time, and gravity.",
    "Mount Everest is the tallest mountain in the world, with a peak that reaches 8,848 meters (29,029 feet) above sea level.",
    # The following passages can be seen as non-matching negatives
    "Madrid is the capital of Spain and its largest city.",
    "J.K. Rowling wrote the Harry Potter series of fantasy novels.",
    "Saturn is the sixth planet from the Sun and the second-largest in the Solar System.",
    "Vincent van Gogh is famous for his post-impressionist paintings like 'Starry Night'.",
    "Sound travels at approximately 343 meters per second in air at standard temperature and pressure.",
    "The chemical symbol for silver is Ag, derived from the Latin word 'Argentum'.",
    "Isaac Newton formulated the laws of motion and universal gravitation.",
    "K2, also known as Mount Godwin-Austen, is the second-highest mountain in the world.",
]


@pytest.fixture(scope="session")
def queries():
    # [FUNCTION DOC]
    # Session fixture returning a copy of the query list.
    """Return a list of sample queries."""
    return QUERIES.copy()


@pytest.fixture(scope="session")
def passages():
    # [FUNCTION DOC]
    # Session fixture returning a copy of the passage list.
    """Return a list of sample passages."""
    return PASSAGES.copy()


@pytest.fixture(scope="session")
def dataset(queries, passages):
    # [FUNCTION DOC]
    # Session fixture returning a dataset of the eight queries with their matching passages.
    """Return a sample dataset with matching queries and passages."""
    return Dataset.from_dict(
        {
            "query": queries,
            "passage": passages[: len(queries)],
        }
    )


@pytest.fixture(scope="session")
def multiple_positive_queries_dataset(queries, passages):
    # [FUNCTION DOC]
    # Session fixture where each query appears twice with two prefixed ("Copy 0", "Copy 1") versions of its passage,
    # so each query has two distinct positives.
    """Return a sample dataset with multiple matching passages for each query."""
    return Dataset.from_dict(
        {
            "query": queries + queries,
            "passage": [f"Copy {idx}, {passage}" for idx in range(2) for passage in passages[: len(queries)]],
        }
    )


@pytest.fixture(scope="session")
def multiple_positive_passages_dataset(queries, passages):
    # [FUNCTION DOC]
    # Session fixture with two prefixed copies of each query paired with the same passage, so each passage is the
    # positive of two distinct queries.
    # KNOWN ISSUE: The docstring says each query has multiple matching passages, but here each passage has multiple
    # matching queries (the queries are the ones duplicated with prefixes).
    """Return a sample dataset with multiple matching passages for each query."""
    return Dataset.from_dict(
        {
            "query": [f"Copy {idx}, {query}" for idx in range(2) for query in queries],
            "passage": passages[: len(queries)] + passages[: len(queries)],
        }
    )


@pytest.fixture(scope="session")
def multiple_positive_dataset(queries, passages):
    # [FUNCTION DOC]
    # Session fixture repeating the whole query-passage dataset twice (exact duplicate rows).
    # KNOWN ISSUE: The docstring claims multiple matching passages per query, but the rows are exact duplicates, so
    # every query still has a single distinct positive.
    """Return a sample dataset with multiple matching passages for each query."""
    return Dataset.from_dict(
        {
            "query": queries + queries,
            "passage": passages[: len(queries)] + passages[: len(queries)],
        }
    )


def test_basic_functionality(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Mines with defaults and asserts query, passage and negative columns and at least as many rows as the input.
    """Test the basic functionality with default parameters."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(dataset=dataset, model=model, verbose=False)

    # Check that the output has expected columns
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Should have same number of rows as input (or more if multiple negatives)
    assert len(result) >= len(dataset)


def test_column_names(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    # [FUNCTION DOC]
    # Asserts custom anchor/positive column names are preserved alongside a negative column.
    """Test specifying custom column names."""
    model = static_retrieval_mrl_en_v1_model
    renamed_dataset = Dataset.from_dict({"question": queries, "answer": passages[: len(queries)]})

    result = mine_hard_negatives(
        dataset=renamed_dataset,
        model=model,
        anchor_column_name="question",
        positive_column_name="answer",
        verbose=False,
    )

    # Check that the output has expected columns
    assert "question" in result.column_names
    assert "answer" in result.column_names
    assert "negative" in result.column_names


def test_fully_custom_column_names(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
):
    # [FUNCTION DOC]
    # Asserts custom column names are preserved in triplet output and in n-tuple output with two negative columns.
    """Test dataset with completely different column names."""
    model = static_retrieval_mrl_en_v1_model
    # Create dataset with completely custom column names
    custom_dataset = Dataset.from_dict({"user_question": queries, "system_response": passages[: len(queries)]})

    # Test with custom column names
    result = mine_hard_negatives(
        dataset=custom_dataset,
        model=model,
        anchor_column_name="user_question",
        positive_column_name="system_response",
        verbose=False,
    )

    # Should preserve custom column names in output
    assert "user_question" in result.column_names
    assert "system_response" in result.column_names
    assert "negative" in result.column_names

    # Test output_format with custom column names
    result_ntuple = mine_hard_negatives(
        dataset=custom_dataset,
        model=model,
        anchor_column_name="user_question",
        positive_column_name="system_response",
        num_negatives=2,
        output_format="n-tuple",
        verbose=False,
    )

    # Should preserve custom column names with n-tuple format
    assert "user_question" in result_ntuple.column_names
    assert "system_response" in result_ntuple.column_names
    assert "negative_1" in result_ntuple.column_names
    assert "negative_2" in result_ntuple.column_names


def test_separate_corpus(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, passages: list[str]
) -> None:
    # [FUNCTION DOC]
    # Mines against a separate corpus (passages from index 5) and asserts at least one negative comes from it.
    """Test using a separate corpus for negative mining."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        corpus=passages[5:],  # Use different passages as corpus
        verbose=False,
    )

    # Check that the output has expected columns
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Check that negatives can come from the separate corpus
    negatives = result["negative"]
    assert any(neg in passages[5:] for neg in negatives), "No negatives found in the separate corpus."


def test_cross_encoder(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, reranker_bert_tiny_model: CrossEncoder
) -> None:
    # [FUNCTION DOC]
    # Mines with a cross-encoder rescorer and a max score filter and asserts the triplet columns.
    """Test using a cross-encoder for rescoring."""
    model = static_retrieval_mrl_en_v1_model
    cross_encoder = reranker_bert_tiny_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        cross_encoder=cross_encoder,
        max_score=0.8,  # Need a filtering criterion for cross-encoder to be used
        verbose=False,
    )

    # Should still have the basic expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_cross_encoder_detailed(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, reranker_bert_tiny_model: CrossEncoder
) -> None:
    # [FUNCTION DOC]
    # Mines with a cross-encoder, range_min 1, score bounds and two negatives in n-tuple format and asserts the
    # columns.
    """Test using a cross-encoder with different parameters."""
    model = static_retrieval_mrl_en_v1_model
    cross_encoder = reranker_bert_tiny_model
    # Test with cross-encoder and various filtering parameters
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        cross_encoder=cross_encoder,
        range_min=1,
        max_score=0.7,
        min_score=0.2,
        num_negatives=2,
        output_format="n-tuple",
        verbose=False,
    )

    # Should have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative_1" in result.column_names
    assert "negative_2" in result.column_names


def test_cross_encoder_with_multiple_positives_non_faiss(
    multiple_positive_dataset: Dataset,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    reranker_bert_tiny_model: CrossEncoder,
) -> None:
    # [FUNCTION DOC]
    # Mines the duplicate-row dataset with a cross-encoder and two negatives and asserts non-empty triplet output.
    model = static_retrieval_mrl_en_v1_model
    cross_encoder = reranker_bert_tiny_model

    result = mine_hard_negatives(
        dataset=multiple_positive_dataset,
        model=model,
        cross_encoder=cross_encoder,
        max_score=0.9,  # ensure CrossEncoder path is exercised
        num_negatives=2,
        verbose=False,
    )

    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names
    assert len(result) > 0


def test_range_parameters(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts mining with range_min 1 and range_max 4 still returns triplet columns.
    """Test range_min and range_max parameters."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        range_min=1,  # Skip the first (closest) match
        range_max=4,  # Only consider top 4 matches
        verbose=False,
    )

    # Output should still have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_score_filters(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts mining with min and max score filters still returns triplet columns.
    """Test max_score and min_score parameters."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        max_score=0.8,  # Only consider candidates with score <= 0.8
        min_score=0.1,  # Only consider candidates with score >= 0.1
        verbose=False,
    )

    # Output should still have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_margin_parameters(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts mining with an absolute margin and, separately, a relative margin both return a negative column.
    """Test absolute_margin and relative_margin parameters."""
    model = static_retrieval_mrl_en_v1_model
    # Test absolute_margin
    result_abs = mine_hard_negatives(
        dataset=dataset,
        model=model,
        absolute_margin=0.1,  # Negative must be at least 0.1 less similar than positive
        verbose=False,
    )

    # Test relative_margin
    result_rel = mine_hard_negatives(
        dataset=dataset,
        model=model,
        relative_margin=0.05,  # Negative must be at most 95% as similar as positive
        verbose=False,
    )

    # Both should have expected structure
    assert "negative" in result_abs.column_names
    assert "negative" in result_rel.column_names


def test_num_negatives(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts n-tuple output with two negatives has negative_1 and negative_2 but no negative_3.
    """Test num_negatives parameter."""
    model = static_retrieval_mrl_en_v1_model
    num_neg = 2
    result = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=num_neg, output_format="n-tuple", verbose=False
    )

    # Should have negative_1 and negative_2 columns
    for i in range(1, num_neg + 1):
        assert f"negative_{i}" in result.column_names

    # Should not have negative_3 column
    assert f"negative_{num_neg + 1}" not in result.column_names


def test_sampling_strategies(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts both "top" and "random" sampling return a negative column.
    """Test different sampling strategies."""
    model = static_retrieval_mrl_en_v1_model
    # Test 'top' strategy
    result_top = mine_hard_negatives(dataset=dataset, model=model, sampling_strategy="top", verbose=False)

    # Test 'random' strategy
    result_random = mine_hard_negatives(dataset=dataset, model=model, sampling_strategy="random", verbose=False)

    # Both should have expected structure
    assert "negative" in result_top.column_names
    assert "negative" in result_random.column_names


def test_prompts(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts raw query and corpus prompts still give triplet columns.
    """Test query_prompt and corpus_prompt parameters."""
    model = static_retrieval_mrl_en_v1_model
    query_prompt = "query: "
    corpus_prompt = "passage: "

    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        query_prompt=query_prompt,
        corpus_prompt=corpus_prompt,
        verbose=False,
    )

    # Should still have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_include_positives(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts include_positives forces n-tuple style output even when triplet format is requested.
    """Test include_positives parameter."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset, model=model, include_positives=True, output_format="triplet", verbose=False
    )

    # Should use n-tuple format (as enforced by include_positives=True)
    assert "negative_1" in result.column_names


def test_include_positives_with_labeled_formats(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    # [FUNCTION DOC]
    # Asserts that with include_positives, requested labeled-pair and labeled-list formats both produce n-tuple
    # style columns (negative_1 to negative_3) where negative_1 equals the positive passage.
    # KNOWN ISSUE: The docstring and comments speak of labeled formats and labels, but include_positives forces the
    # n-tuple layout, so neither labeled format nor any label is actually verified here.
    """Test include_positives with labeled pair and list formats."""
    model = static_retrieval_mrl_en_v1_model
    # Test with labeled-pair format
    result_pair = mine_hard_negatives(
        dataset=dataset,
        model=model,
        include_positives=True,
        output_format="labeled-pair",
        verbose=False,
    )

    # Should have expected structure
    assert "query" in result_pair.column_names
    assert "passage" in result_pair.column_names
    assert "negative_1" in result_pair.column_names
    assert "negative_3" in result_pair.column_names

    # First label in each row should be 1 (positive)
    for positive, negative_1 in zip(result_pair["passage"], result_pair["negative_1"]):
        assert positive == negative_1

    # Test with labeled-list format
    result_list = mine_hard_negatives(
        dataset=dataset,
        model=model,
        include_positives=True,
        output_format="labeled-list",
        verbose=False,
    )

    # Should have expected structure
    assert "query" in result_list.column_names
    assert "passage" in result_list.column_names
    assert "negative_1" in result_list.column_names
    assert "negative_3" in result_list.column_names

    # First label in each row should be 1 (positive)
    # Note that this isn't necessarily the case, sometimes there's predicted negatives with higher
    # similarity scores than the positive, but our documents are quite distinct here
    for positive, negative_1 in zip(result_list["passage"], result_list["negative_1"]):
        assert positive == negative_1


def test_output_formats(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Exhaustively checks the columns of every output format with and without scores: triplet (scores as a
    # two-element list), n-tuple (scores list of one plus negatives), the deprecated n-tuple-scores alias,
    # labeled-pair (label 0/1, or a score column replacing the label) and labeled-list (list passages with labels
    # 1 then 0s, or a scores list replacing labels).
    """Test all output_format options."""
    model = static_retrieval_mrl_en_v1_model

    # Test triplet format without scores (default)
    result_triplet = mine_hard_negatives(dataset=dataset, model=model, output_format="triplet", verbose=False)
    assert "query" in result_triplet.column_names
    assert "passage" in result_triplet.column_names
    assert "negative" in result_triplet.column_names
    # No score-related columns by default
    assert "positive_score" not in result_triplet.column_names
    assert "negative_score" not in result_triplet.column_names
    assert "scores" not in result_triplet.column_names
    assert len(result_triplet.column_names) == 3

    # Test triplet format with scores
    result_triplet_scores = mine_hard_negatives(
        dataset=dataset, model=model, output_format="triplet", output_scores=True, verbose=False
    )
    assert "query" in result_triplet_scores.column_names
    assert "passage" in result_triplet_scores.column_names
    assert "negative" in result_triplet_scores.column_names
    # Scores are exposed via a single `scores` column as [pos, neg]
    assert "positive_score" not in result_triplet_scores.column_names
    assert "negative_score" not in result_triplet_scores.column_names
    assert "scores" in result_triplet_scores.column_names
    assert len(result_triplet_scores.column_names) == 4

    # Verify scores are lists of numeric values [positive_score, negative_score]
    if len(result_triplet_scores) > 0:
        for score_list in result_triplet_scores["scores"]:
            assert isinstance(score_list, list)
            assert len(score_list) == 2
            assert all(isinstance(score, (int, float)) for score in score_list)

    # Test n-tuple format without scores (default)
    result_ntuple = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=2, output_format="n-tuple", verbose=False
    )
    assert "query" in result_ntuple.column_names
    assert "passage" in result_ntuple.column_names
    assert "negative_1" in result_ntuple.column_names
    assert "negative_2" in result_ntuple.column_names
    assert "score" not in result_ntuple.column_names
    assert "scores" not in result_ntuple.column_names

    # Test n-tuple format with scores
    result_ntuple_scores = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=2, output_format="n-tuple", output_scores=True, verbose=False
    )
    assert "query" in result_ntuple_scores.column_names
    assert "passage" in result_ntuple_scores.column_names
    assert "negative_1" in result_ntuple_scores.column_names
    assert "negative_2" in result_ntuple_scores.column_names
    # Unified `scores` column: [pos_score, neg1_score, neg2_score]
    assert "score" not in result_ntuple_scores.column_names
    assert "scores" in result_ntuple_scores.column_names

    # Verify scores are lists of expected length (1 positive + num_negatives)
    if len(result_ntuple_scores) > 0:
        for score_list in result_ntuple_scores["scores"]:
            assert isinstance(score_list, list)
            assert len(score_list) == 3
            assert all(isinstance(score, (int, float)) for score in score_list)

    # Test n-tuple-scores format
    result_scores = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=2, output_format="n-tuple-scores", verbose=False
    )
    assert "query" in result_scores.column_names
    assert "passage" in result_scores.column_names
    assert "negative_1" in result_scores.column_names
    assert "negative_2" in result_scores.column_names
    # Deprecated alias should behave like n-tuple + output_scores
    assert "score" not in result_scores.column_names
    assert "scores" in result_scores.column_names

    # Verify scores are lists of expected length (1 positive + num_negatives)
    assert all(len(score) == 3 for score in result_scores["scores"])

    # Test labeled-pair format without scores (default)
    result_pair = mine_hard_negatives(dataset=dataset, model=model, output_format="labeled-pair", verbose=False)
    assert "query" in result_pair.column_names
    assert "passage" in result_pair.column_names
    assert "label" in result_pair.column_names
    assert "score" not in result_pair.column_names
    assert len(result_pair.column_names) == 3

    # Test labeled-pair format with scores
    result_pair_scores = mine_hard_negatives(
        dataset=dataset, model=model, output_format="labeled-pair", output_scores=True, verbose=False
    )
    assert "query" in result_pair_scores.column_names
    assert "passage" in result_pair_scores.column_names
    # When output_scores=True we expose scores and drop labels
    assert "label" not in result_pair_scores.column_names
    assert "score" in result_pair_scores.column_names
    assert len(result_pair_scores.column_names) == 3

    # Verify labels are 0 or 1
    labels = set(result_pair["label"])
    assert labels == {0, 1}

    # Verify scores are numeric values
    if len(result_pair_scores) > 0:
        assert all(isinstance(score, (int, float)) for score in result_pair_scores["score"])

    # Test labeled-list format without scores (default)
    result_list = mine_hard_negatives(dataset=dataset, model=model, output_format="labeled-list", verbose=False)
    assert "query" in result_list.column_names
    assert "passage" in result_list.column_names
    assert "labels" in result_list.column_names
    assert "scores" not in result_list.column_names
    assert len(result_list.column_names) == 3

    # Test labeled-list format with scores
    result_list_scores = mine_hard_negatives(
        dataset=dataset, model=model, output_format="labeled-list", output_scores=True, verbose=False
    )
    assert "query" in result_list_scores.column_names
    assert "passage" in result_list_scores.column_names
    # With scores we replace labels by scores
    assert "labels" not in result_list_scores.column_names
    assert "scores" in result_list_scores.column_names
    assert len(result_list_scores.column_names) == 3

    # Verify each item in 'passage' is a list
    assert all(isinstance(p, list) for p in result_list["passage"])

    # Verify each item in 'labels' is a list with first element 1 (positive) and others 0 (negative)
    for label_list in result_list["labels"]:
        assert isinstance(label_list, list)
        assert label_list[0] == 1
        assert all(label == 0 for label in label_list[1:])

    # Verify each item in 'scores' is a list with numeric values
    if len(result_list_scores) > 0:
        for score_list in result_list_scores["scores"]:
            assert isinstance(score_list, list)
            assert all(isinstance(score, (int, float)) for score in score_list)


def _compute_similarity(model: SentenceTransformer, query: str, passage: str) -> float:
    # [FUNCTION DOC]
    # HELPER: Encodes the query with the query encoder and the passage with the document encoder (normalized
    # tensors, no progress bar) and returns the model's pairwise similarity as a float.
    """Compute similarity between a single query and passage using the model.

    This helper mirrors the internal scoring logic in `mine_hard_negatives` by
    encoding the query and passage with the dedicated query/document encoders
    and then using `similarity_pairwise`.
    """

    query_emb = model.encode_query([query], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    passage_emb = model.encode_document(
        [passage], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
    )
    return model.similarity_pairwise(query_emb, passage_emb).item()


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_triplet_output_scores_match_similarity(
    request: pytest.FixtureRequest, static_retrieval_mrl_en_v1_model: SentenceTransformer, test_dataset: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over the four dataset fixtures; mines two negatives in triplet format with scores and asserts
    # each row's positive and negative scores match recomputed similarities within 1e-5.
    """Verify that triplet scores match recomputed query-positive/negative similarity.

    We validate both the simple dataset and the multiple-positive dataset to
    ensure that score indexing remains correct when queries have more than
    one positive passage.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        output_format="triplet",
        output_scores=True,
        num_negatives=2,
        verbose=False,
    )
    assert result

    for row in result:
        pos_score_reported, neg_score_reported = row["scores"]

        pos_score_actual = _compute_similarity(model, row["query"], row["passage"])
        neg_score_actual = _compute_similarity(model, row["query"], row["negative"])

        assert pos_score_reported == pytest.approx(pos_score_actual, rel=1e-5, abs=1e-5)
        assert neg_score_reported == pytest.approx(neg_score_actual, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_n_tuple_output_scores_match_similarity(
    request: pytest.FixtureRequest, static_retrieval_mrl_en_v1_model: SentenceTransformer, test_dataset: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over the dataset fixtures; asserts n-tuple scores have one plus two entries matching recomputed
    # similarities of the positive and each negative.
    """Verify that n-tuple scores match recomputed query-positive/negatives similarity.

    Runs over both the single- and multiple-positive datasets.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    num_negatives = 2
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        num_negatives=num_negatives,
        output_format="n-tuple",
        output_scores=True,
        verbose=False,
    )
    assert result

    for row in result:
        scores = row["scores"]
        assert len(scores) == 1 + num_negatives

        # Positive
        pos_score_reported = scores[0]
        pos_score_actual = _compute_similarity(model, row["query"], row["passage"])
        assert pos_score_reported == pytest.approx(pos_score_actual, rel=1e-5, abs=1e-5)

        # Negatives
        for i in range(num_negatives):
            neg_col = f"negative_{i + 1}"
            neg_score_reported = scores[i + 1]
            neg_score_actual = _compute_similarity(model, row["query"], row[neg_col])
            assert neg_score_reported == pytest.approx(neg_score_actual, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_n_tuple_scores_alias_output_scores_match_similarity(
    request: pytest.FixtureRequest, static_retrieval_mrl_en_v1_model: SentenceTransformer, test_dataset: str
) -> None:
    # [FUNCTION DOC]
    # Same as the previous test using the deprecated n-tuple-scores format.
    """Verify that n-tuple-scores (deprecated alias) exposes correct similarity scores.

    Runs over both the single- and multiple-positive datasets.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    num_negatives = 2
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        num_negatives=num_negatives,
        output_format="n-tuple-scores",
        verbose=False,
    )
    assert result

    for row in result:
        scores = row["scores"]
        assert len(scores) == 1 + num_negatives

        pos_score_reported = scores[0]
        pos_score_actual = _compute_similarity(model, row["query"], row["passage"])
        assert pos_score_reported == pytest.approx(pos_score_actual, rel=1e-5, abs=1e-5)

        for i in range(num_negatives):
            neg_col = f"negative_{i + 1}"
            neg_score_reported = scores[i + 1]
            neg_score_actual = _compute_similarity(model, row["query"], row[neg_col])
            assert neg_score_reported == pytest.approx(neg_score_actual, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_labeled_pair_output_scores_match_similarity(
    request: pytest.FixtureRequest, static_retrieval_mrl_en_v1_model: SentenceTransformer, test_dataset: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over the dataset fixtures; asserts each labeled-pair score matches the recomputed similarity of
    # its query and passage.
    """Verify that labeled-pair scores match recomputed query-passage similarity.

    Runs over both the single- and multiple-positive datasets.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        output_format="labeled-pair",
        output_scores=True,
        verbose=False,
    )
    assert result

    for row in result:
        score_reported = row["score"]
        score_actual = _compute_similarity(model, row["query"], row["passage"])
        assert score_reported == pytest.approx(score_actual, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_labeled_list_output_scores_match_similarity(
    request: pytest.FixtureRequest, static_retrieval_mrl_en_v1_model: SentenceTransformer, test_dataset: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over the dataset fixtures; asserts each labeled-list row has as many scores as passages and each
    # matches the recomputed similarity.
    """Verify that labeled-list scores match recomputed query-passage similarity for all items.

    Runs over both the single- and multiple-positive datasets.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        output_format="labeled-list",
        output_scores=True,
        verbose=False,
    )
    assert result

    for row in result:
        query = row["query"]
        passages = row["passage"]
        scores = row["scores"]

        assert len(passages) == len(scores)

        for passage, score_reported in zip(passages, scores):
            score_actual = _compute_similarity(model, query, passage)
            assert score_reported == pytest.approx(score_actual, rel=1e-5, abs=1e-5)


def test_batch_size(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts a small encoding batch size still returns triplet columns.
    """Test batch_size parameter."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        batch_size=2,  # Small batch size for testing
        verbose=False,
    )

    # Should still produce expected output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
def test_faiss(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Skipped without faiss. Mines twice with FAISS and a small FAISS batch size and asserts triplet columns.
    # KNOWN ISSUE: The two mining calls are identical, so the second run adds no coverage (likely meant to vary a
    # parameter).
    """Test use_faiss parameter."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        use_faiss=True,
        faiss_batch_size=2,  # Small batch size for testing
        verbose=False,
    )

    # Should still produce expected output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        use_faiss=True,
        faiss_batch_size=2,  # Small batch size for testing
        verbose=False,
    )

    # Should still produce expected output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_non_faiss_batched_search_matches_single_batch(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC]
    # Monkeypatches the model's similarity function to record query chunk sizes, mines with FAISS batch size 3 and
    # then 100 (without FAISS), asserts chunk sizes 3, 3, 2 and then 8, and that both results have identical columns,
    # texts and approximately equal scores.
    """Test that batching the non-FAISS similarity search splits the queries and does not change the mined negatives."""
    model = static_retrieval_mrl_en_v1_model
    model.similarity_fn_name  # noqa: B018 (trigger lazy init of _similarity)
    original_similarity = model._similarity
    chunk_sizes = []

    def similarity_spy(embeddings1, embeddings2):
        # [FUNCTION DOC]
        # HELPER: Records the number of query embeddings and delegates to the original similarity function.
        chunk_sizes.append(len(embeddings1))
        return original_similarity(embeddings1, embeddings2)

    monkeypatch.setattr(model, "_similarity", similarity_spy)

    # faiss_batch_size=3 splits the 8 queries into uneven batches (3, 3, 2), 100 is a single batch
    results = []
    for faiss_batch_size, expected_chunk_sizes in ((3, [3, 3, 2]), (100, [8])):
        chunk_sizes.clear()
        results.append(
            mine_hard_negatives(
                dataset=dataset,
                model=model,
                num_negatives=2,
                range_max=3,
                output_format="n-tuple",
                output_scores=True,
                faiss_batch_size=faiss_batch_size,
                verbose=False,
            )
        )
        assert chunk_sizes == expected_chunk_sizes
    batched, single = results

    assert batched.column_names == single.column_names
    for column in batched.column_names:
        if column == "scores":
            for batched_scores, single_scores in zip(batched[column], single[column]):
                assert batched_scores == pytest.approx(single_scores, rel=1e-6, abs=1e-7)
        else:
            assert batched[column] == single[column]


def test_cache(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Mines twice with a cache folder in a temp directory; asserts query and corpus embedding cache files are created
    # and both runs return the same number of rows.
    """Test cache_folder parameter."""
    model = static_retrieval_mrl_en_v1_model
    cache_dir = os.path.join(tmp_path, "embeddings_cache")

    # First run should create cache
    result1 = mine_hard_negatives(dataset=dataset, model=model, cache_folder=cache_dir, verbose=False)

    # Check that cache files were created
    cache_files = os.listdir(cache_dir)
    assert len(cache_files) > 0
    assert any("query_embeddings" in f for f in cache_files)
    assert any("corpus_embeddings" in f for f in cache_files)

    # Second run should use cache
    result2 = mine_hard_negatives(dataset=dataset, model=model, cache_folder=cache_dir, verbose=False)

    # Results should be the same
    assert len(result1) == len(result2)


def test_cache_respects_prompt(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Mines twice with different query prompts into the same cache folder and asserts a second, distinct query
    # embedding cache file is added.
    """The cache key must incorporate the prompt arguments, so a different `query_prompt`
    produces its own cache file rather than silently reusing embeddings computed with a different prompt."""
    model = static_retrieval_mrl_en_v1_model
    cache_dir = os.path.join(tmp_path, "embeddings_cache")

    mine_hard_negatives(
        dataset=dataset,
        model=model,
        cache_folder=cache_dir,
        query_prompt="Represent this question for retrieval: ",
        verbose=False,
    )
    query_files_a = {f for f in os.listdir(cache_dir) if f.startswith("query_embeddings")}

    mine_hard_negatives(
        dataset=dataset,
        model=model,
        cache_folder=cache_dir,
        query_prompt="Represent this statement for retrieval: ",
        verbose=False,
    )
    query_files_b = {f for f in os.listdir(cache_dir) if f.startswith("query_embeddings")}

    # A second, distinct query_prompt must add a new cache file, not reuse the first prompt's.
    assert query_files_a != query_files_b
    assert query_files_a.issubset(query_files_b)


def test_multiple_positives_per_query(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
):
    # [FUNCTION DOC]
    # Mines a dataset where some queries repeat with different positives and asserts triplet columns and at least as
    # many rows as input.
    """Test dataset with multiple positives per query."""
    model = static_retrieval_mrl_en_v1_model
    # Create dataset with duplicate queries
    queries_dup = queries[:3] + queries[:2]
    passages_dup = passages[:3] + passages[5:7]  # Different positives for repeated queries

    dataset_dup = Dataset.from_dict({"query": queries_dup, "passage": passages_dup})

    result = mine_hard_negatives(dataset=dataset_dup, model=model, range_max=3, verbose=False)

    # Should still have expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Should handle the duplicates correctly
    assert len(result) >= len(dataset_dup)


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
def test_faiss_multiple_positives_matches_non_faiss(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    # [FUNCTION DOC]
    # Skipped without faiss. With repeated queries (two positives) and three negatives with range_max 3, asserts the
    # FAISS and non-FAISS paths mine the same number of rows.
    """FAISS and non-FAISS paths must mine the same number of negatives when queries have multiple positives.

    Regression test for the FAISS branch requesting only ``range_max + 1`` candidates from
    ``index.search`` instead of ``range_max + max_positives`` (as the non-FAISS ``torch.topk`` sibling
    does). With 2+ positives per query the FAISS path retrieved too few candidates and could mine fewer
    negatives than the non-FAISS path for the same input. This crosses ``use_faiss=True`` with the
    multi-positive case, which neither ``test_faiss`` nor ``test_multiple_positives_per_query`` covered.
    """
    model = static_retrieval_mrl_en_v1_model
    # Duplicate queries with different positives -> repeated queries have multiple positives (max_positives == 2).
    queries_dup = queries[:3] + queries[:2]
    passages_dup = passages[:3] + passages[5:7]
    dataset_dup = Dataset.from_dict({"query": queries_dup, "passage": passages_dup})

    # num_negatives must reach into the tail of the candidate pool (num_negatives == range_max), otherwise the
    # missing candidates are never selected and the assertion below passes even without the fix.
    kwargs = dict(dataset=dataset_dup, model=model, num_negatives=3, range_max=3, faiss_batch_size=2, verbose=False)
    faiss_result = mine_hard_negatives(use_faiss=True, **kwargs)
    non_faiss_result = mine_hard_negatives(use_faiss=False, **kwargs)

    # The invariant the fix restores: both paths mine the same number of negatives for the same input
    # (default "triplet" output_format yields one negative per row, so len == number of mined negatives).
    assert len(faiss_result) == len(non_faiss_result)


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
@pytest.mark.parametrize("range_max", [3, 10])
def test_faiss_corpus_smaller_than_k(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer, range_max: int
) -> None:
    # [FUNCTION DOC]
    # Skipped without faiss; parametrized over range_max. Uses a four-document corpus smaller than the number of
    # requested candidates and asserts no mined negative is the row's passage or any positive of its query.
    """Candidates must never include a query's own positives when ``k`` exceeds the corpus size.

    FAISS pads short result sets with index -1 and score -3.4e38. The score survives the -inf filters and
    the index resolves to the last corpus entry, so the final document could be mined as a negative for
    every query, including for the query it is a positive of.
    """
    model = static_retrieval_mrl_en_v1_model
    corpus = passages[:4]
    # The first query has two positives (max_positives == 2), so k = range_max + 2 overruns the 4 document corpus.
    queries_dup = queries[:3] + queries[:1]
    passages_dup = [passages[0], passages[1], passages[2], passages[3]]
    dataset_dup = Dataset.from_dict({"query": queries_dup, "passage": passages_dup})

    result = mine_hard_negatives(
        dataset=dataset_dup,
        model=model,
        corpus=corpus,
        num_negatives=3,
        range_max=range_max,
        use_faiss=True,
        faiss_batch_size=2,
        verbose=False,
    )

    positives_per_query = defaultdict(set)
    for query, positive in zip(dataset_dup["query"], dataset_dup["passage"]):
        positives_per_query[query].add(positive)

    for row in result:
        assert row["negative"] != row["passage"]
        assert row["negative"] not in positives_per_query[row["query"]]


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
def test_faiss_corpus_smaller_than_k_with_cross_encoder(
    queries: list[str],
    passages: list[str],
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    reranker_bert_tiny_model: CrossEncoder,
) -> None:
    # [FUNCTION DOC]
    # Same as the previous test with a cross-encoder rescorer and a negative absolute margin, asserting FAISS padding
    # slots are never mined as negatives.
    """The FAISS -1 padding must stay disqualified after the CrossEncoder rescoring.

    The rescoring assigns a fresh score to every candidate, so disqualifying the padding at search time
    is not enough: the padded slots get a genuine score back and can be mined as negatives. A negative
    ``absolute_margin`` keeps the margin filter from incidentally removing them.
    """
    model = static_retrieval_mrl_en_v1_model
    corpus = passages[:4]
    # The first query has two positives (max_positives == 2), so k = 3 + 2 overruns the 4 document corpus.
    queries_dup = queries[:3] + queries[:1]
    passages_dup = [passages[0], passages[1], passages[2], passages[3]]
    dataset_dup = Dataset.from_dict({"query": queries_dup, "passage": passages_dup})

    result = mine_hard_negatives(
        dataset=dataset_dup,
        model=model,
        corpus=corpus,
        cross_encoder=reranker_bert_tiny_model,
        num_negatives=3,
        range_max=3,
        absolute_margin=-10.0,
        use_faiss=True,
        faiss_batch_size=2,
        verbose=False,
    )

    positives_per_query = defaultdict(set)
    for query, positive in zip(dataset_dup["query"], dataset_dup["passage"]):
        positives_per_query[query].add(positive)

    for row in result:
        assert row["negative"] != row["passage"]
        assert row["negative"] not in positives_per_query[row["query"]]


def test_deprecated_parameters(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts deprecated parameters still work: as_triplets True gives three columns, as_triplets False gives
    # n-tuple columns, and margin gives triplet columns.
    """Test deprecated parameters: as_triplets and margin."""
    model = static_retrieval_mrl_en_v1_model
    # Test as_triplets=True
    result_triplet = mine_hard_negatives(dataset=dataset, model=model, range_max=3, as_triplets=True, verbose=False)
    assert "negative" in result_triplet.column_names
    assert len(result_triplet.column_names) == 3

    # Test as_triplets=False
    result_ntuple = mine_hard_negatives(
        dataset=dataset, model=model, as_triplets=False, range_max=3, num_negatives=2, verbose=False
    )
    assert "negative_1" in result_ntuple.column_names
    assert "negative_2" in result_ntuple.column_names

    # Test margin
    result_margin = mine_hard_negatives(dataset=dataset, model=model, range_max=3, margin=0.1, verbose=False)
    assert "query" in result_margin.column_names
    assert "passage" in result_margin.column_names
    assert "negative" in result_margin.column_names


def test_margin_with_default_range_max(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    # [FUNCTION DOC]
    # Asserts the deprecated margin with the auto-derived range_max (larger than the corpus) gives non-empty
    # triplet output.
    """Test the deprecated margin parameter with the auto-derived range_max, which exceeds the
    corpus size and relies on the candidate set being capped at the corpus size."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        margin=0.2,  # Use deprecated margin parameter
        verbose=False,
    )

    # Should still produce valid output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Confirm we have results
    assert len(result) > 0


def test_multi_process(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, reranker_bert_tiny_model: CrossEncoder
) -> None:
    # [FUNCTION DOC]
    # Skipped when CI is set. Mines with a cross-encoder, relative margin and multi-processing and asserts triplet
    # columns.
    """Test use_multi_process parameter if multiple CPUs available."""
    model = static_retrieval_mrl_en_v1_model
    cross_encoder = reranker_bert_tiny_model
    # Skip on CI environments where multi-processing might be restricted
    if os.environ.get("CI"):
        pytest.skip("Skipping multi-process test in CI environment")

    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        cross_encoder=cross_encoder,
        num_negatives=2,
        relative_margin=0.1,
        use_multi_process=True,
        verbose=False,
    )

    # Should still produce expected output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_empty_dataset(static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts an empty dataset raises ValueError.
    """Test behavior with an empty dataset."""
    model = static_retrieval_mrl_en_v1_model
    empty_dataset = Dataset.from_dict({"query": [], "passage": []})

    with pytest.raises(ValueError):
        mine_hard_negatives(dataset=empty_dataset, model=model, verbose=False)


@pytest.mark.parametrize("sampling_strategy", ["top", "random"])
def test_num_negatives_exceeds_range_window(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, sampling_strategy: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over sampling strategies; asserts ten negatives with a window of three raises a ValueError
    # explaining only three candidates remain.
    """Requesting more negatives than the range window holds must explain itself, not fail on a tensor shape.

    Both strategies reach the same invalid window by a different route: "top" slices the candidate matrix to
    range_max - range_min columns while the anchor index matrix keeps num_negatives columns, and "random" clamps
    its number of options to num_negatives and then samples past the width of that same matrix.
    """
    with pytest.raises(ValueError, match="Cannot mine 10 negatives per query: only 3 candidates remain"):
        mine_hard_negatives(
            dataset=dataset,
            model=static_retrieval_mrl_en_v1_model,
            num_negatives=10,
            range_min=0,
            range_max=3,
            sampling_strategy=sampling_strategy,
            verbose=False,
        )


@pytest.mark.parametrize("num_negatives", [0, -1, -5])
def test_num_negatives_must_be_positive(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, num_negatives: int
) -> None:
    # [FUNCTION DOC]
    # Parametrized over 0, -1 and -5; asserts a ValueError saying num_negatives must be at least 1.
    """range_max is derived from num_negatives, so a non-positive value must be named rather than mined around.

    Without the check, 0 quietly returns the dataset with no negative columns at all, and a negative value
    derives a range_max at or below range_min, reporting a range problem the caller never configured.
    """
    with pytest.raises(ValueError, match=f"num_negatives must be at least 1, got num_negatives={num_negatives}"):
        mine_hard_negatives(
            dataset=dataset,
            model=static_retrieval_mrl_en_v1_model,
            num_negatives=num_negatives,
            verbose=False,
        )


def test_num_negatives_exceeds_range_window_capped_by_faiss(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC]
    # Skips without faiss; monkeypatches faiss to report one GPU and asserts the error mentions range_max being capped
    # to 2047 for FAISS on GPU.
    """The FAISS GPU cap can shrink an auto-derived range_max below num_negatives, so the error mentions the cap."""
    faiss = pytest.importorskip("faiss")
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 1)

    with pytest.raises(ValueError, match="range_max was capped to 2047 because FAISS on GPU"):
        mine_hard_negatives(
            dataset=dataset,
            model=static_retrieval_mrl_en_v1_model,
            num_negatives=10,
            range_min=2040,
            use_faiss=True,
            verbose=False,
        )


def test_use_faiss_on_cpu_is_not_capped(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC]
    # Skips without faiss; monkeypatches faiss to report no GPUs and removes the GPU cloner options, and asserts the
    # same arguments mine without error, returning zero rows.
    """FAISS on CPU has no 2048-document limit, so the same arguments must mine instead of raising."""
    faiss = pytest.importorskip("faiss")
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 0)
    monkeypatch.delattr(faiss, "GpuMultipleClonerOptions", raising=False)

    result = mine_hard_negatives(
        dataset=dataset,
        model=static_retrieval_mrl_en_v1_model,
        num_negatives=10,
        range_min=2040,
        use_faiss=True,
        verbose=False,
    )

    assert len(result) == 0


@pytest.mark.parametrize("sampling_strategy", ["top", "random"])
def test_num_negatives_matching_range_window_is_allowed(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, sampling_strategy: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over sampling strategies; asserts a window equal to num_negatives works, giving five n-tuple
    # columns and one row per input.
    """A window exactly the size of num_negatives is still satisfiable, under either strategy, and must not raise."""
    result = mine_hard_negatives(
        dataset=dataset,
        model=static_retrieval_mrl_en_v1_model,
        num_negatives=3,
        range_min=0,
        range_max=3,
        sampling_strategy=sampling_strategy,
        output_format="n-tuple",
        verbose=False,
    )

    assert len(result.column_names) == 2 + 3
    assert len(result) == len(dataset)


def test_range_min_must_be_non_negative(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    # [FUNCTION DOC]
    # Asserts a negative range_min raises ValueError.
    """A negative range_min would silently slice the candidate window from the wrong end."""
    with pytest.raises(ValueError, match="range_min must be non-negative"):
        mine_hard_negatives(
            dataset=dataset,
            model=static_retrieval_mrl_en_v1_model,
            range_min=-5,
            range_max=8,
            verbose=False,
        )


@pytest.mark.parametrize(("range_min", "range_max"), [(3, 3), (5, 3)])
def test_range_min_must_be_below_range_max(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, range_min: int, range_max: int
) -> None:
    # [FUNCTION DOC]
    # Parametrized; asserts range_min at or above range_max raises ValueError.
    """A range_min at or above range_max leaves no candidate window at all."""
    with pytest.raises(ValueError, match="range_min must be smaller than range_max"):
        mine_hard_negatives(
            dataset=dataset,
            model=static_retrieval_mrl_en_v1_model,
            range_min=range_min,
            range_max=range_max,
            verbose=False,
        )


def test_range_min_above_faiss_capped_range_max(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC]
    # Skips without faiss; with one GPU reported, asserts range_min above the capped range_max raises a ValueError
    # mentioning the cap.
    """The FAISS GPU cap can push an auto-derived range_max below range_min, so the error mentions the cap."""
    faiss = pytest.importorskip("faiss")
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 1)

    with pytest.raises(ValueError, match="range_min must be smaller than range_max.*range_max was capped to 2047"):
        mine_hard_negatives(
            dataset=dataset,
            model=static_retrieval_mrl_en_v1_model,
            range_min=2050,
            use_faiss=True,
            verbose=False,
        )


def test_larger_dataset_with_combinations(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    # [FUNCTION DOC]
    # Mines a 32-row dataset (queries suffixed with an index) with combined range, score, margin and batch options in
    # n-tuple format and asserts exactly two negative columns.
    """Test with a larger dataset and challenging parameter combinations."""
    model = static_retrieval_mrl_en_v1_model
    # Create a larger dataset with 32 entries
    queries_large = [
        f"{query} {i}"  # Create unique queries by appending an index
        for i in range(4)
        for query in queries
    ]
    passages_large = passages[: len(queries)] * 4  # Match the first passages to each query

    larger_dataset = Dataset.from_dict({"query": queries_large, "passage": passages_large})

    # Test a combination of parameters
    result = mine_hard_negatives(
        dataset=larger_dataset,
        model=model,
        range_min=1,
        range_max=15,
        max_score=0.9,
        min_score=0.1,
        absolute_margin=0.05,
        num_negatives=2,
        output_format="n-tuple",
        batch_size=4,
        verbose=False,
    )

    # Should have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative_1" in result.column_names
    assert "negative_2" in result.column_names

    # Should not have more negatives than specified
    assert "negative_3" not in result.column_names


def test_prompt_combinations(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Deep-copies the model with named prompts and asserts mining with prompt names, and with a raw query prompt
    # overriding the name, both give triplet columns only.
    """Test various prompt configurations."""
    model = static_retrieval_mrl_en_v1_model
    # Test with prompt_names instead of raw prompts
    model_with_prompts = deepcopy(model)
    model_with_prompts.prompts = {"query_prompt": "Query: ", "passage_prompt": "Passage: "}

    result_prompt_names = mine_hard_negatives(
        dataset=dataset,
        model=model_with_prompts,
        query_prompt_name="query_prompt",
        corpus_prompt_name="passage_prompt",
        verbose=False,
    )

    # Should have expected structure
    assert "query" in result_prompt_names.column_names
    assert "passage" in result_prompt_names.column_names
    assert "negative" in result_prompt_names.column_names

    # Test with mixed prompts - raw prompt overriding prompt_name
    result_mixed = mine_hard_negatives(
        dataset=dataset,
        model=model_with_prompts,
        query_prompt_name="query_prompt",  # This should be ignored since query_prompt is provided
        query_prompt="Direct query: ",  # This should be used
        corpus_prompt_name="passage_prompt",
        verbose=False,
    )

    # Should still have expected structure
    assert "query" in result_mixed.column_names
    assert "passage" in result_mixed.column_names
    assert "negative" in result_mixed.column_names
    assert "negative_1" not in result_mixed.column_names  # Ensure negative_1 is not present
    assert "negative_2" not in result_mixed.column_names  # Ensure negative_2 is not present


def test_n_tuple_scores_format_details(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    # [FUNCTION DOC]
    # Asserts the n-tuple-scores format gives scores lists of three entries where the positive score is at least
    # every negative score.
    """Test details of n-tuple-scores output format."""
    model = static_retrieval_mrl_en_v1_model
    num_neg = 2
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        num_negatives=num_neg,
        output_format="n-tuple-scores",
        verbose=False,
    )

    # Check column structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative_1" in result.column_names
    assert "negative_2" in result.column_names
    # Deprecated alias should behave like n-tuple + output_scores
    assert "score" not in result.column_names
    assert "scores" in result.column_names

    # Check score structure - should be a list with 1 + num_negatives elements (positive + negatives)
    for score_list in result["scores"]:
        # Should be a list
        assert isinstance(score_list, list)

        # Should have exactly 1 + num_negatives elements
        assert len(score_list) == 1 + num_neg

        # First score (for positive) should be higher than all negative scores
        pos_score = score_list[0]
        for neg_score in score_list[1:]:
            assert pos_score >= neg_score


def test_tiny_corpus(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    # [FUNCTION DOC]
    # Mines two pairs with one negative and range_max 1 and asserts a dataset with triplet columns and at least one
    # row.
    """Test with a very small corpus to test edge case handling."""
    model = static_retrieval_mrl_en_v1_model
    # Create a dataset with just 2 pairs
    tiny_dataset = Dataset.from_dict({"query": queries[:2], "passage": passages[:2]})

    # Test with minimal corpus
    result = mine_hard_negatives(
        dataset=tiny_dataset,
        model=model,
        num_negatives=1,  # Request only 1 negative
        range_max=1,  # Very restrictive range_max
        verbose=False,
    )

    # Should still produce a valid dataset with expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Some rows might be missing if constraints couldn't be satisfied
    assert isinstance(result, Dataset)
    assert len(result) >= 1


def test_verbose_mode(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts verbose mode runs and returns non-empty triplet output.
    """Test that verbose=True doesn't cause any crashes."""
    model = static_retrieval_mrl_en_v1_model

    # Simply check that running with verbose=True doesn't crash
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        range_max=3,  # Small range to keep test fast
        verbose=True,  # Enable verbose output
    )

    # Verify we still get valid results
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names
    assert len(result) > 0


@pytest.mark.skipif(not is_datasets_available(), reason="datasets library is not available")
def test_mine_hard_negatives_with_prompt(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Replaces the model's preprocess with a recorder, mines without a prompt and asserts every call had task query
    # or document with an empty prompt, then mines with a query prompt and asserts query calls carry it while
    # document calls do not; both results are non-empty. Restores the original preprocess at the end.
    # KNOWN ISSUE: The original preprocess is restored only on success (not in a finally block), so a failing assertion
    # leaves the patched method on the fixture model.
    """
    Tests that mine_hard_negatives runs with and without a prompt.
    """
    model = paraphrase_distilroberta_base_v1_model
    # 1. Create a test dataset
    data = {
        "anchor": [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Who wrote 'Hamlet'?",
            "What is the boiling point of water?",
            "Describe the theory of relativity.",
        ],
        "positive": [
            "Paris is the capital of France.",
            "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
            "William Shakespeare wrote 'Hamlet'.",
            "Water boils at 100 degrees Celsius.",
            "Relativity theory describes gravity as a property of spacetime.",
        ],
        "negative_pool": [
            "Berlin is the capital of Germany.",
            "Cellular respiration releases energy.",
            "Christopher Marlowe wrote 'Doctor Faustus'.",
            "Ethanol boils at 78 degrees Celsius.",
            "Quantum mechanics describes the smallest scales of energy.",
        ],
    }

    dataset = Dataset.from_dict(data)
    corpus = data["positive"] + data["negative_pool"]  # Corpus includes positives and other docs

    query_prompt = "query: "
    num_negatives = 1

    # Set up monkeypatching to verify prompt usage
    original_preprocess = model.preprocess
    preprocess_calls = []

    def mock_preprocess(texts, **kwargs) -> dict[str, Tensor]:
        # [FUNCTION DOC]
        # HELPER: Records the keyword arguments and delegates to the original preprocess.
        preprocess_calls.append(kwargs)
        return original_preprocess(texts, **kwargs)

    # 2. Run without prompt - check that no prompt is added
    model.preprocess = mock_preprocess
    preprocess_calls.clear()

    result_no_prompt = mine_hard_negatives(
        dataset=dataset,
        model=model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        corpus=corpus,
        num_negatives=num_negatives,
        batch_size=4,
        verbose=False,
        output_format="triplet",
    )

    # Verify that preprocess was called without prompts for queries and corpus
    for kwargs in preprocess_calls:
        assert kwargs == {"task": "query", "prompt": ""} or kwargs == {"task": "document", "prompt": ""}

    # Assert basic success criteria
    assert isinstance(result_no_prompt, Dataset)
    assert "negative" in result_no_prompt.column_names
    assert len(result_no_prompt) > 0  # Check that some negatives were found

    # 3. Run with prompt - check that prompt is actually added
    preprocess_calls.clear()

    result_with_prompt = mine_hard_negatives(
        dataset=dataset,
        model=model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        corpus=corpus,
        num_negatives=num_negatives,
        batch_size=4,
        verbose=False,
        query_prompt=query_prompt,
        output_format="triplet",
    )

    # Verify that tokenize was called with prompts, and the corpus without prompts
    for kwargs in preprocess_calls:
        assert kwargs == {"task": "query", "prompt": query_prompt} or kwargs == {"task": "document", "prompt": ""}

    # Assert basic success criteria
    assert isinstance(result_with_prompt, Dataset)
    assert "negative" in result_with_prompt.column_names
    assert len(result_with_prompt) > 0

    # Restore original preprocess function
    model.preprocess = original_preprocess


@pytest.mark.parametrize("output_format", ["triplet", "labeled-pair", "n-tuple", "n-tuple-scores", "labeled-list"])
@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_multiple_positives(
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    output_format: str,
    test_dataset: str,
):
    # [FUNCTION DOC]
    # Parametrized over output formats and dataset fixtures; mines three negatives and asserts all rows are unique and
    # that the row count is: unique pairs times 3 for triplet, unique pairs plus unique queries times 3 for
    # labeled-pair, and unique pairs for the other formats.
    model = static_retrieval_mrl_en_v1_model
    # Get the actual dataset from the fixture using the parameter name
    dataset = request.getfixturevalue(test_dataset)
    unique_dataset_length = len(set([tuple(sample.values()) for sample in dataset]))

    num_negatives = 3
    result = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=num_negatives, output_format=output_format, verbose=False
    )

    # Ensure all rows in result are unique
    unique_result_length = len(set([str(sample) for sample in result]))
    assert unique_result_length == len(result)

    # Should have the expected number of rows, accounting for the dataset type
    if output_format == "triplet":
        assert len(result) == unique_dataset_length * num_negatives
    elif output_format == "labeled-pair":
        # For the "labeled-pair" output format:
        # - For each unique (query, positive) pair in the dataset, we create one labeled pair with label 1 (positive).
        # - For each unique query, we mine `num_negatives` negatives, each paired with the query and labeled 0 (negative).
        # Therefore, the total number of result rows is:
        #   (number of unique pairs) + (number of unique queries) * num_negatives
        #   == unique_dataset_length + len(set(dataset["query"])) * num_negatives
        assert len(result) == unique_dataset_length + len(set(dataset["query"])) * num_negatives
    elif output_format in ("n-tuple", "n-tuple-scores", "labeled-list"):
        assert len(result) == unique_dataset_length


@pytest.mark.parametrize("output_format", ["triplet", "labeled-pair", "n-tuple", "n-tuple-scores", "labeled-list"])
@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_missing_negatives(
    capsys: pytest.CaptureFixture,
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    output_format: str,
    test_dataset: str,
):
    # [FUNCTION DOC]
    # Parametrized like the previous test; asserts the default mining reaches the expected maximum row count, then
    # mines with max score 0 and range_max 4 and asserts fewer rows and a printed "Could not find enough negatives"
    # message.
    model = static_retrieval_mrl_en_v1_model
    # Get the actual dataset from the fixture using the parameter name
    dataset = request.getfixturevalue(test_dataset)
    unique_dataset_length = len(set([tuple(sample.values()) for sample in dataset]))

    num_negatives = 3
    expected_max_count = len(
        mine_hard_negatives(
            dataset,
            model=model,
            output_format=output_format,
            num_negatives=num_negatives,
            verbose=True,
        )
    )
    if output_format == "triplet":
        assert expected_max_count == unique_dataset_length * num_negatives
    elif output_format == "labeled-pair":
        assert expected_max_count == unique_dataset_length + len(set(dataset["query"])) * num_negatives
    elif output_format in ("n-tuple", "n-tuple-scores", "labeled-list"):
        assert expected_max_count == unique_dataset_length

    capsys.readouterr()  # Clear any previous output
    result = mine_hard_negatives(
        dataset,
        model=model,
        max_score=0,
        range_max=4,
        output_format=output_format,
        num_negatives=num_negatives,
        verbose=True,
    )
    assert len(result) < expected_max_count
    captured = capsys.readouterr()
    assert "Could not find enough negatives" in captured.out


class ControlledNegativeScoreModel:
    # [CLASS DOC]
    # HELPER: Deterministic stand-in model with a CPU device and minimal card data. Holds fixed 2-D unit embeddings
    # for a query "q", a positive "p" with negative similarity, a negative more similar than the positive, and a far
    # negative.
    """Deterministic model whose anchor-positive score is negative, with a candidate negative
    that is *more* similar to the anchor than the positive (see #3819)."""

    device = torch.device("cpu")
    model_card_data = SimpleNamespace(base_model="controlled-negative-score-model")

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Stores the fixed embedding table.
        self.embeddings = {
            "q": torch.tensor([1.0, 0.0]),
            "p": torch.tensor([-0.50, math.sqrt(1 - 0.50**2)]),
            "n_more_similar": torch.tensor([-0.49, math.sqrt(1 - 0.49**2)]),
            "n_far": torch.tensor([-0.80, math.sqrt(1 - 0.80**2)]),
        }

    def encode_query(self, texts, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns a numpy array stacking the embeddings of the given texts (unknown texts raise KeyError).
        return torch.stack([self.embeddings[text] for text in texts]).numpy()

    def encode_document(self, texts, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Same as the query encoder, for documents.
        return torch.stack([self.embeddings[text] for text in texts]).numpy()

    def similarity(self, queries, corpus):
        # [FUNCTION DOC]
        # HELPER: Returns the dot-product matrix between float32 query and corpus embeddings.
        return torch.as_tensor(queries, dtype=torch.float32) @ torch.as_tensor(corpus, dtype=torch.float32).T

    def similarity_pairwise(self, queries, positives):
        # [FUNCTION DOC]
        # HELPER: Returns the row-wise dot products of float32 query and positive embeddings.
        return (torch.as_tensor(queries, dtype=torch.float32) * torch.as_tensor(positives, dtype=torch.float32)).sum(
            dim=1
        )


def test_relative_margin_with_negative_positive_score() -> None:
    # [FUNCTION DOC]
    # Mines with a relative margin of 0.05 where the positive score is negative and asserts the more-similar negative
    # is rejected, returning the far negative with a lower score than the positive.
    """`relative_margin` must not keep a negative that is more similar to the anchor than the
    positive pair when the positive-pair score is negative (#3819)."""
    dataset = Dataset.from_dict({"query": ["q"], "positive": ["p"]})
    result = mine_hard_negatives(
        dataset=dataset,
        model=ControlledNegativeScoreModel(),
        anchor_column_name="query",
        positive_column_name="positive",
        corpus=["p", "n_more_similar", "n_far"],
        range_max=2,
        relative_margin=0.05,
        num_negatives=1,
        output_scores=True,
        verbose=False,
    )
    row = result[0]
    positive_score, negative_score = row["scores"]
    assert row["negative"] == "n_far"
    assert negative_score < positive_score


def test_range_max_larger_than_corpus_does_not_crash() -> None:
    # [FUNCTION DOC]
    # Mines with range_max 10 on a three-document corpus without FAISS and asserts one row whose negative is the more
    # similar candidate with scores -0.50 and -0.49.
    """Requesting more candidates than the corpus holds must not crash the default (non-FAISS)
    path. `torch.topk` cannot return more columns than the corpus size, so the candidate set is
    capped at the corpus size and padded up, mirroring the FAISS branch."""
    dataset = Dataset.from_dict({"query": ["q"], "positive": ["p"]})
    # The corpus has only 3 unique entries, yet range_max + max_positives (11) asks for far more.
    result = mine_hard_negatives(
        dataset=dataset,
        model=ControlledNegativeScoreModel(),
        anchor_column_name="query",
        positive_column_name="positive",
        corpus=["p", "n_more_similar", "n_far"],
        range_max=10,
        num_negatives=1,
        use_faiss=False,
        output_scores=True,
        verbose=False,
    )
    assert len(result) == 1
    row = result[0]
    # The top real negative must be returned, with the positive excluded and no padding leaking through.
    assert row["negative"] == "n_more_similar"
    positive_score, negative_score = row["scores"]
    assert positive_score == pytest.approx(-0.50)
    assert negative_score == pytest.approx(-0.49)


def test_missing_negatives_message_names_range_max(capsys: pytest.CaptureFixture) -> None:
    # [FUNCTION DOC]
    # Mines five negatives from a tiny corpus with verbose output and asserts the printed message suggests adjusting
    # range_max.
    """With only range_max to suggest, the "Could not find enough negatives" message must still
    name it instead of printing an empty parameter list."""
    dataset = Dataset.from_dict({"query": ["q"], "positive": ["p"]})
    mine_hard_negatives(
        dataset=dataset,
        model=ControlledNegativeScoreModel(),
        anchor_column_name="query",
        positive_column_name="positive",
        corpus=["p", "n_more_similar", "n_far"],
        num_negatives=5,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "Consider adjusting the range_max parameter if" in captured.out
