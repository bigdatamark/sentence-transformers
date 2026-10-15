# [MODULE DOC]
# Pytest tests for CrossEncoderRerankingEvaluator, which scores (query, document) pairs with a CrossEncoder and
# reports ranking metrics (MAP, MRR at k, NDCG at k) keyed with the evaluator name as prefix. Samples are dicts with
# "query", "positive" (string or list) and exactly one of "negative" (list, scored together with positives) or
# "documents" (a candidate list to rerank, which also yields "base_" metrics for the original order). Uses the tiny
# reranker fixture.
"""Tests for CrossEncoderRerankingEvaluator batched prediction."""

from __future__ import annotations

import pytest

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator


def test_negative_format(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Evaluates three samples in negative format and asserts MAP, MRR at k and NDCG at k keys exist (k from the
    # evaluator) and every metric value lies in 0 to 1.
    """Basic test with 'negative' format produces valid metrics."""
    samples = [
        {
            "query": "What is Python?",
            "positive": ["Python is a programming language"],
            "negative": ["Java is a language", "C++ is fast"],
        },
        {
            "query": "What is the sun?",
            "positive": ["The sun is a star"],
            "negative": ["The moon orbits earth", "Mars is red"],
        },
        {"query": "What is water?", "positive": ["Water is H2O"], "negative": ["Salt is NaCl", "Gold is Au"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="neg-test")
    results = evaluator(reranker_bert_tiny_model)

    assert "neg-test_map" in results
    assert f"neg-test_mrr@{evaluator.at_k}" in results
    assert f"neg-test_ndcg@{evaluator.at_k}" in results
    for value in results.values():
        assert 0.0 <= value <= 1.0


def test_documents_format(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Evaluates two samples in documents format and asserts both reranked and base MAP and NDCG at k metrics are
    # reported.
    """Test with 'documents' format produces both base and reranked metrics."""
    samples = [
        {
            "query": "What is Python?",
            "positive": ["Python is a programming language"],
            "documents": ["Java is a language", "Python is a programming language", "C++ is fast"],
        },
        {
            "query": "What is the sun?",
            "positive": ["The sun is a star"],
            "documents": ["The moon orbits earth", "The sun is a star", "Mars is red"],
        },
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="doc-test")
    results = evaluator(reranker_bert_tiny_model)

    # Should have both base and reranked metrics
    assert "doc-test_map" in results
    assert "doc-test_base_map" in results
    assert f"doc-test_ndcg@{evaluator.at_k}" in results
    assert f"doc-test_base_ndcg@{evaluator.at_k}" in results


def test_no_relevant_docs_in_documents(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Evaluates one documents-format sample whose positive does not appear among its documents and asserts NDCG at k
    # is still reported (no crash).
    """Edge case: positive not in documents list."""
    samples = [
        {"query": "empty", "positive": ["x"], "documents": ["a", "b", "c"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="edge-test")
    results = evaluator(reranker_bert_tiny_model)

    assert f"edge-test_ndcg@{evaluator.at_k}" in results


def test_single_sample(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts a single negative-format sample still yields NDCG at k.
    """Single sample still works."""
    samples = [
        {"query": "test", "positive": ["yes"], "negative": ["no"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="single")
    results = evaluator(reranker_bert_tiny_model)

    assert f"single_ndcg@{evaluator.at_k}" in results


def test_string_positive(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts a positive given as a plain string (not a list) is accepted and NDCG at k is reported.
    """Positive as a string (not list) is handled correctly."""
    samples = [
        {"query": "test", "positive": "yes", "negative": ["no", "nope"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="str-pos")
    results = evaluator(reranker_bert_tiny_model)

    assert f"str-pos_ndcg@{evaluator.at_k}" in results


def test_validation_errors() -> None:
    # [FUNCTION DOC]
    # Asserts calling evaluators (with no model) on invalid samples raises ValueError: missing query (message mentions
    # "query"), missing positive ("positive"), neither negative nor documents ("exactly one"), and both negative and
    # documents ("exactly one"). Validation happens before the model is used.
    """Invalid samples raise clear errors."""
    with pytest.raises(ValueError, match="query"):
        CrossEncoderRerankingEvaluator([{"positive": ["a"], "negative": ["b"]}])(None)

    with pytest.raises(ValueError, match="positive"):
        CrossEncoderRerankingEvaluator([{"query": "q", "negative": ["b"]}])(None)

    with pytest.raises(ValueError, match="exactly one"):
        CrossEncoderRerankingEvaluator([{"query": "q", "positive": ["a"]}])(None)

    with pytest.raises(ValueError, match="exactly one"):
        CrossEncoderRerankingEvaluator([{"query": "q", "positive": ["a"], "negative": ["b"], "documents": ["c"]}])(
            None
        )
