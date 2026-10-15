# [MODULE DOC]
# Tests InformationRetrievalEvaluator: the set of metric keys it reports, invariance of metrics to corpus chunk size under exact score
# ties, and exact metric values using a deterministic keyword-based mock model.
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim


@pytest.fixture
def mock_model():
    # [FUNCTION DOC]
    # HELPER: Fixture returning a mock model. Its encoders embed each sentence as a 5-D vector that is the sum of one-hot vectors for the
    # keywords pokemon, car, vehicle, fruit and vegetable found as substrings. It uses cosine similarity and a model-card stub that
    # ignores every method call.
    def mock_encode(sentences: str | list[str], **kwargs) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Accepts a string or list of strings; returns a stacked tensor with one 5-D keyword-count vector per sentence (substring
        # matching, so overlapping keywords add up).
        """
        We simply one-hot encode the sentences. If a sentence contains a keyword, the corresponding one-hot
        encoding is added to the sentence embedding.
        """
        one_hot_encodings = {
            "pokemon": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
            "car": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
            "vehicle": torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
            "fruit": torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
            "vegetable": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = []
        for sentence in sentences:
            encoding = torch.zeros(5)
            for keyword, one_hot in one_hot_encodings.items():
                if keyword in sentence:
                    encoding += one_hot
            embeddings.append(encoding)
        return torch.stack(embeddings)

    class _MockModelCardData:
        # [CLASS DOC]
        # HELPER: Model-card data stub; any attribute lookup returns a function that accepts anything and returns None.
        def __getattr__(self, name):
            # [FUNCTION DOC]
            # HELPER: Returns a no-op callable for any missing attribute name.
            return lambda *args, **kwargs: None

    class _MockModel:
        # [CLASS DOC]
        # HELPER: Mock model with similarity_fn_name "cosine" and a stub model_card_data, exposing similarity, encode, encode_query and
        # encode_document.
        similarity_fn_name = "cosine"
        model_card_data = _MockModelCardData()

        def similarity(self, a, b):
            # [FUNCTION DOC]
            # HELPER: Returns the cosine similarity matrix between the two embedding batches.
            return cos_sim(a, b)

        def encode(self, sentences, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Delegates to the keyword one-hot encoder.
            return mock_encode(sentences, **kwargs)

        def encode_query(self, sentences, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Query encoder; delegates to the keyword one-hot encoder.
            return mock_encode(sentences, **kwargs)

        def encode_document(self, sentences, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Document encoder; delegates to the keyword one-hot encoder.
            return mock_encode(sentences, **kwargs)

    return _MockModel()


@pytest.fixture
def test_data():
    # [FUNCTION DOC]
    # HELPER: Fixture returning a tuple of five queries, a five-document corpus (both keyed by string ids) and the relevance mapping from
    # query id to the set of relevant doc ids (the vehicle query has two relevant docs).
    queries = {
        "0": "What is a pokemon?",
        "1": "What is a vegetable?",
        "2": "What is a fruit?",
        "3": "What is a vehicle?",
        "4": "What is a car?",
    }
    corpus = {
        "0": "A pokemon is a fictional creature",
        "1": "A vegetable is a plant",
        "2": "A fruit is a plant",
        "3": "A vehicle is a machine",
        "4": "A car is a vehicle",
    }
    relevant_docs = {"0": {"0"}, "1": {"1"}, "2": {"2"}, "3": {"3", "4"}, "4": {"4"}}
    return queries, corpus, relevant_docs


def test_simple(test_data, stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path):
    # [FUNCTION DOC]
    # Runs the evaluator named "test" with the tiny model and specific cutoffs, writing CSV to tmp_path, and asserts the returned keys are
    # exactly the expected cosine accuracy, precision, recall, ndcg, mrr and map metrics at those cutoffs.
    queries, corpus, relevant_docs = test_data
    model = stsb_bert_tiny_model

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[5],
    )
    results = ir_evaluator(model, output_path=str(tmp_path))
    expected_keys = [
        "test_cosine_accuracy@1",
        "test_cosine_accuracy@3",
        "test_cosine_precision@1",
        "test_cosine_precision@3",
        "test_cosine_recall@1",
        "test_cosine_recall@3",
        "test_cosine_ndcg@3",
        "test_cosine_mrr@3",
        "test_cosine_map@5",
    ]
    assert set(results.keys()) == set(expected_keys)


def test_metrics_are_independent_of_corpus_chunk_size(stsb_bert_tiny_model: SentenceTransformer):
    # [FUNCTION DOC]
    # Builds a corpus of 20 identical documents so all scores tie, and evaluates two queries with corpus chunk sizes 4, 8, 20 and 50 (batch
    # size 4, no CSV). Asserts every run's metrics equal the chunk size 50 run (ties broken by corpus id) and that accuracy@1 is 0.5
    # (only the first query's relevant d00 ranks first).
    """Duplicate documents produce exact score ties: breaking them by corpus_id keeps every metric
    identical across corpus_chunk_size values."""
    queries = {"q0": "What is the capital of France?", "q1": "Who painted the Mona Lisa?"}
    corpus = {f"d{idx:02d}": "Paris is the capital of France." for idx in range(20)}
    relevant_docs = {"q0": {"d00", "d15"}, "q1": {"d07"}}

    results_per_chunk_size = {}
    # Every effective chunk length (4, 8, 20, 20) is a multiple of batch_size 4, so every encode
    # batch holds 4 identical texts and the duplicates tie bit-exactly in every chunking (larger
    # batches vary in the last ulp).
    for corpus_chunk_size in [4, 8, 20, 50]:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="chunked",
            corpus_chunk_size=corpus_chunk_size,
            batch_size=4,
            accuracy_at_k=[1, 5],
            precision_recall_at_k=[1, 5],
            mrr_at_k=[10],
            ndcg_at_k=[10],
            map_at_k=[10],
            write_csv=False,
        )
        results_per_chunk_size[corpus_chunk_size] = ir_evaluator(stsb_bert_tiny_model)

    baseline = results_per_chunk_size[50]
    for chunk_size, results in results_per_chunk_size.items():
        assert results == baseline, f"corpus_chunk_size={chunk_size} changed the metrics"
    # All 20 documents tie, so ranks follow ascending corpus_id: only q0 has its d00 at rank 1.
    assert baseline["chunked_cosine_accuracy@1"] == 0.5


def test_metrics(test_data, mock_model, tmp_path: Path):
    # [FUNCTION DOC]
    # Evaluates the mock model on the test data and asserts exact metric values: perfect accuracy, precision@1, recall@3, ndcg, mrr and
    # map; precision@3 of 0.4 and recall@1 of 0.9 (the vehicle query has two relevant docs). Writes CSV to tmp_path.
    queries, corpus, relevant_docs = test_data

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[5],
    )
    results = ir_evaluator(mock_model, output_path=str(tmp_path))
    # We expect test_cosine_precision@3 to be 0.4, since 6 out of 15 (5 queries * 3) are True Positives
    # We expect test_cosine_recall@1 to be 0.9: the average of 4 times a recall of 1 and once a recall of 0.5
    expected_results = {
        "test_cosine_accuracy@1": 1.0,
        "test_cosine_accuracy@3": 1.0,
        "test_cosine_precision@1": 1.0,
        "test_cosine_precision@3": 0.4,
        "test_cosine_recall@1": 0.9,
        "test_cosine_recall@3": 1.0,
        "test_cosine_ndcg@3": 1.0,
        "test_cosine_mrr@3": 1.0,
        "test_cosine_map@5": 1.0,
    }

    for key, expected_value in expected_results.items():
        assert results[key] == pytest.approx(expected_value, abs=1e-9)
