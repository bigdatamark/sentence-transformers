# [MODULE DOC]
# Tests BinaryClassificationEvaluator: its static threshold search helpers for best F1 and best accuracy (checked against sklearn),
# correct direction for distance metrics, max_* aggregation across several similarity functions, and CSV output column alignment.
"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import evaluation


def test_BinaryClassificationEvaluator_find_best_f1_and_threshold() -> None:
    # [FUNCTION DOC]
    # Generates 1000 random binary labels and random scores, finds the best-F1 threshold (higher score means similar), thresholds the
    # scores at greater-or-equal, and asserts sklearn's F1 matches the returned best F1 within 1e-6. Non-seeded randomness.
    """Tests that the F1 score for the computed threshold is correct"""
    y_true = np.random.randint(0, 2, 1000)
    y_pred_cosine = np.random.randn(1000)
    (
        best_f1,
        best_precision,
        best_recall,
        threshold,
    ) = evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(
        y_pred_cosine, y_true, high_score_more_similar=True
    )
    y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
    sklearn_f1score = f1_score(y_true, y_pred_labels)
    assert np.abs(best_f1 - sklearn_f1score) < 1e-6


def test_BinaryClassificationEvaluator_find_best_accuracy_and_threshold() -> None:
    # [FUNCTION DOC]
    # Same as the F1 test but for the best-accuracy threshold search, comparing with sklearn's accuracy within 1e-6.
    """Tests that the Acc score for the computed threshold is correct"""
    y_true = np.random.randint(0, 2, 1000)
    y_pred_cosine = np.random.randn(1000)
    (
        max_acc,
        threshold,
    ) = evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(
        y_pred_cosine, y_true, high_score_more_similar=True
    )
    y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
    sklearn_acc = accuracy_score(y_true, y_pred_labels)
    assert np.abs(max_acc - sklearn_acc) < 1e-6


@pytest.mark.parametrize("similarity_fn_name", ["euclidean", "manhattan"])
def test_BinaryClassificationEvaluator_distance_metrics_direction(similarity_fn_name: str) -> None:
    # [FUNCTION DOC]
    # Parametrized over euclidean and manhattan. Uses a mock encoder mapping fixed string keys to 2-D points forming two close pairs
    # (distances 1 and 2, label 1) and two far pairs (distances 10 and 20, label 0). Asserts accuracy, F1, precision, recall, AP and MCC
    # are all 1.0 and that the accuracy and F1 thresholds are the positive distance 6 (midpoint of 2 and 10).
    """The euclidean/manhattan metrics must treat smaller distances as more similar.

    ``pairwise_euclidean_sim``/``pairwise_manhattan_sim`` return negative distances (i.e.
    similarities), but were passed unnegated to the ``greater_is_better=False`` metric
    computations, which expect positive distances. That inverted the ranking, so accuracy,
    F1, precision, recall, AP and MCC were all computed for the opposite classifier
    ("larger distance means similar") and the reported thresholds were negative.
    """
    embeddings = {
        "a1": [0.0, 0.0],
        "a2": [1.0, 0.0],  # distance 1 to a1, similar pair (label 1)
        "b1": [0.0, 10.0],
        "b2": [0.0, 12.0],  # distance 2 to b1, similar pair (label 1)
        "c1": [0.0, 0.0],
        "c2": [10.0, 0.0],  # distance 10 to c1, dissimilar pair (label 0)
        "d1": [5.0, 5.0],
        "d2": [5.0, 25.0],  # distance 20 to d1, dissimilar pair (label 0)
    }

    class MockEncoder:
        # [CLASS DOC]
        # HELPER: Minimal encoder exposing only an encode method.
        def encode(self, sentences, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns a float32 array of the predefined embeddings for the given sentence keys, ignoring extra keyword arguments.
            return np.array([embeddings[sentence] for sentence in sentences], dtype=np.float32)

    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=["a1", "b1", "c1", "d1"],
        sentences2=["a2", "b2", "c2", "d2"],
        labels=[1, 1, 0, 0],
        similarity_fn_names=[similarity_fn_name],
    )
    scores = evaluator.compute_metrics(MockEncoder())[similarity_fn_name]

    # The similar pairs (distances 1 and 2) are perfectly separable from the
    # dissimilar pairs (distances 10 and 20), so every metric should be perfect.
    for metric in ["accuracy", "f1", "precision", "recall", "ap", "mcc"]:
        assert scores[metric] == pytest.approx(1.0), f"{metric}: {scores[metric]}"

    # The thresholds should be positive distances separating the similar from the
    # dissimilar pairs: (2 + 10) / 2 = 6 for both euclidean and manhattan.
    assert scores["accuracy_threshold"] == pytest.approx(6.0)
    assert scores["f1_threshold"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "similarity_fn_names",
    [["dot", "euclidean"], ["cosine", "dot"], ["cosine", "dot", "euclidean", "manhattan"]],
)
def test_BinaryClassificationEvaluator_multiple_similarity_fn_names(
    stsb_bert_tiny_model: SentenceTransformer, similarity_fn_names: list[str]
) -> None:
    # [FUNCTION DOC]
    # Parametrized over similarity function lists that may omit cosine. Evaluates three labelled sentence pairs with the tiny model and
    # asserts the primary metric is "max_ap" and each max_ metric equals the maximum of that metric across the requested functions.
    """Tests that the max_* aggregation does not assume that cosine was requested"""
    model = stsb_bert_tiny_model
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=["A man is eating food.", "A cat sits outside.", "The girl plays guitar."],
        sentences2=["A man eats something.", "The sky is blue.", "A woman plays a guitar."],
        labels=[1, 0, 1],
        similarity_fn_names=similarity_fn_names,
    )
    metrics = evaluator(model)

    assert evaluator.primary_metric == "max_ap"
    for metric in ["accuracy", "f1", "precision", "recall", "ap", "mcc"]:
        assert metrics[f"max_{metric}"] == max(metrics[f"{name}_{metric}"] for name in similarity_fn_names)


@pytest.mark.parametrize("similarity_fn_names", [["cosine"], ["cosine", "dot", "euclidean", "manhattan"]])
def test_BinaryClassificationEvaluator_csv_columns_are_aligned(
    stsb_bert_tiny_model: SentenceTransformer, tmp_path, similarity_fn_names: list[str]
) -> None:
    # [FUNCTION DOC]
    # Evaluates with output to tmp_path, reads the written CSV, and asserts it has a header plus one data row with equal column counts
    # (threshold metrics with extra underscores must not be dropped from the data row). Writes and reads a file.
    """The results CSV data row must have as many columns as the header row.

    The ``*_accuracy_threshold`` and ``*_f1_threshold`` metrics contain a second underscore, so a
    ``count("_") == 1`` filter dropped them from the data row while keeping them in the header,
    shifting every later column and leaving the trailing ``*_ap``/``*_mcc`` columns empty.
    """
    model = stsb_bert_tiny_model
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=["A man is eating food.", "A cat sits outside.", "The girl plays guitar."],
        sentences2=["A man eats something.", "The sky is blue.", "A woman plays a guitar."],
        labels=[1, 0, 1],
        similarity_fn_names=similarity_fn_names,
    )
    evaluator(model, output_path=str(tmp_path))

    with open(tmp_path / evaluator.csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 2
    assert len(rows[1]) == len(rows[0])
