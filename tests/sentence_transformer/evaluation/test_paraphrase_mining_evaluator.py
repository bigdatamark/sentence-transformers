# [MODULE DOC]
# Tests for ParaphraseMiningEvaluator, which mines paraphrase pairs from a set of sentences and scores them against gold pairs.
# KNOWN ISSUE: The module docstring says it tests BinaryClassificationEvaluator, but the module actually tests ParaphraseMiningEvaluator.
"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import ParaphraseMiningEvaluator


def test_ParaphraseMiningEvaluator(
    paraphrase_distilroberta_base_v1_model: SentenceTransformer, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Builds four sentences keyed by integer id forming two obvious paraphrase pairs (0,1) and (2,3), runs the evaluator with
    # those gold pairs, writing output files to tmp_path, and asserts the primary metric is above 0.99.
    # Inputs: paraphrase_distilroberta_base_v1_model fixture and tmp_path.
    """Tests that the ParaphraseMiningEvaluator can be loaded"""
    model = paraphrase_distilroberta_base_v1_model
    sentences = {
        0: "Hello World",
        1: "Hello World!",
        2: "The cat is on the table",
        3: "On the table the cat is",
    }
    data_eval = ParaphraseMiningEvaluator(sentences, [(0, 1), (2, 3)])
    metrics = data_eval(model, output_path=str(tmp_path))
    assert metrics[data_eval.primary_metric] > 0.99
