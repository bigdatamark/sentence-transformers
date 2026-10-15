# [MODULE DOC]
# Pytest tests that the deprecated CrossEncoder evaluator classes (binary accuracy, binary classification, F1, softmax
# accuracy, reranking, correlation) remain importable both from their legacy per-class module paths under the
# cross-encoder evaluation package and from the package itself, and that these paths expose the very same class
# objects defined in the deprecated module.
from __future__ import annotations

import importlib

import pytest

from sentence_transformers.cross_encoder.evaluation.deprecated import (
    CEBinaryAccuracyEvaluator,
    CEBinaryClassificationEvaluator,
    CECorrelationEvaluator,
    CEF1Evaluator,
    CERerankingEvaluator,
    CESoftmaxAccuracyEvaluator,
)


@pytest.mark.parametrize(
    ("module_names", "module_attributes"),
    [
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CEBinaryAccuracyEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CEBinaryAccuracyEvaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CEBinaryClassificationEvaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CEF1Evaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CEF1Evaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CESoftmaxAccuracyEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CESoftmaxAccuracyEvaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CERerankingEvaluator,
            ],
        ),
        (
            [
                "sentence_transformers.cross_encoder.evaluation.CECorrelationEvaluator",
                "sentence_transformers.cross_encoder.evaluation",
            ],
            [
                CECorrelationEvaluator,
            ],
        ),
    ],
)
def test_import(module_names: list[str], module_attributes: list[object]) -> None:
    # [FUNCTION DOC]
    # Parametrised with a list of module paths and a list of expected classes. Imports each module path (raises
    # ImportError if a legacy path is missing) and asserts each class is available on it under its own name and is
    # identical to the expected class object.
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for module_attribute in module_attributes:
            obj = getattr(module, module_attribute.__name__, None)
            assert obj is module_attribute
