# [PACKAGE DOC]
# Base evaluation package shared by all model families (sentence embedders, cross encoders, sparse encoders).
# Re-exports the abstract evaluator base class (BaseEvaluator), its deprecated alias SentenceEvaluator, and
# SequentialEvaluator, which runs several evaluators in a row and merges their metric dictionaries.
from __future__ import annotations

from .evaluator import BaseEvaluator, SentenceEvaluator
from .sequential import SequentialEvaluator

__all__ = [
    "BaseEvaluator",
    "SentenceEvaluator",
    "SequentialEvaluator",
]
