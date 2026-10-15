# [MODULE DOC]
# Deprecated legacy CrossEncoder evaluator names kept for backwards compatibility. Each class is marked
# deprecated (emitting a deprecation warning on use) and subclasses one of the new evaluators. The
# classification-style ones add a constructor from legacy InputExample objects.
from __future__ import annotations

from typing_extensions import deprecated

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation.classification import CrossEncoderClassificationEvaluator
from sentence_transformers.cross_encoder.evaluation.correlation import CrossEncoderCorrelationEvaluator
from sentence_transformers.cross_encoder.evaluation.reranking import CrossEncoderRerankingEvaluator


@deprecated(
    "This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator. "
    "Please use CrossEncoderClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CEBinaryAccuracyEvaluator(CrossEncoderClassificationEvaluator):
    # [CLASS DOC]
    # Deprecated alias of the classification evaluator (formerly binary accuracy). Behavior is inherited.
    """
    This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        # [FUNCTION DOC]
        # Build an evaluator from a list of InputExample objects: collects each example's texts as a pair and its
        # label, then calls the constructor with those pairs, labels and any extra kwargs. Returns the instance.
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


@deprecated(
    "This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator. "
    "Please use CrossEncoderClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CEBinaryClassificationEvaluator(CrossEncoderClassificationEvaluator):
    # [CLASS DOC]
    # Deprecated alias of the classification evaluator (formerly binary classification). Behavior inherited.
    """
    This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        # [FUNCTION DOC]
        # Build an evaluator from InputExample objects: pairs from each example's texts, labels from its label;
        # extra kwargs forwarded to the constructor. Returns the instance.
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


@deprecated(
    "This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator. "
    "Please use CrossEncoderClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CEF1Evaluator(CrossEncoderClassificationEvaluator):
    # [CLASS DOC]
    # Deprecated alias of the classification evaluator (formerly F1). Behavior inherited.
    """
    This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        # [FUNCTION DOC]
        # Build an evaluator from InputExample objects: pairs from each example's texts, labels from its label;
        # extra kwargs forwarded to the constructor. Returns the instance.
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


@deprecated(
    "This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator. "
    "Please use CrossEncoderClassificationEvaluator instead, which supports both binary and multi-class "
    "evaluation. It accepts approximately the same inputs as this evaluator."
)
class CESoftmaxAccuracyEvaluator(CrossEncoderClassificationEvaluator):
    # [CLASS DOC]
    # Deprecated alias of the classification evaluator (formerly softmax accuracy). Behavior inherited.
    """
    This evaluator has been deprecated in favor of the more general CrossEncoderClassificationEvaluator.
    """

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        # [FUNCTION DOC]
        # Build an evaluator from InputExample objects: pairs from each example's texts, labels from its label;
        # extra kwargs forwarded to the constructor. Returns the instance.
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


@deprecated(
    "The CECorrelationEvaluator has been renamed to CrossEncoderCorrelationEvaluator. "
    "Please use CrossEncoderCorrelationEvaluator instead."
)
class CECorrelationEvaluator(CrossEncoderCorrelationEvaluator):
    # [CLASS DOC]
    # Deprecated alias of the correlation evaluator; adds nothing.
    pass


@deprecated(
    "The CERerankingEvaluator has been renamed to CrossEncoderCorrelationEvaluator. "
    "Please use CrossEncoderCorrelationEvaluator instead."
)
class CERerankingEvaluator(CrossEncoderRerankingEvaluator):
    # [CLASS DOC]
    # Deprecated alias of the reranking evaluator; adds nothing.
    # KNOWN ISSUE: The deprecation message says this was renamed to CrossEncoderCorrelationEvaluator, but the
    # class actually subclasses (and should point users to) CrossEncoderRerankingEvaluator.
    pass
