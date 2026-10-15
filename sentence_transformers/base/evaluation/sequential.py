# [MODULE DOC]
# Defines SequentialEvaluator, an evaluator that runs a sequence of sub-evaluators one after another on the same model
# and combines all their metrics into one dictionary plus an aggregated "sequential_score".
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel


class SequentialEvaluator(BaseEvaluator):
    # [CLASS DOC]
    # Composite evaluator. Attributes: evaluators (the iterable of sub-evaluators, stored as given) and main_score_function
    # (callable taking the list of per-evaluator main scores and returning one aggregate score; defaults to taking the last
    # score). Its primary metric is always "sequential_score".
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    All scores are passed to 'main_score_function', which derives one final score value

    Args:
        evaluators (Iterable[BaseEvaluator]): A collection of BaseEvaluator objects.
        main_score_function (function, optional): A function that takes a list of scores and returns the main score.
            Defaults to selecting the last score in the list.

    Example:
        ::

            evaluator1 = BinaryClassificationEvaluator(...)
            evaluator2 = InformationRetrievalEvaluator(...)
            evaluator3 = MSEEvaluator(...)
            seq_evaluator = SequentialEvaluator([evaluator1, evaluator2, evaluator3])
    """

    def __init__(self, evaluators: Iterable[BaseEvaluator], main_score_function=lambda scores: scores[-1]):
        # [FUNCTION DOC]
        # Calls the base initialiser (greater_is_better True, primary_metric None) and stores the evaluators and the main score
        # function unchanged.
        # KNOWN ISSUE: evaluators is typed as any Iterable and stored as is; if a one-shot iterator or generator is passed, it
        # is exhausted after the first evaluation and later calls evaluate nothing.
        super().__init__()
        self.evaluators = evaluators
        self.main_score_function = main_score_function

    def __call__(
        self, model: BaseModel, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        # [FUNCTION DOC]
        # Runs every sub-evaluator with the same model, output_path, epoch and steps. For each result: if it is a plain score,
        # the score is collected and the result is wrapped into a dict under the key "evaluator_<index>"; if it is a dict, the
        # value under the sub-evaluator's primary_metric (or, if that is unset, the first key's value) is collected as its
        # score. Sets self.primary_metric to "sequential_score". The aggregate is 0.0 when no scores were collected, otherwise
        # main_score_function applied to the score list. Returns one flat dict merging all sub-results (later keys overwrite
        # earlier equal keys) plus "sequential_score" holding the aggregate.
        evaluations = []
        scores = []
        for evaluator_idx, evaluator in enumerate(self.evaluators):
            evaluation = evaluator(model, output_path, epoch, steps)

            if not isinstance(evaluation, dict):
                scores.append(evaluation)
                evaluation = {f"evaluator_{evaluator_idx}": evaluation}
            else:
                if hasattr(evaluator, "primary_metric") and evaluator.primary_metric is not None:
                    scores.append(evaluation[evaluator.primary_metric])
                else:
                    scores.append(evaluation[list(evaluation.keys())[0]])

            evaluations.append(evaluation)

        self.primary_metric = "sequential_score"
        if not scores:
            main_score = 0.0
        else:
            main_score = self.main_score_function(scores)
        results = {key: value for evaluation in evaluations for key, value in evaluation.items()}
        results["sequential_score"] = main_score
        return results
