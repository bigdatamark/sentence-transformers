# [MODULE DOC]
# Defines BaseEvaluator, the abstract base for every evaluator used during or after training. An evaluator is a
# callable that receives a model (plus optional output path, epoch and step) and returns either a single score or a
# dictionary of named metrics. The base class tracks whether higher scores are better and which metric key is the
# "primary" one (used for best-checkpoint selection), offers helpers to prefix metric names, to push metrics into the
# model card data, to derive a human-readable description from the class name, and to expose config for the model card.
# Also defines SentenceEvaluator as a backward-compatible deprecated alias of BaseEvaluator.
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel


class BaseEvaluator:
    # [CLASS DOC]
    # Abstract evaluator. Attributes: greater_is_better (bool, default True; tells the trainer whether a larger primary
    # score is better) and primary_metric (str or None; the key of the main metric when a dict is returned). Subclasses
    # must implement the call operator; the other methods are shared utilities.
    """
    Base class for all evaluators. Notably, this class introduces the ``greater_is_better`` and ``primary_metric``
    attributes. The former is a boolean indicating whether a higher evaluation score is better, which is used
    for choosing the best checkpoint if ``load_best_model_at_end`` is set to ``True`` in the training arguments.

    The latter is a string indicating the primary metric for the evaluator. This has to be defined whenever
    the evaluator returns a dictionary of metrics, and the primary metric is the key pointing to the primary
    metric, i.e. the one that is used for model selection and/or logging.

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initialises greater_is_better to True and primary_metric to None. No other side effects.
        self.greater_is_better = True
        self.primary_metric = None

    def __call__(
        self, model: BaseModel, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> float | dict[str, float]:
        # [FUNCTION DOC]
        # Abstract evaluation entry point. Inputs: model to evaluate, optional output_path for writing result files, epoch and
        # steps (-1 means "test data" / "end of epoch", used by subclasses for file name prefixes). Should return a float or a
        # dict of metric name to float (in which case primary_metric must be set). The base implementation always raises
        # NotImplementedError.
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        Args:
            model: the model to evaluate
            output_path: path where predictions and metrics are written to
            epoch: the epoch where the evaluation takes place. This is used for the file prefixes. If this is -1,
                then we assume evaluation on test data.
            steps: the steps in the current epoch at time of the evaluation. This is used for the file prefixes.
                If this is -1, then we assume evaluation at the end of the epoch.

        Returns:
            Either a score for the evaluation with a higher score indicating a better result, or a dictionary with
            scores. If the latter is chosen, then `evaluator.primary_metric` must be defined
        """
        raise NotImplementedError

    def prefix_name_to_metrics(self, metrics: dict[str, float], name: str) -> dict[str, float]:
        # [FUNCTION DOC]
        # Converts every metric value to float where possible and, if name is non-empty, prefixes every key with the name and
        # an underscore. Inputs: metrics dict and a name string. Returns a new dict. Side effect: when name is non-empty and
        # primary_metric is set and does not already start with the prefix, primary_metric is mutated to carry the prefix too
        # (so repeated calls do not double-prefix it). With an empty name, only the float conversion is applied.
        # KNOWN ISSUE: the inner conversion only catches ValueError, so a value that float() rejects with TypeError (for
        # example None, a list or a dict) raises instead of being passed through unchanged.
        def maybe_to_float(value: Any) -> Any:
            # [FUNCTION DOC]
            # HELPER: tries to convert a value to float; returns the original value if conversion raises ValueError.
            try:
                return float(value)
            except ValueError:
                return value

        if not name:
            return {key: maybe_to_float(value) for key, value in metrics.items()}
        metrics = {name + "_" + key: maybe_to_float(value) for key, value in metrics.items()}
        if (
            hasattr(self, "primary_metric")
            and self.primary_metric is not None
            and not self.primary_metric.startswith(name + "_")
        ):
            self.primary_metric = name + "_" + self.primary_metric
        return metrics

    def store_metrics_in_model_card_data(
        self, model: BaseModel, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        # [FUNCTION DOC]
        # Stores the given metrics for this evaluator in the model's model_card_data (via its set_evaluation_metrics method)
        # together with the epoch and step, so they can be rendered into the generated model card. Returns nothing; mutates
        # the model's card data.
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch, step)

    @property
    def description(self) -> str:
        # [FUNCTION DOC]
        # Read-only property returning a human-readable name derived from the class name: a leading "CE" is expanded to
        # "CrossEncoder", everything from the first occurrence of "Evaluator" onward is cut off (if present), and a space is
        # inserted between every lowercase letter followed by an uppercase letter. Example: "BinaryClassificationEvaluator"
        # becomes "Binary Classification".
        """
        Returns a human-readable description of the evaluator: BinaryClassificationEvaluator -> Binary Classification

        1. Replace "CE" prefix with "CrossEncoder"
        2. Remove "Evaluator" from the class name
        3. Add a space before every capital letter
        """
        class_name = self.__class__.__name__

        if class_name.startswith("CE"):
            class_name = "CrossEncoder" + class_name[2:]

        try:
            index = class_name.index("Evaluator")
            class_name = class_name[:index]
        except ValueError:
            pass

        return re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", class_name)

    def get_config_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns a dict of meaningful configuration values for the model card. The base implementation returns an empty dict;
        # subclasses override it.
        """
        Return a dictionary with all meaningful configuration values of the evaluator to store in the model card.
        """
        return {}


# Backward compatibility alias, deprecated
SentenceEvaluator = BaseEvaluator
