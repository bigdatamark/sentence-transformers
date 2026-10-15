# [MODULE DOC] Triplet evaluator: for (anchor, positive, negative) triplets, measures the fraction where the anchor is more
# similar to the positive than to the negative by more than a margin, for one or more similarity functions.
# Anchors use the model's query encoder and positives/negatives its document encoder. Optionally writes CSV.
from __future__ import annotations

import csv
import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.sentence_transformer.readers import InputExample
from sentence_transformers.util import (
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)
from sentence_transformers.util.similarity import SimilarityFunction

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class TripletEvaluator(BaseEvaluator):
    # [CLASS DOC] Evaluator state: anchors, positives, negatives (equal length), name, truncate_dim, main_similarity_function
    # (optional SimilarityFunction selecting the primary metric), similarity_fn_names (subset of cosine, dot,
    # manhattan, euclidean; empty means use the model's function at call time), margin (dict from function name
    # to margin, defaults 0), batch_size, show_progress_bar, write_csv, CSV file name and headers.
    # KNOWN ISSUE: The docstring says that without main_similarity_function all four similarity functions are used; the code uses similarity_fn_names, defaulting to the model's single similarity function.
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
    Checks if ``similarity(sentence, positive_example) > similarity(sentence, negative_example) + margin``.

    Anchors are embedded with the model's ``encode_query`` and positives / negatives with
    ``encode_document``, so an asymmetric model's ``query`` / ``document`` prompts are applied during
    evaluation (a no-op when those prompts are unset). This assumes retrieval-style triplets
    (anchor = query, positive / negative = document). For a symmetric model, either leave the
    ``query`` / ``document`` prompts unset so all three inputs are embedded identically, or set both
    to the same prompt to apply one prefix uniformly.

    Args:
        anchors (List[str]): Sentences to check similarity to. (e.g. a query)
        positives (List[str]): List of positive sentences
        negatives (List[str]): List of negative sentences
        main_similarity_function (Union[str, SimilarityFunction], optional):
            The similarity function to use. If not specified, use cosine similarity,
            dot product, Euclidean, and Manhattan similarity. Defaults to None.
        margin (Union[float, Dict[str, float]], optional): Margins for various similarity metrics.
            If a float is provided, it will be used as the margin for all similarity metrics.
            If a dictionary is provided, the keys must be 'cosine', 'dot', 'manhattan', and 'euclidean'.
            The value specifies the minimum margin by which the negative sample should be further from
            the anchor than the positive sample. Defaults to None.
        name (str): Name for the output. Defaults to "".
        batch_size (int): Batch size used to compute embeddings. Defaults to 16.
        show_progress_bar (bool): If true, prints a progress bar. Defaults to False.
        write_csv (bool): Write results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
            `None` uses the model's current truncation dimension. Defaults to None.
        similarity_fn_names (List[str], optional): List of similarity function names to evaluate.
            If not specified, evaluate using the ``model.similarity_fn_name``.
            Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

            # Load a dataset with (anchor, positive, negative) triplets
            dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

            # Initialize the TripletEvaluator using anchors, positives, and negatives
            triplet_evaluator = TripletEvaluator(
                anchors=dataset[:1000]["anchor"],
                positives=dataset[:1000]["positive"],
                negatives=dataset[:1000]["negative"],
                name="all_nli_dev",
            )
            results = triplet_evaluator(model)
            '''
            TripletEvaluator: Evaluating the model on the all-nli-dev dataset:
            Accuracy Cosine Similarity:        95.60%
            '''
            print(triplet_evaluator.primary_metric)
            # => "all_nli_dev_cosine_accuracy"
            print(results[triplet_evaluator.primary_metric])
            # => 0.956
    """

    def __init__(
        self,
        anchors: Sequence[SingleInput],
        positives: Sequence[SingleInput],
        negatives: Sequence[SingleInput],
        main_similarity_function: str | SimilarityFunction | None = None,
        margin: float | dict[str, float] | None = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
        similarity_fn_names: list[Literal["cosine", "dot", "euclidean", "manhattan"]] | None = None,
        main_distance_function: str | SimilarityFunction | None = "deprecated",
    ):
        # [FUNCTION DOC] Store inputs and assert equal lengths. The deprecated main_distance_function is used as the main similarity
        # (with a warning) when given and no main similarity is set. Convert the main similarity to a SimilarityFunction
        # (ValueError if invalid). Raise ValueError for unknown similarity names, unknown margin keys or a margin that
        # is neither number, dict nor None. Build the margin dict (a number applies to all functions; a dict overrides
        # the zero defaults). If show_progress_bar is None, enable it at INFO/DEBUG logging. Build CSV name and headers.
        super().__init__()
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.name = name
        self.truncate_dim = truncate_dim

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        if main_distance_function != "deprecated" and main_similarity_function is None:
            main_similarity_function = main_distance_function
            logger.warning(
                "The 'main_distance_function' parameter is deprecated. Please use 'main_similarity_function' instead. "
                "'main_distance_function' will be removed in a future release."
            )

        self.main_similarity_function = (
            SimilarityFunction(main_similarity_function) if main_similarity_function else None
        )
        self.similarity_fn_names = similarity_fn_names or []

        fn_keys = list(self._get_similarity_functions().keys())
        if unknown_names := sorted(set(self.similarity_fn_names) - set(fn_keys)):
            raise ValueError(
                f"`similarity_fn_names` got unexpected names {unknown_names}, expected names in {fn_keys}."
            )
        if margin is None:
            self.margin = {k: 0 for k in fn_keys}
        elif isinstance(margin, (float, int)):
            self.margin = {k: margin for k in fn_keys}
        elif isinstance(margin, dict):
            if unknown_keys := sorted(set(margin) - set(fn_keys)):
                raise ValueError(f"`margin` got unexpected keys {unknown_keys}, expected keys in {fn_keys}.")
            self.margin = {**{k: 0 for k in fn_keys}, **margin}
        else:
            raise ValueError(f"`margin` should be a float or a dictionary with keys in {fn_keys}.")

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps"]
        self.write_csv = write_csv

        self._append_csv_headers(self.similarity_fn_names)

    def _append_csv_headers(self, similarity_fn_names):
        # [FUNCTION DOC] HELPER: Append an "accuracy_<fn>" CSV header for each given function name.
        for fn_name in similarity_fn_names:
            self.csv_headers.append(f"accuracy_{fn_name}")

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        # [FUNCTION DOC] Alternate constructor from InputExamples using texts[0], texts[1], texts[2] as anchor, positive, negative.
        anchors = []
        positives = []
        negatives = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        return cls(anchors, positives, negatives, **kwargs)

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        # [FUNCTION DOC] Log a header; embed anchors with the query encoder and positives/negatives with the document encoder.
        # Adopt the model's similarity function if none configured. For each known function compute pairwise
        # anchor-positive and anchor-negative scores (distances as negative similarities) and accuracy as the mean of
        # positive score greater than negative score plus the margin; store as "<fn>_accuracy" and log. Optionally
        # write/append a CSV row of values. With several functions add max_accuracy. Primary metric: main function's
        # accuracy if set, else max_accuracy for several functions, else the single function's accuracy. Returns
        # name-prefixed metrics after storing them in model card data.
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"TripletEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        embeddings_anchors = self.embed_inputs(model, self.anchors, encode_fn_name="query")
        embeddings_positives = self.embed_inputs(model, self.positives, encode_fn_name="document")
        embeddings_negatives = self.embed_inputs(model, self.negatives, encode_fn_name="document")

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = self._get_similarity_functions()

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                positive_scores, negative_scores = similarity_functions[fn_name](
                    embeddings_anchors, embeddings_positives, embeddings_negatives
                )
                accuracy = (positive_scores > negative_scores + self.margin[fn_name]).float().mean().item()
                metrics[f"{fn_name}_accuracy"] = accuracy
                logger.info(f"Accuracy {fn_name.capitalize()} Similarity:\t{accuracy:.2%}")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps] + list(metrics.values()))

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps] + list(metrics.values()))

        if len(self.similarity_fn_names) > 1:
            metrics["max_accuracy"] = max(metrics.values())

        if self.main_similarity_function:
            self.primary_metric = f"{self.main_similarity_function.value}_accuracy"
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "max_accuracy"
            else:
                self.primary_metric = f"{self.similarity_fn_names[0]}_accuracy"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def embed_inputs(
        self,
        model: SentenceTransformer,
        sentences: SingleInput | Sequence[SingleInput] | np.ndarray,
        encode_fn_name: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        # [FUNCTION DOC] HELPER: Encode inputs to numpy with the query encoder ("query"), document encoder ("document") or generic
        # encoder otherwise, using batch size, progress bar and truncation settings.
        if encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        else:
            encode_fn = model.encode
        return encode_fn(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            truncate_dim=self.truncate_dim,
            **kwargs,
        )

    def _get_similarity_functions(self) -> dict:
        # [FUNCTION DOC] HELPER: Return a dict mapping "cosine", "dot", "manhattan", "euclidean" to functions that take anchor,
        # positive and negative embeddings and return the pair of pairwise anchor-positive and anchor-negative scores.
        return {
            "cosine": lambda a, p, n: (pairwise_cos_sim(a, p), pairwise_cos_sim(a, n)),
            "dot": lambda a, p, n: (pairwise_dot_score(a, p), pairwise_dot_score(a, n)),
            "manhattan": lambda a, p, n: (pairwise_manhattan_sim(a, p), pairwise_manhattan_sim(a, n)),
            "euclidean": lambda a, p, n: (pairwise_euclidean_sim(a, p), pairwise_euclidean_sim(a, n)),
        }

    def get_config_dict(self):
        # [FUNCTION DOC] Return a config dict with margin (only when any margin is non-zero) and truncate_dim when set.
        config_dict = {}
        if self.margin != {key: 0 for key in self._get_similarity_functions()}:
            config_dict["margin"] = self.margin
        if self.truncate_dim is not None:
            config_dict["truncate_dim"] = self.truncate_dim
        return config_dict
