# [MODULE DOC]
# Base data collator used by the trainers: turns a list of dataset rows into a model-ready batch dictionary by extracting a
# label column and preprocessing every remaining text (or other input) column with the model's preprocess function,
# resolving per-column/per-dataset prompts, router tasks and max-length overrides. DEFAULT_LABEL_COLUMNS lists the column
# names treated as labels, in priority order. Uses the transformers logger (supports warning_once).
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.utils import logging as transformers_logging

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)

# Keep in sync with util.dataset.DEFAULT_LABEL_COLUMNS (a shared import would be circular).
DEFAULT_LABEL_COLUMNS: tuple[str, ...] = ("label", "labels", "score", "scores")


@dataclass
class BaseDataCollator:
    # [CLASS DOC]
    # Dataclass collator. Attributes: preprocess_fn (callable taking a list of inputs plus prompt, task and optionally
    # max_length keywords, returning a dict of tensors), valid_label_columns (default the DEFAULT_LABEL_COLUMNS list),
    # router_mapping (column -> task, or dataset name -> (column -> task); default empty), prompts (a single string, column ->
    # prompt, or dataset name -> (column -> prompt); default empty), max_length (int for all columns, dict keyed by task, or
    # None), and the private _warned_columns set of column-name tuples already checked for ordering. Calling the instance
    # collates a batch; output keys are "<column>_<preprocessed key>", optional "label", "dataset_name" and "<column>_task".
    """Base data collator for Sentence Transformers models.

    Preprocesses text columns via ``preprocess_fn`` (typically ``model.preprocess``),
    producing ``{column}_input_ids``, ``{column}_attention_mask``, etc.  Handles prompt
    resolution (per-column or per-dataset) and Router task mapping.

    It is important that the columns are in the expected order. For example, if your dataset has columns
    "answer", "question" in that order, then the MultipleNegativesRankingLoss will consider
    "answer" as the anchor and "question" as the positive, and it will (unexpectedly) optimize for
    "given the answer, what is the question?".
    """

    preprocess_fn: Callable
    valid_label_columns: list[str] = field(default_factory=lambda: list(DEFAULT_LABEL_COLUMNS))
    router_mapping: dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict, repr=False)
    prompts: str | dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict, repr=False)
    max_length: int | dict[str, int] | None = field(default=None, repr=False)

    _warned_columns: set[tuple[str, ...]] = field(default_factory=set, init=False, repr=False)

    def _resolve_router_mapping(self, batch: dict[str, Any]) -> dict[str, str]:
        # [FUNCTION DOC]
        # HELPER: return the column -> task mapping for this batch. If router_mapping is non-empty and its first value is a dict
        # (per-dataset form), return the entry for the batch's dataset_name, or an empty dict if absent; otherwise return
        # router_mapping, or an empty dict if it is falsy.
        """Resolve the router mapping for this batch, handling nested (per-dataset) mappings."""
        router_mapping = self.router_mapping
        if (
            router_mapping
            and isinstance(router_mapping, dict)
            and isinstance(next(iter(router_mapping.values())), dict)
        ):
            if "dataset_name" in batch and batch["dataset_name"] in router_mapping:
                return router_mapping[batch["dataset_name"]]
            return {}
        return router_mapping or {}

    def _resolve_prompts(self, batch: dict[str, Any]) -> str | dict[str, str] | None:
        # [FUNCTION DOC]
        # HELPER: return the prompts applicable to this batch. Non-dict or empty prompts are returned unchanged. If the batch has a
        # dataset_name that is a key of prompts, return that entry. If prompts is nested (first value is a dict): raise ValueError
        # when the batch has no dataset_name, else return an empty dict (dataset without prompts). Otherwise return prompts as a
        # column -> prompt mapping.
        """Resolve the prompts for this batch, handling nested (per-dataset) mappings."""
        prompts = self.prompts
        if not isinstance(prompts, dict) or not prompts:
            return prompts

        is_multi_dataset = "dataset_name" in batch
        if is_multi_dataset and batch["dataset_name"] in prompts:
            return prompts[batch["dataset_name"]]

        if isinstance(next(iter(prompts.values())), dict):
            if not is_multi_dataset:
                raise ValueError(
                    "The prompts provided to the trainer are a nested dictionary. In this setting, the first "
                    "level of the dictionary should map to dataset names and the second level to column names. "
                    "However, as the provided dataset is a not a DatasetDict, no dataset names can be inferred. "
                    f"The keys to the provided prompts dictionary are {list(prompts.keys())!r}"
                )
            return {}

        return prompts

    def _get_prompt_for_column(self, prompts: str | dict[str, str], column_name: str) -> str | None:
        # [FUNCTION DOC]
        # HELPER: return the prompt for a column: the string itself when prompts is a string, the column's entry when prompts is a
        # dict containing it, else None.
        """Get the prompt string for a specific column."""
        if isinstance(prompts, str):
            return prompts
        elif isinstance(prompts, dict) and column_name in prompts:
            return prompts[column_name]
        return None

    def _get_task_for_column(
        self, column_name: str, column_position: int, router_mapping: dict[str, str]
    ) -> str | None:
        # [FUNCTION DOC]
        # HELPER: return the task for a column from the router mapping (None if unmapped). Subclasses override this to add
        # position-based defaults (column_position is unused here).
        """Resolve the task a column is preprocessed with. Subclasses may add positional defaults."""
        return router_mapping.get(column_name)

    def _get_max_length_for_task(self, task: str | None) -> int | None:
        # [FUNCTION DOC]
        # HELPER: return the max-length override for a column: max_length itself when it is an int, the task's entry when it is a
        # dict and task is not None (None if missing), else None.
        """Resolve the training max-length override for a column: an int applies to every column,
        a dict is keyed by task."""
        if isinstance(self.max_length, int):
            return self.max_length
        if isinstance(self.max_length, dict) and task is not None:
            return self.max_length.get(task)
        return None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Collate a list of row dicts into a batch. Returns an empty dict for no rows. Column names come from the first row;
        # "dataset_name" is removed from them and copied from the first row into the batch. Column order is checked once per
        # distinct column tuple. The first valid label column present becomes batch "label" as a tensor of all rows' values and is
        # removed. Resolves router mapping and prompts, logs a one-time warning when columns look like IDs ("id" or ending in
        # _id/_ids/_idx). Then, for each remaining column in order, determines task, prompt and max-length override, calls
        # preprocess_fn on the column values (passing max_length only when set), stores each output under "<column>_<key>", and
        # records "<column>_task" when a task is set. Returns the batch dict. Raises ValueError via prompt resolution for nested
        # prompts without a dataset name.
        if not features:
            return {}

        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        router_mapping = self._resolve_router_mapping(batch)
        prompts = self._resolve_prompts(batch)

        id_like = [col for col in column_names if col == "id" or col.endswith(("_id", "_ids", "_idx"))]
        if id_like:
            logger.warning_once(
                f"Column(s) {id_like} look like ID columns, but they will be tokenized and trained on as "
                "text. If they hold IDs, resolve them to their values first, e.g. with "
                "`sentence_transformers.util.resolve_ids`, or remove them from the dataset."
            )

        for column_position, column_name in enumerate(column_names):
            task = self._get_task_for_column(column_name, column_position, router_mapping)
            prompt = self._get_prompt_for_column(prompts, column_name)
            inputs = [row[column_name] for row in features]
            max_length = self._get_max_length_for_task(task)
            # Only forward the override when set, so preprocess_fns without **kwargs keep working.
            if max_length is not None:
                preprocessed = self.preprocess_fn(inputs, prompt=prompt, task=task, max_length=max_length)
            else:
                preprocessed = self.preprocess_fn(inputs, prompt=prompt, task=task)
            for key, value in preprocessed.items():
                batch[f"{column_name}_{key}"] = value
            # Stamp the resolved task so losses can re-run the model under the task each column was
            # tokenized with (e.g. router_mapping overrides or a subclass positional default).
            if task is not None:
                batch[f"{column_name}_task"] = task

        return batch

    def maybe_warn_about_column_order(self, column_names: list[str]) -> None:
        # [FUNCTION DOC]
        # Warn when well-known column names (anchor/positive/negative, question/answer, query/response,
        # hypothesis/entailment/contradiction) are not at their usual positions, suggesting a select_columns call with the
        # conventional order; warns at most once per call, then records the column tuple in _warned_columns so it is not checked
        # again. Returns None. Side effects: logging and mutation of _warned_columns.
        """Warn the user if the columns are likely not in the expected order."""
        # A mapping from common column names to the expected index in the dataset
        column_name_to_expected_idx = {
            "anchor": 0,
            "positive": 1,
            "negative": 2,
            "question": 0,
            "answer": 1,
            "query": 0,
            "response": 1,
            "hypothesis": 0,
            "entailment": 1,
            "contradiction": 2,
        }
        for column_name, expected_idx in column_name_to_expected_idx.items():
            if column_name in column_names and column_names.index(column_name) != expected_idx:
                if column_name in ("anchor", "positive", "negative"):
                    proposed_fix_columns = ["anchor", "positive", "negative"]
                elif column_name in ("question", "answer"):
                    proposed_fix_columns = ["question", "answer"]
                elif column_name in ("query", "response"):
                    proposed_fix_columns = ["query", "response"]
                elif column_name in ("hypothesis", "entailment", "contradiction"):
                    proposed_fix_columns = ["hypothesis", "entailment", "contradiction"]

                logger.warning(
                    f"Column {column_name!r} is at index {column_names.index(column_name)}, whereas "
                    f"a column with this name is usually expected at index {expected_idx}. Note that the column "
                    "order can be important for some losses, e.g. MultipleNegativesRankingLoss will always "
                    "consider the first column as the anchor and the second as the positive, regardless of "
                    "the dataset column names. Consider renaming the columns to match the expected order, e.g.:\n"
                    f"dataset = dataset.select_columns({proposed_fix_columns})"
                )
                # We only need to warn once per list of column names to prevent spamming the user
                break

        self._warned_columns.add(tuple(column_names))
