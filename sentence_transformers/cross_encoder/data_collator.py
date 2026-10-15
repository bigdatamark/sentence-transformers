# [MODULE DOC]
# Data collator for CrossEncoder training. Unlike embedding-model collators it does not tokenize: it
# gathers raw text columns, the label, and a per-batch prompt and router task, leaving tokenization to the
# losses (which pair the columns themselves).
from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import torch

from sentence_transformers.base.data_collator import BaseDataCollator


@dataclass
class CrossEncoderDataCollator(BaseDataCollator):
    # [CLASS DOC]
    # Dataclass collator inheriting fields from BaseDataCollator (notably the prompts setting, the router
    # mapping, and the list of recognized label column names). Column order matters, since losses treat
    # columns positionally (e.g. first column is the query/anchor). Prompts and router mapping may be a single
    # string or a mapping from dataset name to string, never per-column.
    """Collator for a CrossEncoder model.
    This returns raw text columns (not tokenized), as CrossEncoder loss functions handle
    preprocessing. A single prompt and task (resolved per dataset if needed) are passed
    through in the batch so that losses can forward them to ``model.preprocess``.

    It is important that the columns are in the expected order. For example, if your dataset has columns
    "answer", "question" in that order, then the MultipleNegativesRankingLoss will consider
    "answer" as the anchor and "question" as the positive, and it will (unexpectedly) optimize for
    "given the answer, what is the question?".
    """

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate configuration after dataclass init. Raises ValueError if the prompts mapping or router mapping
        # contains a nested mapping as any value (which would mean per-column configuration). No other effects.
        # For cross-encoders, inputs from multiple columns are combined into pairs, so
        # per-column prompts and per-column router mappings don't make sense.
        # Only a single value or a per-dataset mapping is allowed for each.
        if isinstance(self.prompts, dict):
            for value in self.prompts.values():
                if isinstance(value, dict):
                    raise ValueError(
                        "CrossEncoder prompts cannot be per-column. Inputs from multiple columns are "
                        "combined into pairs, so only a single prompt string or a per-dataset mapping "
                        "of prompt strings is supported. For example: prompts='Search: ' or "
                        "prompts={'dataset_a': 'Search: ', 'dataset_b': 'Retrieve: '}"
                    )
        if isinstance(self.router_mapping, dict):
            for value in self.router_mapping.values():
                if isinstance(value, dict):
                    raise ValueError(
                        "CrossEncoder router_mapping cannot be per-column. Inputs from multiple columns "
                        "are combined into pairs, so only a per-dataset mapping of task strings is "
                        "supported. For example: router_mapping={'dataset_a': 'rerank', 'dataset_b': 'rerank'}"
                    )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Collate a list of row dicts (all sharing the keys of the first row) into a batch dict. Steps: if a
        # "dataset_name" column exists, remove it from the columns and store the first row's value in the batch.
        # Find the first recognized label column present; if its values are collections, store a list of per-row
        # tensors under "label", else a single tensor of all values; remove it from the columns. Resolve one prompt
        # and one task for the batch (see the resolver) and store them under "prompt" / "task" only when non-empty.
        # Every remaining column becomes a list of raw values under its own name. Returns the batch dict.
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                # If the label column is a list/tuple/collection, we create a list of tensors
                if isinstance(features[0][label_column], Collection):
                    batch["label"] = [torch.tensor(row[label_column]) for row in features]
                else:
                    # Otherwise, if it's e.g. single values, we create a tensor
                    batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        # Resolve prompt and task to single values for this batch
        prompt = self._resolve_scalar(self.prompts, batch)
        if prompt:
            batch["prompt"] = prompt
        task = self._resolve_scalar(self.router_mapping, batch)
        if task:
            batch["task"] = task

        for column_name in column_names:
            batch[column_name] = [row[column_name] for row in features]

        return batch

    @staticmethod
    def _resolve_scalar(mapping: str | dict[str, str] | None, batch: dict[str, Any]) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Resolve a prompt or task setting to one string for a batch. Returns None for an empty/absent
        # setting, the string itself if it is a string, the value for the batch's dataset name if the setting is a
        # mapping and the batch carries a dataset name (None if that name is missing from the mapping), and None
        # otherwise.
        """Resolve a string-or-per-dataset mapping to a single string for this batch."""
        if not mapping:
            return None
        if isinstance(mapping, str):
            return mapping
        if isinstance(mapping, dict) and "dataset_name" in batch:
            return mapping.get(batch["dataset_name"], None)
        return None
