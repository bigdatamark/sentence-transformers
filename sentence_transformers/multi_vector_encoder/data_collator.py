# [MODULE DOC]
# HELPER: Data collator for MultiVectorEncoder training. Subclasses the shared BaseDataCollator (which
# tokenizes dataset columns into feature batches and may route columns to router tasks) and only changes
# how a column's router task is chosen when no explicit mapping exists.
from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.base.data_collator import BaseDataCollator


@dataclass
class MultiVectorEncoderDataCollator(BaseDataCollator):
    # [CLASS DOC]
    # HELPER: Dataclass collator identical to BaseDataCollator except for default task assignment: a column
    # with no entry in the router mapping gets task "query" if it is the first column (position 0) and
    # "document" otherwise. Column names are never inspected for defaults.
    """Data collator for :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` training.

    Differs from :class:`~sentence_transformers.base.data_collator.BaseDataCollator` only in the
    default task assignment: when ``router_mapping`` does not specify a task for a column, the first
    column defaults to ``"query"`` and the rest to ``"document"``. This matches the losses, which
    assign positionally (column 0 = query). Column names are not consulted. Use ``router_mapping``
    to override per-column.
    """

    def _get_task_for_column(self, column_name: str, column_position: int, router_mapping: dict[str, str]) -> str:
        # [FUNCTION DOC]
        # HELPER: Return the router task for one column. Inputs: the column name, its zero-based position, and a
        # mapping from column name to task. Returns the mapped task if present; otherwise "query" for position 0
        # and "document" for any other position. Pure, no side effects.
        task = router_mapping.get(column_name)
        if task is None:
            task = "query" if column_position == 0 else "document"
        return task
