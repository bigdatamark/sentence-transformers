# [MODULE DOC]
# Dataset utilities for training data that references texts by ID. resolve_ids builds a picklable
# batched transform (for datasets' set_transform or map) that replaces ID columns, such as query_id
# or a list of document_ids, with the values looked up from separate lookup datasets, optionally
# expanding ID lists into numbered columns, and passes label columns through.
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from transformers.utils import logging as transformers_logging

if TYPE_CHECKING:
    from datasets import Dataset

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)

# Keep in sync with base.data_collator.DEFAULT_LABEL_COLUMNS (importing it here would be circular).
DEFAULT_LABEL_COLUMNS: tuple[str, ...] = ("label", "labels", "score", "scores")


def _infer_id_col(dataset: Dataset, input_col: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the single column of the lookup dataset named "id" or ending in "_id"; raises
    # ValueError naming the candidates if there is not exactly one.
    candidates = [col for col in dataset.column_names if col == "id" or col.endswith("_id")]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot infer the ID column of lookups[{input_col!r}]: "
            f"expected exactly one column named 'id' or ending in '_id', found {candidates or 'none'} "
            f"among {sorted(dataset.column_names)}. Pass an explicit (dataset, id_col) tuple."
        )
    return candidates[0]


def _infer_value_col(dataset: Dataset, id_col: str, input_col: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the single lookup dataset column other than the ID column; raises ValueError if
    # there is not exactly one.
    remaining = [col for col in dataset.column_names if col != id_col]
    if len(remaining) != 1:
        raise ValueError(
            f"Cannot infer the value column of lookups[{input_col!r}]: "
            f"expected exactly one column besides {id_col!r}, found {remaining or 'none'}. "
            f"Pass an explicit (dataset, id_col, value_col) tuple."
        )
    return remaining[0]


def _output_col(input_col: str, plural: bool = False) -> str:
    # [FUNCTION DOC]
    # HELPER: Derives an output column name from an input ID column name. In normal mode strips a
    # trailing "_ids", "_id" or "_idx". In plural mode turns "y_ids" into "ies", "_ids" into "s", and
    # strips "_id" or "_idx". Names without these suffixes are returned unchanged.
    # Strip the ID suffix ("query_id" -> "query"), or pluralise in plural mode ("document_ids" -> "documents").
    suffixes = {"y_ids": "ies", "_ids": "s", "_id": "", "_idx": ""} if plural else {"_ids": "", "_id": "", "_idx": ""}
    for suffix, replacement in suffixes.items():
        if input_col.endswith(suffix):
            return input_col.removesuffix(suffix) + replacement
    return input_col


def _is_string_encoded_list(value: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns true if the value is a string whose first non-whitespace character is an opening
    # square bracket (a list stored as text).
    return isinstance(value, str) and value.lstrip().startswith("[")


def _parse_string_encoded_lists(column: list, col_name: str) -> list:
    # [FUNCTION DOC]
    # HELPER: Parses each row of a column holding lists stored as text using a safe literal evaluator
    # and returns the rows as lists. Raises ValueError, naming the column and showing the start of the
    # row, if a row fails to parse or does not parse to a list or tuple.
    # Some KD datasets (e.g. lightonai/ms-marco-en-bge-gemma) store ID and score lists as strings.
    parsed = []
    for row in column:
        try:
            value = ast.literal_eval(row)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"Column {col_name!r} looks like it holds string-encoded lists, but a row failed to "
                f"parse ({e}): {row[:100]!r}"
            ) from None
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Column {col_name!r} looks like it holds string-encoded lists, but a row parsed to "
                f"{type(value).__name__} instead of a list: {row[:100]!r}"
            )
        parsed.append(list(value))
    return parsed


class _SortedIdIndex:
    # [CLASS DOC]
    # HELPER: Memory-efficient ID-to-row-position lookup over sorted numpy arrays, used for numeric or
    # narrow string IDs. Attributes: keys (sorted unique IDs) and positions (their row positions).
    """``id -> row position`` lookup backed by sorted numpy arrays.

    A python dict retains a hash table plus every key object, and rebuilds that table in each
    DataLoader worker on spawn-start platforms. Sorted arrays hold the same lookup in a fraction of
    the memory and pickle as flat buffers, so they also unpickle far faster. Lookups are a
    searchsorted rather than a hash, so they take 6x to 16x as long per batch as the dict, which is
    immaterial next to the forward pass. Only used for homogeneous numeric ids and for strings
    narrow enough that the fixed-width array stays under the dict's footprint, since numpy sizes
    such an array by its longest entry.
    """

    __slots__ = ("keys", "positions")

    def __init__(self, keys: np.ndarray) -> None:
        # [FUNCTION DOC]
        # Stable-sorts the keys, and for runs of equal keys keeps only the last occurrence (matching dict
        # semantics where a later duplicate wins), storing the sorted unique keys and their original
        # positions.
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        # Dict semantics: a duplicated ID resolves to its last occurrence. Stable sort keeps
        # duplicates in original order, so the last of each equal run is the one to keep.
        keep = np.ones(len(sorted_keys), dtype=bool)
        if len(sorted_keys) > 1:
            keep[:-1] = sorted_keys[1:] != sorted_keys[:-1]
        self.keys = sorted_keys[keep]
        self.positions = order[keep]

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of unique keys.
        return len(self.keys)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns the row positions for an array of IDs via binary search. Raises KeyError with the first
        # missing ID (as a Python number for numeric IDs, else a string), including when the index is empty.
        """Return the row positions for ``ids``, raising ``KeyError(missing_id)`` on a miss."""
        if len(self.keys) == 0:
            missing = ids[0]
        else:
            insertion_points = np.searchsorted(self.keys, ids)
            clipped = np.minimum(insertion_points, len(self.keys) - 1)
            found = self.keys[clipped] == ids
            if found.all():
                return self.positions[clipped]
            missing = ids[int(np.argmin(found))]
        raise KeyError(missing.item() if ids.dtype.kind in "iuf" else str(missing))


def _take(ids: list, value_view, index: _SortedIdIndex | dict, value_col: str, input_col: str) -> list:
    # [FUNCTION DOC]
    # HELPER: Looks up the values for a list of IDs: returns an empty list for no IDs; maps IDs to row
    # positions with a _SortedIdIndex (raising KeyError for the first ID if the ID array type does not
    # match the key type) or a dict, then returns the value column of those rows from the value view.
    # Converts a TypeError (unhashable IDs, typically from a formatted dataset) into a TypeError with
    # advice to remove the format, and a KeyError into a KeyError naming the missing ID and input
    # column (with a hint when it looks like a list stored as text).
    if not ids:
        return []
    try:
        if isinstance(index, _SortedIdIndex):
            queries = np.asarray(ids)
            if queries.dtype.kind == "O" or queries.dtype.kind != index.keys.dtype.kind:
                # Mixed / mistyped ids can't hit the typed sorted keys: every id would "miss", so
                # report the first one as not found, matching the dict path's behaviour.
                raise KeyError(ids[0])
            rows = index.lookup(queries).tolist()
        else:
            rows = [index[key] for key in ids]
    except TypeError:
        raise TypeError(
            f"IDs from input column {input_col!r} are not hashable, which typically means the train "
            "dataset carries a format (e.g. with_format('numpy')). Call dataset.with_format(None) "
            "before applying the transform."
        ) from None
    except KeyError as e:
        missing = e.args[0]
        hint = ""
        if isinstance(missing, str) and missing.lstrip().startswith("["):
            hint = " The ID looks like a string-encoded list. Convert the column to native lists first."
        raise KeyError(
            f"ID {missing!r} from input column {input_col!r} was not found in its lookup dataset.{hint}"
        ) from None
    return value_view[rows][value_col]


def _add_output_column(resolved: dict[str, Any], name: str, values: list) -> None:
    # [FUNCTION DOC]
    # HELPER: Adds a named column of values to the output dict (mutating it); raises ValueError if the
    # name already exists.
    # Output names depend on the data (list-ness, list length), so collisions are caught here.
    if name in resolved:
        raise ValueError(
            f"Output column {name!r} is produced more than once (e.g. two lookups entries resolving "
            "to the same name, or an expanded list column colliding with another entry). Rename the "
            "input columns to disambiguate."
        )
    resolved[name] = values


def _resolve_ids_batch(
    batch: dict[str, Any],
    resolutions: dict[str, tuple[Any, _SortedIdIndex | dict, str, str, str]],
    keep_columns: tuple[str, ...],
    max_list_length: int | None,
    output_format: Literal["columns", "lists"],
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: The batched transform. Raises TypeError if the batch is not a mapping. For each configured
    # input column: raises TypeError if the column is not a list (formatted dataset); parses lists stored
    # as text; then, if rows are lists and output_format is "columns", truncates rows to
    # max_list_length, requires all rows to have the same non-zero length (else ValueError), looks up
    # all IDs at once and adds one column per position named with the stripped name plus an underscore
    # and the one-based position; if rows are lists and output_format is "lists", truncates rows, looks
    # up all IDs and adds a nested list column under the plural name; otherwise looks up the scalar IDs
    # and adds them under the stripped name. Then passes through each keep column present in the batch
    # (parsing lists stored as text and truncating per-row sequences to max_list_length). Raises
    # ValueError on output name collisions. Returns the new batch dict containing only resolved and
    # kept columns.
    # The batched transform built by resolve_ids, bound via functools.partial. Module-level rather
    # than a closure so that datasets carrying it via set_transform stay picklable, which DataLoader
    # workers require on spawn-start platforms.
    if not isinstance(batch, Mapping):
        raise TypeError(
            f"resolve_ids expects batches as plain column dicts, got {type(batch).__name__}: the train "
            "dataset carries a format (e.g. with_format('pandas')). Call dataset.with_format(None) "
            "before applying the transform."
        )
    resolved: dict[str, Any] = {}

    for input_col, (value_view, index, value_col, stripped_name, plural_name) in resolutions.items():
        column = batch[input_col]
        if not isinstance(column, list):
            raise TypeError(
                f"Input column {input_col!r} holds a {type(column).__name__} rather than a list, which "
                "means the train dataset carries a format (e.g. with_format('numpy')). Call "
                "dataset.with_format(None) before applying the transform."
            )
        if len(column) > 0 and _is_string_encoded_list(column[0]):
            column = _parse_string_encoded_lists(column, input_col)
        # Detect list-per-row vs single-per-row from the actual data. Uniform within a column.
        is_list = len(column) > 0 and isinstance(column[0], (list, tuple))
        if is_list and output_format == "columns":
            if max_list_length is not None:
                column = [row[:max_list_length] for row in column]
            lengths = {len(row) for row in column}
            if len(lengths) != 1:
                raise ValueError(
                    f"Rows of input column {input_col!r} hold lists of differing lengths "
                    f"{sorted(lengths)}. List columns expand into one output column per position, "
                    "which requires the same length on every row. Set max_list_length to cap all "
                    'rows to a common length, or output_format="lists" to keep nested lists '
                    "(the CrossEncoder listwise shape, which allows ragged rows)."
                )
            num_positions = lengths.pop()
            if num_positions == 0:
                raise ValueError(
                    f"Rows of input column {input_col!r} hold empty ID lists, so there are no "
                    "positions to expand into columns. Filter such rows out of the dataset."
                )
            flat_values = _take([k for row in column for k in row], value_view, index, value_col, input_col)
            for position in range(num_positions):
                # flat_values is row-major, so a stride slice gathers one position across all rows.
                _add_output_column(resolved, f"{stripped_name}_{position + 1}", flat_values[position::num_positions])
        elif is_list:  # So, output_format == "lists"
            if max_list_length is not None:
                column = [row[:max_list_length] for row in column]
            flat_values = _take([k for row in column for k in row], value_view, index, value_col, input_col)
            nested, offset = [], 0
            for row in column:
                nested.append(flat_values[offset : offset + len(row)])
                offset += len(row)
            _add_output_column(resolved, plural_name, nested)
        else:
            _add_output_column(resolved, stripped_name, _take(list(column), value_view, index, value_col, input_col))

    # Pass through the kept columns (labels by default). Truncate to max_list_length when the column is
    # a list-per-row, keeping it aligned with a resolved ID list. Leave scalar values alone.
    for keep_col in keep_columns:
        if keep_col not in batch:
            continue
        values = batch[keep_col]
        if len(values) > 0 and _is_string_encoded_list(values[0]):
            values = _parse_string_encoded_lists(values, keep_col)
        # Sized but not a string covers lists as well as the arrays a formatted label column holds.
        is_row_sequence = len(values) > 0 and not isinstance(values[0], (str, bytes)) and hasattr(values[0], "__len__")
        if max_list_length is not None and is_row_sequence:
            values = [row[:max_list_length] for row in values]
        _add_output_column(resolved, keep_col, values)

    return resolved


def resolve_ids(
    lookups: dict[str, Dataset | tuple[Dataset, str] | tuple[Dataset, str, str]],
    keep_columns: list[str] | tuple[str, ...] = DEFAULT_LABEL_COLUMNS,
    max_list_length: int | None = None,
    output_format: Literal["columns", "lists"] = "columns",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    # [FUNCTION DOC]
    # Builds and returns the ID-resolving batched transform. Raises ImportError without the datasets
    # library, TypeError if keep_columns is a string, ValueError if max_list_length is below 1 or
    # output_format is not "columns" or "lists". For each lookups entry (a Dataset, or a tuple of
    # dataset and ID column, optionally with a value column; other tuple lengths raise ValueError),
    # raises TypeError for a DatasetDict or non-Dataset, removes any format from the dataset, infers or
    # validates the ID and value columns (ValueError if missing), and builds an ID index shared by
    # entries with the same original dataset object and ID column. The index reads IDs in batches of
    # 50000; it is a _SortedIdIndex for one-dimensional numeric or string IDs of at most 128 bytes,
    # else a dict. Logs a one-time warning if duplicate IDs exist (the last occurrence wins). Stores a
    # single-column value view, the index, value column and stripped/plural output names per input
    # column. Returns a partial application of the module-level batch function with these settings, so
    # it stays picklable.
    """Build a batched transform that resolves ID columns to their values via a join against lookup
    datasets. Useful for IR datasets that store query / document IDs alongside separate text (or
    image) datasets, e.g. `lightonai/ms-marco-en-bge <https://huggingface.co/datasets/lightonai/ms-marco-en-bge>`_.

    An input column may hold one ID per row (e.g. ``positive_id: str``) or a list of IDs per row
    (e.g. ``document_ids: list[str]``), detected from the row values. An ID that is missing from its
    lookup dataset raises a :class:`KeyError` when the batch is read.

    Pass the returned callable to :meth:`~datasets.Dataset.set_transform` (lazy, no caching) or
    :meth:`~datasets.Dataset.map` with ``batched=True`` (eager, cached). With ``map``, also pass
    ``remove_columns=list(lookups)`` so the raw ID columns are dropped from the result: unlike
    ``set_transform``, ``map`` merges the transform output with the untouched input columns. ``map``
    also hands the transform batches in the dataset's current format, so remove any format first
    with ``with_format(None)``.

    Args:
        lookups: ``{input_col: lookup}`` for every input column to resolve, where ``lookup`` is:

            - a :class:`datasets.Dataset`: the ID column (the single column named ``id`` or ending
              in ``_id``) and the value column (the single remaining column) are inferred. Ambiguity
              raises at build time.
            - ``(dataset, id_col)``: explicit ID column, value column inferred.
            - ``(dataset, id_col, value_col)``: fully explicit.

            All other input columns are dropped (except ``keep_columns``), as the losses read batch
            columns positionally. Insertion order sets the output column order, so list the anchor
            (query) column first.
        keep_columns: Input columns to pass through unresolved. Defaults to the label columns the
            data collators recognize by default (``label``, ``labels``, ``score``, ``scores``).
            Mirror your collator's ``valid_label_columns`` if you customized it. Absent columns
            are ignored.
        max_list_length: Truncate every list-per-row column (ID lists and list-valued kept columns) to the
            first ``max_list_length`` entries. Must be a positive integer or ``None`` (default, keep all).
            Note that this takes the first N as stored, not top-N by teacher score.
        output_format: ``"columns"`` (default) expands each list column into numbered columns
            (``document_1``, ..., ``document_N``), requiring a uniform list length per row.
            ``"lists"`` keeps list columns nested under a plural name (``documents: list``),
            allowing ragged rows. Use ``"lists"`` for CrossEncoder listwise training.

    Returns:
        A picklable batched transform mapping a batch dict of ID columns to a batch dict of
        resolved columns, for :meth:`~datasets.Dataset.set_transform` or
        :meth:`~datasets.Dataset.map` with ``batched=True``.

    Example (KD, LightOn ms-marco-en-bge)::

        from datasets import load_dataset
        from sentence_transformers.util import resolve_ids

        train = load_dataset("lightonai/ms-marco-en-bge", "train", split="train")
        queries = load_dataset("lightonai/ms-marco-en-bge", "queries", split="train")
        documents = load_dataset("lightonai/ms-marco-en-bge", "documents", split="train")

        train.set_transform(resolve_ids({
            "query_id": queries,
            "document_ids": documents,
        }, max_list_length=32))
        # -> rows of {"query": str, "document_1": str, ..., "document_32": str, "scores": list[float]}

    Example (triplet with IDs)::

        train.set_transform(resolve_ids({
            "query_id": queries,
            "positive_id": documents,
            "negative_id": documents,
        }))
        # -> rows of {"query": str, "positive": str, "negative": str}

    Example (CrossEncoder listwise, nested lists)::

        train.set_transform(resolve_ids({
            "query_id": queries,
            "document_ids": documents,
        }, output_format="lists"))
        # -> rows of {"query": str, "documents": list[str], "scores": list[float]}

    Example (streaming)::

        from datasets import Features, Sequence, Value

        # Only the train split streams. The lookup datasets are random-access joins, so they
        # must stay regular (materialized) Datasets.
        train = load_dataset("lightonai/ms-marco-en-bge", "train", split="train", streaming=True)
        train = train.map(
            resolve_ids({"query_id": queries, "document_ids": documents}, max_list_length=32),
            batched=True,
            remove_columns=["query_id", "document_ids"],
            features=Features({
                "query": Value("string"),
                **{f"document_{i}": Value("string") for i in range(1, 33)},
                "scores": Sequence(Value("float32")),
            }),
        )
        # Streaming map cannot infer the output schema, and the trainers require it, so pass
        # ``features=`` explicitly.
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as e:
        raise ImportError(
            "resolve_ids requires the `datasets` library. Install it with `pip install datasets`."
        ) from e

    if isinstance(keep_columns, str):
        raise TypeError(f"keep_columns must be a list or tuple of column names, got the string {keep_columns!r}.")
    if max_list_length is not None and max_list_length < 1:
        raise ValueError(f"max_list_length must be a positive integer or None, got {max_list_length}.")
    if output_format not in ("columns", "lists"):
        raise ValueError(f'output_format must be "columns" or "lists", got {output_format!r}.')

    # Per input column: (value view, id -> row index, value column, stripped output name, plural
    # output name). Output names are precomputed here so the per-batch transform only formats the
    # position suffix. ID indices are shared between entries that use the same dataset and ID
    # column (e.g. positive_id and negative_id).
    resolutions: dict[str, tuple[Any, _SortedIdIndex | dict, str, str, str]] = {}
    indices: dict[tuple[int, str], _SortedIdIndex | dict] = {}
    for input_col, lookup_spec in lookups.items():
        if isinstance(lookup_spec, tuple):
            spec = list(lookup_spec)
            if len(spec) not in (2, 3):
                raise ValueError(
                    f"lookups[{input_col!r}] must be a Dataset, (dataset, id_col), or "
                    f"(dataset, id_col, value_col), got a tuple of length {len(spec)}."
                )
            dataset = spec[0]
            id_col = spec[1]
            value_col = spec[2] if len(spec) == 3 else None
        else:
            dataset, id_col, value_col = lookup_spec, None, None

        if isinstance(dataset, DatasetDict):
            raise TypeError(
                f"lookups[{input_col!r}] holds a `DatasetDict`, expected a `Dataset`. Pass a specific "
                "split (e.g. `dataset_dict['train']`), or load with `load_dataset(..., split='...')`."
            )
        if not isinstance(dataset, Dataset):
            raise TypeError(f"lookups[{input_col!r}] must hold a `Dataset`, got {type(dataset).__name__}.")
        # Plain python values for index keys and output values, also when the caller formatted the
        # lookup (e.g. with_format("torch") would produce unhashable 0-d tensor keys). Key the shared
        # index on the ORIGINAL object, since with_format returns a new one per entry.
        original_dataset_id = id(dataset)
        dataset = dataset.with_format(None)

        if id_col is None:
            id_col = _infer_id_col(dataset, input_col)
        elif id_col not in dataset.column_names:
            raise ValueError(
                f"lookups[{input_col!r}]: ID column {id_col!r} is not a column of the lookup dataset "
                f"(columns: {sorted(dataset.column_names)})."
            )
        if value_col is None:
            value_col = _infer_value_col(dataset, id_col, input_col)
        elif value_col not in dataset.column_names:
            raise ValueError(
                f"lookups[{input_col!r}]: value column {value_col!r} is not a column of the lookup "
                f"dataset (columns: {sorted(dataset.column_names)})."
            )

        index_key = (original_dataset_id, id_col)
        if index_key not in indices:
            # Batched iteration is ~10x faster than materializing the column via dataset[id_col],
            # and unlike raw Arrow access it respects an indices mapping (select / shuffle / filter).
            id_values: list = []
            for id_batch in dataset.select_columns([id_col]).iter(batch_size=50_000):
                id_values.extend(id_batch[id_col])
            keys = np.asarray(id_values)
            # numpy pads a string array to its longest id, so past 128 bytes it costs more than the dict.
            if keys.ndim == 1 and keys.dtype.kind in "iufUS" and keys.itemsize <= 128:
                index = _SortedIdIndex(keys)
            else:
                # Mixed-type, wide, or otherwise non-sortable ids: the dict handles any hashable key.
                index = {value: position for position, value in enumerate(id_values)}
            if len(index) != dataset.num_rows:
                logger.warning_once(
                    f"lookups[{input_col!r}] contains duplicate {id_col!r} values "
                    f"({dataset.num_rows - len(index)} rows shadowed). Each duplicated ID resolves to its "
                    "last occurrence."
                )
            indices[index_key] = index
        # A single-column view, so that batched row access only decodes the value column.
        resolutions[input_col] = (
            dataset.select_columns([value_col]),
            indices[index_key],
            value_col,
            _output_col(input_col),
            _output_col(input_col, plural=True),
        )

    return partial(
        _resolve_ids_batch,
        resolutions=resolutions,
        keep_columns=tuple(keep_columns),
        max_list_length=max_list_length,
        output_format=output_format,
    )
