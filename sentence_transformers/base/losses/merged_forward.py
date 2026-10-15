# [MODULE DOC]
# Utilities that let losses embed several input columns (e.g. anchor, positive, negatives) in one merged forward pass
# instead of one forward per column, for speed. Provides: a context manager that disables merging (via a context
# variable), merge_feature_batches (pads and concatenates homogeneous per-column feature dicts row-wise, column-major,
# or refuses by returning None), chunked forward helpers for multi-vector (token-level) models that split a batch into
# row chunks and re-pad the outputs, and embed_columns / embed_columns_padded which try merging and fall back to
# per-column forwards. Results are identical either way, up to dropout sampling. Logs a one-time warning when merging
# is refused for a reason the user could fix.
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.utils import logging

from sentence_transformers.base.losses.gradcache import _create_minibatch, _get_batch_size, _minibatch_ranges
from sentence_transformers.util import cat_padded_token_embeddings

logger = logging.get_logger(__name__)

_MERGING_DISABLED: ContextVar[bool] = ContextVar("merging_disabled", default=False)


def _separate_forwards(reason: str) -> None:
    # [FUNCTION DOC]
    # HELPER: logs a one-time warning that columns are embedded in separate forward passes because of the given reason,
    # then returns None so callers can return its result directly as the "refused" value.
    """Note why the columns are embedded separately, then return None so the caller falls back.

    Speed only: the per-column path produces the same loss and gradients. Worth surfacing because
    otherwise a setup that never qualifies looks identical to one that always does. Reserved for
    reasons with a call-site fix: guards that a model architecture can never pass refuse silently.
    """
    logger.warning_once(
        f"Note: embedding each input column in its own forward pass, because {reason}. Columns that "
        "preprocess identically are combined into one forward pass instead, which is faster. The loss "
        "and gradients are the same either way, up to dropout sampling."
    )
    return None


@contextmanager
def column_merging_disabled() -> Iterator[None]:
    # [FUNCTION DOC]
    # Context manager (generator based) that sets the module-level "merging disabled" context variable to True for the
    # duration of the block and resets it afterwards (also on exceptions). While active, merge_feature_batches always
    # returns None silently. Intended for wrapper losses that pass state between calls through the same feature dicts.
    """Force :func:`merge_feature_batches` to refuse for the duration of the block.

    For wrapper losses that call an inner loss several times with the *same* feature dicts and pass
    state between those calls through them, e.g.
    :class:`~sentence_transformers.sentence_transformer.losses.AdaptiveLayerLoss`, whose ``TransformerDecorator`` writes
    ``all_layer_embeddings`` into each dict on the first call and reads it back on the rest. Merging
    builds fresh dicts, so that hand-off would land on a dict the next call never sees.
    """
    token = _MERGING_DISABLED.set(True)
    try:
        yield
    finally:
        _MERGING_DISABLED.reset(token)


def merge_feature_batches(features_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # Merges a list of per-column feature dicts into one dict whose tensors hold all columns' rows concatenated in column
    # order, or returns None if merging is not possible. Returns None silently when merging is disabled, when the batch is
    # flattened ("cu_seq_lens_q" present), or when neither "input_ids" nor "attention_mask" is a tensor of at least 2 dims.
    # Returns None with a warning when columns have different key sets, or for any key where: a tensor value is missing or
    # has a different number of dims in some column; some value is 0-dimensional or its first dimension differs from the
    # batch size; dims beyond the second differ; or the second (token) dimension differs and the tensor is floating point or
    # complex (only integer/bool tensors are right-padded with zeros to the maximum width). Non-tensor values and the
    # "prompt_length" key are metadata: they must compare equal across columns (tensors by exact equality) and the first
    # column's value is kept; incomparable values (comparison raising RuntimeError or ValueError) or unequal values cause a
    # warned refusal. Input: non-empty list of dicts. Output: merged dict or None.
    """Pad and concatenate homogeneous per-column feature batches into one flat batch.

    Running ``N`` columns through the model one by one means ``N`` forwards of ``batch_size`` rows
    each, which leaves the GPU mostly idle at typical training sizes. Merging restores a single
    ``(batch_size * N)``-row forward, ordered column-major (all of column 1, then all of column 2,
    and so on). Returns None when the columns are not homogeneous (mismatched keys, shapes, or
    metadata), when the batch is flattened for FA2 unpadding, or when a tensor's row axis is not the
    batch axis (flattened VLM grids), in which case the caller falls back to per-column forwards.

    The merged batch pads every row to the cross-column max width, so one long outlier document
    widens all rows (a per-column fallback only widens its own column). Combine with
    :func:`chunked_padded_forward` (which re-trims per chunk) or ``TrainingArguments.max_length``
    to bound this.
    """
    # Deliberate, from a wrapper loss that needs the per-column dicts, so nothing to report.
    if _MERGING_DISABLED.get():
        return None
    first = features_list[0]
    if any(set(features) != set(first) for features in features_list[1:]):
        return _separate_forwards("the columns preprocess to different sets of features")
    # A flattened batch packs its rows into one sequence whose cu_seq_lens_q offsets concatenation would scramble.
    if "cu_seq_lens_q" in first:
        return None
    # No batched text tensor to align rows by, e.g. StaticEmbedding's flattened 1D token stream.
    if not any(isinstance(first.get(key), Tensor) and first[key].ndim >= 2 for key in ("input_ids", "attention_mask")):
        return None
    rows = _get_batch_size(first)
    merged: dict[str, Any] = {}
    for key, first_value in first.items():
        values = [features[key] for features in features_list]
        # prompt_length is batch metadata in every form Pooling supports (int, shape-1 tensor, or
        # per-row tensor read as one value), so it must compare equal rather than concatenate.
        # The per-row form is what an IterableDataset yields, so it has to merge, not just refuse.
        if isinstance(first_value, Tensor) and key != "prompt_length":
            if any(not isinstance(value, Tensor) or value.ndim != first_value.ndim for value in values):
                return _separate_forwards(f"the columns give {key!r} a different shape each")
            # Tensors whose row axis is not the batch axis (flattened VLM grids and friends) would
            # silently misalign against the concatenated text rows.
            if any(value.ndim == 0 or value.shape[0] != rows for value in values):
                return _separate_forwards(f"{key!r} is not indexed by sample, so its rows cannot be combined")
            if first_value.ndim >= 2:
                if any(value.shape[2:] != first_value.shape[2:] for value in values):
                    return _separate_forwards(f"the columns give {key!r} a different shape each")
                max_length = max(value.shape[1] for value in values)
                if any(value.shape[1] != max_length for value in values):
                    # Only token-style integer/bool tensors can be zero-padded safely: the added
                    # positions carry attention_mask 0 and never reach the scoring mask.
                    if first_value.is_floating_point() or first_value.is_complex():
                        return _separate_forwards(f"{key!r} is a float tensor that would need padding to combine")
                    values = [
                        F.pad(value, [0, 0] * (value.ndim - 2) + [0, max_length - value.shape[1]]) for value in values
                    ]
            merged[key] = torch.cat(values, dim=0)
        else:
            try:
                if isinstance(first_value, Tensor):
                    # Elementwise ``!=`` does not reduce to a plain bool for the per-row shape.
                    unequal = any(
                        not isinstance(value, Tensor) or not torch.equal(value, first_value) for value in values[1:]
                    )
                else:
                    unequal = any(bool(value != first_value) for value in values[1:])
            except (RuntimeError, ValueError):
                # A value that cannot be compared (numpy arrays and the like, reachable through a
                # custom ``preprocess_fn``, or tensors on different devices) cannot be shown equal.
                return _separate_forwards(f"the columns cannot be compared on {key!r}")
            if unequal:
                return _separate_forwards(f"the columns disagree on {key!r}")
            merged[key] = first_value
    return merged


def _chunked_outputs(
    model: nn.Module,
    features: dict[str, Any],
    mini_batch_size: int | None,
    task: str | None,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: runs the model on the features, optionally in row chunks. If mini_batch_size is given, splits the features
    # into consecutive sample ranges of that size (using the GradCache range and slicing helpers, so each chunk is trimmed
    # to its own width); if there is no size or only one range, runs a single forward. Passes task as a keyword argument
    # only when not None. Returns the list of model output dicts, one per chunk. All chunks keep gradients.
    """Run ``model`` over row chunks of ``features`` and return one output dict per chunk.

    A single-element result means the batch ran in one forward, i.e. ``mini_batch_size`` is None or
    is not smaller than the batch. The chunks come from the GradCache mini-batch machinery, so
    padded, FA2-flattened and VLM batches are all sliced correctly, and each chunk is embedded at
    its own width: a single long outlier row only widens its own chunk.

    Unlike GradCache this is a single pass with gradients enabled, so every chunk's activations stay
    live for the backward pass: the win is bounded padding waste, not fewer activations. Reach for a
    cached (GradCache) loss to bound memory by chunk size instead.
    """
    forward_kwargs = {} if task is None else {"task": task}
    ranges = _minibatch_ranges(features, mini_batch_size) if mini_batch_size is not None else None
    if ranges is None or len(ranges) == 1:
        return [model(features, **forward_kwargs)]
    return [model(_create_minibatch(features, begin, end), **forward_kwargs) for begin, end in ranges]


def chunked_padded_forward(
    model: nn.Module,
    features: dict[str, Any],
    task: str | None,
    mini_batch_size: int | None,
) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC]
    # Runs the chunked forward for a multi-vector model and returns the token embeddings and attention mask. With a single
    # chunk they are returned as produced; with several, the per-chunk token embeddings and masks are padded to the widest
    # chunk and concatenated along the batch dimension by a shared utility.
    """Chunked forward for multi-vector models, returning ``(token_embeddings, attention_mask)``.

    The per-chunk outputs are re-padded to the widest chunk before concatenating, since a chunk
    embedded at its own width (or a native-resolution VLM) can emit a shorter token axis than its
    neighbours. See :func:`_chunked_outputs` for the chunking contract.
    """
    outputs = _chunked_outputs(model, features, mini_batch_size, task)
    if len(outputs) == 1:
        return outputs[0]["token_embeddings"], outputs[0]["attention_mask"]
    return cat_padded_token_embeddings(
        [output["token_embeddings"] for output in outputs],
        [output["attention_mask"] for output in outputs],
    )


def embed_columns(
    model: nn.Module, features_list: Iterable[dict[str, Any]], separate_first: bool = True
) -> list[Tensor]:
    # [FUNCTION DOC]
    # Embeds each column and returns one "sentence_embedding" tensor per column, in input order. If separate_first is true
    # (default) and there is at least one column, the first column is embedded on its own and the remaining columns are
    # embedded recursively with separate_first false. Otherwise, with more than one column, tries merge_feature_batches; if
    # it succeeds, runs one forward on the merged batch and splits the embeddings back into equal per-column blocks (by
    # reshaping to column count by rows); if it refuses (or there is at most one column), runs one forward per column, which
    # also lets the model mutate each original feature dict as in the classic per-column pattern.
    """Embed each input column, sharing one merged forward pass across columns when possible.

    Returns one ``sentence_embedding`` tensor per column. When :func:`merge_feature_batches`
    refuses, falls back to one forward per column, which fills each per-column dict exactly like
    the classic ``model(features)`` loss pattern it replaces.

    Only merge columns that share a width profile. The merged batch pads every column to the
    cross-column max width, and that padding costs real compute and peak activation memory: merging
    a narrow query column in with long document columns can end up slower than per-column forwards.
    So by default the first (anchor) column keeps its own forward pass and only the like-width
    candidate columns merge, mirroring the MultiVectorEncoder losses. With two columns that leaves
    nothing to merge, which is deliberate: a two-column loss is as often trained on a query and a
    document as on two sentences of the same kind, and it cannot tell which it has. Pass
    ``separate_first=False`` where every column is known to be a candidate. To force per-column
    forwards regardless, wrap the loss call in :func:`column_merging_disabled`.
    """
    features_list = list(features_list)
    if separate_first and features_list:
        return [
            model(features_list[0])["sentence_embedding"],
            *embed_columns(model, features_list[1:], separate_first=False),
        ]
    merged = merge_feature_batches(features_list) if len(features_list) > 1 else None
    if merged is None:
        return [model(features)["sentence_embedding"] for features in features_list]
    embeddings = model(merged)["sentence_embedding"]
    return list(embeddings.reshape(len(features_list), -1, *embeddings.shape[1:]).unbind(0))


def embed_columns_padded(
    model: nn.Module,
    features_list: list[dict[str, Any]],
    mini_batch_size: int | None = None,
    task_default: str | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    # [FUNCTION DOC]
    # Multi-vector counterpart of embed_columns without first-column separation. With more than one column, tries merging;
    # on success runs one chunked padded forward (task taken from the merged dict's "task" key or task_default) and splits
    # embeddings and masks back per column, converting the masks to boolean. Otherwise runs a chunked padded forward per
    # column. Returns a pair: list of per-column token embedding tensors (batch, tokens, dim) and list of boolean masks.
    """The multi-vector counterpart of :func:`embed_columns`.

    Returns per-column ``(batch_size, tokens, dim)`` embeddings and their boolean scoring masks.
    There is no ``separate_first`` here: the multi-vector losses peel the anchor column off at the
    call site and pass only the candidates, so everything handed to this function merges.
    """
    merged = merge_feature_batches(features_list) if len(features_list) > 1 else None
    if merged is not None:
        embeddings, mask = chunked_padded_forward(model, merged, merged.get("task", task_default), mini_batch_size)
        columns = len(features_list)
        return (
            list(embeddings.reshape(columns, -1, *embeddings.shape[1:]).unbind(0)),
            list(mask.bool().reshape(columns, -1, mask.shape[-1]).unbind(0)),
        )
    outputs = [
        chunked_padded_forward(model, features, features.get("task", task_default), mini_batch_size)
        for features in features_list
    ]
    return [embeddings for embeddings, _ in outputs], [mask.bool() for _, mask in outputs]
