# [MODULE DOC]
# HELPER: Sparse-tensor comparison utility for the sparse encoder tests.
from __future__ import annotations

import torch


def sparse_allclose(
    input: torch.Tensor, other: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    # [FUNCTION DOC]
    # HELPER: Compares two torch sparse tensors for approximate equality.
    # Inputs: two sparse tensors plus relative tolerance, absolute tolerance and an equal-NaN flag (forwarded to
    # the dense closeness check). Returns False if shapes differ; otherwise coalesces both tensors, returns False
    # if their index tensors are not exactly equal, and else returns whether their value tensors are close within
    # the tolerances. No side effects (coalescing produces new tensors).
    # KNOWN ISSUE: The docstring claims support for a dictionary format with "indices" and "values" keys, but the
    # code only accepts sparse tensors (it calls shape, coalesce, indices and values on them); a dict would fail.
    # Also, sparse tensors with the same values but differing explicit zero entries would compare unequal.
    """
    Check if two sparse embeddings are close to each other.

    This function works with sparse embeddings in either:
    1. Tensor format (assuming sparse tensors)
    2. Dictionary format with 'indices' and 'values' keys

    Args:
        input: First sparse embedding (tensor)
        other: Second sparse embedding (tensor)
        rtol: Relative tolerance
        atol: Absolute tolerance
        equal_nan: If True, NaN values in the same locations are considered equal

    Returns:
        bool: True if embeddings are close according to tolerances
    """
    # Check if shape matches
    if input.shape != other.shape:
        return False

    input = input.coalesce()
    other = other.coalesce()

    # Convert dict format to appropriate tensors if needed
    input_indices = input.indices()
    input_values = input.values()

    other_indices = other.indices()
    other_values = other.values()

    # Check if indices are the same
    if not torch.equal(input_indices, other_indices):
        return False

    # Check if values are close
    return torch.allclose(input_values, other_values, rtol=rtol, atol=atol, equal_nan=equal_nan)
