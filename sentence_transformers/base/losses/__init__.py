# [PACKAGE DOC]
# Base losses package with shared machinery used by concrete loss implementations. Re-exports the gradient-cache
# utilities (CachedLossMixin, RandContext, has_static_embedding_input, uses_gradient_cache) for memory-efficient
# "cached" contrastive losses, and column_merging_disabled, a context manager that turns off merging of several input
# columns into one forward pass.
from __future__ import annotations

from .gradcache import (
    CachedLossMixin,
    RandContext,
    has_static_embedding_input,
    uses_gradient_cache,
)
from .merged_forward import column_merging_disabled

__all__ = [
    "CachedLossMixin",
    "RandContext",
    "column_merging_disabled",
    "has_static_embedding_input",
    "uses_gradient_cache",
]
