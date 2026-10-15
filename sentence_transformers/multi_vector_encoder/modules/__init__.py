# [PACKAGE DOC]
# Modules specific to multi-vector (late-interaction, ColBERT-style) encoders.
# Exposes MultiVectorMask (drops/masks token embeddings such as punctuation or padding before scoring)
# and the token pooling family (BaseTokenPooling abstract base, HierarchicalTokenPooling which clusters
# token vectors, LambdaTokenPooling which pools with a user callable) that reduce the number of stored
# token vectors per document. Only re-exports; no side effects.
from __future__ import annotations

from .multi_vector_mask import MultiVectorMask
from .token_pooling import BaseTokenPooling, HierarchicalTokenPooling, LambdaTokenPooling

__all__ = [
    "BaseTokenPooling",
    "HierarchicalTokenPooling",
    "LambdaTokenPooling",
    "MultiVectorMask",
]
