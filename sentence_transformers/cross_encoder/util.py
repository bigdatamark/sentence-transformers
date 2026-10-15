# [MODULE DOC]
# HELPER: Backwards-compatibility shim for the cross encoder package.
# Re-exports the two argument-normalizing decorators (one for the CrossEncoder constructor, one for its
# predict/rank methods) from their canonical home in the shared util.decorators module, so older import
# paths keep working. Defines nothing itself and has no side effects beyond the import.
from __future__ import annotations

# Re-export from the canonical location for backwards compatibility
from sentence_transformers.util.decorators import (  # noqa: F401
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
)
