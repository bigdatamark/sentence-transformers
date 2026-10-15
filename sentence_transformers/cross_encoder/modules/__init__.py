# [PACKAGE DOC]
# Modules used to assemble a CrossEncoder pipeline. Re-exports the shared base modules (Dense, InputModule,
# Module, Asym/Router, Transformer) from the base package so everything needed to build a CrossEncoder is
# importable from one path, plus the cross-encoder-specific LogitScore module that turns causal LM logits
# into relevance scores. Only re-exports; no side effects.
from __future__ import annotations

# Base modules are re-exported here so all modules used to build a CrossEncoder
# can be imported from this single path.
from ...base.modules.dense import Dense
from ...base.modules.input_module import InputModule
from ...base.modules.module import Module
from ...base.modules.router import Asym, Router
from ...base.modules.transformer import Transformer
from .logit_score import LogitScore

__all__ = [
    "Dense",
    "InputModule",
    "Module",
    "Asym",
    "Router",
    "Transformer",
    "LogitScore",
]
