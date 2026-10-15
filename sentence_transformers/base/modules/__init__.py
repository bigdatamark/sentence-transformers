# [PACKAGE DOC]
# Base module package: building blocks shared by all model families. Re-exports Module (abstract base with config and
# weight save/load helpers), InputModule (base for modules that turn raw inputs into features), Transformer (wraps a
# Hugging Face model plus processor/tokenizer), Router (routes inputs to different sub-module chains per task or
# modality), Dense (linear projection with activation and optional residual) and Normalize (L2 normalisation).
from __future__ import annotations

from .dense import Dense
from .input_module import InputModule
from .module import Module
from .normalize import Normalize
from .router import Router
from .transformer import Transformer

__all__ = [
    "Dense",
    "InputModule",
    "Module",
    "Normalize",
    "Router",
    "Transformer",
]
