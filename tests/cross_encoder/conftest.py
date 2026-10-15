# [MODULE DOC]
# HELPER: Pytest fixtures shared by the CrossEncoder test suite. Provides CrossEncoder models loaded from the
# Hugging Face Hub (network access / local cache required). Expensive models are loaded once per session by a
# private session-scoped fixture, and a public function-scoped fixture hands each test a deep copy so tests can
# mutate the model without affecting each other.
from __future__ import annotations

from copy import deepcopy

import pytest

from sentence_transformers import CrossEncoder


@pytest.fixture()
def distilroberta_base_ce_model() -> CrossEncoder:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture. Loads a fresh, untrained CrossEncoder from "distilbert/distilroberta-base"
    # with a single output label (regression/binary head is randomly initialised). Returns the CrossEncoder.
    return CrossEncoder("distilbert/distilroberta-base", num_labels=1)


@pytest.fixture(scope="session")
def _reranker_bert_tiny_model_v54() -> CrossEncoder:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture. Loads the tiny reranker "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-STv54"
    # once per test session and returns it. Not meant to be used directly; use the deep-copying fixture below.
    return CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce-STv54")


@pytest.fixture()
def reranker_bert_tiny_model_v54(_reranker_bert_tiny_model_v54) -> CrossEncoder:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture. Returns a deep copy of the session-cached tiny reranker so each test gets an
    # independent, mutable model.
    return deepcopy(_reranker_bert_tiny_model_v54)


@pytest.fixture(scope="session")
def _nli_minilm_model() -> CrossEncoder:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture. Loads the NLI cross-encoder "cross-encoder/nli-MiniLM2-L6-H768" (three labels)
    # once per session and returns it.
    return CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")


@pytest.fixture()
def nli_minilm_model(_nli_minilm_model) -> CrossEncoder:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture. Returns a deep copy of the session-cached NLI MiniLM cross-encoder.
    return deepcopy(_nli_minilm_model)
