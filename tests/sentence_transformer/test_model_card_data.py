# [MODULE DOC]
# Tests for SentenceTransformer model card data: recorded base model name and revision, and the
# "generated_from_trainer" tag added when a trainer is constructed.
from __future__ import annotations

import pytest

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.util import is_training_available


@pytest.mark.parametrize(
    ("revision", "expected_base_revision"),
    [
        ("f3cb857cba53019a20df283396bcca179cf051a4", "f3cb857cba53019a20df283396bcca179cf051a4"),
        ("f3cb857", "f3cb857"),
        ("main", "valid-revision"),
        (None, "valid-revision"),
    ],
)
def test_model_card_data(revision, expected_base_revision) -> None:
    # [FUNCTION DOC]
    # Parametrized over a revision argument (full commit hash, short hash, "main", None). Loads the tiny STSB
    # BERT model from the Hub (network) with that revision and asserts the model card's base model equals the
    # model name; for "main"/None it asserts the base revision is a 40-character resolved hash, otherwise that it
    # equals the given revision verbatim.
    model_name = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = SentenceTransformer(model_name, revision=revision)

    assert model.model_card_data.base_model == model_name
    if expected_base_revision == "valid-revision":
        assert model.model_card_data.base_model_revision
        assert len(model.model_card_data.base_model_revision) == 40
    else:
        assert model.model_card_data.base_model_revision == expected_base_revision


@pytest.mark.skipif(
    not is_training_available(), reason='Sentence Transformers was not installed with the `["train"]` extra.'
)
def test_generated_from_trainer_tag(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Skipped when training extras are missing. Asserts the "generated_from_trainer" tag is absent, constructs a
    # SentenceTransformerTrainer around the model, and asserts the tag is then present (the trainer mutates the
    # model's card data).
    model = stsb_bert_tiny_model

    assert "generated_from_trainer" not in model.model_card_data.tags
    SentenceTransformerTrainer(model)
    assert "generated_from_trainer" in model.model_card_data.tags
