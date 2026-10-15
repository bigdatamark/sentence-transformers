# [MODULE DOC]
# Unit tests for CoSENTLoss and AnglELoss computed directly from precomputed embeddings, checking the ranking direction of the
# loss formula (log of one plus the sum of exponentiated scaled score differences for wrongly ordered pairs).
from __future__ import annotations

import math

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import AnglELoss, CoSENTLoss


@pytest.fixture
def dummy_model():
    # [FUNCTION DOC]
    # HELPER: Fixture returning an instance of an empty placeholder class used as the "model" argument; the losses under test only
    # use embeddings, so no real model is required.
    class DummyModel:
        # [CLASS DOC]
        # HELPER: Empty stand-in model class with no attributes or behavior.
        pass

    return DummyModel()


def _pairs(scores: list[float]) -> list[torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Given a list of scores, returns two float tensors (a column of the scores and a same-shaped column of ones) whose
    # row-wise dot product equals each score exactly.
    """Embeddings whose dot-product similarity is exactly `scores`, one per input pair."""
    left = torch.tensor([[s] for s in scores], dtype=torch.float32)
    return [left, torch.ones_like(left)]


@pytest.mark.parametrize(
    ("low_label_score", "high_label_score"),
    [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (2.0, -1.0)],
)
def test_cosent_penalises_the_lower_labelled_pair(dummy_model, low_label_score, high_label_score) -> None:
    # [FUNCTION DOC]
    # Parametrized over score pairs. Uses CoSENT with scale 1 and a dot-product similarity, labels 0 then 1. Asserts the loss equals
    # the natural log of one plus the exponential of (low-label score minus high-label score), confirming the lower-labelled pair is penalised.
    """The exponent is s(k,l) - s(i,j): the score of the pair with the *lower* expected similarity
    minus the score of the pair with the higher one, so ranking them the wrong way round costs more."""
    loss = CoSENTLoss(dummy_model, scale=1.0, similarity_fct=lambda a, b: (a * b).sum(-1))
    labels = torch.tensor([0.0, 1.0])

    value = loss.compute_loss_from_embeddings(_pairs([low_label_score, high_label_score]), labels)

    assert value.item() == pytest.approx(math.log(1 + math.exp(low_label_score - high_label_score)), abs=1e-5)


@pytest.mark.parametrize(("labels", "score_difference"), [([0.0, 1.0], -1.0), ([1.0, 0.0], 1.0)])
def test_angle_loss_ranks_actual_angle_scores(dummy_model, labels, score_difference) -> None:
    # [FUNCTION DOC]
    # Uses AnglELoss with scale 2. Checks that its angle similarity gives 0 for the first pair and 1 for the second pair of 2-D vectors,
    # then asserts the loss for the given labels equals log1p of the exponential of 2 times the expected score difference.
    loss = AnglELoss(dummy_model, scale=2.0)
    embeddings = [torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[1.0, 1.0], [1.0, 0.0]])]
    torch.testing.assert_close(loss.similarity_fct(*embeddings), torch.tensor([0.0, 1.0]))

    value = loss.compute_loss_from_embeddings(embeddings, torch.tensor(labels))

    assert value.item() == pytest.approx(math.log1p(math.exp(2.0 * score_difference)), abs=1e-5)
