# [MODULE DOC]
# Tests that TripletLoss decreases when the positive is closer to the anchor than the negative, for each supported distance metric.
from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses.triplet import TripletDistanceMetric, TripletLoss


@pytest.fixture
def dummy_model():
    # [FUNCTION DOC]
    # HELPER: Fixture returning an empty placeholder object passed as the model; the loss is computed from embeddings only.
    class DummyModel:
        # [CLASS DOC]
        # HELPER: Empty stand-in model class.
        pass

    return DummyModel()


@pytest.mark.parametrize(
    "distance_metric",
    [TripletDistanceMetric.COSINE, TripletDistanceMetric.EUCLIDEAN, TripletDistanceMetric.MANHATTAN],
    ids=["cosine", "euclidean", "manhattan"],
)
def test_triplet_loss_correct_direction(dummy_model, distance_metric):
    # [FUNCTION DOC]
    # Parametrized over cosine, euclidean and manhattan distances with margin 1. Builds a single 3-D anchor, a nearby positive and a
    # far negative; computes the loss for the correct triplet and for the swapped one, and asserts the correct triplet's loss is lower.
    """Loss should be lower when the positive is closer to the anchor than the negative."""
    loss_fn = TripletLoss(model=dummy_model, distance_metric=distance_metric, triplet_margin=1.0)

    anchor = torch.tensor([[1.0, 0.0, 0.0]])
    positive = torch.tensor([[0.9, 0.1, 0.0]])  # close to anchor
    negative = torch.tensor([[0.0, 1.0, 0.0]])  # far from anchor

    # Good triplet: positive is closer than negative → should yield low/zero loss
    good_loss = loss_fn.compute_loss_from_embeddings([anchor, positive, negative], labels=None)

    # Bad triplet: swap positive and negative → should yield higher loss
    bad_loss = loss_fn.compute_loss_from_embeddings([anchor, negative, positive], labels=None)

    assert good_loss < bad_loss, (
        f"Good triplet loss ({good_loss:.4f}) should be less than bad triplet loss ({bad_loss:.4f})"
    )
