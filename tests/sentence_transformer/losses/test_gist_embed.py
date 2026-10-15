# [MODULE DOC]
# Regression tests for GISTEmbedLoss (in-batch negatives filtered by a guide model's similarities). Covers the positive-column mask
# offset when embeddings are gathered across devices on a non-zero rank, and the relative margin strategy with negative positive-pair
# scores. Losses are built without real models: both model and guide are fakes that return precomputed embeddings from the feature dicts.
from __future__ import annotations

import math

import pytest
import torch

import sentence_transformers.sentence_transformer.losses.gist_embed as ge
from sentence_transformers.sentence_transformer.losses import GISTEmbedLoss


def _make_loss(margin: float) -> GISTEmbedLoss:
    # [FUNCTION DOC]
    # HELPER: Creates a GISTEmbedLoss bypassing its constructor and sets attributes directly: temperature 0.01, cosine similarity, absolute
    # margin strategy with the given margin, contrasting of anchors and positives enabled, cross-device gathering enabled, no retokenizing,
    # and a cross-entropy criterion. Model and guide are both a function returning the feature's own "sentence_embedding".
    """Build a GISTEmbedLoss without loading real models.

    ``forward`` encodes via ``self.model`` / ``self.guide``. We replace both with a fake
    that returns precomputed embeddings carried on the feature dict, so no model is needed.
    """
    obj = GISTEmbedLoss.__new__(GISTEmbedLoss)
    torch.nn.Module.__init__(obj)
    obj.temperature = 0.01
    obj.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    obj.margin_strategy = "absolute"
    obj.margin = margin
    obj.contrast_anchors = True
    obj.contrast_positives = True
    obj.gather_across_devices = True
    obj.must_retokenize = False
    obj.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    fake = lambda sentence_feature: {"sentence_embedding": sentence_feature["sentence_embedding"]}
    obj.model = fake
    obj.guide = fake
    return obj


@pytest.fixture
def simulate_rank1_world2(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Fixture simulating rank 1 of a world of size 2 in one process. Monkeypatches the loss module's all_gather_with_grad to prepend
    # a seeded random block of 3 rows (rank 0's data) to the given tensor, and get_rank to return 1. Returns the per-rank batch size 3.
    """Run the rank=1/world=2 gather path in a single process: ``all_gather_with_grad``
    prepends a deterministic rank-0 block, and the rank is reported as 1 (offset = batch_size).
    """
    per = 3

    def fake_gather(tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Returns a seeded random 3-row block with the tensor's width concatenated before the tensor along rows.
        g = torch.Generator().manual_seed(1234)
        rank0_block = torch.randn(per, tensor.size(1), generator=g)
        return torch.cat([rank0_block, tensor], dim=0)

    monkeypatch.setattr(ge, "all_gather_with_grad", fake_gather)
    monkeypatch.setattr(ge, "get_rank", lambda: 1)
    return per


def _features(per: int, dim: int = 16, seed: int = 7):
    # [FUNCTION DOC]
    # HELPER: Returns anchor and positive feature dicts with seeded random embeddings of shape per by dim.
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randn(per, dim, generator=g)
    positives = torch.randn(per, dim, generator=g)
    return [{"sentence_embedding": anchors}, {"sentence_embedding": positives}]


def test_gather_rank1_positive_not_masked_with_margin(simulate_rank1_world2):
    # [FUNCTION DOC]
    # With simulated rank 1 and margin 0.1, asserts the loss is finite (each anchor's positive at the offset column must not be masked to -inf).
    """Regression for the rank>0 positive-mask offset bug in the non-cached GISTEmbedLoss.

    The old ``torch.eye(*shape)`` mask protects columns [0..batch-1], but on rank 1 each
    anchor's positive is at gathered column ``offset + row``. With ``margin > 0`` the unprotected
    positive is masked to -inf, the CE-target logit becomes -inf, and the loss is +inf. The
    offset-aware mask keeps it finite.
    """
    per = simulate_rank1_world2
    loss = _make_loss(margin=0.1)(_features(per), labels=None)
    assert torch.isfinite(loss), f"rank>0 loss must be finite, got {loss.item()}"


def test_gather_rank1_margin_zero_baseline(simulate_rank1_world2):
    # [FUNCTION DOC]
    # Baseline with simulated rank 1 and margin 0: asserts the loss is finite.
    """Sanity baseline: with margin=0 the positive equals its own threshold and is not
    suppressed regardless of the mask, so the loss is finite both before and after the fix.
    """
    per = simulate_rank1_world2
    loss = _make_loss(margin=0.0)(_features(per), labels=None)
    assert torch.isfinite(loss)


def test_positive_mask_protects_ce_target_column(simulate_rank1_world2):
    # [FUNCTION DOC]
    # With simulated rank 1 and margin 0.1, replaces the cross-entropy child module by a capturing wrapper (deleting the registered module
    # first), runs the loss, and asserts cross-entropy was called and that for every call the logit at each row's target column is finite,
    # and the loss is finite.
    """Drive the real loss and assert it protects exactly the CE-target column.

    We capture the (masked) score matrix the loss feeds to cross-entropy and check that every
    anchor's CE-target logit survived suppression (is finite). With the old offset-unaware
    ``torch.eye`` mask, on rank 1 that column is set to -inf, so this fails on the pre-fix code.
    """
    per = simulate_rank1_world2
    loss_fn = _make_loss(margin=0.1)

    captured = []
    real_ce = loss_fn.cross_entropy_loss

    def capturing_ce(scores, labels):
        # [FUNCTION DOC]
        # HELPER: Records the scores and labels, then returns the real cross-entropy of them.
        captured.append((scores, labels))
        return real_ce(scores, labels)

    # cross_entropy_loss is registered as an nn.Module child, so drop it before assigning a plain callable.
    del loss_fn.cross_entropy_loss
    loss_fn.cross_entropy_loss = capturing_ce

    loss = loss_fn(_features(per), labels=None)

    assert captured, "cross-entropy was never called"
    for scores, labels in captured:
        target_logits = scores[torch.arange(scores.size(0)), labels]
        assert torch.isfinite(target_logits).all(), (
            f"each anchor's CE-target logit must survive masking, got {target_logits.tolist()}"
        )
    assert torch.isfinite(loss)


def _make_relative_loss(margin: float) -> GISTEmbedLoss:
    # [FUNCTION DOC]
    # HELPER: Like the absolute-margin builder but with the relative margin strategy and no anchor/positive contrasting or gathering.
    """relative-margin GISTEmbedLoss with no gather/contrast, fed precomputed embeddings (#3819)."""
    obj = GISTEmbedLoss.__new__(GISTEmbedLoss)
    torch.nn.Module.__init__(obj)
    obj.temperature = 0.01
    obj.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    obj.margin_strategy = "relative"
    obj.margin = margin
    obj.contrast_anchors = False
    obj.contrast_positives = False
    obj.gather_across_devices = False
    obj.must_retokenize = False
    obj.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    fake = lambda sentence_feature: {"sentence_embedding": sentence_feature["sentence_embedding"]}
    obj.model = fake
    obj.guide = fake
    return obj


def _negative_score_features():
    # [FUNCTION DOC]
    # HELPER: Returns two anchors and two positives (3-D unit vectors) where anchor 0's positive has cosine -0.50 while the other positive
    # has cosine -0.49 with it (closer than its own positive); anchor 1 equals positive 1.
    """anchor0's positive has a negative cosine (-0.50). A non-paired candidate (column 1) is
    *more* similar (-0.49). anchor1 is paired with that candidate (diagonal cosine 1.0)."""
    a0 = [1.0, 0.0, 0.0]
    p0 = [-0.50, math.sqrt(1 - 0.50**2), 0.0]  # cos(a0, p0) = -0.50
    p1 = [-0.49, 0.0, math.sqrt(1 - 0.49**2)]  # cos(a0, p1) = -0.49 (more similar than the positive)
    a1 = p1  # cos(a1, p1) = 1.0
    anchors = torch.tensor([a0, a1])
    positives = torch.tensor([p0, p1])
    return [{"sentence_embedding": anchors}, {"sentence_embedding": positives}]


def test_relative_margin_negative_positive_score_suppresses_closer_negative():
    # [FUNCTION DOC]
    # With relative margin 0.05, captures the score matrix passed to cross-entropy and asserts the candidate closer than anchor 0's negative-scored
    # positive is masked to -inf, while the true positive and the overall loss remain finite.
    """Regression for #3819 in GISTEmbedLoss: with ``margin_strategy="relative"`` and a negative
    positive-pair score, a candidate more similar to the anchor than the positive must still be
    suppressed. The old ``positive * (1 - margin)`` threshold rises for negative scores and lets
    it through, filtering less than ``margin=0``."""
    loss_fn = _make_relative_loss(margin=0.05)

    captured = []
    real_ce = loss_fn.cross_entropy_loss

    def capturing_ce(scores, labels):
        # [FUNCTION DOC]
        # HELPER: Records the scores, then returns the real cross-entropy.
        captured.append(scores)
        return real_ce(scores, labels)

    del loss_fn.cross_entropy_loss
    loss_fn.cross_entropy_loss = capturing_ce

    loss = loss_fn(_negative_score_features(), labels=None)

    assert captured, "cross-entropy was never called"
    scores = captured[0]
    # anchor0's closer-than-positive candidate (column 1) must be suppressed to -inf.
    assert scores[0, 1].item() == float("-inf"), f"closer negative must be masked, got {scores[0, 1].item()}"
    # the true positive (CE target) must survive masking, and the loss stays finite.
    assert torch.isfinite(scores[0, 0]) and torch.isfinite(loss)
