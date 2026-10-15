# [MODULE DOC]
# Tests for SparseDistillKLDivLoss: per-side (student/teacher) temperature handling and the requirement
# that the loss is wrapped inside SpladeLoss or CSRLoss.
from __future__ import annotations

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import SparseDistillKLDivLoss


def _embeddings(count: int = 3) -> list[torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Returns a list of "count" (default 3) random 4x8 sparse tensors with negative entries clamped to
    # zero, generated from a fresh torch generator seeded with 21, so repeated calls return identical tensors.
    generator = torch.Generator().manual_seed(21)
    return [torch.randn(4, 8, generator=generator).relu().to_sparse() for _ in range(count)]


def test_sparse_distill_kl_div_forwards_per_side_temperatures(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Checks that the default loss has temperature, student and teacher temperatures all equal to 2.0; that
    # passing student 0.5 and teacher 2.0 keeps temperature 2.0 and sets the per-side values, and that the
    # student temperature appears in the config dict; and that the split-temperature loss computed from fixed
    # embeddings and 4x2 teacher score labels differs from the default loss.
    """The sparse subclass takes the per-side temperatures and hands them to the dense implementation,
    including its default of 2.0 rather than the dense 1.0."""
    default = SparseDistillKLDivLoss(splade_bert_tiny_model)
    assert (default.temperature, default.student_temperature, default.teacher_temperature) == (2.0, 2.0, 2.0)

    split = SparseDistillKLDivLoss(splade_bert_tiny_model, student_temperature=0.5, teacher_temperature=2.0)
    assert (split.temperature, split.student_temperature, split.teacher_temperature) == (2.0, 0.5, 2.0)
    assert split.get_config_dict()["student_temperature"] == 0.5

    labels = torch.tensor([[4.0, 1.0], [3.5, 0.5], [2.0, 1.5], [5.0, 0.0]])
    assert split.compute_loss_from_embeddings(_embeddings(), labels).item() != pytest.approx(
        default.compute_loss_from_embeddings(_embeddings(), labels).item()
    )


def test_sparse_distill_kl_div_must_be_wrapped(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Checks that calling the loss's forward directly (not wrapped) raises AttributeError whose message contains
    # "should not be used alone".
    """It only makes sense inside SpladeLoss / CSRLoss, which supply the sparsity regularizers, so
    calling it directly fails loudly instead of training an unregularized model."""
    loss = SparseDistillKLDivLoss(splade_bert_tiny_model)
    with pytest.raises(AttributeError, match="should not be used alone"):
        loss([{"input_ids": torch.ones(2, 3, dtype=torch.long)}], torch.zeros(2, 2))
