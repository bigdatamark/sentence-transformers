# [MODULE DOC]
# Tests for SparseCoSENTLoss: verifies that its default similarity function is the pairwise cosine
# similarity (one score per input pair) rather than a full similarity matrix.
from __future__ import annotations

import torch

from sentence_transformers import SparseEncoder, util
from sentence_transformers.sparse_encoder.losses import SparseCoSENTLoss


def test_sparse_cosent_loss_default_matches_explicit_pairwise(
    splade_bert_tiny_model: SparseEncoder,
) -> None:
    # [FUNCTION DOC]
    # Builds one SparseCoSENTLoss with default settings and one with pairwise cosine similarity passed explicitly,
    # asserts both use the pairwise cosine function, then, using seeded random non-negative sparse 4x8 embeddings
    # for two columns and four float labels, asserts both losses compute the same value. Uses the
    # splade_bert_tiny_model fixture; seeds the global torch RNG.
    # CoSENT needs one similarity per input pair. The old cos_sim default returned a
    # full similarity matrix that silently broadcast into a different objective.
    default = SparseCoSENTLoss(splade_bert_tiny_model)
    reference = SparseCoSENTLoss(splade_bert_tiny_model, similarity_fct=util.pairwise_cos_sim)

    assert default.similarity_fct is util.pairwise_cos_sim
    assert reference.similarity_fct is util.pairwise_cos_sim

    torch.manual_seed(12)
    embeddings = [torch.randn(4, 8).relu().to_sparse(), torch.randn(4, 8).relu().to_sparse()]
    labels = torch.tensor([0.9, 0.1, 0.8, 0.2])

    assert torch.allclose(
        default.compute_loss_from_embeddings(embeddings, labels),
        reference.compute_loss_from_embeddings(embeddings, labels),
    )
