# [MODULE DOC]
# Sparse cosine similarity loss: regresses the cosine similarity of sparse embedding pairs onto
# float labels, reusing the dense CosineSimilarityLoss computation. Must be wrapped in SpladeLoss
# or CSRLoss.
from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn
from torch import Tensor

from sentence_transformers.sentence_transformer.losses.cosine_similarity import CosineSimilarityLoss
from sentence_transformers.sparse_encoder.model import SparseEncoder


class SparseCosineSimilarityLoss(CosineSimilarityLoss):
    # [CLASS DOC]
    # CosineSimilarityLoss subclass for SparseEncoder models. The inherited compute_loss_from_embeddings
    # computes the cosine similarity of each pair, applies cos_score_transformation, and compares it to
    # the labels with loss_fct (MSE by default).
    def __init__(
        self,
        model: SparseEncoder,
        loss_fct: nn.Module = nn.MSELoss(),
        cos_score_transformation: nn.Module = nn.Identity(),
    ) -> None:
        # [FUNCTION DOC]
        # Sets the model's similarity_fn_name to "cosine" (mutating the model) and initializes the parent
        # with the model, loss function and score transformation. The default loss and transformation
        # modules are single instances created at definition time and shared between all instances.
        """
        SparseCosineSimilarityLoss expects that the inputs consists of a pair of inputs (e.g., texts) and a float label.
        It computes the vectors ``u = model(input_A)`` and ``v = model(input_B)`` and measures the cosine-similarity
        between the two. By default, it minimizes the following loss:
        ``||input_label - cos_score_transformation(cosine_sim(u,v))||_2``.

        Args:
            model: SparseEncoder model
            loss_fct: Which pytorch loss function should be used to
                compare the ``cosine_similarity(u, v)`` with the
                input_label? By default, MSE is used: ``||input_label -
                cosine_sim(u, v)||_2``
            cos_score_transformation: The cos_score_transformation
                function is applied on top of cosine_similarity. By
                default, the identify function is used (i.e. no change).

        Requirements:
            - Need to be used in SpladeLoss or CSRLoss as a loss function.
            - Input pairs with corresponding similarity scores in range `[0, 1]`

        Inputs:
            +--------------------------------+------------------------+
            | Inputs                         | Labels                 |
            +================================+========================+
            | (input_A, input_B) pairs       | float similarity score |
            +--------------------------------+------------------------+

        Relations:
            - :class:`SparseAnglELoss` is :class:`SparseCoSENTLoss` with ``pairwise_angle_sim`` as the metric, rather than ``pairwise_cos_sim``.

        Example:
            ::

                from datasets import Dataset
                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("distilbert/distilbert-base-uncased")
                train_dataset = Dataset.from_dict(
                    {
                        "sentence1": ["It's nice weather outside today.", "He drove to work."],
                        "sentence2": ["It's so sunny.", "She walked to the store."],
                        "score": [1.0, 0.3],
                    }
                )
                loss = losses.SpladeLoss(
                    model=model,
                    loss=losses.SparseCosineSimilarityLoss(model),
                    document_regularizer_weight=5e-5,
                    use_document_regularizer_only=True,
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        model.similarity_fn_name = "cosine"
        super().__init__(model, loss_fct=loss_fct, cos_score_transformation=cos_score_transformation)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Always raises AttributeError: this loss must be used inside SpladeLoss or CSRLoss.
        raise AttributeError("SparseCosineSimilarityLoss should not be used alone. Use it with SpladeLoss or CSRLoss.")
