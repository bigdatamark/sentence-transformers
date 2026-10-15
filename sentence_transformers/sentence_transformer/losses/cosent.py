# [MODULE DOC] CoSENT loss: a ranking loss over pairs of labeled sentence pairs that pushes the model's pair similarity order
# to match the order of the gold similarity labels.
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.base.losses.merged_forward import embed_columns
from sentence_transformers.sentence_transformer.model import SentenceTransformer


class CoSENTLoss(nn.Module):
    # [CLASS DOC] Loss module. Attributes: model, similarity_fct (pairwise similarity between aligned rows; default pairwise
    # cosine), scale (inverse temperature, default 20).
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.pairwise_cos_sim) -> None:
        # [FUNCTION DOC] Store the model, similarity function and scale.
        """
        This class implements CoSENT (Consistent SENTence embedding) loss.
        It expects that each of the inputs consists of a pair of inputs (e.g., texts) and a float valued label,
        representing the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = log(1 + sum(exp(scale * (s(k,l) - s(i,j)))))``, where ``s`` is the score returned by
        ``similarity_fct``. The sum is over all pairs of input pairs in the batch such that the similarity label of
        ``(i,j)`` is greater than that of ``(k,l)``.

        Anecdotal experiments show that this loss function produces a more powerful training signal than :class:`CosineSimilarityLoss`,
        resulting in faster convergence and a final model with superior performance. Consequently, CoSENTLoss may be used
        as a drop-in replacement for :class:`CosineSimilarityLoss` in any training script.

        Args:
            model: SentenceTransformerModel
            similarity_fct: Function to compute the PAIRWISE similarity
                between embeddings. Default is
                ``util.pairwise_cos_sim``.
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.

        References:
            - For further details, see: https://penghao-bdsc.github.io/papers/CoSENT_TASLP2024.pdf

        Requirements:
            - Input pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Inputs                         | Labels                 |
            +================================+========================+
            | (input_A, input_B) pairs       | float similarity score |
            +--------------------------------+------------------------+

        Relations:
            - :class:`AnglELoss` is CoSENTLoss with ``pairwise_angle_sim`` as the metric, rather than ``pairwise_cos_sim``.
            - :class:`CosineSimilarityLoss` seems to produce a weaker training signal than CoSENTLoss. In our experiments, CoSENTLoss is recommended.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                    "score": [1.0, 0.3],
                })
                loss = losses.CoSENTLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # [FUNCTION DOC] Embed every input column with the shared column embedding helper and compute the loss with the labels.
        embeddings = embed_columns(self.model, sentence_features)

        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        # [FUNCTION DOC] Score each aligned pair from the first two embedding columns (extra columns are ignored here), multiply by
        # scale, and form the matrix of score differences (row pair minus column pair). Keep only entries where the
        # row pair's label is lower than the column pair's (others get a huge negative offset), append a single zero,
        # and return the log-sum-exp of all entries, i.e. log(1 + sum of exp(lower-labeled score minus higher-labeled
        # score)).
        """
        Compute the CoSENT loss from embeddings.

        Args:
            embeddings: List of embeddings
            labels: Labels indicating the similarity scores of the pairs

        Returns:
            Loss value
        """

        scores = self.similarity_fct(embeddings[0], embeddings[1])
        scores = scores * self.scale
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a config dict with scale and the similarity function's name (requires the function to have a name).
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

    @property
    def citation(self) -> str:
        # [FUNCTION DOC] Return the BibTeX citation for the CoSENT paper.
        return """
@article{10531646,
    author={Huang, Xiang and Peng, Hao and Zou, Dongcheng and Liu, Zhiwei and Li, Jianxin and Liu, Kay and Wu, Jia and Su, Jianlin and Yu, Philip S.},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    title={CoSENT: Consistent Sentence Embedding via Similarity Ranking},
    year={2024},
    doi={10.1109/TASLP.2024.3402087}
}
"""
