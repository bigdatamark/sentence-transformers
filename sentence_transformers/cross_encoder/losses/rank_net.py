# [MODULE DOC]
# RankNet pairwise learning-to-rank loss for CrossEncoder, implemented as LambdaLoss with no weighting.
from __future__ import annotations

from typing import Literal

from torch import nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import LambdaLoss, NoWeightingScheme


class RankNetLoss(LambdaLoss):
    # [CLASS DOC]
    # LambdaLoss subclass fixed to the no-weighting scheme, which reduces LambdaLoss to the RankNet objective:
    # for every pair of documents of a query where one has a higher label, apply a logistic loss on the scaled
    # score difference. Inputs are queries with lists of documents and per-document labels (binary or graded).
    def __init__(
        self,
        model: CrossEncoder,
        k: int | None = None,
        sigma: float = 1.0,
        eps: float = 1e-10,
        reduction_log: Literal["natural", "binary"] = "binary",
        activation_fn: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Forward all parameters (model, optional top-k cutoff, sigma scale, eps, log base for the reduction,
        # activation, mini-batch size) to LambdaLoss with a no-weighting scheme.
        # KNOWN ISSUE: The docstring does not document the k and reduction_log parameters, which are accepted and
        # forwarded to LambdaLoss.
        """
        RankNet loss implementation for learning to rank. This loss function implements the RankNet algorithm,
        which learns a ranking function by optimizing pairwise document comparisons using a neural network.
        The implementation is optimized to handle padded documents efficiently by only processing valid
        documents during model inference.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            sigma (float): Score difference weight used in sigmoid (default: 1.0)
            eps (float): Small constant for numerical stability (default: 1e-10)
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the
                loss. Defaults to :class:`~torch.nn.Identity`.
            mini_batch_size (int, optional): Number of samples to process in each forward pass. This has a significant
                impact on the memory consumption and speed of the training process. Three cases are possible:
                - If ``mini_batch_size`` is None, the ``mini_batch_size`` is set to the batch size.
                - If ``mini_batch_size`` is greater than 0, the batch is split into mini-batches of size ``mini_batch_size``.
                - If ``mini_batch_size`` is <= 0, the entire batch is processed at once.
                Defaults to None.

        References:
            - Learning to Rank using Gradient Descent: https://icml.cc/Conferences/2015/wp-content/uploads/2015/06/icml_ranking.pdf
            - `Cross Encoder > Training Examples > MS MARCO <../../../examples/cross_encoder/training/ms_marco/README.html>`_

        Requirements:
            1. Query with multiple documents (pairwise approach)
            2. Documents must have relevance scores/labels. Both binary and continuous labels are supported.

        Inputs:
            +----------------------------------------+--------------------------------+-------------------------------+
            | Inputs                                 | Labels                         | Number of Model Output Labels |
            +========================================+================================+===============================+
            | (query, [doc1, doc2, ..., docN])       | [score1, score2, ..., scoreN]  | 1                             |
            +----------------------------------------+--------------------------------+-------------------------------+

        Recommendations:
            - Use :class:`~sentence_transformers.util.hard_negatives.mine_hard_negatives` with ``output_format="labeled-list"``
              to convert question-answer pairs to the required input format with hard negatives.

        Relations:
            - :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` can be seen as an extension of this loss
              where each score pair is weighted. Alternatively, this loss can be seen as a special case of the
              :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` without a weighting scheme.
            - :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` with its default NDCGLoss2++ weighting
              scheme anecdotally performs better than the other losses with the same input format.

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "docs": [
                        ["Pandas are a kind of bear.", "Pandas are kind of like fish."],
                        ["The capital of France is Paris.", "Paris is the capital of France.", "Paris is quite large."],
                    ],
                    "labels": [[1, 0], [1, 1, 0]],
                })
                loss = losses.RankNetLoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__(
            model=model,
            weighting_scheme=NoWeightingScheme(),
            k=k,
            sigma=sigma,
            eps=eps,
            reduction_log=reduction_log,
            activation_fn=activation_fn,
            mini_batch_size=mini_batch_size,
        )

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        # [FUNCTION DOC]
        # Return the parent LambdaLoss config with the weighting scheme entry removed (it is fixed).
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        config = super().get_config_dict()
        del config["weighting_scheme"]
        return config

    @property
    def citation(self) -> str:
        # [FUNCTION DOC]
        # Return the BibTeX citation string for Burges et al. 2005 (RankNet).
        return """
@inproceedings{burges2005learning,
  title={Learning to Rank using Gradient Descent},
  author={Burges, Chris and Shaked, Tal and Renshaw, Erin and Lazier, Ari and Deeds, Matt and Hamilton, Nicole and Hullender, Greg},
  booktitle={Proceedings of the 22nd international conference on Machine learning},
  pages={89--96},
  year={2005}
}
"""
