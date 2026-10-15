# [MODULE DOC] Softmax classification loss (the original SBERT NLI objective): a linear classifier over combined features of two
# sentence embeddings, trained with cross-entropy (or a custom loss).
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

import torch
import transformers
from packaging import version
from torch import Tensor, nn

from sentence_transformers.base.losses.merged_forward import embed_columns
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.util.decorators import deprecated_kwargs

logger = logging.getLogger(__name__)


class SoftmaxLoss(nn.Module):
    # [CLASS DOC] Loss module. Attributes: model, num_labels, the three feature flags (concatenate u and v, absolute difference,
    # element-wise product), classifier (linear layer on the model device from the concatenated width to num_labels,
    # trained with the loss), loss_fct (default cross-entropy; a single shared default instance).
    @deprecated_kwargs(sentence_embedding_dimension="embedding_dimension")
    def __init__(
        self,
        model: SentenceTransformer,
        embedding_dimension: int,
        num_labels: int,
        concatenation_sent_rep: bool = True,
        concatenation_sent_difference: bool = True,
        concatenation_sent_multiplication: bool = False,
        loss_fct: Callable = nn.CrossEntropyLoss(),
    ) -> None:
        # [FUNCTION DOC] The deprecated keyword sentence_embedding_dimension is remapped to embedding_dimension by a decorator. Counts
        # concatenated vectors (2 for u and v, plus 1 each for the difference and the product); raises ValueError if the
        # count is 0. Logs the count, creates the classifier and stores the loss function. Logs a warning when the installed
        # transformers is older than 4.43.0, where the classifier cannot be updated.
        """
        This loss was used in our SBERT publication (https://huggingface.co/papers/1908.10084) to train the SentenceTransformer
        model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

        :class:`MultipleNegativesRankingLoss` is an alternative loss function that often yields better results,
        as per https://huggingface.co/papers/2004.09813.

        Args:
            model (SentenceTransformer): The SentenceTransformer model.
            embedding_dimension (int): The dimension of the embeddings.
            num_labels (int): The number of different labels.
            concatenation_sent_rep (bool): Whether to concatenate vectors u,v for the softmax classifier. Defaults to True.
            concatenation_sent_difference (bool): Whether to add abs(u-v) for the softmax classifier. Defaults to True.
            concatenation_sent_multiplication (bool): Whether to add u*v for the softmax classifier. Defaults to False.
            loss_fct (Callable): Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss(). Defaults to nn.CrossEntropyLoss().

        References:
            - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks: https://huggingface.co/papers/1908.10084
            - `Training Examples > Natural Language Inference <../../../examples/sentence_transformer/training/nli/README.html>`_

        Requirements:
            1. input pairs with a class label

        Inputs:
            +---------------------------------------+--------+
            | Inputs                                | Labels |
            +=======================================+========+
            | (input_A, input_B) pairs              | class  |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": [
                        "A person on a horse jumps over a broken down airplane.",
                        "A person on a horse jumps over a broken down airplane.",
                        "A person on a horse jumps over a broken down airplane.",
                        "Children smiling and waving at camera",
                    ],
                    "sentence2": [
                        "A person is training his horse for a competition.",
                        "A person is at a diner, ordering an omelette.",
                        "A person is outdoors, on a horse.",
                        "There are children present.",
                    ],
                    "label": [1, 2, 0, 0],
                })
                loss = losses.SoftmaxLoss(model, model.get_embedding_dimension(), num_labels=3)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        if num_vectors_concatenated == 0:
            raise ValueError(
                "At least one of concatenation_sent_rep, concatenation_sent_difference, "
                "or concatenation_sent_multiplication must be True."
            )
        logger.info(f"Softmax loss: {num_vectors_concatenated} vectors concatenated")
        self.classifier = nn.Linear(num_vectors_concatenated * embedding_dimension, num_labels, device=model.device)
        self.loss_fct = loss_fct

        if version.parse(transformers.__version__) < version.parse("4.43.0"):
            logger.warning(
                "SoftmaxLoss requires transformers >= 4.43.0 to work correctly. "
                "Otherwise, the classifier layer that maps embeddings to the labels cannot be updated. "
                "Consider updating transformers with `pip install transformers>=4.43.0`."
            )

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor | tuple[Tensor, Tensor]:
        # [FUNCTION DOC] Embed the two input columns, build the feature vector from u, v, absolute difference and product per the flags,
        # and apply the classifier. With labels, returns the loss of logits against the flattened labels; without labels,
        # returns a tuple (list of embeddings, logits), as used by the label accuracy evaluator.
        reps = embed_columns(self.model, sentence_features)
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output

    def get_config_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a config dict with num_labels and the three feature flags.
        return {
            "num_labels": self.num_labels,
            "concatenation_sent_rep": self.concatenation_sent_rep,
            "concatenation_sent_difference": self.concatenation_sent_difference,
            "concatenation_sent_multiplication": self.concatenation_sent_multiplication,
        }

    @property
    def citation(self) -> str:
        # [FUNCTION DOC] Return the BibTeX citation for Sentence-BERT.
        return """
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
"""
