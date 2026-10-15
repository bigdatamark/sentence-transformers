# [MODULE DOC] Contrastive Tension (CT) losses for unsupervised training: two copies of the encoder are trained so that
# identical inputs score high and different inputs score low. Contains the pairwise binary cross-entropy
# variant, the in-batch negatives variant, and a legacy data loader producing CT pair batches.
from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.sentence_transformer.readers import InputExample
from sentence_transformers.util import cos_sim, similarity_fct_name


class ContrastiveTensionLoss(nn.Module):
    # [CLASS DOC] Loss module with two encoders: model2 (the original model, used at inference) and model1 (an independent deep
    # copy made at construction). Attribute criterion is binary cross-entropy with logits summed over the batch.
    # Expects two input columns (pairs).
    # KNOWN ISSUE: The docstring opens by saying the loss expects only single inputs, but the code requires exactly two input columns (pairs).
    """
    This loss expects only single inputs, without any labels. Positive and negative pairs are automatically created via random sampling,
    such that a positive pair consists of two identical inputs and a negative pair consists of two different inputs. An independent
    copy of the encoder model is created, which is used for encoding the first input of each pair. The original encoder model encodes the
    second input. The embeddings are compared and scored using the generated labels (1 if positive, 0 if negative) using the binary cross
    entropy objective.

    Generally, :class:`ContrastiveTensionLossInBatchNegatives` is recommended over this loss, as it gives a stronger training signal.

    Args:
        model: SentenceTransformer model

    References:
        * Semantic Re-Tuning with Contrastive Tension: https://openreview.net/pdf?id=Ov_sMNau-PF
        * `Unsupervised Learning > CT <../../../examples/sentence_transformer/unsupervised_learning/CT/README.html>`_

    Inputs:
        +------------------------------------------+
        | Inputs                         | Labels  |
        +================================+=========+
        | (input_A, input_B) pairs       | None    |
        +------------------------------------------+

    Relations:
        * :class:`ContrastiveTensionLossInBatchNegatives` uses in-batch negative sampling, which gives a stronger training signal than this loss.

    Example:

        Using a dataset with input pairs that sometimes are identical (positive pairs) and sometimes different (negative pairs):

        ::

            import random
            from datasets import Dataset
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.losses import ContrastiveTensionDataLoader
            from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
            from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer

            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # The dataset pairs must sometimes contain identical sentences (positive pairs) and sometimes different sentences (negative pairs).
            train_dataset = Dataset.from_dict({
                "sentence_A": [
                    "It's nice weather outside today.",
                    "He drove to work.",
                    "It's so sunny.",
                ]
                "sentence_B": [
                    "It's nice weather outside today.",
                    "He drove to work.",
                    "It's so sunny.",
                ]
            })
            train_loss = ContrastiveTensionLoss(model=model)

            args = SentenceTransformerTrainingArguments(
                num_train_epochs=10,
                per_device_train_batch_size=32,
                eval_steps=0.1,
                logging_steps=0.01,
                learning_rate=5e-5,
                save_strategy="no",
                fp16=True,
            )
            trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=train_loss)
            trainer.train()

        With a dataset with single input pairs:

        ::

            import random
            from datasets import Dataset
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.losses import ContrastiveTensionLoss
            from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
            from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer

            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            train_dataset = Dataset.from_dict({
                "text1": [
                    "It's nice weather outside today.",
                    "He drove to work.",
                    "It's so sunny.",
                ]
            })
            sentences = train_dataset['text1']

            def to_ct_pairs(sample, pos_neg_ratio=8):
                pos_neg_ratio = 1 / pos_neg_ratio
                sample["text2"] = sample["text1"] if random.random() < pos_neg_ratio else random.choice(sentences)
                return sample

            pos_neg_ratio = 8  # 1 positive pair for 7 negative pairs
            train_dataset = train_dataset.map(to_ct_pairs, fn_kwargs={"pos_neg_ratio": pos_neg_ratio})

            train_loss = ContrastiveTensionLoss(model=model)

            args = SentenceTransformerTrainingArguments(
                num_train_epochs=10,
                per_device_train_batch_size=32,
                eval_steps=0.1,
                logging_steps=0.01,
                learning_rate=5e-5,
                save_strategy="no",
                fp16=True,
            )
            trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=train_loss)
            trainer.train()
    """

    def __init__(self, model: SentenceTransformer) -> None:
        # [FUNCTION DOC] Store the model as model2, deep-copy it into model1 (doubling parameter memory), and create the summed BCE criterion.
        super().__init__()
        self.model2 = model  # This will be the final model used during the inference time.
        self.model1 = copy.deepcopy(model)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor:
        # [FUNCTION DOC] Unpack exactly two feature dicts (ValueError otherwise). Embed the first with model1 and the second with
        # model2; the logit per pair is their dot product. When labels are not given, derive them: 1 if the two
        # inputs' token ids are identical (comparing only non-padded tokens when both attention masks exist, else
        # comparing full id rows), otherwise 0. Returns the summed BCE-with-logits loss.
        sentence_features1, sentence_features2 = tuple(sentence_features)
        reps_1 = self.model1(sentence_features1)["sentence_embedding"]  # (bsz, hdim)
        reps_2 = self.model2(sentence_features2)["sentence_embedding"]

        sim_scores = (
            torch.matmul(reps_1[:, None], reps_2[:, :, None]).squeeze(-1).squeeze(-1)
        )  # (bsz,) dot product, i.e. S1S2^T

        if labels is None:
            # identical sentence -> 1, different -> 0
            input_ids1 = sentence_features1["input_ids"]
            input_ids2 = sentence_features2["input_ids"]
            attention_mask1 = sentence_features1.get("attention_mask")
            attention_mask2 = sentence_features2.get("attention_mask")

            if attention_mask1 is not None and attention_mask2 is not None:
                # Compare only non-padded tokens so sentences are considered identical even if they differ by left/right padding.
                labels = torch.as_tensor(
                    [
                        torch.equal(inputs1[mask1], inputs2[mask2])
                        for inputs1, mask1, inputs2, mask2 in zip(
                            input_ids1, attention_mask1.bool(), input_ids2, attention_mask2.bool()
                        )
                    ],
                    device=sim_scores.device,
                    dtype=sim_scores.dtype,
                )
            else:
                labels = (input_ids1 == input_ids2).all(dim=1).float()

        loss = self.criterion(sim_scores, labels.type_as(sim_scores))
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return an empty config dict.
        return {}

    @property
    def citation(self) -> str:
        # [FUNCTION DOC] Return the BibTeX citation for Semantic Re-tuning with Contrastive Tension.
        return """
@inproceedings{carlsson2021semantic,
    title={Semantic Re-tuning with Contrastive Tension},
    author={Fredrik Carlsson and Amaru Cuba Gyllensten and Evangelia Gogoulou and Erik Ylip{\"a}{\"a} Hellqvist and Magnus Sahlgren},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Ov_sMNau-PF}
}
"""


class ContrastiveTensionLossInBatchNegatives(nn.Module):
    # [CLASS DOC] In-batch negatives CT loss. Attributes: model2 (original model), model1 (deep copy), similarity_fct (default
    # cosine), cross_entropy_loss, logit_scale (learnable parameter initialized to the log of scale).
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=cos_sim) -> None:
        # [FUNCTION DOC] Store the model, deep-copy it, store the similarity function and create the learnable log-scale parameter.
        """
        This loss expects only single inputs, without any labels. Positive and negative pairs are automatically created via random sampling,
        such that a positive pair consists of two identical inputs and a negative pair consists of two different inputs. An independent
        copy of the encoder model is created, which is used for encoding the first input of each pair. The original encoder model encodes the
        second input. Unlike :class:`ContrastiveTensionLoss`, this loss uses the batch negative sampling strategy, i.e. the negative pairs
        are sampled from the batch. Using in-batch negative sampling gives a stronger training signal than the original :class:`ContrastiveTensionLoss`.
        The performance usually increases with increasing batch sizes.

        The training :class:`datasets.Dataset` must contain one text column.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        References:
            - Semantic Re-Tuning with Contrastive Tension: https://openreview.net/pdf?id=Ov_sMNau-PF
            - `Unsupervised Learning > CT (In-Batch Negatives) <../../../examples/sentence_transformer/unsupervised_learning/CT_In-Batch_Negatives/README.html>`_

        Relations:
            * :class:`ContrastiveTensionLoss` does not select negative pairs in-batch, resulting in a weaker training signal than this loss.

        Inputs:
            +------------------+--------+
            | Inputs           | Labels |
            +==================+========+
            | single inputs    | none   |
            +------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses
                from datasets import Dataset

                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

                train_dataset = Dataset.from_dict({
                    "sentence": [
                       "It's nice weather outside today.",
                       "He drove to work.",
                       "It's so sunny.",
                    ]
                })

                train_loss = losses.ContrastiveTensionLossInBatchNegatives(model=model)

                args = SentenceTransformerTrainingArguments(
                    num_train_epochs=10,
                    per_device_train_batch_size=32,
                    eval_steps=0.1,
                    logging_steps=0.01,
                    learning_rate=5e-5,
                    save_strategy="no",
                    fp16=True,
                )

                trainer = SentenceTransformerTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    loss=train_loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model2 = model  # This will be the final model used during the inference time.
        self.model1 = copy.deepcopy(model)
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(scale))

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # [FUNCTION DOC] Use only the first input column: embed it with both encoders, compute the full similarity matrix times the
        # exponentiated learnable scale, and return the average of cross-entropy over rows and over columns with the
        # diagonal as targets. The passed labels are ignored.
        sentence_features = sentence_features[0]
        embeddings_a = self.model1(sentence_features)["sentence_embedding"]  # (bsz, hdim)
        embeddings_b = self.model2(sentence_features)["sentence_embedding"]

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.logit_scale.exp()  # self.scale
        labels = torch.arange(len(scores), dtype=torch.long, device=scores.device)
        return (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(scores.t(), labels)) / 2

    def get_config_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the config: current scale (exponentiated parameter) and similarity function name.
        return {
            "scale": self.logit_scale.exp().item(),
            "similarity_fct": similarity_fct_name(self.similarity_fct),
        }

    @property
    def citation(self) -> str:
        # [FUNCTION DOC] Return the BibTeX citation for Semantic Re-tuning with Contrastive Tension.
        return """
@inproceedings{carlsson2021semantic,
    title={Semantic Re-tuning with Contrastive Tension},
    author={Fredrik Carlsson and Amaru Cuba Gyllensten and Evangelia Gogoulou and Erik Ylip{\"a}{\"a} Hellqvist and Magnus Sahlgren},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Ov_sMNau-PF}
}
"""


# CT Data Loader
# For CT, we need batches in a specific format
# In each batch, we have one positive pair (i.e. [sentA, sentA]) and 7 negative pairs (i.e. [sentA, sentB]).
# To achieve this, we create a custom DataLoader that produces batches with this property


class ContrastiveTensionDataLoader:
    # [CLASS DOC] Legacy iterable data loader yielding batches of CT InputExample pairs where every pos_neg_ratio-th example is
    # an identical (positive, label 1) pair and the rest are pairs of consecutive different sentences (label 0).
    # Attributes: sentences, batch_size, pos_neg_ratio, collate_fn (set externally; None yields raw lists).
    def __init__(self, sentences, batch_size, pos_neg_ratio=8):
        # [FUNCTION DOC] Store settings. Raises ValueError if batch_size is not divisible by pos_neg_ratio.
        self.sentences = sentences
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        self.collate_fn = None

        if self.batch_size % self.pos_neg_ratio != 0:
            raise ValueError(
                f"ContrastiveTensionDataLoader was loaded with a pos_neg_ratio of {pos_neg_ratio} and a batch size of {batch_size}. The batch size must be divisible by the pos_neg_ratio"
            )

    def __iter__(self):
        # [FUNCTION DOC] Shuffle the sentence list in place (global random), then walk it: at positions where the batch length is a
        # multiple of pos_neg_ratio, pair the current sentence with itself (label 1) and advance by one; otherwise pair
        # it with the next sentence (label 0) and advance by two. Yield each full batch (collated if collate_fn is set);
        # a trailing partial batch is dropped.
        random.shuffle(self.sentences)
        sentence_idx = 0
        batch = []

        while sentence_idx + 1 < len(self.sentences):
            s1 = self.sentences[sentence_idx]
            if len(batch) % self.pos_neg_ratio > 0:  # Negative (different) pair
                sentence_idx += 1
                s2 = self.sentences[sentence_idx]
                label = 0
            else:  # Positive (identical pair)
                s2 = self.sentences[sentence_idx]
                label = 1

            sentence_idx += 1
            batch.append(InputExample(texts=[s1, s2], label=label))

            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch) if self.collate_fn is not None else batch
                batch = []

    def __len__(self):
        # [FUNCTION DOC] Return the approximate number of batches: floor of sentence count divided by twice the batch size.
        return math.floor(len(self.sentences) / (2 * self.batch_size))
