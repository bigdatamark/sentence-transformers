# [MODULE DOC] Trainer for MultiVectorEncoder (ColBERT-style) models. A thin subclass of the shared BaseTrainer that
# only plugs in multi-vector-specific default classes (model, model card data/callback, data collator,
# training arguments) and the default loss factory.
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from transformers import (
    EvalPrediction,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.image_processing_utils import BaseImageProcessor

from sentence_transformers.base.evaluation import BaseEvaluator
from sentence_transformers.base.trainer import BaseTrainer
from sentence_transformers.multi_vector_encoder.data_collator import MultiVectorEncoderDataCollator
from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.model_card import (
    MultiVectorEncoderModelCardCallback,
    MultiVectorEncoderModelCardData,
)
from sentence_transformers.multi_vector_encoder.training_args import MultiVectorEncoderTrainingArguments
from sentence_transformers.util import is_datasets_available
from sentence_transformers.util.decorators import deprecated_kwargs

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset

logger = logging.getLogger(__name__)


class MultiVectorEncoderTrainer(BaseTrainer):
    # [CLASS DOC] Trainer specialization for MultiVectorEncoder. Class attributes name the model class, model card data
    # class, model card callback class, data collator class and training arguments class that BaseTrainer
    # uses when constructing defaults. All training behavior lives in BaseTrainer.
    """Trainer for :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` (multi-vector / ColBERT-style) models.

    Inherits all functionality from :class:`~sentence_transformers.base.trainer.BaseTrainer`: dataset-name-based
    loss dispatch, prompt threading, model-card callback, mixed precision, gradient checkpointing, multi-GPU
    via accelerate, etc. Overrides only the default classes and the default-loss factory.

    Args:
        model: The :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` to train.
        args: Training arguments.
        train_dataset: Training dataset. Standard ST formats are supported (pair / triplet / multi-negative),
            plus the knowledge-distillation format ``(query, document_1, ..., document_N, scores)`` where
            ``scores`` is a list of N teacher scores per row. You can use
            :func:`~sentence_transformers.util.dataset.resolve_ids` to build the numbered columns from
            ID-only KD datasets.
        eval_dataset: Evaluation dataset.
        loss: A loss class, dict of dataset-name → loss, callable returning a loss, or dict of callables.
            Defaults to :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorMultipleNegativesRankingLoss`.
        evaluator: A :class:`~sentence_transformers.base.evaluation.BaseEvaluator` (or list of them) for
            mid-training evaluation.
        callbacks: Extra :class:`~transformers.TrainerCallback` instances.
        optimizers: ``(optimizer, scheduler)`` tuple.
    """

    model_class = MultiVectorEncoder
    model_card_data_class = MultiVectorEncoderModelCardData
    model_card_callback_class = MultiVectorEncoderModelCardCallback
    data_collator_class = MultiVectorEncoderDataCollator
    training_args_class = MultiVectorEncoderTrainingArguments

    @deprecated_kwargs(tokenizer="processing_class")
    def __init__(
        self,
        model: MultiVectorEncoder | None = None,
        args: MultiVectorEncoderTrainingArguments | None = None,
        train_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        eval_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        loss: nn.Module
        | dict[str, nn.Module]
        | Callable[[MultiVectorEncoder], torch.nn.Module]
        | dict[str, Callable[[MultiVectorEncoder], torch.nn.Module]]
        | None = None,
        evaluator: BaseEvaluator | list[BaseEvaluator] | None = None,
        data_collator: MultiVectorEncoderDataCollator | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], MultiVectorEncoder] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        # [FUNCTION DOC] Construct the trainer by forwarding every argument unchanged to BaseTrainer (model, args, datasets, loss,
        # evaluator, data collator, processing class, model_init, compute_metrics, callbacks, optimizers, optimizer
        # class/kwargs, logits preprocessing). The deprecated keyword "tokenizer" is remapped to
        # "processing_class" by a decorator. Afterwards only annotates the narrower types of model, args and
        # data_collator. Side effects: those of BaseTrainer construction.
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
            data_collator=data_collator,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.model: MultiVectorEncoder
        self.args: MultiVectorEncoderTrainingArguments
        self.data_collator: MultiVectorEncoderDataCollator

    def get_default_loss(self, model: MultiVectorEncoder) -> torch.nn.Module:
        # [FUNCTION DOC] Return the default loss used when none is passed: logs an info message saying the multi-vector
        # multiple-negatives ranking loss is being used, then lazily imports and returns a new instance of that
        # loss bound to the given model.
        from sentence_transformers.multi_vector_encoder.losses import (
            MultiVectorMultipleNegativesRankingLoss,
        )

        logger.info(
            "No `loss` passed, using `MultiVectorMultipleNegativesRankingLoss` as the default. "
            "Provide a different loss via the `loss=` argument if you want to train with knowledge distillation, "
            "margin-MSE, or a cached / large-batch InfoNCE objective."
        )
        return MultiVectorMultipleNegativesRankingLoss(model=model)
