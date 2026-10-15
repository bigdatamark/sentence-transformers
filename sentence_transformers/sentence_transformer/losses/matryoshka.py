# [MODULE DOC] Matryoshka loss modifier: trains embeddings so that their leading dimensions are useful on their own by applying
# a wrapped loss at several truncation sizes and summing weighted results. For ordinary losses the model forward is
# temporarily replaced by a caching decorator; for gradient-caching losses their internal loss computation is
# wrapped instead.
from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.base.losses.gradcache import uses_gradient_cache
from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


def shrink(tensor: Tensor, dim: int) -> Tensor:
    # [FUNCTION DOC] HELPER: Truncate the last axis of a tensor to the first dim entries and L2-normalize along that axis. Raises
    # ValueError if dim exceeds the tensor's last-axis size.
    tensor_dim = tensor.shape[-1]
    if dim > tensor_dim:
        raise ValueError(
            f"Dimension {dim} in matryoshka_dims cannot be greater than the model's embedding dimension: {tensor_dim}"
        )
    tensor = tensor[..., :dim]
    tensor = F.normalize(tensor, p=2, dim=-1)
    return tensor


class ForwardDecorator:
    # [CLASS DOC] HELPER: Callable replacing the model's forward. Attributes: fn (original forward), dim (current truncation size),
    # cache (list of forward outputs from the first, largest dimension), cache_dim (dimension the cache was filled at),
    # idx (position in the cache for the current dimension's pass).
    """
    This decorator is used to cache the output of the Sentence Transformer's forward pass,
    so that it can be shrank and reused for multiple loss calculations. This prevents the
    model from recalculating the embeddings for each desired Matryoshka dimensionality.

    This decorator is applied to `SentenceTransformer.forward`.
    """

    def __init__(self, fn) -> None:
        # [FUNCTION DOC] Store the original forward and initialize an empty cache with no dimension.
        self.fn = fn

        self.dim = None
        self.cache = []
        self.cache_dim = None
        self.idx = 0

    def set_dim(self, dim) -> None:
        # [FUNCTION DOC] Set the current truncation dimension and reset the cache position to 0.
        self.dim = dim
        self.idx = 0

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        # [FUNCTION DOC] While the cache is empty or still at the dimension that filled it, run the real forward and append the output;
        # otherwise reuse the cached output at the current position. Truncate and normalize token embeddings (if present)
        # and the sentence embedding in place in the output dict (so the cache holds the last truncation, which is fine
        # because dimensions are processed from largest to smallest). Advance the position and return the output.
        # Growing cache:
        if self.cache_dim is None or self.cache_dim == self.dim:
            output = self.fn(features)
            self.cache.append(output)
            self.cache_dim = self.dim
        # Using cache:
        else:
            output = self.cache[self.idx]
        if "token_embeddings" in output:
            output["token_embeddings"] = shrink(output["token_embeddings"], self.dim)
        output["sentence_embedding"] = shrink(output["sentence_embedding"], self.dim)
        self.idx += 1
        return output


class CachedLossDecorator:
    # [CLASS DOC] HELPER: Wrapper around a gradient-caching loss's loss computation that evaluates it once per Matryoshka
    # dimension. Attributes: fn (original loss computation), matryoshka_dims, matryoshka_weights, n_dims_per_step.
    """
    This decorator is used with the Cached... losses to compute the underlying loss function
    for each Matryoshka dimensionality. This is done by shrinking the pre-computed embeddings
    to the desired dimensionality and then passing them to the underlying loss function once
    for each desired dimensionality.

    This decorator is applied to the `calculate_loss` method of the Cached... losses.
    """

    def __init__(
        self,
        fn,
        matryoshka_dims: Sequence[int],
        matryoshka_weights: Sequence[float] | Sequence[int],
        n_dims_per_step: int = -1,
    ) -> None:
        # [FUNCTION DOC] Store the wrapped function, dimensions, weights and per-step sample size.
        self.fn = fn
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights
        self.n_dims_per_step = n_dims_per_step

    def __call__(self, reps: list[list[Tensor]], *args, **kwargs) -> Tensor:
        # [FUNCTION DOC] Choose all dimension indices, or a random sample of n_dims_per_step of them (global random) when that is
        # positive and smaller. For each chosen dimension, truncate and normalize every cached embedding chunk; if
        # gradients are enabled, pass detached leaf copies to the wrapped function (so its internal backward only reaches
        # them) and then back-propagate each copy's gradient times the weight through the truncation to the original
        # chunk. Accumulates and returns the weighted sum of losses. Extra arguments are forwarded.
        dim_indices = range(len(self.matryoshka_dims))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = random.sample(dim_indices, self.n_dims_per_step)

        loss = 0.0
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]

            truncated = [[shrink(r, dim) for r in minibatch] for minibatch in reps]
            compute_gradients = torch.is_grad_enabled()
            # we need to detach the truncated embeddings,
            # otherwise the first backward pass of the underlying function will clear the computation graph of the embedding truncation
            if compute_gradients:
                matryoshka_reps = [[r.detach().requires_grad_() for r in minibatch] for minibatch in truncated]
            else:
                matryoshka_reps = truncated
            loss += weight * self.fn(matryoshka_reps, *args, **kwargs)
            # After computing the gradients in minibatches, we need to continue the backward pass through the truncation calculation
            # the gradients must be multiplied with the weights because otherwise the matryoshka weights are not considered in the backward pass
            if compute_gradients:
                for t_minibatch, d_minibatch in zip(truncated, matryoshka_reps):
                    for t, d in zip(t_minibatch, d_minibatch):
                        t.backward(weight * d.grad)
        return loss


class MatryoshkaLoss(nn.Module):
    # [CLASS DOC] Loss module wrapping another loss. Attributes: model, loss, matryoshka_dims and matryoshka_weights (tuples sorted
    # together by descending dimension), n_dims_per_step, uses_gradient_cache (whether the wrapped loss caches gradients).
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        matryoshka_dims: Sequence[int],
        matryoshka_weights: Sequence[float] | Sequence[int] | None = None,
        n_dims_per_step: int = -1,
    ) -> None:
        # [FUNCTION DOC] Validate: raises ValueError for empty dims, non-positive dims, weights of mismatched length, or dims exceeding the
        # model's embedding dimension (when known); warns if the full dimension is not included. Weights default to 1.
        # Sorts dims with their weights in descending order. If the wrapped loss uses gradient caching, replaces its
        # loss computation method with the Matryoshka wrapper (mutates the wrapped loss).
        """
        The MatryoshkaLoss can be seen as a loss *modifier* that allows you to use other loss functions at various
        different embedding dimensions. This is useful for when you want to train a model where users have the option
        to lower the embedding dimension to improve their embedding comparison speed and costs.

        This loss is also compatible with the Cached... losses, which are in-batch negative losses that allow for
        higher batch sizes. The higher batch sizes allow for more negatives, and often result in a stronger model.

        Args:
            model: SentenceTransformer model
            loss: The loss function to be used, e.g.
                :class:`MultipleNegativesRankingLoss`,
                :class:`CoSENTLoss`, etc.
            matryoshka_dims: A list of embedding dimensions to be used
                for the loss function, e.g. [768, 512, 256, 128, 64].
            matryoshka_weights: A list of weights to be used for the
                loss function, e.g. [1, 1, 1, 1, 1]. If None, then the
                weights will be set to 1 for all dimensions.
            n_dims_per_step: The number of dimensions to use per step.
                If -1, then all dimensions are used. If > 0, then a
                random sample of n_dims_per_step dimensions are used per
                step. The default value is -1.

        References:
            - The concept was introduced in this paper: https://huggingface.co/papers/2205.13147
            - `Matryoshka Embeddings <../../../examples/sentence_transformer/training/matryoshka/README.html>`_

        Inputs:
            +---------------------------------------+--------+
            | Inputs                                | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

        Relations:
            - :class:`Matryoshka2dLoss` uses this loss in combination with :class:`AdaptiveLayerLoss` which allows for
                layer reduction for faster inference.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)
                loss = losses.MatryoshkaLoss(model, loss, [768, 512, 256, 128, 64])

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.loss = loss

        if not matryoshka_dims:
            raise ValueError("You must provide at least one dimension in matryoshka_dims.")
        if any(dim <= 0 for dim in matryoshka_dims):
            raise ValueError("All dimensions passed to a matryoshka loss must be > 0.")
        if matryoshka_weights is None:
            matryoshka_weights = [1] * len(matryoshka_dims)
        elif len(matryoshka_weights) != len(matryoshka_dims):
            raise ValueError("matryoshka_weights must be the same length as matryoshka_dims.")

        model_embedding_dim = model.get_embedding_dimension()
        if model_embedding_dim is not None:
            if any(d > model_embedding_dim for d in matryoshka_dims):
                raise ValueError(
                    f"Dimensions in matryoshka_dims cannot exceed the model's embedding dimension of {model_embedding_dim}."
                )
            if model_embedding_dim not in matryoshka_dims:
                logger.warning(
                    f"The model's embedding dimension {model_embedding_dim} is not included in matryoshka_dims: {matryoshka_dims}. "
                    "This means that the full model dimension won't be trained, which may lead to degraded performance "
                    "when using the model without specifying a lower truncation dimension. It is strongly recommended to include "
                    f"{model_embedding_dim} in matryoshka_dims."
                )

        # Sort the dimensions and weights in descending order
        dims_weights = zip(matryoshka_dims, matryoshka_weights)
        self.matryoshka_dims: tuple[int, ...]
        self.matryoshka_weights: tuple[float, ...] | tuple[int, ...]
        self.matryoshka_dims, self.matryoshka_weights = zip(*sorted(dims_weights, key=lambda x: x[0], reverse=True))
        self.n_dims_per_step = n_dims_per_step

        # Gradient-caching losses back-propagate from a hook that fires after the ForwardDecorator
        # is removed, so decorate their `calculate_loss` to shrink the cached embeddings instead.
        self.uses_gradient_cache = uses_gradient_cache(loss)
        if self.uses_gradient_cache:
            loss.calculate_loss = CachedLossDecorator(
                loss.calculate_loss, self.matryoshka_dims, self.matryoshka_weights, self.n_dims_per_step
            )

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # [FUNCTION DOC] For gradient-caching losses, simply call the wrapped loss (the wrapper handles dimensions). Otherwise install the
        # caching forward decorator on the model, pick all or a random (then sorted) subset of dimensions, and for each set
        # the decorator dimension, truncate 2-D tensor labels whose width equals the model's embedding dimension to that
        # dimension (for embedding distillation), and add weight times the wrapped loss. Always restores the original
        # forward. Returns the summed loss.
        # KNOWN ISSUE: When labels are embeddings, the student embeddings are truncated and re-normalized but the label embeddings are only truncated, so distillation targets at smaller dimensions are not normalized consistently with the predictions.
        # For the gradient-caching losses, the CachedLossDecorator has been applied to the `calculate_loss`
        # method, so we can directly call the loss function.
        if self.uses_gradient_cache:
            return self.loss(sentence_features, labels)

        # Otherwise, we apply the ForwardDecorator to the model's forward pass, which will cache the output
        # embeddings for each Matryoshka dimensionality, allowing it to be reused for the smaller dimensions.
        original_forward = self.model.forward
        try:
            decorated_forward = ForwardDecorator(original_forward)
            self.model.forward = decorated_forward

            dim_indices = range(len(self.matryoshka_dims))
            if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
                dim_indices = random.sample(dim_indices, self.n_dims_per_step)
                dim_indices.sort()

            loss = 0.0
            for idx in dim_indices:
                dim = self.matryoshka_dims[idx]
                weight = self.matryoshka_weights[idx]
                decorated_forward.set_dim(dim)
                # If the labels seem to be embeddings, truncate them to match the soon-to-be-truncated predicted embeddings
                # This allows for MatryoshkaLoss with a direct distillation loss
                dim_labels = labels
                if (
                    isinstance(labels, torch.Tensor)
                    and labels.ndim == 2
                    and labels.size(-1) == self.model.get_embedding_dimension()
                ):
                    dim_labels = labels[:, :dim]

                loss += weight * self.loss(sentence_features, dim_labels)
        finally:
            self.model.forward = original_forward
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the config: wrapped loss class name, dims, weights and n_dims_per_step.
        return {
            "loss": self.loss.__class__.__name__,
            "matryoshka_dims": self.matryoshka_dims,
            "matryoshka_weights": self.matryoshka_weights,
            "n_dims_per_step": self.n_dims_per_step,
        }

    @property
    def citation(self) -> str:
        # [FUNCTION DOC] Return the BibTeX citation for Matryoshka Representation Learning.
        return """
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
