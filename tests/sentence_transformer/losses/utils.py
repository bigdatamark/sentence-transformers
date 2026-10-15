# [MODULE DOC]
# HELPER: Shared utilities for loss tests that compare gradients between equivalent loss implementations: disabling dropout
# deterministically, collecting parameter gradients, and checking that gradients are non-trivial.
from __future__ import annotations

import torch

from sentence_transformers.base.model import BaseModel
from sentence_transformers.util import batch_to_device


def disable_dropout(model: BaseModel, canary_inputs: list | None = None) -> None:
    # [FUNCTION DOC]
    # HELPER: Makes a model deterministic in train mode. For every submodule, sets the rate of each dropout layer to zero and zeroes any
    # float attribute named dropout_prob, attention_dropout or attention_probs_dropout_prob (covers older transformers SDPA attention).
    # Then verifies: remembers the training flag, switches to train mode, preprocesses canary_inputs (or two default sentences) onto the
    # model device, runs two forward passes without gradients, reads "sentence_embedding" if present otherwise "scores", restores the
    # training flag, and asserts both outputs are identical. Inputs: any BaseModel; optional canary inputs (pairs for CrossEncoder).
    # Returns nothing; mutates the model in place; raises AssertionError if dropout is still active.
    """Disable every dropout, so that gradient-equivalence tests are deterministic.

    transformers<5 keeps the SDPA attention dropout as a plain float attribute rather than an ``nn.Dropout``
    module, so zeroing only the modules would leave attention dropout active there.

    Works for SentenceTransformer, SparseEncoder, and CrossEncoder models. CrossEncoder callers must
    pass pair-shaped ``canary_inputs``, e.g. ``[["a query", "an answer"]] * 2``.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for attribute in ("dropout_prob", "attention_dropout", "attention_probs_dropout_prob"):
            if isinstance(getattr(module, attribute, None), float):
                setattr(module, attribute, 0.0)

    # If a future transformers release keeps the rate somewhere this doesn't look, the equivalence tests
    # would start failing as a confusing "gradients do not match", pointing at the loss rather than here.
    was_training = model.training
    model.train()
    features = batch_to_device(
        model.preprocess(canary_inputs or ["a canary sentence", "another canary sentence"]), model.device
    )
    with torch.no_grad():
        outputs = model(features)
        key = "sentence_embedding" if "sentence_embedding" in outputs else "scores"
        first, second = outputs[key], model(features)[key]
    model.train(was_training)
    assert torch.equal(first, second), "dropout is still active in training mode; disable_dropout needs updating"


def gradients(model: BaseModel) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Returns a dict from parameter name to a clone of its gradient, for every parameter whose gradient is not None.
    return {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}


def assert_trained(grads: dict[str, torch.Tensor]) -> None:
    # [FUNCTION DOC]
    # HELPER: Asserts the gradient dict is non-empty and that the sum of absolute values over all gradients is positive, so
    # gradient-equality checks are not vacuous. Raises AssertionError otherwise.
    """Guard against a vacuous comparison: an empty or all-zero gradient dict passes any assert_close loop."""
    assert grads, "no parameter received a gradient at all"
    assert sum(grad.abs().sum() for grad in grads.values()) > 0, "every gradient is zero"
