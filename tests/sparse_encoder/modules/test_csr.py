# [MODULE DOC]
# Tests for CSR (Contrastive Sparse Representation) sparse encoder models: which output keys the forward pass
# produces in inference versus training mode.
from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from sentence_transformers import SparseEncoder


# Create a wrapper to measure outputs of the forward method
class ForwardMethodWrapper:
    # [CLASS DOC]
    # HELPER: Callable wrapper that replaces a model's forward method to record outputs.
    # Attributes: the wrapped model, its original forward method, an is_inference flag, and a list of recorded
    # outputs.
    def __init__(self, model, is_inference: bool = True):
        # [FUNCTION DOC]
        # HELPER: Stores the model, captures its current forward method, stores the inference flag (default True) and
        # initialises an empty outputs list.
        self.model = model
        self.original_forward = model.forward
        self.is_inference = is_inference
        self.outputs = []

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Calls the original forward with all given arguments, inside torch inference mode when is_inference is
        # set (otherwise with no special context), appends the output to the outputs list, and returns it.
        # Set the model to training mode if is_train is True
        with torch.inference_mode() if self.is_inference else nullcontext():
            output = self.original_forward(*args, **kwargs)
        self.outputs.append(output)
        return output

    def reset(self):
        # [FUNCTION DOC]
        # HELPER: Clears the recorded outputs list.
        self.outputs = []


@pytest.mark.parametrize(
    ["is_inference", "expected_keys"],
    [
        (
            False,
            {
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "token_embeddings",
                "sentence_embedding",
                "sentence_embedding_backbone",
                "sentence_embedding_encoded",
                "sentence_embedding_encoded_4k",
                "auxiliary_embedding",
                "decoded_embedding_k",
                "decoded_embedding_4k",
                "decoded_embedding_aux",
                "decoded_embedding_k_pre_bias",
                "modality",
            },
        ),
        (
            True,
            {"input_ids", "attention_mask", "token_type_ids", "token_embeddings", "sentence_embedding", "modality"},
        ),
    ],
)
def test_csr_outputs(csr_bert_tiny_model: SparseEncoder, is_inference: bool, expected_keys: set) -> None:
    # [FUNCTION DOC]
    # Parametrized over inference mode and the expected set of output keys (training mode additionally exposes the
    # backbone, encoded, auxiliary and decoded embeddings). Replaces the model's forward with the wrapper,
    # preprocesses one sentence, moves tensor inputs to the model device, calls the model, and asserts the first
    # recorded output's keys equal the expected set. Mutates the fixture model's forward permanently.
    model = csr_bert_tiny_model

    # Create the wrapper and replace the forward method
    wrapper = ForwardMethodWrapper(model, is_inference=is_inference)
    model.forward = wrapper

    # Run the encode method which should call forward internally
    inputs = model.preprocess(["This is a test sentence."])
    inputs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
    }
    model(inputs)

    # Check that the model was called in the correct mode, and that the outputs contain the expected keys
    assert set(wrapper.outputs[0].keys()) == expected_keys
    # We don't have to restore the original forward method, as the model will not be reused
