# [MODULE DOC]
# Tests MatryoshkaLoss construction-time validation (dimensions non-empty, positive, not exceeding the model dimension, weights matching
# in length), the warning when the full model dimension is omitted, sorting of dimensions (with weights reordered alongside), default
# weights, and a forward pass wrapping MSELoss. Uses the static-retrieval-mrl-en-v1 model (1024-D) fixture.
from __future__ import annotations

import warnings

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss, MSELoss


@pytest.fixture
def mock_loss():
    # [FUNCTION DOC]
    # HELPER: Fixture returning a trivial base loss module whose forward ignores inputs and returns a scalar zero tensor.
    """Create a mock base loss for testing."""

    class _FakeLoss(torch.nn.Module):
        # [CLASS DOC]
        # HELPER: Fake loss module returning a constant zero.
        def forward(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns a scalar tensor 0.0 regardless of arguments.
            return torch.tensor(0.0)

    return _FakeLoss()


def test_empty_matryoshka_dims_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts an empty dimension list raises ValueError requiring at least one dimension.
    """Test that empty matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="You must provide at least one dimension in matryoshka_dims."):
        MatryoshkaLoss(model=static_retrieval_mrl_en_v1_model, loss=mock_loss, matryoshka_dims=[])


def test_zero_dimension_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts a zero among the dimensions raises ValueError requiring all dimensions to be positive.
    """Test that zero dimension in matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="All dimensions passed to a matryoshka loss must be > 0."):
        MatryoshkaLoss(model=static_retrieval_mrl_en_v1_model, loss=mock_loss, matryoshka_dims=[512, 0, 256])


def test_negative_dimension_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts a negative dimension raises the same positivity ValueError.
    """Test that negative dimension in matryoshka_dims raises ValueError."""
    with pytest.raises(ValueError, match="All dimensions passed to a matryoshka loss must be > 0."):
        MatryoshkaLoss(model=static_retrieval_mrl_en_v1_model, loss=mock_loss, matryoshka_dims=[512, -100, 256])


def test_weights_length_mismatch_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts three dimensions with two weights raises ValueError requiring equal lengths.
    """Test that length mismatch between weights and dims raises ValueError."""
    with pytest.raises(ValueError, match="matryoshka_weights must be the same length as matryoshka_dims."):
        MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[512, 256, 128],
            matryoshka_weights=[1.0, 0.5],  # Only 2 weights for 3 dims
        )


def test_dimension_exceeds_model_dimension_raises_error(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts a dimension of 2048 (above the model's 1024) raises ValueError naming the model's embedding dimension.
    """Test that dimension exceeding model's embedding dimension raises ValueError."""
    expected_msg = "Dimensions in matryoshka_dims cannot exceed the model's embedding dimension of 1024."
    with pytest.raises(ValueError, match=expected_msg):
        MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[512, 1024, 2048, 256],  # 2048 > 1024
        )


def test_model_dimension_not_in_dims_warns(static_retrieval_mrl_en_v1_model, mock_loss, caplog):
    # [FUNCTION DOC]
    # Asserts that omitting 1024 from the dimensions logs exactly one WARNING record with the exact recommendation message. Uses caplog.
    """Test that model dimension not in matryoshka_dims produces warning."""
    with caplog.at_level("WARNING"):
        MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[512, 256, 128],  # 1024 not included
        )

    # Check that warning was logged
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    expected_msg = (
        "The model's embedding dimension 1024 is not included in matryoshka_dims: [512, 256, 128]. "
        "This means that the full model dimension won't be trained, which may lead to degraded performance "
        "when using the model without specifying a lower truncation dimension. It is strongly recommended to include "
        "1024 in matryoshka_dims."
    )
    assert caplog.records[0].message == expected_msg


def test_model_dimension_none_no_validation(mock_loss):
    # [FUNCTION DOC]
    # With a mock model whose embedding dimension is None, asserts construction with large dimensions neither raises nor emits any warning.
    """Test that when model dimension is None, no validation or warning occurs."""

    class _MockModel:
        # [CLASS DOC]
        # HELPER: Mock model reporting an unknown (None) embedding dimension.
        def get_embedding_dimension(self):
            # [FUNCTION DOC]
            # HELPER: Returns None.
            return None

    model = _MockModel()

    # This should not raise any error or warning, even with large dimensions
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        MatryoshkaLoss(
            model=model,
            loss=mock_loss,
            matryoshka_dims=[2048, 1024, 512],  # Large dimensions, but no validation
        )


def test_valid_initialization_no_warnings(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts a valid dimension list that includes 1024 constructs without any warning (warnings are turned into errors).
    """Test that valid initialization with model dimension included produces no warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[1024, 768, 512, 256, 128],  # 1024 included
        )


def test_valid_initialization_with_weights(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts valid dims and weights construct without warnings and are stored as tuples in the given (already descending) order.
    """Test that valid initialization with custom weights works correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        loss = MatryoshkaLoss(
            model=static_retrieval_mrl_en_v1_model,
            loss=mock_loss,
            matryoshka_dims=[1024, 768, 512, 256],
            matryoshka_weights=[1.0, 0.8, 0.6, 0.4],
        )

        # Verify that dimensions and weights are sorted in descending order
        assert loss.matryoshka_dims == (1024, 768, 512, 256)
        assert loss.matryoshka_weights == (1.0, 0.8, 0.6, 0.4)


def test_dimensions_sorted_descending(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts unsorted dims are stored sorted descending as a tuple, with each weight moved along with its dimension (so weights follow
    # the dims order rather than being sorted themselves).
    """Test that dimensions and weights are sorted in descending order."""
    loss = MatryoshkaLoss(
        model=static_retrieval_mrl_en_v1_model,
        loss=mock_loss,
        matryoshka_dims=[256, 768, 512, 1024],  # Unsorted
        matryoshka_weights=[0.6, 1.0, 0.8, 0.4],  # Corresponding weights
    )

    # Should be sorted in descending order
    assert loss.matryoshka_dims == (1024, 768, 512, 256)
    assert loss.matryoshka_weights == (0.4, 1.0, 0.8, 0.6)


def test_default_weights_when_none(static_retrieval_mrl_en_v1_model, mock_loss):
    # [FUNCTION DOC]
    # Asserts that weights default to a tuple of ones, one per dimension, when None is given.
    """Test that default weights (all 1s) are used when weights is None."""
    loss = MatryoshkaLoss(
        model=static_retrieval_mrl_en_v1_model,
        loss=mock_loss,
        matryoshka_dims=[1024, 768, 512, 256],
        matryoshka_weights=None,
    )

    assert loss.matryoshka_weights == (1, 1, 1, 1)


def test_mse_loss_matryoshka(static_retrieval_mrl_en_v1_model):
    # [FUNCTION DOC]
    # Wraps MSELoss in MatryoshkaLoss with seven dimensions on a CPU model, feeds one preprocessed sentence with a random 1x1024 teacher
    # embedding as labels, and asserts the result is a scalar tensor.
    """Test MSELoss with multiple inputs (matryoshka)."""
    model = static_retrieval_mrl_en_v1_model.cpu()
    orig_loss = MSELoss(model=model)
    loss = MatryoshkaLoss(model=model, loss=orig_loss, matryoshka_dims=[32, 64, 128, 256, 512, 768, 1024])
    sentence_features = [model.preprocess(["This is an input text"])]
    x = torch.randn(1, 1024)  # Original dimension (teacher)

    # Call the loss function and verify output type and shape
    output = loss(sentence_features, x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([])  # MSELoss returns a scalar
