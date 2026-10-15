# [MODULE DOC]
# Tests for the SparseEncoder model class: decoding sparse embeddings into (token, weight) lists (shapes, types,
# top_k limits, sparse/dense input, empty and invalid inputs, sorting), inference-free SPLADE routing and max
# sequence length propagation, max_active_dims handling for SPLADE and CSR models, architecture detection
# (fill-mask backbones become SPLADE, others CSR), sparsity statistics, pooling chunk size, embedding
# intersection, encoding from datasets and numpy arrays, empty inputs, extra keyword argument validation (also
# through a Router), similarity functions and their persistence, prompts and their persistence, query/document
# prompt resolution in encode_query and encode_document, forward hooks, and converting saved models of other
# families.
from __future__ import annotations

import copy
import json
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sentence_transformers.sentence_transformer.modules import Pooling
from sentence_transformers.sparse_encoder.model import SparseEncoder
from sentence_transformers.sparse_encoder.modules import Router, SparseAutoEncoder, SpladePooling, Transformer
from sentence_transformers.util.similarity import SimilarityFunction


@pytest.mark.parametrize(
    ("texts", "top_k", "expected_shape"),
    [
        # Single text, default top_k (None)
        (["The weather is nice!"], None, 1),
        # Single text, specific top_k
        (["The weather is nice!"], 3, 1),
        # String text, specific top_k, expect a non-nested list
        ("The weather is nice!", 8, 8),
        # Multiple texts, default top_k (None)
        (["The weather is nice!", "It's sunny outside"], None, 2),
        # Multiple texts, specific top_k
        (["The weather is nice!", "It's sunny outside"], 3, 2),
    ],
)
def test_decode_shapes(
    splade_bert_tiny_model: SparseEncoder, texts: list[str] | str, top_k: int, expected_shape: int
) -> None:
    # [FUNCTION DOC]
    # Parametrized over text inputs and top_k. Encodes and decodes, asserting the decoded length (one list per
    # input text, or top_k tuples for a bare string) and, for lists, that each item is a list no longer than top_k.
    model = splade_bert_tiny_model
    embeddings = model.encode(texts)
    decoded = model.decode(embeddings, top_k=top_k)

    assert len(decoded) == expected_shape

    if isinstance(texts, list):
        if len(texts) == 1:
            assert isinstance(decoded[0], tuple) or isinstance(decoded, list)
            if top_k is not None:
                assert len(decoded) <= top_k
        else:
            assert isinstance(decoded, list)
            for item in decoded:
                assert isinstance(item, list)
                if top_k is not None:
                    assert len(item) <= top_k


@pytest.mark.parametrize(
    ("text", "expected_token_types"),
    [
        ("The weather is nice!", str),
        ("It's sunny outside", str),
    ],
)
def test_decode_token_types(splade_bert_tiny_model: SparseEncoder, text: str, expected_token_types: type) -> None:
    # [FUNCTION DOC]
    # Parametrized over texts; asserts every decoded pair of a single string is a string token with a float weight.
    model = splade_bert_tiny_model
    embeddings = model.encode(text)
    decoded = model.decode(embeddings)

    # Check the first item in the batch
    for token, weight in decoded:
        assert isinstance(token, expected_token_types)
        assert isinstance(weight, float)


@pytest.mark.parametrize(
    ("text", "top_k"),
    [
        ("The weather is nice!", 1),
        ("It's sunny outside", 3),
        ("Hello world", 5),
    ],
)
def test_decode_top_k_respects_limit(splade_bert_tiny_model: SparseEncoder, text: str, top_k: int) -> None:
    # [FUNCTION DOC]
    # Asserts decoding a one-item list with top_k gives at most top_k entries.
    model = splade_bert_tiny_model
    embeddings = model.encode([text])
    decoded = model.decode(embeddings, top_k=top_k)

    assert len(decoded) <= top_k


@pytest.mark.parametrize(
    ("texts", "format_type"),
    [
        ("The weather is nice!", "1d"),
        (["The weather is nice!"], "1d"),
        (["The weather is nice!", "It's sunny outside"], "2d"),
    ],
)
def test_decode_handles_sparse_dense_inputs(
    splade_bert_tiny_model: SparseEncoder, texts: list[str] | str, format_type: str
):
    # [FUNCTION DOC]
    # Parametrized over inputs; decodes the sparse and dense forms of the embeddings and asserts equal lengths (and,
    # for batches, equal per-item lengths).
    # KNOWN ISSUE: The 2-D branch sorts both decoded results "to ensure consistent comparison" but then only compares
    # lengths, so the token/weight contents of sparse and dense decoding are never compared.
    model = splade_bert_tiny_model
    # Get embeddings and test both sparse and dense format handling
    embeddings = model.encode(texts)

    # Test with sparse tensor
    if not embeddings.is_sparse:
        embeddings_sparse = embeddings.to_sparse()
    else:
        embeddings_sparse = embeddings

    decoded_sparse = model.decode(embeddings_sparse)

    # Test with dense tensor
    if embeddings.is_sparse:
        embeddings_dense = embeddings.to_dense()
    else:
        embeddings_dense = embeddings

    decoded_dense = model.decode(embeddings_dense)

    # Verify both produce the same result structure
    if format_type == "1d":
        assert len(decoded_sparse) == len(decoded_dense)
    else:
        assert len(decoded_sparse) == len(decoded_dense)
        for i in range(len(decoded_sparse)):
            # Sort both results to ensure consistent comparison
            sorted_sparse = sorted(decoded_sparse[i], key=lambda x: (x[1], x[0]), reverse=True)
            sorted_dense = sorted(decoded_dense[i], key=lambda x: (x[1], x[0]), reverse=True)
            assert len(sorted_sparse) == len(sorted_dense)


def test_decode_empty_tensor(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts decoding an empty 1 x vocabulary sparse tensor gives no entries.
    model = splade_bert_tiny_model
    # Create an empty sparse tensor
    empty_sparse = torch.sparse_coo_tensor(
        indices=torch.zeros((2, 0), dtype=torch.long),
        values=torch.zeros((0,), dtype=torch.float),
        size=(1, model.get_embedding_dimension()),
    )

    decoded = model.decode(empty_sparse)
    assert len(decoded) == 0 or (isinstance(decoded, list) and all(not item for item in decoded))


@pytest.mark.parametrize("top_k", [0, -1, -5])
def test_decode_invalid_top_k(splade_bert_tiny_model: SparseEncoder, top_k: int) -> None:
    # [FUNCTION DOC]
    # Parametrized over 0, -1, -5; asserts decode raises a ValueError requiring a positive top_k.
    model = splade_bert_tiny_model
    embeddings = model.encode("Hello world")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        model.decode(embeddings, top_k=top_k)


def test_decode_invalid_input_type(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts decoding a plain list raises TypeError expecting a tensor.
    model = splade_bert_tiny_model
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        model.decode([1, 2, 3])


def test_decode_invalid_ndim(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts decoding a 3-D tensor raises ValueError requiring 1-D or 2-D input.
    model = splade_bert_tiny_model
    tensor_3d = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="Input tensor must be 1D or 2D"):
        model.decode(tensor_3d)


def test_decode_batch_with_empty_sample(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts decoding a 2-row sparse batch whose second row is empty gives two entries for row 0 and an empty list
    # for row 1.
    model = splade_bert_tiny_model
    vocab_size = model.get_embedding_dimension()
    # Create a batch where the first sample has values but the second is all zeros
    indices = torch.tensor([[0, 0], [1, 5]])  # both non-zero entries in sample 0
    values = torch.tensor([1.0, 2.0])
    batch_sparse = torch.sparse_coo_tensor(indices, values, size=(2, vocab_size))

    decoded = model.decode(batch_sparse)
    assert len(decoded) == 2
    assert len(decoded[0]) == 2  # sample 0 has 2 non-zero entries
    assert decoded[1] == []  # sample 1 is empty


@pytest.mark.parametrize("top_k", [None, 5, 1000])
@pytest.mark.parametrize(
    "texts",
    [
        ("The weather is nice!"),
        (["The weather is nice!"]),
        (["The weather is nice!", "It's sunny outside", "Hello world"]),
        (["Short text", "This is a longer text with more words to encode"]),
    ],
)
def test_decode_returns_sorted_weights(
    splade_bert_tiny_model: SparseEncoder, texts: list[str] | str, top_k: int | None
) -> None:
    # [FUNCTION DOC]
    # Parametrized over texts and top_k; asserts decoded weights are in non-increasing order.
    model = splade_bert_tiny_model
    embeddings = model.encode(texts)
    decoded = model.decode(embeddings, top_k=top_k)

    if isinstance(texts, list):
        for item in decoded:
            weights = [weight for _, weight in item]
            assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))
    else:
        weights = [weight for _, weight in decoded]
        assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def test_inference_free_splade(inference_free_splade_bert_tiny_model: SparseEncoder):
    # [FUNCTION DOC]
    # Encodes a query and a document with an inference-free model and asserts 1-D embeddings of the vocabulary
    # size, query active dims equal to its token count, at least 50 document active dims, and that setting the
    # model's max sequence length from 512 to 256 updates both router branches.
    model = inference_free_splade_bert_tiny_model
    dimensionality = model.get_embedding_dimension()

    query = "What is the capital of France?"
    document = "The capital of France is Paris."
    query_embeddings = model.encode_query(query)
    document_embeddings = model.encode_document(document)

    assert query_embeddings.shape == (dimensionality,)
    assert document_embeddings.shape == (dimensionality,)

    decoded_query = model.decode(query_embeddings)
    decoded_document = model.decode(document_embeddings)
    assert len(decoded_query) == len(model.preprocess(query, task="query")["input_ids"][0])
    assert len(decoded_document) >= 50

    assert model.max_seq_length == 512
    assert model[0].sub_modules["query"][0].max_seq_length == 512
    assert model[0].sub_modules["document"][0].max_seq_length == 512

    model.max_seq_length = 256
    assert model.max_seq_length == 256
    assert model[0].sub_modules["query"][0].max_seq_length == 256
    assert model[0].sub_modules["document"][0].max_seq_length == 256


def test_inference_free_splade_max_active_dims_routing(inference_free_splade_bert_tiny_model: SparseEncoder):
    # [FUNCTION DOC]
    # Asserts encoding with max_active_dims 50 keeps a subset of the baseline active indices with at most 50
    # entries, for both query and document routes.
    model = inference_free_splade_bert_tiny_model
    query = "What is the capital of France?"
    document = "The capital of France is Paris."

    # Encode without max_active_dims: baseline
    query_emb = model.encode_query(query)
    doc_emb = model.encode_document(document)

    # Encode with max_active_dims: should route to the same sub-modules
    query_emb_mad = model.encode_query(query, max_active_dims=50)
    doc_emb_mad = model.encode_document(document, max_active_dims=50)

    # The non-zero indices of the max_active_dims result should be a subset of the baseline
    query_baseline_indices = query_emb.coalesce().indices()[0]
    query_mad_indices = query_emb_mad.coalesce().indices()[0]
    assert set(query_mad_indices.tolist()).issubset(set(query_baseline_indices.tolist()))
    assert query_emb_mad._nnz() <= 50

    doc_baseline_indices = doc_emb.coalesce().indices()[0]
    doc_mad_indices = doc_emb_mad.coalesce().indices()[0]
    assert set(doc_mad_indices.tolist()).issubset(set(doc_baseline_indices.tolist()))
    assert doc_emb_mad._nnz() <= 50


def test_encode_advanced_parameters(splade_bert_tiny_model: SparseEncoder, monkeypatch: pytest.MonkeyPatch):
    # [FUNCTION DOC]
    # Replaces encode with a recorder and asserts encode_query forwards normalization, batch size, progress bar,
    # max_active_dims, chunk size and an arbitrary custom keyword.
    """Test that additional parameters are correctly passed to encode"""
    model = splade_bert_tiny_model

    encode_calls = []

    def spy_encode(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the positional and keyword arguments of the call; returns None.
        encode_calls.append((args, kwargs))

    monkeypatch.setattr(model, "encode", spy_encode)
    # Call with advanced parameters
    model.encode_query(
        "test",
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
        max_active_dims=128,
        chunk_size=10,
        custom_param="value",
    )

    # Verify all parameters were passed correctly
    _, kwargs = encode_calls[0]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 64
    assert kwargs["show_progress_bar"] is True
    assert kwargs["max_active_dims"] == 128
    assert kwargs["chunk_size"] == 10
    assert kwargs["custom_param"] == "value"


def test_csr_max_active_dims_passed_to_forward(csr_bert_tiny_model: SparseEncoder, monkeypatch: pytest.MonkeyPatch):
    # [FUNCTION DOC]
    # Asserts the CSR model's last module is a SparseAutoEncoder with k 16, and, by spying on its forward, that an
    # explicit max_active_dims of 5 is passed through and that otherwise the model-level default is passed.
    model = csr_bert_tiny_model
    assert isinstance(model[-1], SparseAutoEncoder)
    assert model[-1].k == 16

    # Verify that max_active_dims is passed to SparseAutoEncoder.forward()
    forward_calls = []
    original_forward = model[-1].forward

    def spy_forward(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the keyword arguments and delegates to the original forward.
        forward_calls.append(kwargs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model[-1], "forward", spy_forward)

    model.encode("Hello world", max_active_dims=5)
    assert len(forward_calls) == 1
    assert forward_calls[0]["max_active_dims"] == 5

    # Without max_active_dims, the model's default max_active_dims is used
    forward_calls.clear()
    model.encode("Hello world")
    assert len(forward_calls) == 1
    assert forward_calls[0]["max_active_dims"] == model.max_active_dims


def test_max_active_dims_set_init(splade_bert_tiny_model: SparseEncoder, csr_bert_tiny_model: SparseEncoder, tmp_path):
    # [FUNCTION DOC]
    # Saves SPLADE and CSR models to a temp directory and asserts reloaded max_active_dims is None for SPLADE, 16 for
    # CSR (from k), and 13 when given explicitly.
    splade_bert_tiny_model.save_pretrained(str(tmp_path / "splade_bert_tiny"))
    csr_bert_tiny_model.save_pretrained(str(tmp_path / "csr_bert_tiny"))

    # Load the models with max_active_dims set
    loaded_model = SparseEncoder(str(tmp_path / "splade_bert_tiny"))
    assert loaded_model.max_active_dims is None
    loaded_model = SparseEncoder(str(tmp_path / "splade_bert_tiny"), max_active_dims=13)
    assert loaded_model.max_active_dims == 13

    loaded_model = SparseEncoder(str(tmp_path / "csr_bert_tiny"))
    assert loaded_model.max_active_dims == 16  # Based on the SparseAutoEncoder's k value
    loaded_model = SparseEncoder(str(tmp_path / "csr_bert_tiny"), max_active_dims=13)
    assert loaded_model.max_active_dims == 13


def test_detect_mlm():
    # [FUNCTION DOC]
    # Asserts loading a DistilBERT checkpoint (network) yields a fill-mask Transformer followed by SpladePooling.
    model = SparseEncoder("distilbert/distilbert-base-uncased")

    assert isinstance(model[0], Transformer)
    assert model[0].transformer_task == "fill-mask"
    assert isinstance(model[1], SpladePooling)


def test_default_to_csr():
    # [FUNCTION DOC]
    # Asserts loading the tiny STSB BERT checkpoint yields Transformer, Pooling and SparseAutoEncoder (CSR).
    # NOTE: bert-tiny is actually MLM-based, but the config isn't modern enough to allow us to detect it,
    # so we should default to CSR here.
    model = SparseEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert isinstance(model[0], Transformer)
    assert isinstance(model[1], Pooling)
    assert isinstance(model[2], SparseAutoEncoder)


def test_sparsity(splade_bert_tiny_model: SparseEncoder):
    # [FUNCTION DOC]
    # Asserts sparsity statistics of two query embeddings report between 1 and 99 active dims and a ratio in
    # [0.99, 1), identical for the dense form, and that the batch active dims equal the mean of the per-row values.
    model = splade_bert_tiny_model

    # Check that the sparsity is applied correctly
    embeddings = model.encode_query(["What is the capital of France?", "Who has won the World Cup in 2016?"])
    sparsity = model.sparsity(embeddings)
    assert isinstance(sparsity, dict)
    assert "active_dims" in sparsity
    assert "sparsity_ratio" in sparsity
    assert sparsity["active_dims"] < 100 and sparsity["active_dims"] > 0
    assert sparsity["sparsity_ratio"] < 1.0 and sparsity["sparsity_ratio"] >= 0.99

    # Also check with dense tensors
    dense_sparsity = model.sparsity(embeddings.to_dense())
    assert dense_sparsity == sparsity, "Sparsity should be the same for dense and sparse tensors"

    # Check that 1-dimensional embeddings work correctly
    sparsity_one = model.sparsity(embeddings[0])
    sparsity_two = model.sparsity(embeddings[1])
    assert (sparsity_one["active_dims"] + sparsity_two["active_dims"]) / 2 == sparsity["active_dims"]


def test_splade_pooling_chunk_size(splade_bert_tiny_model: SparseEncoder):
    # [FUNCTION DOC]
    # Asserts the pooling chunk size defaults to None and setting it updates the SpladePooling module.
    model = splade_bert_tiny_model

    # The chunk size defaults to None, i.e. no chunking
    assert model.splade_pooling_chunk_size is None
    # But we can chunk the pooling to save memory at the cost of some speed
    model.splade_pooling_chunk_size = 13
    assert model.splade_pooling_chunk_size == 13
    assert isinstance(model[1], SpladePooling)
    assert model[1].chunk_size == 13


def test_intersection(splade_bert_tiny_model: SparseEncoder):
    # [FUNCTION DOC]
    # Asserts the intersection of a query and document embedding is a 1-D tensor with fewer active dims than either,
    # and that intersecting one query with two documents gives a 2 x vocabulary tensor decodable into two lists.
    model = splade_bert_tiny_model

    # Test intersection with a single text
    query = "Where can I deposit my money?"
    document = "I'm sitting by the river."
    query_embeddings = model.encode_query(query)
    document_embeddings = model.encode_document(document)
    query_sparsity = model.sparsity(query_embeddings)
    document_sparsity = model.sparsity(document_embeddings)

    # Let's check that the intersection is a tensor and has the correct shape
    intersection = model.intersection(query_embeddings, document_embeddings)
    assert isinstance(intersection, torch.Tensor)
    assert intersection.shape == (model.get_embedding_dimension(),)

    # Check that the intersection sparsity is less than both query and document sparsities
    intersection_sparsity = model.sparsity(intersection)
    assert (
        intersection_sparsity["active_dims"] < query_sparsity["active_dims"]
        and intersection_sparsity["active_dims"] < document_sparsity["active_dims"]
    )

    # Test with multiple texts
    query = "Who has won the World Cup in 2016?"
    documents = ["The capital of France is Paris.", "Germany won the World Cup in 2014."]
    query_embeddings = model.encode_query(query)
    document_embeddings = model.encode_document(documents)

    intersection_batch = model.intersection(query_embeddings, document_embeddings)
    assert isinstance(intersection_batch, torch.Tensor)
    assert intersection_batch.shape == (len(documents), model.get_embedding_dimension())

    decoded_intersection_batch = model.decode(intersection_batch)
    assert len(decoded_intersection_batch) == len(documents)


def test_encode_with_dataset_column(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts encoding a datasets column gives a 2 x vocabulary tensor.
    """Test that encode can handle a dataset column as input."""
    model = splade_bert_tiny_model
    from datasets import Dataset

    # Create a simple dataset with a text column
    dataset = Dataset.from_dict({"text": ["This is a test.", "Another sentence."]})

    # Encode the dataset column
    embeddings = model.encode(dataset["text"], convert_to_tensor=True)

    # Check the shape of the embeddings
    assert embeddings.shape == (2, model.get_embedding_dimension())


def test_encode_per_call_processing_kwargs(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts per-call processing kwargs truncating text to 4 tokens give a different embedding than the full text.
    """Per-call ``processing_kwargs`` should be accepted by ``encode`` and reach ``preprocess``.

    If the kwarg were silently dropped (or rejected by encode's allowlist), truncated and full
    encodings of the same text would either raise or produce identical sparse embeddings.
    """
    model = splade_bert_tiny_model
    text = "this sentence is much longer than four tokens for sure"
    truncated = model.encode(
        [text],
        convert_to_tensor=True,
        save_to_cpu=True,
        processing_kwargs={"text": {"max_length": 4, "truncation": True}},
    )
    full = model.encode([text], convert_to_tensor=True, save_to_cpu=True)
    assert truncated.shape == full.shape
    assert not torch.allclose(truncated.to_dense(), full.to_dense())


def test_encode_numpy_1d_string_array(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts a 1-D numpy string array is encoded per element, matching the list input.
    """Regression test for #3718: encoding a 1D numpy string array should produce one embedding per element."""
    model = splade_bert_tiny_model
    texts = np.array(["Access Management", "Press Coordination", "Financial Reports"])
    embeddings = model.encode(texts, convert_to_tensor=True, save_to_cpu=True)
    expected = model.encode(texts.tolist(), convert_to_tensor=True, save_to_cpu=True)
    assert embeddings.shape == (3, model.get_embedding_dimension())
    assert torch.allclose(embeddings.to_dense(), expected.to_dense())


def test_encode_numpy_2d_string_array(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts a 2-D numpy string array of pairs matches encoding the nested list.
    """Encoding a 2D numpy string array should match encoding the equivalent nested list."""
    model = splade_bert_tiny_model
    pairs = np.array([["what is AI?", "AI is artificial intelligence."], ["what is ML?", "ML is machine learning."]])
    embeddings = model.encode(pairs, convert_to_tensor=True, save_to_cpu=True)
    expected = model.encode(pairs.tolist(), convert_to_tensor=True, save_to_cpu=True)
    assert embeddings.shape == (2, model.get_embedding_dimension())
    assert torch.allclose(embeddings.to_dense(), expected.to_dense())


def test_encode_numpy_empty(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts an empty numpy string array encodes to an empty tensor equal to encoding an empty list.
    """Encoding an empty string ndarray should return an empty tensor, like ``encode([])``."""
    model = splade_bert_tiny_model
    embeddings = model.encode(np.array([], dtype=str), convert_to_tensor=True, save_to_cpu=True)
    expected = model.encode([], convert_to_tensor=True, save_to_cpu=True)
    assert embeddings.numel() == 0
    assert torch.equal(embeddings.to_dense(), expected.to_dense())


@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_sparse_tensor", [True, False])
@pytest.mark.parametrize("save_to_cpu", [True, False])
@pytest.mark.parametrize("max_active_dims", [None, 64, 128])
def test_empty_encode(
    splade_bert_tiny_model: SparseEncoder,
    convert_to_tensor: bool,
    convert_to_sparse_tensor: bool,
    save_to_cpu: bool,
    max_active_dims: int | None,
):
    # [FUNCTION DOC]
    # Parametrized over tensor/sparse conversion, CPU saving and max_active_dims; asserts encoding an empty list
    # gives an empty tensor on the right device with the right sparsity, or an empty list without tensor conversion.
    model = splade_bert_tiny_model
    embeddings = model.encode(
        [],
        convert_to_tensor=convert_to_tensor,
        convert_to_sparse_tensor=convert_to_sparse_tensor,
        save_to_cpu=save_to_cpu,
        max_active_dims=max_active_dims,
    )

    if convert_to_tensor:
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.numel() == 0
        if save_to_cpu:
            assert embeddings.device == torch.device("cpu")
        else:
            assert embeddings.device == model.device

        if convert_to_sparse_tensor:
            assert embeddings.is_sparse
        else:
            assert not embeddings.is_sparse
    else:
        assert embeddings == []


def test_get_model_kwargs(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts get_model_kwargs is empty by default and unknown encode kwargs raise ValueError; after setting
    # forward kwargs on the modules, asserts they are listed, unknown kwargs still raise, and passing them reaches the
    # modules (raising TypeError since they do not really accept them); then replaces the pooling module with a Router
    # of differently configured copies and asserts the kwargs include both routes plus task and modality, with the same
    # error behaviour. Mutates the fixture model.
    """Test that get_model_kwargs returns the correct keyword arguments."""
    model = splade_bert_tiny_model

    # Check that the forward kwargs are as expected, i.e. no extra forward kwargs
    # for this basic model
    forward_kwargs = model.get_model_kwargs()
    assert forward_kwargs == []
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SparseEncoder.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SparseEncoder.get_model_kwargs(), this model does not accept "
            "any additional keyword arguments."
        ),
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", normalize=True)
    # This should run fine
    model.encode("Test sentence")
    model.encode_query("Test sentence")

    # If one of the modules has additional forward kwargs, they should be included
    model[0].forward_kwargs = {"foo"}
    model[1].forward_kwargs = {"bar", "baz"}
    assert set(model.get_model_kwargs()) == {"foo", "bar", "baz"}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SparseEncoder.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SparseEncoder.get_model_kwargs(), the valid additional keyword"
            " arguments are: "
        )
        + r"\[('foo'|'bar'|'baz'|, ){5}\].",
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", normalize=True)
    # This should run fine
    model.encode("Test sentence")
    model.encode_query("Test sentence")
    with pytest.raises(
        TypeError,
        match=r"(Transformer\.)?forward\(\) got an unexpected keyword argument '(foo|bar)'",
    ):
        # This would run fine, except the model can't actually accept these arguments (we monkeypatched the modules'
        # forward_kwargs for this test, after all). The model does send the args down to the underlying modules, though!
        model.encode("Test sentence", foo=True, bar=False)

    # And also if we have a Router in place
    query_pooling_copy = copy.deepcopy(model[1])
    query_pooling_copy.forward_kwargs = {"query_arg"}
    document_pooling_copy = copy.deepcopy(model[1])
    document_pooling_copy.forward_kwargs = {"document_arg_1", "document_arg_2"}
    model[1] = Router.for_query_document(
        query_modules=[query_pooling_copy],
        document_modules=[document_pooling_copy],
    )
    assert set(model.get_model_kwargs()) == {
        "foo",
        "task",
        "query_arg",
        "document_arg_1",
        "document_arg_2",
        "modality",
    }
    with pytest.raises(
        ValueError,
        match=re.escape(
            "SparseEncoder.encode() has been called with additional keyword arguments that this model does "
            "not use: ['normalize']. As per SparseEncoder.get_model_kwargs(), the valid additional keyword"
            " arguments are: "
        )
        + r"\[('foo'|'task'|'query_arg'|'document_arg_1'|'document_arg_2'|'modality'|, ){11}\].",
    ):
        # There is no "normalize" argument, this should crash
        model.encode("Test sentence", task="query", normalize=True)
    # This should run fine
    model.encode("Test sentence", task="document")
    model.encode_query("Test sentence")
    with pytest.raises(
        TypeError,
        match=r"(Transformer\.)?forward\(\) got an unexpected keyword argument '(foo|document_arg_1)'",
    ):
        # This would run fine, except the model can't actually accept these arguments (we monkeypatched the modules'
        # forward_kwargs for this test, after all). The model does send the args down to the underlying modules, though!
        model.encode("Test sentence", task="document", foo=True, document_arg_1=12)


# MaxSim is for multi-vector (3D) embeddings. SparseEncoder is single-vector so it's not applicable.
@pytest.mark.parametrize(
    "similarity_fn_name",
    [v for v in SimilarityFunction.possible_values() if v in SparseEncoder.SUPPORTED_SIMILARITY_FN_NAMES],
)
def test_similarity_score(splade_bert_tiny_model: SparseEncoder, similarity_fn_name: str) -> None:
    # [FUNCTION DOC]
    # Parametrized over supported similarity names; encodes four sentences densely and asserts a 4x4 score matrix
    # with self-similarities of 1 for cosine, 0 for euclidean/manhattan and positive for dot, and a pairwise score
    # vector of length two.
    model = splade_bert_tiny_model
    model.similarity_fn_name = similarity_fn_name
    sentences = [
        "The weather is so nice!",
        "It's so sunny outside.",
        "He's driving to the movie theater.",
        "She's going to the cinema.",
    ]
    embeddings = model.encode(sentences, convert_to_sparse_tensor=False)
    scores = model.similarity(embeddings, embeddings)
    assert scores.shape == (len(sentences), len(sentences))
    diag = np.diag(scores.cpu().numpy())
    if similarity_fn_name == "cosine":
        np.testing.assert_almost_equal(diag, np.ones(len(sentences), dtype=float), decimal=4)
    elif similarity_fn_name in ("euclidean", "manhattan"):
        np.testing.assert_almost_equal(diag, np.zeros(len(sentences), dtype=float), decimal=4)
    else:  # dot product - self-similarity of non-zero sparse vectors is positive
        assert (diag > 0).all()

    pairwise_scores = model.similarity_pairwise(embeddings[::2], embeddings[1::2])
    assert pairwise_scores.shape == (len(sentences) // 2,)


def test_similarity_score_save(splade_bert_tiny_model: SparseEncoder, tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts the default similarity is dot, and a cosine setting survives save and reload.
    model = splade_bert_tiny_model
    assert model.similarity_fn_name == "dot"

    model.similarity_fn_name = "cosine"
    model.save(str(tmp_path))
    loaded_model = SparseEncoder(str(tmp_path))
    assert loaded_model.similarity_fn_name == "cosine"


def test_similarity_fn_name_set_via_enum(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts setting the similarity via the enum stores the string name.
    model = splade_bert_tiny_model
    model.similarity_fn_name = SimilarityFunction.COSINE
    assert model.similarity_fn_name == "cosine"
    model.similarity_fn_name = SimilarityFunction.DOT
    assert model.similarity_fn_name == "dot"


def test_similarity_fn_name_constructor_overrides_saved(splade_bert_tiny_model: SparseEncoder, tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts a similarity name given to the constructor overrides the saved one.
    splade_bert_tiny_model.similarity_fn_name = "cosine"
    splade_bert_tiny_model.save(str(tmp_path))
    model = SparseEncoder(str(tmp_path), similarity_fn_name="dot")
    assert model.similarity_fn_name == "dot"


def test_prompts(splade_bert_tiny_model: SparseEncoder, caplog: pytest.LogCaptureFixture) -> None:
    # [FUNCTION DOC]
    # Asserts default prompts are empty; that prompted texts embed differently; that passing the prompt directly,
    # or by name, reproduces the manually prompted embedding; that passing both logs a single warning and uses the
    # prompt; and that an unknown prompt name raises ValueError.
    model = splade_bert_tiny_model
    assert model.prompts == {"query": "", "document": ""}
    assert model.default_prompt_name is None
    texts = ["How to bake a chocolate cake", "Symptoms of the flu"]
    no_prompt_embedding = model.encode(texts, convert_to_sparse_tensor=False, save_to_cpu=True)
    prompt_embedding = model.encode(
        [f"query: {text}" for text in texts], convert_to_sparse_tensor=False, save_to_cpu=True
    )
    assert not np.array_equal(no_prompt_embedding, prompt_embedding)

    query = "query: "
    # Test prompt="query: "
    model.prompts = {"query": "", "document": ""}
    assert np.array_equal(
        model.encode(texts, prompt=query, convert_to_sparse_tensor=False, save_to_cpu=True), prompt_embedding
    )

    # Test prompt_name="..."
    model.prompts = {"query": query, "document": ""}
    assert np.array_equal(
        model.encode(texts, prompt_name="query", convert_to_sparse_tensor=False, save_to_cpu=True), prompt_embedding
    )

    caplog.clear()
    # Test prompt_name="..." & prompt="..."
    with caplog.at_level(logging.WARNING):
        assert np.array_equal(
            model.encode(texts, prompt=query, prompt_name="query", convert_to_sparse_tensor=False, save_to_cpu=True),
            prompt_embedding,
        )
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2] == "Provide either a `prompt`, a `prompt_name`, or neither, but not both. "
            "Ignoring the `prompt_name` in favor of `prompt`."
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Prompt name 'invalid_prompt_name' not found in the configured prompts dictionary with keys ['query', 'document']."
        ),
    ):
        model.encode(texts, prompt_name="invalid_prompt_name")


def test_save_load_prompts() -> None:
    # [FUNCTION DOC]
    # Asserts an unknown default prompt name raises ValueError at construction; then that given prompts are merged
    # with an empty document prompt, saved to the config JSON in a temp directory, and restored on reload together
    # with the default prompt name. Downloads the model from the Hub.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Default prompt name 'invalid_prompt_name' not found in the configured prompts dictionary with keys ['query', 'document']."
        ),
    ):
        SparseEncoder(
            "sparse-encoder-testing/splade-bert-tiny-nq",
            prompts={"query": "query: "},
            default_prompt_name="invalid_prompt_name",
        )

    model = SparseEncoder(
        "sparse-encoder-testing/splade-bert-tiny-nq",
        prompts={"query": "query: "},
        default_prompt_name="query",
    )
    assert model.prompts == {"query": "query: ", "document": ""}
    assert model.default_prompt_name == "query"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        model.save(str(model_path))
        config_path = model_path / "config_sentence_transformers.json"
        assert config_path.exists()
        with open(config_path, encoding="utf8") as f:
            saved_config = json.load(f)
        assert saved_config["prompts"] == {"query": "query: ", "document": ""}
        assert saved_config["default_prompt_name"] == "query"

        fresh_model = SparseEncoder(str(model_path))
        assert fresh_model.prompts == {"query": "query: ", "document": ""}
        assert fresh_model.default_prompt_name == "query"


@pytest.mark.parametrize("sentences", ["Hello world", ["Hello world", "This is a test"], [], [""]])
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_sparse_tensor", [True, False])
@pytest.mark.parametrize("prompt_name", [None, "query", "custom"])
@pytest.mark.parametrize("prompt", [None, "Custom prompt: "])
def test_encode_query(
    splade_bert_tiny_model: SparseEncoder,
    sentences: str | list[str],
    convert_to_tensor: bool,
    convert_to_sparse_tensor: bool,
    prompt_name: str | None,
    prompt: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    # [FUNCTION DOC]
    # Parametrized over inputs, conversion flags, prompt name and prompt. Replaces encode with a recorder and asserts
    # encode_query passes the inputs, flags and prompt through, sets task "query", and resolves the prompt name to
    # the given name, else None when a prompt is given, else "query".
    model = splade_bert_tiny_model
    model.prompts = {"query": "query: ", "custom": "custom: "}

    encode_calls = []

    def spy_encode(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the call's arguments; returns None.
        encode_calls.append((args, kwargs))

    monkeypatch.setattr(model, "encode", spy_encode)
    model.encode_query(
        sentences=sentences,
        batch_size=32,
        convert_to_tensor=convert_to_tensor,
        convert_to_sparse_tensor=convert_to_sparse_tensor,
        prompt_name=prompt_name,
        prompt=prompt,
    )

    if prompt_name:
        expected_prompt_name = prompt_name
    elif prompt is not None:
        expected_prompt_name = None
    else:
        expected_prompt_name = "query"

    assert len(encode_calls) == 1
    _, kwargs = encode_calls[0]

    assert kwargs["inputs"] == sentences
    assert kwargs["convert_to_tensor"] == convert_to_tensor
    assert kwargs["convert_to_sparse_tensor"] == convert_to_sparse_tensor
    assert kwargs["prompt"] == prompt
    assert kwargs["prompt_name"] == expected_prompt_name
    assert kwargs["task"] == "query"


@pytest.mark.parametrize("sentences", ["Hello world", ["Hello world", "This is a test"], [], [""]])
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_sparse_tensor", [True, False])
@pytest.mark.parametrize("prompt_name", [None, "document", "passage", "corpus", "custom"])
@pytest.mark.parametrize("prompt", [None, "Custom prompt: "])
def test_encode_document(
    splade_bert_tiny_model: SparseEncoder,
    sentences: str | list[str],
    convert_to_tensor: bool,
    convert_to_sparse_tensor: bool,
    prompt_name: str | None,
    prompt: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    # [FUNCTION DOC]
    # Same as the previous test for encode_document, whose fallback prompt name is "document" and whose task is
    # "document".
    model = splade_bert_tiny_model
    model.prompts = {"document": "document: ", "passage": "passage: ", "corpus": "corpus: ", "custom": "custom: "}

    encode_calls = []

    def spy_encode(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the call's arguments; returns None.
        encode_calls.append((args, kwargs))

    monkeypatch.setattr(model, "encode", spy_encode)
    model.encode_document(
        sentences=sentences,
        batch_size=32,
        convert_to_tensor=convert_to_tensor,
        convert_to_sparse_tensor=convert_to_sparse_tensor,
        prompt_name=prompt_name,
        prompt=prompt,
    )

    assert len(encode_calls) == 1
    _, kwargs = encode_calls[0]

    if prompt_name:
        expected_prompt_name = prompt_name
    elif prompt is not None:
        expected_prompt_name = None
    else:
        expected_prompt_name = "document"

    assert kwargs["inputs"] == sentences
    assert kwargs["convert_to_tensor"] == convert_to_tensor
    assert kwargs["convert_to_sparse_tensor"] == convert_to_sparse_tensor
    assert kwargs["prompt"] == prompt
    assert kwargs["prompt_name"] == expected_prompt_name
    assert kwargs["task"] == "document"


def test_encode_document_prompt_priority(splade_bert_tiny_model: SparseEncoder, monkeypatch: pytest.MonkeyPatch):
    # [FUNCTION DOC]
    # Asserts encode_document picks the prompt name "document", then "passage", then "corpus" depending on which are
    # configured, and None when none of them is.
    """Test that proper prompt priority is respected when multiple options are available"""
    model = splade_bert_tiny_model
    model.prompts = {
        "document": "document: ",
        "passage": "passage: ",
        "corpus": "corpus: ",
    }

    encode_calls = []

    def spy_encode(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the call's arguments; returns None.
        encode_calls.append((args, kwargs))

    monkeypatch.setattr(model, "encode", spy_encode)

    model.encode_document("test")
    _, kwargs = encode_calls[-1]
    assert kwargs["prompt_name"] == "document"

    # Remove document, should fall back to passage
    encode_calls.clear()
    model.prompts = {"passage": "passage: ", "corpus": "corpus: "}
    model.encode_document("test")
    _, kwargs = encode_calls[-1]
    assert kwargs["prompt_name"] == "passage"

    # Remove passage, should fall back to corpus
    encode_calls.clear()
    model.prompts = {"corpus": "corpus: "}
    model.encode_document("test")
    _, kwargs = encode_calls[-1]
    assert kwargs["prompt_name"] == "corpus"

    # No document/passage/corpus, should use None
    encode_calls.clear()
    model.prompts = {"custom": "custom: "}
    model.encode_document("test")
    _, kwargs = encode_calls[-1]
    assert kwargs["prompt_name"] is None


def test_encode_routes_through_module_call(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Registers a forward hook, encodes once (removing the hook in a finally block) and asserts the hook fired, so
    # encode calls the model through its call method.
    """encode() must run the forward pass via __call__ so that model.compile() applies to inference."""
    model = splade_bert_tiny_model
    calls = []
    handle = model.register_forward_hook(lambda module, args, output: calls.append(True))
    try:
        model.encode("Hello world")
    finally:
        handle.remove()
    assert calls, "encode() should invoke the model via __call__, not call forward() directly"


def test_similarity_fn_name_rejects_multi_vector_names(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts setting a multi-vector similarity name raises a ValueError mentioning MultiVectorEncoder.
    """MaxSim scores ragged token embeddings: setting it on a sparse model would produce wrong shapes
    silently, so it must fail loud with a pointer to MultiVectorEncoder."""
    with pytest.raises(ValueError, match="MultiVectorEncoder"):
        splade_bert_tiny_model.similarity_fn_name = "maxsim"


def test_parse_model_config_ignores_multi_vector_similarity(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Temporarily clears the stored similarity name, parses a config with "maxsim", asserts it stays unset, and
    # restores the original in a finally block.
    """A saved multi-vector similarity falls through to the default instead of raising in the strict
    setter during config parsing."""
    model = splade_bert_tiny_model
    original = model._similarity_fn_name
    try:
        model._similarity_fn_name = None
        model._parse_model_config({"similarity_fn_name": "maxsim"})
        assert model._similarity_fn_name is None
    finally:
        model._similarity_fn_name = original


def test_conversion_ignores_source_prompts_and_similarity(tmp_path) -> None:
    # [FUNCTION DOC]
    # Saves a MultiVectorEncoder with custom prompts and default prompt name, loads it as a SparseEncoder and asserts
    # the default empty prompts, no default prompt name, and dot similarity.
    """Converting a MultiVectorEncoder save rebuilds this family's default modules, so the source's
    prompts, default_prompt_name and similarity_fn_name are not inherited (they would silently apply
    to every encode call and persist on save). SentenceTransformer saves are the exception: their
    modules are reused, see SparseEncoder._load_converted_modules."""
    from sentence_transformers import MultiVectorEncoder

    source = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    source.prompts = {"query": "find: ", "document": "text: "}
    source.default_prompt_name = "query"
    source.save_pretrained(str(tmp_path))

    model = SparseEncoder(str(tmp_path))
    assert model.prompts == {"query": "", "document": ""}
    assert model.default_prompt_name is None
    assert model.similarity_fn_name == "dot"


def test_conversion_from_sentence_transformer_keeps_prompts_and_similarity(tmp_path) -> None:
    # [FUNCTION DOC]
    # Saves a SentenceTransformer with a query prompt, loads it as a SparseEncoder and asserts a SparseAutoEncoder is
    # appended and the prompt and cosine similarity are kept.
    """The SentenceTransformer branch reuses the saved modules and appends a SparseAutoEncoder
    (CSR), so the source's prompts and similarity stay meaningful for the kept backbone and survive.
    Carrying the dense base's "cosine" over is what the CSR recipe wants, rather than the "dot" that
    a SPLADE-shaped SparseEncoder defaults to."""
    from sentence_transformers import SentenceTransformer

    source = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        prompts={"query": "Represent this sentence: "},
    )
    source.save_pretrained(str(tmp_path))

    model = SparseEncoder(str(tmp_path))
    assert isinstance(model[-1], SparseAutoEncoder)
    assert model.prompts["query"] == "Represent this sentence: "
    assert model.similarity_fn_name == "cosine"
