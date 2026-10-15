# [MODULE DOC]
# Tests the ONNX and OpenVINO inference backends for MultiVectorEncoder (per-token embeddings for late interaction): exporting on load,
# saving and reloading exported models, ONNX graph optimization and dynamic int8 quantization, and backend name validation. Outputs are
# compared with a PyTorch reference computed from the same saved checkpoint. Skipped without optimum backends or in CI.
"""
Test different backends (PyTorch, ONNX, OpenVINO) for MultiVectorEncoder.

This module tests exporting, loading, and using models with different inference backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils import is_ci

try:
    from optimum.intel import OVModelForFeatureExtraction
    from optimum.onnxruntime import ORTModelForFeatureExtraction
except ImportError:
    pytest.skip("OpenVINO and ONNX backends are not available", allow_module_level=True)

from sentence_transformers import (
    MultiVectorEncoder,
    export_dynamic_quantized_onnx_model,
    export_optimized_onnx_model,
)

if is_ci():
    pytest.skip("Skip test in CI to try and avoid 429 Client Error", allow_module_level=True)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> str:
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture. Loads the tiny BERT checkpoint as a MultiVectorEncoder (which adds a randomly initialized projection),
    # sets fixed query expansion to 16 tokens, saves it once to a module temp dir and returns the path, so every backend loads identical weights.
    # Network and disk.
    # The tiny checkpoint gets a randomly initialised projection on conversion, so save it once
    # and reload from disk per backend. Otherwise every load would produce different embeddings.
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model[0].query_expansion = {"strategy": "fixed", "length": 16}
    save_dir = str(tmp_path_factory.mktemp("mve_backends") / "mve-tiny")
    model.save_pretrained(save_dir)
    return save_dir


@pytest.fixture(scope="module")
def torch_embeddings(model_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture returning the PyTorch backend's numpy per-token embeddings for one query and two documents.
    model = MultiVectorEncoder(model_dir, backend="torch")
    # Arrays, so the torch reference compares against the CPU-only backends without a device hop.
    queries = model.encode_query(["What is the capital of France?"], convert_to_numpy=True)
    documents = model.encode_document(
        ["Paris is the capital of France.", "Berlin is the capital of Germany."], convert_to_numpy=True
    )
    return queries, documents


def encode_and_compare(model: MultiVectorEncoder, torch_embeddings, atol: float = 1e-5) -> None:
    # [FUNCTION DOC]
    # HELPER: Encodes the same query and documents with the given model as numpy arrays, asserts the query has 16 token vectors (fixed expansion)
    # of the embedding dimension, and that each output has the reference's shape and matches it within atol.
    reference_queries, reference_documents = torch_embeddings
    queries = model.encode_query(["What is the capital of France?"], convert_to_numpy=True)
    documents = model.encode_document(
        ["Paris is the capital of France.", "Berlin is the capital of Germany."], convert_to_numpy=True
    )
    assert queries[0].shape == (16, model.get_embedding_dimension())
    for reference, embedding in zip(reference_queries + reference_documents, queries + documents):
        assert reference.shape == embedding.shape
        assert np.allclose(reference, embedding, atol=atol)


@pytest.mark.parametrize(
    ["backend", "expected_auto_model_class"],
    [
        ("onnx", ORTModelForFeatureExtraction),
        ("openvino", OVModelForFeatureExtraction),
    ],
)
@pytest.mark.parametrize(
    "model_kwargs", [{}, {"file_name": "wrong_file_name"}]
)  # <- Using a file_name is fine when exporting
def test_backend_export(backend, expected_auto_model_class, model_kwargs, model_dir, torch_embeddings) -> None:
    # [FUNCTION DOC]
    # Parametrized over onnx/openvino with the expected optimum class and over model_kwargs (none or an ignored wrong file name). Loads with that
    # backend (exporting), asserts the backend name and optimum model class, and compares outputs to the reference.
    model = MultiVectorEncoder(model_dir, backend=backend, model_kwargs=model_kwargs)
    assert model.get_backend() == backend
    assert isinstance(model[0].auto_model, expected_auto_model_class)
    assert isinstance(model.transformers_model, expected_auto_model_class)
    encode_and_compare(model, torch_embeddings)


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_backend_save_and_load(backend, model_dir, torch_embeddings, tmp_path) -> None:
    # [FUNCTION DOC]
    # Parametrized over onnx/openvino. Loads with export, saves to tmp_path, reloads the exported directory with the same backend, and compares
    # outputs to the reference.
    export_dir = str(tmp_path / "exported")
    model = MultiVectorEncoder(model_dir, backend=backend)
    model.save_pretrained(export_dir)

    exported_model = MultiVectorEncoder(export_dir, backend=backend)
    encode_and_compare(exported_model, torch_embeddings)


def test_onnx_optimize_and_quantize(model_dir, torch_embeddings, tmp_path) -> None:
    # [FUNCTION DOC]
    # Exports to ONNX and saves; applies O1 optimization and reloads that file, comparing to the reference within 1e-3; applies AVX512-VNNI dynamic
    # quantization and reloads it, asserting only output shapes match the reference.
    export_dir = str(tmp_path / "exported")
    model = MultiVectorEncoder(model_dir, backend="onnx")
    model.save_pretrained(export_dir)

    export_optimized_onnx_model(model, "O1", export_dir)
    optimized_model = MultiVectorEncoder(export_dir, backend="onnx", model_kwargs={"file_name": "onnx/model_O1.onnx"})
    encode_and_compare(optimized_model, torch_embeddings, atol=1e-3)

    export_dynamic_quantized_onnx_model(model, "avx512_vnni", export_dir)
    quantized_model = MultiVectorEncoder(
        export_dir, backend="onnx", model_kwargs={"file_name": "onnx/model_qint8_avx512_vnni.onnx"}
    )
    reference_queries, reference_documents = torch_embeddings
    queries = quantized_model.encode_query(["What is the capital of France?"])
    documents = quantized_model.encode_document(
        ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    )
    # Quantization shifts values, so only the shapes are compared.
    for reference, embedding in zip(reference_queries + reference_documents, queries + documents):
        assert reference.shape == embedding.shape


def test_incorrect_backend(model_dir) -> None:
    # [FUNCTION DOC]
    # Asserts an unknown backend name raises ValueError.
    with pytest.raises(ValueError):
        MultiVectorEncoder(model_dir, backend="incorrect_backend")
