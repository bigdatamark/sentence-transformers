# [MODULE DOC]
# Pytest tests for CrossEncoder inference backends (PyTorch, ONNX via optimum onnxruntime, OpenVINO via optimum
# intel): exporting on load, loading pre-exported repositories, provider and OpenVINO config handling, numeric parity
# with PyTorch, and the file-name/subfolder arguments forwarded to the optimum loader. Skips the module if optimum's
# ONNX or OpenVINO classes cannot be imported, and in CI (to avoid Hub rate limiting). Loads tiny GooAQ reranker
# repositories from the Hub.
"""
Test different backends (PyTorch, ONNX, OpenVINO) for the CrossEncoder class.

This module tests loading and using CrossEncoder models with different inference backends.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from packaging.version import Version, parse

from tests.utils import is_ci

try:
    from optimum.intel import OVModelForSequenceClassification
    from optimum.intel.version import __version__ as optimum_intel_version
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.version import __version__ as optimum_version
except ImportError:
    pytest.skip("OpenVINO and ONNX backends are not available", allow_module_level=True)

from sentence_transformers import CrossEncoder

if is_ci():
    pytest.skip("Skip test in CI to try and avoid 429 Client Error", allow_module_level=True)


## Testing exporting:
@pytest.mark.parametrize(
    ["backend", "expected_auto_model_class"],
    [
        ("onnx", ORTModelForSequenceClassification),
        ("openvino", OVModelForSequenceClassification),
    ],
)
@pytest.mark.parametrize(
    "model_kwargs", [{}, {"file_name": "wrong_file_name"}]
)  # <- Using a file_name is fine when exporting
def test_backend_export(backend, expected_auto_model_class, model_kwargs) -> None:
    # [FUNCTION DOC]
    # Parametrised over backend (onnx or openvino, with the expected optimum sequence-classification class) and model
    # kwargs (none, or a wrong file name, which is ignored when exporting). Loads the plain PyTorch reranker with that
    # backend, asserts the reported backend and the underlying model class, and asserts predicting one pair returns an
    # array of shape one.
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce", backend=backend, model_kwargs=model_kwargs
    )
    assert model.get_backend() == backend
    assert isinstance(model.model, expected_auto_model_class)

    # Test encoding with the model
    scores = model.predict([("Hello, World!", "How are you?")])
    assert scores.shape == (1,)


def test_backend_no_export_crash():
    # [FUNCTION DOC]
    # Loads the plain reranker with export disabled: for ONNX expects OSError on optimum older than 1.25.0, otherwise
    # success with an ONNX model (export is auto-enabled); for OpenVINO expects OSError on optimum-intel older than
    # 1.19.0, otherwise success with an OpenVINO model.
    # Prior to optimum v1.25.0, ONNX Crashes when it can't export & the model repo/path doesn't contain an exported model
    # Since then, it auto-updates export to True
    with pytest.raises(OSError) if parse(optimum_version) < Version("1.25.0") else nullcontext():
        model = CrossEncoder(
            "cross-encoder-testing/reranker-bert-tiny-gooaq-bce", backend="onnx", model_kwargs={"export": False}
        )
        assert isinstance(model.model, ORTModelForSequenceClassification)

    # OpenVINO will forcibly override the export=False if the model repo/path doesn't contain an exported model
    # But only starting from optimum-intel=v1.19.0
    with pytest.raises(OSError) if parse(optimum_intel_version) < Version("1.19.0") else nullcontext():
        model = CrossEncoder(
            "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
            backend="openvino",
            model_kwargs={"export": False},
        )
        assert isinstance(model.model, OVModelForSequenceClassification)


## Testing loading exported models:
@pytest.mark.parametrize(
    ["backend", "model_id"],
    [
        ("onnx", "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-onnx"),
        ("openvino", "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-openvino"),
    ],
)
@pytest.mark.parametrize(
    ["model_kwargs", "exception"],
    [
        [{}, False],
        [{"file_name": "wrong_file_name", "export": True}, False],  # Using a file_name is fine when exporting
        [{"file_name": "wrong_file_name", "export": False}, True],  # ... but fails when not exporting
    ],
)
def test_backend_load(backend, model_id, model_kwargs, exception) -> None:
    # [FUNCTION DOC]
    # Parametrised over pre-exported ONNX/OpenVINO repositories and model kwargs. When the case expects an error
    # (wrong file name with export disabled), asserts loading raises OSError or RuntimeError; otherwise asserts the
    # backend and that predicting one pair gives shape one.
    if exception:
        with pytest.raises((OSError, RuntimeError)):
            CrossEncoder(model_id, backend=backend, model_kwargs=model_kwargs)
    else:
        model = CrossEncoder(model_id, backend=backend, model_kwargs=model_kwargs)
        assert model.get_backend() == backend
        scores = model.predict([("Hello, World!", "How are you?")])
        assert scores.shape == (1,)


def test_onnx_provider_crash() -> None:
    # [FUNCTION DOC]
    # Asserts loading the ONNX repository with an invalid execution provider raises ValueError.
    with pytest.raises(ValueError):
        CrossEncoder(
            "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-onnx",
            backend="onnx",
            model_kwargs={"provider": "incorrect_provider"},
        )


def test_openvino_provider() -> None:
    # [FUNCTION DOC]
    # Asserts that an OpenVINO config given as a dict ("f32" precision hint) or as a path to a JSON file ("f16") is
    # applied, with the performance hint defaulting to "LATENCY" in both cases. Writes a temporary JSON file.
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-openvino",
        backend="openvino",
        model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "f32"}},
    )
    assert model.model.ov_config == {
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        ov_config_path = os.path.join(temp_dir, "ov_config.json")
        with open(ov_config_path, "w") as ov_config_file:
            json.dump({"INFERENCE_PRECISION_HINT": "f16"}, ov_config_file)

        model = CrossEncoder(
            "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-openvino",
            backend="openvino",
            model_kwargs={"ov_config": ov_config_path},
        )
        assert model.model.ov_config == {
            "INFERENCE_PRECISION_HINT": "f16",
            "PERFORMANCE_HINT": "LATENCY",
        }


def test_incorrect_backend() -> None:
    # [FUNCTION DOC]
    # Asserts an unknown backend name raises ValueError.
    with pytest.raises(ValueError):
        CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", backend="incorrect_backend")


def test_openvino_backend() -> None:
    # [FUNCTION DOC]
    # Asserts OpenVINO predictions on two pairs match PyTorch within 1e-6 (f32 precision hint). Then loads with an
    # OpenVINO config file setting two inference streams and asserts the inference request reports 2 streams; saves
    # that model into the temporary directory, reloads it locally with OpenVINO and asserts its predictions match the
    # in-memory OpenVINO model. Deletes the local model and runs garbage collection before the directory is removed.
    model_id = "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"
    test_pairs = [("Hello there!", "General Kenobi!"), ("How are you?", "I'm fine, thanks.")]

    # Test that OpenVINO output is close to PyTorch output
    pytorch_model = CrossEncoder(model_id)
    openvino_model = CrossEncoder(
        model_id,
        backend="openvino",
        model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "f32"}},
    )
    pytorch_result = pytorch_model.predict(test_pairs)
    openvino_result = openvino_model.predict(test_pairs)
    assert np.allclose(openvino_result, pytorch_result, atol=0.000001), "OpenVINO and Pytorch outputs are not close"

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test that loading with ov_config file works as expected
        config_file = str(Path(tmpdirname) / "ov_config.json")
        with open(Path(config_file), "w") as f:
            f.write('{"NUM_STREAMS" : "2"}')
        openvino_model_with_config = CrossEncoder(
            model_id,
            backend="openvino",
            model_kwargs={"ov_config": config_file},
        )
        # The transformers model is an Optimum model with an OpenVINO inference request property
        assert openvino_model_with_config.model.request.get_property("NUM_STREAMS") == 2

        # Test that saving and loading local OpenVINO models works as expected
        openvino_model_with_config.save_pretrained(tmpdirname)
        local_openvino_model = CrossEncoder(
            tmpdirname, backend="openvino", model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "f32"}}
        )
        local_openvino_result = local_openvino_model.predict(test_pairs)
        assert np.allclose(local_openvino_result, openvino_result), (
            "OpenVINO saved model output differs from in-memory converted model"
        )
        del local_openvino_model
        gc.collect()


def test_export_false_subfolder() -> None:
    # [FUNCTION DOC]
    # Wraps the OpenVINO class's loading class method with an asserting decorator, then loads the pre-exported
    # OpenVINO repository with export disabled; the decorator asserts the loader received export false, subfolder
    # "openvino" and file name "openvino_model.xml".
    # KNOWN ISSUE: The class method is replaced globally and never restored, so the asserting wrapper leaks into every
    # later test in the session (and stacks with the next test's wrapper, whose calls may then fail on the export
    # assertion or a missing "export" key).
    model_id = "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-openvino"

    def from_pretrained_decorator(method):
        # [FUNCTION DOC]
        # HELPER: Takes a loader method and returns a wrapper that asserts the expected keyword arguments before
        # delegating.
        def decorator(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Asserts export is falsy, subfolder is "openvino" and file name is "openvino_model.xml" (KeyError
            # if any is absent), then calls the wrapped loader and returns its result.
            assert not kwargs["export"]
            assert kwargs["subfolder"] == "openvino"
            assert kwargs["file_name"] == "openvino_model.xml"
            return method(*args, **kwargs)

        return decorator

    OVModelForSequenceClassification.from_pretrained = from_pretrained_decorator(
        OVModelForSequenceClassification.from_pretrained
    )
    CrossEncoder(model_id, backend="openvino", model_kwargs={"export": False})


def test_export_set_nested_filename() -> None:
    # [FUNCTION DOC]
    # Wraps the OpenVINO loader with an asserting decorator and loads the pre-exported repository with file name
    # "openvino/openvino_model.xml", checking that the nested path is split into subfolder "openvino" and file name
    # "openvino_model.xml".
    # KNOWN ISSUE: Same unrestored global patch of the OpenVINO loader as the previous test; when run after it, the
    # earlier wrapper is still active as well.
    model_id = "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-openvino"

    def from_pretrained_decorator(method):
        # [FUNCTION DOC]
        # HELPER: Takes a loader method and returns a wrapper that asserts the subfolder and file name before
        # delegating.
        def decorator(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Asserts subfolder is "openvino" and file name is "openvino_model.xml", then calls the wrapped
            # loader and returns its result.
            assert kwargs["subfolder"] == "openvino"
            assert kwargs["file_name"] == "openvino_model.xml"
            return method(*args, **kwargs)

        return decorator

    OVModelForSequenceClassification.from_pretrained = from_pretrained_decorator(
        OVModelForSequenceClassification.from_pretrained
    )
    CrossEncoder(model_id, backend="openvino", model_kwargs={"file_name": "openvino/openvino_model.xml"})
