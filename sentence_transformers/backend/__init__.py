# [PACKAGE DOC]
# sentence_transformers.backend: support for running models on the ONNX Runtime and OpenVINO backends through Hugging
# Face Optimum. Re-exports the loaders (load_onnx_model, load_openvino_model), the exporters
# (export_optimized_onnx_model, export_dynamic_quantized_onnx_model, export_static_quantized_openvino_model) and the shared
# helpers from utils (_save_pretrained_wrapper, backend_should_export, backend_warn_to_save, save_or_push_to_hub_model).
# Importing it has no side effects beyond importing these submodules; Optimum itself is only imported lazily inside functions.
from __future__ import annotations

from .load import load_onnx_model, load_openvino_model
from .optimize import export_optimized_onnx_model
from .quantize import export_dynamic_quantized_onnx_model, export_static_quantized_openvino_model
from .utils import _save_pretrained_wrapper, backend_should_export, backend_warn_to_save, save_or_push_to_hub_model

__all__ = [
    "load_onnx_model",
    "load_openvino_model",
    "export_optimized_onnx_model",
    "export_dynamic_quantized_onnx_model",
    "export_static_quantized_openvino_model",
    "_save_pretrained_wrapper",
    "backend_should_export",
    "backend_warn_to_save",
    "save_or_push_to_hub_model",
]
