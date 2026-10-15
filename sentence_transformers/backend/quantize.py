# [MODULE DOC]
# Quantization exports: dynamic int8 quantization for ONNX-backed models (Optimum ORTQuantizer) and post-training static
# quantization with a calibration dataset for OpenVINO-backed models (Optimum-Intel OVQuantizer). Both delegate saving or
# uploading to save_or_push_to_hub_model.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from sentence_transformers.backend.utils import save_or_push_to_hub_model
from sentence_transformers.util import disable_datasets_caching, is_datasets_available

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder

    try:
        from optimum.intel import OVQuantizationConfig
    except ImportError:
        pass
    try:
        from optimum.onnxruntime.configuration import QuantizationConfig
    except ImportError:
        pass


def export_dynamic_quantized_onnx_model(
    model: SentenceTransformer | SparseEncoder | CrossEncoder | MultiVectorEncoder,
    quantization_config: QuantizationConfig | Literal["arm64", "avx2", "avx512", "avx512_vnni"],
    model_name_or_path: str,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str | None = None,
) -> None:
    # [FUNCTION DOC]
    # Dynamically quantize the ONNX model of a model loaded with backend "onnx" and save/upload it as onnx/model_<suffix>.onnx.
    # Inputs: model (transformers_model must be an ORTModel), quantization_config (a QuantizationConfig or one of "arm64",
    # "avx2", "avx512", "avx512_vnni", turned into the matching AutoQuantizationConfig preset with is_static False),
    # model_name_or_path, push_to_hub, create_pr, file_suffix. Default suffix: "<weights dtype lowercased>_<preset name>" for
    # string presets, else "<weights dtype lowercased>_quantized". Returns None.
    # Raises ImportError if Optimum/ONNX Runtime are missing, ValueError for a non-ONNX model or unknown preset.
    # Side effects: temporary files, local writes or Hub upload.
    """
    Export a quantized ONNX model from a SentenceTransformer, SparseEncoder, CrossEncoder,
    or MultiVectorEncoder model.

    This function applies dynamic quantization, i.e. without a calibration dataset.
    Each of the default quantization configurations quantize the model to int8, allowing
    for faster inference on CPUs, but are likely slower on GPUs.

    See the following pages for more information & benchmarks:

    - `Sentence Transformer > Speeding up Inference <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>`_
    - `Cross Encoder > Speeding up Inference <https://sbert.net/docs/cross_encoder/usage/efficiency.html>`_

    Args:
        model (SentenceTransformer | SparseEncoder | CrossEncoder | MultiVectorEncoder): The SentenceTransformer,
            SparseEncoder, CrossEncoder, or MultiVectorEncoder model to be quantized. Must be loaded with `backend="onnx"`.
        quantization_config (QuantizationConfig): The quantization configuration.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the quantized model will be saved.
        push_to_hub (bool, optional): Whether to push the quantized model to the Hugging Face Hub. Defaults to False.
        create_pr (bool, optional): Whether to create a pull request when pushing to the Hugging Face Hub. Defaults to False.
        file_suffix (str | None, optional): The suffix to add to the quantized model file name. Defaults to None.

    Raises:
        ImportError: If the required packages `optimum` and `onnxruntime` are not installed.
        ValueError: If the provided model is not a valid SentenceTransformer, SparseEncoder, CrossEncoder,
            or MultiVectorEncoder model loaded with `backend="onnx"`.
        ValueError: If the provided quantization_config is not valid.

    Returns:
        None
    """

    try:
        from optimum.onnxruntime import ORTModel, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        raise ImportError(
            "Please install Optimum and ONNX Runtime to use this function. "
            "You can install them with pip: `pip install sentence-transformers[onnx]` "
            "or `pip install sentence-transformers[onnx-gpu]`"
        )

    ort_model = model.transformers_model
    if not isinstance(ort_model, ORTModel):
        raise ValueError(
            "The model must be a Transformer-based SentenceTransformer, SparseEncoder, CrossEncoder, "
            'or MultiVectorEncoder model loaded with `backend="onnx"`.'
        )

    quantizer = ORTQuantizer.from_pretrained(ort_model)

    if isinstance(quantization_config, str):
        if quantization_config not in ["arm64", "avx2", "avx512", "avx512_vnni"]:
            raise ValueError(
                "quantization_config must be an QuantizationConfig instance or one of 'arm64', 'avx2', 'avx512', or 'avx512_vnni'."
            )

        quantization_config_name = quantization_config[:]
        quantization_config = getattr(AutoQuantizationConfig, quantization_config)(is_static=False)
        file_suffix = file_suffix or f"{quantization_config.weights_dtype.name.lower()}_{quantization_config_name}"

    if file_suffix is None:
        file_suffix = f"{quantization_config.weights_dtype.name.lower()}_quantized"

    save_or_push_to_hub_model(
        export_function=lambda save_dir: quantizer.quantize(quantization_config, save_dir, file_suffix=file_suffix),
        export_function_name="export_dynamic_quantized_onnx_model",
        config=quantization_config,
        model_name_or_path=model_name_or_path,
        push_to_hub=push_to_hub,
        create_pr=create_pr,
        file_suffix=file_suffix,
        backend="onnx",
        model=model,
    )


def export_static_quantized_openvino_model(
    model: SentenceTransformer | SparseEncoder | CrossEncoder | MultiVectorEncoder,
    quantization_config: OVQuantizationConfig | dict | None,
    model_name_or_path: str,
    dataset_name: str | None = None,
    dataset_config_name: str | None = None,
    dataset_split: str | None = None,
    column_name: str | None = None,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str = "qint8_quantized",
) -> None:
    # [FUNCTION DOC]
    # Statically quantize the OpenVINO model of a model loaded with backend "openvino" using a calibration dataset, and
    # save/upload it as openvino/openvino_model_<suffix>.xml plus the matching .bin.
    # Inputs: model (transformers_model must be an OVModel), quantization_config (OVQuantizationConfig, dict, or None meaning a
    # default OVQuantizationConfig), model_name_or_path, dataset_name/dataset_config_name/dataset_split/column_name (all four
    # given or all omitted; defaults nyu-mll/glue, sst2, train, sentence), push_to_hub, create_pr, file_suffix (default
    # "qint8_quantized"). Builds an OVConfig and OVQuantizer, loads the calibration dataset with datasets caching disabled,
    # tokenizing the chosen column to padded length 384, with num_samples taken from the config. Returns None.
    # Raises ImportError if optimum-intel/openvino or datasets are missing, ValueError for a non-OpenVINO model or a partial set
    # of dataset arguments. Side effects: dataset download, temporary files, local writes or Hub upload.
    # KNOWN ISSUE: the docstring allows a dict quantization_config, but the code reads its num_samples attribute, which fails for a dict; the fallback of 300 samples is dead code because the config is never None at that point.
    """
    Export a quantized OpenVINO model from a SentenceTransformer, SparseEncoder, CrossEncoder,
    or MultiVectorEncoder model.

    This function applies Post-Training Static Quantization (PTQ) using a calibration dataset, which calibrates
    quantization constants without requiring model retraining. Each default quantization configuration converts
    the model to int8 precision, enabling faster inference while maintaining accuracy.

    See the following pages for more information & benchmarks:

    - `Sentence Transformer > Speeding up Inference <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>`_
    - `Cross Encoder > Speeding up Inference <https://sbert.net/docs/cross_encoder/usage/efficiency.html>`_

    Args:
        model (SentenceTransformer | SparseEncoder | CrossEncoder | MultiVectorEncoder): The SentenceTransformer,
            SparseEncoder, CrossEncoder, or MultiVectorEncoder model to be quantized. Must be loaded with `backend="openvino"`.
        quantization_config (OVQuantizationConfig | dict | None): The quantization configuration. If None, default values are used.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the quantized model will be saved.
        dataset_name(str, optional): The name of the dataset to load for calibration.
            If not specified, the `sst2` subset of the `glue` dataset will be used by default.
        dataset_config_name (str, optional): The specific configuration of the dataset to load.
        dataset_split (str, optional): The split of the dataset to load (e.g., 'train', 'test'). Defaults to None.
        column_name (str, optional): The column name in the dataset to use for calibration. Defaults to None.
        push_to_hub (bool, optional): Whether to push the quantized model to the Hugging Face Hub. Defaults to False.
        create_pr (bool, optional): Whether to create a pull request when pushing to the Hugging Face Hub. Defaults to False.
        file_suffix (str, optional): The suffix to add to the quantized model file name. Defaults to `qint8_quantized`.

    Raises:
        ImportError: If the required packages `optimum` and `openvino` are not installed.
        ValueError: If the provided model is not a valid SentenceTransformer, SparseEncoder, CrossEncoder,
            or MultiVectorEncoder model loaded with `backend="openvino"`.
        ValueError: If the provided quantization_config is not valid.

    Returns:
        None
    """
    try:
        from optimum.intel.openvino import (
            OVConfig,
            OVQuantizationConfig,
            OVQuantizer,
        )
        from optimum.intel.openvino.modeling import OVModel
    except ImportError:
        raise ImportError(
            "Please install datasets, optimum-intel and openvino to use this function. "
            "You can install them with pip: `pip install datasets sentence-transformers[openvino]`"
        )
    if not is_datasets_available():
        raise ImportError(
            "Please install datasets to use this function. You can install it with pip: `pip install datasets`"
        )

    ov_model = model.transformers_model
    if not isinstance(ov_model, OVModel):
        raise ValueError(
            "The model must be a Transformer-based SentenceTransformer, SparseEncoder, CrossEncoder, "
            'or MultiVectorEncoder model loaded with `backend="openvino"`.'
        )

    if quantization_config is None:
        quantization_config = OVQuantizationConfig()

    ov_config = OVConfig(quantization_config=quantization_config)
    quantizer = OVQuantizer.from_pretrained(ov_model)

    if any(param is not None for param in [dataset_name, dataset_config_name, dataset_split, column_name]) and not all(
        param is not None for param in [dataset_name, dataset_config_name, dataset_split, column_name]
    ):
        raise ValueError(
            "Either specify all of `dataset_name`, `dataset_config_name`, `dataset_split`, and `column_name`, or leave them all unspecified."
        )

    def preprocess_function(examples):
        # [FUNCTION DOC]
        # HELPER: tokenize a list of calibration texts with the model's tokenizer, padding to max length 384 and truncating.
        return model.tokenizer(examples, padding="max_length", max_length=384, truncation=True)

    dataset_name = dataset_name if dataset_name is not None else "nyu-mll/glue"
    dataset_config_name = dataset_config_name if dataset_config_name is not None else "sst2"
    dataset_split = dataset_split if dataset_split is not None else "train"
    column_name = column_name if column_name is not None else "sentence"
    with disable_datasets_caching():
        calibration_dataset = quantizer.get_calibration_dataset(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            preprocess_function=lambda examples: preprocess_function(examples[column_name]),
            num_samples=quantization_config.num_samples if quantization_config is not None else 300,
            dataset_split=dataset_split,
        )

    save_or_push_to_hub_model(
        export_function=lambda save_dir: quantizer.quantize(
            calibration_dataset, save_directory=save_dir, ov_config=ov_config
        ),
        export_function_name="export_static_quantized_openvino_model",
        config=quantization_config,
        model_name_or_path=model_name_or_path,
        push_to_hub=push_to_hub,
        create_pr=create_pr,
        file_suffix=file_suffix,
        backend="openvino",
        model=model,
    )
