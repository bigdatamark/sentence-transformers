# [MODULE DOC]
# HELPER: Root pytest conftest for the whole test suite. Provides (a) autouse fixtures that reset transformers'
# global "warning once" cache before each test and that mock Hub metadata lookups made by model-card generation
# for the whole session, and (b) shared model, tokenizer, embedding and dataset fixtures. Expensive models are
# loaded once per session by private session-scoped fixtures (leading underscore); public function-scoped fixtures
# return a deep copy so tests can mutate them freely. Most fixtures download from the Hugging Face Hub (network or
# local cache). The datasets library is imported only when available.
from __future__ import annotations

import logging
import os
import tempfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub.utils import validate_repo_id
from tokenizers import Tokenizer

from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.sentence_transformer.modules import Pooling, StaticEmbedding, Transformer
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import DatasetDict, load_dataset


def _clear_warning_once_cache() -> None:
    # [FUNCTION DOC]
    # HELPER: Clears the global cache behind the logger's "warning once" method if that method is wrapped in an LRU
    # cache (as transformers does); otherwise does nothing. No inputs, returns nothing. Mutates global logging state.
    # Nothing to clear when warning_once is not the lru_cache-wrapped transformers version, as then
    # it does not suppress repeats in the first place.
    cache_clear = getattr(getattr(logging.Logger, "warning_once", None), "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


@pytest.fixture(autouse=True)
def clear_warning_once_cache():
    # [FUNCTION DOC]
    # HELPER: Autouse fixture run for every test. Clears the warning-once cache before the test and returns the
    # clearing function so a test can request it by name and clear the cache again mid-test.
    """transformers caches ``warning_once`` globally, so without this a warning emitted by one test
    silences it in every later test. Request the fixture by name to clear the cache again part-way
    through a test.
    """
    _clear_warning_once_cache()
    return _clear_warning_once_cache


def _fake_model_info(model_id: str) -> SimpleNamespace:
    # [FUNCTION DOC]
    # HELPER: Fake replacement for the Hub "get model info" call. Validates the string as a Hub repo id (raising the
    # Hub validation error if malformed), raises ValueError if the string is an existing local path, and otherwise
    # returns a simple namespace with the given id and a fixed 40-character fake commit sha. Reads the filesystem only.
    validate_repo_id(model_id)
    if os.path.exists(model_id):
        raise ValueError(f"{model_id!r} is a local path, not a Hub model ID")
    return SimpleNamespace(id=model_id, sha="0123456789abcdef0123456789abcdef01234567")


def _fake_dataset_info(dataset_id: str) -> SimpleNamespace:
    # [FUNCTION DOC]
    # HELPER: Fake replacement for the Hub "get dataset info" call. Same validation as the model variant (invalid repo
    # id error, ValueError for an existing local path); returns a simple namespace with the id and no card data.
    validate_repo_id(dataset_id)
    if os.path.exists(dataset_id):
        raise ValueError(f"{dataset_id!r} is a local path, not a Hub dataset ID")
    return SimpleNamespace(id=dataset_id, cardData=None)


@pytest.fixture(scope="session", autouse=True)
def mock_hub_info():
    # [FUNCTION DOC]
    # HELPER: Session-scoped autouse fixture. For the whole session, monkeypatches the model-card module's model-info
    # and dataset-info lookups with the two fakes above so no Hub requests are made; restores them on teardown.
    # Yields nothing.
    """Mock the Hub requests checking whether base models and datasets exist, e.g. to dodge rate limits in CI."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("sentence_transformers.base.model_card.get_model_info", _fake_model_info)
        monkeypatch.setattr("sentence_transformers.base.model_card.get_dataset_info", _fake_dataset_info)
        yield


# Sentence Transformers
@pytest.fixture(scope="session")
def _stsb_bert_tiny_model() -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture. Loads the tiny STSB BERT SentenceTransformer (safetensors weights), disables
    # widget example generation on its model card data, and returns it.
    model_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = SentenceTransformer(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def stsb_bert_tiny_model(_stsb_bert_tiny_model: SentenceTransformer) -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture returning a deep copy of the session tiny STSB BERT model.
    return deepcopy(_stsb_bert_tiny_model)


@pytest.fixture(scope="session")
def _avg_word_embeddings_levy() -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture. Loads the average-word-embeddings (Levy dependency) SentenceTransformer, disables
    # widget example generation, and returns it.
    model_id = "sentence-transformers/average_word_embeddings_levy_dependency"
    model = SentenceTransformer(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def avg_word_embeddings_levy(_avg_word_embeddings_levy: SentenceTransformer) -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture returning a deep copy of the session average-word-embeddings model.
    return deepcopy(_avg_word_embeddings_levy)


@pytest.fixture()
def stsb_bert_tiny_model_onnx() -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Fixture loading the tiny STSB BERT SentenceTransformer repository variant that ships ONNX weights.
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-onnx")


@pytest.fixture()
def stsb_bert_tiny_model_openvino() -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Fixture loading the tiny STSB BERT SentenceTransformer repository variant that ships OpenVINO weights.
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-openvino")


@pytest.fixture()
def paraphrase_distilroberta_base_v1_model() -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Fixture loading the paraphrase-distilroberta-base-v1 SentenceTransformer (fresh per test).
    return SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1")


@pytest.fixture(scope="session")
def _static_retrieval_mrl_en_v1_model() -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture loading the static-retrieval-mrl-en-v1 SentenceTransformer (StaticEmbedding based).
    model_id = "sentence-transformers/static-retrieval-mrl-en-v1"
    return SentenceTransformer(model_id)


@pytest.fixture()
def static_retrieval_mrl_en_v1_model(_static_retrieval_mrl_en_v1_model: SentenceTransformer) -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture returning a deep copy of the session static retrieval MRL model.
    return deepcopy(_static_retrieval_mrl_en_v1_model)


@pytest.fixture()
def clip_vit_b_32_model() -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Fixture loading the CLIP ViT-B-32 multimodal SentenceTransformer (fresh per test).
    return SentenceTransformer("sentence-transformers/clip-ViT-B-32")


@pytest.fixture(scope="session")
def _distilbert_base_uncased_model() -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture building a SentenceTransformer from scratch: a Transformer module for
    # distilbert-base-uncased followed by a Pooling module sized to the transformer's embedding dimension. If the model
    # card data has no base model recorded, sets it to the distilbert model id. Returns the model.
    model_id = "distilbert/distilbert-base-uncased"
    word_embedding_model = Transformer(model_id)
    pooling_model = Pooling(word_embedding_model.get_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if not model.model_card_data.base_model:
        model.model_card_data.base_model = model_id
    return model


@pytest.fixture()
def distilbert_base_uncased_model(_distilbert_base_uncased_model: SentenceTransformer) -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture returning a deep copy of the session distilbert-plus-pooling model.
    return deepcopy(_distilbert_base_uncased_model)


# Cross Encoders
@pytest.fixture(scope="session")
def _reranker_bert_tiny_model() -> CrossEncoder:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture loading the tiny GooAQ BCE reranker CrossEncoder, with widget example generation
    # disabled. Returns it.
    model_id = "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"
    model = CrossEncoder(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def reranker_bert_tiny_model(_reranker_bert_tiny_model) -> CrossEncoder:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture returning a deep copy of the session tiny reranker CrossEncoder.
    return deepcopy(_reranker_bert_tiny_model)


# Sparse Encoders
@pytest.fixture(scope="session")
def _splade_bert_tiny_model() -> SparseEncoder:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture loading the tiny SPLADE BERT SparseEncoder (NQ), widget examples disabled.
    model_id = "sparse-encoder-testing/splade-bert-tiny-nq"
    model = SparseEncoder(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def splade_bert_tiny_model(_splade_bert_tiny_model: SparseEncoder) -> SparseEncoder:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture returning a deep copy of the session tiny SPLADE model.
    return deepcopy(_splade_bert_tiny_model)


@pytest.fixture(scope="session")
def _inference_free_splade_bert_tiny_model() -> SparseEncoder:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture loading the tiny inference-free SPLADE BERT SparseEncoder, widget examples disabled.
    model_id = "sparse-encoder-testing/inference-free-splade-bert-tiny-nq"
    model = SparseEncoder(model_id)
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def inference_free_splade_bert_tiny_model(_inference_free_splade_bert_tiny_model: SparseEncoder) -> SparseEncoder:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture returning a deep copy of the session inference-free SPLADE model.
    return deepcopy(_inference_free_splade_bert_tiny_model)


@pytest.fixture(scope="session")
def _csr_bert_tiny_model() -> SparseEncoder:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture building a CSR (sparse autoencoder) SparseEncoder from the tiny STSB BERT dense
    # checkpoint. Sets the last module's top-k to 16 and auxiliary top-k to 32, disables widget examples, returns it.
    model_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = SparseEncoder(model_id)
    model[-1].k = 16
    model[-1].k_aux = 32
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
    return model


@pytest.fixture()
def csr_bert_tiny_model(_csr_bert_tiny_model: SparseEncoder) -> SparseEncoder:
    # [FUNCTION DOC]
    # HELPER: Function-scoped fixture returning a deep copy of the session CSR model.
    return deepcopy(_csr_bert_tiny_model)


# Tokenization & Datasets
@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture loading the bert-base-uncased fast tokenizer (tokenizers library) from the Hub.
    return Tokenizer.from_pretrained("google-bert/bert-base-uncased")


@pytest.fixture
def embedding_weights():
    # [FUNCTION DOC]
    # HELPER: Fixture returning a random float32 matrix of shape 30522 by 768 (BERT vocabulary size by hidden size),
    # drawn uniformly from 0 to 1 with the global NumPy random state.
    return np.random.rand(30522, 768).astype(np.float32)


@pytest.fixture
def static_embedding(tokenizer: Tokenizer, embedding_weights) -> StaticEmbedding:
    # [FUNCTION DOC]
    # HELPER: Fixture building a StaticEmbedding module from the tokenizer and random embedding weights fixtures.
    return StaticEmbedding(tokenizer, embedding_weights=embedding_weights)


@pytest.fixture
def static_embedding_model(static_embedding: StaticEmbedding) -> SentenceTransformer:
    # [FUNCTION DOC]
    # HELPER: Fixture wrapping the static embedding module in a single-module SentenceTransformer with widget example
    # generation disabled.
    model = SentenceTransformer(modules=[static_embedding])
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.fixture(scope="session")
def stsb_dataset_dict() -> DatasetDict:
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture loading the "sentence-transformers/stsb" dataset as a DatasetDict (all splits).
    # Requires the datasets library.
    return load_dataset("sentence-transformers/stsb")


@pytest.fixture()
def cache_dir():
    # [FUNCTION DOC]
    # HELPER: Generator fixture supplying a cache directory for model downloads. If the CI environment variable is set,
    # yields a fresh temporary directory (removed afterwards, ignoring cleanup errors for Windows); otherwise yields
    # None so the default Hugging Face cache is used.
    """
    In the CI environment, we use a temporary directory as `cache_dir`
    to avoid keeping the downloaded models on disk after the test.
    """
    if os.environ.get("CI", None):
        # Note: `ignore_cleanup_errors=True` is used to avoid NotADirectoryError in Windows on GitHub Actions.
        # See https://github.com/python/cpython/issues/107408, https://www.scivision.dev/python-tempfile-permission-error-windows/
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            yield tmp_dir
    else:
        yield None
