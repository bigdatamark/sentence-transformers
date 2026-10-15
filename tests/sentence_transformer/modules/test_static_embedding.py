# [MODULE DOC]
# Tests for the StaticEmbedding module (a token-embedding lookup averaged into a sentence embedding, no transformer): construction,
# tokenization, forward, save/load, creation via model2vec distillation or download, and modality validation. Defines skip markers
# for when model2vec is missing and for transformers 5 or newer. Relies on tokenizer, embedding_weights and static_embedding fixtures
# (BERT vocabulary of 30522 tokens, 768-D) from conftest.
from __future__ import annotations

import math
from pathlib import Path

import pytest
from packaging.version import Version
from tokenizers import Tokenizer
from transformers import __version__ as transformers_version

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules.static_embedding import StaticEmbedding

try:
    import model2vec
except ImportError:
    model2vec = None

skip_if_no_model2vec = pytest.mark.skipif(model2vec is None, reason="The model2vec library is not installed.")
skip_if_transformers_5_or_higher = pytest.mark.skipif(
    Version(transformers_version) >= Version("5.0.0rc0"), reason="Transformers version is v5.0.0rc0 or higher."
)


def test_initialization_with_embedding_weights(tokenizer: Tokenizer, embedding_weights) -> None:
    # [FUNCTION DOC]
    # Asserts that constructing from a tokenizer and given embedding weights yields a 30522x768 embedding matrix.
    model = StaticEmbedding(tokenizer, embedding_weights=embedding_weights)
    assert model.embedding.weight.shape == (30522, 768)


def test_initialization_with_embedding_dim(tokenizer: Tokenizer) -> None:
    # [FUNCTION DOC]
    # Asserts that constructing from a tokenizer and only an embedding dimension of 768 yields a randomly initialized 30522x768 matrix.
    model = StaticEmbedding(tokenizer, embedding_dim=768)
    assert model.embedding.weight.shape == (30522, 768)


def test_tokenize(static_embedding: StaticEmbedding) -> None:
    # [FUNCTION DOC]
    # Asserts preprocessing two texts returns features containing "input_ids" and "offsets" (flattened bag-of-tokens layout).
    texts = ["Hello world!", "How are you?"]
    tokens = static_embedding.preprocess(texts)
    assert "input_ids" in tokens
    assert "offsets" in tokens


def test_forward(static_embedding: StaticEmbedding) -> None:
    # [FUNCTION DOC]
    # Asserts that running the module on preprocessed texts returns a dict containing "sentence_embedding".
    texts = ["Hello world!", "How are you?"]
    tokens = static_embedding.preprocess(texts)
    output = static_embedding(tokens)
    assert "sentence_embedding" in output


def test_save_and_load(tmp_path: Path, static_embedding: StaticEmbedding) -> None:
    # [FUNCTION DOC]
    # Saves the module to a new subdirectory of tmp_path, reloads it, and asserts the embedding matrix shape is unchanged. Writes files.
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    static_embedding.save(str(save_dir))

    loaded_model = StaticEmbedding.load(str(save_dir))
    assert loaded_model.embedding.weight.shape == static_embedding.embedding.weight.shape


@skip_if_transformers_5_or_higher()  # Model2vec distillation is not yet compatible with transformers v5+
@skip_if_no_model2vec()
def test_from_distillation() -> None:
    # [FUNCTION DOC]
    # Skipped without model2vec or on transformers 5+. Distills a static embedding from a tiny BERT model with PCA to 32 dims and asserts
    # the vocabulary size is within 5 of 29525 (it varies across model2vec versions) and the dimension is 32. Downloads a model.
    model = StaticEmbedding.from_distillation("sentence-transformers-testing/stsb-bert-tiny-safetensors", pca_dims=32)
    # The shape has been 29528 for <0.5.0, 29525 for 0.5.0, and 29524 for >=0.6.0, so let's make a safer test
    # that checks the first dimension is close to 29525 and the second dimension is 32.
    assert abs(model.embedding.weight.shape[0] - 29525) < 5
    assert model.embedding.weight.shape[1] == 32


@skip_if_no_model2vec()
def test_from_model2vec() -> None:
    # [FUNCTION DOC]
    # Skipped without model2vec. Loads a model2vec model from the Hub and asserts a 29528x256 embedding matrix. Network.
    model = StaticEmbedding.from_model2vec("minishlab/M2V_base_output")
    assert model.embedding.weight.shape == (29528, 256)


def test_unsupported_modality(static_embedding: StaticEmbedding) -> None:
    # [FUNCTION DOC]
    # Wraps the static embedding in a text-only SentenceTransformer and asserts ValueError with specific messages for an image-only input,
    # for a combined text+image dict, and for a batch mixing a text+image dict with a text-only dict (naming image as unsupported).
    from PIL import Image

    model = SentenceTransformer(modules=[static_embedding])
    dummy_image = Image.new("RGB", (10, 10))

    # Image-only input
    with pytest.raises(
        ValueError,
        match="Modality 'image' is not supported by this SentenceTransformer model. Supported modalities: text",
    ):
        model.encode([dummy_image])

    # Mixed text+image input via multimodal dict
    with pytest.raises(
        ValueError,
        match="Modality 'image\\+text' is not supported by this SentenceTransformer model. Supported modalities: text",
    ):
        model.encode([{"text": "a cat", "image": dummy_image}])

    # Mixed-modality batch (a multimodal dict alongside a text-only dict) is inferred as 'message'.
    # The model genuinely cannot handle image, so it is named as the unsupported modality.
    with pytest.raises(
        ValueError,
        match=(
            r"This batch mixes multiple modalities \(image\+text, text\), but this SentenceTransformer "
            r"model does not support image\. Supported modalities: text"
        ),
    ):
        model.encode([{"text": "a cat", "image": dummy_image}, {"text": "a dog"}])


def test_loading_model2vec() -> None:
    # [FUNCTION DOC]
    # Loads a model2vec-format model directly as a SentenceTransformer from the Hub; asserts embedding dimension 256, unbounded max
    # sequence length, a 2x256 output for two sentences, and similarity above 0.7 between two sentences about sunshine. Network.
    model = SentenceTransformer("minishlab/potion-base-8M")
    assert model.get_embedding_dimension() == 256
    assert model.max_seq_length == math.inf

    test_sentences = ["It's so sunny outside!", "The sun is shining outside!"]
    embeddings = model.encode(test_sentences)
    assert embeddings.shape == (2, 256)
    similarity = model.similarity(embeddings[0], embeddings[1])
    assert similarity.item() > 0.7
