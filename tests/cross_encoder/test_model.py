# [MODULE DOC]
# Pytest tests for the CrossEncoder model class: loading options (classifier dropout, revision, device, dtype,
# deprecated argument names), predict and rank behaviour (input shapes, output types, softmax, low precision, empty
# and NumPy inputs, per-call processing kwargs), serialization, number of labels and default activation for module
# stacks without a Transformer, Hub pushing with mocked HfApi, argument-renaming decorators, activation function
# persistence, chat-template handling for query/document pair roles, Qwen3 reranker parity (CUDA only), and conversion
# from a SentenceTransformer checkpoint. Many tests download tiny models from the Hub; fixtures come from the root and
# cross-encoder conftests. Module constants include a ChatML template that echoes roles, a structured-content rerank
# template, two Berlin query/document pairs, and the error substring used when a chat template cannot carry the pair
# roles.
from __future__ import annotations

import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from huggingface_hub import CommitInfo, HfApi, RepoUrl
from packaging.version import Version, parse
from PIL import Image
from pytest import FixtureRequest
from transformers import __version__ as transformers_version

from sentence_transformers import CrossEncoder
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.util import fullname
from sentence_transformers.util.decorators import (
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
)
from tests.utils import GENERIC_INSTRUCT_TEMPLATE, RERANKER_TEMPLATE, skip_bfloat16_cpu_crash


def test_classifier_dropout_is_set() -> None:
    # [FUNCTION DOC]
    # Loads the tiny reranker with a classifier dropout of 0.1234 and asserts it is set on both the CrossEncoder
    # config and the underlying transformer config.
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", classifier_dropout=0.1234)
    assert model.config.classifier_dropout == 0.1234
    assert model.model.config.classifier_dropout == 0.1234


def test_classifier_dropout_default_value(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts the tiny reranker fixture has no classifier dropout set (None) on both configs.
    model = reranker_bert_tiny_model
    assert model.config.classifier_dropout is None
    assert model.model.config.classifier_dropout is None


def test_load_with_revision() -> None:
    # [FUNCTION DOC]
    # Loads the tiny STSB BERT checkpoint (one label, randomly initialised classifier) at revision main, at a commit
    # equal to main, and at an older commit; copies main's classifier weight and bias into the other two, then asserts
    # predictions on one pair are identical for main and the equal commit and different for the older commit.
    model_name = "sentence-transformers-testing/stsb-bert-tiny-safetensors"

    main_model = CrossEncoder(model_name, num_labels=1, revision="main")
    latest_model = CrossEncoder(
        model_name,
        num_labels=1,
        revision="f3cb857cba53019a20df283396bcca179cf051a4",
    )
    older_model = CrossEncoder(
        model_name,
        num_labels=1,
        revision="ba33022fdf0b0fc2643263f0726f44d0a07d0e24",
    )

    # Set the classifier.bias and classifier.weight equal among models. This
    # is needed because the AutoModelForSequenceClassification randomly initializes
    # the classifier.bias and classifier.weight for each (model) initialization.
    # The test is only possible if all models have the same classifier.bias
    # and classifier.weight parameters.
    latest_model.model.classifier.bias = main_model.model.classifier.bias
    latest_model.model.classifier.weight = main_model.model.classifier.weight
    older_model.model.classifier.bias = main_model.model.classifier.bias
    older_model.model.classifier.weight = main_model.model.classifier.weight

    test_sentences = [["Hello there!", "Hello, World!"]]
    main_prob = main_model.predict(test_sentences, convert_to_tensor=True)
    assert torch.equal(main_prob, latest_model.predict(test_sentences, convert_to_tensor=True))
    assert not torch.equal(main_prob, older_model.predict(test_sentences, convert_to_tensor=True))


@pytest.mark.parametrize(
    argnames="return_documents",
    argvalues=[True, False],
    ids=["return-docs", "no-return-docs"],
)
def test_rank(return_documents: bool, request: FixtureRequest) -> None:
    # [FUNCTION DOC]
    # Parametrised on returning documents. Ranks nine sentences against the query "A man is eating pasta." with the
    # pretrained stsb-distilroberta-base cross-encoder and asserts the corpus id order equals the expected ranking;
    # when documents are returned (identified via the parametrize id), also asserts the returned texts are exactly the
    # corpus.
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    # We want to compute the similarity between the query sentence
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]
    expected_ranking = [0, 1, 3, 6, 2, 5, 7, 4, 8]

    # 1. We rank all sentences in the corpus for the query
    ranks = model.rank(query=query, documents=corpus, return_documents=return_documents)
    if request.node.callspec.id == "return-docs":
        assert {*corpus} == {rank.get("text") for rank in ranks}

    pred_ranking = [rank["corpus_id"] for rank in ranks]
    assert pred_ranking == expected_ranking


def test_rank_scores_are_floats(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts rank results contain plain Python float scores and are JSON serialisable.
    ranks = reranker_bert_tiny_model.rank("A man is eating pasta.", ["A man is eating food.", "A monkey is playing."])
    assert [type(rank["score"]) for rank in ranks] == [float, float]
    json.dumps(ranks)


@pytest.mark.parametrize("kwarg", ["convert_to_numpy", "convert_to_tensor"])
def test_rank_convert_kwargs_deprecated(reranker_bert_tiny_model: CrossEncoder, kwarg: str, caplog) -> None:
    # [FUNCTION DOC]
    # Parametrised over the removed convert-to-numpy and convert-to-tensor keywords. Asserts passing one to rank logs
    # a warning naming it and still returns float scores.
    # Annotated as Any so type checkers don't match the removed kwarg against every remaining parameter
    deprecated_kwargs: dict[str, Any] = {kwarg: True}
    with caplog.at_level(logging.WARNING):
        ranks = reranker_bert_tiny_model.rank("A man is eating pasta.", ["A man is eating food."], **deprecated_kwargs)

    assert kwarg in caplog.text
    assert type(ranks[0]["score"]) is float


def test_rank_multiple_labels(nli_minilm_model: CrossEncoder):
    # [FUNCTION DOC]
    # Asserts rank on the three-label NLI model raises ValueError stating rank only works with one label and
    # suggesting predict.
    model = nli_minilm_model
    with pytest.raises(
        ValueError,
        match=re.escape(
            "CrossEncoder.rank() only works for models with num_labels=1. "
            "Consider using CrossEncoder.predict() with input pairs instead."
        ),
    ):
        model.rank(
            query="A man is eating pasta.",
            documents=[
                "A man is eating food.",
                "A man is eating a piece of bread.",
                "The girl is carrying a baby.",
            ],
        )


def test_predict_softmax(nli_minilm_model: CrossEncoder):
    # [FUNCTION DOC]
    # On the NLI model, asserts that with softmax applied each pair's three scores sum to 1, and that without softmax
    # they do not all sum to 1.
    model = nli_minilm_model
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
    ]
    scores = model.predict([(query, doc) for doc in corpus], apply_softmax=True, convert_to_tensor=True)
    assert torch.isclose(scores.sum(1), torch.ones(len(corpus), device=scores.device)).all()
    scores = model.predict([(query, doc) for doc in corpus], apply_softmax=False, convert_to_tensor=True)
    assert not torch.isclose(scores.sum(1), torch.ones(len(corpus), device=scores.device)).all()


@skip_bfloat16_cpu_crash
def test_predict_low_precision_logits_upcast_to_float32(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Skipped where bfloat16 CPU crashes; xfails on CPU with torch 2.5.0 or older. Predicts two pairs in float32 as
    # reference, casts the underlying transformer to bfloat16, predicts again, and asserts the result is still float32
    # and within 1e-2 of the reference (activation applied after upcasting logits).
    """The activation over cross-encoder logits must run in float32 for low-precision models (HPS).

    Applying sigmoid/softmax to float16/bfloat16 logits buckets the (0, 1) probability range
    coarsely, collapsing distinct relevance scores into spurious ties.
    """
    model = reranker_bert_tiny_model
    if model.device.type == "cpu" and Version(torch.__version__) <= Version("2.5.0"):
        pytest.xfail("bfloat16 CPU inference is unreliable on older torch")
    pairs = [
        ("A man is eating pasta.", "A man is eating food."),
        ("A man is eating pasta.", "The girl is carrying a baby."),
    ]
    reference = model.predict(pairs, convert_to_tensor=True)
    assert reference.dtype == torch.float32

    model.model.to(torch.bfloat16)
    scores = model.predict(pairs, convert_to_tensor=True)

    assert scores.dtype == torch.float32
    assert torch.allclose(scores, reference, atol=1e-2)


@pytest.mark.parametrize("model_fixture", ["reranker_bert_tiny_model", "nli_minilm_model"])
def test_predict_single_input(model_fixture: str, request: FixtureRequest):
    # [FUNCTION DOC]
    # Parametrised over the tiny reranker and the NLI model (resolved via the fixture request). Asserts a list
    # containing one pair returns an array of shape one (or one by the label count), and a bare pair returns a NumPy
    # float32 scalar for one label or a vector of label-count length otherwise.
    model = request.getfixturevalue(model_fixture)
    nested_pair_score = model.predict([["A man is eating pasta.", "A man is eating food."]])
    assert isinstance(nested_pair_score, np.ndarray)
    if model.num_labels == 1:
        assert nested_pair_score.shape == (1,)
    else:
        assert nested_pair_score.shape == (1, model.num_labels)

    pair_score = model.predict(["A man is eating pasta.", "A man is eating food."])
    if model.num_labels == 1:
        assert isinstance(pair_score, np.float32)
    else:
        assert isinstance(pair_score, np.ndarray)
        assert pair_score.shape == (model.num_labels,)


def test_is_singular_input_numpy_1d_pair(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts a 1-D NumPy string array is considered a single input.
    """A 1D numpy string array represents a single (query, document) pair."""
    assert reranker_bert_tiny_model.is_singular_input(np.array(["query", "document"])) is True


def test_is_singular_input_numpy_2d_pairs(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts a 2-D NumPy string array is considered a batch, not a single input.
    """A 2D numpy string array is a batch of pairs."""
    assert reranker_bert_tiny_model.is_singular_input(np.array([["q1", "d1"], ["q2", "d2"]])) is False


def test_is_singular_input_numpy_empty(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts an empty NumPy string array is considered an empty batch, not a single input.
    """An empty 1D string ndarray is an empty batch, not a singular pair, matching ``predict([])``."""
    assert reranker_bert_tiny_model.is_singular_input(np.array([], dtype=str)) is False


def test_predict_numpy_empty(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts predicting an empty NumPy string array returns an empty array of shape zero equal to predicting an empty
    # list.
    """Predicting on an empty string ndarray should return an empty array, like ``predict([])``."""
    scores = reranker_bert_tiny_model.predict(np.array([], dtype=str), show_progress_bar=False)
    expected = reranker_bert_tiny_model.predict([], show_progress_bar=False)
    assert scores.shape == (0,)
    assert np.array_equal(scores, expected)


def test_predict_numpy_1d_pair(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts predicting a 1-D NumPy pair returns a float32 scalar close to predicting the equivalent tuple.
    """Predicting on a 1D numpy string array (a single pair) should match the tuple equivalent
    and return a scalar score. Exercises the singular-branch .tolist() conversion."""
    model = reranker_bert_tiny_model
    pair = np.array(["what is AI?", "AI is artificial intelligence."])
    score = model.predict(pair, show_progress_bar=False)
    expected = model.predict(tuple(pair.tolist()), show_progress_bar=False)
    assert isinstance(score, np.float32)
    assert np.allclose(score, expected)


def test_predict_numpy_2d_pairs(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts predicting a 2-D NumPy array of two pairs returns two scores matching the nested-list equivalent.
    """Predicting on a 2D numpy string array should match predicting on the equivalent nested list."""
    pairs = np.array([["what is AI?", "AI is artificial intelligence."], ["what is ML?", "ML is machine learning."]])
    scores = reranker_bert_tiny_model.predict(pairs, show_progress_bar=False)
    expected = reranker_bert_tiny_model.predict(pairs.tolist(), show_progress_bar=False)
    assert scores.shape == (2,)
    assert np.allclose(scores, expected)


def test_predict_batch_size_1(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Regression test: replaces the model's forward with a wrapper that squeezes the trailing dimension of the
    # "scores" output (mimicking models that return one-dimensional scores), predicts two pairs with batch size 1, and
    # asserts a NumPy array of shape two (no zero-dimensional collapse).
    """Regression test: batch_size=1 with num_labels=1 used to fail because squeeze produced a 0-d tensor.

    Some models (e.g. jinaai/jina-reranker-m0) return scores with shape [batch_size] instead of [batch_size, 1].
    With batch_size=1, squeeze(-1) would collapse [1] to a 0-d scalar, causing .extend() to fail.
    We mock forward to reproduce this by stripping the trailing dimension.
    """
    model = reranker_bert_tiny_model
    pairs = [
        ["A man is eating pasta.", "A man is eating food."],
        ["The girl is carrying a baby.", "A man is riding a horse."],
    ]

    original_forward = model.forward

    def forward_without_trailing_dim(features, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Calls the original forward, removes the last dimension of the "scores" tensor in the output dict,
        # and returns the dict.
        out = original_forward(features, **kwargs)
        # Simulate models that return [batch_size] instead of [batch_size, 1]
        out["scores"] = out["scores"].squeeze(-1)
        return out

    model.forward = forward_without_trailing_dim

    scores = model.predict(pairs, batch_size=1)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)


@pytest.mark.parametrize("convert_to_numpy", [True, False])
@pytest.mark.parametrize("convert_to_tensor", [True, False])
def test_empty_predict(reranker_bert_tiny_model: CrossEncoder, convert_to_numpy: bool, convert_to_tensor: bool):
    # [FUNCTION DOC]
    # Parametrised over convert-to-NumPy and convert-to-tensor. Asserts predicting an empty list returns an empty
    # tensor on the model's device when tensors are requested, else an empty NumPy array when NumPy is requested, else
    # an empty list.
    model = reranker_bert_tiny_model
    result = model.predict([], convert_to_numpy=convert_to_numpy, convert_to_tensor=convert_to_tensor)

    if convert_to_tensor:
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 0
        assert result.device == model.model.device
    elif convert_to_numpy:
        assert isinstance(result, np.ndarray)
        assert result.size == 0
    else:
        assert result == []


@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
def test_predict_output_types(
    reranker_bert_tiny_model: CrossEncoder, convert_to_tensor: bool, convert_to_numpy: bool
) -> None:
    # [FUNCTION DOC]
    # Parametrised over the same flags. Predicts one pair and asserts a float32 tensor when tensors are requested,
    # else a float32 NumPy array when NumPy is requested, else a list whose first element is a float32 tensor.
    model = reranker_bert_tiny_model
    embeddings = model.predict(
        [["One sentence", "Another sentence"]],
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
    )
    if convert_to_tensor:
        assert embeddings[0].dtype == torch.float32
        assert isinstance(embeddings, torch.Tensor)
    elif convert_to_numpy:
        assert embeddings[0].dtype == np.float32
        assert isinstance(embeddings, np.ndarray)
    else:
        assert embeddings[0].dtype == torch.float32
        assert isinstance(embeddings, list)


@pytest.mark.parametrize("safe_serialization", [True, False, None])
def test_safe_serialization(reranker_bert_tiny_model: CrossEncoder, safe_serialization: bool) -> None:
    # [FUNCTION DOC]
    # Parametrised over safe serialization true, false and unset. Saves into a temporary directory and asserts exactly
    # one safetensors file when true or unset; when false, only on transformers older than 5.0 asserts exactly one
    # PyTorch bin file (v5 ignores the option, so nothing is checked).
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_folder:
        model = reranker_bert_tiny_model
        if safe_serialization:
            model.save_pretrained(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        elif safe_serialization is None:
            model.save_pretrained(cache_folder)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        else:
            # For transformers v5.0, safe_serialization is quietly ignored
            if parse(transformers_version) < Version("5.0.0dev0"):
                model.save_pretrained(cache_folder, safe_serialization=safe_serialization)
                model_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
                assert 1 == len(model_files)


def test_bfloat16() -> None:
    # [FUNCTION DOC]
    # Skips on Windows without CUDA. Loads the tiny reranker with the deprecated automodel argument setting bfloat16,
    # and asserts predict returns a NumPy array and rank returns a list.
    if sys.platform == "win32" and not torch.cuda.is_available():
        pytest.skip(
            "bfloat16 CPU matmul can hard-crash (0xc000001d) on some Windows machines. Skipping to avoid CI failures."
        )
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce", automodel_args={"torch_dtype": torch.bfloat16}
    )
    score = model.predict([["Hello there!", "Hello, World!"]])
    assert isinstance(score, np.ndarray)

    ranking = model.rank("Hello there!", ["Hello, World!", "Heya!"])
    assert isinstance(ranking, list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_assignment(device):
    # [FUNCTION DOC]
    # Requires CUDA; parametrised over cpu and cuda. Asserts loading with that device reports it.
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device=device)
    assert model.device.type == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_device_switching():
    # [FUNCTION DOC]
    # Requires CUDA. Loads on CPU, asserts the model and underlying transformer are on CPU, moves to CUDA and asserts
    # both report CUDA, then frees the model and empties the CUDA cache.
    # test assignment using .to
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device="cpu")
    assert model.device.type == "cpu"
    assert model.model.device.type == "cpu"

    model.to("cuda")
    assert model.device.type == "cuda"
    assert model.model.device.type == "cuda"

    del model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_target_device_backwards_compat():
    # [FUNCTION DOC]
    # Requires CUDA. Asserts the legacy private target-device attribute reads as CPU and that assigning "cuda" to it
    # moves the model to CUDA.
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device="cpu")
    assert model.device.type == "cpu"

    assert model._target_device.type == "cpu"
    model._target_device = "cuda"
    assert model.device.type == "cuda"


def test_num_labels_fresh_model():
    # [FUNCTION DOC]
    # Asserts a fresh CrossEncoder from a checkpoint without a classification head defaults to one label.
    model = CrossEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert model.num_labels == 1


# The transformer-less stacks below aren't runnable end-to-end (no non-Transformer InputModule
# currently preprocesses CrossEncoder pairs), but we technically allow them so the property logic
# (config/model/num_labels/activation_fn) is exercised in isolation. It's what makes stacks like
# Transformer + Pooling + Dense(scores) usable as rerankers.
@pytest.mark.parametrize("out_features", [1, 3, 7])
def test_num_labels_from_dense_scores_module(static_embedding: StaticEmbedding, out_features: int):
    # [FUNCTION DOC]
    # Parametrised over output sizes 1, 3 and 7. Builds a CrossEncoder from the static embedding plus a Dense layer
    # whose output name is "scores" and asserts the label count equals the Dense output size.
    """A Dense module with ``module_output_name="scores"`` should dictate ``num_labels``."""
    from sentence_transformers.base.modules import Dense

    dense = Dense(
        static_embedding.get_embedding_dimension(),
        out_features,
        activation_function=None,
        module_output_name="scores",
    )
    model = CrossEncoder(modules=[static_embedding, dense])
    assert model.num_labels == out_features


def test_num_labels_ignores_dense_without_scores_output(static_embedding: StaticEmbedding):
    # [FUNCTION DOC]
    # Builds a CrossEncoder from the static embedding plus a 64-unit Dense layer with the default output name and
    # asserts the label count falls back to 1.
    """A trailing Dense whose ``module_output_name`` is not ``"scores"`` must not be
    treated as the classification head by ``num_labels``.
    """
    from sentence_transformers.base.modules import Dense

    dense = Dense(
        static_embedding.get_embedding_dimension(),
        64,
        activation_function=None,
    )
    model = CrossEncoder(modules=[static_embedding, dense])
    # No Transformer/LogitScore in the stack, and the Dense doesn't expose "scores",
    # so num_labels falls through to the default of 1 rather than picking up Dense.out_features.
    assert model.num_labels == 1


def test_config_and_model_none_without_transformer(static_embedding: StaticEmbedding):
    # [FUNCTION DOC]
    # Asserts that for a stack without a Transformer (static embedding plus a scores Dense), the config and model
    # properties are None rather than raising.
    """``.config`` and ``.model`` must degrade to ``None`` when the module stack has no
    underlying HuggingFace Transformer rather than raising AttributeError on ``self[0].model``.
    """
    from sentence_transformers.base.modules import Dense

    dense = Dense(static_embedding.get_embedding_dimension(), 1, module_output_name="scores")
    model = CrossEncoder(modules=[static_embedding, dense])

    assert model.config is None
    assert model.model is None


@pytest.mark.parametrize(
    ["out_features", "expected_activation_cls"],
    [(1, torch.nn.Sigmoid), (3, torch.nn.Identity)],
)
def test_default_activation_fn_without_transformer(
    static_embedding: StaticEmbedding, out_features: int, expected_activation_cls: type
):
    # [FUNCTION DOC]
    # Parametrised over one output (expects Sigmoid) and three outputs (expects Identity). Asserts the default
    # activation for a Transformer-less stack with a scores Dense is chosen from the label count.
    """``get_default_activation_fn`` must pick Sigmoid/Identity based on ``num_labels`` even
    when there's no Transformer (and thus no ``config``) to inspect for a saved activation.
    """
    from sentence_transformers.base.modules import Dense

    dense = Dense(
        static_embedding.get_embedding_dimension(),
        out_features,
        activation_function=None,
        module_output_name="scores",
    )
    model = CrossEncoder(modules=[static_embedding, dense])
    assert isinstance(model.activation_fn, expected_activation_cls)


def test_push_to_hub(
    reranker_bert_tiny_model: CrossEncoder, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # [FUNCTION DOC]
    # Monkeypatches HfApi repository creation, folder upload and branch creation with fakes, then pushes the tiny
    # reranker three times and asserts the repository id passed to upload and the returned URL: a commit URL with hash
    # 123456 for a plain push, hash 678901 and the revision forwarded when a revision is given, and the discussion URL
    # when creating a pull request.
    model = reranker_bert_tiny_model

    def mock_create_repo(self, repo_id, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake repository creation returning a repo URL for the id.
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    mock_upload_folder_kwargs = {}

    def mock_upload_folder(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake folder upload: stores the keyword arguments in the enclosing variable and returns commit info
        # with commit URL using hash 123456 when no revision is given else 678901, plus a pull request URL ending in
        # discussion 123. Retries construction with an extra endpoint argument for newer huggingface_hub versions that
        # require it.
        nonlocal mock_upload_folder_kwargs
        mock_upload_folder_kwargs = kwargs
        commit_hash = "123456" if kwargs.get("revision") is None else "678901"
        commit_info_kwargs = {
            "commit_url": f"https://huggingface.co/{kwargs.get('repo_id')}/commit/{commit_hash}",
            "commit_message": "commit_message",
            "commit_description": "commit_description",
            "oid": "oid",
            "pr_url": f"https://huggingface.co/{kwargs.get('repo_id')}/discussions/123",
        }
        try:
            return CommitInfo(**commit_info_kwargs)
        except TypeError:
            # Required as of https://github.com/huggingface/huggingface_hub/pull/3679
            return CommitInfo(**commit_info_kwargs, _endpoint=None)

    def mock_create_branch(self, repo_id, branch, revision=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake branch creation that does nothing and returns None.
        return None

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "create_branch", mock_create_branch)

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base")
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/123456"
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base", revision="revision_test")
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert mock_upload_folder_kwargs["revision"] == "revision_test"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/678901"
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base", create_pr=True)
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/discussions/123"
    mock_upload_folder_kwargs.clear()


@pytest.mark.parametrize(
    ["in_args", "in_kwargs", "out_args", "out_kwargs"],
    [
        [
            tuple(),
            {"model_name": "cross-encoder-testing/reranker-bert-tiny-gooaq-bce", "classifier_dropout": 0.1234},
            tuple(),
            {
                "model_name_or_path": "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
                "config_kwargs": {"classifier_dropout": 0.1234},
            },
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {"classifier_dropout": 0.1234},
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {"config_kwargs": {"classifier_dropout": 0.1234}},
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "automodel_args": {"foo": "bar"},
                "tokenizer_args": {"foo": "baz"},
            },
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "model_kwargs": {"foo": "bar"},
                "processor_kwargs": {"foo": "baz"},
            },
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "config_args": {"foo": "bar"},
                "cache_dir": "local_tmp",
            },
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "config_kwargs": {"foo": "bar"},
                "cache_folder": "local_tmp",
            },
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "automodel_args": {"foo": "bar"},
                "model_kwargs": {"faa": "baz"},
            },
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "model_kwargs": {"faa": "baz"},
            },
        ],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "default_activation_function": "torch.nn.Sigmoid",
            },
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {
                "activation_fn": "torch.nn.Sigmoid",
            },
        ],
        [tuple(), {}, tuple(), {}],
        [
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {},
            ("cross-encoder-testing/reranker-bert-tiny-gooaq-bce",),
            {},
        ],
        [
            tuple(),
            {
                "model_name": "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
                "automodel_args": {"foo": "bar"},
                "tokenizer_args": {"foo": "baz"},
                "config_args": {"foo": "bar"},
                "cache_dir": "local_tmp",
            },
            tuple(),
            {
                "model_name_or_path": "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
                "model_kwargs": {"foo": "bar"},
                "processor_kwargs": {"foo": "baz"},
                "config_kwargs": {"foo": "bar"},
                "cache_folder": "local_tmp",
            },
        ],
    ],
)
def test_init_args_decorator(
    monkeypatch: pytest.MonkeyPatch, in_args: tuple, in_kwargs: dict, out_args: tuple, out_kwargs: dict
):
    # [FUNCTION DOC]
    # Parametrised with input positional/keyword arguments and expected outputs. Replaces the CrossEncoder constructor
    # with a dummy wrapped by the init-argument decorator that records what it receives, constructs a CrossEncoder,
    # and asserts the decorator mapped deprecated names: model name to model name or path, classifier dropout into
    # config kwargs, automodel args to model kwargs, tokenizer args to processor kwargs, config args to config kwargs,
    # cache dir to cache folder, and default activation function to activation fn; when both an old and new name are
    # given the new one wins; empty inputs pass through.
    decorated_out_args = None
    decorated_out_kwargs = None

    @cross_encoder_init_args_decorator
    def mock_init(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Dummy constructor decorated with the init-argument decorator; stores the positional and keyword
        # arguments it receives in the enclosing variables.
        nonlocal decorated_out_args
        nonlocal decorated_out_kwargs
        decorated_out_args = args
        decorated_out_kwargs = kwargs
        return None

    monkeypatch.setattr(CrossEncoder, "__init__", mock_init)

    CrossEncoder(*in_args, **in_kwargs)
    assert decorated_out_args == out_args
    assert decorated_out_kwargs == out_kwargs


@pytest.mark.parametrize(
    ["in_kwargs", "out_kwargs"],
    [
        [
            {"inputs": [["Hello there!", "Hello, World!"]], "num_workers": 2},
            {"inputs": [["Hello there!", "Hello, World!"]]},
        ],
        [
            {
                "inputs": [["Hello there!", "Hello, World!"]],
                "activation_fct": torch.nn.Identity,
                "activation_fn": torch.nn.Sigmoid,
            },
            {"inputs": [["Hello there!", "Hello, World!"]], "activation_fn": torch.nn.Sigmoid},
        ],
        [
            {"sentences": [["Hello there!", "Hello, World!"]]},
            {"inputs": [["Hello there!", "Hello, World!"]]},
        ],
    ],
)
def test_predict_rank_args_decorator(
    reranker_bert_tiny_model: CrossEncoder, monkeypatch: pytest.MonkeyPatch, caplog, in_kwargs: dict, out_kwargs: dict
):
    # [FUNCTION DOC]
    # Parametrised with input and expected keyword arguments. Replaces predict with a dummy wrapped by the
    # predict/rank argument decorator, calls it, and asserts a warning was logged and that the decorator dropped num
    # workers, dropped the old activation keyword when the new one is present, and renamed sentences to inputs.
    model = reranker_bert_tiny_model
    decorated_out_kwargs = None

    @cross_encoder_predict_rank_args_decorator
    def mock_predict(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Dummy predict decorated with the predict/rank decorator; stores the keyword arguments it receives.
        nonlocal decorated_out_kwargs
        decorated_out_kwargs = kwargs
        return None

    monkeypatch.setattr(CrossEncoder, "predict", mock_predict)

    with caplog.at_level(logging.WARNING):
        model.predict(**in_kwargs)
        assert caplog.text != ""
    assert decorated_out_kwargs == out_kwargs


def test_logger_warning(caplog):
    # [FUNCTION DOC]
    # Asserts loading with classifier dropout, automodel args, tokenizer args or config args each logs the
    # corresponding deprecation/rename warning.
    model_name = "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"
    with caplog.at_level(logging.WARNING):
        CrossEncoder(model_name, classifier_dropout=0.1234)
        assert "`classifier_dropout` argument is deprecated" in caplog.text

    with caplog.at_level(logging.WARNING):
        CrossEncoder(model_name, automodel_args={"torch_dtype": torch.float32})
        assert "`automodel_args` argument was renamed and is now deprecated" in caplog.text

    with caplog.at_level(logging.WARNING):
        CrossEncoder(model_name, tokenizer_args={"model_max_length": 8192})
        assert "`tokenizer_args` argument was renamed and is now deprecated" in caplog.text

    with caplog.at_level(logging.WARNING):
        CrossEncoder(model_name, config_args={"classifier_dropout": 0.2})
        assert "`config_args` argument was renamed and is now deprecated" in caplog.text


@pytest.mark.parametrize(
    ["num_labels", "activation_fn", "saved_activation_fn"],
    [
        [
            1,
            torch.nn.Sigmoid(),
            "torch.nn.modules.activation.Sigmoid",
        ],
        [
            1,
            torch.nn.Identity(),
            "torch.nn.modules.linear.Identity",
        ],
        [
            1,
            torch.nn.Tanh(),
            "torch.nn.modules.activation.Tanh",
        ],
        [
            1,
            torch.nn.Softmax(),
            "torch.nn.modules.activation.Softmax",
        ],
        [
            1,
            None,
            "torch.nn.modules.activation.Sigmoid",
        ],
        [
            3,
            None,
            "torch.nn.modules.linear.Identity",
        ],
    ],
)
def test_load_activation_fn_from_kwargs(num_labels: int, activation_fn: str, saved_activation_fn: str, tmp_path: Path):
    # [FUNCTION DOC]
    # Parametrised over label count, activation passed to the constructor (Sigmoid, Identity, Tanh, Softmax or None)
    # and the expected saved full class path (None gives Sigmoid for one label, Identity for three). Asserts the
    # model's activation has that path, that saving writes it to the Sentence Transformers config, that reloading
    # restores it, that a per-call activation in predict does not change the stored one, and that passing an
    # activation when reloading overrides it.
    model = CrossEncoder(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", num_labels=num_labels, activation_fn=activation_fn
    )
    assert fullname(model.activation_fn) == saved_activation_fn

    model.save_pretrained(tmp_path)
    with open(tmp_path / "config_sentence_transformers.json") as f:
        config = json.load(f)
    assert config["activation_fn"] == saved_activation_fn

    loaded_model = CrossEncoder(str(tmp_path))
    assert fullname(loaded_model.activation_fn) == saved_activation_fn

    # Setting the activation function via a predict call only updates it for that call
    loaded_model.predict([["Hello there!", "Hello, World!"]], activation_fn=torch.nn.Identity())
    assert fullname(loaded_model.activation_fn) == saved_activation_fn

    # But we can also override it again when loading the model
    loaded_model = CrossEncoder(str(tmp_path), activation_fn=torch.nn.Identity())
    assert fullname(loaded_model.activation_fn) == "torch.nn.modules.linear.Identity"


@pytest.mark.parametrize(
    "tanh_model_name",
    [
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v3",
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v4",
    ],
)
def test_load_activation_fn_from_config(tanh_model_name: str, tmp_path):
    # [FUNCTION DOC]
    # Parametrised over two tiny checkpoints saved with a Tanh activation in v3 and v4 formats. Asserts the activation
    # loads as Tanh, is written back on save, and is restored on reload.
    saved_activation_fn = "torch.nn.modules.activation.Tanh"

    model = CrossEncoder(tanh_model_name)
    assert fullname(model.activation_fn) == saved_activation_fn

    model.save_pretrained(tmp_path)
    with open(tmp_path / "config_sentence_transformers.json") as f:
        config = json.load(f)
    assert config["activation_fn"] == saved_activation_fn

    loaded_model = CrossEncoder(str(tmp_path))
    assert fullname(loaded_model.activation_fn) == saved_activation_fn


def test_load_activation_fn_from_config_custom(reranker_bert_tiny_model: CrossEncoder, tmp_path: Path, caplog):
    # [FUNCTION DOC]
    # Saves the tiny reranker, rewrites its config to name an untrusted custom activation path, asserts loading logs
    # that the path is not trusted and the default is used, and asserts loading with trust remote code raises
    # ModuleNotFoundError because the module does not exist.
    model = reranker_bert_tiny_model

    model.save_pretrained(tmp_path)
    with open(tmp_path / "config_sentence_transformers.json") as f:
        config = json.load(f)
    config["activation_fn"] = "sentence_transformers.custom.activations.CustomActivation"
    with open(tmp_path / "config_sentence_transformers.json", "w") as f:
        json.dump(config, f)

    with caplog.at_level(logging.WARNING):
        CrossEncoder(str(tmp_path))
        assert (
            "Activation function path 'sentence_transformers.custom.activations.CustomActivation' is not trusted, using default activation function instead."
            in caplog.text
        )

    # If we use trust_remote_code, it'll try to load the custom activation function, which doesn't exist
    with pytest.raises(ModuleNotFoundError):
        model = CrossEncoder(str(tmp_path), trust_remote_code=True)


def test_default_activation_fn(reranker_bert_tiny_model: CrossEncoder):
    # [FUNCTION DOC]
    # Asserts the tiny reranker's activation is Sigmoid and that the deprecated default activation property returns
    # the same while emitting a DeprecationWarning.
    model = reranker_bert_tiny_model
    assert fullname(model.activation_fn) == "torch.nn.modules.activation.Sigmoid"
    with pytest.warns(
        DeprecationWarning, match="The `default_activation_function` property was renamed and is now deprecated.*"
    ):
        assert fullname(model.default_activation_function) == "torch.nn.modules.activation.Sigmoid"


def test_bge_reranker_max_length():
    # [FUNCTION DOC]
    # Loads BAAI/bge-reranker-base and asserts max length and tokenizer max length are 512; after setting max length
    # to 256 both report 256.
    model = CrossEncoder("BAAI/bge-reranker-base")
    assert model.max_length == 512
    assert model.tokenizer.model_max_length == 512

    model.max_length = 256
    assert model.max_length == 256
    assert model.tokenizer.model_max_length == 256


def test_predict_with_dataset_column(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts predicting a Hugging Face Dataset column holding two text pairs (as tensors) returns two scores.
    """Test that predict can handle a dataset column as input."""
    model = reranker_bert_tiny_model
    from datasets import Dataset

    # Create a simple dataset with a text column
    dataset = Dataset.from_dict(
        {
            "text": [
                ["This is the start of a pair.", "And this the end."],
                ["This is a second pair.", "And this the end of the second pair."],
            ]
        }
    )

    # Encode the dataset column
    embeddings = model.predict(dataset["text"], convert_to_tensor=True)

    # Check the shape of the embeddings
    assert embeddings.shape == (2,)


def test_predict_per_call_processing_kwargs(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Asserts predicting one pair with per-call processing kwargs that truncate text to 4 tokens gives a different
    # score than predicting without them, proving the kwargs reach preprocessing.
    """Per-call ``processing_kwargs`` should be accepted by ``predict`` and reach ``preprocess``.

    If the kwarg were silently dropped, truncated and full predictions on the same pair would
    produce identical scores.
    """
    model = reranker_bert_tiny_model
    pair = ["this is a moderately long query sentence", "this is a moderately long candidate sentence"]
    truncated = model.predict(
        [pair],
        processing_kwargs={"text": {"max_length": 4, "truncation": True}},
    )
    full = model.predict([pair])
    assert not np.isclose(truncated[0], full[0])


# ChatML emits the role rather than branching on it, so unknown roles carry their content through
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
)
# Structured-content variant of a rerank template, as a multimodal processor would need
STRUCTURED_RERANKER_TEMPLATE = (
    '<Query>: {{ (messages | selectattr("role", "eq", "query") | first).content[0].text }}\n'
    '<Document>: {{ (messages | selectattr("role", "eq", "document") | first).content[0].text }}'
)
BERLIN_PAIRS = [
    ["How many people live in Berlin?", "Berlin has a population of 3,520,031 registered inhabitants."],
    ["How many people live in Berlin?", "Berlin is well known for its museums."],
]
PAIR_ROLES_DROPPED = "cannot carry a 'query'/'document' pair"


def test_replacing_a_rejected_chat_template_takes_effect() -> None:
    # [FUNCTION DOC]
    # Loads the tiny reranker with a generic instruct chat template that cannot render query/document roles and
    # asserts predicting pairs raises ValueError about the pair roles; then sets the processor's chat template to the
    # reranker template and asserts preprocessing a pair now renders the query text.
    """The error names a remedy, so applying it to a loaded model has to work."""
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": GENERIC_INSTRUCT_TEMPLATE},
    )
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model.predict(BERLIN_PAIRS)

    model.processor.chat_template = RERANKER_TEMPLATE
    decoded = model[0].tokenizer.decode(
        model[0].preprocess(BERLIN_PAIRS[:1])["input_ids"][0], skip_special_tokens=True
    )
    assert "< query > : how many people live in berlin?" in decoded


def test_preprocess_raises_on_non_text_pair_without_pair_roles() -> None:
    # [FUNCTION DOC]
    # With the generic instruct template, asserts preprocessing an (image, text) pair raises the pair-roles
    # ValueError, also when a text pair shares the batch.
    """Non-text pairs reach the roles by a different route, and are dropped just the same."""
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": GENERIC_INSTRUCT_TEMPLATE},
    )
    image = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model[0].preprocess([(image, "a document")])
    # A text pair sharing a batch with a non-text one takes the same route
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model[0].preprocess([(image, "a document"), ("a query", "a document")])


def test_predict_raises_on_chat_template_without_pair_roles() -> None:
    # [FUNCTION DOC]
    # With the generic instruct template, asserts the transformer has a message modality and reports a pair-roles
    # failure, that predicting pairs raises the pair-roles ValueError, and that single texts still preprocess and
    # decode containing the input.
    """A chat template that drops the pair roles must be reported, not worked around.

    Backbones such as the Qwen3 instruct models carry a template that only branches on
    ``system``/``user``/``assistant``. Rendering the ``query``/``document`` roles through it drops both
    messages, leaving the model with a zero-length input. Tokenizing the pair directly instead would
    score an input the model was never trained on, so the misconfiguration is raised rather than hidden.
    """
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": GENERIC_INSTRUCT_TEMPLATE},
    )
    transformer = model[0]
    # The template is picked up, it just cannot render the roles that text pairs take
    assert "message" in transformer.modality_config
    assert transformer.input_formatter.pair_roles_failure() is not None

    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model.predict(BERLIN_PAIRS)

    # Single texts never take those roles, so they are unaffected
    features = transformer.preprocess(["a solo text"])
    assert "a solo text" in transformer.tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)


def test_predict_raises_on_mixed_batch_containing_a_pair() -> None:
    # [FUNCTION DOC]
    # With the generic instruct template, asserts a batch mixing a single text and a pair raises the pair-roles
    # ValueError.
    """One pair in a batch is enough, its content would be dropped just the same."""
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": GENERIC_INSTRUCT_TEMPLATE},
    )
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED):
        model[0].preprocess(["a solo text", ("How many people live in Berlin?", "Berlin has 3.5M.")])


def test_predict_applies_chat_template_with_pair_roles() -> None:
    # [FUNCTION DOC]
    # With the reranker template, asserts no pair-roles failure is reported and that the decoded preprocessed pair
    # contains the query and document sections.
    """A template that does reference the ``query``/``document`` roles must still be applied."""
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": RERANKER_TEMPLATE},
    )
    transformer = model[0]
    assert transformer.input_formatter.pair_roles_failure() is None

    features = transformer.preprocess(BERLIN_PAIRS[:1])
    decoded = transformer.tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)
    assert "< query > : how many people live in berlin?" in decoded
    assert "< document > : berlin has a population" in decoded


def test_predict_applies_chat_template_that_only_echoes_the_role() -> None:
    # [FUNCTION DOC]
    # With the ChatML template that echoes roles, asserts no pair-roles failure is reported and the decoded pair
    # contains both role names and both texts.
    """A template that carries unknown roles through must keep working, not raise.

    Pass-through ChatML templates such as SmolLM2's emit ``message['role']`` verbatim, so they never
    mention ``query``/``document`` yet render both messages in full. Deciding from the template source
    rather than from a render would break models that are fine today.
    """
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": CHATML_TEMPLATE},
    )
    transformer = model[0]
    assert transformer.input_formatter.pair_roles_failure() is None

    features = transformer.preprocess(BERLIN_PAIRS[:1])
    decoded = transformer.tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)
    assert "query" in decoded and "how many people live in berlin?" in decoded
    assert "document" in decoded and "berlin has a population" in decoded


def test_message_input_with_a_single_pair_role_is_not_checked() -> None:
    # [FUNCTION DOC]
    # With a template that renders only the query role, asserts a hand-written message using only the query role
    # preprocesses and decodes with the instruct prefix (no pair-roles check applies).
    """A hand-written message may use one pair role alone, and that is not a pair.

    The check guards the pair mapping, which always produces both roles in one conversation, so a
    template rendering just the role the user actually sent must not be rejected for dropping a role
    the batch never uses.
    """
    query_only = "{% for m in messages %}{% if m.role == 'query' %}Instruct: {{ m.content }}{% endif %}{% endfor %}"
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={"chat_template": query_only},
    )
    features = model[0].preprocess([[{"role": "query", "content": "hello world"}]])
    decoded = model[0].tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)
    assert "instruct : hello world" in decoded


def test_probe_uses_the_configured_chat_template_kwargs() -> None:
    # [FUNCTION DOC]
    # With a dict of named templates (default generic instruct, plus "reranker" structured template), asserts
    # preprocessing a pair raises the pair-roles ValueError whose remedy mentions structured content access; then
    # selects the "reranker" template via processing kwargs and asserts the pair renders the query section.
    """A model may select a named rerank template via processing kwargs, as Qwen3-VL-Reranker does.

    The probe must measure the selected template rather than the default one, since rejecting the
    model for a default that drops the pair roles would break a correctly configured reranker.
    """
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
        processor_kwargs={
            "chat_template": {"default": GENERIC_INSTRUCT_TEMPLATE, "reranker": STRUCTURED_RERANKER_TEMPLATE}
        },
    )
    transformer = model[0]
    with pytest.raises(ValueError, match=PAIR_ROLES_DROPPED) as excinfo:
        transformer.preprocess(BERLIN_PAIRS[:1])
    # The dict template infers structured format, so the remedy example must read typed content
    assert "content[0].text" in str(excinfo.value)

    transformer.processing_kwargs["chat_template"] = {"chat_template": "reranker"}
    features = transformer.preprocess(BERLIN_PAIRS[:1])
    decoded = transformer.tokenizer.decode(features["input_ids"][0], skip_special_tokens=True)
    assert "< query > : how many people live in berlin?" in decoded


# Test suite converted from demo_3406_simple_og.py
def format_queries(query, instruction=None):
    # [FUNCTION DOC]
    # HELPER: Formats a query for the Qwen3 reranker: a fixed system prompt asking for a yes/no relevance judgement,
    # then a user turn containing an Instruct line (default web search instruction when none is given) and a Query
    # line. Returns the string.
    """Helper function to format queries with the template."""
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document):
    # [FUNCTION DOC]
    # HELPER: Formats a document for the Qwen3 reranker as a Document line followed by the end-of-user and
    # start-of-assistant tokens with an empty think block. Returns the string.
    """Helper function to format documents with the template."""
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_reranker_formatted_pairs():
    # [FUNCTION DOC]
    # Requires CUDA. Loads the sequence-classification conversion of Qwen3-Reranker-0.6B with identity activation,
    # predicts four manually formatted query/document pairs, and asserts the scores match fixed expected logits within
    # 1e-4.
    """Test Qwen3 Reranker with manually formatted query-document pairs."""
    model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", activation_fn=torch.nn.Identity())
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
    ]

    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    pairs = [[format_queries(query, task), format_document(doc)] for query, doc in zip(queries, documents)]
    scores = model.predict(pairs)
    expected_scores = [-3.109297752380371, 7.120389938354492, -0.3787546157836914, 3.541637420654297]

    # Assert scores match expected values with tolerance
    assert scores == pytest.approx(expected_scores, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_reranker_with_chat_template():
    # [FUNCTION DOC]
    # Requires CUDA. Same model and expected scores, but with a chat template rendering the system/query/document
    # roles, a "web_search" prompt set as default, and plain pairs.
    """Test Qwen3 Reranker with Chat template."""
    chat_template = """\
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n\n"""

    task = "Given a web search query, retrieve relevant passages that answer the query"
    model = CrossEncoder(
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        activation_fn=torch.nn.Identity(),
        processor_kwargs={"chat_template": chat_template},
        prompts={"web_search": task},
        default_prompt_name="web_search",
    )

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    expected_scores = [-3.109297752380371, 7.120389938354492, -0.3787546157836914, 3.541637420654297]

    # Assert scores match expected values with tolerance
    assert scores == pytest.approx(expected_scores, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_reranker_original_with_identity_activation():
    # [FUNCTION DOC]
    # Requires CUDA. Loads the original Qwen/Qwen3-Reranker-0.6B (converted on load to a logit-scoring CrossEncoder)
    # in float32 with identity activation, the role-rendering chat template and the default web-search prompt; asserts
    # float32 dtype and that scores on the four pairs match the same expected values.
    """Test original Qwen3 Reranker with Identity activation function."""
    chat_template = """\
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n\n"""

    task = "Given a web search query, retrieve relevant passages that answer the query"
    model = CrossEncoder(
        "Qwen/Qwen3-Reranker-0.6B",
        prompts={"web_search": task},
        default_prompt_name="web_search",
        activation_fn=torch.nn.Identity(),
        model_kwargs={"torch_dtype": torch.float32},
        processor_kwargs={"chat_template": chat_template},
    )
    assert model.dtype == torch.float32

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    expected_scores = [-3.109297752380371, 7.120389938354492, -0.3787546157836914, 3.541637420654297]

    # Assert scores match expected values with tolerance
    assert scores == pytest.approx(expected_scores, abs=1e-4)


def test_conversion_ignores_source_prompts_and_default_prompt_name(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Saves a SentenceTransformer with a query prompt set as default, loads the directory as a CrossEncoder
    # (conversion adds a fresh head), and asserts the prompts are empty and no default prompt name is inherited.
    """Converting a SentenceTransformer save appends a fresh classification head, so the source's
    embedding prompts and default_prompt_name are not inherited: they would otherwise silently be
    prepended to every predict call."""
    from sentence_transformers import SentenceTransformer

    source = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        prompts={"query": "Represent this sentence: "},
        default_prompt_name="query",
    )
    source.save_pretrained(str(tmp_path))

    model = CrossEncoder(str(tmp_path))
    assert model.prompts == {}
    assert model.default_prompt_name is None


def test_predict_routes_through_module_call(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Registers a forward hook on the CrossEncoder that records calls, predicts one pair, removes the hook, and
    # asserts the hook fired (prediction goes through the module call so compilation applies).
    """predict() must run the forward pass via __call__ so that model.compile() applies to inference."""
    model = reranker_bert_tiny_model
    calls = []
    handle = model.register_forward_hook(lambda module, args, output: calls.append(True))
    try:
        model.predict([["Hello there!", "Hello, World!"]])
    finally:
        handle.remove()
    assert calls, "predict() should invoke the model via __call__, not call forward() directly"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_reranker_original_without_prompt():
    # [FUNCTION DOC]
    # Requires CUDA. Same as the original Qwen3 test but without any prompt; relies on the chat template's default
    # instruction and asserts the same expected scores.
    # KNOWN ISSUE: The docstring is copied from the identity-activation test and does not mention that this variant
    # omits the prompt.
    """Test original Qwen3 Reranker with Identity activation function."""
    chat_template = """\
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n\n"""

    model = CrossEncoder(
        "Qwen/Qwen3-Reranker-0.6B",
        activation_fn=torch.nn.Identity(),
        model_kwargs={"torch_dtype": torch.float32},
        processor_kwargs={"chat_template": chat_template},
    )
    assert model.dtype == torch.float32

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    expected_scores = [-3.109297752380371, 7.120389938354492, -0.3787546157836914, 3.541637420654297]

    # Assert scores match expected values with tolerance
    assert scores == pytest.approx(expected_scores, abs=1e-4)
