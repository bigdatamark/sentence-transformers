# [MODULE DOC]
# Tests SentenceTransformer.encode output shapes and values: token embeddings, single and batched sentences, normalization,
# sentence-pair (tuple) inputs, and that encoding goes through the module call (so hooks and compilation apply). Expected sums are
# regression values for the paraphrase-distilroberta-base-v1 model with 768-D embeddings.
"""
Computes embeddings
"""

from __future__ import annotations

import numpy as np

from sentence_transformers import SentenceTransformer


def test_encode_token_embeddings(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Encodes five sentences with output_value "token_embeddings" and batch size 2. Asserts one output per sentence and that each
    # output's token count equals the length of that sentence's tokenized input ids.
    """
    Test that encode(output_value='token_embeddings') works
    """
    model = paraphrase_distilroberta_base_v1_model
    sent = [
        "Hello Word, a test sentence",
        "Here comes another sentence",
        "My final sentence",
        "Sentences",
        "Sentence five five five five five five five",
    ]
    emb = model.encode(sent, output_value="token_embeddings", batch_size=2)
    assert len(emb) == len(sent)

    for s, e in zip(sent, emb):
        assert len(model.preprocess([s])["input_ids"][0]) == e.shape[0]


def test_encode_single_sentences(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts a single string encodes to a 768-vector, a one-element list to a 1x768 array (both summing to about 7.98), and a
    # three-sentence list to a 3x768 array summing to about 22.97 within tolerance.
    model = paraphrase_distilroberta_base_v1_model
    # Single sentence
    emb = model.encode("Hello Word, a test sentence")
    assert emb.shape == (768,)
    assert abs(np.sum(emb) - 7.9811716) < 0.002

    # Single sentence as list
    emb = model.encode(["Hello Word, a test sentence"])
    assert emb.shape == (1, 768)
    assert abs(np.sum(emb) - 7.9811716) < 0.002

    # Sentence list
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ]
    )
    assert emb.shape == (3, 768)
    assert abs(np.sum(emb) - 22.968266) < 0.007


def test_encode_normalize(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Encodes three sentences with normalize_embeddings enabled and asserts the shape is 3x768 and each row has unit L2 norm within 0.001.
    model = paraphrase_distilroberta_base_v1_model
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ],
        normalize_embeddings=True,
    )
    assert emb.shape == (3, 768)
    for norm in np.linalg.norm(emb, axis=1):
        assert abs(norm - 1) < 0.001


def test_encode_tuple_sentences(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Encodes one and then three sentence tuples (pairs fed jointly to the model); asserts shapes 1x768 and 3x768 and that the sums
    # match the reference values within 0.002.
    model = paraphrase_distilroberta_base_v1_model
    # Input a sentence tuple
    emb = model.encode([("Hello Word, a test sentence", "Second input for model")])
    assert emb.shape == (1, 768)
    assert abs(np.sum(emb) - 9.503508) < 0.002

    # List of sentence tuples
    emb = model.encode(
        [
            ("Hello Word, a test sentence", "Second input for model"),
            ("My second tuple", "With two inputs"),
            ("Final tuple", "final test"),
        ]
    )
    assert emb.shape == (3, 768)
    assert abs(np.sum(emb) - 32.14627) < 0.002


def test_encode_routes_through_module_call(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Registers a forward hook on the tiny model that records each call, encodes one string, removes the hook (always), and asserts
    # the hook fired, proving encode invokes the model via its call method rather than forward directly. Temporarily mutates hooks.
    """encode() must run the forward pass via __call__ so that model.compile() applies to inference."""
    model = stsb_bert_tiny_model
    calls = []
    handle = model.register_forward_hook(lambda module, args, output: calls.append(True))
    try:
        model.encode("Hello world")
    finally:
        handle.remove()
    assert calls, "encode() should invoke the model via __call__, not call forward() directly"
