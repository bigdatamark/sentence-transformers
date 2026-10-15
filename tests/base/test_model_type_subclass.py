# [MODULE DOC]
# Pytest tests (regression for issue 3536) that subclasses of SentenceTransformer, SparseEncoder and CrossEncoder
# inherit their parent's model type string, so loading a checkpoint saved by the same archetype uses the normal
# config-based module loading rather than the conversion loader; and that a subclass declaring its own model type opts
# out and uses the conversion path. Loads tiny models from the Hub.
from __future__ import annotations

from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder


def test_model_type_inherited_for_sentence_transformer_subclass(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts the tiny STSB fixture reports model type "SentenceTransformer", then defines a subclass whose
    # converted-module loader sets a flag before delegating, loads the same tiny checkpoint with it, and asserts its
    # model type is still "SentenceTransformer" and the converted loader was not called.
    """A subclass of SentenceTransformer should still report `model_type == "SentenceTransformer"`,
    so `_load_modules` takes the config path (not the conversion path) for pre-saved ST models.
    Reproduces #3536."""
    assert stsb_bert_tiny_model.model_type == "SentenceTransformer"

    converted_called = False

    class MySentenceTransformer(SentenceTransformer):
        # [CLASS DOC]
        # HELPER: Local SentenceTransformer subclass instrumented to record use of the converted-module loader.
        def _load_converted_modules(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Sets the enclosing test's flag to True, then delegates to the parent loader with the same
            # arguments and returns its result.
            nonlocal converted_called
            converted_called = True
            return super()._load_converted_modules(*args, **kwargs)

    subclassed = MySentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert subclassed.model_type == "SentenceTransformer"
    assert not converted_called, "Plain ST subclass should not hit the conversion loader path"


def test_model_type_inherited_for_sparse_encoder_subclass(splade_bert_tiny_model: SparseEncoder) -> None:
    # [FUNCTION DOC]
    # Same check for SparseEncoder with the tiny SPLADE checkpoint: model type stays "SparseEncoder" and the converted
    # loader is not called.
    """Same inheritance and routing for SparseEncoder subclasses."""
    assert splade_bert_tiny_model.model_type == "SparseEncoder"

    converted_called = False

    class MySparseEncoder(SparseEncoder):
        # [CLASS DOC]
        # HELPER: Local SparseEncoder subclass instrumented to record use of the converted-module loader.
        def _load_converted_modules(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Sets the enclosing test's flag, then delegates to the parent loader and returns its result.
            nonlocal converted_called
            converted_called = True
            return super()._load_converted_modules(*args, **kwargs)

    subclassed = MySparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq")
    assert subclassed.model_type == "SparseEncoder"
    assert not converted_called, "Plain SparseEncoder subclass should not hit the conversion loader path"


def test_model_type_inherited_for_cross_encoder_subclass(reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Same check for CrossEncoder with the tiny GooAQ reranker: model type stays "CrossEncoder" and the converted
    # loader is not called.
    """Same inheritance and routing for CrossEncoder subclasses."""
    assert reranker_bert_tiny_model.model_type == "CrossEncoder"

    converted_called = False

    class MyCrossEncoder(CrossEncoder):
        # [CLASS DOC]
        # HELPER: Local CrossEncoder subclass instrumented to record use of the converted-module loader.
        def _load_converted_modules(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Sets the enclosing test's flag, then delegates to the parent loader and returns its result.
            nonlocal converted_called
            converted_called = True
            return super()._load_converted_modules(*args, **kwargs)

    subclassed = MyCrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
    assert subclassed.model_type == "CrossEncoder"
    assert not converted_called, "Plain CrossEncoder subclass should not hit the conversion loader path"


def test_model_type_override_takes_precedence() -> None:
    # [FUNCTION DOC]
    # Defines a SentenceTransformer subclass with an explicit model type attribute "OverrideSentenceTransformer" and
    # an instrumented converted loader, loads the tiny STSB checkpoint, and asserts the model type is the override and
    # that the conversion loader was used.
    """A subclass that explicitly declares its own `model_type` opts out of the archetype
    identity, restoring the pre-#3536 conversion-path behavior on archetype-typed models."""
    converted_called = False

    class OverrideSentenceTransformer(SentenceTransformer):
        # [CLASS DOC]
        # HELPER: SentenceTransformer subclass that overrides the model type class attribute and records
        # conversion-loader use.
        model_type = "OverrideSentenceTransformer"

        def _load_converted_modules(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Sets the enclosing test's flag, then delegates to the parent loader and returns its result.
            nonlocal converted_called
            converted_called = True
            return super()._load_converted_modules(*args, **kwargs)

    overridden = OverrideSentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert overridden.model_type == "OverrideSentenceTransformer"
    assert converted_called, "Overriding `model_type` should route through the conversion loader path"
