# [MODULE DOC]
# Tests for InputModule's preprocess/tokenize backward compatibility: legacy subclasses that only
# override tokenize are delegated to (with a warning and prompt prepending), proper preprocess
# subclasses work silently, subclasses implementing neither raise NotImplementedError, and calling
# the deprecated tokenize on a preprocess subclass warns and delegates. Defines three minimal
# InputModule subclasses as fixtures.
from __future__ import annotations

import logging

import pytest

from sentence_transformers.base.modules.input_module import InputModule


class _TokenizeOnlyModule(InputModule):
    # [CLASS DOC]
    # HELPER: Test double for the legacy pattern: overrides tokenize (returning a dict with the texts
    # under "tokens") but not preprocess; forward is identity and save does nothing.
    """Subclass that overrides tokenize() but not preprocess() (legacy pattern)."""

    def forward(self, features):
        # [FUNCTION DOC]
        # HELPER: Identity forward; returns the features unchanged.
        return features

    def tokenize(self, texts, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns a dict mapping "tokens" to the given texts; extra keyword arguments ignored.
        return {"tokens": texts}

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        # [FUNCTION DOC]
        # HELPER: No-op save.
        pass


class _PreprocessModule(InputModule):
    # [CLASS DOC]
    # HELPER: Test double implementing preprocess properly; forward is identity and save does nothing.
    """Subclass that properly overrides preprocess()."""

    def forward(self, features):
        # [FUNCTION DOC]
        # HELPER: Identity forward; returns the features unchanged.
        return features

    def preprocess(self, inputs, prompt=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: If a prompt is given, prepends it to the inputs using the base class's prompt helper;
        # returns a dict mapping "tokens" to the (possibly prefixed) inputs.
        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)
        return {"tokens": inputs}

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        # [FUNCTION DOC]
        # HELPER: No-op save.
        pass


class _BareModule(InputModule):
    # [CLASS DOC]
    # HELPER: Test double that overrides neither preprocess nor tokenize.
    """Subclass that overrides neither preprocess() nor tokenize()."""

    def forward(self, features):
        # [FUNCTION DOC]
        # HELPER: Identity forward; returns the features unchanged.
        return features

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        # [FUNCTION DOC]
        # HELPER: No-op save.
        pass


class TestPreprocessBackwardCompat:
    # [CLASS DOC]
    # Tests that InputModule.preprocess works for legacy tokenize-only, proper preprocess and bare subclasses.
    def test_tokenize_only_subclass_delegates_and_warns(self, caplog):
        # [FUNCTION DOC]
        # Calling preprocess on the tokenize-only module returns tokenize's result and logs a warning that
        # the subclass overrides tokenize instead of preprocess.
        module = _TokenizeOnlyModule()
        with caplog.at_level(logging.WARNING):
            result = module.preprocess(["hello", "world"])
        assert result == {"tokens": ["hello", "world"]}
        assert any("overrides `tokenize` instead of `preprocess`" in r.message for r in caplog.records)

    def test_tokenize_only_subclass_prepends_prompt(self):
        # [FUNCTION DOC]
        # Preprocess with a prompt on the tokenize-only module prepends the prompt before delegating.
        module = _TokenizeOnlyModule()
        result = module.preprocess(["hello"], prompt="search: ")
        assert result == {"tokens": ["search: hello"]}

    def test_preprocess_subclass_works_without_warning(self, caplog):
        # [FUNCTION DOC]
        # Preprocess on a proper subclass returns its result and logs no warning mentioning tokenize.
        module = _PreprocessModule()
        with caplog.at_level(logging.WARNING):
            result = module.preprocess(["hello"])
        assert result == {"tokens": ["hello"]}
        assert not any("tokenize" in r.message for r in caplog.records)

    def test_preprocess_subclass_with_prompt(self):
        # [FUNCTION DOC]
        # Preprocess with a prompt on a proper subclass yields the prompt-prefixed inputs.
        module = _PreprocessModule()
        result = module.preprocess(["hello"], prompt="query: ")
        assert result == {"tokens": ["query: hello"]}

    def test_bare_subclass_raises_not_implemented(self):
        # [FUNCTION DOC]
        # Preprocess on the bare subclass raises NotImplementedError saying preprocess must be implemented.
        module = _BareModule()
        with pytest.raises(NotImplementedError, match="must implement the `preprocess` method"):
            module.preprocess(["hello"])


class TestTokenizeDeprecation:
    # [CLASS DOC]
    # Tests for the deprecated InputModule.tokenize entry point.
    def test_tokenize_on_preprocess_subclass_warns_and_delegates(self, caplog):
        # [FUNCTION DOC]
        # Calling tokenize on a preprocess subclass logs a deprecation warning and returns preprocess's result.
        module = _PreprocessModule()
        with caplog.at_level(logging.WARNING):
            result = module.tokenize(["hello"])
        assert result == {"tokens": ["hello"]}
        assert any("tokenize" in r.message and "deprecated" in r.message for r in caplog.records)

    def test_tokenize_on_bare_subclass_raises(self):
        # [FUNCTION DOC]
        # Calling tokenize on the bare subclass raises NotImplementedError.
        module = _BareModule()
        with pytest.raises(NotImplementedError):
            module.tokenize(["hello"])
