# [MODULE DOC]
# Unit tests for the keyword-argument compatibility decorators in sentence_transformers.util.decorators:
# deprecated_kwargs (generic renames with warnings), transformer_kwargs_decorator (model/tokenizer/config arg
# renames and cache_dir distribution), cross_encoder_init_args_decorator (CrossEncoder init renames, classifier
# dropout and legacy positional args), cross_encoder_predict_rank_args_decorator (predict/rank renames and
# dropping of removed kwargs), and save_to_hub_args_decorator. Each test decorates a small function that returns
# the keyword arguments it received and inspects the result and captured warning logs.
"""Unit tests for decorator utilities in sentence_transformers.util.decorators.

Each decorator is tested with mock functions to verify:
- Old (deprecated) keyword arguments are renamed and produce warnings
- New keyword arguments work without warnings
- When both old and new are provided, the new value wins
"""

from __future__ import annotations

import logging

import pytest

from sentence_transformers.util.decorators import (
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
    deprecated_kwargs,
    save_to_hub_args_decorator,
    transformer_kwargs_decorator,
)


class TestDeprecatedKwargs:
    # [CLASS DOC]
    # Groups tests for the generic deprecated_kwargs decorator, which renames old keyword names to new ones and
    # logs a deprecation warning.
    def test_old_name_is_renamed(self, caplog):
        # [FUNCTION DOC]
        # Asserts an old keyword is renamed to the new one and a warning mentioning the old name and "deprecated" is
        # logged.
        @deprecated_kwargs(old_dim="new_dim")
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(old_dim=128)

        assert result == {"new_dim": 128}
        assert "old_dim" in caplog.text
        assert "deprecated" in caplog.text

    def test_new_name_no_warning(self, caplog):
        # [FUNCTION DOC]
        # Asserts the new keyword passes through without a deprecation warning.
        @deprecated_kwargs(old_dim="new_dim")
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(new_dim=256)

        assert result == {"new_dim": 256}
        assert "deprecated" not in caplog.text

    def test_both_provided_new_wins(self, caplog):
        # [FUNCTION DOC]
        # Asserts that when both names are given, the new value is kept, the old dropped, and a warning logged.
        @deprecated_kwargs(old_size="new_size")
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(old_size=128, new_size=256)

        assert result == {"new_size": 256}
        assert "old_size" in caplog.text

    def test_multiple_renames(self, caplog):
        # [FUNCTION DOC]
        # Asserts several renames apply at once.
        @deprecated_kwargs(alpha="a", beta="b")
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(alpha=1, beta=2)

        assert result == {"a": 1, "b": 2}

    def test_preserves_positional_args(self):
        # [FUNCTION DOC]
        # Asserts positional arguments are passed through unchanged while a keyword is renamed.
        @deprecated_kwargs(old="new")
        def func(pos1, pos2, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return pos1, pos2, kwargs

        assert func("x", "y", old=1) == ("x", "y", {"new": 1})

    def test_works_on_methods(self, caplog):
        # [FUNCTION DOC]
        # Asserts the decorator works on an instance method of a locally defined class and logs a warning.
        class Foo:
            # [CLASS DOC]
            # HELPER: Local class with one decorated method.
            @deprecated_kwargs(old_field="new_field")
            def method(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns the keyword arguments it received as a dict.
                return kwargs

        with caplog.at_level(logging.WARNING):
            result = Foo().method(old_field=42)

        assert result == {"new_field": 42}
        assert "deprecated" in caplog.text


class TestTransformerKwargsDecorator:
    # [CLASS DOC]
    # Groups tests for transformer_kwargs_decorator (model_args, tokenizer_args, config_args renamed to
    # model_kwargs, processor_kwargs, config_kwargs; cache_dir copied into each kwargs dict).
    @pytest.mark.parametrize(
        "old_name, new_name",
        [
            ("model_args", "model_kwargs"),
            ("tokenizer_args", "processor_kwargs"),
            ("config_args", "config_kwargs"),
        ],
    )
    def test_rename(self, caplog, old_name, new_name):
        # [FUNCTION DOC]
        # Parametrized over the renames; asserts the old name is renamed with a deprecation warning naming it.
        @transformer_kwargs_decorator
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(**{old_name: {"key": "val"}})

        assert result == {new_name: {"key": "val"}}
        assert old_name in caplog.text
        assert "deprecated" in caplog.text

    @pytest.mark.parametrize(
        "old_name, new_name",
        [
            ("model_args", "model_kwargs"),
            ("tokenizer_args", "processor_kwargs"),
            ("config_args", "config_kwargs"),
        ],
    )
    def test_new_name_no_warning(self, caplog, old_name, new_name):
        # [FUNCTION DOC]
        # Parametrized over the renames; asserts the new name passes through without warning about the old name.
        @transformer_kwargs_decorator
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(**{new_name: {"key": "val"}})

        assert result == {new_name: {"key": "val"}}
        assert old_name not in caplog.text

    @pytest.mark.parametrize(
        "old_name, new_name",
        [
            ("model_args", "model_kwargs"),
            ("tokenizer_args", "processor_kwargs"),
            ("config_args", "config_kwargs"),
        ],
    )
    def test_both_provided_new_wins(self, caplog, old_name, new_name):
        # [FUNCTION DOC]
        # Parametrized over the renames; asserts the new value wins and the old name is warned about.
        @transformer_kwargs_decorator
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(**{old_name: {"old": True}, new_name: {"new": True}})

        assert result == {new_name: {"new": True}}
        assert old_name in caplog.text

    def test_cache_dir_distributed(self, caplog):
        # [FUNCTION DOC]
        # Asserts a cache_dir is removed and placed into all three kwargs dicts, with a warning mentioning cache_dir.
        @transformer_kwargs_decorator
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(cache_dir="/tmp/cache")

        assert result == {
            "model_kwargs": {"cache_dir": "/tmp/cache"},
            "processor_kwargs": {"cache_dir": "/tmp/cache"},
            "config_kwargs": {"cache_dir": "/tmp/cache"},
        }
        assert "cache_dir" in caplog.text

    def test_cache_dir_does_not_override_existing(self, caplog):
        # [FUNCTION DOC]
        # Asserts cache_dir does not override a cache_dir already present in model_kwargs but fills the others.
        @transformer_kwargs_decorator
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(cache_dir="/tmp/cache", model_kwargs={"cache_dir": "/existing"})

        assert result["model_kwargs"]["cache_dir"] == "/existing"
        assert result["processor_kwargs"]["cache_dir"] == "/tmp/cache"

    def test_cache_dir_none_no_warning(self, caplog):
        # [FUNCTION DOC]
        # Asserts a None cache_dir is simply dropped without a warning.
        @transformer_kwargs_decorator
        def func(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(cache_dir=None)

        assert result == {}
        assert "cache_dir" not in caplog.text


class TestCrossEncoderInitArgsDecorator:
    # [CLASS DOC]
    # Groups tests for cross_encoder_init_args_decorator on functions shaped like CrossEncoder's constructor.
    @pytest.mark.parametrize(
        "old_name, new_name",
        [
            ("model_name", "model_name_or_path"),
            ("automodel_args", "model_kwargs"),
            ("tokenizer_args", "processor_kwargs"),
            ("tokenizer_kwargs", "processor_kwargs"),
            ("config_args", "config_kwargs"),
            ("cache_dir", "cache_folder"),
            ("default_activation_function", "activation_fn"),
        ],
    )
    def test_rename(self, caplog, old_name, new_name):
        # [FUNCTION DOC]
        # Parametrized over the legacy init names; asserts each is renamed to its new name with a warning.
        @cross_encoder_init_args_decorator
        def func(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(None, **{old_name: "value"})

        assert result == {new_name: "value"}
        assert old_name in caplog.text

    @pytest.mark.parametrize(
        "old_name, new_name",
        [
            ("model_name", "model_name_or_path"),
            ("automodel_args", "model_kwargs"),
            ("tokenizer_args", "processor_kwargs"),
            ("tokenizer_kwargs", "processor_kwargs"),
            ("config_args", "config_kwargs"),
            ("cache_dir", "cache_folder"),
            ("default_activation_function", "activation_fn"),
        ],
    )
    def test_both_provided_new_wins(self, caplog, old_name, new_name):
        # [FUNCTION DOC]
        # Parametrized over the legacy init names; asserts the new value wins when both are given.
        @cross_encoder_init_args_decorator
        def func(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(None, **{old_name: "old_val", new_name: "new_val"})

        assert result == {new_name: "new_val"}

    def test_classifier_dropout(self, caplog):
        # [FUNCTION DOC]
        # Asserts classifier_dropout is moved into config_kwargs with a warning.
        @cross_encoder_init_args_decorator
        def func(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(None, classifier_dropout=0.5)

        assert result == {"config_kwargs": {"classifier_dropout": 0.5}}
        assert "classifier_dropout" in caplog.text

    def test_classifier_dropout_merges_into_existing_config_kwargs(self, caplog):
        # [FUNCTION DOC]
        # Asserts classifier_dropout is merged into existing config_kwargs.
        @cross_encoder_init_args_decorator
        def func(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the keyword arguments it received as a dict.
            return kwargs

        with caplog.at_level(logging.WARNING):
            result = func(None, classifier_dropout=0.5, config_kwargs={"other": True})

        assert result == {"config_kwargs": {"classifier_dropout": 0.5, "other": True}}

    def test_positional_num_labels(self, caplog):
        # [FUNCTION DOC]
        # Asserts a positional argument after the model name becomes num_labels with a warning mentioning num_labels
        # and "positional".
        @cross_encoder_init_args_decorator
        def func(self, model_name_or_path, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the model name argument and the received keyword arguments.
            return model_name_or_path, kwargs

        with caplog.at_level(logging.WARNING):
            model_name, kwargs = func(None, "my-model", 3)

        assert model_name == "my-model"
        assert kwargs == {"num_labels": 3}
        assert "num_labels" in caplog.text
        assert "positional" in caplog.text

    def test_positional_all_four_legacy_args(self, caplog):
        # [FUNCTION DOC]
        # Asserts the legacy positional order (num_labels, max_length, activation function, device) is mapped to the
        # matching keyword arguments.
        """Old signature: CrossEncoder(model_name, num_labels, max_length, activation_fn, device)."""
        import torch

        @cross_encoder_init_args_decorator
        def func(self, model_name_or_path, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the model name argument and the received keyword arguments.
            return model_name_or_path, kwargs

        with caplog.at_level(logging.WARNING):
            model_name, kwargs = func(None, "my-model", 2, 512, torch.nn.Sigmoid(), "cuda")

        assert model_name == "my-model"
        assert kwargs["num_labels"] == 2
        assert kwargs["max_length"] == 512
        assert isinstance(kwargs["activation_fn"], torch.nn.Sigmoid)
        assert kwargs["device"] == "cuda"

    def test_positional_does_not_override_kwarg(self, caplog):
        # [FUNCTION DOC]
        # Asserts an explicit num_labels keyword wins over a positional value.
        @cross_encoder_init_args_decorator
        def func(self, model_name_or_path, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the model name argument and the received keyword arguments.
            return model_name_or_path, kwargs

        with caplog.at_level(logging.WARNING):
            model_name, kwargs = func(None, "my-model", 2, num_labels=5)

        assert model_name == "my-model"
        # Keyword arg wins over positional
        assert kwargs == {"num_labels": 5}

    def test_no_positional_no_warning(self, caplog):
        # [FUNCTION DOC]
        # Asserts keyword-only usage produces no positional warning.
        @cross_encoder_init_args_decorator
        def func(self, model_name_or_path, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the model name argument and the received keyword arguments.
            return model_name_or_path, kwargs

        with caplog.at_level(logging.WARNING):
            model_name, kwargs = func(None, "my-model", num_labels=2)

        assert model_name == "my-model"
        assert kwargs == {"num_labels": 2}
        assert "positional" not in caplog.text


class TestCrossEncoderPredictRankArgsDecorator:
    # [CLASS DOC]
    # Groups tests for cross_encoder_predict_rank_args_decorator, which renames sentences and activation_fct,
    # removes num_workers, and drops convert_to_numpy/convert_to_tensor only for methods that no longer declare
    # them, naming the wrapped method in every warning. Defines predict-like and rank-like static functions.
    """The decorator adapts to the wrapped method: it only drops convert_to_numpy / convert_to_tensor
    for methods that no longer declare them, and names the method it wrapped in every warning."""

    @staticmethod
    def _predict(self, inputs=None, convert_to_numpy=True, convert_to_tensor=False, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Predict-like function returning a dict of its inputs, both conversion flags (defaults True and False)
        # and any extra keyword arguments.
        return {
            "inputs": inputs,
            "convert_to_numpy": convert_to_numpy,
            "convert_to_tensor": convert_to_tensor,
            **kwargs,
        }

    @staticmethod
    def _rank(self, query=None, top_k=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Rank-like function returning a dict of query, top_k (both default None) and extra keyword arguments.
        return {"query": query, "top_k": top_k, **kwargs}

    def test_all_deprecated_kwargs(self, caplog):
        # [FUNCTION DOC]
        # Asserts sentences becomes inputs, activation_fct becomes activation_fn, and num_workers is removed with a
        # warning naming the wrapped method.
        """Renames (sentences→inputs, activation_fct→activation_fn) are wired via deprecated_kwargs.
        num_workers is removed as a no-op."""
        predict = cross_encoder_predict_rank_args_decorator(self._predict)

        with caplog.at_level(logging.WARNING):
            result = predict(None, sentences=[["a", "b"]], activation_fct="sigmoid", num_workers=4)

        assert result["inputs"] == [["a", "b"]]
        assert result["activation_fn"] == "sigmoid"
        assert "CrossEncoder._predict `num_workers`" in caplog.text

    def test_convert_kwargs_kept_for_predict(self, caplog):
        # [FUNCTION DOC]
        # Asserts conversion flags pass through for the predict-like function with no log output.
        """predict still declares both, so they must pass through untouched and unwarned."""
        predict = cross_encoder_predict_rank_args_decorator(self._predict)

        with caplog.at_level(logging.WARNING):
            result = predict(None, convert_to_tensor=True)

        assert result["convert_to_tensor"] is True
        assert caplog.text == ""

    def test_convert_kwargs_dropped_for_rank(self, caplog):
        # [FUNCTION DOC]
        # Asserts conversion flags are dropped for the rank-like function with warnings naming it, and activation_fct is
        # renamed.
        """rank no longer declares them, so they are dropped as no-ops and the warning names rank."""
        rank = cross_encoder_predict_rank_args_decorator(self._rank)

        with caplog.at_level(logging.WARNING):
            result = rank(None, activation_fct="sigmoid", convert_to_numpy=False, convert_to_tensor=True)

        assert result == {"query": None, "top_k": None, "activation_fn": "sigmoid"}
        assert "CrossEncoder._rank `convert_to_numpy`" in caplog.text
        assert "CrossEncoder._rank `convert_to_tensor`" in caplog.text

    def test_no_warning_without_deprecated_kwargs(self, caplog):
        # [FUNCTION DOC]
        # Asserts no log output when only current keywords are used.
        rank = cross_encoder_predict_rank_args_decorator(self._rank)

        with caplog.at_level(logging.WARNING):
            result = rank(None, top_k=3)

        assert result == {"query": None, "top_k": 3}
        assert caplog.text == ""


class TestSaveToHubArgsDecorator:
    # [CLASS DOC]
    # Groups tests for save_to_hub_args_decorator (repo_name renamed to repo_id, and a None inserted for the token
    # argument when two or more positional arguments are passed).
    def test_repo_name_renamed_to_repo_id(self, caplog):
        # [FUNCTION DOC]
        # Asserts repo_name is renamed to repo_id with a warning.
        @save_to_hub_args_decorator
        def func(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the positional arguments (excluding self) and keyword arguments it received.
            return args, kwargs

        with caplog.at_level(logging.WARNING):
            _, kwargs = func(None, repo_name="my-repo")

        assert kwargs == {"repo_id": "my-repo"}
        assert "repo_name" in caplog.text

    def test_repo_id_no_warning(self, caplog):
        # [FUNCTION DOC]
        # Asserts repo_id passes through without a warning.
        @save_to_hub_args_decorator
        def func(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the positional arguments (excluding self) and keyword arguments it received.
            return args, kwargs

        with caplog.at_level(logging.WARNING):
            _, kwargs = func(None, repo_id="my-repo")

        assert kwargs == {"repo_id": "my-repo"}
        assert "repo_name" not in caplog.text

    def test_both_provided_repo_id_wins(self, caplog):
        # [FUNCTION DOC]
        # Asserts repo_id wins when both are given.
        @save_to_hub_args_decorator
        def func(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the positional arguments (excluding self) and keyword arguments it received.
            return args, kwargs

        with caplog.at_level(logging.WARNING):
            _, kwargs = func(None, repo_name="old", repo_id="new")

        assert kwargs == {"repo_id": "new"}

    def test_positional_args_adjusted(self):
        # [FUNCTION DOC]
        # Asserts two positional arguments become three with None appended at position three.
        """When >= 2 positional args are passed, a None is inserted at position 2 for the `token` arg."""

        @save_to_hub_args_decorator
        def func(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the positional arguments (excluding self) and keyword arguments it received.
            return args, kwargs

        args, _ = func(None, "repo", "org")
        # Original (repo, org) → (repo, org, None)
        assert args == ("repo", "org", None)

    def test_single_positional_arg_unchanged(self):
        # [FUNCTION DOC]
        # Asserts a single positional argument is left unchanged.
        @save_to_hub_args_decorator
        def func(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the positional arguments (excluding self) and keyword arguments it received.
            return args, kwargs

        args, _ = func(None, "repo")
        assert args == ("repo",)
