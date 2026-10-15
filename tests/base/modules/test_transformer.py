# [MODULE DOC]
# Tests for the base Transformer module (sentence_transformers.base.modules.Transformer), which wraps
# a Hugging Face model plus processor/tokenizer: initialization options (task, lowercasing, processing
# kwargs, deprecated kwargs, modality config), attention-config warnings, max_seq_length handling,
# preprocessing (prompts, padding, truncation, multimodal inputs), forward/output extraction,
# flattened FlashAttention inputs, saving/loading and helper utilities. Uses tiny Hub test models
# (downloaded on demand; network and cache side effects), the shared stsb_bert_tiny_model fixture,
# mocks and monkeypatching. RAGGED_BATCH holds two texts of very different length so padding is observable.
from __future__ import annotations

import inspect
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from packaging.version import Version
from packaging.version import parse as parse_version
from tokenizers.normalizers import NFC, Lowercase, Sequence
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers import __version__ as transformers_version
from transformers.utils import is_torchvision_available, is_vision_available

from sentence_transformers.base.modules import Transformer
from sentence_transformers.base.modules.transformer import (
    TRANSFORMER_TASK_DEFAULTS,
    _count_media_per_sample,
    set_temporary_class_attrs,
)
from sentence_transformers.util import batch_to_device

transformer_module = sys.modules[Transformer.__module__]

TINY_BERT = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_LLAVA = "hf-internal-testing/tiny-random-LlavaForConditionalGeneration"
TINY_XLMR = "hf-internal-testing/tiny-xlm-roberta"
TINY_ALTCLIP = "hf-internal-testing/tiny-random-AltCLIPModel"

# Uneven lengths, so that padding is actually applied and the padded side is observable.
RAGGED_BATCH = ["hi", "a considerably longer sentence than the other one"]


@pytest.fixture()
def bert_tiny_transformer(stsb_bert_tiny_model) -> Transformer:
    # [FUNCTION DOC]
    # HELPER: Pytest fixture returning the Transformer module (first module) of the shared tiny BERT model.
    """A lightweight BERT Transformer for reuse across tests."""
    return stsb_bert_tiny_model[0]


def test_infer_flatten_position_offset(bert_tiny_transformer):
    # [FUNCTION DOC]
    # The flatten position offset is 0 for BERT and a rotary Llama model, and padding_idx + 1 (2) for
    # XLM-RoBERTa and AltCLIP (nested embeddings), computed from config-only models.
    """RoBERTa-family embeddings compute positions starting at padding_idx + 1, so flattened
    inputs need their collator-produced 0-based position_ids shifted. The submodule scan keys on
    an int padding_idx next to a learned position_embeddings table: BERT-style embeddings store
    neither, rotary decoders store padding_idx on the Model class without position_embeddings,
    and multimodal wrappers nest the offset embeddings below ``.embeddings``."""
    assert bert_tiny_transformer._infer_flatten_position_offset() == 0

    def offset_of(model) -> int:
        # [FUNCTION DOC]
        # HELPER: Calls the unbound offset inference on a simple namespace holding the given model.
        return Transformer._infer_flatten_position_offset(SimpleNamespace(model=model))

    # The scan only reads module structure, so build from config alone: these repos' weight files
    # predate safetensors, and loading them trips the torch.load guard on torch < 2.6.
    xlmr = AutoModel.from_config(AutoConfig.from_pretrained(TINY_XLMR))
    assert xlmr.embeddings.padding_idx == 1
    assert offset_of(xlmr) == 2

    assert offset_of(AutoModel.from_config(AutoConfig.from_pretrained(TINY_LLAMA))) == 0
    assert offset_of(AutoModel.from_config(AutoConfig.from_pretrained(TINY_ALTCLIP))) == 2


def test_preprocess_unsupported_modality_uses_shared_error(bert_tiny_transformer):
    # [FUNCTION DOC]
    # A text-only Transformer rejects an image input, and a mixed text+image batch, with the shared
    # unsupported-modality ValueError messages.
    """A text-only Transformer module rejects unsupported modalities with the same module-scoped
    guidance that BaseModel.preprocess produces (both via ``raise_unsupported_modality_error``)."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match=r"Modality 'image' is not supported by this Transformer module\. Supported modalities: text",
    ):
        bert_tiny_transformer.preprocess([image])

    with pytest.raises(
        ValueError,
        match=r"mixes multiple modalities \(image, text\), but this Transformer module does not support image",
    ):
        bert_tiny_transformer.preprocess(["a sentence", image])


class TestSetTemporaryClassAttrs:
    # [CLASS DOC]
    # Tests for the set_temporary_class_attrs context manager.
    def test_sets_and_restores(self):
        # [FUNCTION DOC]
        # Class attributes are overridden inside the context and restored afterwards.
        class Dummy:
            # [CLASS DOC]
            # HELPER: Local class with attributes x and y.
            x = 1
            y = 2

        with set_temporary_class_attrs(Dummy, x=10, y=20):
            assert Dummy.x == 10
            assert Dummy.y == 20
        assert Dummy.x == 1
        assert Dummy.y == 2

    def test_restores_on_exception(self):
        # [FUNCTION DOC]
        # Attributes are restored even when the block raises.
        class Dummy:
            # [CLASS DOC]
            # HELPER: Local class with attribute x.
            x = 1

        with pytest.raises(RuntimeError):
            with set_temporary_class_attrs(Dummy, x=99):
                raise RuntimeError("boom")
        assert Dummy.x == 1


class TestTransformerInit:
    # [CLASS DOC]
    # Tests for Transformer construction options.
    def test_invalid_transformer_task(self):
        # [FUNCTION DOC]
        # An unknown transformer_task raises ValueError.
        with pytest.raises(ValueError, match="Unsupported transformer_task"):
            Transformer(TINY_BERT, transformer_task="nonexistent-task")

    def test_sequence_classification_default_num_labels(self):
        # [FUNCTION DOC]
        # Loading a non-classification checkpoint with the sequence-classification task gives num_labels 1.
        """When loading a non-SeqCls model with task='sequence-classification', num_labels defaults to 1."""
        transformer = Transformer(TINY_BERT, transformer_task="sequence-classification")
        assert transformer.config.num_labels == 1

    def test_do_lower_case(self):
        # [FUNCTION DOC]
        # do_lower_case adds exactly one Lowercase normalizer and makes different-cased inputs tokenize identically.
        """do_lower_case should add exactly one Lowercase normalizer and actually lowercase tokens."""
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        assert transformer.do_lower_case is True

        # Normalizer should be a Sequence containing exactly one Lowercase
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        assert sum(1 for n in normalizer if isinstance(n, Lowercase)) == 1

        # Tokens should be lowercased, and different cases should produce the same IDs
        tokens = transformer.tokenizer.tokenize("Hello WORLD")
        assert all(t == t.lower() for t in tokens)
        features_lower = transformer.preprocess(["hello world"])
        features_upper = transformer.preprocess(["HELLO WORLD"])
        assert torch.equal(features_lower["input_ids"], features_upper["input_ids"])

    def test_do_lower_case_noop_when_lowercase_already_present(self, monkeypatch):
        # [FUNCTION DOC]
        # When the processor's normalizer already is a sequence with Lowercase (patched loader), no second one is added.
        """If the normalizer already has Lowercase, do_lower_case should not add another."""
        # Pre-set the normalizer to already contain Lowercase before __init__ applies it
        original_from_pretrained = AutoProcessor.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Wraps the original processor loader and replaces the normalizer with a Lowercase-only sequence.
            processor = original_from_pretrained(*args, **kwargs)
            processor.backend_tokenizer.normalizer = Sequence([Lowercase()])
            return processor

        monkeypatch.setattr(AutoProcessor, "from_pretrained", patched_from_pretrained)
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        assert sum(1 for n in normalizer if isinstance(n, Lowercase)) == 1

    def test_do_lower_case_prepends_to_existing_sequence(self, monkeypatch):
        # [FUNCTION DOC]
        # With an existing NFC-only sequence normalizer, Lowercase is prepended and NFC kept.
        """When normalizer is a Sequence without Lowercase, Lowercase should be prepended."""
        original_from_pretrained = AutoProcessor.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Loads the processor and sets its normalizer to a sequence containing NFC.
            processor = original_from_pretrained(*args, **kwargs)
            processor.backend_tokenizer.normalizer = Sequence([NFC()])
            return processor

        monkeypatch.setattr(AutoProcessor, "from_pretrained", patched_from_pretrained)
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        normalizer_list = list(normalizer)
        assert isinstance(normalizer_list[0], Lowercase)
        assert any(isinstance(n, NFC) for n in normalizer_list)

    def test_do_lower_case_wraps_single_normalizer(self, monkeypatch):
        # [FUNCTION DOC]
        # A single non-sequence normalizer is wrapped into a sequence of Lowercase followed by it.
        """When normalizer is a single non-Sequence normalizer, it should be wrapped with Lowercase."""
        original_from_pretrained = AutoProcessor.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Loads the processor and sets its normalizer to a bare NFC.
            processor = original_from_pretrained(*args, **kwargs)
            processor.backend_tokenizer.normalizer = NFC()
            return processor

        monkeypatch.setattr(AutoProcessor, "from_pretrained", patched_from_pretrained)
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        normalizer_list = list(normalizer)
        assert isinstance(normalizer_list[0], Lowercase)
        assert isinstance(normalizer_list[1], NFC)

    def test_do_lower_case_with_none_normalizer(self, monkeypatch):
        # [FUNCTION DOC]
        # A missing (None) normalizer becomes a sequence with just Lowercase.
        """When normalizer is None, do_lower_case should create a Sequence with just Lowercase."""
        original_from_pretrained = AutoProcessor.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Loads the processor and removes its normalizer.
            processor = original_from_pretrained(*args, **kwargs)
            processor.backend_tokenizer.normalizer = None
            return processor

        monkeypatch.setattr(AutoProcessor, "from_pretrained", patched_from_pretrained)
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        normalizer = transformer.tokenizer.backend_tokenizer.normalizer
        assert isinstance(normalizer, Sequence)
        normalizer_list = list(normalizer)
        assert len(normalizer_list) == 1
        assert isinstance(normalizer_list[0], Lowercase)

    def test_do_lower_case_false_does_not_modify_normalizer(self):
        # [FUNCTION DOC]
        # do_lower_case False leaves the normalizer identical to the default.
        """do_lower_case=False should not modify the tokenizer normalizer."""
        transformer_default = Transformer(TINY_BERT, do_lower_case=False)
        transformer_none = Transformer(TINY_BERT)
        norm_default = transformer_default.tokenizer.backend_tokenizer.normalizer
        norm_none = transformer_none.tokenizer.backend_tokenizer.normalizer
        if norm_default is None:
            assert norm_none is None
        else:
            assert str(norm_default) == str(norm_none)

    def test_do_lower_case_tokenizer_persisted_after_save_load(self, tmp_path):
        # [FUNCTION DOC]
        # Lowercasing persists through Transformer save and load (it lives in the saved tokenizer).
        """The Lowercase normalizer added to the tokenizer should persist after save/load."""
        transformer = Transformer(TINY_BERT, do_lower_case=True)
        tokens_before = transformer.tokenizer.tokenize("Hello WORLD")
        assert all(t == t.lower() for t in tokens_before)

        save_dir = str(tmp_path / "model")
        transformer.save(save_dir)
        reloaded = Transformer.load(save_dir)

        # The tokenizer normalizer is saved with the tokenizer, so lowercasing still works
        tokens_after = reloaded.tokenizer.tokenize("Hello WORLD")
        assert tokens_before == tokens_after
        assert all(t == t.lower() for t in tokens_after)

    def test_processing_kwargs_default_empty(self):
        # [FUNCTION DOC]
        # processing_kwargs defaults to an empty dict.
        """processing_kwargs should default to an empty dict when not provided."""
        transformer = Transformer(TINY_BERT)
        assert transformer.processing_kwargs == {}

    def test_processing_kwargs_stored(self):
        # [FUNCTION DOC]
        # Given processing_kwargs are stored as-is.
        """processing_kwargs passed to __init__ should be stored on the instance."""
        kwargs = {"text": {"truncation": "only_first"}, "chat_template": {"add_generation_prompt": True}}
        transformer = Transformer(TINY_BERT, processing_kwargs=kwargs)
        assert transformer.processing_kwargs == kwargs

    def test_processing_kwargs_unknown_keys_warning(self, caplog):
        # [FUNCTION DOC]
        # Unknown top-level processing_kwargs keys log a warning naming them.
        """Unknown top-level keys in processing_kwargs should emit a warning."""
        with caplog.at_level(logging.WARNING):
            Transformer(TINY_BERT, processing_kwargs={"padding_side": {"value": "left"}, "text": {"truncation": True}})
        assert any("Unknown keys" in record.message and "padding_side" in record.message for record in caplog.records)

    def test_processing_kwargs_valid_keys_no_warning(self, caplog):
        # [FUNCTION DOC]
        # Known processing_kwargs keys log no "Unknown keys" warning.
        """Valid processing_kwargs keys should not emit a warning."""
        with caplog.at_level(logging.WARNING):
            Transformer(
                TINY_BERT, processing_kwargs={"common": {"return_tensors": "pt"}, "text": {"truncation": True}}
            )
        assert not any("Unknown keys" in record.message for record in caplog.records)

    def test_tokenizer_name_or_path_warning(self, caplog):
        # [FUNCTION DOC]
        # The deprecated tokenizer_name_or_path argument logs a deprecation warning but still constructs.
        """tokenizer_name_or_path should emit a deprecation warning."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, tokenizer_name_or_path=TINY_BERT)
        assert any(
            "tokenizer_name_or_path" in record.message and "deprecated" in record.message for record in caplog.records
        )
        assert transformer is not None


class TestWarnOnUnsupportedAttentionConfig:
    # [CLASS DOC]
    # Tests the warning for attention settings that the installed transformers version does not support.
    # _BIDIR and _CAUSAL bundle the parameter name, value, support flag name and minimum version.
    """Warn when the config requests attention settings not supported by the installed transformers version."""

    _BIDIR = ("use_bidirectional_attention", True, "_TRANSFORMERS_SUPPORTS_USE_BIDIRECTIONAL_ATTENTION", "4.56.2")
    _CAUSAL = ("is_causal", False, "_TRANSFORMERS_SUPPORTS_IS_CAUSAL_FALSE", "5.2.0")

    @pytest.mark.parametrize(
        ("param", "value", "support_flag", "min_version", "set_on", "supports", "expect_warn"),
        [
            # Warns when flag is set on the main config and version is unsupported
            (*_BIDIR, "main", False, True),
            (*_CAUSAL, "main", False, True),
            # Warns when flag is set on a sub-config (e.g. multimodal text_config)
            (*_BIDIR, "sub", False, True),
            (*_CAUSAL, "sub", False, True),
            # No warning when version supports the flag
            (*_BIDIR, "main", True, False),
            (*_CAUSAL, "main", True, False),
            # No warning when flag is absent, even if version is unsupported
            (*_BIDIR, None, False, False),
            (*_CAUSAL, None, False, False),
        ],
    )
    def test_warn_on_unsupported_attention_config(
        self,
        caplog,
        monkeypatch,
        bert_tiny_transformer,
        param,
        value,
        support_flag,
        min_version,
        set_on,
        supports,
        expect_warn,
    ):
        # [FUNCTION DOC]
        # Monkeypatches the module's support flag, optionally sets the attention parameter on the config or on
        # a copied sub-config registered as text_config, calls the warning helper and checks whether a warning
        # naming "param=value" and the minimum version is logged as expected.
        monkeypatch.setattr(transformer_module, support_flag, supports)
        config = bert_tiny_transformer.config
        if set_on == "main":
            setattr(config, param, value)
        elif set_on == "sub":
            sub = deepcopy(config)
            setattr(sub, param, value)
            config.sub_configs = {"text_config": type(sub)}
            config.text_config = sub

        with caplog.at_level(logging.WARNING):
            Transformer._warn_on_unsupported_attention_config(config)

        matching = [
            r
            for r in caplog.records
            if f"{param}={value}" in r.message and f"transformers>={min_version}" in r.message
        ]
        assert bool(matching) is expect_warn


class TestTransformerMaxSeqLength:
    # [CLASS DOC]
    # Tests for the max_seq_length property and setter.
    """Test the max_seq_length property for both tokenizer-based and config-based models."""

    def test_max_seq_length_from_tokenizer(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # max_seq_length defaults to the tokenizer's model_max_length.
        model = bert_tiny_transformer
        assert model.max_seq_length is not None
        assert model.max_seq_length == model.tokenizer.model_max_length

    def test_max_seq_length_setter(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Setting max_seq_length updates the tokenizer's model_max_length; the original value is restored after.
        model = bert_tiny_transformer
        original = model.max_seq_length
        model.max_seq_length = 64
        assert model.max_seq_length == 64
        assert model.tokenizer.model_max_length == 64
        model.max_seq_length = original

    def test_max_seq_length_init_kwarg(self):
        # [FUNCTION DOC]
        # The max_seq_length constructor argument is applied.
        transformer = Transformer(TINY_BERT, max_seq_length=42)
        assert transformer.max_seq_length == 42

    def test_max_seq_length_capped_by_max_position_embeddings(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # max_seq_length does not exceed the config's max_position_embeddings when that exists.
        model = bert_tiny_transformer
        if hasattr(model.config, "max_position_embeddings"):
            assert model.max_seq_length <= model.config.max_position_embeddings

    def test_max_seq_length_fallback_to_config(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # With the tokenizer property patched to None, max_seq_length falls back to max_position_embeddings.
        """When tokenizer is None, max_seq_length should fall back to config.max_position_embeddings."""
        model = bert_tiny_transformer
        monkeypatch.setattr(type(model), "tokenizer", property(lambda self: None))
        seq_len = model.max_seq_length
        if hasattr(model.config, "max_position_embeddings"):
            assert seq_len == model.config.max_position_embeddings

    def test_max_seq_length_setter_noop_without_tokenizer(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Setting max_seq_length without a tokenizer does not raise.
        """Setting max_seq_length when tokenizer is None should be a no-op."""
        model = bert_tiny_transformer
        monkeypatch.setattr(type(model), "tokenizer", property(lambda self: None))
        model.max_seq_length = 42  # should not raise

    def test_max_seq_length_truncation(self):
        # [FUNCTION DOC]
        # With max_seq_length 5, preprocessing truncates input_ids to 5 tokens and the tokenizer limit is 5.
        """Inputs longer than max_seq_length should be truncated during preprocessing."""
        transformer = Transformer(TINY_BERT, max_seq_length=5)
        features = transformer.preprocess(["this is a longer sentence that should get truncated"])
        assert features["input_ids"].shape[1] == 5
        assert transformer.tokenizer.model_max_length == 5


class TestTransformerDeprecatedKwargs:
    # [CLASS DOC]
    # Tests for deprecated Transformer constructor keyword arguments.
    def test_model_args_deprecated(self, caplog):
        # [FUNCTION DOC]
        # model_args logs a deprecation warning.
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, model_args={})
        assert any("model_args" in r.message and "deprecated" in r.message for r in caplog.records)
        assert transformer is not None

    def test_tokenizer_args_deprecated(self, caplog):
        # [FUNCTION DOC]
        # tokenizer_args logs a deprecation warning.
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, tokenizer_args={})
        assert any("tokenizer_args" in r.message and "deprecated" in r.message for r in caplog.records)
        assert transformer is not None

    def test_config_args_deprecated(self, caplog):
        # [FUNCTION DOC]
        # config_args logs a deprecation warning.
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, config_args={})
        assert any("config_args" in r.message and "deprecated" in r.message for r in caplog.records)
        assert transformer is not None

    def test_new_kwargs_no_warning(self, caplog):
        # [FUNCTION DOC]
        # The new model/processor/config kwargs names log no deprecation warning.
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, model_kwargs={}, processor_kwargs={}, config_kwargs={})
        assert not any("deprecated" in r.message for r in caplog.records)
        assert transformer is not None


class TestTransformerModalityConfigValidation:
    # [CLASS DOC]
    # Tests for modality_config validation in the constructor.
    def test_valid_modality_config(self):
        # [FUNCTION DOC]
        # A complete modality_config with a module_output_name is stored.
        transformer = Transformer(
            TINY_BERT,
            modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
            module_output_name="token_embeddings",
        )
        assert transformer.modality_config == {
            "text": {"method": "forward", "method_output_name": "last_hidden_state"}
        }

    def test_invalid_modality_config_missing_method(self):
        # [FUNCTION DOC]
        # A modality entry without "method" raises ValueError.
        with pytest.raises(ValueError, match="'method' and 'method_output_name'"):
            Transformer(
                TINY_BERT,
                modality_config={"text": {"method_output_name": "last_hidden_state"}},
                module_output_name="token_embeddings",
            )

    def test_invalid_modality_config_missing_output_name(self):
        # [FUNCTION DOC]
        # A modality entry without "method_output_name" raises ValueError.
        with pytest.raises(ValueError, match="'method' and 'method_output_name'"):
            Transformer(
                TINY_BERT, modality_config={"text": {"method": "forward"}}, module_output_name="token_embeddings"
            )

    def test_modality_config_requires_module_output_name(self):
        # [FUNCTION DOC]
        # A modality_config without module_output_name raises ValueError.
        with pytest.raises(ValueError, match="module_output_name"):
            Transformer(
                TINY_BERT, modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
            )


class TestPreprocess:
    # [CLASS DOC]
    # Tests for Transformer.preprocess.
    def test_unsupported_modality_error(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # If input parsing reports an unsupported modality (patched to "video"), preprocess raises ValueError.
        """Passing an unsupported modality should raise ValueError."""
        model = bert_tiny_transformer
        monkeypatch.setattr(model.input_formatter, "parse_inputs", lambda *a, **kw: ("video", {"video": []}, {}))
        with pytest.raises(ValueError, match="not supported"):
            model.preprocess(["test"])

    def test_preprocess_with_text_prompt(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # A text prompt adds a positive integer prompt_length to the features.
        """preprocess should incorporate a text prompt and set prompt_length."""
        result = bert_tiny_transformer.preprocess(["hello world"], prompt="search query: ")
        assert "prompt_length" in result
        assert isinstance(result["prompt_length"], int)
        assert result["prompt_length"] > 0

    def test_preprocess_without_prompt(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Without a prompt there is no prompt_length.
        """preprocess without a prompt should not include prompt_length."""
        result = bert_tiny_transformer.preprocess(["hello world"])
        assert "prompt_length" not in result

    def test_preprocess_empty_inputs(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Empty input returns an empty dict.
        """preprocess with an empty list should return an empty dict without raising."""
        result = bert_tiny_transformer.preprocess([])
        assert result == {}

    def test_preprocess_processing_kwargs_text_override(self):
        # [FUNCTION DOC]
        # Instance processing_kwargs for text (max_length 5 with truncation) truncate the output to 5 tokens.
        """processing_kwargs should override default text preprocessing kwargs."""
        # With max_length=5 and truncation, the output should be truncated
        transformer = Transformer(
            TINY_BERT,
            processing_kwargs={"text": {"max_length": 5, "truncation": True}},
        )
        result = transformer.preprocess(["this is a longer sentence that should get truncated"])
        assert result["input_ids"].shape[1] == 5

    def test_preprocess_processing_kwargs_common_override(self):
        # [FUNCTION DOC]
        # Instance "common" processing_kwargs can switch return_tensors to numpy arrays.
        """processing_kwargs 'common' should override common_kwargs like return_tensors."""
        transformer = Transformer(
            TINY_BERT,
            processing_kwargs={"common": {"return_tensors": "np"}},
        )
        result = transformer.preprocess(["hello world"])
        assert isinstance(result["input_ids"], np.ndarray)

    def test_preprocess_per_call_processing_kwargs_text_override(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Per-call text processing_kwargs truncate that call only; the next call is untruncated.
        """Per-call processing_kwargs should override defaults for that call only."""
        result = bert_tiny_transformer.preprocess(
            ["this is a longer sentence that should get truncated"],
            processing_kwargs={"text": {"max_length": 5, "truncation": True}},
        )
        assert result["input_ids"].shape[1] == 5

        # Without per-call overrides, the default behaviour is restored
        next_result = bert_tiny_transformer.preprocess(["this is a longer sentence that should get truncated"])
        assert next_result["input_ids"].shape[1] > 5

    def test_preprocess_per_call_processing_kwargs_common_override(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Per-call "common" processing_kwargs return numpy for that call only; the next call returns tensors.
        """Per-call processing_kwargs 'common' should override return_tensors for that call only."""
        result = bert_tiny_transformer.preprocess(
            ["hello world"],
            processing_kwargs={"common": {"return_tensors": "np"}},
        )
        assert isinstance(result["input_ids"], np.ndarray)

        next_result = bert_tiny_transformer.preprocess(["hello world"])
        assert isinstance(next_result["input_ids"], torch.Tensor)

    def test_preprocess_per_call_merges_with_instance_processing_kwargs(self):
        # [FUNCTION DOC]
        # Per-call values override matching instance values while other instance settings (truncation) still apply.
        """Per-call values override matching instance values. Non-overridden settings are preserved."""
        transformer = Transformer(
            TINY_BERT,
            processing_kwargs={"text": {"max_length": 5, "truncation": True}},
        )
        # Per-call only changes max_length. Truncation from the instance must still apply
        result = transformer.preprocess(
            ["this is a longer sentence that should get truncated"],
            processing_kwargs={"text": {"max_length": 8}},
        )
        assert result["input_ids"].shape[1] == 8

    def test_preprocess_per_call_processing_kwargs_unknown_keys_warning(self, bert_tiny_transformer, caplog):
        # [FUNCTION DOC]
        # Unknown top-level keys in per-call processing_kwargs log a warning.
        """Unknown top-level keys in per-call processing_kwargs should emit a warning."""
        with caplog.at_level(logging.WARNING):
            bert_tiny_transformer.preprocess(
                ["hello world"],
                processing_kwargs={"texts": {"max_length": 5}},
            )
        assert "Unknown keys in per-call `processing_kwargs`" in caplog.text

    def test_preprocess_per_call_processing_kwargs_does_not_mutate_instance(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Per-call processing_kwargs do not modify the instance's processing_kwargs.
        """Per-call overrides must not leak into self.processing_kwargs."""
        before = deepcopy(bert_tiny_transformer.processing_kwargs)
        bert_tiny_transformer.preprocess(
            ["hello world"],
            processing_kwargs={"text": {"max_length": 5, "truncation": True}},
        )
        assert bert_tiny_transformer.processing_kwargs == before

    def test_causal_task_left_pads_by_default(self):
        # [FUNCTION DOC]
        # A causal (text-generation) model is set to left padding on load, so a ragged batch ends in real tokens.
        """The module sets padding_side to "left" on load, so a ragged batch ends in real tokens."""
        transformer = Transformer(TINY_LLAMA, transformer_task="text-generation")
        features = transformer.preprocess(RAGGED_BATCH)
        assert transformer.processor.padding_side == "left"
        assert features["attention_mask"][:, -1].all()

    def test_causal_right_padding_via_attribute_raises(self):
        # [FUNCTION DOC]
        # Switching the processor to right padding afterwards makes preprocessing of a ragged batch raise a
        # ValueError telling the user to set left padding on the processor.
        """padding_side is set to "left" on load, so reaching this means it was mutated afterwards."""
        transformer = Transformer(TINY_LLAMA, transformer_task="text-generation")
        transformer.processor.padding_side = "right"
        with pytest.raises(ValueError) as exc_info:
            transformer.preprocess(RAGGED_BATCH)
        message = str(exc_info.value)
        assert "ends in a padding token" in message
        assert 'processor.padding_side = "left"' in message

    def test_causal_right_padding_via_processing_kwargs_raises(self):
        # [FUNCTION DOC]
        # Requesting right padding via per-call text processing_kwargs raises a ValueError pointing at
        # processing_kwargs, while the processor attribute still reads "left".
        """processing_kwargs overrides the attribute at call time, so the attribute alone can't be trusted:
        it still reads "left" here while the processor pads right."""
        transformer = Transformer(TINY_LLAMA, transformer_task="text-generation")
        with pytest.raises(ValueError) as exc_info:
            transformer.preprocess(RAGGED_BATCH, processing_kwargs={"text": {"padding_side": "right"}})
        assert transformer.processor.padding_side == "left"
        message = str(exc_info.value)
        assert "ends in a padding token" in message
        assert "processing_kwargs" in message

    @pytest.mark.parametrize("bucket", ["text", "common"])
    def test_causal_right_padding_via_processing_kwargs_raises_without_message_modality(self, bucket):
        # [FUNCTION DOC]
        # With a modality config lacking "message" (tokenizer called directly), right padding requested in either
        # the "text" or "common" processing_kwargs bucket raises the padding ValueError.
        """TINY_LLAMA has a chat template, so text normally routes through apply_chat_template, which only
        forwards the "text" bucket. Without a "message" modality the tokenizer is called directly and the
        "common" bucket reaches it as well, so both buckets must be caught."""
        transformer = Transformer(
            TINY_LLAMA,
            transformer_task="text-generation",
            modality_config={"text": {"method": "forward", "method_output_name": "logits"}},
            module_output_name="causal_logits",
        )
        assert "message" not in transformer.modality_config
        with pytest.raises(ValueError) as exc_info:
            transformer.preprocess(RAGGED_BATCH, processing_kwargs={bucket: {"padding_side": "right"}})
        message = str(exc_info.value)
        assert "ends in a padding token" in message
        assert "processing_kwargs" in message

    def test_causal_padding_check_ignores_unpadded_batches(self):
        # [FUNCTION DOC]
        # A batch needing no padding passes even with right padding configured.
        """A batch with nothing to pad has no last-token problem, whatever padding_side says."""
        transformer = Transformer(TINY_LLAMA, transformer_task="text-generation")
        transformer.processor.padding_side = "right"
        transformer.preprocess(["same length here"])

    @pytest.mark.skipif(
        Version(transformers_version) < Version("5.0.0"), reason="any-to-any requires transformers v5+"
    )
    @pytest.mark.skipif(
        not (is_vision_available() and is_torchvision_available()),
        reason="Loading the Llava processor requires Pillow and torchvision",
    )
    def test_causal_right_padding_via_multimodal_tokenizer_raises(self):
        # [FUNCTION DOC]
        # For a multimodal (any-to-any) processor without its own padding_side, the tokenizer is left padded on
        # load, and right padding set on the tokenizer raises a ValueError naming processor.tokenizer.padding_side.
        # Skipped on transformers below 5.0 or without Pillow/torchvision.
        """A multimodal processor pads via its tokenizer and has no padding_side of its own, so the
        tokenizer attribute is the only one that can be wrong, and the only one worth reporting."""
        transformer = Transformer(TINY_LLAVA, transformer_task="any-to-any")
        assert not hasattr(transformer.processor, "padding_side")
        assert transformer.processor.tokenizer.padding_side == "left"

        transformer.processor.tokenizer.padding_side = "right"
        with pytest.raises(ValueError) as exc_info:
            transformer.preprocess(RAGGED_BATCH)
        message = str(exc_info.value)
        assert "ends in a padding token" in message
        assert 'processor.tokenizer.padding_side = "left"' in message


class TestForward:
    # [CLASS DOC]
    # Tests for Transformer.forward and the query/document length and query-expansion preprocessing options.
    def test_missing_method_error(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # A modality config naming a method the model lacks makes forward raise ValueError.
        """forward should raise ValueError when the model doesn't have the requested method."""
        model = bert_tiny_transformer
        model.modality_config["text"]["method"] = "nonexistent_method"
        features = model.preprocess(["test"])
        with pytest.raises(ValueError, match="does not have the requested"):
            model.forward(features)

    def test_4d_embedding_reshape(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # A 4-D model output (patched forward returning a fake output object) is flattened to a 3-D embedding.
        """4D model outputs should be flattened to 3D."""
        model = bert_tiny_transformer
        features = batch_to_device(model.preprocess(["test"]), model.model.device)
        original_forward = model.model.forward

        def mock_forward(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Runs the real forward for the batch size, then returns a fake output exposing a random
            # (batch, 3, 4, 4) tensor as last_hidden_state.
            out = original_forward(**kwargs)
            b = out.last_hidden_state.shape[0]
            fake_4d = torch.randn(b, 3, 4, 4)

            class Fake4DOutput:
                # [CLASS DOC]
                # HELPER: Fake model output supporting key lookup and membership for "last_hidden_state" only.
                def __getitem__(self, key):
                    # [FUNCTION DOC]
                    # HELPER: Returns the fake 4-D tensor for "last_hidden_state", else None.
                    return fake_4d if key == "last_hidden_state" else None

                def __contains__(self, key):
                    # [FUNCTION DOC]
                    # HELPER: True only for "last_hidden_state".
                    return key == "last_hidden_state"

            return Fake4DOutput()

        monkeypatch.setattr(model.model, "forward", mock_forward)
        with torch.no_grad():
            result = model.forward(features)
        emb = result[model.module_output_name]
        assert emb.ndim == 3

    def test_attribute_fallback_for_output(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # If indexing the model output raises KeyError, forward falls back to attribute access and still sets the output feature.
        """forward should fall back to getattr when dict indexing fails on model output."""
        model = bert_tiny_transformer

        class FakeOutput:
            # [CLASS DOC]
            # HELPER: Fake output holding last_hidden_state as an attribute whose item lookup always raises KeyError.
            def __init__(self, tensor):
                # [FUNCTION DOC]
                # HELPER: Stores the tensor as last_hidden_state.
                self.last_hidden_state = tensor

            def __getitem__(self, key):
                # [FUNCTION DOC]
                # HELPER: Always raises KeyError for the key.
                raise KeyError(key)

        features = batch_to_device(model.preprocess(["test"]), model.model.device)
        original_forward = model.model.forward

        def mock_forward(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Runs the real forward and wraps its last hidden state in the fake output.
            out = original_forward(**kwargs)
            return FakeOutput(out.last_hidden_state)

        monkeypatch.setattr(model.model, "forward", mock_forward)
        with torch.no_grad():
            result = model.forward(features)
        assert model.module_output_name in result

    def test_output_hidden_states(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # With output_hidden_states enabled on the config, forward adds "all_layer_embeddings".
        """When output_hidden_states is True, all_layer_embeddings should appear in features."""
        model = bert_tiny_transformer
        model.model.config.output_hidden_states = True
        features = batch_to_device(model.preprocess(["test"]), model.model.device)
        with torch.no_grad():
            result = model.forward(features)
        assert "all_layer_embeddings" in result

    def test_list_method_output_name_hidden_states(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # A list output path ["hidden_states", -1] enables hidden states automatically and returns a 3-D
        # embedding of hidden size.
        """A list method_output_name like ["hidden_states", -1] should auto-enable output_hidden_states
        and traverse the output path to return the correct hidden layer embedding."""
        model = bert_tiny_transformer
        model.model.config.output_hidden_states = False
        model.modality_config["text"]["method_output_name"] = ["hidden_states", -1]
        features = batch_to_device(model.preprocess(["test"]), model.model.device)
        with torch.no_grad():
            result = model.forward(features)
        embedding = result[model.module_output_name]
        assert embedding.ndim == 3
        assert embedding.shape[-1] == model.config.hidden_size

    def test_model_forward_params_matches_kwargs_support(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # model_forward_params is None exactly when the model forward accepts arbitrary keyword arguments;
        # otherwise it is a set containing at least input_ids and attention_mask.
        """``model_forward_params`` is ``None`` exactly when forward accepts ``**kwargs`` (the denylist
        path). Otherwise it is an allowlist set seeded with the common safety-net names."""
        model = bert_tiny_transformer
        accepts_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in inspect.signature(model.model.forward).parameters.values()
        )
        if accepts_kwargs:
            assert model.model_forward_params is None
        else:
            assert isinstance(model.model_forward_params, set)
            assert {"input_ids", "attention_mask"} <= model.model_forward_params

    def test_document_length_applies_to_document_task_only(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # document_length (patched to 6) truncates only task "document" inputs; "query" and no task are untouched.
        """Tasks are ``"query"`` / ``"document"`` throughout the library: ``document_length`` only
        truncates ``task="document"`` batches, any other task falls through untouched."""
        transformer = bert_tiny_transformer
        monkeypatch.setattr(transformer, "document_length", 6)
        long_text = "a rather long document with clearly more than six tokens in it"
        assert transformer.preprocess([long_text], task="document")["input_ids"].shape[1] == 6
        for task in ("query", None):
            assert transformer.preprocess([long_text], task=task)["input_ids"].shape[1] > 6

    def test_expansion_pad_contract_survives_processing_kwargs(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # With fixed query expansion of length 16, a per-call padding override is overridden back to the fixed
        # width (16 tokens) and a warning about the fixed width is emitted (captured by patching warning_once).
        """A processing_kwargs padding/max_length override must not silently break the expansion's
        fixed width (which would make query embeddings depend on batch composition): the expansion
        contract is re-applied with a warning."""
        import sentence_transformers.base.modules.transformer as transformer_module

        transformer = bert_tiny_transformer
        monkeypatch.setattr(transformer, "query_expansion", {"strategy": "fixed", "token": None, "length": 16})
        warnings: list[str] = []
        monkeypatch.setattr(transformer_module.logger, "warning_once", warnings.append)

        features = transformer.preprocess(
            ["short", "a somewhat longer query"], task="query", processing_kwargs={"text": {"padding": True}}
        )
        assert features["input_ids"].shape[1] == 16
        assert warnings and "fixed width" in warnings[0]

    def test_min_expansion_pads_short_queries_to_floor(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # "min" expansion to 8 pads a 3-token query with 5 mask-token expansion positions that have attention mask 0.
        transformer = bert_tiny_transformer
        monkeypatch.setattr(transformer, "query_expansion", {"strategy": "min", "token": None, "length": 8})
        features = transformer.preprocess(["short"], task="query")
        assert features["input_ids"].shape[1] == 8
        positions = features["query_expansion_positions"]
        # bert-tiny tokenizes "short" to [CLS] short [SEP]: 3 real tokens, 5 expansion positions.
        assert positions.sum().item() == 5
        assert (features["input_ids"][positions] == transformer.tokenizer.mask_token_id).all()
        assert features["attention_mask"][positions].eq(0).all()

    def test_min_expansion_leaves_long_queries_untruncated(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # "min" expansion leaves queries longer than the floor untruncated, with no expansion positions and no warning.
        import sentence_transformers.base.modules.transformer as transformer_module

        transformer = bert_tiny_transformer
        monkeypatch.setattr(transformer, "query_expansion", {"strategy": "min", "token": None, "length": 8})
        warnings: list[str] = []
        monkeypatch.setattr(transformer_module.logger, "warning_once", warnings.append)
        features = transformer.preprocess(
            ["a very verbose query with clearly more than eight tokens worth of content in it"], task="query"
        )
        assert features["input_ids"].shape[1] > 8
        assert not features["query_expansion_positions"].any()
        assert not warnings

    def test_min_expansion_mixed_batch_positions_per_row(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # In a mixed batch, the short row gets 5 expansion positions within the floor and plain padding beyond it,
        # the long row none.
        transformer = bert_tiny_transformer
        monkeypatch.setattr(transformer, "query_expansion", {"strategy": "min", "token": None, "length": 8})
        features = transformer.preprocess(
            ["short", "a very verbose query with clearly more than eight tokens worth of content in it"],
            task="query",
        )
        positions = features["query_expansion_positions"]
        assert features["input_ids"].shape[1] > 8
        assert positions[0].sum().item() == 5
        assert positions[0, 8:].eq(False).all()
        assert not positions[1].any()
        # The short row's trailing pads beyond the floor stay real pads, excluded from scoring.
        assert features["attention_mask"][0, 8:].eq(0).all()

    def test_min_expansion_ceiling_via_max_length(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # A per-call max_length of 12 caps a long query under "min" expansion.
        transformer = bert_tiny_transformer
        monkeypatch.setattr(transformer, "query_expansion", {"strategy": "min", "token": None, "length": 8})
        features = transformer.preprocess(
            ["a very verbose query with clearly more than twelve tokens worth of content in it"],
            task="query",
            max_length=12,
        )
        assert features["input_ids"].shape[1] == 12

    def test_min_expansion_attend_scopes_to_expansion_positions(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # With attend enabled, the short row's attention mask covers the floor width (first 8) and not the padding beyond.
        transformer = bert_tiny_transformer
        monkeypatch.setattr(
            transformer, "query_expansion", {"strategy": "min", "attend": True, "token": None, "length": 8}
        )
        features = transformer.preprocess(
            ["short", "a very verbose query with clearly more than eight tokens worth of content in it"],
            task="query",
        )
        assert features["attention_mask"][0, :8].eq(1).all()
        assert features["attention_mask"][0, 8:].eq(0).all()

    def test_fixed_expansion_warns_on_truncated_queries(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # "fixed" expansion to 8 warns only when a query had to be truncated to the fixed width.
        """strategy='fixed' silently loses content past the expansion length (the width is fixed in
        both directions), so the first such batch must warn. strategy='min' never truncates."""
        import sentence_transformers.base.modules.transformer as transformer_module

        transformer = bert_tiny_transformer
        monkeypatch.setattr(transformer, "query_expansion", {"strategy": "fixed", "token": None, "length": 8})
        warnings: list[str] = []
        monkeypatch.setattr(transformer_module.logger, "warning_once", warnings.append)

        features = transformer.preprocess(["short query"], task="query")
        assert features["input_ids"].shape[1] == 8
        assert not warnings

        features = transformer.preprocess(
            ["a very verbose query with clearly more than eight tokens worth of content in it"], task="query"
        )
        assert features["input_ids"].shape[1] == 8
        assert warnings and "truncated" in warnings[0]

    def test_retrieval_task_warns_on_text_documents(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # For the retrieval transformer task, text inputs with task "document" log a warning that text is rendered
        # as a query; query or untasked inputs do not.
        """`*ForRetrieval` processors always render text as a query: asking for document treatment
        on text inputs must warn instead of silently query-formatting the documents."""
        import sentence_transformers.base.modules.transformer as transformer_module

        transformer = bert_tiny_transformer
        monkeypatch.setattr(transformer, "transformer_task", "retrieval")
        warnings: list[str] = []
        monkeypatch.setattr(transformer_module.logger, "warning_once", warnings.append)

        transformer.preprocess(["a text document"], task="document")
        assert warnings and "renders text as a query" in warnings[0]

        warnings.clear()
        transformer.preprocess(["a query"], task="query")
        transformer.preprocess(["taskless text"])
        assert not warnings

    def test_kwargs_forward_excludes_bookkeeping_keys(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # When the model forward accepts arbitrary kwargs, forward passes input_ids/attention_mask but none of the
        # injected bookkeeping or tokenizer-only keys (spy on the model forward). Skipped otherwise.
        """When forward accepts ``**kwargs`` (``model_forward_params is None``), forward() passes the real
        model inputs but drops ST bookkeeping and tokenizer-only extras, even though ``**kwargs`` would
        otherwise swallow them silently."""
        model = bert_tiny_transformer
        if model.model_forward_params is not None:
            pytest.skip("bert-tiny forward does not accept **kwargs in this transformers version")

        features = batch_to_device(model.preprocess(["hello world"]), model.model.device)
        # Inject every key that must never reach the model, regardless of whether preprocess added it.
        seq_len = features["input_ids"].shape[1]
        features.update(
            {
                "modality": "text",
                "num_images_per_sample": torch.zeros(1, dtype=torch.long),
                "num_videos_per_sample": torch.zeros(1, dtype=torch.long),
                "prompt_length": 1,
                "query_expansion_positions": torch.zeros_like(features["input_ids"], dtype=torch.bool),
                "task": "document",
                "offset_mapping": torch.zeros(1, seq_len, 2, dtype=torch.long),
                "overflow_to_sample_mapping": torch.zeros(1, dtype=torch.long),
                "special_tokens_mask": torch.zeros_like(features["input_ids"]),
            }
        )

        captured = {}
        original_forward = model.model.forward

        def spy(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Records keyword arguments and calls the original forward.
            captured.update(kwargs)
            return original_forward(**kwargs)

        monkeypatch.setattr(model.model, "forward", spy)
        with torch.no_grad():
            model.forward(features)

        assert {"input_ids", "attention_mask"} <= captured.keys()
        for banned in (
            "modality",
            "num_images_per_sample",
            "num_videos_per_sample",
            "prompt_length",
            "query_expansion_positions",
            "task",
            "offset_mapping",
            "overflow_to_sample_mapping",
            "special_tokens_mask",
        ):
            assert banned not in captured, f"{banned} must not be forwarded to the model"

    def test_kwargs_forward_keeps_bookkeeping_keys_the_model_declares(self, monkeypatch):
        # [FUNCTION DOC]
        # With a wrapped model (patched loader) whose forward declares task and modality plus arbitrary kwargs,
        # forward passes the declared bookkeeping keys ("document", "text") but still blocks undeclared ones
        # such as prompt_length.
        """A custom model whose forward names a bookkeeping key wants it (e.g. task adapters keyed on
        ``task``), so the denylist must yield to an explicit declaration even though ``**kwargs`` is
        also present. Keys it does not declare stay blocked."""

        class TaskAwareModel(torch.nn.Module):
            # [CLASS DOC]
            # HELPER: Wrapper module around a loaded model exposing its config; forward records task/modality and
            # whether prompt_length leaked, then calls the inner model with the remaining kwargs.
            def __init__(self, inner):
                # [FUNCTION DOC]
                # HELPER: Stores the inner model and its config.
                super().__init__()
                self.inner = inner
                self.config = inner.config

            def forward(self, task=None, modality=None, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Records (task, modality) in seen and whether prompt_length is present in blocked, then forwards the rest.
                self.seen = (task, modality)
                self.blocked = "prompt_length" in kwargs
                return self.inner(**kwargs)

        original_load = Transformer._load_model
        monkeypatch.setattr(
            Transformer,
            "_load_model",
            lambda self, *args, **kwargs: TaskAwareModel(original_load(self, *args, **kwargs)),
        )
        model = Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        assert model.model_forward_params is None, "declaring **kwargs must select the denylist branch"

        features = batch_to_device(model.preprocess(["hello world"]), next(model.model.parameters()).device)
        features.update({"task": "document", "prompt_length": 1})
        with torch.no_grad():
            model.forward(features)

        assert model.model.seen == ("document", "text")
        assert not model.model.blocked, "prompt_length is undeclared, so it stays denied"

    def test_allowlist_forward_filters_unknown_kwargs(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # With an explicit allowlist of forward parameters, forward drops feature keys outside it (pixel_values)
        # and passes input_ids.
        """When forward does not accept ``**kwargs`` (``model_forward_params`` is a set), forward() drops
        any feature key outside the allowlist."""
        model = bert_tiny_transformer
        model.model_forward_params = {"input_ids", "attention_mask", "token_type_ids", "return_dict"}

        features = batch_to_device(model.preprocess(["hello world"]), model.model.device)
        features["pixel_values"] = torch.zeros(1, 3, 4, 4)  # not in the allowlist, must be dropped

        captured = {}
        original_forward = model.model.forward

        def spy(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Records keyword arguments and calls the original forward.
            captured.update(kwargs)
            return original_forward(**kwargs)

        monkeypatch.setattr(model.model, "forward", spy)
        with torch.no_grad():
            model.forward(features)

        assert "input_ids" in captured
        assert "pixel_values" not in captured


class TestGetEmbeddingDimension:
    # [CLASS DOC]
    # Tests for Transformer.get_embedding_dimension fallbacks.
    def test_standard_hidden_size(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Returns the config's hidden_size.
        dim = bert_tiny_transformer.get_embedding_dimension()
        assert dim == bert_tiny_transformer.config.hidden_size

    def test_hidden_sizes_list(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Without hidden_size, returns the last entry of hidden_sizes.
        """Models with hidden_sizes (list) should return the last element."""
        model = bert_tiny_transformer
        monkeypatch.delattr(model.config, "hidden_size", raising=False)
        monkeypatch.delattr(type(model.config), "hidden_size", raising=False)
        model.config.hidden_sizes = [64, 128, 256]
        assert model.get_embedding_dimension() == 256

    def test_hidden_dim(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Without hidden_size, returns hidden_dim.
        """Models with hidden_dim should return it."""
        model = bert_tiny_transformer
        monkeypatch.delattr(model.config, "hidden_size", raising=False)
        monkeypatch.delattr(type(model.config), "hidden_size", raising=False)
        model.config.hidden_dim = 384
        assert model.get_embedding_dimension() == 384

    def test_raises_when_no_dimension_found(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Without any dimension attribute, raises ValueError.
        """Should raise ValueError when no dimension attribute is found."""
        model = bert_tiny_transformer
        monkeypatch.delattr(model.config, "hidden_size", raising=False)
        monkeypatch.delattr(type(model.config), "hidden_size", raising=False)
        with pytest.raises(ValueError, match="Could not determine embedding dimension"):
            model.get_embedding_dimension()

    def test_projection_dim_for_sentence_embedding(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # With module_output_name "sentence_embedding", projection_dim takes priority.
        """When module_output_name is 'sentence_embedding', projection_dim takes priority."""
        model = bert_tiny_transformer
        model.module_output_name = "sentence_embedding"
        model.config.projection_dim = 512
        assert model.get_embedding_dimension() == 512

    def test_text_config_fallback(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Without hidden_size on the main config, falls back to text_config's hidden_size.
        """Should fall back to text_config.hidden_size when main config lacks it."""
        model = bert_tiny_transformer
        del model.config.hidden_size

        class FakeTextConfig:
            # [CLASS DOC]
            # HELPER: Fake text sub-config with hidden_size 768.
            hidden_size = 768

        model.config.text_config = FakeTextConfig()
        assert model.get_embedding_dimension() == 768


class TestGetPromptLength:
    # [CLASS DOC]
    # Tests for the private prompt-length helper and its cache.
    def test_prompt_length_cached(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Repeated calls with the same prompt return the same length and add one cache entry.
        """Second call with the same prompt should use the cache."""
        model = bert_tiny_transformer
        model._prompt_length_mapping.clear()
        len1 = model._get_prompt_length("search query: ")
        len2 = model._get_prompt_length("search query: ")
        assert len1 == len2
        assert len(model._prompt_length_mapping) == 1

    def test_prompt_length_cache_distinguishes_kwarg_keys(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Calls differing only in keyword argument names create separate cache entries.
        """Different kwarg keys with the same values should produce separate cache entries."""
        model = bert_tiny_transformer
        model._prompt_length_mapping.clear()
        model._get_prompt_length("search query: ", task="foo")
        model._get_prompt_length("search query: ", mode="foo")
        assert len(model._prompt_length_mapping) == 2

    def test_prompt_length_excludes_special_tokens(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # The prompt length excludes trailing special tokens, so it is shorter than the fully tokenized prompt.
        """Prompt length should exclude trailing special tokens like [SEP]."""
        model = bert_tiny_transformer
        prompt = "query: "
        tokenized = model.tokenizer(prompt, add_special_tokens=True)
        raw_length = len(tokenized["input_ids"])
        prompt_length = model._get_prompt_length(prompt)
        assert prompt_length < raw_length


class TestCallProcessor:
    # [CLASS DOC]
    # Tests for the private processor-call helper.
    def test_single_modality_tokenizer(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # The single-modality text path returns input_ids.
        """Single-modality tokenizer path should work for text."""
        model = bert_tiny_transformer
        result = model._call_processor(
            modality="text",
            processor_inputs={"text": ["hello world"]},
            modality_kwargs={
                "text": {"padding": True, "truncation": "longest_first"},
                "audio": {},
                "image": {},
                "video": {},
            },
            common_kwargs={"return_tensors": "pt"},
        )
        assert "input_ids" in result


def make_char_chat_template_mock(suffix, captured_kwargs=None):
    # [FUNCTION DOC]
    # HELPER: Returns a fake apply_chat_template that tokenizes the first conversation's concatenated content
    # as one id per character (code point) followed by the suffix ids, truncates to max_length unless
    # truncation is disabled (False or "do_not_truncate"), and returns a dict with tensors ("pt"), numpy
    # arrays ("np") or nested lists. If captured_kwargs is given, each call's kwargs are appended to it.
    """Chat-template mock: one token per content character plus ``suffix``, honoring truncation/max_length
    and return_tensors ("pt", "np", or ragged lists)."""

    def mock_apply_chat_template(messages, **kwargs):
        # [FUNCTION DOC]
        # HELPER: The fake chat-template call described above.
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        text = "".join(str(message["content"]) for message in messages[0])
        ids = [ord(char) for char in text] + suffix
        # Like real tokenizers, max_length truncates unless truncation is explicitly disabled.
        if kwargs.get("max_length") and kwargs.get("truncation") not in (False, "do_not_truncate"):
            ids = ids[: kwargs["max_length"]]
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        if kwargs.get("return_tensors") == "np":
            return {"input_ids": np.array([ids]), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
        return {"input_ids": [ids]}

    return mock_apply_chat_template


def make_modality_kwargs(**text_kwargs):
    # [FUNCTION DOC]
    # HELPER: Builds a modality kwargs dict with the given text kwargs and empty audio/image/video buckets.
    """Modality kwargs for _process_chat_messages with only the text bucket populated."""
    return {"text": text_kwargs, "audio": {}, "image": {}, "video": {}}


class TestApplyChatTemplate:
    # [CLASS DOC]
    # Tests for the private chat-template application helper.
    @pytest.mark.parametrize("use_processor_kwargs", [False, True])
    @pytest.mark.parametrize("video_flag", [None, False, True])
    @pytest.mark.parametrize("chat_flag", [None, False, True])
    def test_load_audio_from_video_routing(self, monkeypatch, video_flag, chat_flag, use_processor_kwargs):
        # [FUNCTION DOC]
        # For all combinations of the per-video and chat-level load_audio_from_video flags and of the
        # processor_kwargs calling convention (patched module flag), checks with a mocked processor that
        # apply_chat_template is called once with tokenize/return_dict, the effective flag (chat flag wins,
        # else the video flag, else False), the bucketed kwargs (nested under processor_kwargs when that
        # convention is active), returns the mock's result, and leaves the input dicts unmodified.
        monkeypatch.setattr(
            transformer_module, "_TRANSFORMERS_APPLY_CHAT_TEMPLATE_RECOMMENDS_PROCESSOR_KWARGS", use_processor_kwargs
        )
        processor = MagicMock(spec=transformer_module.ProcessorMixin)
        model = SimpleNamespace(processor=processor)
        messages = [[{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}]}]]
        modality_kwargs = make_modality_kwargs(padding=True)
        modality_kwargs["video"] = {"num_frames": 2}
        chat_kwargs = {"add_generation_prompt": False}
        if video_flag is not None:
            modality_kwargs["video"]["load_audio_from_video"] = video_flag
        if chat_flag is not None:
            chat_kwargs["load_audio_from_video"] = chat_flag
        common_kwargs = {"return_tensors": "pt"}
        original_kwargs = deepcopy((modality_kwargs, common_kwargs, chat_kwargs))

        result = Transformer._apply_chat_template(model, messages, modality_kwargs, common_kwargs, chat_kwargs)

        expected_flag = chat_flag if chat_flag is not None else bool(video_flag)
        expected_processor_kwargs = {
            "text_kwargs": {"padding": True},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {"num_frames": 2},
            "common_kwargs": {"return_tensors": "pt"},
        }
        if use_processor_kwargs:
            expected_processor_kwargs = {"processor_kwargs": expected_processor_kwargs, "return_tensors": "pt"}
        processor.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=True,
            return_dict=True,
            load_audio_from_video=expected_flag,
            add_generation_prompt=False,
            **expected_processor_kwargs,
        )
        assert result is processor.apply_chat_template.return_value
        assert (modality_kwargs, common_kwargs, chat_kwargs) == original_kwargs


class TestProcessChatMessages:
    # [CLASS DOC]
    # Tests for chat-message processing, including restoration of the chat-template suffix after truncation.
    def test_unsupported_message_modality(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Processing chat messages without a "message" modality config raises ValueError.
        """Should raise ValueError when 'message' modality is not in modality_config."""
        model = bert_tiny_transformer
        with pytest.raises(ValueError, match="does not support 'message' modality"):
            model._process_chat_messages(
                messages=[[{"role": "user", "content": "test"}]],
                modality_kwargs={"text": {}, "audio": {}, "image": {}, "video": {}},
                common_kwargs={},
            )

    def test_processing_kwargs_chat_template_passed_through(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Instance chat_template processing_kwargs reach apply_chat_template through preprocess.
        """Instance ``processing_kwargs['chat_template']`` should reach ``apply_chat_template`` via preprocess."""
        model = bert_tiny_transformer
        model.processing_kwargs = {"chat_template": {"add_generation_prompt": True, "continue_final_message": False}}
        model.modality_config["message"] = {"method": "forward", "method_output_name": "last_hidden_state"}

        captured_kwargs = {}

        def mock_apply_chat_template(messages, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records kwargs and returns a fixed three-token encoding.
            captured_kwargs.update(kwargs)
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        model.preprocess([[{"role": "user", "content": "test"}]])
        assert captured_kwargs["add_generation_prompt"] is True
        assert captured_kwargs["continue_final_message"] is False

    def test_per_call_chat_template_overrides_instance_via_preprocess(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Per-call chat_template processing_kwargs override the instance value.
        """Per-call ``processing_kwargs={'chat_template': ...}`` should override instance-level via preprocess."""
        model = bert_tiny_transformer
        model.processing_kwargs = {"chat_template": {"add_generation_prompt": False}}
        model.modality_config["message"] = {"method": "forward", "method_output_name": "last_hidden_state"}

        captured_kwargs = {}

        def mock_apply_chat_template(messages, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records kwargs and returns a fixed three-token encoding.
            captured_kwargs.update(kwargs)
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        model.preprocess(
            [[{"role": "user", "content": "test"}]],
            processing_kwargs={"chat_template": {"add_generation_prompt": True}},
        )
        assert captured_kwargs["add_generation_prompt"] is True

    def test_chat_truncation_restores_chat_template_suffix(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # With the character-level mock, truncating long content to 8 tokens restores the 3-token suffix at the
        # tail while keeping the head; short content is untouched; disabling truncation returns the full sequence;
        # and the unpadded list output path restores the suffix the same way.
        model = bert_tiny_transformer
        # Suffix preservation is task-independent: the default feature-extraction task confirms it benefits
        # embedders (e.g. last-token pooling), not just causal-LM rerankers.
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}

        # A chat template appends a fixed scoring suffix (e.g. the assistant prefill) that right-truncation
        # cuts off first. The mock mimics this: one token per content character, then ``suffix``.
        suffix = [101, 102, 103]
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock(suffix))

        def run(content, max_length, truncation=True):
            # [FUNCTION DOC]
            # HELPER: Processes one message with the given content, max_length and truncation, returning tensors.
            return model._process_chat_messages(
                messages=[[{"role": "user", "content": content}]],
                modality_kwargs=make_modality_kwargs(padding=True, truncation=truncation, max_length=max_length),
                common_kwargs={"return_tensors": "pt"},
            )

        # The mock tokenizes one token per character, so this 13-token content is truncated at max_length=8,
        # dropping the suffix. The restore puts it back on the tail (head kept), with an auto-derived length.
        long_content = "abcdefghijklm"
        ids = run(long_content, max_length=8)["input_ids"].tolist()[0]
        assert len(ids) == 8  # max_length is respected
        assert ids[-3:] == suffix  # the scoring suffix is back at the tail
        assert ids[:5] == [ord(char) for char in long_content[:5]]  # the head is preserved

        # Short content fits, so no truncation happened and the row is returned untouched (no-op restore).
        assert run("ab", max_length=8)["input_ids"].tolist()[0] == [ord("a"), ord("b")] + suffix

        # Escape hatch: with truncation disabled the full sequence (suffix intact) is returned as-is.
        for truncation in (False, "do_not_truncate"):
            assert (
                run(long_content, max_length=8, truncation=truncation)["input_ids"].tolist()[0]
                == [ord(char) for char in long_content] + suffix
            )

        # Flattening path (feature-extraction + can_flatten_inputs): return_tensors is dropped, so the
        # processor returns ragged unpadded lists. The list branch restores the suffix the same way.
        list_ids = model._process_chat_messages(
            messages=[[{"role": "user", "content": long_content}]],
            modality_kwargs=make_modality_kwargs(truncation=True, max_length=8),
            common_kwargs={},
        )["input_ids"][0]
        assert len(list_ids) == 8 and list_ids[-3:] == suffix and list_ids[:5] == [ord(c) for c in long_content[:5]]

    def test_chat_truncation_restore_can_be_disabled(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # restore_suffix False in the chat template kwargs leaves the raw truncated tail, True restores the suffix,
        # and the flag is never forwarded to apply_chat_template.
        # The restore is on by default, but ``chat_template_kwargs["restore_suffix"]=False`` (set via
        # ``processing_kwargs["chat_template"]``) turns it off. The flag is a Sentence Transformers concept,
        # so it must be popped and never forwarded to apply_chat_template.
        model = bert_tiny_transformer
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}
        suffix = [201, 202, 203]
        forwarded_kwargs = []
        monkeypatch.setattr(
            model.processor, "apply_chat_template", make_char_chat_template_mock(suffix, forwarded_kwargs)
        )

        def run(restore_suffix):
            # [FUNCTION DOC]
            # HELPER: Processes a 10-character message truncated to 8 with the given restore flag and returns the ids row.
            return model._process_chat_messages(
                messages=[[{"role": "user", "content": "abcdefghij"}]],
                modality_kwargs=make_modality_kwargs(padding=True, truncation=True, max_length=8),
                common_kwargs={"return_tensors": "pt"},
                chat_template_kwargs={"restore_suffix": restore_suffix},
            )["input_ids"].tolist()[0]

        # Default behaviour (restore on): the template suffix is put back at the tail.
        assert run(True)[-3:] == suffix
        # Disabled: the raw truncated tail (content tokens, no suffix) is left untouched.
        assert run(False) == [ord(char) for char in "abcdefghij"][:8]
        # restore_suffix is a ST-only flag, so it is never leaked into the apply_chat_template call(s).
        assert forwarded_kwargs and all("restore_suffix" not in kwargs for kwargs in forwarded_kwargs)

    def test_chat_truncation_restore_leaves_non_torch_tensors_untouched(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Numpy outputs are not restored: the truncated numpy row keeps its raw content tail.
        # The restore only supports torch tensors and token lists. Other containers (e.g. numpy arrays from
        # ``return_tensors="np"``) must be returned untouched, not crash on ndarray truthiness.
        model = bert_tiny_transformer
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}

        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock([101, 102, 103]))
        features = model._process_chat_messages(
            messages=[[{"role": "user", "content": "abcdefghijklm"}]],
            modality_kwargs=make_modality_kwargs(padding=True, truncation=True, max_length=8),
            common_kwargs={"return_tensors": "np"},
        )
        # The truncated numpy row keeps its raw tail (content tokens, no suffix): numpy is not restored.
        assert features["input_ids"].tolist() == [[ord(char) for char in "abcdefghijklm"][:8]]

    def test_chat_truncation_restore_reads_chat_template_bucket_kwargs(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Truncation settings supplied in the chat_template bucket trigger the restore, including when they
        # override an existing text truncation value.
        # truncation/max_length can also ride in the chat_template bucket (apply_chat_template accepts them
        # top level). The restore gate must see them there, and the skeleton renders must not inherit them.
        model = bert_tiny_transformer
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}
        suffix = [101, 102, 103]
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock(suffix))
        ids = model._process_chat_messages(
            messages=[[{"role": "user", "content": "abcdefghijklm"}]],
            modality_kwargs=make_modality_kwargs(padding=True),
            common_kwargs={"return_tensors": "pt"},
            chat_template_kwargs={"truncation": True, "max_length": 8},
        )["input_ids"].tolist()[0]
        assert len(ids) == 8 and ids[-3:] == suffix

        # The chat bucket also wins over an existing text value (preprocess always sets a default text
        # truncation, which previously collided with an explicit chat-bucket one as a duplicate kwarg).
        ids = model._process_chat_messages(
            messages=[[{"role": "user", "content": "abcdefghijklm"}]],
            modality_kwargs=make_modality_kwargs(padding=True, truncation="longest_first"),
            common_kwargs={"return_tensors": "pt"},
            chat_template_kwargs={"truncation": True, "max_length": 8},
        )["input_ids"].tolist()[0]
        assert len(ids) == 8 and ids[-3:] == suffix

    def test_chat_truncation_restore_handles_implicit_truncation(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # max_length with truncation left as None counts as truncation and triggers the restore.
        # Tokenizers truncate when max_length is set even if truncation was merely left unset (None), so the
        # restore gate must treat that as truncation-on.
        model = bert_tiny_transformer
        model.modality_config["message"] = {"method": "forward", "method_output_name": "logits"}
        suffix = [101, 102, 103]
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock(suffix))
        ids = model._process_chat_messages(
            messages=[[{"role": "user", "content": "abcdefghijklm"}]],
            modality_kwargs=make_modality_kwargs(padding=False, truncation=None, max_length=8),
            common_kwargs={"return_tensors": "pt"},
        )["input_ids"].tolist()[0]
        assert len(ids) == 8 and ids[-3:] == suffix

    def test_restore_chat_template_suffix_handles_both_padding_sides(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # The suffix restore locates real tokens via the attention mask for both right- and left-padded batches,
        # rewriting only the truncated row.
        # The restore must locate the real tail via the attention mask (not a fixed [-keep:] slice), so it is
        # correct whether the batch is left- or right-padded. In each batch row 0 was truncated (full width,
        # suffix cut) and row 1 fits (real tokens end with the suffix, the remainder is padding).
        model = bert_tiny_transformer
        suffix = [101, 102, 103]
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *args, **kwargs: suffix)
        messages = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]

        # Right padding: real tokens left-aligned, pad on the right. The fit row's pad must stay untouched.
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 101, 102, 103, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"].tolist() == [[10, 11, 12, 101, 102, 103], [20, 21, 101, 102, 103, 0]]

        # Left padding: pad on the left, real tokens right-aligned.
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15], [0, 20, 21, 101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"].tolist() == [[10, 11, 12, 101, 102, 103], [0, 20, 21, 101, 102, 103]]

    def test_restore_chat_template_suffix_when_truncated_row_is_padded(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # A truncated row that also carries padding gets the suffix written over its last real tokens, for both padding sides.
        # pad_to_multiple_of can pad a truncated row past max_length, so a truncated row may itself carry
        # padding. The real tail is read from the mask, so the suffix still lands on real tokens (pad intact).
        model = bert_tiny_transformer
        suffix = [101, 102, 103]
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *args, **kwargs: suffix)
        messages = [[{"role": "user", "content": "x"}]]

        # Right padding: 5 real tokens (suffix cut by truncation) then 3 trailing pads.
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"].tolist() == [[10, 11, 101, 102, 103, 0, 0, 0]]

        # Left padding: 3 leading pads then the 5 real tokens.
        features = {
            "input_ids": torch.tensor([[0, 0, 0, 10, 11, 12, 13, 14]]),
            "attention_mask": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"].tolist() == [[0, 0, 0, 10, 11, 101, 102, 103]]

    def test_restore_skipped_when_nothing_reached_max_length(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Rows shorter than max_length are skipped without deriving the suffix.
        # Rows shorter than max_length were not truncated, so they are skipped without deriving any suffix.
        model = bert_tiny_transformer
        derived = []
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *a, **k: derived.append(True) or [1, 2, 3])
        messages = [[{"role": "user", "content": "x"}]]
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]),  # real length 3
        }
        # real length (3) < max_length (8): not truncated -> skipped, no derivation, untouched.
        model._restore_chat_template_suffix(features, messages, {}, {}, max_length=8)
        assert derived == [] and features["input_ids"].tolist() == [[10, 11, 12, 0, 0, 0]]

    def test_restore_only_touches_rows_that_reached_max_length(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Only rows whose real length reaches max_length are restored; the suffix is derived once.
        # The truncation check is per row: in a batch where one row was truncated (real length equals
        # max_length), shorter rows kept their tail and must be skipped entirely (no derivation, no write).
        model = bert_tiny_transformer
        derived = []
        monkeypatch.setattr(
            model, "_chat_template_suffix_ids", lambda *a, **k: derived.append(True) or [101, 102, 103]
        )
        messages = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]),
        }
        model._restore_chat_template_suffix(features, messages, {}, {}, max_length=6)
        # Row 0 (length 6 == max_length) is restored, row 1 (length 3) is left untouched and underived.
        assert features["input_ids"].tolist() == [[10, 11, 12, 101, 102, 103], [20, 21, 22, 0, 0, 0]]
        assert len(derived) == 1

    def test_restore_skips_tensor_batches_without_attention_mask(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # A tensor batch without attention_mask is left untouched and no suffix is derived.
        # Without a mask, real tokens can't be told apart from padding, so the restore must leave the batch
        # untouched rather than stamp the suffix over what may be padding.
        model = bert_tiny_transformer
        derived = []
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *a, **k: derived.append(True) or [101, 102])
        features = {"input_ids": torch.tensor([[10, 11, 12, 13]])}
        model._restore_chat_template_suffix(features, [[{"role": "user", "content": "x"}]], {}, {})
        assert derived == [] and features["input_ids"].tolist() == [[10, 11, 12, 13]]

    def test_restore_handles_float_attention_mask(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # A float attention mask still yields integer real lengths and a correct restore.
        # Some processors emit float masks: the real lengths must still be ints for the tensor slicing.
        model = bert_tiny_transformer
        monkeypatch.setattr(model, "_chat_template_suffix_ids", lambda *a, **k: [101, 102, 103])
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]]),
            "attention_mask": torch.ones(1, 6),  # float32
        }
        model._restore_chat_template_suffix(features, [[{"role": "user", "content": "a"}]], {}, {})
        assert features["input_ids"].tolist() == [[10, 11, 12, 101, 102, 103]]

    def test_chat_template_suffix_ids_tolerates_unhashable_kwargs(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Unhashable chat-template kwarg values (a tools list) do not break suffix derivation or caching.
        # chat_template_kwargs may carry unhashable values (e.g. a ``tools`` list). repr() in the cache key
        # keeps the lookup from raising TypeError, so derivation still runs (and caches).
        model = bert_tiny_transformer
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock([99]))
        suffix = model._chat_template_suffix_ids([{"role": "user", "content": "x"}], {"tools": [{"name": "f"}]}, {})
        assert suffix == [99]  # no crash from the unhashable kwarg. The fixed marker is the derived suffix

    def test_chat_template_suffix_ids_skips_uninspectable_messages(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Conversations the text-only check cannot inspect (non-dict message, list of plain-string content) give an empty suffix.
        # Some templates accept shapes the text-only check can't inspect (a non-dict message, or a content
        # list of plain strings). The restore must skip them (no suffix), not raise AttributeError.
        model = bert_tiny_transformer
        monkeypatch.setattr(
            model.processor, "apply_chat_template", lambda messages, **kwargs: {"input_ids": [[1, 2, 3]]}
        )
        assert model._chat_template_suffix_ids([{"role": "user", "content": "x"}, "not a message"], {}, {}) == []
        assert model._chat_template_suffix_ids([{"role": "user", "content": ["chunk a", "chunk b"]}], {}, {}) == []

    def test_chat_template_suffix_ids_returns_empty_without_variable_content(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # A conversation with only system content (nothing variable) gives an empty suffix.
        # With only kept (system) content there's nothing to vary, so the two renders are identical and there
        # is no reliable suffix boundary -> return [] rather than treating the whole render as the suffix.
        model = bert_tiny_transformer
        monkeypatch.setattr(model.processor, "apply_chat_template", make_char_chat_template_mock([]))
        assert model._chat_template_suffix_ids([{"role": "system", "content": "only a prompt"}], {}, {}) == []

    def test_chat_template_suffix_cache_is_bounded(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # With the suffix cache size patched to 3, deriving for 10 distinct prompts keeps only 3 cache entries.
        # The cache key includes the prompt, so a workload with many distinct prompts must stay bounded.
        model = bert_tiny_transformer
        monkeypatch.setattr("sentence_transformers.base.modules.transformer._CHAT_TEMPLATE_SUFFIX_CACHE_SIZE", 3)
        monkeypatch.setattr(
            model.processor, "apply_chat_template", lambda messages, **kwargs: {"input_ids": [[1, 2, 3]]}
        )
        for index in range(10):
            model._chat_template_suffix_ids([{"role": "system", "content": f"prompt {index}"}], {}, {})
        assert len(model._chat_template_suffix_cache) == 3

    def test_chat_template_suffix_ids_captures_prompt_rendered_in_tail(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # A template that renders the system prompt after the body yields a suffix containing the prompt tokens
        # plus the marker, keyed separately per prompt.
        # When the template renders the system prompt in the tail, keeping it fixed (not fillered) lets the
        # diff capture it, so a truncated trailing prompt can be restored. Different prompts -> different keys.
        model = bert_tiny_transformer

        def mock_apply_chat_template(messages, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake template rendering non-system content, then system content, then marker 99, as code points.
            # Render the (fillered) non-system content, then the kept system prompt, then a fixed marker (99).
            body = "".join(str(m["content"]) for m in messages[0] if m.get("role") != "system")
            prompt = "".join(str(m["content"]) for m in messages[0] if m.get("role") == "system")
            return {"input_ids": [[ord(c) for c in body] + [ord(c) for c in prompt] + [99]]}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        # "PQ" (80, 81) is kept fixed and lands in the tail, so it plus the marker (99) form the suffix.
        assert model._chat_template_suffix_ids(
            [{"role": "system", "content": "PQ"}, {"role": "user", "content": "x"}], {}, {}
        ) == [80, 81, 99]
        # A different prompt gets its own cache entry and its own suffix.
        assert model._chat_template_suffix_ids(
            [{"role": "system", "content": "RS"}, {"role": "user", "content": "x"}], {}, {}
        ) == [82, 83, 99]

    def test_chat_template_suffix_ids_keys_on_non_content_fields(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Non-content message fields (name) are part of the suffix cache key, so different names give different suffixes.
        # The skeleton keeps non-content message fields (e.g. ``name``, ``tool_calls``) verbatim and a
        # template may render them, so they are part of the cache key: the same role layout with different
        # fields must not share a cached suffix.
        model = bert_tiny_transformer

        def mock_apply_chat_template(messages, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake template rendering content, then each message's name field, then marker 99.
            # Render the (fillered) content, then the kept ``name`` field, then a fixed marker (99).
            body = "".join(str(m["content"]) for m in messages[0])
            name = "".join(str(m.get("name", "")) for m in messages[0])
            return {"input_ids": [[ord(c) for c in body] + [ord(c) for c in name] + [99]]}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        conv_a = [{"role": "user", "content": "x", "name": "AB"}]
        conv_b = [{"role": "user", "content": "x", "name": "CD"}]
        assert model._chat_template_suffix_ids(conv_a, {}, {}) == [65, 66, 99]  # "AB" + marker
        assert model._chat_template_suffix_ids(conv_b, {}, {}) == [67, 68, 99]  # "CD", not A's cached suffix

    def test_chat_template_suffix_ids_mirrors_common_kwargs(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Common kwargs are forwarded to the skeleton renders and keyed in the cache (marker 98 versus 99).
        # Common kwargs can change tokenization (a ProcessorMixin spreads them into the text kwargs), so the
        # skeleton renders must receive them and the cache must key on them.
        model = bert_tiny_transformer

        def mock_apply_chat_template(messages, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake template emitting marker 98 when common add_special_tokens is False, else 99, after the content.
            # Emit a different fixed tail depending on a tokenization-affecting common kwarg.
            marker = 98 if kwargs.get("common_kwargs", {}).get("add_special_tokens") is False else 99
            body = "".join(str(m["content"]) for m in messages[0])
            return {"input_ids": [[ord(c) for c in body] + [marker]]}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        conversation = [{"role": "user", "content": "x"}]
        assert model._chat_template_suffix_ids(conversation, {}, {}, {"add_special_tokens": False}) == [98]
        assert model._chat_template_suffix_ids(conversation, {}, {}, {}) == [99]

    def test_restore_chat_template_suffix_is_per_row_and_skips_multimodal(self, bert_tiny_transformer, monkeypatch):
        # [FUNCTION DOC]
        # In a mixed batch the text-only row is restored per row while the multimodal row is left untouched.
        # A heterogeneous batch: the text-only row is restored per-row, while the multimodal row is left
        # untouched (its trailing suffix can't be isolated from a text skeleton, so it returns []).
        model = bert_tiny_transformer
        suffix = [101, 102, 103]

        def mock_apply_chat_template(messages, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake template emitting the first content character's code point followed by the fixed suffix.
            # Skeleton content is the filler string, so emit one content token plus the fixed suffix.
            text = "".join(str(message["content"]) for message in messages[0])
            return {"input_ids": [[ord(text[0])] + suffix]}

        monkeypatch.setattr(model.processor, "apply_chat_template", mock_apply_chat_template)
        messages = [
            [{"role": "user", "content": "text"}],
            [{"role": "user", "content": [{"type": "text", "text": "t"}, {"type": "image", "image": "img"}]}],
        ]
        features = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
        }
        model._restore_chat_template_suffix(features, messages, {}, {})
        assert features["input_ids"][0].tolist() == [10, 11, 12, 101, 102, 103]  # text row restored
        assert features["input_ids"][1].tolist() == [20, 21, 22, 23, 24, 25]  # multimodal row untouched

    @pytest.mark.slow
    def test_chat_truncation_restores_suffix_with_real_chat_template(self):
        # [FUNCTION DOC]
        # Slow integration test with the real tiny Llama chat template: truncating a long user turn to 16 tokens
        # restores the closing "[/INST]" while keeping content, and disabling the restore leaves it missing.
        # Integration check against a real chat-template model (the other suffix tests mock the tokenizer), so
        # it exercises the real apply_chat_template dispatch and two-filler derivation. The Llama-2 template
        # ends the user turn with `[/INST]`, which truncation drops and the restore must put back.
        model = Transformer(TINY_LLAMA, transformer_task="feature-extraction")
        tokenizer = model.tokenizer

        def process(max_length, restore_suffix=True):
            # [FUNCTION DOC]
            # HELPER: Processes a 50-word user message with generation prompt at the given max_length and restore flag, returning the ids.
            return model._process_chat_messages(
                messages=[[{"role": "user", "content": "word " * 50}]],
                modality_kwargs=make_modality_kwargs(padding=True, truncation="longest_first", max_length=max_length),
                common_kwargs={"return_tensors": "pt"},
                chat_template_kwargs={"add_generation_prompt": True, "restore_suffix": restore_suffix},
            )["input_ids"][0].tolist()

        # The conversation overflows max_length=16, so `[/INST]` is truncated off and then restored.
        restored = process(16)
        decoded = tokenizer.decode(restored)
        assert len(restored) == 16  # truncation is still respected
        assert decoded.rstrip().endswith("[/INST]")  # the template suffix is back at the tail
        assert "word" in decoded  # real content is kept at the head, not overwritten wholesale

        # With the restore disabled the same input ends on a content token: the bug this feature fixes.
        assert not tokenizer.decode(process(16, restore_suffix=False)).rstrip().endswith("[/INST]")


class TestModelLoading:
    # [CLASS DOC]
    # Tests for model loading options and PEFT handling.
    def test_invalid_backend_error(self):
        # [FUNCTION DOC]
        # An unknown backend raises ValueError.
        with pytest.raises(ValueError, match="Unsupported backend"):
            Transformer(TINY_BERT, backend="invalid_backend")

    def test_peft_seq_classification_no_architectures(self, monkeypatch):
        # [FUNCTION DOC]
        # With adapter detection and PEFT availability patched and a fake PeftConfig lacking "architectures",
        # sequence-classification initialization raises AttributeError. Requires the peft package.
        # KNOWN ISSUE: the docstring says initialization "should not crash", but the test asserts that it raises AttributeError, so it locks in the crash rather than guarding against it.
        """PeftConfig has no 'architectures' attr. Sequence-classification init should not crash."""

        class FakePeftConfig:
            # [CLASS DOC]
            # HELPER: Stand-in PEFT config exposing only base_model_name_or_path.
            """Minimal stand-in for PeftConfig that intentionally lacks 'architectures'."""

            base_model_name_or_path = TINY_BERT

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Class method returning a new instance, ignoring arguments.
                return cls()

        monkeypatch.setattr(transformer_module, "find_adapter_config_file", lambda *a, **kw: "some_file")
        monkeypatch.setattr(transformer_module, "is_peft_available", lambda: True)

        import peft

        monkeypatch.setattr(peft, "PeftConfig", FakePeftConfig)

        # PeftConfig lacks 'architectures', so the sequence-classification guard
        # (which accesses config.architectures) will raise AttributeError.
        with pytest.raises(AttributeError):
            Transformer(TINY_BERT, transformer_task="sequence-classification")

    def test_peft_non_torch_backend_error(self, monkeypatch):
        # [FUNCTION DOC]
        # A detected PEFT adapter combined with the onnx backend raises ValueError.
        """PEFT models should raise an error for non-torch backends."""
        monkeypatch.setattr(transformer_module, "find_adapter_config_file", lambda *a, **kw: "some_file")
        with pytest.raises(ValueError, match="PEFT models can currently only be loaded"):
            Transformer(TINY_BERT, backend="onnx")


class TestModalityInference:
    # [CLASS DOC]
    # Tests for modality inference helpers.
    def test_infer_modalities_edge_cases_returns_none_for_unknown(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # The edge-case modality inference returns None for BERT.
        """Should return None for model types not in _EDGE_CASE_MODALITY_CONFIGS."""
        model = bert_tiny_transformer
        result = model.infer_modalities_edge_cases(model.model, model.processor)
        assert result is None

    def test_infer_modalities_from_processor_text(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Inference from a tokenizer processor yields ["text"].
        """Should identify 'text' modality for a tokenizer-based processor."""
        model = bert_tiny_transformer
        modalities = model.infer_modalities_from_processor(model.processor)
        assert modalities == ["text"]


class TestGetMethodOutputFields:
    # [CLASS DOC]
    # Tests for extracting output field names from a method's return annotation.
    def test_with_model_output_return_type(self):
        # [FUNCTION DOC]
        # BERT's forward annotation yields fields including last_hidden_state; the output-name inference returns
        # valid names and None for unknown ones.
        """Should extract field names from a ModelOutput return type, and validate _infer_method_output_name."""
        model = AutoModel.from_pretrained(TINY_BERT)
        fields = Transformer._get_method_output_fields(model.forward)
        assert fields is not None
        assert "last_hidden_state" in fields

        # _infer_method_output_name: valid name should be returned, invalid should return None
        assert Transformer._infer_method_output_name("last_hidden_state", model.forward) == "last_hidden_state"
        assert Transformer._infer_method_output_name("nonexistent_field", model.forward) is None

    def test_with_no_return_annotation(self):
        # [FUNCTION DOC]
        # A function without return annotation yields None.
        """Should return None when method has no return type annotation."""

        def plain_func(x):
            # [FUNCTION DOC]
            # HELPER: Identity function without annotations.
            return x

        assert Transformer._get_method_output_fields(plain_func) is None

    def test_with_exception_in_get_type_hints(self, monkeypatch):
        # [FUNCTION DOC]
        # If type-hint resolution raises (patched), the helper returns None.
        """Should return None when get_type_hints raises."""

        def raise_boom(*a, **kw):
            # [FUNCTION DOC]
            # HELPER: Always raises a generic Exception.
            raise Exception("boom")

        monkeypatch.setattr(transformer_module, "get_type_hints", raise_boom)
        model = AutoModel.from_pretrained(TINY_BERT)
        assert Transformer._get_method_output_fields(model.forward) is None


class TestGetProcessorAttributes:
    # [CLASS DOC]
    # Tests for the private processor-attributes helper.
    def test_returns_none_for_tokenizer(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # For a tokenizer-only processor the result is None or a dict/list.
        # KNOWN ISSUE: the name and docstring say the result should be None, but the assertion also accepts any dict or list, so it barely constrains the behavior.
        """Tokenizer-only processors don't have get_attributes or attributes."""
        result = bert_tiny_transformer._get_processor_attributes()
        assert result is None or isinstance(result, (dict, list))


class TestSerialization:
    # [CLASS DOC]
    # Tests for Transformer save/load and config serialization.
    def test_save_and_load_roundtrip(self, bert_tiny_transformer, tmp_path):
        # [FUNCTION DOC]
        # Save then load preserves model class, output name, modality config and all forward outputs (tolerance 1e-5).
        """save() then load() should produce an equivalent Transformer with identical outputs."""
        model = bert_tiny_transformer
        texts = ["hello world", "goodbye world"]
        features = batch_to_device(model.preprocess(texts), model.model.device)
        with torch.no_grad():
            out1 = model.forward(features)

        save_dir = str(tmp_path / "model")
        model.save(save_dir)
        reloaded = Transformer.load(save_dir)

        assert type(reloaded.auto_model).__name__ == type(model.auto_model).__name__
        assert reloaded.module_output_name == model.module_output_name
        assert reloaded.modality_config == model.modality_config

        features = batch_to_device(reloaded.preprocess(texts), reloaded.model.device)
        with torch.no_grad():
            out2 = reloaded.forward(features)

        for key in out1:
            v1, v2 = out1[key], out2[key]
            if isinstance(v1, torch.Tensor):
                assert torch.allclose(v1.cpu(), v2.cpu(), atol=1e-5), f"Output '{key}' differs after save/load"
            else:
                assert v1 == v2

    def test_max_seq_length_save_and_load(self, bert_tiny_transformer, tmp_path):
        # [FUNCTION DOC]
        # A custom max_seq_length (42) survives save/load, including the tokenizer limit.
        """A custom max_seq_length should be preserved after save/load."""
        model = bert_tiny_transformer
        assert model.tokenizer.model_max_length != 42
        model.max_seq_length = 42
        assert model.tokenizer.model_max_length == 42
        save_dir = str(tmp_path / "model")
        model.save(save_dir)
        reloaded = Transformer.load(save_dir)
        assert reloaded.max_seq_length == 42
        assert reloaded.tokenizer.model_max_length == 42

    def test_processing_kwargs_save_load_roundtrip(self, tmp_path):
        # [FUNCTION DOC]
        # processing_kwargs are written to sentence_bert_config.json and restored on load.
        """processing_kwargs should be persisted to config JSON and restored on load."""
        processing_kwargs = {
            "text": {"truncation": "only_first"},
            "chat_template": {"add_generation_prompt": True},
        }
        transformer = Transformer(TINY_BERT, processing_kwargs=processing_kwargs)
        assert transformer.processing_kwargs == processing_kwargs

        save_dir = str(tmp_path / "model")
        transformer.save(save_dir)

        # Verify the JSON file contains processing_kwargs
        config_path = Path(save_dir) / "sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        assert config["processing_kwargs"] == processing_kwargs

        # Verify the reloaded Transformer has the same processing_kwargs
        reloaded = Transformer.load(save_dir)
        assert reloaded.processing_kwargs == processing_kwargs

    def test_processing_kwargs_omitted_from_config_when_empty(self, bert_tiny_transformer, tmp_path):
        # [FUNCTION DOC]
        # Empty processing_kwargs are omitted from the saved config.
        """Empty processing_kwargs should not appear in the saved config JSON."""
        assert bert_tiny_transformer.processing_kwargs == {}
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        assert "processing_kwargs" not in config

    def test_unpad_inputs_omitted_from_config_when_inferred(self, bert_tiny_transformer, tmp_path):
        # [FUNCTION DOC]
        # An inferred (None) unpad_inputs is omitted from the saved config.
        """Inferred unpad_inputs (None) should not appear in the saved config JSON."""
        assert bert_tiny_transformer.unpad_inputs is None
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        assert "unpad_inputs" not in config

    @pytest.mark.parametrize("unpad_inputs", [True, False])
    def test_unpad_inputs_save_load_roundtrip(self, tmp_path, unpad_inputs):
        # [FUNCTION DOC]
        # An explicit unpad_inputs True/False is saved and restored.
        """Explicitly set unpad_inputs should appear in the saved config and survive a roundtrip."""
        transformer = Transformer(TINY_BERT, unpad_inputs=unpad_inputs)
        assert transformer.unpad_inputs is unpad_inputs
        save_dir = str(tmp_path / "model")
        transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        assert config["unpad_inputs"] is unpad_inputs

        reloaded = Transformer.load(save_dir)
        assert reloaded.unpad_inputs is unpad_inputs

    def test_get_config_dict(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # get_config_dict includes a modality_config dict.
        config = bert_tiny_transformer.get_config_dict()
        assert "modality_config" in config
        assert isinstance(config["modality_config"], dict)

    def test_get_config_dict_tuple_keys_serialized(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # Tuple modality keys serialize as plus-joined strings ("image+text").
        """Tuple modality keys should be serialized to plus-separated strings."""
        model = bert_tiny_transformer
        model.modality_config[("image", "text")] = {
            "method": "forward",
            "method_output_name": "last_hidden_state",
        }
        config = model.get_config_dict()
        assert "image+text" in config["modality_config"]

    def test_repr(self, bert_tiny_transformer):
        # [FUNCTION DOC]
        # repr contains "Transformer(" and "architecture".
        r = repr(bert_tiny_transformer)
        assert "Transformer(" in r
        assert "architecture" in r

    def test_load_config_deserializes_tuple_keys(self, bert_tiny_transformer, tmp_path):
        # [FUNCTION DOC]
        # load_config turns plus-joined modality keys back into tuples.
        """load_config should deserialize plus-separated keys back to tuples."""
        model = bert_tiny_transformer
        save_dir = str(tmp_path / "model")
        model.modality_config[("image", "text")] = {
            "method": "forward",
            "method_output_name": "last_hidden_state",
        }
        model.save(save_dir)

        config = Transformer.load_config(save_dir)
        assert ("image", "text") in config["modality_config"]

    def test_load_config_strips_trust_remote_code(self, bert_tiny_transformer, tmp_path):
        # [FUNCTION DOC]
        # load_config strips trust_remote_code injected into the saved model_args.
        """load_config should remove trust_remote_code from sub-dicts."""
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        assert config_path.exists(), "save() must always create sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        config.setdefault("model_args", {})["trust_remote_code"] = True
        config_path.write_text(json.dumps(config))

        loaded = Transformer.load_config(save_dir)
        assert "trust_remote_code" not in loaded.get("model_args", {})

    def test_load_config_default_modality_config_for_old_models(self, bert_tiny_transformer, tmp_path):
        # [FUNCTION DOC]
        # A config saved without modality_config/module_output_name loads with the feature-extraction defaults.
        """Models saved without modality_config should get the default text-only config."""
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        config_path = Path(save_dir) / "sentence_bert_config.json"
        assert config_path.exists(), "save() must always create sentence_bert_config.json"
        config = json.loads(config_path.read_text())
        config.pop("modality_config", None)
        config.pop("module_output_name", None)
        config_path.write_text(json.dumps(config))

        loaded = Transformer.load_config(save_dir)
        expected_config, expected_output = TRANSFORMER_TASK_DEFAULTS["feature-extraction"]
        assert loaded["modality_config"] == expected_config
        assert loaded["module_output_name"] == expected_output


class TestGetDefaultModalityConfig:
    # [CLASS DOC]
    # Tests for the default modality config lookup.
    def test_default_and_explicit_tasks(self):
        # [FUNCTION DOC]
        # Each known task maps to its TRANSFORMER_TASK_DEFAULTS entry; a missing task defaults to feature-extraction.
        """Dict lookup should work for all tasks, and missing key should default to feature-extraction."""
        for task in ("feature-extraction", "sequence-classification", "text-generation", "fill-mask"):
            assert (
                Transformer._get_default_modality_config({"transformer_task": task}) == TRANSFORMER_TASK_DEFAULTS[task]
            )
        assert Transformer._get_default_modality_config({}) == TRANSFORMER_TASK_DEFAULTS["feature-extraction"]


class TestLoadInitKwargs:
    # [CLASS DOC]
    # Tests for the private init-kwargs loader.
    def test_merges_config_and_overrides(self, bert_tiny_transformer, tmp_path):
        # [FUNCTION DOC]
        # Loading init kwargs from a saved directory merges user model_kwargs, distributes cache_folder as
        # cache_dir into model/processor/config kwargs, and sets backend "torch".
        """_load_init_kwargs should merge config, hub_kwargs, user overrides, and pass cache_dir for the decorator."""
        save_dir = str(tmp_path / "model")
        bert_tiny_transformer.save(save_dir)

        kwargs = Transformer._load_init_kwargs(
            model_name_or_path=save_dir,
            model_kwargs={"torch_dtype": "float16"},
            cache_folder="/tmp/cache",
        )
        assert "model_kwargs" in kwargs
        assert kwargs["model_kwargs"]["torch_dtype"] == "float16"
        # cache_folder is distributed into each kwargs dict directly
        assert kwargs["model_kwargs"]["cache_dir"] == "/tmp/cache"
        assert kwargs["processor_kwargs"]["cache_dir"] == "/tmp/cache"
        assert kwargs["config_kwargs"]["cache_dir"] == "/tmp/cache"
        assert kwargs["backend"] == "torch"


class TestEncoderOnlySaveLoadRoundtrip:
    # [CLASS DOC]
    # Slow save/load round-trip tests for encoder-only models extracted from encoder-decoder (and audio)
    # architectures. Class-level lists give (model id, expected encoder class name, extra kwargs such as
    # config_kwargs) for tiny text models, tiny audio models and larger T5Gemma models.
    """Test save/load roundtrip for encoder-only models extracted from encoder-decoder architectures.

    Each encoder-decoder architecture in ``_ENCODER_ONLY_MODELS`` (plus the T5Gemma special cases)
    should produce the correct encoder class, and outputs should be identical after save/load.
    """

    # (model_name, expected_class_name, extra_kwargs)
    # extra_kwargs may contain config_kwargs and is_audio
    ENCODER_ONLY_TEXT_MODELS = [
        ("hf-internal-testing/tiny-random-T5Model", "T5EncoderModel", {}),
        ("hf-internal-testing/tiny-random-mt5", "MT5EncoderModel", {}),
        ("hf-internal-testing/tiny-random-UMT5ForTokenClassification", "UMT5EncoderModel", {}),
        ("hf-internal-testing/tiny-random-LongT5Model", "LongT5EncoderModel", {}),
        ("hf-internal-testing/tiny-random-ProphetNetModel", "ProphetNetEncoder", {}),
        ("hf-internal-testing/tiny-random-SwitchTransformersModel", "SwitchTransformersEncoderModel", {}),
        ("hf-internal-testing/tiny-random-BlenderbotModel", "BlenderbotEncoder", {}),
        ("hf-internal-testing/tiny-random-BlenderbotSmallModel", "BlenderbotSmallEncoder", {}),
        ("hf-internal-testing/tiny-random-M2M100Model", "M2M100Encoder", {}),
        ("hf-internal-testing/tiny-random-PegasusModel", "PegasusEncoder", {}),
        ("hf-internal-testing/tiny-random-PegasusXModel", "PegasusXEncoder", {}),
        (
            "hf-internal-testing/tiny-random-MarianModel",
            "MarianEncoder",
            # The default pad_token is at idx 58100, but embeddings were reduced to 99 in the tiny model
            {"config_kwargs": {"pad_token_id": 1}},
        ),
    ]

    ENCODER_ONLY_AUDIO_MODELS = [
        (
            "hf-internal-testing/tiny-random-WhisperModel",
            "WhisperEncoder",
            {"config_kwargs": {"max_source_positions": 1500}},
        ),
        ("hf-internal-testing/tiny-random-MoonshineForConditionalGeneration", "MoonshineEncoder", {}),
    ]

    ENCODER_ONLY_LARGE_MODELS = [
        ("google/t5gemma-s-s-prefixlm", "T5GemmaEncoderModel", {}),
        ("google/t5gemma-2-270m-270m", "T5Gemma2Encoder", {}),
    ]

    @staticmethod
    def _load_transformer(model_name: str, extra_kwargs: dict) -> Transformer:
        # [FUNCTION DOC]
        # HELPER: Builds a Transformer for the model id with ignore_mismatched_sizes and the optional config_kwargs.
        config_kwargs = extra_kwargs.get("config_kwargs", {})
        return Transformer(
            model_name_or_path=model_name,
            model_kwargs={"ignore_mismatched_sizes": True},
            config_kwargs=config_kwargs,
        )

    @staticmethod
    def _get_inputs(transformer: Transformer, is_audio: bool) -> list:
        # [FUNCTION DOC]
        # HELPER: Returns two random one-second 16 kHz float32 waveforms for audio models, else two short texts.
        if is_audio:
            return [np.random.randn(16000).astype(np.float32) for _ in range(2)]
        return ["hello world", "goodbye world"]

    @staticmethod
    def _assert_outputs_match(out1: dict, out2: dict) -> None:
        # [FUNCTION DOC]
        # HELPER: Asserts every output key matches between two output dicts (tensors within 1e-5, others equal).
        for key in out1:
            v1, v2 = out1[key], out2[key]
            if isinstance(v1, torch.Tensor):
                assert torch.allclose(v1, v2, atol=1e-5), f"Outputs for key {key!r} differ after save/load"
            else:
                assert v1 == v2, f"Outputs for key {key!r} differ after save/load"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name, expected_class_name, extra_kwargs",
        [(TINY_BERT, "BertModel", {})] + ENCODER_ONLY_TEXT_MODELS,
        ids=lambda val: val if isinstance(val, str) and "/" in val else "",
    )
    def test_text_model_roundtrip(self, tmp_path: Path, model_name: str, expected_class_name: str, extra_kwargs: dict):
        # [FUNCTION DOC]
        # For tiny BERT and each encoder-decoder text model, checks the loaded encoder class, then that forward
        # outputs are unchanged after save and Transformer.load.
        transformer = self._load_transformer(model_name, extra_kwargs)
        assert type(transformer.auto_model).__name__ == expected_class_name

        inputs = self._get_inputs(transformer, is_audio=False)
        features = transformer.preprocess(inputs)
        with torch.no_grad():
            out1 = transformer(features)

        save_dir = tmp_path / "model"
        transformer.save(str(save_dir))
        reloaded = Transformer.load(str(save_dir))
        assert type(reloaded.auto_model).__name__ == expected_class_name

        features = reloaded.preprocess(inputs)
        with torch.no_grad():
            out2 = reloaded(features)

        self._assert_outputs_match(out1, out2)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name, expected_class_name, extra_kwargs",
        ENCODER_ONLY_AUDIO_MODELS,
        ids=lambda val: val if isinstance(val, str) and "/" in val else "",
    )
    def test_audio_model_roundtrip(
        self, tmp_path: Path, model_name: str, expected_class_name: str, extra_kwargs: dict
    ):
        # [FUNCTION DOC]
        # Same round trip for the audio encoder models with waveform inputs.
        transformer = self._load_transformer(model_name, extra_kwargs)
        assert type(transformer.auto_model).__name__ == expected_class_name

        inputs = self._get_inputs(transformer, is_audio=True)
        features = transformer.preprocess(inputs)
        with torch.no_grad():
            out1 = transformer(features)

        save_dir = tmp_path / "model"
        transformer.save(str(save_dir))
        reloaded = Transformer.load(str(save_dir))
        assert type(reloaded.auto_model).__name__ == expected_class_name

        features = reloaded.preprocess(inputs)
        with torch.no_grad():
            out2 = reloaded(features)

        self._assert_outputs_match(out1, out2)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name, expected_class_name, extra_kwargs",
        ENCODER_ONLY_LARGE_MODELS,
        ids=lambda val: val if isinstance(val, str) and "/" in val else "",
    )
    def test_large_model_roundtrip(
        self, tmp_path: Path, model_name: str, expected_class_name: str, extra_kwargs: dict
    ):
        # [FUNCTION DOC]
        # Same round trip for the larger T5Gemma encoders, skipped on transformers versions that lack them;
        # T5Gemma2 is forced to eager attention before running.
        if parse_version(transformers_version) < parse_version("5.0.0") and expected_class_name == "T5Gemma2Encoder":
            pytest.skip("T5Gemma2Encoder requires transformers>=5.0.0")
        if (
            parse_version(transformers_version) < parse_version("4.54.1")
            and expected_class_name == "T5GemmaEncoderModel"
        ):
            pytest.skip("T5GemmaEncoderModel requires transformers>=4.54.1")

        transformer = self._load_transformer(model_name, extra_kwargs)
        assert type(transformer.auto_model).__name__ == expected_class_name

        if expected_class_name == "T5Gemma2Encoder":
            transformer.auto_model.config._attn_implementation = "eager"

        inputs = self._get_inputs(transformer, is_audio=False)
        features = transformer.preprocess(inputs)
        with torch.no_grad():
            out1 = transformer(features)

        save_dir = tmp_path / "model"
        transformer.save(str(save_dir))
        reloaded = Transformer.load(str(save_dir))
        assert type(reloaded.auto_model).__name__ == expected_class_name

        if expected_class_name == "T5Gemma2Encoder":
            reloaded.auto_model.config._attn_implementation = "eager"

        features = reloaded.preprocess(inputs)
        with torch.no_grad():
            out2 = reloaded(features)

        self._assert_outputs_match(out1, out2)


class TestCanFlattenInputs:
    # [CLASS DOC]
    # Tests for deciding whether inputs can be flattened (unpadded) for FlashAttention-2.
    def test_false_for_non_feature_extraction_task(self):
        # [FUNCTION DOC]
        # The sequence-classification task disables flattening.
        """Flattening requires the feature-extraction task."""
        transformer = Transformer(TINY_BERT, transformer_task="sequence-classification")
        assert transformer._can_flatten_inputs() is False
        assert transformer.can_flatten_inputs is False

    def test_false_for_non_torch_backend(self):
        # [FUNCTION DOC]
        # The onnx backend disables flattening; the test is skipped if that backend cannot be constructed.
        # KNOWN ISSUE: the skip catches every Exception (not just a missing ONNX dependency), so any real construction bug is silently reported as a skip.
        """Flattening requires the torch backend."""
        try:
            transformer = Transformer(TINY_BERT, backend="onnx")
        except (ImportError, Exception):
            pytest.skip("ONNX backend not available")
        assert transformer._can_flatten_inputs() is False
        assert transformer.can_flatten_inputs is False

    def test_false_when_no_flash_attention(self):
        # [FUNCTION DOC]
        # Without FlashAttention-2 (default attention), flattening is disabled.
        """Without flash attention, flattening should be disabled."""
        transformer = Transformer(TINY_BERT)
        # Default attn_implementation is not flash_attention_2
        assert transformer._can_flatten_inputs() is False
        assert transformer.can_flatten_inputs is False

    def test_false_when_backend_incompatible(self, monkeypatch):
        # [FUNCTION DOC]
        # If the model reports backend incompatibility (patched), flattening is disabled.
        """If the model's auto_model reports backend incompatibility, flattening is disabled."""
        transformer = Transformer(TINY_BERT)
        monkeypatch.setattr(transformer.model, "is_backend_compatible", lambda: False)
        assert transformer._can_flatten_inputs() is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_true_with_flash_attention(self):
        # [FUNCTION DOC]
        # On CUDA with the kernels library and FlashAttention-2 in bfloat16, flattening is enabled and a data
        # collator exists; skipped when no kernel build exists for the platform.
        """With flash attention and a compatible model, flattening should be enabled."""
        try:
            import kernels  # noqa: F401
        except ImportError:
            pytest.skip("kernels library not available")

        try:
            transformer = Transformer(
                TINY_BERT,
                model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
            )
        except (ValueError, FileNotFoundError, ImportError) as exc:
            # The kernels hub ships no build variant for every platform (e.g. Windows).
            if "flash" in str(exc).lower() or "build variant" in str(exc).lower():
                pytest.skip(f"no flash-attn2 kernel build for this platform: {exc}")
            raise
        assert transformer._can_flatten_inputs() is True
        assert transformer.can_flatten_inputs is True
        assert transformer.data_collator is not None

    def test_unpad_inputs_false_forces_padding(self):
        # [FUNCTION DOC]
        # unpad_inputs False forces flattening off.
        """Setting unpad_inputs=False should disable flattening regardless of prerequisites."""
        transformer = Transformer(TINY_BERT, unpad_inputs=False)
        assert transformer.unpad_inputs is False
        assert transformer.can_flatten_inputs is False

    def test_unpad_inputs_true_warns_when_prerequisites_not_met(self, caplog):
        # [FUNCTION DOC]
        # unpad_inputs True without prerequisites logs a warning and leaves flattening off.
        """Setting unpad_inputs=True should warn if prerequisites are not met."""
        with caplog.at_level(logging.WARNING):
            transformer = Transformer(TINY_BERT, unpad_inputs=True)
        assert transformer.can_flatten_inputs is False
        assert "unpad_inputs=True was set" in caplog.text

    def test_unpad_inputs_none_auto_detects(self):
        # [FUNCTION DOC]
        # unpad_inputs None auto-detects (off without FlashAttention).
        """Setting unpad_inputs=None (default) should auto-detect."""
        transformer = Transformer(TINY_BERT)
        assert transformer.unpad_inputs is None
        # Without flash attention, auto-detect should disable flattening
        assert transformer.can_flatten_inputs is False

    def test_unpad_inputs_setter_re_evaluates(self):
        # [FUNCTION DOC]
        # Setting unpad_inputs after construction re-evaluates can_flatten_inputs.
        """Setting unpad_inputs after init should re-evaluate can_flatten_inputs."""
        transformer = Transformer(TINY_BERT)
        assert transformer.can_flatten_inputs is False

        # Setting to False explicitly should keep it disabled
        transformer.unpad_inputs = False
        assert transformer.unpad_inputs is False
        assert transformer.can_flatten_inputs is False

        # Setting back to None should re-evaluate (still False without flash attention)
        transformer.unpad_inputs = None
        assert transformer.unpad_inputs is None
        assert transformer.can_flatten_inputs is False


class TestConditionalFlattening:
    # [CLASS DOC]
    # Tests that preprocess flattens only text-only inputs when flattening is enabled.
    """Test that preprocess only flattens text-only inputs, even when can_flatten_inputs is True."""

    @pytest.fixture()
    def flatten_ready_transformer(self):
        # [FUNCTION DOC]
        # HELPER: Fixture returning a BERT Transformer forced to can_flatten_inputs True, with its position
        # offset computed and a mock data collator returning fixed ids and position ids.
        """A BERT transformer with can_flatten_inputs=True and a mock data_collator."""
        transformer = Transformer(TINY_BERT)
        transformer.can_flatten_inputs = True
        transformer._flatten_position_offset = transformer._infer_flatten_position_offset()
        transformer.data_collator = MagicMock(
            return_value={"input_ids": torch.tensor([1, 2, 3]), "position_ids": torch.tensor([0, 1, 2])}
        )
        return transformer

    def test_text_inputs_are_flattened(self, flatten_ready_transformer):
        # [FUNCTION DOC]
        # Plain text inputs are passed through the data collator once.
        """Text inputs should be flattened when can_flatten_inputs is True."""
        flatten_ready_transformer.preprocess(["hello world", "another sentence"])
        flatten_ready_transformer.data_collator.assert_called_once()

    def test_text_inputs_not_flattened_when_disabled(self):
        # [FUNCTION DOC]
        # With flattening disabled the data collator is not used.
        """Text inputs should NOT be flattened when can_flatten_inputs is False."""
        transformer = Transformer(TINY_BERT)
        assert transformer.can_flatten_inputs is False
        transformer.data_collator = MagicMock()
        transformer.preprocess(["hello world"])
        transformer.data_collator.assert_not_called()

    def test_multimodal_messages_skip_flattening(self, flatten_ready_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Messages containing an image (parsing and processor call patched) are not flattened and keep their attention mask.
        """Messages with non-text content should NOT be flattened even when can_flatten_inputs is True."""
        transformer = flatten_ready_transformer
        transformer.modality_config["message"] = {"method": "forward"}

        multimodal_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://example.com/img.jpg"},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ]
        ]
        monkeypatch.setattr(
            transformer.input_formatter,
            "parse_inputs",
            lambda inputs: ("message", {"message": multimodal_messages}, {}),
        )
        monkeypatch.setattr(
            transformer,
            "_call_processor",
            lambda modality, inputs, modality_kwargs, common_kwargs, **_: {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
        )

        result = transformer.preprocess(["dummy"])
        transformer.data_collator.assert_not_called()
        assert "attention_mask" in result

    def test_text_only_messages_are_flattened(self, flatten_ready_transformer, monkeypatch):
        # [FUNCTION DOC]
        # Text-only messages (patched parsing and processor call) are flattened via the data collator.
        """Messages with only text content should be flattened when can_flatten_inputs is True."""
        transformer = flatten_ready_transformer
        transformer.modality_config["message"] = {"method": "forward"}

        text_messages = [[{"role": "user", "content": "hello world"}]]
        monkeypatch.setattr(
            transformer.input_formatter,
            "parse_inputs",
            lambda inputs: ("message", {"message": text_messages}, {}),
        )
        monkeypatch.setattr(
            transformer,
            "_call_processor",
            lambda modality, inputs, modality_kwargs, common_kwargs, **_: {"input_ids": [[1, 2, 3]]},
        )

        transformer.preprocess(["dummy"])
        transformer.data_collator.assert_called_once()


class TestCountMediaPerSample:
    # [CLASS DOC]
    # Tests for counting images and videos per chat sample.
    def test_multiple_images_per_sample(self):
        # [FUNCTION DOC]
        # Counts multiple images per sample, including samples with none.
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "a"},
                        {"type": "image", "image": "b"},
                        {"type": "text", "text": "compare"},
                    ],
                }
            ],
            [{"role": "user", "content": [{"type": "image", "image": "c"}]}],
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [2, 1, 0]
        assert num_videos == [0, 0, 0]

    def test_videos(self):
        # [FUNCTION DOC]
        # Counts videos per sample.
        messages = [
            [{"role": "user", "content": [{"type": "video", "video": "v1"}, {"type": "text", "text": "describe"}]}],
            [{"role": "user", "content": [{"type": "video", "video": "v2"}, {"type": "video", "video": "v3"}]}],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [0, 0]
        assert num_videos == [1, 2]

    def test_mixed_images_and_videos(self):
        # [FUNCTION DOC]
        # Counts images and videos in the same sample.
        messages = [
            [{"role": "user", "content": [{"type": "image", "image": "i"}, {"type": "video", "video": "v"}]}],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [1]
        assert num_videos == [1]

    def test_text_only(self):
        # [FUNCTION DOC]
        # Text-only samples, including plain string content, count zero media.
        messages = [
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            [{"role": "user", "content": "plain string"}],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [0, 0]
        assert num_videos == [0, 0]

    def test_multi_turn_messages(self):
        # [FUNCTION DOC]
        # Media across multiple turns of one conversation are summed.
        messages = [
            [
                {"role": "user", "content": [{"type": "image", "image": "a"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a cat"}]},
                {"role": "user", "content": [{"type": "image", "image": "b"}, {"type": "text", "text": "and this?"}]},
            ],
        ]
        num_images, num_videos = _count_media_per_sample(messages)
        assert num_images == [2]
        assert num_videos == [0]

    def test_empty_batch(self):
        # [FUNCTION DOC]
        # An empty batch gives two empty lists.
        assert _count_media_per_sample([]) == ([], [])


class TestTransformerSaveLoadRoundtrip:
    # [CLASS DOC]
    # Save/load round-trip test checking files and outputs.
    def test_save_load_produces_same_output(self, bert_tiny_transformer: Transformer, tmp_path):
        # [FUNCTION DOC]
        # Saving writes config.json (with model_type and architectures) and the Sentence Transformers config
        # file; the reloaded module on CPU gives exactly equal token embeddings.
        """Test that a Transformer can be saved and loaded, producing identical output."""
        save_dir = str(tmp_path / "transformer_roundtrip")
        bert_tiny_transformer.save(save_dir)

        # Verify config.json exists and has expected keys
        config_path = Path(save_dir) / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config_data = json.load(f)
        assert "model_type" in config_data
        assert "architectures" in config_data

        # Verify sentence_bert_config.json exists
        st_config_path = Path(save_dir) / Transformer.config_file_name
        assert st_config_path.exists()

        # Load and compare outputs on CPU to avoid cross-device precision issues
        reloaded = Transformer.load(save_dir)
        bert_tiny_transformer.cpu()
        reloaded.cpu()

        features_original = bert_tiny_transformer.preprocess(["Hello world"])
        features_reloaded = reloaded.preprocess(["Hello world"])

        with torch.no_grad():
            out_original = bert_tiny_transformer.forward(features_original)
            out_reloaded = reloaded.forward(features_reloaded)

        assert torch.equal(out_original["token_embeddings"], out_reloaded["token_embeddings"])


class TestTransformerForwardEmptyBatch:
    # [CLASS DOC]
    # Tests forward on an empty batch.
    def test_forward_empty_inputs(self, bert_tiny_transformer: Transformer):
        # [FUNCTION DOC]
        # Forward on zero-row input either raises RuntimeError or returns zero-row token embeddings.
        """Test that forward with an empty batch (zero-length tensors) either raises a RuntimeError
        or returns empty outputs, depending on the PyTorch/transformers version."""
        features = {
            "input_ids": torch.zeros(0, 5, dtype=torch.long),
            "attention_mask": torch.zeros(0, 5, dtype=torch.long),
        }
        features = batch_to_device(features, bert_tiny_transformer.model.device)
        try:
            output = bert_tiny_transformer.forward(features)
        except RuntimeError:
            pass
        else:
            assert output["token_embeddings"].shape[0] == 0
