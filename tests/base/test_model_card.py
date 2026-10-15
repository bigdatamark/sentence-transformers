# [MODULE DOC]
# Pytest tests for BaseModelCardData, the model-card metadata object shared by all model types: widget example
# selection from training datasets, detection and formatting of media values (images, audio and video dicts), saving
# non-text usage examples as asset files, usage snippet generation (text and non-text), dataset statistics, and
# multimodal support further below. Optional dependencies (Pillow, datasets with its Audio/Image/Video features, PyAV,
# torchcodec) are imported when available and set to None otherwise; tests needing them are skipped. Skips the whole
# module without the training extra.
"""Tests for BaseModelCardData: snippet generation, asset saving, dataset metrics, and multimodal support."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import contextmanager
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.model_card import BaseModelCardData, generate_model_card
from sentence_transformers.util import is_datasets_available, is_training_available

try:
    from PIL import Image as PILModule
except ImportError:
    PILModule = None

if is_datasets_available():
    from datasets import Dataset, DatasetDict

    try:
        from datasets import Audio as AudioFeature
        from datasets import Image as ImageFeature
    except ImportError:
        AudioFeature = None
        ImageFeature = None

    try:
        from datasets import Video as VideoFeature
    except ImportError:
        VideoFeature = None

try:
    import av as _av
except ImportError:
    _av = None

try:
    import torchcodec as _torchcodec
except ImportError:
    _torchcodec = None

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def _make_model_card_data(cls=BaseModelCardData, **kwargs) -> BaseModelCardData:
    # [FUNCTION DOC]
    # HELPER: Constructs a model card data object of the given class (default the base class) with the given keyword
    # arguments and returns it.
    # KNOWN ISSUE: The docstring mentions "common defaults", but no defaults are applied beyond the constructor's own.
    """Create a BaseModelCardData instance with common defaults for testing."""
    data = cls(**kwargs)
    return data


def _make_pil_image(width: int = 64, height: int = 64) -> PILModule.Image:
    # [FUNCTION DOC]
    # HELPER: Returns a solid red RGB PIL image of the given width and height (default 64 by 64). Requires Pillow.
    """Create a small dummy PIL image."""
    return PILModule.new("RGB", (width, height), color=(255, 0, 0))


class _FakeLoss:
    # [CLASS DOC]
    # HELPER: Empty placeholder standing in for a loss object when computing dataset metrics.
    """Minimal stand-in for a loss object in compute_dataset_metrics calls."""

    pass


@pytest.mark.skipif(not is_datasets_available(), reason="datasets is not installed")
class TestSetWidgetExamples:
    # [CLASS DOC]
    # Tests for choosing widget and usage examples from a training dataset (skipped without datasets).
    def test_generates_examples_for_custom_transform_dataset(self) -> None:
        # [FUNCTION DOC]
        # Builds a 10-row dataset of query ids, document id lists and scores with an on-the-fly transform that
        # produces query and two document strings (keeping scores). Asserts widget and usage examples are produced,
        # all usage examples are strings, and no raw document id leaks into them (rows must be sampled through the
        # transform).
        # select_columns would starve a set_transform dataset (features are pre-transform), so rows
        # are sampled through the transform: widgets come from the transformed string values only.
        dataset = Dataset.from_dict(
            {
                "query_id": [f"q{i}" for i in range(10)],
                "document_ids": [[f"d{i}a", f"d{i}b"] for i in range(10)],
                "scores": [[1.0, 0.5]] * 10,
            }
        )

        def transform(batch):
            # [FUNCTION DOC]
            # HELPER: Transform producing descriptive query and document strings from the query ids and passing the
            # scores through.
            # Mimics resolve_ids: needs the "scores" column that select_columns would drop.
            return {
                "query": [f"query about topic {qid}" for qid in batch["query_id"]],
                "document_1": [f"first document for {qid}" for qid in batch["query_id"]],
                "document_2": [f"second document for {qid}" for qid in batch["query_id"]],
                "scores": batch["scores"],
            }

        dataset.set_transform(transform)

        data = _make_model_card_data()
        data.set_widget_examples(dataset)
        assert data.widget
        assert data.usage_examples
        assert all(isinstance(text, str) for text in data.usage_examples)
        assert not any("d0a" in text for text in data.usage_examples), "raw IDs must not leak into examples"

    def test_widget_examples_skipped_when_transform_fails(self) -> None:
        # [FUNCTION DOC]
        # Asserts that a dataset whose transform raises KeyError on access results in an empty widget list instead of
        # an error.
        # A transform that fails on row access (e.g. a lookup miss) must be skipped, not crash the card.
        dataset = Dataset.from_dict({"query_id": ["q1", "q2", "q3"]})

        def transform(batch):
            # [FUNCTION DOC]
            # HELPER: Transform that always raises KeyError.
            raise KeyError("lookup miss")

        dataset.set_transform(transform)

        data = _make_model_card_data()
        data.set_widget_examples(dataset)
        assert data.widget == []

    def test_widget_examples_prefer_shortest_rows(self) -> None:
        # [FUNCTION DOC]
        # Builds 8 rows by 4 text columns whose length grows with the row index, sets widget examples, and asserts
        # widgets exist and none of the three longest rows (5 to 7) appear in the usage examples or widget texts
        # (shortest sampled rows are preferred, read back from the same rows).
        # lengths are measured on the sampled rows, so the picks must be read back from those same
        # rows: the 3 longest rows must never surface (regression for the index-space mismatch).
        dataset = Dataset.from_dict(
            {f"text{column}": [f"{'x' * (20 * i)} row{i} col{column}" for i in range(8)] for column in range(4)}
        )
        data = _make_model_card_data()
        data.set_widget_examples(dataset)
        assert data.widget
        widget_texts = [
            text
            for entry in data.widget
            for text in (entry.get("source_sentence", ""), entry.get("text", ""), *entry.get("sentences", []))
        ]
        all_texts = " ".join([*data.usage_examples, *widget_texts])
        assert not any(f"row{i}" in all_texts for i in (5, 6, 7))

    def test_generates_examples_for_plain_text_dataset(self) -> None:
        # [FUNCTION DOC]
        # Asserts a plain two-column text dataset without a transform still produces widget examples.
        # A plain text dataset (no transform) must still produce widget examples: the skip above is targeted.
        dataset = Dataset.from_dict(
            {
                "anchor": [f"anchor sentence {i}" for i in range(10)],
                "positive": [f"positive sentence {i}" for i in range(10)],
            }
        )
        data = _make_model_card_data()
        data.set_widget_examples(dataset)
        assert data.widget


class TestIsTypedMediaDict:
    # [CLASS DOC]
    # Tests the static check for "typed media" dicts (audio or video payloads).
    def test_audio_dict(self) -> None:
        # [FUNCTION DOC]
        # Asserts a dict with an array and a sampling rate is a typed media dict.
        assert BaseModelCardData._is_typed_media_dict({"array": np.zeros(10), "sampling_rate": 16000}) is True

    def test_video_dict(self) -> None:
        # [FUNCTION DOC]
        # Asserts a dict with an array and video metadata is a typed media dict.
        assert (
            BaseModelCardData._is_typed_media_dict({"array": np.zeros((8, 3, 64, 64)), "video_metadata": {"fps": 30}})
            is True
        )

    def test_multimodal_dict(self) -> None:
        # [FUNCTION DOC]
        # Asserts a text-plus-image multimodal dict is not a typed media dict.
        assert BaseModelCardData._is_typed_media_dict({"text": "hello", "image": "url"}) is False

    def test_empty_dict(self) -> None:
        # [FUNCTION DOC]
        # Asserts an empty dict is not a typed media dict.
        assert BaseModelCardData._is_typed_media_dict({}) is False

    def test_non_dict(self) -> None:
        # [FUNCTION DOC]
        # Asserts a non-dict value is not a typed media dict.
        assert BaseModelCardData._is_typed_media_dict("not a dict") is False


class TestFormatSnippetValue:
    # [CLASS DOC]
    # Tests converting asset paths to URLs and formatting values for code snippets.
    """Test _asset_path_to_url and _format_snippet_value."""

    def test_asset_url_with_model_id(self) -> None:
        # [FUNCTION DOC]
        # Asserts an asset path becomes the Hub resolve-main URL under the model id.
        data = _make_model_card_data()
        data.model_id = "user/my-model"
        url = data._asset_path_to_url("assets/image_0.jpg")
        assert url == "https://huggingface.co/user/my-model/resolve/main/assets/image_0.jpg"

    def test_asset_url_without_model_id(self) -> None:
        # [FUNCTION DOC]
        # Asserts an asset path is returned unchanged when there is no model id.
        data = _make_model_card_data()
        data.model_id = None
        url = data._asset_path_to_url("assets/image_0.jpg")
        assert url == "assets/image_0.jpg"

    def test_plain_string(self) -> None:
        # [FUNCTION DOC]
        # Asserts a plain string is formatted as a single-quoted literal.
        data = _make_model_card_data()
        assert data._format_snippet_value("hello") == "'hello'"

    def test_asset_path_with_model_id(self) -> None:
        # [FUNCTION DOC]
        # Asserts an asset path with a model id is formatted as a string containing the Hub URL and the path.
        data = _make_model_card_data()
        data.model_id = "user/model"
        result = data._format_snippet_value("assets/image_0.jpg")
        assert "huggingface.co/user/model" in result
        assert "assets/image_0.jpg" in result

    def test_asset_path_without_model_id(self) -> None:
        # [FUNCTION DOC]
        # Asserts an asset path without a model id is formatted as the quoted path.
        data = _make_model_card_data()
        data.model_id = None
        result = data._format_snippet_value("assets/image_0.jpg")
        assert result == "'assets/image_0.jpg'"


class TestFormatExampleValue:
    # [CLASS DOC]
    # Tests formatting of values for the dataset examples table.
    """Test formatting for the dataset examples table in the model card."""

    def test_short_string(self) -> None:
        # [FUNCTION DOC]
        # Asserts a short string is returned as is.
        assert BaseModelCardData._format_example_value("hello") == "hello"

    def test_long_string_truncated(self) -> None:
        # [FUNCTION DOC]
        # Asserts a 1500-character string is truncated to 1000 characters plus "...".
        long_str = "x" * 1500
        result = BaseModelCardData._format_example_value(long_str)
        assert result.endswith("...")
        assert len(result) == 1003  # 1000 chars + "..."

    def test_list_truncated(self) -> None:
        # [FUNCTION DOC]
        # Asserts a seven-element list is shown truncated, ending with "...]".
        result = BaseModelCardData._format_example_value([1, 2, 3, 4, 5, 6, 7])
        assert "...]" in result

    def test_short_list_not_truncated(self) -> None:
        # [FUNCTION DOC]
        # Asserts a three-element list is not truncated.
        result = BaseModelCardData._format_example_value([1, 2, 3])
        assert "..." not in result

    def test_newlines_replaced(self) -> None:
        # [FUNCTION DOC]
        # Asserts newlines are replaced by HTML line breaks.
        result = BaseModelCardData._format_example_value("line1\nline2")
        assert "<br>" in result
        assert "\n" not in result

    def test_pipe_escaped(self) -> None:
        # [FUNCTION DOC]
        # Asserts pipe characters are escaped with a backslash (Markdown table safety).
        result = BaseModelCardData._format_example_value("a|b")
        assert "\\|" in result

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_pil_image_placeholder(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts a 100x200 PIL image is shown as an image placeholder including its size.
        img = _make_pil_image(100, 200)
        result = BaseModelCardData._format_example_value(img)
        assert "image" in result
        assert "100x200" in result

    def test_audio_dict_placeholder(self) -> None:
        # [FUNCTION DOC]
        # Asserts a one-second 16 kHz audio dict is shown as an audio placeholder with "1.00s" and "16000 Hz".
        audio = {"array": np.zeros(16000), "sampling_rate": 16000}
        result = BaseModelCardData._format_example_value(audio)
        assert "audio" in result
        assert "1.00s" in result
        assert "16000 Hz" in result

    def test_video_dict_placeholder(self) -> None:
        # [FUNCTION DOC]
        # Asserts a video dict is shown as a placeholder containing "video".
        video = {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 30}}
        result = BaseModelCardData._format_example_value(video)
        assert "video" in result


class TestSavePredictExampleAssets:
    # [CLASS DOC]
    # Tests saving non-text usage examples into an assets folder of the save directory and recording display paths.
    """Test saving non-text predict examples to files."""

    @pytest.mark.parametrize(
        "usage_examples",
        [
            ["hello", "world"],
            [["q", "a1"], ["q", "a2"]],
        ],
        ids=["flat_strings", "list_of_lists"],
    )
    def test_text_only_does_not_create_assets_dir(self, usage_examples) -> None:
        # [FUNCTION DOC]
        # Parametrised over flat strings and CrossEncoder-style pairs. Asserts that with only text examples no assets
        # folder is created and no display examples are set.
        """When all examples are text (flat or CrossEncoder-style pairs), no assets dir is created."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = usage_examples
            data.save_usage_example_assets()

            assert not os.path.exists(os.path.join(tmpdir, "assets"))
            assert data.usage_examples_display is None

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_pil_image_saved_as_jpg(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts one PIL image is saved as assets/image_0.jpg and recorded as its display path.
        """A PIL image in usage_examples is saved to assets/image_0.jpg."""
        data = _make_model_card_data()
        img = _make_pil_image(32, 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = [img]
            data.save_usage_example_assets()

            assert data.usage_examples_display == ["assets/image_0.jpg"]
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_multiple_images_indexed_correctly(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts three differently sized images are saved as image_0 to image_2 and recorded in
        # order.
        """Multiple distinct images get sequential indices."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            # Use different sizes so they're not deduplicated
            data.usage_examples = [_make_pil_image(32, 32), _make_pil_image(48, 48), _make_pil_image(64, 64)]
            data.save_usage_example_assets()

            assert data.usage_examples_display == [
                "assets/image_0.jpg",
                "assets/image_1.jpg",
                "assets/image_2.jpg",
            ]
            for i in range(3):
                assert os.path.isfile(os.path.join(tmpdir, "assets", f"image_{i}.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_duplicate_images_deduplicated(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts three references to the same image are saved once and all display entries
        # point to image_0.
        """Identical images are saved only once and share the same path."""
        data = _make_model_card_data()
        img = _make_pil_image(32, 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = [img, img, img]
            data.save_usage_example_assets()

            assert data.usage_examples_display == [
                "assets/image_0.jpg",
                "assets/image_0.jpg",
                "assets/image_0.jpg",
            ]
            # Only one file saved
            assert os.listdir(os.path.join(tmpdir, "assets")) == ["image_0.jpg"]

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_multimodal_dict_with_text_and_image(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts a text-plus-image dict keeps its text and replaces the image with its saved
        # asset path.
        """Multimodal dict: text kept as-is, image saved to file."""
        data = _make_model_card_data()
        img = _make_pil_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = [{"text": "A cat", "image": img}]
            data.save_usage_example_assets()

            assert len(data.usage_examples_display) == 1
            display = data.usage_examples_display[0]
            assert display["text"] == "A cat"
            assert display["image"] == "assets/image_0.jpg"
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_mixed_text_and_images(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts strings stay unchanged and an image in between is replaced by its asset path.
        """A mix of strings and images in usage_examples."""
        data = _make_model_card_data()
        img = _make_pil_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = ["hello", img, "world"]
            data.save_usage_example_assets()

            assert data.usage_examples_display[0] == "hello"
            assert data.usage_examples_display[1] == "assets/image_0.jpg"
            assert data.usage_examples_display[2] == "world"

    def test_audio_dict_saved_as_wav(self, tmp_path) -> None:
        # [FUNCTION DOC]
        # Skips without a working torchaudio save backend (checked by writing a test file). Asserts a 16 kHz audio
        # dict is saved as assets/audio_0.wav and recorded.
        """AudioDict saved to assets/ as .wav file."""
        torchaudio = pytest.importorskip("torchaudio")
        # Verify torchaudio can actually save (may fail without a backend like soundfile/sox)
        try:
            torchaudio.save(str(tmp_path / "check.wav"), torch.randn(1, 100), 16000)
        except Exception:
            pytest.skip("torchaudio backend cannot save wav files")

        data = _make_model_card_data()
        audio = {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000}
        data.save_dir = str(tmp_path)
        data.usage_examples = [audio]
        data.save_usage_example_assets()

        assert data.usage_examples_display == ["assets/audio_0.wav"]
        assert (tmp_path / "assets" / "audio_0.wav").is_file()

    def test_video_dict_saved_as_mp4(self) -> None:
        # [FUNCTION DOC]
        # Skips without PyAV. Asserts an 8-frame uint8 video dict (frames, channels, height, width) at 30 fps is saved
        # as assets/video_0.mp4 and recorded.
        """VideoDict saved to assets/ as .mp4 file."""
        pytest.importorskip("av")

        data = _make_model_card_data()
        # (T, C, H, W) uint8 video tensor
        video = {"array": np.random.randint(0, 255, (8, 3, 64, 64), dtype=np.uint8), "video_metadata": {"fps": 30}}
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = [video]
            data.save_usage_example_assets()

            assert data.usage_examples_display == ["assets/video_0.mp4"]
            assert os.path.isfile(os.path.join(tmpdir, "assets", "video_0.mp4"))

    def test_no_save_dir_is_noop(self) -> None:
        # [FUNCTION DOC]
        # Asserts that without a save directory nothing is saved and no display examples are set.
        """When save_dir is not set, nothing happens."""
        data = _make_model_card_data()
        data.usage_examples = [_make_pil_image()] if PILModule else ["text"]
        data.save_dir = None
        data.save_usage_example_assets()

        assert data.usage_examples_display is None

    def test_no_usage_examples_is_noop(self) -> None:
        # [FUNCTION DOC]
        # Asserts that without usage examples nothing is saved.
        """When usage_examples is None, nothing happens."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = None
            data.save_usage_example_assets()

            assert data.usage_examples_display is None


class TestGenerateUsageSnippet:
    # [CLASS DOC]
    # Tests usage snippet generation for SentenceTransformer card data (text and non-text inputs).
    """Test snippet generation for both text-only and non-text inputs."""

    def test_text_default_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts the default snippet is a Python code block using SentenceTransformer, the default weather example
        # sentence, encode and similarity calls.
        """Default examples are used when usage_examples is None."""
        model = stsb_bert_tiny_model
        snippet = model.model_card_data.generate_usage_snippet()

        assert "```python" in snippet
        assert "```" in snippet
        assert "SentenceTransformer" in snippet
        assert "'The weather is lovely today.'" in snippet
        assert "model.encode(sentences)" in snippet
        assert "model.similarity(embeddings, embeddings)" in snippet

    def test_text_custom_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts custom usage strings appear quoted and the embedding shape comment starts with 3 rows.
        """usage_examples strings appear in the snippet."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["Hello", "World", "Test"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "'Hello'" in snippet
        assert "'World'" in snippet
        assert "'Test'" in snippet
        assert "# [3," in snippet

    def test_text_with_similarities(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts that preset similarities are printed in the snippet.
        """When similarities are computed, they appear in the snippet."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["A", "B"]
        model.model_card_data.similarities = "# tensor([[1.0, 0.5],\n#         [0.5, 1.0]])"
        snippet = model.model_card_data.generate_usage_snippet()

        assert "print(similarities)" in snippet
        assert "# tensor([[1.0, 0.5]," in snippet

    def test_text_without_similarities(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts that without similarities the snippet prints the similarity shape with a 2 by 2 comment.
        """When no similarities, shape comment is shown."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["A", "B"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "print(similarities.shape)" in snippet
        assert "# [2, 2]" in snippet

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("tomaarsen/my-cool-model", 'SentenceTransformer("tomaarsen/my-cool-model")'),
            (None, "sentence_transformers_model_id"),
        ],
        ids=["custom_id", "default_placeholder"],
    )
    def test_text_model_id(self, stsb_bert_tiny_model: SentenceTransformer, model_id, expected) -> None:
        # [FUNCTION DOC]
        # Parametrised over a custom model id and None. Asserts the loading line uses the id, or the placeholder id
        # when None.
        """model_id appears in the loading line, or a placeholder is used when None."""
        model = stsb_bert_tiny_model
        model.model_card_data.model_id = model_id
        model.model_card_data.usage_examples = ["test"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert expected in snippet

    def test_text_output_dimensionality(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts the embedding shape comment shows two rows and the model's embedding dimension.
        """Embedding dimension appears in the shape comment."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["A", "B"]
        dim = model.get_embedding_dimension()
        snippet = model.model_card_data.generate_usage_snippet()

        assert f"# [2, {dim}]" in snippet

    def test_display_precedence(self) -> None:
        # [FUNCTION DOC]
        # Asserts display examples take precedence over raw usage examples in the snippet.
        """usage_examples_display takes precedence over usage_examples for rendering."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.usage_examples = ["original A", "original B"]
        data.usage_examples_display = ["display A", "display B"]
        snippet = data.generate_usage_snippet()

        assert "'display A'" in snippet
        assert "'display B'" in snippet
        assert "original" not in snippet

    def test_non_text_multimodal_dict(self) -> None:
        # [FUNCTION DOC]
        # Asserts the private non-text snippet for multimodal dict displays builds an inputs list of dicts with text
        # and image keys, converts asset paths to Hub URLs, and encodes and compares the inputs.
        """Multimodal dicts produce inputs = [{...}, ...] format."""
        data = _make_model_card_data()
        data.model_id = "user/model"
        data.similarities = None
        data.model = None
        display = [
            {"text": "A cat", "image": "assets/image_0.jpg"},
            {"text": "A dog", "image": "assets/image_1.jpg"},
        ]
        snippet = data._generate_non_text_snippet(display)

        assert "inputs = [" in snippet
        assert "'text'" in snippet
        assert "'image'" in snippet
        assert "huggingface.co/user/model" in snippet
        assert "model.encode(inputs)" in snippet
        assert "model.similarity(" in snippet

    def test_non_text_single_modality(self) -> None:
        # [FUNCTION DOC]
        # Asserts the non-text snippet for single-modality asset paths builds an inputs list with Hub URLs and encodes
        # it.
        """Single non-text modality produces inputs = [...] format."""
        data = _make_model_card_data()
        data.model_id = "user/model"
        data.similarities = None
        data.model = None
        display = ["assets/image_0.jpg", "assets/image_1.jpg"]
        snippet = data._generate_non_text_snippet(display)

        assert "inputs = [" in snippet
        assert "huggingface.co/user/model" in snippet
        assert "model.encode(inputs)" in snippet

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_non_text_dispatch_multimodal(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts the public snippet generator dispatches to the multimodal form (inputs list
        # with a text key) when usage examples are dicts with display paths.
        """generate_usage_snippet dispatches to multimodal for dict usage_examples."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.usage_examples = [{"text": "A", "image": _make_pil_image()}]
        data.usage_examples_display = [{"text": "A", "image": "assets/image_0.jpg"}]
        snippet = data.generate_usage_snippet()

        assert "inputs = [" in snippet
        assert "'text'" in snippet

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_non_text_dispatch_single_modality(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts PIL image usage examples dispatch to the single-modality form and render the
        # display asset path.
        """PIL images in usage_examples cause single-modality dispatch, rendering display paths."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.usage_examples = [_make_pil_image(), _make_pil_image()]
        data.usage_examples_display = ["assets/image_0.jpg", "assets/image_1.jpg"]
        snippet = data.generate_usage_snippet()

        assert "inputs = [" in snippet
        assert "assets/image_0.jpg" in snippet


@pytest.mark.skipif(
    PILModule is None or not is_datasets_available() or ImageFeature is None,
    reason="PIL, datasets, or datasets.Image not available",
)
class TestSetMultimodalPredictExample:
    # [CLASS DOC]
    # Tests choosing multimodal usage examples from a DatasetDict via the private setter, with the model's modalities
    # property temporarily replaced by a property mock (skipped without Pillow, datasets or its Image feature).
    """Test multimodal usage_examples extraction from datasets."""

    def _make_text_image_dataset(self, n: int = 5) -> DatasetDict:
        # [FUNCTION DOC]
        # HELPER: Returns a DatasetDict with a "train" split of n rows (default 5) having "text i" strings and
        # distinct-sized images (64+i square) cast to the datasets Image feature.
        images = [_make_pil_image(width=64 + i, height=64 + i) for i in range(n)]
        ds = Dataset.from_dict({"text": [f"text {i}" for i in range(n)], "image": images})
        ds = ds.cast_column("image", ImageFeature())
        return DatasetDict(train=ds)

    def _make_image_dataset(self, n: int = 5) -> DatasetDict:
        # [FUNCTION DOC]
        # HELPER: Returns a DatasetDict with a "train" split of n distinct-sized images only.
        ds = Dataset.from_dict({"image": [_make_pil_image(width=64 + i, height=64 + i) for i in range(n)]})
        ds = ds.cast_column("image", ImageFeature())
        return DatasetDict(train=ds)

    def test_combined_modality_builds_dicts(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With modalities text, image and the image+text tuple, asserts three usage examples are chosen and they are
        # dicts with text and image keys.
        """A model with tuple modality ("image", "text") builds multimodal dicts from text+image columns."""
        model = stsb_bert_tiny_model
        dd = self._make_text_image_dataset()

        # BLIP-like: supports text, image, and the combined ("image", "text")
        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=["text", "image", ("image", "text")],
        ):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples is not None
        assert len(model.model_card_data.usage_examples) == 3
        first = model.model_card_data.usage_examples[0]
        assert isinstance(first, dict)
        assert "text" in first
        assert "image" in first

    def test_separate_modalities_no_combined_dict(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With text and image but no combined tuple, asserts three examples are chosen and they are single images, not
        # dicts.
        """A CLIP-like model (text + image separately, no tuple) does NOT build combined dicts.

        Instead, it picks the first non-text modality and shows single-modality examples.
        """
        model = stsb_bert_tiny_model
        dd = self._make_text_image_dataset()

        # CLIP-like: supports text and image independently, but no ("image", "text") tuple
        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=["text", "image"],
        ):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples is not None
        assert len(model.model_card_data.usage_examples) == 3
        # Should be raw images, NOT dicts, CLIP can't process combined inputs
        assert not isinstance(model.model_card_data.usage_examples[0], dict)

    def test_image_only_model(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With an image-only model, asserts three image examples (not dicts).
        """Image-only model: usage_examples is a list of images, not dicts."""
        model = stsb_bert_tiny_model
        dd = self._make_image_dataset()

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples is not None
        assert len(model.model_card_data.usage_examples) == 3
        assert not isinstance(model.model_card_data.usage_examples[0], dict)

    def test_text_only_model_skips(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With a text-only model, asserts existing usage examples are left untouched.
        """Text-only model does not produce multimodal usage_examples."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["original"]
        dd = self._make_text_image_dataset()

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["text"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples == ["original"]

    def test_combined_only_model(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With only the image+text tuple as modality, asserts three combined dict examples.
        """A Kosmos-like model with only ("image", "text") still builds combined dicts."""
        model = stsb_bert_tiny_model
        dd = self._make_text_image_dataset()

        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=[("image", "text")],
        ):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert len(model.model_card_data.usage_examples) == 3
        first = model.model_card_data.usage_examples[0]
        assert isinstance(first, dict)
        assert "text" in first and "image" in first

    def test_small_dataset_fewer_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts a one-row image dataset yields a single example.
        """Dataset smaller than 3 examples: uses what's available."""
        model = stsb_bert_tiny_model
        dd = self._make_image_dataset(n=1)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert len(model.model_card_data.usage_examples) == 1

    def test_no_matching_columns_skips(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With an audio-only model and a text-only dataset, asserts existing examples are left untouched.
        """If dataset has no columns matching the model's non-text modalities, nothing happens."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["original"]

        # Dataset with only text, but model wants audio
        ds = Dataset.from_dict({"text": ["hello", "world"]})
        dd = DatasetDict(train=ds)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["audio"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples == ["original"]

    def test_duplicate_images_deduplicated(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With three copies of one image followed by three distinct images, asserts three examples are chosen with
        # three distinct sizes (duplicates skipped).
        """Duplicate images in the first rows are skipped. Unique images are picked from later rows."""
        model = stsb_bert_tiny_model

        # First 3 rows have the same image, rows 3-5 have distinct images
        same_image = _make_pil_image(64, 64)
        images = [same_image.copy() for _ in range(3)] + [_make_pil_image(w, w) for w in (32, 48, 96)]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())
        dd = DatasetDict(train=ds)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        examples = model.model_card_data.usage_examples
        assert examples is not None
        assert len(examples) == 3
        # All three should be distinct images (different sizes confirm uniqueness)
        sizes = {img.size for img in examples}
        assert len(sizes) == 3


@pytest.mark.skipif(
    PILModule is None or not is_datasets_available() or ImageFeature is None,
    reason="PIL, datasets, or datasets.Image not available",
)
class TestComputeDatasetMetricsNonText:
    # [CLASS DOC]
    # Tests dataset statistics and example tables for image columns (skipped without Pillow, datasets or its Image
    # feature).
    """Test dataset statistics for image/audio columns."""

    def test_image_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts an image column's stats have dtype "image" with minimum "32x32 px" and maximum "128x128 px".
        """Image columns show width x height stats."""
        model = stsb_bert_tiny_model
        images = [_make_pil_image(w, w) for w in [32, 64, 128]]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        info = {"name": "test"}
        result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

        assert "stats" in result
        assert "image" in result["stats"]
        assert result["stats"]["image"]["dtype"] == "image"
        assert "32x32 px" in result["stats"]["image"]["data"]["min"]
        assert "128x128 px" in result["stats"]["image"]["data"]["max"]

    def test_image_example_placeholder_without_save_dir(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts that without a save directory the examples table shows image placeholders and no img tags.
        """Without save_dir, image examples show placeholder text."""
        model = stsb_bert_tiny_model
        model.model_card_data.save_dir = None
        images = [_make_pil_image(64, 64) for _ in range(5)]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        info = {"name": "test"}
        result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

        assert "examples_table" in result
        assert "image" in result["examples_table"].lower() or "64x64" in result["examples_table"]
        assert "<img" not in result["examples_table"]

    def test_image_example_saved_with_save_dir(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts that with a save directory the examples table uses img tags referencing example_image files and
        # three such JPEG files (one per example row) are written to the assets folder.
        """With save_dir, image examples are saved as files and rendered with <img> tags."""
        model = stsb_bert_tiny_model
        # Use different sizes so they're not deduplicated
        images = [_make_pil_image(32 + i * 16, 32 + i * 16) for i in range(5)]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            info = {"name": "test"}
            result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

            assert "examples_table" in result
            assert "<img" in result["examples_table"]
            assert "example_image_0.jpg" in result["examples_table"]
            # 3 example rows, 1 image column, all distinct = 3 images saved
            for i in range(3):
                assert os.path.isfile(os.path.join(tmpdir, "assets", f"example_image_{i}.jpg"))


class TestComputeDatasetMetricsStringMediaPath:
    # [CLASS DOC]
    # Tests stats for string columns that contain media file paths.
    # KNOWN ISSUE: This class and the next lack the Pillow/datasets skip guard used by the neighbouring classes, so
    # they fail with NameError instead of skipping when those optional packages are missing.
    """Test dataset statistics for string columns containing media file paths."""

    def test_image_path_column_skips_preprocess(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Writes three JPEG files, builds a "positive" column of their paths, and with modalities text and image plus
        # a spy on the model's preprocess, asserts the column is reported as a string column of image modality with
        # empty data and that preprocess was never called.
        model = stsb_bert_tiny_model

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                p = os.path.join(tmpdir, f"img_{i}.jpg")
                _make_pil_image(32, 32).save(p, "JPEG")
                paths.append(p)

            ds = Dataset.from_dict({"positive": paths})

            preprocess_calls = []
            original_preprocess = model.preprocess

            def _tracking_preprocess(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Spy that records the call arguments and delegates to the original preprocess.
                preprocess_calls.append((args, kwargs))
                return original_preprocess(*args, **kwargs)

            info = {"name": "test"}
            with (
                patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["text", "image"]),
                patch.object(model, "preprocess", side_effect=_tracking_preprocess),
            ):
                result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

            assert "stats" in result
            assert "positive" in result["stats"]
            assert result["stats"]["positive"]["dtype"] == "string"
            assert result["stats"]["positive"]["modality"] == "image"
            assert result["stats"]["positive"]["data"] == {}
            assert preprocess_calls == [], f"model.preprocess was called on a media-path column: {preprocess_calls!r}"

    def test_text_column_still_tokenizes(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts a plain text column still gets min, mean and max token-length statistics.
        """Plain text columns still get token-length stats (no regression for the common case)."""
        model = stsb_bert_tiny_model

        ds = Dataset.from_dict({"anchor": ["hello world", "sentence transformers are great", "short"]})
        info = {"name": "test"}
        result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

        assert "stats" in result
        assert result["stats"]["anchor"]["dtype"] == "string"
        data = result["stats"]["anchor"]["data"]
        assert "min" in data and "tokens" in data["min"]
        assert "mean" in data and "tokens" in data["mean"]
        assert "max" in data and "tokens" in data["max"]


class TestComputeDatasetMetricsColumns:
    # [CLASS DOC]
    # Tests the remaining column-type branches of the private column-statistics helper.
    """Cover the remaining column-type branches in _compute_column_stats."""

    def test_int_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts an integer label column has dtype "int", empty modality, and per-value percentages (60 percent
        # zeros, 40 percent ones).
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"label": [0, 1, 0, 1, 0]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["label"]
        assert entry["dtype"] == "int"
        assert entry["modality"] == ""
        assert set(entry["data"].keys()) == {0, 1}
        assert "60" in entry["data"][0]
        assert "40" in entry["data"][1]

    def test_bool_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts a boolean column has dtype "bool", empty modality, and keys True and False.
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"flag": [True, False, True, True]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["flag"]
        assert entry["dtype"] == "bool"
        assert entry["modality"] == ""
        assert set(entry["data"].keys()) == {True, False}

    def test_float_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts a float column has dtype "float" and min/mean/max of 0.1, 0.5 and 0.9.
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"score": [0.1, 0.5, 0.9]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["score"]
        assert entry["dtype"] == "float"
        assert entry["modality"] == ""
        assert entry["data"] == {"min": 0.1, "mean": 0.5, "max": 0.9}

    def test_list_column_all_same_length(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts a list column of equal-length lists reports a single size of "3 elements".
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"embeddings": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["embeddings"]
        assert entry["dtype"] == "list"
        assert entry["modality"] == ""
        assert entry["data"] == {"size": "3 elements"}

    def test_list_column_varying_lengths(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts a list column of varying lengths reports min "1 elements", max "3 elements" and a mean containing
        # "2.00".
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"tokens": [[1], [2, 3], [4, 5, 6]]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["tokens"]
        assert entry["dtype"] == "list"
        assert entry["modality"] == ""
        assert entry["data"]["min"] == "1 elements"
        assert entry["data"]["max"] == "3 elements"
        assert "2.00" in entry["data"]["mean"]

    def test_audio_dict_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts an audio-dict column has dtype and modality "audio", min "1.00s", max "2.00s" and sampling rate
        # "16000 Hz".
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict(
            {
                "audio": [
                    {"array": [0.0] * 16000, "sampling_rate": 16000},
                    {"array": [0.0] * 32000, "sampling_rate": 16000},
                ]
            }
        )
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["audio"]
        assert entry["dtype"] == "audio"
        assert entry["modality"] == "audio"
        assert entry["data"]["min"] == "1.00s"
        assert entry["data"]["max"] == "2.00s"
        assert entry["data"]["sampling_rate"] == "16000 Hz"

    def test_tensor_video_falls_through_to_fallback(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With a video-only model and a NumPy-formatted column of 4-D arrays, asserts the stats fall back: modality
        # "video", a dtype naming the ndarray class, and empty data.
        """A 4D tensor column (video modality) isn't handled by the VideoDecoder branch,
        so it hits the fallback: dtype is the fully-qualified class name, modality is populated."""
        model = stsb_bert_tiny_model

        videos = [np.zeros((8, 3, 32, 32), dtype=np.uint8) for _ in range(3)]
        ds = Dataset.from_dict({"video": videos}).with_format("numpy")

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["video"]):
            result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())

        entry = result["stats"]["video"]
        assert entry["modality"] == "video"
        assert "ndarray" in entry["dtype"]
        assert entry["data"] == {}

    def test_modality_row_in_stats_table(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts the rendered stats table contains exactly one modality row and that it lists "text" (for the text
        # column).
        """The rendered stats_table contains a modality row alongside type and details."""
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict(
            {
                "anchor": ["hello", "world", "foo"],
                "score": [0.1, 0.5, 0.9],
            }
        )
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())

        table = result["stats_table"]
        lines = [line.strip() for line in table.splitlines() if line.strip()]
        modality_lines = [line for line in lines if line.startswith("| modality")]
        assert len(modality_lines) == 1, f"Expected one modality row, got: {modality_lines}"
        # anchor → "text", score → "" (numeric has no modality)
        assert "text" in modality_lines[0]


class TestEndToEnd:
    # [CLASS DOC]
    # End-to-end tests of full model card generation.
    """End-to-end: model card generation produces valid output."""

    def test_text_model_card_valid(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With local files only, asserts the card contains a Python code block, mentions SentenceTransformer, and has
        # no triple newlines.
        """Text model generates a valid card with no triple newlines."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model_card = generate_model_card(model)

        assert "```python" in model_card
        assert "SentenceTransformer" in model_card
        assert "\n\n\n" not in model_card

    def test_model_card_with_model_id(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts the model id appears in the card's loading line.
        """model_id flows through to the generated card."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "tomaarsen/test-model"
        model_card = generate_model_card(model)

        assert 'SentenceTransformer("tomaarsen/test-model")' in model_card

    def test_text_modality_in_card(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts a text-only model's card lists supported modality "Text".
        """Text-only model shows 'Text' in supported modalities."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.usage_examples = ["A", "B"]
        model_card = generate_model_card(model)

        assert "Supported Modality" in model_card
        assert "Text" in model_card

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_model_card_with_image_usage_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. With text-plus-image dict usage examples, a model id and a save directory, asserts the
        # card has a multimodal inputs snippet with text and image keys and that assets/image_0.jpg was written.
        """Image usage_examples causes multimodal snippet generation."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "user/multimodal-model"
        model.model_card_data.usage_examples = [
            {"text": "A cat", "image": _make_pil_image()},
            {"text": "A dog", "image": _make_pil_image()},
        ]
        # We need a save_dir for asset saving
        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            model_card = generate_model_card(model)

            # Should have multimodal snippet
            assert "inputs = [" in model_card
            assert "'text'" in model_card
            assert "'image'" in model_card
            # Assets should be saved
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_model_card_with_image_only_usage_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. With image-only usage examples, asserts the card has an inputs snippet with Hub URLs
        # for the model id.
        """Image-only usage_examples generates single-modality snippet."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "user/image-model"
        model.model_card_data.usage_examples = [_make_pil_image(), _make_pil_image()]
        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            model_card = generate_model_card(model)

            assert "inputs = [" in model_card
            assert "huggingface.co/user/image-model" in model_card


class _MockVideoMetadata:
    # [CLASS DOC]
    # HELPER: Minimal stand-in for torchcodec video metadata (path, duration in seconds, width, height, frame count,
    # average fps).
    """Minimal stand-in for ``torchcodec`` video metadata."""

    def __init__(
        self,
        path: str | None = None,
        duration_seconds: float = 2.0,
        width: int = 64,
        height: int = 64,
        num_frames: int = 8,
        average_fps: float = 24.0,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores the given metadata values (defaults: no path, 2 seconds, 64x64, 8 frames, 24 fps).
        self.path = path
        self.duration_seconds = duration_seconds
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.average_fps = average_fps


class _MockVideoDecoder:
    # [CLASS DOC]
    # HELPER: Minimal mock of torchcodec's video decoder, holding metadata, an encoded-source dict, and a frame
    # tensor; can be configured to fail on batch or single frame access.
    """Minimal mock for ``torchcodec.decoders.VideoDecoder``.

    When ``fail_batch=True``, :meth:`get_frames_at` always raises.
    When ``fail_single=True``, :meth:`__getitem__` always raises.
    """

    def __init__(
        self,
        source: str | None = None,
        *,
        metadata: _MockVideoMetadata | None = None,
        frames: torch.Tensor | None = None,
        fail_batch: bool = False,
        fail_single: bool = False,
        hf_encoded: dict | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Uses the given metadata or creates default metadata (path taken from the source if it is a string).
        # Sets the encoded-source marker (as datasets' Video feature does) to the given dict or to the metadata path
        # with no bytes. Uses the given frames or random uint8 frames of shape frame count by 3 by height by width.
        # Stores the failure flags.
        if metadata is not None:
            self.metadata = metadata
        else:
            path = source if isinstance(source, str) else None
            self.metadata = _MockVideoMetadata(path=path)
        # datasets' Video feature stamps the encoded source onto every decoder it hands out.
        self._hf_encoded = hf_encoded if hf_encoded is not None else {"path": self.metadata.path, "bytes": None}
        nf = self.metadata.num_frames
        h, w = self.metadata.height, self.metadata.width
        self._frames = frames if frames is not None else torch.randint(0, 255, (nf, 3, h, w), dtype=torch.uint8)
        self._fail_batch = fail_batch
        self._fail_single = fail_single

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # HELPER: Returns the number of frames.
        return len(self._frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Returns the frame at the index; raises RuntimeError when single-frame failure is enabled.
        if self._fail_single:
            raise RuntimeError("single frame access failed")
        return self._frames[idx]

    def get_frames_at(self, indices: list[int]):
        # [FUNCTION DOC]
        # HELPER: Returns a small result object whose data attribute holds the frames at the given indices; raises
        # RuntimeError when batch failure is enabled.
        if self._fail_batch:
            raise RuntimeError("batch decode failed")

        class _Result:
            # [CLASS DOC]
            # HELPER: Tiny container exposing the decoded frames as its data attribute.
            def __init__(self, data: torch.Tensor):
                # [FUNCTION DOC]
                # HELPER: Stores the frame tensor as data.
                self.data = data

        return _Result(self._frames[indices])


@contextmanager
def _patch_video_decoder():
    # [FUNCTION DOC]
    # HELPER: Context manager that replaces the model-card module's VideoDecoder reference with the mock decoder class
    # so isinstance checks accept mocks; restores it on exit.
    """Patch ``model_card.VideoDecoder`` so ``_MockVideoDecoder`` instances pass ``isinstance`` checks."""
    with patch("sentence_transformers.base.model_card.VideoDecoder", _MockVideoDecoder):
        yield


def _make_mock_video_decoder(**kwargs) -> _MockVideoDecoder:
    # [FUNCTION DOC]
    # HELPER: Returns a mock decoder whose metadata is built from the given keyword overrides.
    """Create a ``_MockVideoDecoder`` with custom metadata overrides."""
    return _MockVideoDecoder(metadata=_MockVideoMetadata(**kwargs))


class TestHashAssetVideoDecoder:
    # [CLASS DOC]
    # Tests the private asset-hashing helper (used for deduplication) on video decoders.
    """Test _hash_asset for VideoDecoder inputs."""

    def test_consistent_hash(self) -> None:
        # [FUNCTION DOC]
        # Asserts two mock decoders with the same path and metadata produce the same non-None hash.
        """The same encoded source produces the same hash."""
        with _patch_video_decoder():
            kwargs = dict(path="/v.mp4", duration_seconds=2.0, width=64, height=64, num_frames=8)
            dec1 = _make_mock_video_decoder(**kwargs)
            dec2 = _make_mock_video_decoder(**kwargs)
            h1 = BaseModelCardData._hash_asset(dec1)
            h2 = BaseModelCardData._hash_asset(dec2)
            assert h1 is not None
            assert h1 == h2

    def test_different_source_different_hash(self) -> None:
        # [FUNCTION DOC]
        # Asserts decoders with different source paths hash differently.
        with _patch_video_decoder():
            dec1 = _make_mock_video_decoder(path="/one.mp4")
            dec2 = _make_mock_video_decoder(path="/two.mp4")
            assert BaseModelCardData._hash_asset(dec1) != BaseModelCardData._hash_asset(dec2)

    def test_same_shape_different_bytes_are_distinct(self) -> None:
        # [FUNCTION DOC]
        # Asserts two path-less decoders with identical metadata but different encoded bytes hash differently.
        # Decoder metadata (path, duration, resolution, frame count) is identical for any two
        # in-memory clips of the same shape, so hashing it collapsed them into one asset.
        with _patch_video_decoder():
            dec1 = _MockVideoDecoder(hf_encoded={"path": None, "bytes": b"first clip"})
            dec2 = _MockVideoDecoder(hf_encoded={"path": None, "bytes": b"second clip"})
            assert dec1.metadata.__dict__ == dec2.metadata.__dict__
            assert BaseModelCardData._hash_asset(dec1) != BaseModelCardData._hash_asset(dec2)

    def test_unsourced_decoder_is_not_hashable(self) -> None:
        # [FUNCTION DOC]
        # Asserts a decoder with neither a source path nor bytes is not hashable (None).
        # Nothing to identify it by, so report None and let the caller keep both examples rather
        # than collapse every such decoder onto one hash.
        with _patch_video_decoder():
            dec = _MockVideoDecoder(hf_encoded={"path": None, "bytes": None})
            assert BaseModelCardData._hash_asset(dec) is None


class TestVideoDecoderToDict:
    # [CLASS DOC]
    # Tests the private conversion of a video decoder to a video dict with fallback strategies.
    """Test _video_decoder_to_dict fallback strategies."""

    def test_batch_decode_succeeds(self) -> None:
        # [FUNCTION DOC]
        # Asserts a successful batch decode returns a dict with an array of 4 frames and metadata with 24 fps and 4
        # total frames.
        """get_frames_at with all frame indices succeeds on first try."""
        with _patch_video_decoder():
            dec = _make_mock_video_decoder(num_frames=4, width=32, height=32)
            result = BaseModelCardData._video_decoder_to_dict(dec)
            assert "array" in result and "video_metadata" in result
            assert result["array"].shape[0] == 4
            assert result["video_metadata"]["fps"] == 24.0
            assert result["video_metadata"]["total_num_frames"] == 4

    def test_batch_n_minus_1_fallback(self) -> None:
        # [FUNCTION DOC]
        # With a decoder subclass that fails only when all frame indices are requested, asserts the conversion falls
        # back to all but the last frame (5 of 6).
        """get_frames_at(n) fails but get_frames_at(n-1) succeeds."""

        class _FailLastFrame(_MockVideoDecoder):
            # [CLASS DOC]
            # HELPER: Mock decoder whose batch decode raises when asked for every frame and otherwise delegates.
            def get_frames_at(self, indices):
                # [FUNCTION DOC]
                # HELPER: Raises RuntimeError if the index count equals the frame count, else returns the parent's
                # result.
                if len(indices) == self.metadata.num_frames:
                    raise RuntimeError("last frame broken")
                return super().get_frames_at(indices)

        with patch("sentence_transformers.base.model_card.VideoDecoder", _FailLastFrame):
            dec = _FailLastFrame(metadata=_MockVideoMetadata(num_frames=6, width=32, height=32))
            result = BaseModelCardData._video_decoder_to_dict(dec)
            assert result["array"].shape[0] == 5  # n - 1

    def test_single_frame_fallback(self) -> None:
        # [FUNCTION DOC]
        # Asserts that when batch decoding always fails, conversion falls back to single-frame access and yields all 4
        # frames.
        """All batch decodes fail, falls back to single-frame access."""
        with _patch_video_decoder():
            dec = _MockVideoDecoder(metadata=_MockVideoMetadata(num_frames=4, width=32, height=32), fail_batch=True)
            result = BaseModelCardData._video_decoder_to_dict(dec)
            assert result["array"].shape[0] == 4

    def test_recreate_from_path(self) -> None:
        # [FUNCTION DOC]
        # Asserts that when the original decoder fails entirely but has a source path, conversion recreates a decoder
        # from that path (patched to return a working mock) and succeeds with 4 frames.
        """Original decoder fails completely, fresh one from source path succeeds."""
        meta = _MockVideoMetadata(path="/fake/video.mp4", num_frames=4, width=32, height=32)
        broken = _MockVideoDecoder(metadata=meta, fail_batch=True, fail_single=True)

        working = _MockVideoDecoder(metadata=_MockVideoMetadata(num_frames=4, width=32, height=32))
        with patch("sentence_transformers.base.model_card.VideoDecoder", lambda path: working):
            result = BaseModelCardData._video_decoder_to_dict(broken)
            assert "array" in result
            assert result["array"].shape[0] == 4

    def test_all_fail_raises(self) -> None:
        # [FUNCTION DOC]
        # Asserts that when every strategy fails and there is no source path, conversion raises RuntimeError ("Could
        # not decode").
        """When all strategies fail (no source path), RuntimeError is raised."""
        with _patch_video_decoder():
            dec = _MockVideoDecoder(
                metadata=_MockVideoMetadata(num_frames=4, width=32, height=32),
                fail_batch=True,
                fail_single=True,
            )
            with pytest.raises(RuntimeError, match="Could not decode"):
                BaseModelCardData._video_decoder_to_dict(dec)


class TestPrepareForInference:
    # [CLASS DOC]
    # Tests the private preparation of example values for inference (decoders become video dicts).
    """Test _prepare_for_inference conversion of VideoDecoder objects."""

    def test_video_decoder_converted(self) -> None:
        # [FUNCTION DOC]
        # Asserts a decoder is converted to a dict with array and video metadata.
        with _patch_video_decoder():
            dec = _make_mock_video_decoder(num_frames=4, width=32, height=32)
            result = BaseModelCardData._prepare_for_inference(dec)
            assert isinstance(result, dict)
            assert "array" in result and "video_metadata" in result

    def test_dict_with_video_decoder_value(self) -> None:
        # [FUNCTION DOC]
        # Asserts decoder values inside a dict are converted while other values are kept.
        """Nested dict: VideoDecoder values are converted, others kept."""
        with _patch_video_decoder():
            dec = _make_mock_video_decoder(num_frames=4, width=32, height=32)
            result = BaseModelCardData._prepare_for_inference({"text": "hello", "video": dec})
            assert result["text"] == "hello"
            assert isinstance(result["video"], dict)
            assert "array" in result["video"]

    def test_passthrough_string(self) -> None:
        # [FUNCTION DOC]
        # Asserts strings pass through unchanged.
        assert BaseModelCardData._prepare_for_inference("hello") == "hello"

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_passthrough_pil_image(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts a PIL image passes through as the same object.
        img = _make_pil_image()
        assert BaseModelCardData._prepare_for_inference(img) is img


class TestFormatExampleValueVideoDecoder:
    # [CLASS DOC]
    # Tests table formatting of video decoder values.
    """Test _format_example_value for VideoDecoder inputs."""

    def test_video_decoder_placeholder(self) -> None:
        # [FUNCTION DOC]
        # Asserts a decoder placeholder includes "video", duration "3.50s", resolution "1920x1080" and "30fps".
        with _patch_video_decoder():
            dec = _make_mock_video_decoder(duration_seconds=3.50, width=1920, height=1080, average_fps=30.0)
            result = BaseModelCardData._format_example_value(dec)
            assert "video" in result
            assert "3.50s" in result
            assert "1920x1080" in result
            assert "30fps" in result


class TestSavePredictExampleAssetsVideoDecoder:
    # [CLASS DOC]
    # Tests saving video decoder usage examples as assets.
    """Test save_usage_example_assets and _save_asset for VideoDecoder inputs."""

    def test_video_decoder_with_source_path(self) -> None:
        # [FUNCTION DOC]
        # Asserts a decoder with a real source file has that file copied byte-for-byte to assets/video_0.mp4 and
        # recorded as display path.
        """VideoDecoder with a real source file copies it to assets/."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "source_video.mp4")
                with open(source, "wb") as f:
                    f.write(b"fake video content")

                dec = _MockVideoDecoder(metadata=_MockVideoMetadata(path=source))
                data.save_dir = tmpdir
                data.usage_examples = [dec]
                data.save_usage_example_assets()

                assert data.usage_examples_display == ["assets/video_0.mp4"]
                assert os.path.isfile(os.path.join(tmpdir, "assets", "video_0.mp4"))
                # Verify file content was copied
                with open(os.path.join(tmpdir, "assets", "video_0.mp4"), "rb") as f:
                    assert f.read() == b"fake video content"

    def test_video_decoder_preserves_extension(self) -> None:
        # [FUNCTION DOC]
        # Asserts the source file extension (.webm) is preserved in the asset name.
        """Source file extension is preserved (e.g. .webm)."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "clip.webm")
                with open(source, "wb") as f:
                    f.write(b"webm data")

                dec = _MockVideoDecoder(metadata=_MockVideoMetadata(path=source))
                data.save_dir = tmpdir
                data.usage_examples = [dec]
                data.save_usage_example_assets()

                assert data.usage_examples_display == ["assets/video_0.webm"]

    def test_video_decoder_duplicate_deduplicated(self) -> None:
        # [FUNCTION DOC]
        # Asserts two decoders sharing the same source metadata are saved once and share a display path.
        """Identical VideoDecoders (same metadata hash) are saved only once."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "source.mp4")
                with open(source, "wb") as f:
                    f.write(b"video bytes")

                meta = _MockVideoMetadata(path=source)
                dec1 = _MockVideoDecoder(metadata=meta)
                dec2 = _MockVideoDecoder(metadata=meta)

                data.save_dir = tmpdir
                data.usage_examples = [dec1, dec2]
                data.save_usage_example_assets()

                assert data.usage_examples_display[0] == data.usage_examples_display[1]
                assert len(os.listdir(os.path.join(tmpdir, "assets"))) == 1


class TestFormatAndSaveExampleVideoDecoder:
    # [CLASS DOC]
    # Tests the private example formatter that saves assets for the examples table.
    """Test _format_and_save_example for VideoDecoder inputs."""

    def test_with_save_dir_and_source_path(self) -> None:
        # [FUNCTION DOC]
        # Asserts a decoder with a save directory and source file produces a video tag with duration "1.50s",
        # increments the counter to 1, and writes assets/example_video_0.mp4.
        """VideoDecoder with save_dir and source file produces <video> tag."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "src.mp4")
                with open(source, "wb") as f:
                    f.write(b"vid")

                dec = _make_mock_video_decoder(path=source, duration_seconds=1.50)
                data.save_dir = tmpdir
                html, counter = data._format_and_save_example(dec, 0)

                assert "<video" in html
                assert "1.50s" in html
                assert counter == 1
                assert os.path.isfile(os.path.join(tmpdir, "assets", "example_video_0.mp4"))

    def test_cached_video_not_saved_again(self) -> None:
        # [FUNCTION DOC]
        # Asserts formatting the same decoder twice reuses the cached asset path: the first call increments the
        # counter, the second returns the given counter unchanged, and both reference example_video_0.
        """Second call for same VideoDecoder returns cached path without incrementing counter."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "src.mp4")
                with open(source, "wb") as f:
                    f.write(b"vid")

                meta = _MockVideoMetadata(path=source, duration_seconds=2.0)
                dec = _MockVideoDecoder(metadata=meta)
                data.save_dir = tmpdir

                html1, c1 = data._format_and_save_example(dec, 0)
                html2, c2 = data._format_and_save_example(dec, 5)

                assert c1 == 1  # incremented
                assert c2 == 5  # NOT incremented (cached)
                assert "example_video_0" in html1
                assert "example_video_0" in html2  # same cached path

    def test_without_save_dir_returns_placeholder(self) -> None:
        # [FUNCTION DOC]
        # Asserts that without a save directory a decoder yields a code placeholder mentioning video and the counter
        # stays 0.
        """Without save_dir, returns text placeholder in <code> tag."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            data.save_dir = None
            dec = _make_mock_video_decoder(duration_seconds=3.0, width=640, height=480, average_fps=30)
            html, counter = data._format_and_save_example(dec, 0)

            assert "<code>" in html
            assert "video" in html
            assert counter == 0


_can_test_video_dataset = (
    is_datasets_available() and VideoFeature is not None and _av is not None and _torchcodec is not None
)


@pytest.mark.skipif(not _can_test_video_dataset, reason="datasets Video feature or torchcodec encoder not available")
def test_hash_asset_distinguishes_real_same_shape_videos() -> None:
    # [FUNCTION DOC]
    # Skips without the datasets Video feature, PyAV or torchcodec. Encodes two different random 8-frame 64x64 H.264
    # clips in memory, loads them as a Video column, and asserts their decoders share shape metadata yet hash
    # differently, and that re-reading the same row gives the same hash.
    """Two real in-memory clips of identical shape and duration must not collapse to one asset."""

    def encode(seed: int) -> bytes:
        # [FUNCTION DOC]
        # HELPER: Encodes 8 random RGB frames (NumPy generator seeded with the given seed) as a 64x64 yuv420p H.264
        # MP4 at 10 fps into memory with PyAV and returns the bytes.
        buffer = io.BytesIO()
        with _av.open(buffer, mode="w", format="mp4") as container:
            stream = container.add_stream("h264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
            generator = np.random.default_rng(seed)
            for _ in range(8):
                frame_data = generator.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                for packet in stream.encode(_av.VideoFrame.from_ndarray(frame_data, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return buffer.getvalue()

    dataset = Dataset.from_dict({"video": [{"bytes": encode(1), "path": None}, {"bytes": encode(2), "path": None}]})
    dataset = dataset.cast_column("video", VideoFeature())
    first, second = dataset[0]["video"], dataset[1]["video"]
    assert (first.metadata.width, first.metadata.num_frames) == (second.metadata.width, second.metadata.num_frames)
    assert BaseModelCardData._hash_asset(first) != BaseModelCardData._hash_asset(second)
    # And the decoder is rebuilt per access, so the hash must survive a re-read of the same row.
    assert BaseModelCardData._hash_asset(first) == BaseModelCardData._hash_asset(dataset[0]["video"])


@contextmanager
def _create_video_dataset(n: int = 5):
    # [FUNCTION DOC]
    # HELPER: Context manager that writes n (default 5) synthetic 4-frame H.264 MP4 files with growing resolution
    # (32+16i square) and frame rate (10+5i) into a temporary directory, yields a Dataset whose "video" column of
    # those paths is cast to the datasets Video feature, and removes the directory afterwards. Requires PyAV and
    # datasets.
    """Create a small Dataset with a Video feature column from synthetic video files."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        paths = []
        for i in range(n):
            w, h = 32 + i * 16, 32 + i * 16
            num_frames = 4
            fps = 10 + i * 5
            frames = torch.randint(0, 255, (num_frames, h, w, 3), dtype=torch.uint8).numpy()
            path = os.path.join(tmpdir, f"video_{i}.mp4")
            with _av.open(path, mode="w") as container:
                stream = container.add_stream("h264", rate=fps)
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"
                for frame_data in frames:
                    frame = _av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            paths.append(path)
        ds = Dataset.from_dict({"video": paths})
        ds = ds.cast_column("video", VideoFeature())
        yield ds


@pytest.mark.skipif(not _can_test_video_dataset, reason="datasets Video feature or torchcodec encoder not available")
class TestSetMultimodalPredictExampleVideo:
    # [CLASS DOC]
    # Tests multimodal usage-example selection from Video columns (requires datasets Video, PyAV and torchcodec).
    """Test _set_multimodal_usage_examples with Video feature columns."""

    def test_video_column_detected(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # With a video-only model, asserts three usage examples are chosen and all are torchcodec video decoders.
        """A model supporting 'video' modality picks VideoDecoder examples from Video columns."""
        from torchcodec.decoders import VideoDecoder

        with _create_video_dataset(n=5) as ds:
            model = stsb_bert_tiny_model
            dd = DatasetDict(train=ds)
            with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["video"]):
                model.model_card_data._set_multimodal_usage_examples(dd)

            assert model.model_card_data.usage_examples is not None
            assert len(model.model_card_data.usage_examples) == 3
            assert all(isinstance(v, VideoDecoder) for v in model.model_card_data.usage_examples)


@pytest.mark.skipif(not _can_test_video_dataset, reason="datasets Video feature or torchcodec encoder not available")
class TestComputeDatasetMetricsVideo:
    # [CLASS DOC]
    # Tests dataset statistics for Video columns (same requirements).
    """Test compute_dataset_metrics with VideoDecoder columns."""

    def test_video_decoder_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Asserts a video column reports dtype "video" with min, max, mean and fps entries.
        """VideoDecoder columns produce duration/resolution/fps stats."""
        with _create_video_dataset(n=5) as ds:
            model = stsb_bert_tiny_model
            info = {"name": "test"}
            result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

            assert "stats" in result
            assert "video" in result["stats"]
            assert result["stats"]["video"]["dtype"] == "video"
            data = result["stats"]["video"]["data"]
            assert "min" in data and "max" in data and "mean" in data
            assert "fps" in data


class TestSaveAssetVideoDecoderNoSourcePath:
    # [CLASS DOC]
    # Tests saving a path-less video decoder by decoding and re-encoding it.
    """Test _save_asset when VideoDecoder has no source path (decode + encode fallback)."""

    @pytest.mark.skipif(_av is None, reason="av (PyAV) not available")
    def test_video_decoder_without_source_path_decoded_and_saved(self) -> None:
        # [FUNCTION DOC]
        # Skips without PyAV. Asserts a mock decoder without a source path is decoded and encoded to
        # assets/video_0.mp4 and recorded.
        """VideoDecoder with path=None falls back to decode → VideoDict → av encoding."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                dec = _MockVideoDecoder(
                    metadata=_MockVideoMetadata(path=None, num_frames=4, width=32, height=32, average_fps=10.0)
                )
                data.save_dir = tmpdir
                data.usage_examples = [dec]
                data.save_usage_example_assets()

                assert data.usage_examples_display == ["assets/video_0.mp4"]
                assert os.path.isfile(os.path.join(tmpdir, "assets", "video_0.mp4"))


@pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
class TestGenerateModelCardUrlRewriting:
    # [CLASS DOC]
    # Tests that generated model cards rewrite relative asset paths into absolute Hub URLs (skipped without Pillow).
    """Test that generate_model_card rewrites asset src paths to absolute Hub URLs."""

    def test_asset_src_urls_rewritten(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Injects training dataset info containing an image example and a relative img source, generates the card with
        # a model id and save directory, and asserts no relative asset sources remain and the Hub resolve-main URL is
        # used instead.
        """Training example table asset paths are rewritten to absolute Hub URLs."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "user/my-model"
        # Inject training dataset info with an image example that will be re-rendered
        # during to_dict() (normally set by Trainer during training)
        img = _make_pil_image(32, 32)
        model.model_card_data.train_datasets = [
            {
                "name": "test",
                "columns": ["<code>image</code>"],
                "stats_table": "",
                "examples": {"image": [img]},
                "examples_table": '<img src="assets/placeholder.jpg">',
                "_example_columns": ["image"],
                "loss": {"fullname": "test.FakeLoss"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            model_card = generate_model_card(model)

            # Relative src="assets/..." should have been rewritten
            assert 'src="assets/' not in model_card
            assert 'src="https://huggingface.co/user/my-model/resolve/main/assets/' in model_card


def _make_ir_evaluator():
    # [FUNCTION DOC]
    # HELPER: Returns a minimal InformationRetrievalEvaluator with one query, one document and one relevant pair.
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

    return InformationRetrievalEvaluator(
        queries={"q1": "query"},
        corpus={"d1": "document"},
        relevant_docs={"q1": {"d1"}},
    )


def _make_similarity_evaluator():
    # [FUNCTION DOC]
    # HELPER: Returns a minimal EmbeddingSimilarityEvaluator with one sentence pair.
    from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

    return EmbeddingSimilarityEvaluator(sentences1=["a"], sentences2=["b"], scores=[0.5])


def _make_nano_beir_evaluator():
    # [FUNCTION DOC]
    # HELPER: Returns a NanoBEIREvaluator instance created without running its constructor (avoids dataset downloads;
    # only useful for type checks).
    from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator

    # Bypass __init__ to skip downloading NanoBEIR datasets. We only need the class for isinstance.
    return NanoBEIREvaluator.__new__(NanoBEIREvaluator)


def _make_sparse_ir_evaluator():
    # [FUNCTION DOC]
    # HELPER: Returns a minimal SparseInformationRetrievalEvaluator (an IR evaluator subclass).
    from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator

    # Subclass of InformationRetrievalEvaluator: exercises the isinstance path.
    return SparseInformationRetrievalEvaluator(
        queries={"q1": "query"},
        corpus={"d1": "document"},
        relevant_docs={"q1": {"d1"}},
    )


def _make_sequential_with_ir_evaluator():
    # [FUNCTION DOC]
    # HELPER: Returns a SequentialEvaluator wrapping a similarity evaluator and an IR evaluator.
    from sentence_transformers.base.evaluation import SequentialEvaluator

    return SequentialEvaluator([_make_similarity_evaluator(), _make_ir_evaluator()])


class TestTryToSetIrModelFromEvaluators:
    # [CLASS DOC]
    # Tests the lazy detection that marks the model card as an information-retrieval model when evaluation results
    # come from IR evaluators.
    """Lazy IR-model fallback that fires when evaluators reveal IR intent."""

    @pytest.mark.parametrize(
        "evaluator_factory",
        [
            _make_ir_evaluator,
            _make_nano_beir_evaluator,
            _make_sparse_ir_evaluator,
            _make_sequential_with_ir_evaluator,
        ],
        ids=["information_retrieval", "nano_beir", "sparse_ir_subclass", "sequential_with_ir"],
    )
    def test_ir_evaluator_sets_ir_model(self, evaluator_factory) -> None:
        # [FUNCTION DOC]
        # Parametrised over IR, NanoBEIR, sparse IR and a sequential evaluator containing IR. Asserts recording a
        # result for it and running the detection sets the IR flag to True.
        data = _make_model_card_data()
        data.eval_results_dict[evaluator_factory()] = {"ndcg@10": 0.5}
        data.try_to_set_ir_model_from_evaluators()
        assert data.ir_model is True

    def test_non_ir_evaluator_does_not_set_ir_model(self) -> None:
        # [FUNCTION DOC]
        # Asserts a similarity evaluator alone leaves the IR flag None.
        data = _make_model_card_data()
        data.eval_results_dict[_make_similarity_evaluator()] = {"spearman_cosine": 0.7}
        data.try_to_set_ir_model_from_evaluators()
        assert data.ir_model is None

    def test_to_dict_skips_when_already_set(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Sets the IR flag to True, records an IR evaluator result, runs the card's dict conversion, and asserts the
        # flag is still True.
        # KNOWN ISSUE: Because the recorded evaluator is an IR evaluator, re-running detection would also yield True,
        # so this test cannot detect whether an already-set flag gets overwritten.
        """to_dict guards the call so an already-set ir_model is never overwritten."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.ir_model = True
        model.model_card_data.eval_results_dict[_make_ir_evaluator()] = {"ndcg@10": 0.5}

        model.model_card_data.to_dict()

        assert model.model_card_data.ir_model is True

    def test_to_dict_triggers_lazy_detection(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        # [FUNCTION DOC]
        # Clears the IR flag, sets query/document usage examples and records an IR evaluator result, runs the dict
        # conversion, and asserts the flag becomes True and the usage snippet uses the encode-query and
        # encode-document calls.
        """End-to-end: to_dict() runs the fallback so the snippet uses encode_query/encode_document."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.ir_model = None
        model.model_card_data.usage_examples = ["q?", "d1", "d2"]
        model.model_card_data.eval_results_dict[_make_ir_evaluator()] = {"ndcg@10": 0.5}

        result = model.model_card_data.to_dict()

        assert model.model_card_data.ir_model is True
        assert "model.encode_query(queries)" in result["usage_snippet"]
        assert "model.encode_document(documents)" in result["usage_snippet"]
