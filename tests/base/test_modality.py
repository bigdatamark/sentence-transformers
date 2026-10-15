# [MODULE DOC]
# Pytest unit tests for the modality utilities: URL/path detectors for images, videos and audio; per-sample and
# per-batch modality inference (strings, arrays, tensors, PIL images, chat messages, media dicts and multimodal
# dicts); the modality-to-processor-argument mapping; and the InputFormatter, which infers flat versus structured chat
# message format from a processor's chat template, converts inputs to chat messages, normalises messages, parses
# batches into processor inputs, and handles query/document pair roles. Further tests below cover chat-template
# rendering and pair handling with fake processors. Requires numpy, torch, Pillow, jinja2 and transformers.
from __future__ import annotations

import numpy as np
import pytest
import torch
from jinja2 import Environment
from PIL import Image
from transformers.video_utils import make_batched_videos

from sentence_transformers.base.modality import (
    InputFormatter,
    _is_non_text_pair,
    infer_batch_modality,
    infer_modality,
    is_audio_url_or_path,
    is_image_url_or_path,
    is_video_url_or_path,
)
from sentence_transformers.base.modality_types import MODALITY_TO_PROCESSOR_ARG
from tests.utils import GENERIC_INSTRUCT_TEMPLATE, RERANKER_TEMPLATE


class TestIsImageUrlOrPath:
    # [CLASS DOC]
    # Tests image URL/path detection.
    def test_https_jpg(self):
        # [FUNCTION DOC]
        # Asserts an https URL ending in .jpg is an image.
        assert is_image_url_or_path("https://example.com/photo.jpg") is True

    def test_https_with_query_params(self):
        # [FUNCTION DOC]
        # Asserts an image URL with query parameters is an image.
        assert is_image_url_or_path("https://cdn.example.com/photo.jpg?width=200&token=abc") is True

    def test_https_with_fragment(self):
        # [FUNCTION DOC]
        # Asserts an image URL with a fragment is an image.
        assert is_image_url_or_path("https://example.com/photo.png#section") is True

    def test_http_url(self):
        # [FUNCTION DOC]
        # Asserts an http .webp URL is an image.
        assert is_image_url_or_path("http://example.com/photo.webp") is True

    def test_data_uri(self):
        # [FUNCTION DOC]
        # Asserts an image data URI is an image.
        assert is_image_url_or_path("data:image/png;base64,iVBOR") is True

    def test_plain_text_not_image(self):
        # [FUNCTION DOC]
        # Asserts plain text is not an image.
        assert is_image_url_or_path("hello world") is False

    def test_url_without_image_extension(self):
        # [FUNCTION DOC]
        # Asserts a URL with a non-image extension is not an image.
        assert is_image_url_or_path("https://example.com/page.html") is False

    def test_local_file_exists(self, tmp_path):
        # [FUNCTION DOC]
        # Asserts an existing local .jpg file path is an image.
        img_file = tmp_path / "test.jpg"
        img_file.write_text("fake image")
        assert is_image_url_or_path(str(img_file)) is True

    def test_local_file_not_exists(self):
        # [FUNCTION DOC]
        # Asserts a non-existent local .jpg path is not an image.
        assert is_image_url_or_path("/nonexistent/path/photo.jpg") is False

    def test_empty_string(self):
        # [FUNCTION DOC]
        # Asserts the empty string is not an image.
        assert is_image_url_or_path("") is False

    def test_case_insensitive_extension(self):
        # [FUNCTION DOC]
        # Asserts extension matching is case insensitive.
        assert is_image_url_or_path("https://example.com/PHOTO.JPG") is True

    def test_url_with_spaces_not_image(self):
        # [FUNCTION DOC]
        # Asserts an image URL followed by spaces and text is not an image.
        assert is_image_url_or_path("https://example.com/photo.jpg this is a sentence") is False

    def test_malformed_url_crash(self):
        # [FUNCTION DOC]
        # Asserts malformed URLs (unclosed brackets) return False instead of raising.
        """Malformed URLs (e.g. containing unclosed brackets) should not crash."""
        assert is_image_url_or_path("https://[[ooga-booga<<<<") is False
        assert is_image_url_or_path("https://www.google.com)[google.com]") is False


class TestIsVideoUrlOrPath:
    # [CLASS DOC]
    # Tests video URL/path detection.
    def test_https_mp4(self):
        # [FUNCTION DOC]
        # Asserts an https .mp4 URL is a video.
        assert is_video_url_or_path("https://example.com/video.mp4") is True

    def test_https_with_query_params(self):
        # [FUNCTION DOC]
        # Asserts a video URL with query parameters is a video.
        assert is_video_url_or_path("https://cdn.example.com/clip.mp4?token=abc") is True

    def test_youtube_www(self):
        # [FUNCTION DOC]
        # Asserts a www YouTube watch URL is a video.
        assert is_video_url_or_path("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_youtube_short(self):
        # [FUNCTION DOC]
        # Asserts a youtu.be short URL is a video.
        assert is_video_url_or_path("https://youtu.be/dQw4w9WgXcQ") is True

    def test_youtube_mobile(self):
        # [FUNCTION DOC]
        # Asserts a mobile YouTube URL is a video.
        assert is_video_url_or_path("https://m.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_plain_text_not_video(self):
        # [FUNCTION DOC]
        # Asserts plain text is not a video.
        assert is_video_url_or_path("hello world") is False

    def test_empty_string(self):
        # [FUNCTION DOC]
        # Asserts the empty string is not a video.
        assert is_video_url_or_path("") is False

    def test_youtube_url_with_spaces_not_video(self):
        # [FUNCTION DOC]
        # Asserts a YouTube URL followed by a sentence is not a video.
        # Text that starts with a YouTube URL but contains spaces (i.e. is really a sentence)
        text = "http://youtu.be/YYPc7CRL39o - Love Song, The Butterflys sing You Are So Beautiful"
        assert is_video_url_or_path(text) is False

    def test_video_url_with_spaces_not_video(self):
        # [FUNCTION DOC]
        # Asserts a video URL followed by extra text is not a video.
        assert is_video_url_or_path("https://example.com/video.mp4 some extra text") is False

    def test_malformed_url_crash(self):
        # [FUNCTION DOC]
        # Asserts malformed URLs return False instead of raising.
        """Malformed URLs (e.g. containing unclosed brackets) should not crash."""
        assert is_video_url_or_path("https://[[ooga-booga<<<<") is False
        assert is_video_url_or_path("https://www.google.com)[google.com]") is False


class TestIsAudioUrlOrPath:
    # [CLASS DOC]
    # Tests audio URL/path detection.
    def test_https_mp3(self):
        # [FUNCTION DOC]
        # Asserts an https .mp3 URL is audio.
        assert is_audio_url_or_path("https://example.com/clip.mp3") is True

    def test_https_with_query_params(self):
        # [FUNCTION DOC]
        # Asserts an audio URL with query parameters is audio.
        assert is_audio_url_or_path("https://cdn.example.com/clip.wav?token=abc") is True

    def test_plain_text_not_audio(self):
        # [FUNCTION DOC]
        # Asserts plain text is not audio.
        assert is_audio_url_or_path("hello world") is False

    def test_empty_string(self):
        # [FUNCTION DOC]
        # Asserts the empty string is not audio.
        assert is_audio_url_or_path("") is False

    def test_local_file_exists(self, tmp_path):
        # [FUNCTION DOC]
        # Asserts an existing local .mp3 file path is audio.
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("fake audio")
        assert is_audio_url_or_path(str(audio_file)) is True

    def test_audio_url_with_spaces_not_audio(self):
        # [FUNCTION DOC]
        # Asserts an audio URL followed by text is not audio.
        assert is_audio_url_or_path("https://example.com/clip.mp3 some description") is False

    def test_malformed_url_crash(self):
        # [FUNCTION DOC]
        # Asserts malformed URLs return False instead of raising.
        """Malformed URLs (e.g. containing unclosed brackets) should not crash."""
        assert is_audio_url_or_path("https://[[ooga-booga<<<<") is False
        assert is_audio_url_or_path("https://www.google.com)[google.com]") is False


class TestInferModality:
    # [CLASS DOC]
    # Tests single-sample modality inference.
    @pytest.mark.parametrize("plural", ["images", "videos", "audios", "texts"])
    def test_plural_keys_rejected_without_mutating_input(self, plural):
        # [FUNCTION DOC]
        # Parametrised over plural keys (images, videos, audios, texts). Asserts a dict with that key raises
        # ValueError about unrecognized modality keys and the input dict is not mutated.
        sample = {plural: ["a", "b"]}
        with pytest.raises(ValueError, match="unrecognized modality keys"):
            infer_modality(sample)
        assert sample == {plural: ["a", "b"]}

    def test_plain_text(self):
        # [FUNCTION DOC]
        # Asserts plain text infers as "text".
        assert infer_modality("hello world") == "text"

    def test_text_pair_tuple(self):
        # [FUNCTION DOC]
        # Asserts a tuple of two strings infers as "text" (a text pair).
        assert infer_modality(("query", "document")) == "text"

    def test_text_pair_list(self):
        # [FUNCTION DOC]
        # Asserts a list of two strings infers as "text".
        assert infer_modality(["query", "document"]) == "text"

    def test_image_https_url(self):
        # [FUNCTION DOC]
        # Asserts an image URL infers as "image".
        assert infer_modality("https://example.com/photo.jpg") == "image"

    def test_audio_https_url(self):
        # [FUNCTION DOC]
        # Asserts an audio URL infers as "audio".
        assert infer_modality("https://example.com/clip.mp3") == "audio"

    def test_video_https_url(self):
        # [FUNCTION DOC]
        # Asserts a video URL infers as "video".
        assert infer_modality("https://example.com/video.mp4") == "video"

    def test_malformed_url_does_not_crash(self):
        # [FUNCTION DOC]
        # Asserts malformed URLs infer as "text" without raising.
        """Malformed URLs like bare brackets should return text instead of crashing."""
        assert infer_modality("https://[[ooga-booga<<<<") == "text"
        assert infer_modality("https://www.google.com)[google.com]") == "text"

    def test_pil_image(self):
        # [FUNCTION DOC]
        # Asserts a PIL image infers as "image".
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        assert infer_modality(img) == "image"

    def test_ndarray_1d_is_audio(self):
        # [FUNCTION DOC]
        # Asserts a 1-D NumPy array infers as "audio".
        assert infer_modality(np.zeros(16000)) == "audio"

    def test_ndarray_2d_is_audio(self):
        # [FUNCTION DOC]
        # Asserts a 2-D NumPy array infers as "audio" (multi-channel).
        assert infer_modality(np.zeros((2, 16000))) == "audio"

    def test_ndarray_3d_is_image(self):
        # [FUNCTION DOC]
        # Asserts a 3-D NumPy array infers as "image".
        assert infer_modality(np.zeros((224, 224, 3))) == "image"

    def test_ndarray_4d_is_video(self):
        # [FUNCTION DOC]
        # Asserts a 4-D NumPy array infers as "video".
        assert infer_modality(np.zeros((8, 3, 224, 224))) == "video"

    def test_ndarray_5d_is_video(self):
        # [FUNCTION DOC]
        # Asserts a 5-D NumPy array infers as "video".
        assert infer_modality(np.zeros((1, 8, 3, 224, 224))) == "video"

    def test_tensor_1d_is_audio(self):
        # [FUNCTION DOC]
        # Asserts a 1-D tensor infers as "audio".
        assert infer_modality(torch.zeros(16000)) == "audio"

    def test_tensor_3d_is_image(self):
        # [FUNCTION DOC]
        # Asserts a 3-D tensor infers as "image".
        assert infer_modality(torch.zeros(3, 224, 224)) == "image"

    def test_tensor_4d_is_video(self):
        # [FUNCTION DOC]
        # Asserts a 4-D tensor infers as "video".
        assert infer_modality(torch.zeros(8, 3, 224, 224)) == "video"

    def test_dict_chat_message(self):
        # [FUNCTION DOC]
        # Asserts a role/content dict infers as "message".
        msg = {"role": "user", "content": "hello"}
        assert infer_modality(msg) == "message"

    def test_list_of_chat_messages(self):
        # [FUNCTION DOC]
        # Asserts a list of role/content dicts infers as "message".
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        assert infer_modality(msgs) == "message"

    def test_dict_audio_dataset_format(self):
        # [FUNCTION DOC]
        # Asserts a dict with array and sampling rate infers as "audio".
        audio = {"array": np.zeros(16000), "sampling_rate": 16000}
        assert infer_modality(audio) == "audio"

    def test_dict_video_with_metadata(self):
        # [FUNCTION DOC]
        # Asserts a dict with array and video metadata infers as "video".
        video = {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 30}}
        assert infer_modality(video) == "video"

    def test_dict_array_without_sampling_rate_raises(self):
        # [FUNCTION DOC]
        # Asserts a dict with only a 1-D array raises ValueError asking for sampling rate or video metadata.
        with pytest.raises(ValueError, match="must also include 'sampling_rate'.*or 'video_metadata'"):
            infer_modality({"array": np.zeros(16000)})

    def test_dict_array_without_video_metadata_raises(self):
        # [FUNCTION DOC]
        # Asserts a dict with only a 4-D array raises the same ValueError.
        with pytest.raises(ValueError, match="must also include 'sampling_rate'.*or 'video_metadata'"):
            infer_modality({"array": np.zeros((8, 3, 224, 224))})

    def test_multimodal_dict_returns_sorted_tuple(self):
        # [FUNCTION DOC]
        # Asserts an image+text dict infers as the alphabetically sorted tuple of modalities.
        # Keys in insertion order: image before text, must still return sorted tuple
        sample = {"image": "cat.jpg", "text": "a photo"}
        assert infer_modality(sample) == ("image", "text")

    def test_multimodal_dict_sorting_is_consistent(self):
        # [FUNCTION DOC]
        # Asserts the inferred tuple does not depend on dict key order.
        # Both orderings should produce the same modality tuple
        sample_a = {"text": "a photo", "image": "cat.jpg"}
        sample_b = {"image": "cat.jpg", "text": "a photo"}
        assert infer_modality(sample_a) == infer_modality(sample_b)

    def test_single_key_multimodal_dict_collapses_to_bare_modality(self):
        # [FUNCTION DOC]
        # Asserts single-key modality dicts infer as that bare modality (image, text, audio, video).
        # A 1-key multimodal dict should be equivalent to a raw value of that modality.
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        assert infer_modality({"image": img}) == "image"
        assert infer_modality({"text": "hello"}) == "text"
        assert infer_modality({"audio": np.zeros(16000)}) == "audio"
        assert infer_modality({"video": np.zeros((8, 3, 224, 224))}) == "video"

    def test_unrecognized_modality_key_raises_without_metadata_hint(self):
        # [FUNCTION DOC]
        # Asserts a dict with an unknown key raises ValueError naming the key and the valid keys, without the
        # array-form hint.
        # A genuinely unknown key reports the invalid key and the valid modality keys, and does not
        # add the array-dict hint (which is reserved for the video_metadata/sampling_rate mistake).
        with pytest.raises(ValueError) as exc_info:
            infer_modality({"image": "cat.jpg", "foo": 1})
        message = str(exc_info.value)
        assert "unrecognized modality keys" in message
        assert "'foo'" in message
        assert "To attach" not in message

    def test_unrecognized_modality_keys_of_mixed_types_still_raise_valueerror(self):
        # [FUNCTION DOC]
        # Asserts unknown keys of mixed types still raise ValueError (not TypeError from sorting).
        # The reported keys are sorted for deterministic output, but keys of mixed types are not
        # mutually comparable, so a plain sort would raise TypeError and mask the actionable error.
        with pytest.raises(ValueError, match="unrecognized modality keys"):
            infer_modality({"image": "cat.jpg", "foo": 1, 0: 2})

    def test_dict_array_with_mixed_key_types_still_raises_valueerror(self):
        # [FUNCTION DOC]
        # Asserts an array dict with a non-string key still raises the sampling-rate ValueError.
        # As above, for the sibling 'array' error that also sorts the reported keys.
        with pytest.raises(ValueError, match="must also include 'sampling_rate'"):
            infer_modality({"array": np.zeros(16000), 0: 2})

    def test_sibling_video_metadata_key_hints_array_form(self):
        # [FUNCTION DOC]
        # Asserts a video dict with a sibling video metadata key raises ValueError naming that key and suggesting the
        # nested array/video-metadata form and disabling frame sampling (issue 3864).
        # video_metadata as a sibling key is a common mistake (issue #3864): the error should name the
        # offending key and point to the {"video": {"array": ..., "video_metadata": ...}} form.
        with pytest.raises(ValueError) as exc_info:
            infer_modality({"video": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 30}})
        message = str(exc_info.value)
        assert "['video_metadata']" in message
        assert '{"video": {"array": frames, "video_metadata"' in message
        assert 'processing_kwargs={"video": {"do_sample_frames": False}}' in message

    def test_sibling_sampling_rate_key_hints_array_form(self):
        # [FUNCTION DOC]
        # Asserts an audio dict with a sibling sampling rate key raises ValueError suggesting the nested audio form,
        # without mentioning video metadata.
        # As above for audio. Hints are picked per modality, so it must not recite the video form.
        with pytest.raises(ValueError) as exc_info:
            infer_modality({"audio": np.zeros(16000), "sampling_rate": 16000})
        message = str(exc_info.value)
        assert "['sampling_rate']" in message
        assert '{"audio": {"array": waveform, "sampling_rate": 16000}}' in message
        assert "video_metadata" not in message

    def test_both_sibling_metadata_keys_hint_both_forms(self):
        # [FUNCTION DOC]
        # Asserts a dict with both sibling mistakes lists both keys and gives both hints.
        # Neither modality wins arbitrarily: a sample carrying both mistakes gets both hints.
        with pytest.raises(ValueError) as exc_info:
            infer_modality(
                {
                    "audio": np.zeros(16000),
                    "sampling_rate": 16000,
                    "video": np.zeros((8, 3, 224, 224)),
                    "video_metadata": {"fps": 30},
                }
            )
        message = str(exc_info.value)
        assert "['sampling_rate', 'video_metadata']" in message
        assert '{"video": {"array": frames, "video_metadata"' in message
        assert '{"audio": {"array": waveform, "sampling_rate": 16000}}' in message

    def test_unsupported_type_raises(self):
        # [FUNCTION DOC]
        # Asserts an integer raises ValueError "Unsupported input type".
        with pytest.raises(ValueError, match="Unsupported input type"):
            infer_modality(12345)

    def test_tensor_bad_ndim_raises(self):
        # [FUNCTION DOC]
        # Asserts a 6-D tensor raises ValueError about unsupported dimensionality.
        with pytest.raises(ValueError, match="Unsupported tensor dimensionality"):
            infer_modality(torch.zeros(2, 3, 4, 5, 6, 7))

    def test_image_url_falls_back_to_text_when_unsupported(self):
        # [FUNCTION DOC]
        # Asserts an image URL infers as "text" when only text is supported.
        assert infer_modality("https://example.com/photo.jpg", supported_modalities=["text"]) == "text"

    def test_video_url_falls_back_to_text_when_unsupported(self):
        # [FUNCTION DOC]
        # Asserts a YouTube URL infers as "text" when only text is supported.
        assert infer_modality("https://youtu.be/dQw4w9WgXcQ", supported_modalities=["text"]) == "text"

    def test_audio_url_falls_back_to_text_when_unsupported(self):
        # [FUNCTION DOC]
        # Asserts an audio URL infers as "text" when only text is supported.
        assert infer_modality("https://example.com/clip.mp3", supported_modalities=["text"]) == "text"

    def test_image_url_detected_when_supported(self):
        # [FUNCTION DOC]
        # Asserts an image URL infers as "image" when image is supported.
        assert infer_modality("https://example.com/photo.jpg", supported_modalities=["text", "image"]) == "image"

    def test_video_url_detected_when_supported(self):
        # [FUNCTION DOC]
        # Asserts a YouTube URL infers as "video" when video is supported.
        assert infer_modality("https://youtu.be/dQw4w9WgXcQ", supported_modalities=["text", "video"]) == "video"

    def test_pil_image_not_affected_by_supported_modalities(self):
        # [FUNCTION DOC]
        # Asserts a PIL image infers as "image" regardless of supported modalities.
        """Non-string inputs are unambiguous and should not be overridden by supported_modalities."""
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        assert infer_modality(img, supported_modalities=["text"]) == "image"

    def test_ndarray_not_affected_by_supported_modalities(self):
        # [FUNCTION DOC]
        # Asserts NumPy arrays keep their audio/image/video inference regardless of supported modalities.
        """Non-string inputs are unambiguous and should not be overridden by supported_modalities."""
        assert infer_modality(np.zeros(16000), supported_modalities=["text"]) == "audio"
        assert infer_modality(np.zeros((224, 224, 3)), supported_modalities=["text"]) == "image"
        assert infer_modality(np.zeros((8, 3, 224, 224)), supported_modalities=["text"]) == "video"

    def test_audio_url_detected_when_supported(self):
        # [FUNCTION DOC]
        # Asserts an audio URL infers as "audio" when audio is supported.
        assert infer_modality("https://example.com/clip.mp3", supported_modalities=["text", "audio"]) == "audio"

    def test_youtube_url_with_trailing_text_is_text(self):
        # [FUNCTION DOC]
        # Asserts a YouTube URL followed by a description infers as "text".
        """The exact case from the bug report: a YouTube URL followed by a description."""
        text = (
            "http://youtu.be/YYPc7CRL39o - Love Song, The Butterflys sing You Are So Beautiful "
            "by Joe Cocker featuring singer/ songwriter Justin Nelson."
        )
        assert infer_modality(text) == "text"

    def test_data_uri_not_affected_by_space_check(self):
        # [FUNCTION DOC]
        # Asserts an image data URI infers as "image" (exempt from the space check).
        """data: URIs bypass the URL space check and should still be detected as images."""
        assert infer_modality("data:image/png;base64,iVBOR") == "image"

    def test_url_with_encoded_spaces_still_detected(self):
        # [FUNCTION DOC]
        # Asserts URLs with percent-encoded spaces are still detected as image, video and audio.
        """URLs with %20 (encoded spaces) are valid and should still be detected."""
        assert infer_modality("https://example.com/my%20photo.jpg") == "image"
        assert infer_modality("https://example.com/my%20video.mp4") == "video"
        assert infer_modality("https://example.com/my%20clip.mp3") == "audio"


class TestInferBatchModality:
    # [CLASS DOC]
    # Tests batch modality inference.
    def test_homogeneous_text(self):
        # [FUNCTION DOC]
        # Asserts a batch of strings infers as "text".
        assert infer_batch_modality(["hello", "world"]) == "text"

    def test_homogeneous_images_ndarray(self):
        # [FUNCTION DOC]
        # Asserts a batch of 3-D arrays infers as "image".
        batch = [np.zeros((224, 224, 3)), np.zeros((224, 224, 3))]
        assert infer_batch_modality(batch) == "image"

    def test_homogeneous_audio_ndarray(self):
        # [FUNCTION DOC]
        # Asserts a batch of 1-D arrays infers as "audio".
        batch = [np.zeros(16000), np.zeros(16000)]
        assert infer_batch_modality(batch) == "audio"

    def test_mixed_text_and_image_returns_message(self):
        # [FUNCTION DOC]
        # Asserts a batch mixing text and a PIL image infers as "message".
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        batch = ["some text", img]
        assert infer_batch_modality(batch) == "message"

    def test_mixed_text_and_audio_returns_message(self):
        # [FUNCTION DOC]
        # Asserts a batch mixing text and audio infers as "message".
        batch = ["some text", np.zeros(16000)]
        assert infer_batch_modality(batch) == "message"

    def test_homogeneous_multimodal_dicts(self):
        # [FUNCTION DOC]
        # Asserts a batch of image+text dicts infers as the sorted tuple.
        batch = [
            {"image": "cat.jpg", "text": "a cat"},
            {"image": "dog.jpg", "text": "a dog"},
        ]
        assert infer_batch_modality(batch) == ("image", "text")

    def test_single_item_batch(self):
        # [FUNCTION DOC]
        # Asserts a single-text batch infers as "text".
        assert infer_batch_modality(["hello"]) == "text"

    def test_video_url_in_text_batch_stays_text_when_unsupported(self):
        # [FUNCTION DOC]
        # Asserts a text batch containing a YouTube URL stays "text" when video is unsupported.
        """A batch of texts with one video URL should remain 'text' if video is not supported."""
        batch = ["hello world", "https://youtu.be/dQw4w9WgXcQ", "another text"]
        assert infer_batch_modality(batch, supported_modalities=["text"]) == "text"

    def test_video_url_in_text_batch_becomes_message_when_supported(self):
        # [FUNCTION DOC]
        # Asserts the same batch becomes "message" when video is supported.
        """A batch of texts with one video URL should become 'message' if video is supported."""
        batch = ["hello world", "https://youtu.be/dQw4w9WgXcQ", "another text"]
        assert infer_batch_modality(batch, supported_modalities=["text", "video"]) == "message"


class TestModalityToProcessorArg:
    # [CLASS DOC]
    # Tests the mapping from modality names to processor argument names.
    def test_contains_expected_modalities(self):
        # [FUNCTION DOC]
        # Asserts the mapping keys are exactly text, image, audio, video and message.
        assert set(MODALITY_TO_PROCESSOR_ARG.keys()) == {"text", "image", "audio", "video", "message"}

    def test_maps_to_processor_arg_names(self):
        # [FUNCTION DOC]
        # Asserts image maps to "images", video to "videos", and text, audio and message to themselves.
        assert MODALITY_TO_PROCESSOR_ARG["image"] == "images"
        assert MODALITY_TO_PROCESSOR_ARG["video"] == "videos"
        assert MODALITY_TO_PROCESSOR_ARG["text"] == "text"
        assert MODALITY_TO_PROCESSOR_ARG["audio"] == "audio"
        assert MODALITY_TO_PROCESSOR_ARG["message"] == "message"


class TestInferModalityPilUnavailable:
    # [CLASS DOC]
    # Tests inference when Pillow is unavailable (the modality module's PIL reference monkeypatched to None).
    def test_pil_unavailable_text_still_works(self, monkeypatch):
        # [FUNCTION DOC]
        # Asserts text still infers as "text".
        monkeypatch.setattr("sentence_transformers.base.modality.PILImage", None)
        assert infer_modality("hello world") == "text"

    def test_pil_unavailable_image_url_still_works(self, monkeypatch):
        # [FUNCTION DOC]
        # Asserts an image URL still infers as "image".
        monkeypatch.setattr("sentence_transformers.base.modality.PILImage", None)
        assert infer_modality("https://example.com/photo.jpg") == "image"


class TestInputFormatterInit:
    # [CLASS DOC]
    # Tests InputFormatter construction and message-format resolution.
    def test_explicit_structured(self):
        # [FUNCTION DOC]
        # Asserts an explicit structured format is kept.
        fmt = InputFormatter(model_type="test", message_format="structured")
        assert fmt.message_format == "structured"

    def test_explicit_flat(self):
        # [FUNCTION DOC]
        # Asserts an explicit flat format is kept.
        fmt = InputFormatter(model_type="test", message_format="flat")
        assert fmt.message_format == "flat"

    def test_auto_without_processor_defaults_to_structured(self):
        # [FUNCTION DOC]
        # Asserts "auto" without a processor resolves to structured.
        fmt = InputFormatter(model_type="test", message_format="auto")
        assert fmt.message_format == "structured"

    def test_auto_with_processor_infers_format(self):
        # [FUNCTION DOC]
        # Asserts "auto" with a processor whose template reads message content directly resolves to flat.
        class FakeProcessor:
            # [CLASS DOC]
            # HELPER: Fake processor with a flat chat template.
            chat_template = "Hello {{ message.content }}"

        fmt = InputFormatter(model_type="test", message_format="auto", processor=FakeProcessor())
        assert fmt.message_format == "flat"


class TestInferFormat:
    # [CLASS DOC]
    # Tests the private format inference from a model type and processor.
    def test_known_model_type(self):
        # [FUNCTION DOC]
        # Asserts the known model type "apertus" infers flat without a processor.
        fmt = InputFormatter(model_type="apertus", message_format="structured")
        assert fmt._infer_format(None) == "flat"

    def test_structured_template_with_type_pattern(self):
        # [FUNCTION DOC]
        # Asserts a template iterating content items and reading their type infers structured.
        class FakeProcessor:
            # [CLASS DOC]
            # HELPER: Fake processor with a structured chat template.
            chat_template = "{% for item in message.content %}{{ item.type }}{% endfor %}"

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(FakeProcessor()) == "structured"

    def test_flat_template_without_structured_patterns(self):
        # [FUNCTION DOC]
        # Asserts a template without structured patterns infers flat.
        class FakeProcessor:
            # [CLASS DOC]
            # HELPER: Fake processor with a flat chat template.
            chat_template = "{{ message.content }}"

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(FakeProcessor()) == "flat"

    def test_no_chat_template(self):
        # [FUNCTION DOC]
        # Asserts a processor without a chat template infers structured.
        class FakeProcessor:
            # [CLASS DOC]
            # HELPER: Fake processor with no chat template attribute.
            pass

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(FakeProcessor()) == "structured"

    def test_chat_template_non_string(self):
        # [FUNCTION DOC]
        # Asserts a non-string (dict) chat template infers structured.
        class FakeProcessor:
            # [CLASS DOC]
            # HELPER: Fake processor whose chat template is a dict.
            chat_template = {"default": "some template"}

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(FakeProcessor()) == "structured"


class TestToMessage:
    # [CLASS DOC]
    # Tests converting a modality dict into a one-message conversation.
    def test_flat_single_text(self):
        # [FUNCTION DOC]
        # Asserts flat format turns a text dict into a user message with string content.
        fmt = InputFormatter(model_type="test", message_format="flat")
        result = fmt.to_message({"text": "hello"})
        assert result == [{"role": "user", "content": "hello"}]

    def test_flat_multimodal_falls_back_to_structured(self):
        # [FUNCTION DOC]
        # Asserts flat format with multiple modalities falls back to structured list content.
        fmt = InputFormatter(model_type="test", message_format="flat")
        result = fmt.to_message({"text": "hello", "image": "cat.jpg"})
        # Falls back to structured when multiple modalities
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)

    def test_structured_single_modality(self):
        # [FUNCTION DOC]
        # Asserts structured format wraps text in a typed content item.
        fmt = InputFormatter(model_type="test", message_format="structured")
        result = fmt.to_message({"text": "hello"})
        assert result == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    def test_structured_multimodal(self):
        # [FUNCTION DOC]
        # Asserts structured multimodal input yields one message with a text item and an image item.
        fmt = InputFormatter(model_type="test", message_format="structured")
        result = fmt.to_message({"text": "a cat", "image": "cat.jpg"})
        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        types = {item["type"] for item in content}
        assert types == {"text", "image"}

    def test_structured_multimodal_preserves_dict_ordering(self):
        # [FUNCTION DOC]
        # Asserts content items follow the input dict's key order in both orders.
        """Content items should follow the input dict's key order, not alphabetical."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        # "text" before "image" in the dict
        result = fmt.to_message({"text": "a cat", "image": "cat.jpg"})
        content = result[0]["content"]
        assert [item["type"] for item in content] == ["text", "image"]

        # Reversed: "image" before "text"
        result = fmt.to_message({"image": "cat.jpg", "text": "a cat"})
        content = result[0]["content"]
        assert [item["type"] for item in content] == ["image", "text"]

    def test_custom_role(self):
        # [FUNCTION DOC]
        # Asserts a custom role ("system") is used.
        fmt = InputFormatter(model_type="test", message_format="flat")
        result = fmt.to_message({"text": "hello"}, role="system")
        assert result == [{"role": "system", "content": "hello"}]


class TestNormalizeMessages:
    # [CLASS DOC]
    # Tests normalising chat messages to the formatter's format.
    def test_flat_to_structured(self):
        # [FUNCTION DOC]
        # Asserts flat string content is converted to a typed text item in structured format.
        fmt = InputFormatter(model_type="test", message_format="structured")
        messages = [{"role": "user", "content": "hello"}]
        result = fmt.normalize_messages(messages)
        assert result == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    def test_structured_to_flat_single_text(self):
        # [FUNCTION DOC]
        # Asserts a single typed text item is flattened to a string in flat format.
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = fmt.normalize_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_structured_to_flat_multi_item_keeps_structured(self):
        # [FUNCTION DOC]
        # Asserts multi-item content cannot be flattened and the messages are returned unchanged.
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "image": "cat.jpg"},
                ],
            }
        ]
        result = fmt.normalize_messages(messages)
        # Can't flatten multi-item, keeps original
        assert result == messages

    def test_already_in_target_format(self):
        # [FUNCTION DOC]
        # Asserts messages already in flat format are unchanged.
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [{"role": "user", "content": "hello"}]
        result = fmt.normalize_messages(messages)
        assert result == messages

    def test_invalid_message_skipped(self):
        # [FUNCTION DOC]
        # Asserts entries without role/content are dropped.
        fmt = InputFormatter(model_type="test", message_format="structured")
        messages = [{"invalid": "message"}, {"role": "user", "content": "hello"}]
        result = fmt.normalize_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_non_string_flat_content_kept_as_is(self):
        # [FUNCTION DOC]
        # Asserts non-string flat content (a PIL image) is kept unchanged in structured mode.
        fmt = InputFormatter(model_type="test", message_format="structured")
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        messages = [{"role": "user", "content": img}]
        result = fmt.normalize_messages(messages)
        assert result == messages


class TestParseInputs:
    # [CLASS DOC]
    # Tests parsing an input batch into (modality, processor inputs dict, extra processor kwargs).
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Creates a structured-format InputFormatter on the test instance before each test.
        self.fmt = InputFormatter(model_type="test", message_format="structured")

    def test_text_inputs(self):
        # [FUNCTION DOC]
        # Asserts a text batch gives modality text, inputs under "text", and no extras.
        modality, inputs, extra = self.fmt.parse_inputs(["hello", "world"])
        assert modality == "text"
        assert inputs == {"text": ["hello", "world"]}
        assert dict(extra) == {}

    def test_text_pair_inputs(self):
        # [FUNCTION DOC]
        # Asserts text pairs give modality text with the pairs under "text".
        modality, inputs, extra = self.fmt.parse_inputs([("q1", "d1"), ("q2", "d2")])
        assert modality == "text"
        assert inputs == {"text": [("q1", "d1"), ("q2", "d2")]}

    def test_image_url_inputs(self):
        # [FUNCTION DOC]
        # Asserts image URLs give modality image with the URLs under "image".
        urls = ["https://example.com/a.jpg", "https://example.com/b.png"]
        modality, inputs, extra = self.fmt.parse_inputs(urls)
        assert modality == "image"
        assert inputs == {"image": urls}

    def test_audio_ndarray_inputs(self):
        # [FUNCTION DOC]
        # Asserts 1-D arrays give modality audio with two audio entries.
        samples = [np.zeros(16000), np.zeros(16000)]
        modality, inputs, extra = self.fmt.parse_inputs(samples)
        assert modality == "audio"
        assert inputs["audio"] is not None
        assert len(inputs["audio"]) == 2

    def test_video_ndarray_inputs(self):
        # [FUNCTION DOC]
        # Asserts 4-D arrays give modality video with two video entries.
        samples = [np.zeros((8, 3, 224, 224)), np.zeros((8, 3, 224, 224))]
        modality, inputs, extra = self.fmt.parse_inputs(samples)
        assert modality == "video"
        assert len(inputs["video"]) == 2

    def test_audio_dict_unwraps_array_and_collects_sampling_rate(self):
        # [FUNCTION DOC]
        # Asserts audio dicts are unwrapped to their arrays and the shared sampling rate is collected into the audio
        # extras.
        audio_dicts = [
            {"array": np.zeros(16000), "sampling_rate": 16000},
            {"array": np.zeros(16000), "sampling_rate": 16000},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(audio_dicts)
        assert modality == "audio"
        assert len(inputs["audio"]) == 2
        assert extra["audio"]["sampling_rate"] == 16000

    def test_video_dict_unwraps_array_and_collects_metadata(self):
        # [FUNCTION DOC]
        # Asserts video dicts are unwrapped and their metadata collected, in order, into the video extras.
        video_dicts = [
            {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 30}},
            {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 24}},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(video_dicts)
        assert modality == "video"
        assert len(inputs["video"]) == 2
        assert extra["video"]["video_metadata"] == [{"fps": 30}, {"fps": 24}]

    def test_video_metadata_stays_aligned_when_mixing_wrapped_and_raw(self):
        # [FUNCTION DOC]
        # Regression for issue 3874: with a raw 4-frame video followed by a wrapped 6-frame video with metadata,
        # asserts two metadata entries where the raw video gets inferred defaults (4 frames, fps None, frame indices 0
        # to 3) and the wrapped video keeps its own metadata.
        """A raw video before a dict-wrapped one must not shift the metadata onto the wrong video.

        https://github.com/huggingface/sentence-transformers/issues/3874
        """
        raw = np.zeros((4, 3, 8, 8))
        wrapped = {"array": np.ones((6, 3, 8, 8)), "video_metadata": {"fps": 15, "total_num_frames": 6}}
        modality, inputs, extra = self.fmt.parse_inputs([raw, wrapped])
        assert modality == "video"
        assert len(inputs["video"]) == 2
        metadata = extra["video"]["video_metadata"]
        assert len(metadata) == 2
        # The raw video gets inferred defaults, mirroring what transformers builds without metadata
        assert metadata[0]["total_num_frames"] == 4
        assert metadata[0]["fps"] is None
        assert metadata[0]["frames_indices"] == [0, 1, 2, 3]
        # The explicit metadata stays with the video it was attached to
        assert metadata[1] == {"fps": 15, "total_num_frames": 6}

    @pytest.mark.parametrize("wrapped_frames", [False, True])
    def test_video_metadata_filled_for_frame_list_video(self, wrapped_frames):
        # [FUNCTION DOC]
        # Parametrised over whether a list of frames is wrapped with null metadata. Asserts metadata is inferred for
        # the frame-list video (4 frames, indices 0 to 3) and the wrapped video's metadata is preserved.
        frames = [np.zeros((3, 8, 8)) for _ in range(4)]
        wrapped = {"array": np.ones((6, 3, 8, 8)), "video_metadata": {"fps": 15, "total_num_frames": 6}}
        sample = {"array": frames, "video_metadata": None} if wrapped_frames else {"video": frames}
        modality, inputs, extra = self.fmt.parse_inputs([sample, wrapped])
        assert modality == "video"
        metadata = extra["video"]["video_metadata"]
        assert metadata[0]["total_num_frames"] == 4
        assert metadata[0]["frames_indices"] == [0, 1, 2, 3]
        assert metadata[1] == {"fps": 15, "total_num_frames": 6}

    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("frame_type", ["pil", "numpy", "torch"])
    @pytest.mark.parametrize("wrapped_frames", [False, True])
    def test_frames_stay_one_video_in_processor(self, container, frame_type, wrapped_frames):
        # [FUNCTION DOC]
        # Parametrised over list/tuple container, PIL/NumPy/torch frames, and wrapping. Asserts parsing two 4x4 frames
        # yields one video that transformers' batching helper turns into a single 2x4x4x3 video.
        frames = [Image.new("RGB", (4, 4)) for _ in range(2)]
        if frame_type == "numpy":
            frames = [np.array(frame) for frame in frames]
        elif frame_type == "torch":
            frames = [torch.from_numpy(np.array(frame)) for frame in frames]
        sample = (
            {"array": container(frames), "video_metadata": None} if wrapped_frames else {"video": container(frames)}
        )
        _, inputs, _ = self.fmt.parse_inputs([sample])
        videos = make_batched_videos(inputs["video"])
        assert len(videos) == 1
        assert videos[0].shape == (2, 4, 4, 3)

    def test_video_metadata_key_dropped_when_no_sample_carries_it(self):
        # [FUNCTION DOC]
        # Asserts that when no sample carries video metadata, no extras are returned.
        samples = [np.zeros((8, 3, 224, 224)), np.zeros((8, 3, 224, 224))]
        modality, inputs, extra = self.fmt.parse_inputs(samples)
        assert modality == "video"
        assert dict(extra) == {}

    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("wrapped_frames", [False, True])
    def test_video_collection_metadata_matches_each_content_item(self, container, wrapped_frames):
        # [FUNCTION DOC]
        # Parametrised over container and wrapping. A sample holds a collection of a raw 2-frame video and a 3-frame
        # frame list, followed by a wrapped 4-frame video with metadata. After parsing and converting to messages,
        # asserts three video content items of 2, 3 and 4 frames align with three metadata entries (frame counts 2, 3,
        # 4; the frame list gets indices 0 to 2; the explicit metadata object is reused as is) and the input sample is
        # not mutated.
        raw = np.zeros((2, 3, 8, 8))
        frames = [np.zeros((3, 8, 8)) for _ in range(3)]
        metadata = {"fps": 15, "total_num_frames": 4}
        wrapped = {"array": np.zeros((4, 3, 8, 8)), "video_metadata": metadata}
        frame_video = {"array": frames, "video_metadata": None} if wrapped_frames else frames
        samples = [
            {"video": container([raw, frame_video])},
            wrapped,
        ]
        modality, inputs, extra = self.fmt.parse_inputs(samples)
        _, result = self.fmt.batch_to_message(modality, inputs)
        videos = [item["video"] for messages in result["message"] for item in messages[0]["content"]]
        entries = extra["video"]["video_metadata"]
        assert len(videos) == len(entries) == 3
        assert [len(video) for video in videos] == [2, 3, 4]
        assert [entry["total_num_frames"] for entry in entries] == [2, 3, 4]
        assert entries[1]["frames_indices"] == [0, 1, 2]
        assert entries[2] is metadata
        assert samples[0]["video"][1] is frame_video

    @pytest.mark.parametrize("container", [list, tuple])
    def test_audio_collection_unwraps_each_recording(self, container):
        # [FUNCTION DOC]
        # Parametrised over container. Asserts a collection of two audio dicts in one sample becomes two audio content
        # items with the raw arrays and sampling rate 16000, and that changing one recording's rate to 8000 makes
        # parsing raise ValueError about sampling rates.
        arrays = [np.zeros(8), np.zeros(16)]
        samples = [{"audio": container({"array": array, "sampling_rate": 16000} for array in arrays)}]
        modality, inputs, extra = self.fmt.parse_inputs(samples)
        _, result = self.fmt.batch_to_message(modality, inputs)
        content = result["message"][0][0]["content"]
        assert [item["audio"] for item in content] == arrays
        assert extra["audio"]["sampling_rate"] == 16000
        samples[0]["audio"][1]["sampling_rate"] = 8000
        with pytest.raises(ValueError, match="sampling rates"):
            self.fmt.parse_inputs(samples)

    def test_video_metadata_with_metadata_less_path_raises(self):
        # [FUNCTION DOC]
        # Asserts mixing a video URL (no metadata possible) with a wrapped video carrying metadata raises ValueError
        # saying metadata is needed for every video.
        wrapped = {"array": np.ones((4, 3, 8, 8)), "video_metadata": {"fps": 15, "total_num_frames": 4}}
        with pytest.raises(ValueError, match="every video"):
            self.fmt.parse_inputs(["https://example.com/a.mp4", wrapped])

    def test_conflicting_sampling_rates_raise(self):
        # [FUNCTION DOC]
        # Regression for issue 3874: asserts audio dicts with different sampling rates raise ValueError.
        """https://github.com/huggingface/sentence-transformers/issues/3874"""
        audio_dicts = [
            {"array": np.zeros(1600), "sampling_rate": 16000},
            {"array": np.zeros(4410), "sampling_rate": 44100},
        ]
        with pytest.raises(ValueError, match="sampling rates"):
            self.fmt.parse_inputs(audio_dicts)

    def test_matching_sampling_rates_do_not_raise(self):
        # [FUNCTION DOC]
        # Asserts audio dicts with equal sampling rates parse without error and report that rate.
        audio_dicts = [
            {"array": np.zeros(1600), "sampling_rate": 16000},
            {"array": np.zeros(3200), "sampling_rate": 16000},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(audio_dicts)
        assert modality == "audio"
        assert extra["audio"]["sampling_rate"] == 16000

    def test_message_dict_inputs(self):
        # [FUNCTION DOC]
        # Asserts single message dicts parse as modality message, each wrapped in its own one-message list.
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(messages)
        assert modality == "message"
        # Single message dicts get wrapped in lists
        assert inputs["message"] == [[messages[0]], [messages[1]]]

    def test_multimodal_dict_inputs(self):
        # [FUNCTION DOC]
        # Asserts image+text dicts parse as the sorted tuple with per-modality lists.
        dicts = [
            {"image": "cat.jpg", "text": "a cat"},
            {"image": "dog.jpg", "text": "a dog"},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(dicts)
        assert modality == ("image", "text")
        assert inputs["image"] == ["cat.jpg", "dog.jpg"]
        assert inputs["text"] == ["a cat", "a dog"]

    def test_multimodal_dict_inputs_preserves_key_ordering(self):
        # [FUNCTION DOC]
        # Asserts the per-modality output keys follow the input dicts' key order, while the modality tuple stays
        # sorted.
        """Output dict key order should match the input dicts' key order, not the modality tuple."""
        # Input dicts have "text" before "image"
        dicts = [
            {"text": "a cat", "image": "cat.jpg"},
            {"text": "a dog", "image": "dog.jpg"},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(dicts)
        # Modality tuple is sorted alphabetically by infer_modality
        assert modality == ("image", "text")
        # But the output dict keys should preserve the input ordering: text first
        assert list(inputs.keys()) == ["text", "image"]

        # Reversed input ordering: "image" before "text"
        dicts = [
            {"image": "cat.jpg", "text": "a cat"},
            {"image": "dog.jpg", "text": "a dog"},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(dicts)
        assert list(inputs.keys()) == ["image", "text"]

    @pytest.mark.parametrize("supported_modalities", [None, ["message"], [("image", "text"), "message"]])
    @pytest.mark.parametrize("collections", [False, True])
    def test_multimodal_batch_preserves_each_samples_key_order(self, supported_modalities, collections):
        # [FUNCTION DOC]
        # Parametrised over the formatter's supported modalities and whether values are collections. With two samples
        # having opposite key orders, asserts the batch becomes "message" and each sample's content items follow its
        # own key order (text then image, and image then text), without mutating the inputs.
        fmt = InputFormatter(model_type="test", message_format="structured", supported_modalities=supported_modalities)
        texts = ["text_a", "text_b"] if collections else "text_a"
        images = ["image_a", "image_b"] if collections else "image_a"
        samples = [{"text": texts, "image": images}, {"image": images, "text": texts}]
        modality, inputs, _ = fmt.parse_inputs(samples)
        assert modality == "message"
        text_content = [{"type": "text", "text": text} for text in (texts if collections else [texts])]
        image_content = [{"type": "image", "image": image} for image in (images if collections else [images])]
        assert inputs["message"] == [
            [{"role": "user", "content": text_content + image_content}],
            [{"role": "user", "content": image_content + text_content}],
        ]
        assert [list(sample) for sample in samples] == [["text", "image"], ["image", "text"]]

    def test_different_key_orders_keep_native_processor_arguments(self):
        # [FUNCTION DOC]
        # With image+text supported natively, asserts samples with different key orders still parse as the tuple
        # modality with plain per-modality lists.
        fmt = InputFormatter(model_type="test", supported_modalities=[("image", "text")])
        modality, inputs, _ = fmt.parse_inputs(
            [
                {"text": "a cat", "image": "cat.jpg"},
                {"image": "dog.jpg", "text": "a dog"},
            ]
        )
        assert modality == ("image", "text")
        assert inputs == {"text": ["a cat", "a dog"], "image": ["cat.jpg", "dog.jpg"]}

    def test_multimodal_dict_audio_only_wrapper_raw_array(self):
        # [FUNCTION DOC]
        # Asserts an audio dict wrapper around a raw array parses as bare audio with the same array object and no
        # extras.
        """A ``{"audio": array}`` wrapper with a raw array should behave like bare audio inputs."""
        arr = np.zeros(16000)
        modality, inputs, extra = self.fmt.parse_inputs([{"audio": arr}])
        assert modality == "audio"
        assert len(inputs["audio"]) == 1
        assert inputs["audio"][0] is arr
        assert dict(extra) == {}

    def test_single_key_image_dict_unwraps(self):
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts an image dict wrapper parses as bare image with the same object.
        """A ``{"image": pil}`` wrapper should behave like a bare PIL image input."""
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        modality, inputs, extra = self.fmt.parse_inputs([{"image": img}])
        assert modality == "image"
        assert len(inputs["image"]) == 1
        assert inputs["image"][0] is img
        assert dict(extra) == {}

    def test_single_key_text_dict_unwraps(self):
        # [FUNCTION DOC]
        # Asserts a text dict wrapper parses as bare text.
        """A ``{"text": "hello"}`` wrapper should behave like a bare text input."""
        modality, inputs, extra = self.fmt.parse_inputs([{"text": "hello"}])
        assert modality == "text"
        assert inputs == {"text": ["hello"]}
        assert dict(extra) == {}

    def test_multimodal_dict_unwraps_nested_audio_dict(self):
        # [FUNCTION DOC]
        # Asserts a nested audio dict (array and sampling rate) under the audio key is unwrapped and its rate
        # collected.
        """A ``{"audio": {"array": ..., "sampling_rate": ...}}`` wrapper should unwrap the nested dict."""
        arr = np.zeros(16000)
        modality, inputs, extra = self.fmt.parse_inputs([{"audio": {"array": arr, "sampling_rate": 16000}}])
        assert modality == "audio"
        assert len(inputs["audio"]) == 1
        assert inputs["audio"][0] is arr
        assert extra["audio"]["sampling_rate"] == 16000

    def test_multimodal_dict_unwraps_nested_video_dict(self):
        # [FUNCTION DOC]
        # Asserts a nested video dict under the video key is unwrapped and its metadata collected.
        """A ``{"video": {"array": ..., "video_metadata": ...}}`` wrapper should unwrap the nested dict."""
        arr = np.zeros((8, 3, 224, 224))
        modality, inputs, extra = self.fmt.parse_inputs([{"video": {"array": arr, "video_metadata": {"fps": 30}}}])
        assert modality == "video"
        assert len(inputs["video"]) == 1
        assert inputs["video"][0] is arr
        assert extra["video"]["video_metadata"] == [{"fps": 30}]

    def test_multimodal_dict_unwraps_audio_alongside_text(self):
        # [FUNCTION DOC]
        # Asserts a nested audio dict combined with text gives the audio+text tuple, the unwrapped array, the text,
        # and the sampling rate.
        """Audio should unwrap even when combined with other modalities in the same dict."""
        arr = np.zeros(16000)
        modality, inputs, extra = self.fmt.parse_inputs(
            [{"audio": {"array": arr, "sampling_rate": 16000}, "text": "describe this"}]
        )
        assert modality == ("audio", "text")
        assert inputs["audio"][0] is arr
        assert inputs["text"] == ["describe this"]
        assert extra["audio"]["sampling_rate"] == 16000

    def test_multimodal_dict_unwraps_video_alongside_text(self):
        # [FUNCTION DOC]
        # Asserts a nested video dict combined with text gives the text+video tuple, the unwrapped array, the text,
        # and the metadata.
        """Video metadata should survive alongside other modalities, as the #3864 error message advises."""
        arr = np.zeros((8, 3, 224, 224))
        modality, inputs, extra = self.fmt.parse_inputs(
            [{"video": {"array": arr, "video_metadata": {"fps": 30}}, "text": "describe this"}]
        )
        assert modality == ("text", "video")
        assert inputs["video"][0] is arr
        assert inputs["text"] == ["describe this"]
        assert extra["video"]["video_metadata"] == [{"fps": 30}]

    def test_multimodal_dict_unwraps_audio_decoder(self):
        # [FUNCTION DOC]
        # Skips without torchcodec. Builds a one-second silent 16 kHz WAV in memory, wraps a torchcodec audio decoder
        # over it in an audio dict, and asserts parsing yields a NumPy array and sampling rate 16000.
        """A ``{"audio": AudioDecoder}`` wrapper should unwrap the decoder into a raw array."""
        pytest.importorskip("torchcodec")
        import io
        import wave

        from torchcodec.decoders import AudioDecoder

        sample_rate = 16000
        pcm = np.zeros(sample_rate, dtype=np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(pcm.tobytes())
        decoder = AudioDecoder(torch.frombuffer(buf.getvalue(), dtype=torch.uint8))

        modality, inputs, extra = self.fmt.parse_inputs([{"audio": decoder}])
        assert modality == "audio"
        assert isinstance(inputs["audio"][0], np.ndarray)
        assert extra["audio"]["sampling_rate"] == sample_rate

    def test_multimodal_dict_unwraps_video_decoder(self):
        # [FUNCTION DOC]
        # Skips without torchcodec or PyAV (or H.264 encoding support). Encodes a 4-frame 32x32 black MP4 in memory,
        # wraps a torchcodec video decoder in a video dict, and asserts parsing yields 4 frames and metadata with 4
        # total frames.
        """A ``{"video": VideoDecoder}`` wrapper should unwrap the decoder and collect metadata."""
        pytest.importorskip("torchcodec")
        av = pytest.importorskip("av")
        import io

        from torchcodec.decoders import VideoDecoder

        num_frames, fps, height, width = 4, 30, 32, 32
        buf = io.BytesIO()
        try:
            with av.open(buf, mode="w", format="mp4") as container:
                stream = container.add_stream("h264", rate=fps)
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                for _ in range(num_frames):
                    frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        except Exception as exc:
            pytest.skip(f"H.264 encoding support is unavailable in this PyAV/FFmpeg build: {exc}")
        decoder = VideoDecoder(torch.frombuffer(buf.getvalue(), dtype=torch.uint8))

        modality, inputs, extra = self.fmt.parse_inputs([{"video": decoder}])
        assert modality == "video"
        assert inputs["video"][0].shape[0] == num_frames
        assert extra["video"]["video_metadata"][0]["total_num_frames"] == num_frames

    def test_mixed_modalities_batch_to_message(self):
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts a batch of text and an image parses as message with two conversations.
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        mixed = ["some text", img]
        modality, inputs, extra = self.fmt.parse_inputs(mixed)
        assert modality == "message"
        assert "message" in inputs
        assert len(inputs["message"]) == 2

    def test_mixed_single_and_multimodal_dict(self):
        # [FUNCTION DOC]
        # Asserts a batch of an image URL and an image+text dict parses as message: the first conversation has one
        # image content item, the second has image and text items.
        """Image + text+image dict should produce valid structured messages for both."""
        inputs = [
            "https://example.com/cat.jpg",  # image URL
            {"image": "https://example.com/dog.jpg", "text": "a dog"},  # multimodal dict
        ]
        modality, result, extra = self.fmt.parse_inputs(inputs)
        assert modality == "message"
        assert len(result["message"]) == 2

        # First: image-only message
        img_msg = result["message"][0]
        assert len(img_msg) == 1
        assert img_msg[0]["content"] == [{"type": "image", "image": "https://example.com/cat.jpg"}]

        # Second: multimodal message with both image and text content entries
        multi_msg = result["message"][1]
        assert len(multi_msg) == 1
        content = multi_msg[0]["content"]
        assert len(content) == 2
        content_types = {item["type"] for item in content}
        assert content_types == {"image", "text"}

    def test_returns_modality_keyed_dict_not_processor_arg_keyed(self):
        # [FUNCTION DOC]
        # Skips without Pillow. Asserts parsed inputs are keyed by modality name ("image"), not processor argument
        # name ("images").
        """Verify parse_inputs uses modality names as keys, not processor arg names."""
        PIL = pytest.importorskip("PIL.Image")
        imgs = [PIL.new("RGB", (10, 10)), PIL.new("RGB", (10, 10))]
        modality, inputs, extra = self.fmt.parse_inputs(imgs)
        assert modality == "image"
        # Key should be "image", not "images"
        assert "image" in inputs
        assert "images" not in inputs

    def test_image_url_classified_as_text_when_unsupported(self):
        # [FUNCTION DOC]
        # With only text supported, asserts an image URL among texts keeps the batch as text with all strings under
        # "text".
        """parse_inputs should respect supported_modalities from InputFormatter."""
        fmt = InputFormatter(model_type="test", message_format="structured", supported_modalities=["text"])
        image_url = "https://example.com/photo.jpg"
        modality, inputs, extra = fmt.parse_inputs(["hello", image_url, "world"])
        assert modality == "text"
        assert inputs == {"text": ["hello", image_url, "world"]}


class TestBatchToMessage:
    # [CLASS DOC]
    # Tests converting parsed processor inputs into chat messages.
    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("frame_type", ["pil", "numpy", "torch", "path", "url"])
    @pytest.mark.parametrize("paired", [False, True])
    def test_video_frames_stay_one_content_item(self, container, frame_type, paired, tmp_path):
        # [FUNCTION DOC]
        # Parametrised over container, frame type (PIL, NumPy, torch, file path, URL) and pairing with a query.
        # Asserts a text+video sample whose video is a list of frames becomes one conversation whose last message has
        # a text item then a single video item holding the very same frame objects in a list, the input sample is
        # unchanged, and no extras are produced.
        fmt = InputFormatter(model_type="test", message_format="structured")
        paths = [str(tmp_path / "frame_a.png"), str(tmp_path / "frame_b.jpg")]
        if frame_type == "path":
            for path in paths:
                Image.new("RGB", (4, 4)).save(path)
        frames = {
            "pil": [Image.new("RGB", (4, 4)) for _ in range(2)],
            "numpy": [np.zeros((4, 4, 3)) for _ in range(2)],
            "torch": [torch.zeros((3, 4, 4)) for _ in range(2)],
            "path": paths,
            "url": ["https://example.com/frame_a.png", "https://example.com/frame_b.jpg"],
        }[frame_type]
        value = container(frames)
        sample = {"text": "Describe this clip", "video": value}
        modality, inputs, extra = fmt.parse_inputs([("query", sample)] if paired else [sample])
        if modality != "message":
            _, inputs = fmt.batch_to_message(modality, inputs)
        messages = inputs["message"][0]
        assert len(messages) == (2 if paired else 1)
        content = messages[-1]["content"]
        assert [item["type"] for item in content] == ["text", "video"]
        assert isinstance(content[1]["video"], list)
        assert len(content[1]["video"]) == len(frames)
        assert all(actual is expected for actual, expected in zip(content[1]["video"], frames))
        assert sample["video"] is value
        assert not extra

    @pytest.mark.parametrize("container", [list, tuple])
    def test_video_frame_detection_checks_every_item(self, container):
        # [FUNCTION DOC]
        # Parametrised over container. Asserts a video collection holding a single frame and a full video is treated
        # as two separate videos (frame detection checks every item).
        fmt = InputFormatter(model_type="test", message_format="structured")
        frame = np.zeros((3, 4, 4))
        video = np.zeros((2, 3, 4, 4))
        modality, inputs, _ = fmt.parse_inputs([{"video": container([frame, video])}])
        _, inputs = fmt.batch_to_message(modality, inputs)
        content = inputs["message"][0][0]["content"]
        assert len(content) == 2
        assert content[0]["video"] is frame
        assert content[1]["video"] is video

    def test_ambiguous_frame_urls_can_be_wrapped(self):
        # [FUNCTION DOC]
        # Asserts ambiguous extension-less frame URLs given directly are treated as separate videos, while the same
        # list wrapped under an array key is one video.
        fmt = InputFormatter(model_type="test", message_format="structured")
        paths = ["https://example.com/frame/1", "https://example.com/frame/2"]
        samples = [{"video": paths}, {"video": {"array": paths}}]
        modality, inputs, _ = fmt.parse_inputs(samples)
        _, inputs = fmt.batch_to_message(modality, inputs)
        contents = [messages[0]["content"] for messages in inputs["message"]]
        assert contents[0] == [{"type": "video", "video": path} for path in paths]
        assert contents[1] == [{"type": "video", "video": paths}]

    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("text_count", [2, 4])
    @pytest.mark.parametrize("images_first", [False, True])
    @pytest.mark.parametrize("paired", [False, True])
    def test_text_and_image_collections_preserve_content_order(self, container, text_count, images_first, paired):
        # [FUNCTION DOC]
        # Parametrised over container, text count, key order and pairing. Asserts text and image collections expand
        # into content items following the sample's key order, with roles query/document when paired or user
        # otherwise.
        fmt = InputFormatter(model_type="test", message_format="structured")
        texts = container(f"text {index}" for index in range(text_count))
        images = container(Image.new("RGB", (2, 2)) for _ in range(2))
        sample = {"image": images, "text": texts} if images_first else {"text": texts, "image": images}
        modality, inputs, _ = fmt.parse_inputs([("query", sample)] if paired else [sample])
        if modality != "message":
            _, inputs = fmt.batch_to_message(modality, inputs)

        text_content = [{"type": "text", "text": text} for text in texts]
        image_content = [{"type": "image", "image": image} for image in images]
        expected = image_content + text_content if images_first else text_content + image_content
        assert len(inputs["message"]) == 1
        messages = inputs["message"][0]
        assert [message["role"] for message in messages] == (["query", "document"] if paired else ["user"])
        assert messages[-1]["content"] == expected

    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("frame_container", [list, tuple])
    @pytest.mark.parametrize("wrapped_frames,paired", [(False, False), (True, False), (False, True)])
    def test_multiple_videos_and_frame_list_preserve_sample_boundaries(
        self, container, frame_container, wrapped_frames, paired
    ):
        # [FUNCTION DOC]
        # Parametrised over containers, frame wrapping and pairing. Asserts a sample with two frame-list videos and a
        # sample with one keep their boundaries: the first message lists text then both videos, the second text then
        # one video, with role document when paired or user otherwise.
        fmt = InputFormatter(model_type="test", message_format="structured")
        frames_a = frame_container(Image.new("RGB", (2, 2)) for _ in range(2))
        frames_b = frame_container(Image.new("RGB", (2, 2)) for _ in range(3))
        video_a = {"array": frames_a, "video_metadata": None} if wrapped_frames else frames_a
        video_b = {"array": frames_b, "video_metadata": None} if wrapped_frames else frames_b
        samples = [
            {"text": "two videos", "video": container([video_a, video_b])},
            {"text": "one video", "video": video_a},
        ]
        detected, result, _ = fmt.parse_inputs([("query", sample) for sample in samples] if paired else samples)
        if detected != "message":
            _, result = fmt.batch_to_message(detected, result)
        documents = [[messages[-1]] for messages in result["message"]]
        assert documents == [
            [
                {
                    "role": "document" if paired else "user",
                    "content": [
                        {"type": "text", "text": "two videos"},
                        {"type": "video", "video": list(frames_a)},
                        {"type": "video", "video": list(frames_b)},
                    ],
                }
            ],
            [
                {
                    "role": "document" if paired else "user",
                    "content": [
                        {"type": "text", "text": "one video"},
                        {"type": "video", "video": list(frames_a)},
                    ],
                }
            ],
        ]

    @pytest.mark.parametrize("container", [list, tuple])
    def test_explicit_text_pair_keeps_pair_roles(self, container):
        # [FUNCTION DOC]
        # Parametrised over container. Asserts a text value that is a two-string collection becomes a query/document
        # pair conversation.
        fmt = InputFormatter(model_type="test", message_format="structured")
        detected, inputs, _ = fmt.parse_inputs([{"text": container(["query", "document"])}])
        _, result = fmt.batch_to_message(detected, inputs)
        assert result["message"] == [
            [
                {"role": "query", "content": [{"type": "text", "text": "query"}]},
                {"role": "document", "content": [{"type": "text", "text": "document"}]},
            ]
        ]

    @pytest.mark.parametrize("modality", ["image", "audio", "video"])
    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("with_text", [False, True])
    def test_media_collection_preserves_sample_boundaries(self, modality, container, with_text):
        # [FUNCTION DOC]
        # Parametrised over modality, container and whether text is added. With two samples holding collections of two
        # and one media items, asserts each sample becomes its own conversation with one content item per media
        # (preceded by the text when present), and that the inputs are not mutated.
        fmt = InputFormatter(model_type="test", message_format="structured")
        media = {
            "image": [Image.new("RGB", (2, 2)) for _ in range(3)],
            "audio": [np.zeros(8) for _ in range(3)],
            "video": [np.zeros((2, 3, 2, 2)) for _ in range(3)],
        }[modality]
        samples = [{modality: container(media[:2])}, {modality: container(media[2:])}]
        if with_text:
            samples = [{"text": "19 inch monitor", **sample} for sample in samples]

        detected, inputs, _ = fmt.parse_inputs(samples)
        _, result = fmt.batch_to_message(detected, inputs)

        assert len(result["message"]) == 2
        for messages, items in zip(result["message"], [media[:2], media[2:]], strict=True):
            expected = [{"type": modality, modality: item} for item in items]
            if with_text:
                expected.insert(0, {"type": "text", "text": "19 inch monitor"})
            assert messages == [{"role": "user", "content": expected}]
        assert list(samples[0]) == (["text", modality] if with_text else [modality])
        assert samples[0][modality][0] is media[0]
        assert len(samples[0][modality]) == 2

    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Creates a structured-format InputFormatter on the test instance before each test.
        self.fmt = InputFormatter(model_type="test", message_format="structured")

    def test_str_modality(self):
        # [FUNCTION DOC]
        # Asserts converting a text batch gives modality message with one conversation per text.
        processor_inputs = {"text": ["hello", "world"]}
        modality, result = self.fmt.batch_to_message("text", processor_inputs)
        assert modality == "message"
        assert "message" in result
        assert len(result["message"]) == 2

    def test_tuple_modality(self):
        # [FUNCTION DOC]
        # Asserts converting an image+text batch gives one conversation list per sample.
        processor_inputs = {"image": ["cat.jpg", "dog.jpg"], "text": ["a cat", "a dog"]}
        modality, result = self.fmt.batch_to_message(("image", "text"), processor_inputs)
        assert modality == "message"
        assert len(result["message"]) == 2
        # Each message should have been created from both modalities
        for msg_list in result["message"]:
            assert isinstance(msg_list, list)

    def test_tuple_modality_preserves_processor_inputs_ordering(self):
        # [FUNCTION DOC]
        # Asserts content items follow the processor inputs' key order in both orders, regardless of the modality
        # tuple order.
        """Content items should follow processor_inputs key order, not the modality tuple order."""
        # processor_inputs has "text" before "image", but modality tuple is alphabetical
        processor_inputs = {"text": ["a cat", "a dog"], "image": ["cat.jpg", "dog.jpg"]}
        modality, result = self.fmt.batch_to_message(("image", "text"), processor_inputs)
        for msg_list in result["message"]:
            content = msg_list[0]["content"]
            assert [item["type"] for item in content] == ["text", "image"]

        # Reversed: "image" before "text" in processor_inputs
        processor_inputs = {"image": ["cat.jpg", "dog.jpg"], "text": ["a cat", "a dog"]}
        modality, result = self.fmt.batch_to_message(("image", "text"), processor_inputs)
        for msg_list in result["message"]:
            content = msg_list[0]["content"]
            assert [item["type"] for item in content] == ["image", "text"]

    def test_roundtrip_parse_then_convert(self):
        # [FUNCTION DOC]
        # Asserts parsing two texts then converting gives modality message with two conversations (no key mapping
        # issues).
        """parse_inputs -> batch_to_message should work without key mapping issues."""
        mod, inputs, extra = self.fmt.parse_inputs(["hello", "world"])
        assert mod == "text"
        new_mod, new_inputs = self.fmt.batch_to_message(mod, inputs)
        assert new_mod == "message"
        assert len(new_inputs["message"]) == 2

    def test_two_element_video_not_routed_to_pair(self, tmp_path):
        # [FUNCTION DOC]
        # Regression for issue 3840: writes two PNG frames, parses a video given as the two paths, and asserts
        # conversion gives a single user message with one video item of both paths, not a query/document pair.
        """A 2-frame video (2 image paths) stays one video message, not a query/document pair (#3840)."""
        frames = [str(tmp_path / "frame1.png"), str(tmp_path / "frame2.png")]
        for frame in frames:
            Image.new("RGB", (4, 4)).save(frame)
        modality, processor_inputs, _ = self.fmt.parse_inputs([{"video": frames}])
        modality, result = self.fmt.batch_to_message(modality, processor_inputs)
        assert modality == "message"
        messages = result["message"][0]
        # A single "user" message, not a ("query", "document") pair
        assert [msg["role"] for msg in messages] == ["user"]
        assert messages[0]["content"] == [{"type": "video", "video": frames}]

    def test_two_element_text_still_routed_to_pair(self):
        # [FUNCTION DOC]
        # Asserts a two-element text entry still converts into query and document roles.
        """A 2-element text input keeps routing to query/document roles (#3840)."""
        processor_inputs = {"text": [["a query", "a document"]]}
        modality, result = self.fmt.batch_to_message("text", processor_inputs)
        assert modality == "message"
        messages = result["message"][0]
        assert [msg["role"] for msg in messages] == ["query", "document"]


class TestPrependPromptToMessages:
    # [CLASS DOC]
    # Tests prepending a prompt to conversations as a system message.
    def test_structured_format(self):
        # [FUNCTION DOC]
        # Asserts structured format adds a system message with a typed text item holding the prompt.
        fmt = InputFormatter(model_type="test", message_format="structured")
        messages = [[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]]
        result = fmt.prepend_prompt_to_messages(messages, "Search: ")
        assert len(result[0]) == 2
        system_msg = result[0][0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == [{"type": "text", "text": "Search: "}]

    def test_flat_format(self):
        # [FUNCTION DOC]
        # Asserts flat format adds a system message whose content is the prompt string.
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [[{"role": "user", "content": "hello"}]]
        result = fmt.prepend_prompt_to_messages(messages, "Search: ")
        assert len(result[0]) == 2
        system_msg = result[0][0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "Search: "

    def test_multiple_message_lists(self):
        # [FUNCTION DOC]
        # Asserts every conversation in the batch gets a system message first.
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "world"}],
        ]
        result = fmt.prepend_prompt_to_messages(messages, "Query: ")
        assert len(result) == 2
        assert all(r[0]["role"] == "system" for r in result)


class TestPrependPromptToTexts:
    # [CLASS DOC]
    # Tests prepending a prompt to plain texts and text pairs.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Creates a structured-format InputFormatter on the test instance before each test.
        self.fmt = InputFormatter(model_type="test", message_format="structured")

    def test_single_texts(self):
        # [FUNCTION DOC]
        # Asserts the prompt is prefixed to each single text.
        result = self.fmt.prepend_prompt_to_texts(["hello", "world"], "Search: ")
        assert result == ["Search: hello", "Search: world"]

    def test_text_pairs(self):
        # [FUNCTION DOC]
        # Asserts for a pair only the first element is prefixed, returned as a list.
        result = self.fmt.prepend_prompt_to_texts([("query", "document")], "Search: ")
        assert result == [["Search: query", "document"]]

    def test_mixed_singles_and_pairs(self):
        # [FUNCTION DOC]
        # Asserts mixed batches prefix singles and the first element of pairs.
        result = self.fmt.prepend_prompt_to_texts(["hello", ["q", "d"]], "P: ")
        assert result == ["P: hello", ["P: q", "d"]]


class TestParseInputsEmpty:
    # [CLASS DOC]
    # Tests parsing an empty batch.
    def test_empty_inputs_returns_empty_text(self):
        # [FUNCTION DOC]
        # Asserts an empty batch parses as text with an empty text list and no extras.
        fmt = InputFormatter(model_type="test", message_format="structured")
        modality, inputs, extra = fmt.parse_inputs([])
        assert modality == "text"
        assert inputs == {"text": []}
        assert dict(extra) == {}


class TestNormalizeMessagesPreservesKeys:
    # [CLASS DOC]
    # Tests that normalisation keeps extra message keys.
    def test_extra_keys_preserved_flat_to_structured(self):
        # [FUNCTION DOC]
        # Asserts converting flat to structured keeps an extra "name" key.
        fmt = InputFormatter(model_type="test", message_format="structured")
        messages = [{"role": "user", "content": "hello", "name": "Alice"}]
        result = fmt.normalize_messages(messages)
        assert result[0]["name"] == "Alice"
        assert result[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_extra_keys_preserved_structured_to_flat(self):
        # [FUNCTION DOC]
        # Asserts converting structured to flat keeps an extra "name" key.
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}], "name": "Bob"}]
        result = fmt.normalize_messages(messages)
        assert result[0]["name"] == "Bob"
        assert result[0]["content"] == "hello"


class TestPrependPromptToMessagesNonShared:
    # [CLASS DOC]
    # Tests that prompt prepending does not share dicts between conversations.
    def test_system_messages_are_independent(self):
        # [FUNCTION DOC]
        # Asserts mutating one conversation's system message does not affect another's.
        """Each message list should get its own system message dict (no shared mutable state)."""
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "world"}],
        ]
        result = fmt.prepend_prompt_to_messages(messages, "Query: ")
        # Mutating one system message should not affect the other
        result[0][0]["extra"] = "mutated"
        assert "extra" not in result[1][0]


class TestInferFormatFlattened:
    # [CLASS DOC]
    # Tests format inference with an empty chat template.
    def test_empty_chat_template_returns_structured(self):
        # [FUNCTION DOC]
        # Asserts an empty string template infers structured.
        class EmptyTemplateProcessor:
            # [CLASS DOC]
            # HELPER: Fake processor with an empty chat template.
            chat_template = ""

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(EmptyTemplateProcessor()) == "structured"


class TestIsNonTextPair:
    # [CLASS DOC]
    # Tests the private check for pairs containing at least one non-text element.
    def test_text_pair_is_not_non_text(self):
        # [FUNCTION DOC]
        # Asserts a tuple of two strings is not a non-text pair.
        assert _is_non_text_pair(("hello", "world")) is False

    def test_text_pair_list_is_not_non_text(self):
        # [FUNCTION DOC]
        # Asserts a list of two strings is not a non-text pair.
        assert _is_non_text_pair(["hello", "world"]) is False

    def test_image_text_pair(self):
        # [FUNCTION DOC]
        # Asserts an (image, text) tuple is a non-text pair.
        img = Image.new("RGB", (32, 32))
        assert _is_non_text_pair((img, "some text")) is True

    def test_text_image_pair(self):
        # [FUNCTION DOC]
        # Asserts a (text, image) tuple is a non-text pair.
        img = Image.new("RGB", (32, 32))
        assert _is_non_text_pair(("some text", img)) is True

    def test_image_image_pair(self):
        # [FUNCTION DOC]
        # Asserts an (image, image) tuple is a non-text pair.
        img1 = Image.new("RGB", (32, 32))
        img2 = Image.new("RGB", (32, 32))
        assert _is_non_text_pair((img1, img2)) is True

    def test_array_text_pair(self):
        # [FUNCTION DOC]
        # Asserts an (array, text) tuple is a non-text pair.
        arr = np.random.randn(16000).astype(np.float32)
        assert _is_non_text_pair((arr, "some text")) is True

    def test_tensor_text_pair(self):
        # [FUNCTION DOC]
        # Asserts a (tensor, text) tuple is a non-text pair.
        tensor = torch.randn(16000)
        assert _is_non_text_pair((tensor, "some text")) is True

    def test_single_element_not_pair(self):
        # [FUNCTION DOC]
        # Asserts a one-element tuple is not a pair.
        assert _is_non_text_pair(("hello",)) is False

    def test_three_elements_not_pair(self):
        # [FUNCTION DOC]
        # Asserts a three-element tuple is not a pair.
        assert _is_non_text_pair(("a", "b", "c")) is False

    def test_string_not_pair(self):
        # [FUNCTION DOC]
        # Asserts a string is not a pair.
        assert _is_non_text_pair("hello") is False

    def test_message_dict_not_pair(self):
        # [FUNCTION DOC]
        # Asserts a tuple of message dicts is not a non-text pair.
        msg = {"role": "user", "content": "hello"}
        assert _is_non_text_pair((msg, msg)) is False

    def test_list_of_message_dicts_not_pair(self):
        # [FUNCTION DOC]
        # Asserts a tuple containing a list of message dicts is not a non-text pair.
        msgs = [{"role": "user", "content": "hello"}]
        assert _is_non_text_pair((msgs, "text")) is False


class TestPairToMessages:
    # [CLASS DOC]
    # Tests converting a (query, document) pair into a two-message conversation with query and document roles.
    @pytest.mark.parametrize("modalities", [("text", "image"), ("image", "text")])
    def test_compound_modality_preserves_user_order_in_both_roles(self, modalities):
        # [FUNCTION DOC]
        # Parametrised over key order. Asserts a pair of identical text+image dicts yields query and document messages
        # whose content items follow the dict's key order.
        fmt = InputFormatter(model_type="test", message_format="structured")
        values = {"text": "description", "image": "image_a"}
        sample = {modality: values[modality] for modality in modalities}
        result = fmt.pair_to_messages((sample, sample))
        assert [message["role"] for message in result] == ["query", "document"]
        for message in result:
            assert message["content"] == [{"type": modality, modality: values[modality]} for modality in modalities]

    @pytest.mark.parametrize("modality", ["image", "audio", "video"])
    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("with_text", [False, True])
    def test_media_collection_matches_single_input_content(self, modality, container, with_text):
        # [FUNCTION DOC]
        # Parametrised over media modality, container and adding text. Asserts the document message for a dict holding
        # a media collection equals the single-input conversion with the role changed to document, and has one item
        # per media element (after the text when present).
        fmt = InputFormatter(model_type="test", message_format="structured")
        sample = {modality: container(["a", "b"])}
        if with_text:
            sample = {"text": "description", **sample}
        result = fmt.pair_to_messages(("query", sample))
        detected, inputs, _ = fmt.parse_inputs([sample])
        _, single = fmt.batch_to_message(detected, inputs)
        assert result[1] == {**single["message"][0][0], "role": "document"}
        assert [item["type"] for item in result[1]["content"]] == (
            ["text", modality, modality] if with_text else [modality, modality]
        )

    def test_structured_same_modality(self):
        # [FUNCTION DOC]
        # Asserts an image/image pair yields query and document messages each with an image item.
        fmt = InputFormatter(model_type="test", message_format="structured")
        img1 = Image.new("RGB", (32, 32))
        img2 = Image.new("RGB", (32, 32))
        result = fmt.pair_to_messages((img1, img2))
        assert len(result) == 2
        assert result[0]["role"] == "query"
        assert result[1]["role"] == "document"
        assert result[0]["content"][0]["type"] == "image"
        assert result[1]["content"][0]["type"] == "image"

    def test_structured_cross_modality(self):
        # [FUNCTION DOC]
        # Asserts an image/text pair yields a query image item and a document text item with the text.
        fmt = InputFormatter(model_type="test", message_format="structured")
        img = Image.new("RGB", (32, 32))
        result = fmt.pair_to_messages((img, "some text"))
        assert len(result) == 2
        assert result[0]["role"] == "query"
        assert result[0]["content"][0]["type"] == "image"
        assert result[1]["role"] == "document"
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "some text"

    def test_flat_format(self):
        # [FUNCTION DOC]
        # Asserts flat format puts the raw image and raw text directly as content.
        fmt = InputFormatter(model_type="test", message_format="flat")
        img = Image.new("RGB", (32, 32))
        result = fmt.pair_to_messages((img, "some text"))
        assert len(result) == 2
        assert result[0]["role"] == "query"
        assert result[0]["content"] is img
        assert result[1]["role"] == "document"
        assert result[1]["content"] == "some text"

    def test_text_text_pair(self):
        # [FUNCTION DOC]
        # Asserts a text/text pair yields typed text items with each text.
        fmt = InputFormatter(model_type="test", message_format="structured")
        result = fmt.pair_to_messages(("query text", "document text"))
        assert len(result) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "query text"
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "document text"

    def test_compound_modality_expanded(self):
        # [FUNCTION DOC]
        # Asserts a multimodal document dict is expanded into two separate content items (image and text) with string
        # types.
        """A dict with multiple modalities (e.g. text+image) should be expanded into
        separate content items, not wrapped as a single compound-typed item."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        doc = {"image": "https://example.com/photo.jpg", "text": "a photo"}
        result = fmt.pair_to_messages(("query text", doc))
        assert len(result) == 2
        assert result[0]["role"] == "query"
        assert result[1]["role"] == "document"

        # Document should have two separate content items, not one compound item
        doc_content = result[1]["content"]
        assert len(doc_content) == 2
        content_types = {item["type"] for item in doc_content}
        assert content_types == {"image", "text"}
        # Each item should have its modality as the key (not a tuple key)
        for item in doc_content:
            assert isinstance(item["type"], str)

    def test_single_key_dict_in_pair_unwraps(self):
        # [FUNCTION DOC]
        # Asserts a single-key image dict in a pair is unwrapped into an image content item holding the bare image.
        """A ``{"image": pil}`` wrapper inside a pair should unwrap to the bare PIL value."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        img = Image.new("RGB", (32, 32))
        result = fmt.pair_to_messages(("a query", {"image": img}))
        assert len(result) == 2
        assert result[1]["role"] == "document"
        # The content should reference the unwrapped image, not the wrapping dict
        assert result[1]["content"] == [{"type": "image", "image": img}]


class TestParseInputsPairs:
    # [CLASS DOC]
    # Tests parsing batches of pairs.
    def test_text_pairs_stay_as_text(self):
        # [FUNCTION DOC]
        # Asserts text pairs stay as text modality.
        """Text pairs should be detected as 'text' modality (tokenizer handles natively)."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        modality, inputs, _ = fmt.parse_inputs([("hello", "world"), ("foo", "bar")])
        assert modality == "text"
        assert "text" in inputs
        assert len(inputs["text"]) == 2

    def test_non_text_pairs_become_message(self):
        # [FUNCTION DOC]
        # Asserts (image, text) pairs become message modality with query and document roles per conversation.
        """Non-text pairs should be converted to message format."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        img1 = Image.new("RGB", (32, 32))
        img2 = Image.new("RGB", (32, 32))
        modality, inputs, _ = fmt.parse_inputs([(img1, "text1"), (img2, "text2")])
        assert modality == "message"
        assert "message" in inputs
        assert len(inputs["message"]) == 2
        # Each message should have query and document roles
        for messages in inputs["message"]:
            assert len(messages) == 2
            assert messages[0]["role"] == "query"
            assert messages[1]["role"] == "document"

    def test_image_image_pairs(self):
        # [FUNCTION DOC]
        # Asserts an image/image pair becomes message modality with image items on both sides.
        """Image-image pairs should be converted to message format."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        img1 = Image.new("RGB", (32, 32))
        img2 = Image.new("RGB", (64, 64))
        modality, inputs, _ = fmt.parse_inputs([(img1, img2)])
        assert modality == "message"
        assert "message" in inputs
        messages = inputs["message"][0]
        assert messages[0]["content"][0]["type"] == "image"
        assert messages[1]["content"][0]["type"] == "image"

    def test_array_text_pairs(self):
        # [FUNCTION DOC]
        # Asserts an (audio array, text) pair becomes message modality with an audio query and text document.
        """Audio array + text pairs should be converted to message format."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        arr = np.random.randn(16000).astype(np.float32)
        modality, inputs, _ = fmt.parse_inputs([(arr, "transcription")])
        assert modality == "message"
        messages = inputs["message"][0]
        assert messages[0]["content"][0]["type"] == "audio"
        assert messages[1]["content"][0]["type"] == "text"

    def test_mixed_text_and_non_text_pairs_all_get_query_document_roles(self):
        # [FUNCTION DOC]
        # Asserts a batch mixing a text pair and an image pair converts both to query/document conversations, keeping
        # the text pair's texts.
        """When a batch mixes text pairs and non-text pairs, text pairs must also
        go through pair_to_messages so they get query/document roles, not user role."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        img = Image.new("RGB", (32, 32))
        inputs = [
            ("query text", "document text"),  # text pair
            (img, "caption"),  # non-text pair
        ]
        modality, result, _ = fmt.parse_inputs(inputs)
        assert modality == "message"
        assert len(result["message"]) == 2

        # Text pair must have query/document roles (not a single "user" message)
        text_pair_msgs = result["message"][0]
        assert len(text_pair_msgs) == 2
        assert text_pair_msgs[0]["role"] == "query"
        assert text_pair_msgs[1]["role"] == "document"
        assert text_pair_msgs[0]["content"][0]["text"] == "query text"
        assert text_pair_msgs[1]["content"][0]["text"] == "document text"

        # Non-text pair should also have query/document roles
        img_pair_msgs = result["message"][1]
        assert len(img_pair_msgs) == 2
        assert img_pair_msgs[0]["role"] == "query"
        assert img_pair_msgs[1]["role"] == "document"


class TestIsTextOnlyMessages:
    # [CLASS DOC]
    # Tests the static check for whether a batch of conversations contains only text.
    def test_flat_text_only(self):
        # [FUNCTION DOC]
        # Asserts flat string content is text only.
        """Flat format messages with string content are text-only."""
        batch = [[{"role": "user", "content": "hello"}]]
        assert InputFormatter.is_text_only_messages(batch) is True

    def test_structured_text_only(self):
        # [FUNCTION DOC]
        # Asserts structured content with only text items is text only.
        """Structured format messages with only text items are text-only."""
        batch = [[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]]
        assert InputFormatter.is_text_only_messages(batch) is True

    def test_structured_text_implicit_type(self):
        # [FUNCTION DOC]
        # Asserts structured items without a type key count as text.
        """Structured items without an explicit 'type' key default to text."""
        batch = [[{"role": "user", "content": [{"text": "hello"}]}]]
        assert InputFormatter.is_text_only_messages(batch) is True

    def test_structured_with_image(self):
        # [FUNCTION DOC]
        # Asserts a message with an image item is not text only.
        """Messages containing an image item are not text-only."""
        batch = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://example.com/img.jpg"},
                        {"type": "text", "text": "describe this"},
                    ],
                }
            ]
        ]
        assert InputFormatter.is_text_only_messages(batch) is False

    def test_structured_with_audio(self):
        # [FUNCTION DOC]
        # Asserts a message with an audio item is not text only.
        """Messages containing an audio item are not text-only."""
        batch = [[{"role": "user", "content": [{"type": "audio", "url": "https://example.com/a.wav"}]}]]
        assert InputFormatter.is_text_only_messages(batch) is False

    def test_mixed_batch_one_multimodal(self):
        # [FUNCTION DOC]
        # Asserts a batch with any non-text sample is not text only.
        """A batch is not text-only if any sample contains non-text content."""
        batch = [
            [{"role": "user", "content": "pure text"}],
            [{"role": "user", "content": [{"type": "image", "url": "https://example.com/img.jpg"}]}],
        ]
        assert InputFormatter.is_text_only_messages(batch) is False

    def test_empty_batch(self):
        # [FUNCTION DOC]
        # Asserts an empty batch is text only.
        """An empty batch is trivially text-only."""
        assert InputFormatter.is_text_only_messages([]) is True

    def test_content_not_str_or_list(self):
        # [FUNCTION DOC]
        # Asserts non-string, non-list content (an integer) is not text only.
        """Non-string, non-list content is treated as non-text."""
        batch = [[{"role": "user", "content": 42}]]
        assert InputFormatter.is_text_only_messages(batch) is False

    def test_multiple_text_messages(self):
        # [FUNCTION DOC]
        # Asserts a multi-turn text conversation is text only.
        """Multi-turn text-only conversations are text-only."""
        batch = [
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ]
        ]
        assert InputFormatter.is_text_only_messages(batch) is True


class FakeProcessor:
    # [CLASS DOC]
    # HELPER: Minimal processor stand-in for the pair-role probe, holding a chat template (string, dict of named
    # templates, or None).
    """Minimal stand-in exposing the two attributes the pair-role probe touches."""

    def __init__(self, chat_template):
        # [FUNCTION DOC]
        # HELPER: Stores the chat template.
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, chat_template=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Renders the messages with Jinja: picks the named template (the given name or "default") when the
        # template is a dict, and passes extra keyword arguments to the template. Tokenize is ignored. Raises whatever
        # Jinja raises for broken templates.
        template = self.chat_template
        if isinstance(template, dict):
            template = template[chat_template or "default"]
        return Environment().from_string(template).render(messages=messages, **kwargs)


class TestPairRolesFailure:
    # [CLASS DOC]
    # Tests the formatter's probe that checks whether the processor's chat template renders both query and document
    # content (returns None if it does, else a reason string).
    @pytest.mark.parametrize(
        ("template", "supported"),
        [
            # Reranker templates select on both roles, either via a filter or a branch
            (RERANKER_TEMPLATE, True),
            (
                "{% for m in messages %}{% if m.role == 'query' %}Q{{ m.content }}"
                "{% elif m.role == 'document' %}D{{ m.content }}{% endif %}{% endfor %}",
                True,
            ),
            # Never mentions the roles yet passes both through, as ChatML templates do
            ("{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}", True),
            # Addresses the pair positionally rather than by role name
            ("Q: {{ messages[0].content }}\nD: {{ messages[1].content }}", True),
            ("{% for m in messages %}{{ m.content }}\n{% endfor %}", True),
            # Mentions both roles, but only in a branch a pair never reaches
            (
                "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}"
                "{% elif m.role == 'query' or m.role == 'document' %}{% endif %}{% endfor %}",
                False,
            ),
            # Half a pair is no better than none, the other side is still dropped
            ("{% for m in messages %}{% if m.role == 'query' %}{{ m.content }}{% endif %}{% endfor %}", False),
            (GENERIC_INSTRUCT_TEMPLATE, False),
            # Anything unrenderable is treated as unable to carry a pair
            ("{% for m in messages %}{{ m.content }}", False),
            ("{{ nope.missing.attribute }}", False),
            (None, False),
        ],
    )
    def test_detection(self, template, supported):
        # [FUNCTION DOC]
        # Parametrised over templates and whether they support pairs: role-selecting rerank templates, role-echoing
        # templates, positional templates and plain content loops pass; templates that only render the user role, only
        # render the query, the generic instruct template, unrenderable or erroring templates, and None fail. Asserts
        # the probe result matches.
        assert (InputFormatter("bert", processor=FakeProcessor(template)).pair_roles_failure() is None) is supported

    @pytest.mark.parametrize(
        "filter_expression",
        ["| upper", "| capitalize", "| truncate(10)", "| wordwrap(4)", "| replace('a', 'z')", "| trim", "[:8]"],
    )
    def test_transformed_content_still_counts_as_rendered(self, filter_expression):
        # [FUNCTION DOC]
        # Parametrised over Jinja filters and slicing. Asserts content reshaped by a filter still counts as rendered.
        """A template may reshape what it renders, and the content still reaches the model."""
        template = "{% for m in messages %}{{ m.content " + filter_expression + " }}\n{% endfor %}"
        assert InputFormatter("bert", processor=FakeProcessor(template)).pair_roles_failure() is None

    def test_probe_covers_structured_message_format(self):
        # [FUNCTION DOC]
        # Asserts a template reading typed content infers structured format and passes the probe, while a structured
        # user-only template fails it.
        """Structured-format processors probe through typed content items rather than plain strings."""
        pass_through = "{% for m in messages %}{{ m.content[0].text }}\n{% endfor %}"
        formatter = InputFormatter("bert", processor=FakeProcessor(pass_through))
        assert formatter.message_format == "structured"
        assert formatter.pair_roles_failure() is None

        user_only = "{% for m in messages %}{% if m.role == 'user' %}{{ m.content[0].text }}{% endif %}{% endfor %}"
        formatter = InputFormatter("bert", processor=FakeProcessor(user_only))
        assert formatter.message_format == "structured"
        assert formatter.pair_roles_failure() is not None

    def test_probe_uses_the_render_kwargs(self):
        # [FUNCTION DOC]
        # With a dict of a default generic template and a "reranker" template, asserts the probe fails by default,
        # passes when kwargs select the reranker template, and fails again by default (results are cached per kwargs).
        """Kwargs can select a named template, so the probe must measure the one the render uses.

        Qwen3-VL-Reranker ships a ``reranker`` template next to a ``default`` that drops the pair
        roles, selected through its configured ``processing_kwargs``.
        """
        processor = FakeProcessor({"default": GENERIC_INSTRUCT_TEMPLATE, "reranker": RERANKER_TEMPLATE})
        formatter = InputFormatter("bert", processor=processor)
        assert formatter.pair_roles_failure() is not None
        assert formatter.pair_roles_failure({"chat_template": "reranker"}) is None
        # Verdicts for different kwargs must not serve each other from the cache
        assert formatter.pair_roles_failure() is not None

    def test_failure_names_the_half_that_is_dropped(self):
        # [FUNCTION DOC]
        # Asserts the failure message names the dropped half ("'document' content") for a query-only template and both
        # halves for the generic template.
        query_only = "{% for m in messages %}{% if m.role == 'query' %}{{ m.content }}{% endif %}{% endfor %}"
        assert "'document' content" in InputFormatter("bert", processor=FakeProcessor(query_only)).pair_roles_failure()

        both = InputFormatter("bert", processor=FakeProcessor(GENERIC_INSTRUCT_TEMPLATE))
        assert "'query' and 'document' content" in both.pair_roles_failure()

    def test_probe_is_rendered_lazily(self):
        # [FUNCTION DOC]
        # Asserts a template assigned to the processor after constructing the formatter is used by the probe.
        """A template assigned after construction must still be picked up."""
        processor = FakeProcessor(GENERIC_INSTRUCT_TEMPLATE)
        formatter = InputFormatter("bert", processor=processor)
        processor.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
        assert formatter.pair_roles_failure() is None

    def test_probe_reruns_after_a_rejected_template_is_replaced(self):
        # [FUNCTION DOC]
        # Asserts that after a failing probe, replacing the template makes the next probe pass (no stale cached
        # verdict).
        """Replacing a rejected template is the documented remedy, so a stale verdict must not pin it."""
        processor = FakeProcessor(GENERIC_INSTRUCT_TEMPLATE)
        formatter = InputFormatter("bert", processor=processor)
        assert formatter.pair_roles_failure() is not None

        processor.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
        assert formatter.pair_roles_failure() is None

    def test_probe_reruns_after_a_named_template_is_edited_in_place(self):
        # [FUNCTION DOC]
        # Asserts that editing one entry of a named-template dict in place also invalidates the cached verdict.
        """Fixing one entry of a named-template dict edits it in place, and must not hit a stale verdict."""
        processor = FakeProcessor({"default": GENERIC_INSTRUCT_TEMPLATE})
        formatter = InputFormatter("bert", processor=processor)
        assert formatter.pair_roles_failure() is not None

        processor.chat_template["default"] = "{% for m in messages %}{{ m.content }}{% endfor %}"
        assert formatter.pair_roles_failure() is None

    def test_failure_distinguishes_an_empty_render_from_a_raise(self):
        # [FUNCTION DOC]
        # Asserts the failure message says the content "does not reach the rendered prompt" for a dropping template,
        # "it raised" for an erroring template, "the model has no chat template" without a processor, and None for the
        # rerank template.
        dropped = InputFormatter("bert", processor=FakeProcessor(GENERIC_INSTRUCT_TEMPLATE))
        assert "does not reach the rendered prompt" in dropped.pair_roles_failure()

        raised = InputFormatter("bert", processor=FakeProcessor("{{ nope() }}"))
        assert "it raised" in raised.pair_roles_failure()

        assert InputFormatter("bert").pair_roles_failure() == "the model has no chat template"
        assert InputFormatter("bert", processor=FakeProcessor(RERANKER_TEMPLATE)).pair_roles_failure() is None


class TestHasPairRoles:
    # [CLASS DOC]
    # Tests detecting whether a batch of conversations contains a query/document pair.
    @pytest.mark.parametrize(
        ("messages_batch", "expected"),
        [
            ([[{"role": "query", "content": "q"}, {"role": "document", "content": "d"}]], True),
            # One pair anywhere in the batch is enough, its content would be dropped just the same
            (
                [
                    [{"role": "user", "content": "hi"}],
                    [{"role": "query", "content": "q"}, {"role": "document", "content": "d"}],
                ],
                True,
            ),
            # One pair role alone is not a pair, e.g. a hand-written query message with no document
            ([[{"role": "query", "content": "q"}]], False),
            ([[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]], False),
            ([[{"content": "no role at all"}]], False),
            ([], False),
        ],
    )
    def test_batches(self, messages_batch, expected):
        # [FUNCTION DOC]
        # Parametrised over batches. Asserts a pair anywhere in the batch is detected, while a lone query role,
        # user/assistant turns, role-less messages and an empty batch are not.
        assert InputFormatter("bert").has_pair_roles(messages_batch) is expected

    def test_covers_non_text_pairs(self):
        # [FUNCTION DOC]
        # Asserts conversations produced from a non-text (image, text) pair are detected as having pair roles.
        """Non-text pairs reach pair_to_messages by a different route, but take the same roles."""
        formatter = InputFormatter("bert")
        messages = formatter.pair_to_messages((Image.new("RGB", (4, 4)), "a document"))
        assert formatter.has_pair_roles([messages]) is True
