# [MODULE DOC]
# Modality detection, input parsing and chat-message conversion for multimodal models. Optionally imports PIL and
# torchcodec decoder classes (None if unavailable). Provides: string heuristics that classify URLs/paths/data URIs as
# image, video or audio; helpers that unwrap dict-wrapped or decoder-wrapped audio/video into raw arrays while
# collecting batch-level processor kwargs (audio sampling rate, per-video metadata); InputFormatter, which groups a
# batch of raw inputs by modality for a processor and converts inputs to chat messages in "structured" (content list
# of typed dicts) or "flat" (content is the raw value) form; and module functions infer_modality, infer_batch_modality,
# format_modality and raise_unsupported_modality_error. Module constants: model types known to need the flat format,
# the pair roles ("query", "document") and probe string pairs used to test whether a chat template carries both roles.
# Logs via the standard logging module.
"""Modality detection, input parsing, and message format conversion."""

from __future__ import annotations

import copy
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, NoReturn
from urllib.parse import urlparse

import numpy as np
import torch
from typing_extensions import TypeIs

from sentence_transformers.base.modality_types import (
    MULTIMODAL_DICT_KEYS,
    AudioInput,
    ImageInput,
    MessageDict,
    MessageFormat,
    Modality,
    PairInput,
    SingleInput,
    VideoInput,
)

try:
    from PIL.Image import Image as PILImage
except ImportError:
    PILImage = None

try:
    from torchcodec.decoders import AudioDecoder, VideoDecoder
except (ImportError, OSError, RuntimeError):
    AudioDecoder = None  # type: ignore[assignment,misc]
    VideoDecoder = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# Override specific model types here when the heuristic in _infer_format gets them wrong.
KNOWN_MODEL_TYPES_MESSAGE_FORMATS = {
    "apertus": "flat",
    "deepseek_v3": "flat",
    "gpt_oss": "flat",
    "seed_oss": "flat",
}


# The roles pair_to_messages assigns to the two halves of a pair.
PAIR_ROLES = ("query", "document")
# A baseline pair, then one probe per role varying only that role's content. Varied values differ from
# their baseline in first character and in length, so even a render that truncates hard still moves.
PAIR_ROLE_PROBES = (("alpha", "bravo"), ("charlie", "bravo"), ("alpha", "foxtrot"))


def _looks_like_url(text: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True if the string starts with "http://" or "https://" and contains no spaces.
    """Check if a string looks like a valid URL (starts with http(s) and has no spaces)."""
    return text.startswith(("http://", "https://")) and " " not in text


def _is_media_url_or_path(text: str, extensions: tuple[str, ...]) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True if the string is a URL whose lower-cased path ends with one of the given extensions (False for
    # URLs that fail to parse), or a non-URL string whose lower-cased value ends with one of the extensions and names an
    # existing local file. Side effect: filesystem existence check.
    """Check if a string is a URL or local file path with one of the given extensions."""
    if _looks_like_url(text):
        try:
            path = urlparse(text).path.lower()
        except ValueError:
            # urlparse can raise ValueError on malformed URLs (e.g. "https://[[invalid]]")
            return False
        return path.endswith(extensions)
    return text.lower().endswith(extensions) and os.path.isfile(text)


def is_image_url_or_path(text: str) -> bool:
    # [FUNCTION DOC]
    # Returns True for strings starting with "data:image/", or image URLs/existing paths with a common image extension
    # (jpg, jpeg, png, gif, bmp, tiff, webp).
    """Check if a string is an image URL, file path, or data URI."""
    if text.startswith("data:image/"):
        return True
    return _is_media_url_or_path(text, (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"))


def is_video_url_or_path(text: str) -> bool:
    # [FUNCTION DOC]
    # Returns True for video URLs/existing paths with a common video extension (mp4, avi, mov, wmv, flv, mkv), or for URLs
    # whose host is one of the YouTube domains; False otherwise (including unparsable URLs).
    """Check if a string is a video URL or file path."""
    if _is_media_url_or_path(text, (".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv")):
        return True
    if not _looks_like_url(text):
        return False
    try:
        netloc = urlparse(text).netloc
    except ValueError:
        return False
    return netloc in (
        "www.youtube.com",
        "youtube.com",
        "youtu.be",
        "m.youtube.com",
    )


def is_audio_url_or_path(text: str) -> bool:
    # [FUNCTION DOC]
    # Returns True for audio URLs/existing paths with an extension among mp3, wav, ogg, flac, aac.
    """Check if a string is an audio URL or file path."""
    return _is_media_url_or_path(text, (".mp3", ".wav", ".ogg", ".flac", ".aac"))


def is_message_dict(value: Any) -> TypeIs[MessageDict]:
    # [FUNCTION DOC]
    # Type guard: returns True if the value is a dict containing both "role" and "content" keys.
    """Check if a value is a single chat message: a dict with ``"role"`` and ``"content"`` keys."""
    return isinstance(value, dict) and "role" in value and "content" in value


def _is_non_text_pair(sample: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True if the sample is a 2-element list/tuple that is not a pair of two strings and in which neither
    # element is a message dict nor a non-empty list starting with a dict. Such pairs must be turned into query/document
    # messages.
    """Check if a sample is a non-text pair (2-element tuple/list with at least one non-string element).

    Text pairs ``(str, str)`` are handled natively by tokenizers and detected as ``"text"`` modality
    by :func:`infer_modality`. This helper detects pairs that contain at least one non-string element
    (e.g. an image, audio array, or dict), which require conversion to message format.
    """
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        return False
    # Text pairs are handled by infer_modality as "text"
    if isinstance(sample[0], str) and isinstance(sample[1], str):
        return False
    # Exclude message dicts and list-of-message-dicts
    for elem in sample:
        if is_message_dict(elem):
            return False
        if isinstance(elem, list) and elem and isinstance(elem[0], dict):
            return False
    return True


def _record_sampling_rate(sampling_rate: int, extra_modality_kwargs: dict[str, dict[str, Any]]) -> None:
    # [FUNCTION DOC]
    # HELPER: stores the sampling rate under the "audio" entry of the extra kwargs mapping (mutated in place). Raises
    # ValueError if a different sampling rate was already recorded for this batch.
    """Record the batch-level ``sampling_rate``, raising if samples disagree.

    Feature extractors accept a single ``sampling_rate`` for the whole batch, so conflicting
    per-sample rates cannot be honored and would silently process some audio at the wrong rate.
    """
    existing = extra_modality_kwargs["audio"].get("sampling_rate")
    if existing is not None and existing != sampling_rate:
        raise ValueError(
            f"This batch mixes audio with different sampling rates ({existing} and {sampling_rate}), but the "
            "processor accepts a single sampling rate per batch. Resample the audio to a common rate, or "
            "encode each sampling rate in a separate call."
        )
    extra_modality_kwargs["audio"]["sampling_rate"] = sampling_rate


def _unwrap_audio(
    audio_value: AudioInput | list[AudioInput] | tuple[AudioInput, ...],
    extra_modality_kwargs: dict[str, dict[str, Any]],
) -> Any:
    # [FUNCTION DOC]
    # HELPER: converts an audio input to a raw value. Lists/tuples are processed element-wise and returned as a list.
    # Dicts record their "sampling_rate" (if present) and return their "array". Audio decoder objects are fully decoded,
    # their sample rate recorded, and the channels averaged into a 1-D numpy array. Anything else is returned unchanged.
    # Mutates the extra kwargs mapping; may raise ValueError on conflicting sampling rates.
    """Unwrap dict-wrapped audio or an ``AudioDecoder`` into a raw array, collecting ``sampling_rate``.

    Passes through unchanged if ``audio_value`` is already a raw array/tensor/URL/path.
    """
    if isinstance(audio_value, (list, tuple)):
        return [_unwrap_audio(item, extra_modality_kwargs) for item in audio_value]
    if isinstance(audio_value, dict):
        if "sampling_rate" in audio_value:
            _record_sampling_rate(audio_value["sampling_rate"], extra_modality_kwargs)
        return audio_value["array"]
    if AudioDecoder is not None and isinstance(audio_value, AudioDecoder):
        samples = audio_value.get_all_samples()
        # AudioDecoder returns (channels, samples): mean over channels to get 1D numpy
        _record_sampling_rate(samples.sample_rate, extra_modality_kwargs)
        return samples.data.mean(dim=0).numpy()
    return audio_value


def _as_sequence(value: Any) -> list | tuple:
    # [FUNCTION DOC]
    # HELPER: returns the value itself if it is a list or tuple, otherwise a one-element list containing it.
    return value if isinstance(value, (list, tuple)) else [value]


def _is_video_frames(value: Any) -> TypeIs[list[ImageInput] | tuple[ImageInput, ...]]:
    # [FUNCTION DOC]
    # HELPER: type guard returning True if the value is a non-empty list/tuple whose every element is inferred as an image
    # (so it represents one video's frames); returns False if any element's modality cannot be inferred.
    if not isinstance(value, (list, tuple)) or not value:
        return False
    try:
        return all(infer_modality(frame) == "image" for frame in value)
    except ValueError:
        return False


def _unwrap_video(
    video_value: VideoInput | list[VideoInput] | tuple[VideoInput, ...],
    extra_modality_kwargs: dict[str, dict[str, Any]],
) -> Any:
    # [FUNCTION DOC]
    # HELPER: converts a video input to a raw value while appending one metadata entry per video to the "video_metadata"
    # list under the "video" entry of the extra kwargs mapping. A frame list is first wrapped as a dict without metadata.
    # Other lists/tuples are processed element-wise and flattened into one list. Dicts append their metadata (or None) and
    # return their array, a frame sequence being wrapped in a one-element list containing the frame list. Video decoder
    # objects are fully decoded and get metadata built from the decoder (average fps, frame count, duration and frame
    # indices). Anything else (paths, URLs, arrays) gets a None metadata entry and is returned unchanged.
    """Unwrap dict-wrapped video or a ``VideoDecoder`` into a raw array, collecting ``video_metadata``.

    Passes through unchanged if ``video_value`` is already a raw array/tensor/URL/path. Appends one
    metadata entry per video (``None`` when the sample carries none) so the batch-level list stays
    index-aligned with the videos. :func:`_reconcile_video_metadata` resolves the ``None`` entries
    once the whole batch has been parsed.
    """
    if _is_video_frames(video_value):
        video_value = {"array": video_value, "video_metadata": None}
    if isinstance(video_value, (list, tuple)):
        return [video for item in video_value for video in _as_sequence(_unwrap_video(item, extra_modality_kwargs))]
    metadata_list = extra_modality_kwargs["video"].setdefault("video_metadata", [])
    if isinstance(video_value, dict):
        metadata_list.append(video_value.get("video_metadata"))
        frames = video_value["array"]
        return [list(frames)] if isinstance(frames, (list, tuple)) else frames
    if VideoDecoder is not None and isinstance(video_value, VideoDecoder):
        frame_batch = video_value.get_frames_in_range(0, len(video_value))
        metadata_list.append(
            {
                "fps": video_value.metadata.average_fps,
                "total_num_frames": video_value.metadata.num_frames,
                "duration": video_value.metadata.duration_seconds,
                "frames_indices": list(range(frame_batch.data.shape[0])),
            }
        )
        return frame_batch.data
    metadata_list.append(None)
    return video_value


def _reconcile_video_metadata(
    typed_inputs: list[tuple[Modality | Literal["pair"], Any]],
    extra_modality_kwargs: dict[str, dict[str, Any]],
) -> None:
    # [FUNCTION DOC]
    # HELPER: post-processes the collected video metadata for a whole batch. If none was collected, returns. If every entry
    # is None, removes the metadata key (and the "video" kwargs entry if it becomes empty) so the processor uses defaults.
    # Otherwise gathers all videos from the typed inputs (plain video inputs and the "video" part of multimodal dicts) and
    # fills each None entry with default metadata derived from the video's length (frame count, frame indices, fps and
    # duration None). Raises ValueError if such a video is a string or has no length, and the strict zip raises if the
    # counts of videos and metadata entries differ. Mutates the kwargs mapping in place.
    """Resolve the per-sample video metadata entries collected by :func:`_unwrap_video`.

    If no video in the batch carried metadata, the key is dropped so the processor infers defaults
    itself. Otherwise every ``None`` entry is filled with the frame count of that video, matching what
    transformers infers for a metadata-less video. This requires the video to be in memory, so
    path/URL videos without metadata raise instead, as passing a partial list would attach metadata
    to the wrong videos.
    """
    video_kwargs = extra_modality_kwargs.get("video")
    if not video_kwargs or "video_metadata" not in video_kwargs:
        return
    metadata_list = video_kwargs["video_metadata"]
    if all(entry is None for entry in metadata_list):
        del video_kwargs["video_metadata"]
        if not video_kwargs:
            del extra_modality_kwargs["video"]
        return
    videos = [
        video
        for mod, value in typed_inputs
        if mod == "video" or (isinstance(mod, tuple) and "video" in mod)
        for video in _as_sequence(value["video"] if isinstance(mod, tuple) else value)
    ]
    for index, (entry, video) in enumerate(zip(metadata_list, videos, strict=True)):
        if entry is not None:
            continue
        if isinstance(video, str) or not hasattr(video, "__len__"):
            raise ValueError(
                "This batch mixes videos that carry per-sample 'video_metadata' with videos that do not. "
                "Default metadata can only be inferred for in-memory videos (frame arrays or lists), so "
                "either pass 'video_metadata' for every video in the batch or for none of them."
            )
        num_frames = len(video)
        metadata_list[index] = {
            "total_num_frames": num_frames,
            "fps": None,
            "duration": None,
            "frames_indices": list(range(num_frames)),
        }


class InputFormatter:
    # [CLASS DOC]
    # Parses raw input batches for a model processor. Attributes: model_type (string from the model config), processor
    # (optional processor/tokenizer), supported_modalities (optional list restricting which media heuristics apply),
    # message_format ("structured" or "flat"), and a private cache of the last pair-role probe result (snapshot of
    # template and kwargs, plus the failure reason or None).
    """Handles input parsing, modality detection, and message format conversion.

    This class manages the complete input preprocessing pipeline:
    1. Parsing raw inputs to detect their modality (text, image, audio, video, message)
    2. Converting inputs to different chat template formats
    3. Normalizing mixed-modality inputs

    Different models require different message/chat template formats:
    - **Structured format**: Content is a list of dicts with type annotations
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    - **Flat format**: Content is the direct value
        [{"role": "user", "content": "hello"}]

    Args:
        model_type: The model type string (e.g. from ``config.model_type``).
        message_format: Message format to use. Options:
            - ``"structured"``: Content is a list of dicts with type/modality keys
            - ``"flat"``: Content is the direct value
            - ``"auto"``: Automatically infer from processor (default)
        processor: Optional processor to infer format from when ``message_format="auto"``.
        supported_modalities: Optional list of modalities supported by the model. When provided,
            string inputs that look like media URLs/paths are only classified as non-text if the
            model actually supports that modality. This prevents text-only models from
            misclassifying text containing media URLs.
    """

    def __init__(
        self,
        model_type: str,
        message_format: MessageFormat = "auto",
        processor=None,
        supported_modalities: list[Modality] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores model_type, processor and supported_modalities, clears the pair-role probe cache, and resolves message_format:
        # "auto" infers it from the processor (or "structured" when there is no processor); other values are used as given.
        self.model_type = model_type
        self.processor = processor
        self.supported_modalities = supported_modalities
        self._pair_role_probe: tuple[tuple[Any, dict[str, Any]], str | None] | None = None
        if message_format == "auto":
            self.message_format = self._infer_format(processor) if processor else "structured"
        else:
            self.message_format = message_format

    def _infer_format(self, processor) -> Literal["structured", "flat"]:
        # [FUNCTION DOC]
        # HELPER: returns the message format the processor expects. Uses the known-model-type override table first; if the
        # processor has no non-empty string chat template, returns "structured"; if the template contains any of several
        # substrings indicating content is indexed or typed (such as content index access or a "type" field), returns
        # "structured"; otherwise "flat".
        """Infer the message format expected by the processor.

        Checks known model types first, then inspects the processor's chat template
        for patterns indicating structured format. Defaults to ``"structured"`` if
        neither approach is conclusive.

        Args:
            processor: The processor/tokenizer to inspect.

        Returns:
            ``"structured"`` or ``"flat"`` message format.
        """
        if self.model_type in KNOWN_MODEL_TYPES_MESSAGE_FORMATS:
            return KNOWN_MODEL_TYPES_MESSAGE_FORMATS[self.model_type]

        template = getattr(processor, "chat_template", None)
        if not isinstance(template, str) or not template:
            return "structured"

        # Patterns that indicate the chat template expects content as a list of dicts
        structured_patterns = [
            "content[0]",
            ".type",
            "'type'",
            '"type"',
            "item.type",
            "message.content[",
        ]
        if any(pattern in template for pattern in structured_patterns):
            return "structured"

        return "flat"

    def parse_inputs(
        self,
        inputs: Sequence[SingleInput | PairInput],
    ) -> tuple[Modality, dict[str, list], defaultdict[str, dict[str, Any]]]:
        # [FUNCTION DOC]
        # Classifies and groups a batch of inputs. Returns a triple: the batch modality (a name, a tuple of names, or
        # "message"), a dict mapping processor input names (modality names) to lists of values, and a defaultdict of extra
        # per-modality processor kwargs. Empty input returns text modality with an empty text list. Each item is classified:
        # non-text pairs are tagged "pair"; otherwise its modality is inferred (respecting supported_modalities), single-key
        # multimodal dicts are unwrapped to their value, audio/video values (also inside multimodal dicts, which are shallow
        # copied) are unwrapped while collecting sampling rate and video metadata, and a single message dict is wrapped in a
        # list. Video metadata is then reconciled. If any non-text pair exists, every item is converted to a message list
        # (pairs and text pairs via pair_to_messages, messages unchanged, others via to_message) and "message" is returned. If
        # all items share one string modality, values are grouped under that key. If they share one tuple modality, values are
        # split per modality key in the first item's key order, except that when message support is allowed (unrestricted, or
        # "message" supported) and items differ in key order, everything is converted to messages instead. Mixed modalities
        # are converted to messages (debug log). May raise ValueError from modality inference or unwrapping.
        """Parse inputs and group by modality.

        Analyzes a list of inputs to detect their modality (text, image, audio, video, message)
        and groups them appropriately for the processor. Handles mixed modalities by converting
        to message format when necessary.

        Non-text pairs (e.g. ``(image, text)`` or ``(image, image)``) are detected and converted
        to message format with ``"query"``/``"document"`` roles via :meth:`pair_to_messages`.

        Args:
            inputs: List of inputs to parse. Can be:
                - str: Text inputs
                - tuple/list of str: Text pairs (for cross-encoders)
                - tuple/list of mixed types: Non-text pairs (e.g. image + text)
                - dict: Chat messages, audio data, or multimodal inputs
                - PIL.Image.Image: Image inputs
                - np.ndarray/torch.Tensor: Audio (1-2D) or video (3-5D) inputs

        Returns:
            Tuple of (modality, processor_inputs_dict, extra_modality_kwargs) where:
                - modality: Detected modality string (``"text"``, ``"image"``, etc.) or tuple of modalities
                - processor_inputs_dict: Dictionary mapping modality names to input lists
                - extra_modality_kwargs: Extra kwargs per modality (e.g. ``sampling_rate`` for audio)
        """
        if not inputs:
            return "text", {"text": []}, defaultdict(dict)

        typed_inputs: list[tuple[Modality | Literal["pair"], Any]] = []
        extra_modality_kwargs = defaultdict(dict)
        has_pairs = False

        for item in inputs:
            # Detect non-text pairs before calling infer_modality (which would raise for them)
            # For text pairs we're fine letting them be classified as "text" modality and handled by tokenizers,
            # but non-text pairs require special handling to convert to messages with query/document roles.
            if _is_non_text_pair(item):
                typed_inputs.append(("pair", item))
                has_pairs = True
                continue

            modality = infer_modality(item, supported_modalities=self.supported_modalities)  # type: ignore[arg-type]  # non-text pairs filtered above

            # Unwrap single-key multimodal dict: {"image": pil} -> use pil as the value
            if isinstance(item, dict) and modality in MULTIMODAL_DICT_KEYS and item.keys() == {modality}:
                item = item[modality]

            # For dict-wrapped audio/video (including inside a multimodal dict), unwrap the array
            # and collect extra kwargs. For a single message dict, wrap it in a list.
            # All other values pass through as-is.
            if modality == "audio":
                value = _unwrap_audio(item, extra_modality_kwargs)
            elif modality == "video":
                value = _unwrap_video(item, extra_modality_kwargs)
            elif isinstance(modality, tuple):
                value = dict(item)
                if "audio" in value:
                    value["audio"] = _unwrap_audio(value["audio"], extra_modality_kwargs)
                if "video" in value:
                    value["video"] = _unwrap_video(value["video"], extra_modality_kwargs)
            elif modality == "message" and isinstance(item, dict):
                value = [item]
            else:
                value = item

            typed_inputs.append((modality, value))

        _reconcile_video_metadata(typed_inputs, extra_modality_kwargs)

        # Non-text pairs require conversion to message format. When the batch contains any
        # non-text pairs, ALL items must be converted to messages for consistency. Text pairs
        # (str, str) must also go through pair_to_messages so they get query/document roles.
        if has_pairs:
            messages = []
            for mod, value in typed_inputs:
                if mod == "pair":
                    messages.append(self.pair_to_messages(value))
                elif mod == "text" and self.is_text_pair(value):
                    messages.append(self.pair_to_messages(value))
                elif mod == "message":
                    messages.append(value)
                else:
                    typed = value if isinstance(mod, tuple) else {mod: value}
                    messages.append(self.to_message(typed))
            return "message", {"message": messages}, extra_modality_kwargs

        modalities, processed_inputs = zip(*typed_inputs)
        processed_inputs = list(processed_inputs)
        unique_modalities = set(modalities)

        if len(unique_modalities) == 1:
            modality = unique_modalities.pop()
            if isinstance(modality, str):
                processed_inputs = {modality: processed_inputs}
            else:
                ordered_keys = tuple(processed_inputs[0])
                if (self.supported_modalities is None or "message" in self.supported_modalities) and any(
                    tuple(entry) != ordered_keys for entry in processed_inputs[1:]
                ):
                    return (
                        "message",
                        {"message": [self.to_message(entry) for entry in processed_inputs]},
                        extra_modality_kwargs,
                    )
                processed_inputs = {mod: [entry[mod] for entry in processed_inputs] for mod in ordered_keys}
        else:
            logger.debug(f"Mixed modalities detected: {unique_modalities}. Converting to 'message' format.")
            processed_inputs = {
                "message": [
                    self.to_message(value if isinstance(modality, tuple) else {modality: value})  # type: ignore[arg-type]
                    for modality, value in typed_inputs
                ]
            }
            modality = "message"

        return modality, processed_inputs, extra_modality_kwargs

    def pair_roles_failure(self, chat_template_kwargs: dict[str, Any] | None = None) -> str | None:
        # [FUNCTION DOC]
        # Returns why the processor's chat template cannot carry a query/document pair, or None if it can. Returns a fixed
        # reason if the processor has no chat template. Uses a cached verdict when the template and kwargs equal the cached
        # snapshot. Otherwise renders three probe pairs (a baseline and one variation per role) through the processor's
        # chat-template rendering without tokenizing, passing the given kwargs; if rendering raises any exception, the reason
        # names the exception; else any role whose varied render equals the baseline is reported as not reaching the prompt.
        # Logs the verdict at debug level and caches it with a deep copy of the template and kwargs.
        """Why the chat template cannot carry a ``query``/``document`` pair, or None when it can.

        Decided by rendering probe pairs and diffing the outputs, never by reading the template source:
        templates can pass roles through without naming them (ChatML) or name them in branches a pair
        never reaches, and may transform content beyond recognition, so only a render that fails to move
        with its input proves the content goes nowhere. A template that raises is reported with the
        exception instead. The verdict is cached against a snapshot of the template and kwargs, so
        fixing either on a loaded model takes effect, which is the documented remedy.

        Args:
            chat_template_kwargs: The ``apply_chat_template`` kwargs the real render passes. These can
                select between named templates, so the probe must render with them.
        """
        template = getattr(self.processor, "chat_template", None)
        if template is None:
            return "the model has no chat template"
        chat_template_kwargs = chat_template_kwargs or {}
        key = (template, chat_template_kwargs)
        if self._pair_role_probe is not None and self._pair_role_probe[0] == key:
            return self._pair_role_probe[1]

        try:
            base, *varied = [
                str(
                    self.processor.apply_chat_template(
                        self.pair_to_messages(pair), tokenize=False, **chat_template_kwargs
                    )
                )
                for pair in PAIR_ROLE_PROBES
            ]
        except Exception as exc:
            failure = f"it raised {type(exc).__name__}: {exc}"
        else:
            dropped = [role for role, render in zip(PAIR_ROLES, varied) if render == base]
            if dropped:
                roles = " and ".join(repr(role) for role in dropped)
                failure = f"the {roles} content does not reach the rendered prompt"
            else:
                failure = None
        logger.debug(f"Pair-role probe for the chat template: {failure or 'pair content survives'}")
        self._pair_role_probe = (copy.deepcopy(key), failure)
        return failure

    def has_pair_roles(self, messages_batch: list[list[dict[str, Any]]]) -> bool:
        # [FUNCTION DOC]
        # Returns True if any conversation in the batch contains messages with both the "query" and the "document" role.
        """Whether any conversation in a batch carries both of the ``query``/``document`` pair roles.

        Anchored on produced messages so both routes into :meth:`pair_to_messages` are covered, and
        requiring both roles keeps hand-written single-role messages outside a check about pairs.
        """
        pair_roles = set(PAIR_ROLES)
        return any(pair_roles <= {message.get("role") for message in messages} for messages in messages_batch)

    def pair_to_messages(self, pair: tuple | list) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Converts a two-element pair into messages with roles "query" (first) and "document" (second); raises ValueError if
        # the pair does not have exactly two elements. Each element's modality is inferred. In flat format each element becomes
        # one message whose content is the raw element. In structured format, dicts that are already modality-keyed are used
        # as typed input, other values are wrapped under their modality key, and to_message builds the message with the role.
        # Returns the list of messages.
        """Convert a pair of inputs to query/document message format.

        Each element of the pair is wrapped in a message with role ``"query"`` (first element)
        or ``"document"`` (second element). The modality of each element is inferred individually
        via :func:`infer_modality`.

        Args:
            pair: A 2-element tuple or list of inputs (e.g. ``(image, text)``).

        Returns:
            List of two message dictionaries with ``"query"`` and ``"document"`` roles.
        """
        messages = []
        for role, item in zip(PAIR_ROLES, pair, strict=True):
            modality = infer_modality(item)
            if self.message_format == "flat":
                messages.append({"role": role, "content": item})
            else:
                typed_input = (
                    item
                    if isinstance(item, dict) and (isinstance(modality, tuple) or item.keys() == {modality})
                    else {modality: item}
                )
                messages.extend(self.to_message(typed_input, role=role))
        return messages

    def to_message(self, typed_input: dict[Modality, Any], role: str = "user") -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Converts a dict from modality to value into a single-message list with the given role (default "user"). In flat
        # format with exactly one modality, content is the raw value; with several modalities it logs a warning and falls back
        # to structured. Structured content is a list of typed entries, each holding a "type" key naming the modality and the
        # item under the modality name as key: video values that are frame sequences count as one video (tuples converted to
        # lists), other multimodal-key values that are lists/tuples are expanded to one entry per item, and any other key
        # (e.g. "message") contributes a single entry.
        """Convert a typed input dictionary to message format.

        Produces a single message with the given ``role``. For query/document pairs,
        use :meth:`pair_to_messages` instead (which is called automatically by :meth:`parse_inputs`).

        Args:
            typed_input: Dictionary mapping modalities to values. Text, image, audio, and video values can be
                lists or tuples of items. Image sequences under video represent one video's frames.
            role: Role for the message (default: ``"user"``).

        Returns:
            List of message dictionaries (single message).
        """
        if self.message_format == "flat":
            if len(typed_input) == 1:
                _, value = next(iter(typed_input.items()))
                return [{"role": role, "content": value}]
            else:
                logger.warning(
                    "Flat message format requested but multiple modalities detected. "
                    "Falling back to structured format."
                )

        content = []
        for modality, value in typed_input.items():
            if modality == "video":
                items = [value] if _is_video_frames(value) else _as_sequence(value)
                items = [list(item) if isinstance(item, tuple) else item for item in items]
            else:
                items = _as_sequence(value) if modality in MULTIMODAL_DICT_KEYS else [value]
            content.extend({"type": modality, modality: item} for item in items)

        return [
            {
                "role": role,
                "content": content,
            }
        ]

    def batch_to_message(
        self, modality: Modality, processor_inputs: dict
    ) -> tuple[Literal["message"], dict[str, list]]:
        # [FUNCTION DOC]
        # Converts grouped processor inputs (modality name to list of values) back to per-sample messages. Returns "message"
        # and a dict with the list of message lists. Empty input yields an empty list. The batch size is taken from the first
        # list. For each index, a typed input is built from the keys present in the given modality (in the input dict's order);
        # a lone text value that is a text pair goes through pair_to_messages, everything else through to_message.
        """Convert a batch of modality-specific inputs into the unified message format.

        Args:
            modality: The modality key (string) or tuple of modality keys.
            processor_inputs: Dictionary mapping modality names to lists of inputs.

        Returns:
            Tuple of ``("message", {"message": [messages_per_sample, ...]})``
        """
        if not processor_inputs:
            return "message", {"message": []}
        modalities = (modality,) if isinstance(modality, str) else modality
        batch_size = len(next(iter(processor_inputs.values())))
        messages = []
        for i in range(batch_size):
            # Use processor_inputs key order (preserves user's original dict ordering)
            typed_input = {mod: processor_inputs[mod][i] for mod in processor_inputs if mod in modalities}
            # Text pairs (e.g. ("query", "document")) are routed to pair_to_messages instead
            if len(typed_input) == 1:
                mod, value = next(iter(typed_input.items()))
                if mod == "text" and self.is_text_pair(value):
                    messages.append(self.pair_to_messages(value))
                    continue
            messages.append(self.to_message(typed_input))  # type: ignore[arg-type]
        return "message", {"message": messages}

    @staticmethod
    def is_text_pair(value: Any) -> bool:
        # [FUNCTION DOC]
        # Static: returns True if the value is a list/tuple of exactly two strings (a cross-encoder text pair).
        """Whether a single input is a text pair, i.e. cross-encoder ``(query, document)`` input."""
        return isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(part, str) for part in value)

    @staticmethod
    def is_text_only_messages(messages_batch: list[list[dict[str, Any]]]) -> bool:
        # [FUNCTION DOC]
        # Static: returns True if every message in every conversation has text-only content: string content counts as text,
        # list content is text only if every item's "type" (default "text") is "text", and any other content type returns False.
        # KNOWN ISSUE: list content items are assumed to be dicts; a list containing non-dict items raises AttributeError
        # instead of returning False.
        """Check whether all messages in a batch contain only text content.

        Works with both flat format (``{"content": "hello"}``) and structured format
        (``{"content": [{"type": "text", "text": "hello"}]}``).

        Args:
            messages_batch: List of message lists, one per sample.

        Returns:
            True if every message contains only text, False if any contain non-text content.
        """
        for messages in messages_batch:
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    continue
                if isinstance(content, list):
                    if any(item.get("type", "text") != "text" for item in content):
                        return False
                else:
                    return False
        return True

    def normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Converts messages to the configured format and returns a new list. Messages lacking "role" or "content" are skipped
        # with a warning. In flat mode, structured content with exactly one item that has a "text" key becomes that text;
        # other structured content is kept with a warning. In structured mode, string content becomes a single text item.
        # Everything else, and all extra keys, are kept unchanged.
        """Normalize messages to the target format (``self.message_format``).

        Extra keys beyond ``"role"`` and ``"content"`` are preserved during conversion.

        Args:
            messages: List of message dictionaries to normalize.

        Returns:
            Normalized list of message dictionaries.
        """
        normalized = []
        for message in messages:
            if "role" not in message or "content" not in message:
                logger.warning(f"Invalid message format: {message}. Skipping.")
                continue

            content = message["content"]
            is_currently_structured = isinstance(content, list) and content and isinstance(content[0], dict)

            if self.message_format == "flat" and is_currently_structured:
                if len(content) == 1 and "text" in content[0]:
                    normalized.append({**message, "content": content[0]["text"]})
                else:
                    logger.warning(
                        f"Cannot convert structured message to flat format: "
                        f"contains {len(content)} content items. Keeping structured."
                    )
                    normalized.append(message)
            elif self.message_format == "structured" and not is_currently_structured:
                if isinstance(content, str):
                    normalized.append({**message, "content": [{"type": "text", "text": content}]})
                else:
                    normalized.append(message)
            else:
                normalized.append(message)

        return normalized

    def prepend_prompt_to_messages(
        self, messages: list[list[dict[str, Any]]], prompt: str
    ) -> list[list[dict[str, Any]]]:
        # [FUNCTION DOC]
        # Returns new message lists with a system message holding the prompt prepended to each; the system content is the raw
        # prompt in flat format or a single text item in structured format.
        """Prepend a system prompt to message format inputs.

        Args:
            messages: List of message lists (each message list represents one input).
            prompt: System prompt to prepend.

        Returns:
            Messages with system prompt prepended to each message list.
        """
        if self.message_format == "flat":
            return [[{"role": "system", "content": prompt}] + message_list for message_list in messages]
        return [
            [{"role": "system", "content": [{"type": "text", "text": prompt}]}] + message_list
            for message_list in messages
        ]

    def prepend_prompt_to_texts(
        self, texts: list[str | tuple[str, str] | list[str]], prompt: str
    ) -> list[str | list[str]]:
        # [FUNCTION DOC]
        # Returns a new list where each string has the prompt prepended, and for pairs (any non-string sequence) the prompt is
        # prepended to the first element only, producing a list.
        """Prepend a prompt to text format inputs.

        For single texts, prepends the prompt directly.
        For text pairs (cross-encoder inputs), prepends only to the first text.

        Args:
            texts: List of text inputs (strings or pairs)
            prompt: Prompt to prepend

        Returns:
            Texts with prompt prepended
        """
        result = []
        for text in texts:
            if isinstance(text, str):
                result.append(prompt + text)
            else:
                result.append([prompt + text[0]] + list(text[1:]))
        return result


def infer_modality(
    sample: SingleInput | PairInput | Any,
    supported_modalities: list[Modality] | None = None,
) -> Modality:
    # [FUNCTION DOC]
    # Infers the modality of one sample by type and structure, without a processor. Order: PIL image is "image", audio
    # decoder "audio", video decoder "video". Strings: image-like, video-like or audio-like URLs/paths map to that modality
    # unless supported_modalities is given and excludes it (then "text"); other strings and two-string tuples/lists are
    # "text". A message dict, or a non-empty list starting with one, is "message". Dicts with "array" plus "sampling_rate"
    # are "audio", with "array" plus "video_metadata" are "video", and with "array" alone raise ValueError. Other non-empty
    # dicts must only use the multimodal keys (else ValueError, with hints when "video_metadata" or "sampling_rate" are
    # used as sibling keys); a single key returns that name, several keys return the sorted tuple of names. Empty dicts
    # raise ValueError. Arrays/tensors: 1-2 dims "audio", 3 dims "image", 4-5 dims "video", other ranks ValueError. Any
    # other type raises ValueError.
    """Infer the modality of a single input sample by inspecting its type/structure.

    Pure type-based detection, does not require a processor or tokenizer.

    Args:
        sample: A single input sample to inspect.
        supported_modalities: Optional list of modalities the model supports. When provided,
            string inputs that would be classified as image/video/audio based on URL/path
            heuristics are instead classified as ``"text"`` if that modality is not supported.
            This prevents misclassification of text that happens to contain media URLs.

    Returns:
        The detected modality string, or a tuple of modality strings for multimodal dict inputs
        with 2+ keys. A 1-key dict like ``{"image": pil}`` collapses to the bare modality string.

    Raises:
        ValueError: If the input type/structure is not recognized.
    """
    # Not a part of the match statement as it would match None if PIL is not installed
    if PILImage is not None and isinstance(sample, PILImage):
        return "image"

    if AudioDecoder is not None and isinstance(sample, AudioDecoder):
        return "audio"

    if VideoDecoder is not None and isinstance(sample, VideoDecoder):
        return "video"

    match sample:
        case str() if is_image_url_or_path(sample):
            if supported_modalities is not None and "image" not in supported_modalities:
                return "text"
            return "image"
        case str() if is_video_url_or_path(sample):
            if supported_modalities is not None and "video" not in supported_modalities:
                return "text"
            return "video"
        case str() if is_audio_url_or_path(sample):
            if supported_modalities is not None and "audio" not in supported_modalities:
                return "text"
            return "audio"
        case str() | (str(), str()) | [str(), str()]:
            return "text"
        case dict() if is_message_dict(sample):
            return "message"
        case list() if sample and is_message_dict(sample[0]):
            return "message"
        case dict() if "array" in sample and "sampling_rate" in sample:
            return "audio"
        case dict() if "array" in sample and "video_metadata" in sample:
            return "video"
        case dict() if "array" in sample:
            raise ValueError(
                "Dict input with 'array' key must also include 'sampling_rate' (for audio) "
                "or 'video_metadata' (for video). "
                f"Got keys: {sorted(sample.keys(), key=str)}"
            )
        case dict() if sample:
            # Single-key dicts collapse to the bare modality string so {"image": pil} matches
            # the same support/routing as a raw PIL input.
            invalid_keys = set(sample.keys()) - MULTIMODAL_DICT_KEYS
            if invalid_keys:
                # Per-sample metadata as a sibling key is a common mistake, so let's point at the array-dict
                # form that actually carries it, nested under its modality key so that it still composes
                # with any sibling modalities in the same sample.
                hints = []
                if "video_metadata" in invalid_keys:
                    hints.append(
                        "To attach per-sample video metadata, pass the video itself as a dict, and keep "
                        "processor options in processing_kwargs, e.g.:\n"
                        '{"video": {"array": frames, "video_metadata": {"fps": 15, "total_num_frames": 3}}}\n'
                        'processing_kwargs={"video": {"do_sample_frames": False}}'
                    )
                if "sampling_rate" in invalid_keys:
                    hints.append(
                        "To attach the sampling rate, pass the audio itself as a dict, e.g.:\n"
                        '{"audio": {"array": waveform, "sampling_rate": 16000}}'
                    )
                hint = (" " + "\n\n".join(hints)) if hints else ""
                raise ValueError(
                    f"Multimodal dict input contains unrecognized modality keys: {sorted(invalid_keys, key=str)}. "
                    f"Expected keys from: {sorted(MULTIMODAL_DICT_KEYS)}.{hint}"
                )
            if len(sample) == 1:
                return next(iter(sample))
            # Multimodal dict: keys are modality names (sorted for consistent route lookups)
            return tuple(sorted(sample.keys()))
        case dict():
            raise ValueError("Empty dict input is not a valid input sample.")
        case np.ndarray() | torch.Tensor():
            if sample.ndim in (1, 2):  # mono or multi-channel waveform
                return "audio"
            elif sample.ndim == 3:  # (H, W, C) or (C, H, W)
                return "image"
            elif sample.ndim in (4, 5):  # (frames, C, H, W) or (batch, frames, C, H, W)
                return "video"
            else:
                raise ValueError(
                    f"Unsupported tensor dimensionality: {sample.ndim}D. "
                    f"Expected 1-2D for audio, 3D for image, or 4-5D for video."
                )
        case _:
            raise ValueError(
                f"Unsupported input type: {type(sample).__name__}. "
                f"Expected one of: str, dict, PIL.Image.Image, np.ndarray, torch.Tensor"
            )


def infer_batch_modality(
    samples: Sequence[SingleInput | PairInput],
    supported_modalities: list[Modality] | None = None,
) -> Modality:
    # [FUNCTION DOC]
    # Returns the shared modality of all samples, "message" if they differ, or "text" for an empty batch. Propagates
    # ValueError from per-sample inference.
    """Infer the modality of a batch of input samples.

    If all samples share the same modality, that modality is returned. If the batch contains
    mixed modalities, ``"message"`` is returned, consistent with how :class:`InputFormatter`
    handles mixed-modality batches in :meth:`~InputFormatter.parse_inputs`.

    Args:
        samples: List of input samples to inspect.
        supported_modalities: Optional list of modalities the model supports. Passed through
            to :func:`infer_modality` to prevent misclassification of text as media modalities.

    Returns:
        The detected modality, or ``"message"`` for mixed-modality batches.
    """
    if not samples:
        return "text"
    modalities = {infer_modality(sample, supported_modalities=supported_modalities) for sample in samples}
    return modalities.pop() if len(modalities) == 1 else "message"


def format_modality(modality: Modality) -> str:
    # [FUNCTION DOC]
    # Returns the modality name, or a tuple's names joined with "+".
    """Format a modality for display, e.g. ``("text", "image")`` becomes ``"text+image"``."""
    if isinstance(modality, tuple):
        return "+".join(modality)
    return modality


def raise_unsupported_modality_error(
    inputs: Sequence[SingleInput | PairInput],
    modality: Modality,
    supported_modalities: list[Modality],
    source: str,
) -> NoReturn:
    # [FUNCTION DOC]
    # Always raises ValueError with a message tailored to why the modality is unsupported by the named source. For the
    # "message" batch modality it re-infers each sample (ignoring inference errors): all-message inputs or a mix including
    # messages report that chat-style inputs are unsupported; other mixes report the base modalities the source lacks, or,
    # if all parts are supported, that they cannot be combined in one batch. A tuple modality whose parts are all supported
    # reports that they cannot be fused in one input. Otherwise reports the modality as unsupported. All messages list the
    # supported modalities.
    """Raise a clear ``ValueError`` explaining why ``modality`` is not supported.

    Shared by :meth:`BaseModel.preprocess` and :meth:`Transformer.preprocess` so both raise the
    same, accurate guidance. ``source`` is a human-readable description of what rejected the input,
    e.g. ``"SentenceTransformer model"`` or ``"Transformer module"``.

    Args:
        inputs: The original batch of inputs, re-inspected per sample to distinguish explicit
            chat-style ``"message"`` inputs from inferred mixed-modality batches (both collapse to
            the ``"message"`` modality at the batch level).
        modality: The inferred (and unsupported) batch modality.
        supported_modalities: The modalities the source actually supports.
        source: Human-readable name of the rejecting component, used verbatim in the message.

    Raises:
        ValueError: Always, with a message tailored to the specific unsupported-modality scenario.
    """
    supported_str = ", ".join(format_modality(m) for m in supported_modalities)

    if modality == "message":
        # "message" is returned both for explicit chat-style inputs and for inferred mixed-modality
        # batches. Re-inspect each sample to tell them apart. If a sample cannot be classified on its
        # own (e.g. a non-text pair), fall back to the generic message below.
        try:
            sample_modalities: set[Modality] = {
                infer_modality(sample, supported_modalities=supported_modalities) for sample in inputs
            }
        except (ValueError, TypeError):
            sample_modalities = set()

        if sample_modalities == {"message"}:
            raise ValueError(
                f"This {source} does not support chat-style 'message' inputs (dicts or lists of dicts with "
                f"'role' and 'content'). Supported modalities: {supported_str}."
            )
        if "message" in sample_modalities:
            # Chat-style inputs are mixed with other inputs. Since this source does not support the
            # message format, the chat-style inputs are the blocking issue. Report that rather than
            # listing "message" as if it were a content modality alongside text/image/audio.
            raise ValueError(
                f"This {source} does not support chat-style 'message' inputs (dicts or lists of dicts with 'role' and "
                f"'content'), which this batch mixes with other modalities. Supported modalities: {supported_str}."
            )
        if sample_modalities:
            present_str = ", ".join(format_modality(m) for m in sorted(sample_modalities, key=format_modality))
            # Decompose into base parts: a part the source cannot handle at all is a genuine error.
            # If every part is supported, the modalities just cannot be combined (handled below).
            base_modalities = {part for m in sample_modalities for part in (m if isinstance(m, tuple) else (m,))}
            unsupported = sorted(part for part in base_modalities if part not in supported_modalities)
            if unsupported:
                raise ValueError(
                    f"This batch mixes multiple modalities ({present_str}), but this {source} does "
                    f"not support {', '.join(unsupported)}. Supported modalities: {supported_str}."
                )
            raise ValueError(
                f"This batch mixes multiple modalities ({present_str}), which this {source} cannot "
                f"encode in a single batch. Encode each modality separately (one call per modality)."
            )

    # A single combined input (e.g. a {"text": ..., "image": ...} dict) with parts the source
    # supports individually but cannot fuse without "message" support: encode each separately.
    if isinstance(modality, tuple) and all(part in supported_modalities for part in modality):
        raise ValueError(
            f"This {source} supports {' and '.join(modality)} individually, but cannot "
            f"combine them in a single input. Encode each modality separately (one call per modality)."
        )

    raise ValueError(
        f"Modality '{format_modality(modality)}' is not supported by this {source}. "
        f"Supported modalities: {supported_str}."
    )
