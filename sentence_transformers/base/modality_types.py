# [MODULE DOC]
# HELPER: type definitions and constants for multimodal input handling. Optionally imports the PIL image class and the
# torchcodec audio/video decoder classes (falling back to None at runtime when unavailable; type checkers always see the
# real classes). Defines typed dicts for wrapped audio (array plus sampling rate), wrapped video (array or frame list
# plus optional metadata), chat messages (role plus content) and multimodal inputs (optional text/image/audio/video
# keys, each a single value or a list/tuple). Defines type aliases for each per-modality input, single inputs, pair
# inputs for cross encoders, the Modality identifier (a modality name or a tuple of names for multimodal dicts), the
# processor argument names and message formats ("auto", "structured", "flat"), the mapping from modality name to
# processor keyword argument name (image to images, video to videos, others unchanged), and the frozen set of valid
# multimodal dict keys.
"""Type definitions and constants for modality handling across different input types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

import numpy as np
import torch

# Type checkers reject possibly-None variables in type expressions, which would leave the aliases
# below partially unknown. Import cleanly for them here, keeping the None fallbacks runtime-only.
if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from torchcodec.decoders import AudioDecoder, VideoDecoder
else:
    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        PILImage = None

    try:
        from torchcodec.decoders import AudioDecoder, VideoDecoder
    except (ImportError, OSError, RuntimeError):
        AudioDecoder = None
        VideoDecoder = None


# Structured input dicts for audio and video, wrapping raw arrays with metadata
class AudioDict(TypedDict):
    # [CLASS DOC]
    # HELPER: typed dict for an audio sample: "array" (numpy array or torch tensor waveform) and "sampling_rate" (int).
    array: np.ndarray | torch.Tensor
    sampling_rate: int


class VideoDict(TypedDict):
    # [CLASS DOC]
    # HELPER: typed dict for a video sample: "array" (array, tensor, or list/tuple of image frames) and "video_metadata"
    # (dict of metadata such as fps and frame count, or None).
    array: np.ndarray | torch.Tensor | list[ImageInput] | tuple[ImageInput, ...]
    video_metadata: dict[str, Any] | None


class MessageDict(TypedDict):
    # [CLASS DOC]
    # HELPER: typed dict for one chat message: "role" (str) and "content" (a string or a list of typed content dicts).
    role: str
    content: str | list[dict[str, Any]]


# Per-modality input types: each defines the accepted formats for a single modality
TextInput: TypeAlias = str
ImageInput: TypeAlias = str | PILImage | np.ndarray | torch.Tensor
AudioInput: TypeAlias = str | np.ndarray | torch.Tensor | AudioDict | AudioDecoder
VideoInput: TypeAlias = (
    str | np.ndarray | torch.Tensor | VideoDict | VideoDecoder | list[ImageInput] | tuple[ImageInput, ...]
)
MessageInput: TypeAlias = MessageDict | list[MessageDict]


class MultimodalInput(TypedDict, total=False):
    # [CLASS DOC]
    # HELPER: typed dict (all keys optional) for one sample combining modalities; each of "text", "image", "audio" and
    # "video" may hold a single item of that modality or a list/tuple of them.
    """One input combining modalities, with optional collections under singular keys."""

    text: TextInput | list[TextInput] | tuple[TextInput, ...]
    image: ImageInput | list[ImageInput] | tuple[ImageInput, ...]
    audio: AudioInput | list[AudioInput] | tuple[AudioInput, ...]
    video: VideoInput | list[VideoInput] | tuple[VideoInput, ...]


SingleInput: TypeAlias = TextInput | ImageInput | AudioInput | VideoInput | MessageInput | MultimodalInput

# Pair types for cross-encoder
PairableInput: TypeAlias = TextInput | ImageInput | AudioInput | VideoInput | MultimodalInput
PairInput: TypeAlias = tuple[PairableInput, PairableInput] | list[PairableInput]

# Modality identifier: a single modality string, or a tuple for multimodal dict inputs
Modality: TypeAlias = (
    Literal["text", "image", "audio", "video", "message"] | tuple[Literal["text", "image", "audio", "video"], ...]
)

# Internal types used in Transformer, Router, InputFormatter
ProcessorArgName: TypeAlias = Literal["text", "images", "audio", "videos", "message"]
MessageFormat: TypeAlias = Literal["auto", "structured", "flat"]
MODALITY_TO_PROCESSOR_ARG: dict[Modality, ProcessorArgName] = {
    "text": "text",
    "image": "images",
    "audio": "audio",
    "video": "videos",
    "message": "message",
}

MULTIMODAL_DICT_KEYS = frozenset({"text", "image", "audio", "video"})
