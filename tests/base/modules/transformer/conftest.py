# [MODULE DOC]
# HELPER: Shared pytest configuration and sample-data generators for the per-task Transformer module tests
# (feature extraction, fill mask, sequence classification, text generation, any-to-any).
# At import time it: optionally imports torchcodec's audio and video decoders (set to None if unavailable) and
# torchaudio (records availability in a private flag); skips the whole module tree when running in CI (the tests
# download many Hub models); and loads "transformers_tiny_models.json" (path relative to the repo root, so tests
# must run from there) into a mapping of architecture key to tiny Hub model id.
# It defines module-level lists/dicts that tests use to mark expectations per architecture: known-broken
# architectures to xfail (in general and only on transformers v4), faulty checkpoints, CUDA-only architectures,
# architectures whose forward pass is expected to fail (None meaning all modalities, else a list of the
# sample-description strings expected to fail), and architectures with expected multimodal success/failure,
# image+video failure, image-only failure and audio failure.
# It also keeps a module-level list of temporary media file paths created by the generators, deleted at session end.
# Each generator returns a dict mapping an input-format name to a list of n samples.
"""Shared fixtures, constants, and sample generators for Transformer task tests."""

from __future__ import annotations

import json
import random
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from sentence_transformers.base.modules import Transformer
from tests.utils import is_ci

try:
    from torchcodec.decoders import AudioDecoder, VideoDecoder
except (ImportError, OSError):
    AudioDecoder = None
    VideoDecoder = None

try:
    import torchaudio

    _has_torchaudio = True
except (ImportError, OSError):
    _has_torchaudio = False

if is_ci():
    pytest.skip(
        "Skipping tests that load a considerable amount of models from the Hugging Face Hub in CI environment",
        allow_module_level=True,
    )

# Curated list of tiny models from hf-internal-testing and tiny-random organizations
# These are pre-built tiny models specifically designed for testing
with open("tests/base/modules/transformer/transformers_tiny_models.json", encoding="utf8") as f:
    TINY_MODEL_MAPPING: dict[str, str] = json.load(f)
    # TINY_MODEL_MAPPING = {arch: model for arch, model in TINY_MODEL_MAPPING.items() if arch == "regnet"}

# Architectures that don't work nicely, but that are low-priority to fix
XFAIL_ARCHITECTURES = [
    # Can't read hidden_size from config
    "cvt",
    "mobilenet_v1",
    "mobilenet_v2",
    "mobilevitv2",
    "perceiver",
    "univnet",
    # Model doesn't output last_hidden_state on forward/expose get_..._features methods
    "dac",
    "dpr",
    "encodec",
    "lxmert",  # Integration could be possible as it does output e.g. language_hidden_states, but low-prio
    "mask2former",  # TODO: Integration could be possible as it does output e.g. encoder_last_hidden_state
    "maskformer",  # Integration could be possible as it does output e.g. encoder_last_hidden_state, but low-prio
    "mimi",
    "oneformer",
    "sam",  # TODO: Integration could be possible as it does have get_image_embeddings
    "seamless_m4t_v2",  # Integration could be possible as it does output e.g. encoder_last_hidden_state, but low-prio
    "vits",
    # Model last_hidden_state on forward or get_..._features doesn't match expected embedding dimension
    "fsmt",
    # Other issues
    "bark",  # Processor doesn't accept 'padding' in text_kwargs as it hardcodes its padding strategy
    "bros",  # Has a hard requirement for bbox, low prio
    "lilt",  # Lilt requires bounding boxes
    "seggpt",  # Has a requirement for prompt_pixel_values and prompt_masks via prompt_images
    "speecht5",  # SpeechT5 is an encoder-decoder that either loads a SpeechT5EncoderWithTextPrenet or SpeechT5EncoderWithSpeechPrenet. There's also no subconfigs to determine which should be used, low prio
    "timesformer",  # Timesformer doesn't nicely preserve the batch dimension when processing videos, low prio
    "speech_to_text",  # Tricky case as it's an encoder-decoder with unusual inputs, low prio
    # Transformers issues
    "focalnet",  # TODO: FocalNetModel.input_modalities still defaults to 'text'
    "fuyu",  # get_image_features accepts pixel_values, but the image processor only returns image_patches (identical, just different name), low prio
    "gemma3n",  # TODO: The Gemma3nModel type hint is not correct, plus the Gemma3nProcessor requires text at all times, despite code where text is created if it doesn't exist, high prio
    "tvp",  # TODO: TvpProcessor.get_attributes() returns ['image_processor', 'tokenizer'] instead of ['video_processor', 'tokenizer']
    "videomae",  # TODO: Models often load with VideoMAEImageProcessor accepting 'images' instead of 'videos'
    "musicgen",  # not compatible with AutoModel
    "musicgen_melody",  # not compatible with AutoModel
    "voxtral",  # VoxtralProcessor doesn't accept audio, only text, while the VoxtralEncoder only processes audio
    "deepseek_v4",  # FP8 checkpoint: save_pretrained re-quantizes to float8_e8m0fnu, which safetensors 0.5.3 cannot serialize. Forward also mixes float32 norms with bf16 Linears
    # Skipped due to extra dependencies
    "layoutlmv2",  # Requires detectron2
    "layoutlmv3",  # Requires pytesseract
    "udop",  # Requires pytesseract
    "dinat",  # Requires natten
    "roformer",  # Requires rjieba
]
TRANSFORMERS_V4_XFAIL_ARCHITECTURES = [
    "blip-2",  # get_text_features is updated in v5
    "chinese_clip",  # get_text_features fails in v4 if there's no pooler
    "flava",  # FlavaModel.forward was incorrectly typed in v4
    "imagegpt",  # ImageGPTProcessor in Transformers v4 doesn't support image URLs or paths. v5 works fine
    "visual_bert",  # Loads a tokenizer, even for text+image
    # Unrecognized processing class
    "splinter",
    "mistral",
    "blenderbot-small",
]
# Model checkpoints that are faulty and cannot be loaded by transformers currently
FAULTY_CHECKPOINTS = [
    "data2vec-audio",  # No vocab file
    "mgp-str",  # Missing preprocessor_config.json / vocab file
    "patchtsmixer",  # Missing processor files
    "patchtst",  # Missing processor files
    "vision-text-dual-encoder",  # preprocessor_config.json mentions now-removed ViTFeatureExtractor instead of ViTImageProcessor
    "yoso",  # tokenizer_config.json tokenizer_class is set to AlbertTokenizer, while it should likely be e.g. TokenizersBackend
]

REQUIRES_CUDA = [
    "qwen2_moe",  # Forward pass uses `torch._grouped_mm`, but data requires 16-bytes alignment on CPU, which isn't guaranteed. Perhaps fixed in https://github.com/pytorch/pytorch/pull/173395, but not yet released.
    "mra",
]
# Models whose checkpoints are faulty, but only with the 'forward', and I still want to test loading, saving,
# configuration reading, etc. None denotes that failure is across all modalities, but for some models, the checkpoint
# may only be faulty for certain modalities (e.g. image), so we can still test other modalities (e.g. text). The
# value is then the modality description that is expected to fail.
EXPECT_FORWARD_FAIL = {
    "llava": None,  # Checkpoint input embedding size is 99, but tokenizer vocab size is the full size
    "markuplm": None,  # Checkpoint xpath embedding size is 20, but xpath_tags_seq is much higher
    "mistral3": None,  # Checkpoint input embedding size is 32k, but tokenizer vocab size is 125k+
    "roc_bert": None,  # Checkpoint shape_embed size is 99, but input_shape_ids is up to 20k+
    "qwen3_next": None,  # Checkpoint config has 2 num_hidden_layers, both linear_attention, but the Qwen3NextDynamicCache requires at least one full_attention layer
    "tapas": None,  # Doesn't accept standard truncation strategies, needs TapasTruncationStrategy, low prio
    "align": [  # Checkpoint image_processor size/crop_size set to 600, doesn't align with num layers, etc., low prio
        "image (url)",
        "image (array)",
        "image (tensor)",
        "image (pil)",
        "image (path)",
    ],
    "clap": [  # Checkpoint has num_mel_bins mismatch, and perhaps audio is too long
        "audio (url)",
        "audio (array)",
        # "audio (tensor)",
        "audio (dict)",
        "audio (path)",
        "audio (audio_decoder)",
    ],
    "idefics": None,  # Checkpoint has "image_size": 224, which I can't override as image_size is normally a dict
    "idefics2": [  # Idefics2 doesn't accept image array/tensors
        "image (array)",
        "image (tensor)",
    ],
    "paligemma": [  # Paligemma doesn't accept URL images if there's also a text
        "image+text (url, text)"
    ],
    "voxtral_realtime": None,  # RuntimeError: The size of tensor a (39) must match the size of tensor b (51) at non-singleton dimension 1
}
# If an architecture outputs sentence_embeddings directly, then they're likely using get_..._features,
# which typically don't support multimodal inputs, so we can expect multimodal inputs to fail for those architectures
# However, there's some exceptions where the model does output sentence embeddings directly, but still supports
# multimodal inputs (e.g. blip), so this is a list of architectures that we still want to test.
EXPECT_MULTIMODAL_SUCCESS = [
    "blip",  # Custom case as it supports text+image with get_multimodal_features and outputs sentence embeddings directly
]
# Some custom cases output token_embeddings via get_..._features, and don't support multimodal inputs.
# We expect failure for multimodal here.
EXPECT_MULTIMODAL_FAILURE = [
    "blip-2",  # Custom case that outputs text or image separately as token_embeddings
    "sam3",  # Custom case that outputs text or image separately as token_embeddings
    "flava",  # Custom case that outputs text or image separately as token_embeddings
]

# Some architectures fail when combining image+video in one input, but support image and video separately, so we can
# expect failure for that specific combination
EXPECT_IMAGE_VIDEO_FAILURE = [
    "glm4v",  # glm4v and glm4v_moe use image tokens for videos as well, and thus lose the ability to assign image features to the "real" image tokens,
    "glm4v_moe",  # and the video features to the image tokens used for videos.
    "glm_ocr",
]
EXPECT_IMAGE_ONLY_FAILURE = [
    "llama4",  # Text is required at all times  # TODO: Or should I update Transformer? But this supports 'message' with only image I think?
]
EXPECT_AUDIO_FAILURE = [
    "gemma4",  # tiny checkpoint ships no audio tower weights, despite architecture supporting audio
]


# Track temporary media files for cleanup
_temp_media_files: list[str] = []


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_media_files():
    # [FUNCTION DOC]
    # HELPER: Session-scoped autouse generator fixture. Yields immediately; at session teardown deletes every file path
    # recorded in the temporary media list if it exists, printing (not raising) any error encountered.
    """Cleanup temporary media files after all tests."""
    yield
    for file_path in _temp_media_files:
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
        except Exception as e:
            print(f"Failed to cleanup {file_path}: {e}")


@pytest.fixture(scope="session", autouse=True)
def patch_torchcodec_int64_indices():
    # [FUNCTION DOC]
    # HELPER: Session-scoped autouse generator fixture that works around a Windows int32 issue in torchcodec. If the
    # video decoder class is unavailable it just yields. Otherwise it replaces the decoder class's "get frames at
    # indices" method with a wrapper that converts NumPy index arrays to 64-bit integer tensors before delegating to
    # the original method, yields for the session, and always restores the original method afterwards. Mutates global
    # class state.
    """Coerce numpy frame indices to int64 for torchcodec on Windows.

    transformers' video samplers return numpy int arrays, which are int32 on Windows
    (where np.dtype(int) is int32). torchcodec's get_frames_at_indices only up-casts
    torch.Tensor/list inputs to int64, not numpy, so its op raises "expected scalar type
    Long but found Int". This is a no-op on platforms where numpy ints are already int64.
    """
    if VideoDecoder is None:
        yield
        return

    original_get_frames_at = VideoDecoder.get_frames_at

    def get_frames_at(self, indices):
        # [FUNCTION DOC]
        # HELPER: Replacement method installed on the video decoder. Inputs: the decoder instance and frame indices. If
        # the indices are a NumPy array they are converted to an int64 torch tensor; then calls the original method and
        # returns its result.
        if isinstance(indices, np.ndarray):
            indices = torch.as_tensor(indices).to(torch.int64)
        return original_get_frames_at(self, indices)

    VideoDecoder.get_frames_at = get_frames_at
    try:
        yield
    finally:
        VideoDecoder.get_frames_at = original_get_frames_at


# Sample data generation functions for each modality type
# Each returns a dict mapping format names to lists of samples
@lru_cache(maxsize=4)
def get_sample_text(n: int = 2) -> dict[str, list[str]]:
    # [FUNCTION DOC]
    # HELPER: Returns sample text inputs. Input n (default 2). Returns a dict with the single key "text" mapping to n
    # strings of the form "This is sentence i for testing." (i from 0). Results are memoised per n (LRU cache of 4),
    # so callers share the same list objects.
    """Generate sample text data.

    Returns:
        Dict with format names as keys and lists of text samples as values.
        Format: 'text' only (for compatibility with messages)
    """
    return {
        "text": [f"This is sentence {i} for testing." for i in range(n)],
    }


@lru_cache(maxsize=4)
def get_sample_images(n: int = 2) -> dict[str, list[Any]]:
    # [FUNCTION DOC]
    # HELPER: Returns n sample images (default 2) in several formats, memoised per n (LRU cache of 4). Keys: "url"
    # (up to two fixed Hub dataset PNG URLs, truncated to n, so fewer than n if n exceeds 2), "array" (random 32x32x3
    # uint8 arrays from the global NumPy RNG), "tensor" (the same arrays as float tensors scaled to 0..1, in HWC
    # layout), "pil" (PIL images from the same arrays) and "path" (paths of newly written temporary PNG files holding
    # different random images). Side effects: writes temp files and records them for session cleanup.
    """Generate sample image data in multiple formats.

    Returns:
        Dict with format names as keys:
        - 'url': List of image URLs
        - 'tensor': List of PyTorch tensors
        - 'array': List of numpy arrays
        - 'pil': List of PIL Images
        - 'path': List of local file paths
    """
    urls = [
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_image_0.png",
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_image_1.png",
    ][:n]

    # Generate PIL images
    pil_images = []
    tensors = []
    arrays = []
    for _ in range(n):
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        arrays.append(img_array)
        tensors.append(torch.from_numpy(img_array).float() / 255.0)
        pil_images.append(Image.fromarray(img_array))

    # Generate local file paths
    paths = []
    for i in range(n):
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(temp_file.name)
        temp_file.close()
        paths.append(temp_file.name)
        _temp_media_files.append(temp_file.name)

    return {
        "url": urls,
        "array": arrays,
        "tensor": tensors,
        "pil": pil_images,
        "path": paths,
    }


@lru_cache(maxsize=4)
def get_sample_audio(n: int = 2) -> dict[str, list[Any]]:
    # [FUNCTION DOC]
    # HELPER: Returns n sample audio inputs (default 2), memoised per n. Creates n random float32 mono waveforms of
    # 1000 to 2000 samples at 16 kHz (global NumPy and Python RNGs). If torchaudio is available, writes each to a
    # temporary WAV file (recorded for cleanup); otherwise uses the fixed URLs as stand-in paths. Returns a dict with
    # "array" (the raw arrays) and "dict" (dicts with array and sampling-rate keys), plus "audio_decoder" (torchcodec
    # audio decoders over the WAV paths) when both torchcodec and torchaudio are available.
    # KNOWN ISSUE: The docstring promises "url" and "path" keys, but those entries are commented out and never returned;
    # the computed URLs and paths are only used for the optional audio decoders.
    """Generate sample audio data in multiple formats.

    Returns:
        Dict with format names as keys:
        - 'url': List of audio URLs (16kHz)
        - 'array': List of numpy arrays
        - 'dict': List of dicts with 'array' and 'sampling_rate' keys
        - 'path': List of local file paths
    """
    urls = [
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_audio_0.wav",
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_audio_1.wav",
    ][:n]

    # Generate arrays (16kHz, 1000-2000 samples)
    sampling_rate = 16000
    arrays = [np.random.randn(random.randint(1000, 2000)).astype(np.float32) for _ in range(n)]
    # tensors = [torch.from_numpy(arr) for arr in arrays]
    # max_length = max(tensor.shape[0] for tensor in tensors)
    # tensors = [torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[0])) for tensor in tensors]

    # Generate dicts with array and sampling_rate
    audio_dicts = [{"array": arr, "sampling_rate": sampling_rate} for arr in arrays]

    # Generate local file paths
    paths = []
    if _has_torchaudio:
        for i, arr in enumerate(arrays):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.close()
            tensor = torch.from_numpy(arr).unsqueeze(0).float()  # (1, num_samples) for torchaudio
            torchaudio.save(temp_file.name, tensor, sampling_rate)
            paths.append(temp_file.name)
            _temp_media_files.append(temp_file.name)
    else:
        # Fallback if torchaudio not available
        paths = urls[:n]

    result = {
        # "url": urls,  # Rarely supported currently
        "array": arrays,
        # "tensor": tensors,  # Tensors is not supported in the generic transformers feature_extraction
        "dict": audio_dicts,
        # "path": paths,  # Rarely supported currently
    }

    if AudioDecoder is not None and _has_torchaudio:
        result["audio_decoder"] = [AudioDecoder(path) for path in paths]

    return result


@lru_cache(maxsize=4)
def get_sample_video(n: int = 2) -> dict[str, list[Any]]:
    # [FUNCTION DOC]
    # HELPER: Returns n sample videos (default 2), memoised per n. Downloads up to two fixed Hub MP4 URLs into
    # temporary files (network I/O; recorded for cleanup). Returns a dict with "url" and "path". When torchcodec's
    # video decoder is available it additionally decodes all frames of each video and adds "tensor" (dicts holding the
    # frame tensor and video metadata: fps, total frame count, duration, and frame indices 0..N-1), "array" (same but
    # with a NumPy frame array), and "video_decoder" (a fresh decoder per path).
    # KNOWN ISSUE: The docstring does not mention the returned "video_decoder" key, and n above 2 silently yields only
    # two samples.
    """Generate sample video data in multiple formats.

    Returns:
        Dict with format names as keys:
        - 'url': List of video URLs
        - 'path': List of locally downloaded file paths
        - 'array': List of numpy arrays (if torchcodec available, extracted from videos)
        - 'tensor': List of PyTorch tensors (if torchcodec available, extracted from videos)
    """
    urls = [
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_video_0.mp4",
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_video_1.mp4",
    ][:n]

    # Download URLs for local file paths
    paths = []
    for i, url in enumerate(urls):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        urllib.request.urlretrieve(url, temp_file.name)
        temp_file.close()
        paths.append(temp_file.name)
        _temp_media_files.append(temp_file.name)

    result = {
        "url": urls,
        "path": paths,
    }

    if VideoDecoder is not None:
        arrays = []
        tensors = []
        for path in paths:
            decoder = VideoDecoder(path)
            frame_batch = decoder.get_frames_in_range(0, decoder.metadata.num_frames)
            metadata = {
                "fps": decoder.metadata.average_fps_from_header,
                "total_num_frames": decoder.metadata.num_frames,
                "duration": decoder.metadata.duration_seconds,
                "frames_indices": list(range(frame_batch.data.shape[0])),
            }
            tensors.append({"array": frame_batch.data, "video_metadata": metadata})
            arrays.append({"array": frame_batch.data.numpy(), "video_metadata": metadata})

        if arrays:
            result["array"] = arrays
        if tensors:
            result["tensor"] = tensors

        result["video_decoder"] = [VideoDecoder(path) for path in paths]

    return result


def get_sample_messages(
    n: int = 2, text_format: str = "text", message_format: str = "structured"
) -> list[list[dict[str, Any]]]:
    # [FUNCTION DOC]
    # HELPER: Builds n chat-format samples, each a one-message conversation with role "user". Inputs: n, the text
    # format key to pull from the sample text generator (falls back to "text" if missing), and the message format.
    # "structured" makes content a list with one dict of type "text" holding the text; "flat" makes content the raw
    # string. Raises ValueError for any other message format. Returns the list of conversations.
    """Generate sample message data (chat format with text).

    Args:
        n: Number of messages to generate
        text_format: Text format to use - "text" only (URL and path formats not supported in messages)
        message_format: Message format - "structured" or "flat"
            - structured: content is list of dicts with type/modality keys
            - flat: content is direct value (string only)

    Returns:
        List of message lists in the specified format
    """
    # Get text samples (only 'text' format is supported for messages)
    text_samples = get_sample_text(n)
    texts = text_samples.get(text_format, text_samples.get("text", []))

    if message_format == "structured":
        messages_list = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in texts]
    elif message_format == "flat":
        messages_list = [[{"role": "user", "content": text}] for text in texts]
    else:
        raise ValueError(f"Unknown format: {message_format}. Must be 'structured' or 'flat'")

    return messages_list


def get_sample_text_pairs(n: int = 2) -> dict[str, list[tuple[str, str]]]:
    # [FUNCTION DOC]
    # HELPER: Returns a dict with key "text_pair" mapping to n (query, document) tuples of the form "Query i",
    # "Document i".
    """Generate sample text pair data for cross-encoder style testing."""
    return {
        "text_pair": [(f"Query {i}", f"Document {i}") for i in range(n)],
    }


def create_modality_pair_samples(model: Transformer, modalities: list[str], n: int = 2) -> dict[str, list[tuple]]:
    # [FUNCTION DOC]
    # HELPER: Builds pair inputs (for cross-encoder style models) for every supported modality combination. Inputs:
    # the model (unused), the list of supported modalities (strings; non-string entries and "message" are excluded
    # from the per-modality list), and n. If "text" is supported, adds key "text+text pair" with the text pairs. If
    # "message" is supported, takes the first format variant of each modality that has a generator and, for every
    # ordered pair of those modalities (including same-modality pairs, but skipping text with text), zips the two
    # sample lists into pairs under the key "A+B pair (formatA, formatB)". Returns the dict of description to pairs.
    """Create test pair samples for all supported modality combinations.

    Generates pairs for:
    1. Text pairs: (text, text) - handled natively by the tokenizer
    2. Cross-modality pairs: (modality_a, modality_b) for each combination of non-message modalities
       These are converted to message format with query/document roles.
    3. Same-modality pairs: (modality, modality) for each non-text modality

    Uses the first format variant for each modality to keep the number of test cases manageable.

    Args:
        model: The Transformer model instance
        modalities: List of supported modalities (e.g., ["text", "image", "message"])
        n: Number of pair samples to generate for each combination

    Returns:
        Dict mapping pair_description to list of pair tuples where:
        - pair_description: Human-readable string like "text+text pair", "image+text pair (pil, text)", etc.
        - pairs: List of (query, document) tuples
    """
    from itertools import product

    samples = {}
    non_message_modalities = [m for m in modalities if m != "message" and isinstance(m, str)]

    # 1. Text pairs (always test if text is supported)
    if "text" in non_message_modalities:
        text_pairs = get_sample_text_pairs(n)["text_pair"]
        samples["text+text pair"] = text_pairs

    # 2. Cross-modality and same-modality pairs (require message support)
    has_message_support = "message" in modalities
    if has_message_support:
        # Get first format variant for each modality
        modality_samples: dict[str, tuple[str, list]] = {}
        for mod in non_message_modalities:
            generator = MODALITY_SAMPLE_GENERATORS.get(mod)
            if not generator:
                continue
            format_variants = generator(n)
            first_format_name, first_format_inputs = next(iter(format_variants.items()))
            modality_samples[mod] = (first_format_name, first_format_inputs)

        # Generate all ordered pairs (including same-modality)
        mods_with_samples = list(modality_samples.keys())
        for mod_a, mod_b in product(mods_with_samples, repeat=2):
            # Skip text+text as it's already handled above (natively by tokenizer)
            if mod_a == "text" and mod_b == "text":
                continue

            fmt_a, inputs_a = modality_samples[mod_a]
            fmt_b, inputs_b = modality_samples[mod_b]
            pairs = list(zip(inputs_a, inputs_b))
            desc = f"{mod_a}+{mod_b} pair ({fmt_a}, {fmt_b})"
            samples[desc] = pairs

    return samples


# Mapping of modality keys to sample data generators
# Note: Each generator returns a dict of format_name -> list of samples
MODALITY_SAMPLE_GENERATORS = {
    "text": get_sample_text,
    "image": get_sample_images,
    "audio": get_sample_audio,
    "video": get_sample_video,
}


def create_modality_samples(
    model: Transformer, modalities: list[str], n: int = 2, message_format: str = "structured"
) -> dict[str, list[Any]]:
    # [FUNCTION DOC]
    # HELPER: Builds single-input samples for every supported modality and combination. Inputs: the model (unused),
    # the modality list (strings are single modalities except "message"; tuples/lists denote explicit multimodal
    # combinations), n, and the message format. Produces keys:
    # 1. "modality (format)" for every format variant of every single modality with a generator.
    # 2. For every combination of two or more single modalities, "a+b (fmtA, fmtB)" mapping to n dicts keyed by
    # modality, using each modality's first format variant.
    # 3. For each explicit multimodal tuple, the same kind of entry, added only if that key is not already present.
    # 4. If "message" is supported: text messages ("message (text, structured)" and/or "message (text, flat)"
    # depending on the message format; both for any value other than structured/flat); for each non-text modality,
    # a structured message whose content item has that modality as type and the first format sample as value,
    # under "modality as message (structured, fmt)"; and for every pair of single modalities, a structured
    # message with one content item per modality under "a+b as message (structured, fmtA, fmtB)".
    # Returns the dict. Raises KeyError if a combined modality has no generator.
    """Create test samples for all supported modalities and their combinations.

    Generates samples for:
    1. Each single modality with different input formats (e.g., image as URL, tensor, or path)
    2. Multi-modal combinations (e.g., text+image as dict inputs)
    3. Message format variations (structured and flat) if "message" is supported
    4. Converting text modality to message format (other formats not supported in messages)

    Args:
        model: The Transformer model instance
        modalities: List of supported modalities (e.g., ["text", "image", "message"])
        n: Number of samples to generate for each combination
        message_format: Message format - "structured" or "flat"
            - structured: content is list of dicts with type/modality keys
            - flat: content is direct value (string only)

    Returns:
        Dict mapping modality_description to inputs where:
        - modality_description: Human-readable string like "text", "image (url)", "image (path)", "text+image (text, url)", etc.
        - inputs: List of input samples in the appropriate format
    """
    from itertools import combinations

    samples = {}

    # Separate message modality from others
    non_message_modalities = [m for m in modalities if m != "message" and isinstance(m, str)]
    multimodal_modalities = [m for m in modalities if isinstance(m, (tuple, list))]
    has_message_support = "message" in modalities

    # 1. Test each single non-message modality with each format variant
    for modality in non_message_modalities:
        generator = MODALITY_SAMPLE_GENERATORS.get(modality)
        if not generator:
            continue

        # Generator returns dict of format_name -> list of samples
        format_variants = generator(n)
        for format_name, inputs in format_variants.items():
            modality_desc = f"{modality} ({format_name})"
            samples[modality_desc] = inputs

    # 2. Test multi-modal combinations (2 or more modalities combined)
    # For simplicity, uses the first format variant for each modality
    if len(non_message_modalities) >= 2:
        for r in range(2, len(non_message_modalities) + 1):
            for combo in combinations(non_message_modalities, r):
                # Generate data for each modality (use first format variant)
                modality_data = {}
                format_names = []
                for mod in combo:
                    generator = MODALITY_SAMPLE_GENERATORS[mod]
                    format_variants = generator(n)
                    # Use the first format variant for multimodal combinations
                    first_format = next(iter(format_variants.items()))
                    modality_data[mod] = first_format[1]
                    format_names.append(first_format[0])

                # Create list of dicts, one per sample
                # Each dict has keys for each modality with the corresponding value
                inputs = [{mod: modality_data[mod][i] for mod in combo} for i in range(n)]
                modality_desc = "+".join(combo) + " (" + ", ".join(format_names) + ")"
                samples[modality_desc] = inputs

    for multimodal_modality in multimodal_modalities:
        modality_data = {}
        format_names = []
        for mod in multimodal_modality:
            generator = MODALITY_SAMPLE_GENERATORS[mod]
            format_variants = generator(n)
            first_format = next(iter(format_variants.items()))
            modality_data[mod] = first_format[1]
            format_names.append(first_format[0])

        # Create list of dicts, one per sample
        inputs = [{mod: modality_data[mod][i] for mod in multimodal_modality} for i in range(n)]
        modality_desc = "+".join(multimodal_modality) + " (" + ", ".join(format_names) + ")"
        if modality_desc not in samples:
            samples[modality_desc] = inputs

    # 3. Test message format variations (if message modality is supported)
    # Note: Only text format is supported in messages (not URLs or paths)
    if has_message_support:
        # 3a. Test text in message format (only text format, structured and flat)
        if message_format == "structured":
            structured_msgs = get_sample_messages(n, text_format="text", message_format="structured")
            samples["message (text, structured)"] = structured_msgs
        elif message_format == "flat":
            flat_msgs = get_sample_messages(n, text_format="text", message_format="flat")
            samples["message (text, flat)"] = flat_msgs
        else:
            # Support both if message_format is something else
            structured_msgs = get_sample_messages(n, text_format="text", message_format="structured")
            samples["message (text, structured)"] = structured_msgs
            flat_msgs = get_sample_messages(n, text_format="text", message_format="flat")
            samples["message (text, flat)"] = flat_msgs

        # 3b. Convert each non-message modality's first format to message format (structured only)
        # Note: Other modalities in messages only work with structured format
        for modality in non_message_modalities:
            if modality == "text":
                # Skip text since we already handled it above
                continue

            generator = MODALITY_SAMPLE_GENERATORS.get(modality)
            if not generator:
                continue

            # Get the first format variant
            format_variants = generator(n)
            first_format_name, first_format_inputs = next(iter(format_variants.items()))

            # Create structured messages with this modality
            structured_msgs = [
                [{"role": "user", "content": [{"type": modality, modality: inp}]}] for inp in first_format_inputs
            ]
            samples[f"{modality} as message (structured, {first_format_name})"] = structured_msgs

        # 3c. Multimodal messages (combining multiple modalities in message format)
        # Only structured format and first format variant for each modality
        if len(non_message_modalities) >= 2:
            # Test pairs (limit complexity to avoid explosion of test cases)
            for combo in combinations(non_message_modalities, 2):
                # Generate data for each modality (use first format variant)
                modality_data = {}
                format_names = []
                for mod in combo:
                    generator = MODALITY_SAMPLE_GENERATORS[mod]
                    format_variants = generator(n)
                    first_format = next(iter(format_variants.items()))
                    modality_data[mod] = first_format[1]
                    format_names.append(first_format[0])

                # Create multimodal messages with structured content
                multimodal_msgs = [
                    [{"role": "user", "content": [{"type": mod, mod: modality_data[mod][i]} for mod in combo]}]
                    for i in range(n)
                ]
                modality_desc = "+".join(combo) + " as message (structured, " + ", ".join(format_names) + ")"
                samples[modality_desc] = multimodal_msgs

    return samples


def get_arch_kwargs(arch: str, transformer_task: str) -> tuple[dict, dict, dict, dict]:
    # [FUNCTION DOC]
    # HELPER: Returns per-architecture overrides needed for tiny checkpoints to run, as four dicts: model kwargs
    # (always ignoring mismatched sizes), config kwargs, processor kwargs, and processing kwargs (per-modality
    # preprocessing options). Inputs: architecture key and transformer task. Applies a fixed set of hard-coded tweaks
    # (e.g. image token id for blip-2, image sizes for flava, decoder mode for reformer text generation, video frame
    # count for gemma4, padding to max length for mra).
    """Get model_kwargs, config_kwargs, processor_kwargs, processing_kwargs for a given architecture.

    Returns:
        Tuple of (model_kwargs, config_kwargs, processor_kwargs, processing_kwargs)
    """
    model_kwargs = {"ignore_mismatched_sizes": True}
    config_kwargs = {}
    processor_kwargs = {}
    processing_kwargs = {}

    # Resolve some minor issues in the checkpoints that prevent forward testing
    if arch == "blip-2":
        config_kwargs["image_token_id"] = 4
    if arch == "flava":
        processor_kwargs["size"] = {"height": 30, "width": 30}
        processor_kwargs["crop_size"] = {"height": 30, "width": 30}
    if arch == "kosmos-2":
        config_kwargs["latent_query_num"] = 64
    if arch == "vilt":
        processor_kwargs["size_divisor"] = 4
    if arch == "whisper":
        config_kwargs["max_source_positions"] = 1500
    if arch == "idefics":
        processor_kwargs["additional_special_tokens"] = []
    if arch == "marian":
        config_kwargs["pad_token_id"] = 1
    if arch == "qwen2_5_vl":
        config_kwargs["hidden_size"] = 16
    if arch == "reformer" and transformer_task == "text-generation":
        config_kwargs["is_decoder"] = True
    if arch == "internvl":
        processing_kwargs["text"] = {"truncation": False}
    if arch == "mra":
        processor_kwargs["model_max_length"] = 256
        processing_kwargs["text"] = {"padding": "max_length"}
    if arch == "qwen3_5" and transformer_task == "sequence-classification":
        config_kwargs["text_config"] = {"num_labels": 1, "id2label": {0: "LABEL_0"}}
    if arch in ("gemma4", "diffusion_gemma"):
        processing_kwargs["video"] = {"num_frames": 8}
    if arch == "gemma4":
        processing_kwargs["audio"] = {"truncation": False}

    return model_kwargs, config_kwargs, processor_kwargs, processing_kwargs


def load_transformer(arch: str, transformer_task: str = "feature-extraction", **extra_kwargs) -> Transformer:
    # [FUNCTION DOC]
    # HELPER: Loads a Sentence Transformers Transformer module for an architecture. Inputs: architecture key (looked
    # up in the tiny model mapping; KeyError if absent), the transformer task (default feature extraction), and extra
    # keyword arguments forwarded to the Transformer constructor. First tries loading with local-files-only set on
    # model, config and processor kwargs (cache only); on any exception retries allowing downloads. Then, if the module
    # has a tokenizer, sets missing tokenizer pad/eos token ids to 0 and missing pad/eos token ids on the model config
    # and its text sub-config to 0. For text-generation and any-to-any tasks, replaces a generation-config pad token id
    # of -1 with the eos token id. Returns the module.
    """Load a Transformer model for the given architecture and task.

    Args:
        arch: Architecture name from TINY_MODEL_MAPPING
        transformer_task: The transformer task (feature-extraction, sequence-classification, text-generation)
        **extra_kwargs: Extra kwargs passed to Transformer()

    Returns:
        Loaded Transformer model
    """
    model_name = TINY_MODEL_MAPPING[arch]
    model_kwargs, config_kwargs, processor_kwargs, processing_kwargs = get_arch_kwargs(arch, transformer_task)

    try:
        model = Transformer(
            model_name,
            transformer_task=transformer_task,
            model_kwargs={**model_kwargs, "local_files_only": True},
            config_kwargs={**config_kwargs, "local_files_only": True},
            processor_kwargs={**processor_kwargs, "local_files_only": True},
            processing_kwargs=processing_kwargs,
            **extra_kwargs,
        )
    except Exception:
        model = Transformer(
            model_name,
            transformer_task=transformer_task,
            model_kwargs=model_kwargs,
            config_kwargs=config_kwargs,
            processor_kwargs=processor_kwargs,
            processing_kwargs=processing_kwargs,
            **extra_kwargs,
        )

    if model.tokenizer:
        # Ensure pad_token_id and eos_token_id are set to avoid issues during testing
        if model.tokenizer.pad_token_id is None:
            model.tokenizer.pad_token_id = 0
        if model.tokenizer.eos_token_id is None:
            model.tokenizer.eos_token_id = 0
        configs = [model.config]
        text_config = model.config.get_text_config()
        if text_config is not model.config:
            configs.append(text_config)
        for config in configs:
            if getattr(config, "pad_token_id", False) is None:
                config.pad_token_id = 0
            if getattr(config, "eos_token_id", False) is None:
                config.eos_token_id = 0

    # Required for saving llama models to disk
    if transformer_task in ("text-generation", "any-to-any") and model.model.generation_config.pad_token_id == -1:
        model.model.generation_config.pad_token_id = model.model.generation_config.eos_token_id

    return model


def modify_processor_for_pairs(model: Transformer):
    # [FUNCTION DOC]
    # HELPER: Mutates the processor's chat template string in place so it accepts "query" and "document" roles
    # wherever it compared against "user", and removes several hard-coded role-alternation exception statements
    # (for gemma3, ministral3, llama and cohere2 templates). Input: the Transformer module; returns nothing. Raises
    # AttributeError if the processor has no chat template string.
    model.processor.chat_template = (
        model.processor.chat_template.replace('== "user"', 'in ["user", "query", "document"]')
        .replace("== 'user'", "in ['user', 'query', 'document']")
        .replace(  # Required for gemma3:
            '{{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}', ""
        )
        .replace(  # Required for ministral3
            "{{- raise_exception('After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results.') }}",
            "",
        )
        .replace(  # Required for llama
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}", ""
        )
        .replace(  # Required for cohere2
            '{{- raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") -}}', ""
        )
    )
