# [MODULE DOC]
# Interpretability helpers for ColPali-style multi-vector models over image documents: locating the real
# query tokens, inferring the image patch grid, computing per-query-token similarity maps over patches, and
# rendering them as colored heatmap overlays. Builds a 256-entry color lookup table approximating seaborn's
# "mako" colormap at import time (global constant). PIL and transformers image utilities are imported lazily.
"""Per-query-token MaxSim heatmap utilities for ColPali-style image documents."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor

from sentence_transformers.base.modules import Transformer

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

# Seaborn's "mako" colormap from 9 sampled keys (linearly interpolated). <11/255 max RGB error
# vs the actual 256-entry seaborn LUT, so the look matches without depending on matplotlib.
_CMAP_KEYS = np.array(
    [
        [11, 4, 5],
        [43, 28, 53],
        [62, 53, 107],
        [59, 86, 152],
        [53, 123, 163],
        [53, 159, 171],
        [75, 194, 173],
        [153, 221, 182],
        [222, 245, 229],
    ],
    dtype=np.float32,
)


def _build_mako_lut(n: int = 256) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: Build an n-entry RGB lookup table by linear interpolation between the nine mako key colors evenly
    # spaced over the table. Returns an n by 3 uint8 array.
    positions = np.linspace(0, len(_CMAP_KEYS) - 1, n)
    indices = positions.astype(int).clip(0, len(_CMAP_KEYS) - 2)
    fractions = (positions - indices)[:, None]
    interp = _CMAP_KEYS[indices] + fractions * (_CMAP_KEYS[indices + 1] - _CMAP_KEYS[indices])
    return interp.astype(np.uint8)


_MAKO_LUT = _build_mako_lut()


def real_query_token_slice(model: MultiVectorEncoder, query: str) -> slice:
    # [FUNCTION DOC]
    # Return the slice of the query's encoded token sequence holding the real content tokens (excluding prompt
    # prefixes, chat-template special tokens, markers, suffixes and expansion tokens). Finds the model's first
    # Transformer module (ValueError if none). Tokenizes an empty query and the actual query with the model's
    # "query" prompt and query task; the slice start is the length of their common prefix of ids. Length is the
    # query's own token count without special tokens when query expansion is configured (since both renders are
    # padded to the same length), otherwise the difference in token counts between the two renders.
    """Return the slice into ``encode_query``'s output that selects the real content tokens.

    Chat-template prefixes (e.g. ``<bos>``), ColBERT query markers (e.g. "[Q] "), suffixes (e.g.
    ``<|im_end|>``), and ``MultiVectorEncoder``'s query-expansion tokens (``<mask>`` / ``<pad>``) wrap
    the actual query and carry attention-sink signals that distort heatmap visualisations. Slicing
    them out keeps only the real tokens:

    .. code-block:: python

        s = real_query_token_slice(model, query)
        query_embedding = model.encode_query([query])[0][s]

    Works for both right-padded (PaliGemma) and left-padded (ColQwen2 / ColModernVBert)
    backbones by comparing the encoded sequences of an empty and the actual query.
    """
    transformer = next((module for module in model if isinstance(module, Transformer)), None)
    if transformer is None:
        raise ValueError(
            "real_query_token_slice needs a Transformer module to re-tokenize the query, but this "
            f"model has none: {[type(module).__name__ for module in model]}."
        )
    # Render with the same prompt encode_query() would apply, otherwise a marker prefix like
    # ColBERT's "[Q] " lands inside the slice and pushes the last real token out of it.
    prompt = model.prompts.get("query")
    empty_ids = transformer.preprocess([""], prompt=prompt, task="query")["input_ids"][0].tolist()
    query_ids = transformer.preprocess([query], prompt=prompt, task="query")["input_ids"][0].tolist()

    prefix = 0
    while prefix < min(len(empty_ids), len(query_ids)) and empty_ids[prefix] == query_ids[prefix]:
        prefix += 1
    expansion = getattr(transformer, "query_expansion", None)
    tokenizer = transformer.tokenizer
    if expansion is not None and tokenizer is not None:
        # Both renders are padded to exactly the expansion length, so the length diff is always 0.
        # Expansion models are plain text tokenizers (the pad strategies raise on chat-template
        # backbones), so the content length is simply the query's own token count.
        n_content = len(tokenizer(query, add_special_tokens=False)["input_ids"])
    else:
        n_content = len(query_ids) - len(empty_ids)
    return slice(prefix, prefix + n_content)


def get_n_patches(model: MultiVectorEncoder, image_size: tuple[int, int]) -> tuple[int, int]:
    # [FUNCTION DOC]
    # Infer the (columns, rows) patch grid the model's processor produces for an image of the given (width,
    # height). Raises NotImplementedError for split-image processors (those with image-splitting or max image
    # size settings). If the image processor has a merge size, processes a blank RGB image of that size and
    # returns grid width and height (from the image grid shape) each divided by the merge size. Else, if the
    # processor's image sequence length is a positive perfect square, returns its square root for both. Otherwise
    # raises ValueError.
    """Infer the ``(n_patches_x, n_patches_y)`` = ``(n_cols, n_rows)`` image-patch grid that
    ``model``'s processor produces for an image of ``image_size`` (``(width, height)``, i.e. PIL's
    ``image.size``). Plugs directly into the ``n_patches`` parameter of
    :func:`maxsim_similarity_map` and :func:`maxsim_heatmap`.

    Dynamic grids (Qwen-VL family) are read from the image processor's own ``image_grid_thw``
    output for a blank image of that size, fixed square grids (PaliGemma / Gemma3) from
    ``processor.image_seq_length``. Idefics3 / SmolVLM split-image processors raise
    :class:`NotImplementedError`: their sub-patch token order is not a plain row-major grid.
    """
    processor = model.processor
    image_processor = getattr(processor, "image_processor", None)
    if (
        getattr(image_processor, "do_image_splitting", None) is not None
        or getattr(image_processor, "max_image_size", None) is not None
    ):
        raise NotImplementedError(
            "Idefics3 / SmolVLM style processors split each image into sub-patch token blocks followed "
            "by a trailing global patch, which is not a row-major (rows, cols) grid: a plain reshape "
            "would scramble the heatmap. Rearrange the embedding into spatial order first, e.g. with "
            "colpali_engine's Idefics3SplitImageInterpretabilityMixin."
        )

    merge_size = getattr(image_processor, "merge_size", None)
    if image_processor is not None and merge_size is not None:
        from PIL import Image

        # The (t, h, w) grid is in pre-merge patch units: merge_size spatial-merges those into tokens.
        grid = image_processor(images=[Image.new("RGB", image_size)], return_tensors="pt")["image_grid_thw"][0]
        return int(grid[2]) // merge_size, int(grid[1]) // merge_size

    image_seq_length = getattr(processor, "image_seq_length", None)
    if isinstance(image_seq_length, int) and image_seq_length > 0:
        side = math.isqrt(image_seq_length)
        if side * side == image_seq_length:
            return side, side

    raise ValueError(
        f"Could not infer a patch grid from {type(processor).__name__}: needs an image_processor with "
        "a Qwen-VL style dynamic grid (merge_size + image_grid_thw) or a square processor.image_seq_length "
        "(PaliGemma / Gemma3). Pass n_patches to maxsim_similarity_map / maxsim_heatmap manually."
    )


def maxsim_similarity_map(
    query_embedding: Tensor,
    image_embedding: Tensor,
    n_patches: tuple[int, int],
    image_mask: Tensor | None = None,
    normalize: bool = False,
) -> Tensor:
    # [FUNCTION DOC]
    # Compute per-query-token similarity over the image patch grid. Inputs: query token embeddings (Qt by D),
    # image token embeddings (Dt by D), the (columns, rows) grid, optional boolean mask selecting patch tokens,
    # and a normalize flag. Applies the mask, raises ValueError if the remaining token count is not columns
    # times rows, reshapes row-major into rows by columns by D, and returns dot products as float32 shaped Qt by
    # rows by columns. With normalize, min-max scales each query token's map to the unit range.
    """Per-query-token similarity over a 2D image-patch grid.

    Args:
        query_embedding: ``(Qt, D)`` per-token query embeddings.
        image_embedding: ``(Dt, D)`` per-token image-document embeddings. Pass ``image_mask`` if
            ``Dt`` includes non-image tokens, otherwise ``Dt == n_patches[0] * n_patches[1]``.
        n_patches: ``(n_cols, n_rows)`` = ``(width, height)``, as returned by
            :func:`get_n_patches`. Patches are row-major.
        image_mask: boolean mask over the ``Dt`` axis (``True`` for image-patch tokens).
        normalize: rescale each per-query-token map to ``[0, 1]``.

    Returns:
        Tensor of shape ``(Qt, n_rows, n_cols)``, float32 regardless of the input dtype.
    """
    if image_mask is not None:
        image_embedding = image_embedding[image_mask.bool()]
    n_cols, n_rows = n_patches
    expected = n_cols * n_rows
    if image_embedding.shape[0] != expected:
        raise ValueError(
            f"image_embedding has {image_embedding.shape[0]} tokens but n_patches={n_patches} "
            f"expects {expected}. Pass image_mask if the embedding includes non-image tokens, or "
            "set MultiVectorMask.keep_only_token_ids to the image-patch token id during encoding."
        )
    grid = image_embedding.view(n_rows, n_cols, -1)
    # Input-precision matmul, fp32 for the downstream heatmap sums, matching maxsim.
    similarity_map = torch.einsum("qd,rcd->qrc", query_embedding, grid).float()
    if normalize:
        flat = similarity_map.reshape(similarity_map.shape[0], -1)
        min_vals = flat.min(dim=-1, keepdim=True).values
        max_vals = flat.max(dim=-1, keepdim=True).values
        similarity_map = ((flat - min_vals) / (max_vals - min_vals + 1e-10)).reshape(-1, n_rows, n_cols)
    return similarity_map


def render_similarity_map_on_image(
    image: PILImage | str,
    similarity_map: Tensor,
    alpha: float = 0.5,
    normalization_range: tuple[float, float] | None = None,
) -> PILImage:
    # [FUNCTION DOC]
    # Overlay a 2D similarity map on an image. Inputs: a PIL image or URL/path (loaded via transformers, may do
    # network or file I/O), a rows by columns map, a constant opacity, and an optional (min, max) range (default
    # the map's own range). Raises ValueError if the map is not 2D. Min-max scales into the unit range with
    # clipping, converts to 8-bit, bicubically upsamples to the image size, maps through the mako lookup table to
    # RGB, adds a constant alpha channel, and returns a new RGBA image alpha-composited over the original.
    """Overlay a 2D similarity map onto an image with the mako colormap.

    Args:
        image: PIL image, URL, or local file path (loaded via :func:`transformers.image_utils.load_image`).
        similarity_map: ``(n_rows, n_cols)`` similarity tensor.
        alpha: constant overlay opacity in ``[0, 1]``.
        normalization_range: ``(min, max)`` for the colour scale. Defaults to the map's own range.
            Pass a shared range across multiple per-token maps so they render proportionally.

    Returns:
        A new RGBA PIL image.
    """
    from PIL import Image as PILImageModule
    from transformers.image_utils import load_image

    if similarity_map.ndim != 2:
        raise ValueError(
            f"render_similarity_map_on_image expects a 2D (n_rows, n_cols) map, got shape {tuple(similarity_map.shape)}."
        )
    image = load_image(image) if not isinstance(image, PILImageModule.Image) else image
    sm = similarity_map.detach().to(torch.float32).cpu().numpy()
    sm_min, sm_max = (float(sm.min()), float(sm.max())) if normalization_range is None else normalization_range
    sm = ((sm - sm_min) / (sm_max - sm_min + 1e-10)).clip(0.0, 1.0)

    # Look up colours AFTER the bicubic upsample so every output pixel stays on the colormap
    # curve (interpolating R/G/B separately can drift off-curve at edges).
    sm_uint = (sm * 255).astype(np.uint8)
    sm_upsampled = np.array(PILImageModule.fromarray(sm_uint).resize(image.size, PILImageModule.Resampling.BICUBIC))
    rgba = np.empty((*sm_upsampled.shape, 4), dtype=np.uint8)
    rgba[..., :3] = _MAKO_LUT[sm_upsampled]
    rgba[..., 3] = int(alpha * 255)

    heatmap = PILImageModule.fromarray(rgba)
    return PILImageModule.alpha_composite(image.convert("RGBA"), heatmap)


def maxsim_heatmap(
    image: PILImage | str,
    query_embedding: Tensor,
    image_embedding: Tensor,
    n_patches: tuple[int, int],
    image_mask: Tensor | None = None,
    aggregate: Literal["sum", "amax", "none"] = "sum",
    alpha: float = 0.5,
) -> PILImage | list[PILImage]:
    # [FUNCTION DOC]
    # One-shot heatmap for a query and image document. Loads the image once, computes the unnormalized
    # similarity map, and then: for "none", renders one overlay per query token sharing a global color range
    # and returns the list; for "sum" or "amax", collapses over query tokens by sum or maximum and returns a
    # single overlay. Raises ValueError for any other aggregate value.
    """One-shot MaxSim heatmap for a (query, image-document) pair.

    Args:
        image: PIL image, URL, or local file path.
        query_embedding: ``(Qt, D)`` per-token query embeddings.
        image_embedding: ``(Dt, D)`` per-token image-document embeddings.
        n_patches: ``(n_cols, n_rows)`` image grid shape, as returned by :func:`get_n_patches`.
        image_mask: optional mask filtering ``image_embedding`` to image patches only.
        aggregate: ``"sum"`` adds every query token's similarity at each patch, giving an aggregate
            similarity heatmap rather than an attribution of the MaxSim score, which counts only each
            query token's single best patch. ``"amax"`` shows the strongest single-token match per
            patch. ``"none"`` returns one heatmap per query token as a list.
        alpha: constant overlay opacity in ``[0, 1]``.

    Returns:
        A single PIL image for ``aggregate in {"sum", "amax"}``, or a list for ``"none"``.
    """
    from PIL import Image as PILImageModule
    from transformers.image_utils import load_image

    # Resolve once so URL fetches aren't repeated per token in the loop below.
    image = load_image(image) if not isinstance(image, PILImageModule.Image) else image
    similarity_map = maxsim_similarity_map(
        query_embedding=query_embedding,
        image_embedding=image_embedding,
        n_patches=n_patches,
        image_mask=image_mask,
    )

    if aggregate == "none":
        # Share one colour scale across the per-token maps so they render proportionally (each map
        # normalized to its own range would make weak tokens look as hot as strong ones).
        shared_range = (float(similarity_map.min()), float(similarity_map.max()))
        return [
            render_similarity_map_on_image(image, similarity_map[i], alpha=alpha, normalization_range=shared_range)
            for i in range(similarity_map.shape[0])
        ]
    if aggregate == "sum":
        collapsed = similarity_map.sum(dim=0)
    elif aggregate == "amax":
        collapsed = similarity_map.amax(dim=0)
    else:
        raise ValueError(f"aggregate must be one of ('sum', 'amax', 'none'), got {aggregate!r}.")
    return render_similarity_map_on_image(image, collapsed, alpha=alpha)


__all__ = [
    "get_n_patches",
    "maxsim_heatmap",
    "maxsim_similarity_map",
    "real_query_token_slice",
    "render_similarity_map_on_image",
]
