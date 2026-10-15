# [MODULE DOC]
# Embedding quantization and approximate/exact semantic search over (possibly quantized) corpora
# using the optional faiss and usearch libraries (imported lazily inside the functions).
# Quantization maps float embeddings to float32, int8/uint8 (per-dimension 256-bucket scalar
# quantization from min/max ranges) or binary/ubinary (sign bits packed into bytes). The search
# functions optionally oversample with quantized vectors and rescore using the original float
# queries. Results are lists (one per query) of dicts with "corpus_id" and "score". Logs warnings.
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import faiss
    import usearch


def semantic_search_faiss(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray | None = None,
    corpus_index: faiss.Index | None = None,
    corpus_precision: Literal["float32", "uint8", "ubinary"] = "float32",
    top_k: int = 10,
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
    rescore: bool = True,
    rescore_multiplier: int = 2,
    exact: bool = True,
    output_index: bool = False,
) -> tuple[list[list[dict[str, int | float]]], float, faiss.Index]:
    # [FUNCTION DOC]
    # Semantic search with faiss.
    # Inputs: query embeddings; exactly one of corpus embeddings or a prebuilt faiss index; corpus
    # precision ("float32", "uint8", "ubinary"); top_k; optional ranges / calibration embeddings for
    # uint8 quantization of the queries; rescore flag and multiplier; exact flag; output_index flag.
    # Raises ValueError if both or neither corpus source is given, or the precision is unsupported.
    # If no index is given, builds one: float32/uint8 use an exact inner-product flat index or a
    # 32-neighbour HNSW index (exact false); ubinary uses a binary flat or binary HNSW index with
    # dimension in bits (bytes times 8); then adds the corpus.
    # If the queries are not int8/uint8: when rescore is on and precision is not float32, keeps the
    # float queries for rescoring and multiplies k by rescore_multiplier (rescore with a float32
    # corpus only logs a warning); then quantizes the queries to the corpus precision. If the queries
    # are already quantized and rescore is on, logs a warning.
    # Searches k neighbours (timed). When rescoring, reconstructs each hit's stored vector (zero
    # placeholder for -1 padding), unpacks bits for ubinary, casts to integers, computes the dot
    # product with the float query, forces padding slots to negative infinity, keeps the top_k by
    # rescored score and reorders indices/scores accordingly.
    # Returns (results, elapsed seconds) or (results, elapsed seconds, index) if output_index; padding
    # hits with id -1 are dropped from the results.
    # KNOWN ISSUE: with exact false and float32/uint8 precision the HNSW flat index uses faiss's default L2 metric, not inner product, so scores are distances (lower is better) and inconsistent with the exact inner-product path.
    # KNOWN ISSUE: without rescoring, binary indexes return Hamming distances (lower is better) yet they are reported as "score" in ascending order.
    """
    Performs semantic search using the FAISS library.

    Rescoring will be performed if:
    1. `rescore` is True
    2. The query embeddings are not quantized
    3. The corpus is quantized, i.e. the corpus precision is not float32
    Only if these conditions are true, will we search for `top_k * rescore_multiplier` samples and then rescore to only
    keep `top_k`.

    Args:
        query_embeddings: Embeddings of the query sentences. Ideally not
            quantized to allow for rescoring.
        corpus_embeddings: Embeddings of the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both. The embeddings can be quantized to "uint8" or
            "ubinary" for more efficient search.
        corpus_index: FAISS index for the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both.
        corpus_precision: Precision of the corpus embeddings. The
            options are "float32", "uint8", or "ubinary". Default is
            "float32".
        top_k: Number of top results to retrieve. Default is 10.
        ranges: Ranges for quantization of embeddings. This is only used
            for uint8 quantization, where the ranges refer to the
            minimum and maximum values for each dimension. So, it's a 2D
            array with shape (2, embedding_dim). Default is None, which
            means that the ranges will be calculated from the
            calibration embeddings.
        calibration_embeddings: Embeddings used for calibration during
            quantization. This is only used for uint8 quantization, where
            the calibration embeddings can be used to compute ranges,
            i.e. the minimum and maximum values for each dimension.
            Default is None, which means that the ranges will be
            calculated from the query embeddings. This is not
            recommended.
        rescore: Whether to perform rescoring. Note that rescoring still
            will only be used if the query embeddings are not quantized
            and the corpus is quantized, i.e. the corpus precision is
            not "float32". Default is True.
        rescore_multiplier: Oversampling factor for rescoring. The code
            will now search `top_k * rescore_multiplier` samples and
            then rescore to only keep `top_k`. Default is 2.
        exact: Whether to use exact search or approximate search.
            Default is True.
        output_index: Whether to output the FAISS index used for the
            search. Default is False.

    Returns:
        A tuple containing a list of search results and the time taken
        for the search. If `output_index` is True, the tuple will also
        contain the FAISS index used for the search.

    Raises:
        ValueError: If both `corpus_embeddings` and `corpus_index` are
            provided or if neither is provided.
        ValueError: If `corpus_precision` is not "float32", "uint8", or
            "ubinary".

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    The time taken for the search is a float value.
    """
    import faiss

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")
    if corpus_precision not in ["float32", "uint8", "ubinary"]:
        raise ValueError('corpus_precision must be "float32", "uint8", or "ubinary" for faiss')

    # If corpus_index is not provided, create a new index
    if corpus_index is None:
        if corpus_precision in ("float32", "uint8"):
            if exact:
                corpus_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
            else:
                corpus_index = faiss.IndexHNSWFlat(corpus_embeddings.shape[1], 16)

        elif corpus_precision == "ubinary":
            if exact:
                corpus_index = faiss.IndexBinaryFlat(corpus_embeddings.shape[1] * 8)
            else:
                corpus_index = faiss.IndexBinaryHNSW(corpus_embeddings.shape[1] * 8, 16)

        corpus_index.add(corpus_embeddings)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # rescore_multiplier
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if rescore:
            if corpus_precision != "float32":
                rescore_embeddings = query_embeddings
                k *= rescore_multiplier
            else:
                logger.warning(
                    "Rescoring is enabled but the corpus is not quantized. Either pass `rescore=False` or "
                    'quantize the corpus embeddings with `quantize_embeddings(embeddings, precision="...") `'
                    'and pass `corpus_precision="..."` to `semantic_search_faiss`.'
                )

        query_embeddings = quantize_embeddings(
            query_embeddings,
            precision=corpus_precision,
            ranges=ranges,
            calibration_embeddings=calibration_embeddings,
        )
    elif rescore:
        logger.warning(
            "Rescoring is enabled but the query embeddings are quantized. Either pass `rescore=False` or don't quantize the query embeddings."
        )

    # Perform the search using the FAISS index
    start_t = time.time()
    scores, indices = corpus_index.search(query_embeddings, k)

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None:
        # FAISS pads short result sets with index -1, and reconstruct(-1) reads out of bounds
        # (or segfaults on an empty index), so the padding slots get a zero placeholder instead
        if corpus_precision == "ubinary":
            zero_embedding = np.zeros(corpus_index.d // 8, dtype=np.uint8)
        else:
            zero_embedding = np.zeros(corpus_index.d, dtype=np.float32)
        top_k_embeddings = np.array(
            [
                [corpus_index.reconstruct(idx.item()) if idx != -1 else zero_embedding for idx in query_indices]
                for query_indices in indices
            ]
        )
        # If the corpus precision is binary, we need to unpack the bits
        if corpus_precision == "ubinary":
            top_k_embeddings = np.unpackbits(top_k_embeddings, axis=-1).astype(int)
        else:
            top_k_embeddings = top_k_embeddings.astype(int)

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings)
        # Disqualify the padding slots so their placeholder scores cannot outrank real candidates
        rescored_scores[indices == -1] = -np.inf
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(query_embeddings))[:, None], rescored_indices]
        scores = rescored_scores[np.arange(len(query_embeddings))[:, None], rescored_indices]

    delta_t = time.time() - start_t

    # FAISS pads short result sets with index -1, which would otherwise resolve to the last corpus entry
    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
                if neighbor != -1
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def semantic_search_usearch(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray | None = None,
    corpus_index: usearch.index.Index | None = None,
    corpus_precision: Literal["float32", "int8", "ubinary", "binary"] = "float32",
    top_k: int = 10,
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
    rescore: bool = True,
    rescore_multiplier: int = 2,
    exact: bool = True,
    output_index: bool = False,
) -> tuple[list[list[dict[str, int | float]]], float, usearch.index.Index]:
    # [FUNCTION DOC]
    # Semantic search with usearch; same contract as the faiss variant.
    # Precisions: "float32" (cosine metric, f32), "int8" (inner product, i8), "binary" (Hamming, i8)
    # and "ubinary" (Hamming, b1 with dimension in bits). Validation errors as in the faiss version.
    # If no index is given, builds one and adds the corpus with keys 0..n-1.
    # Quantizes non-int queries to the corpus precision; enables rescoring (k times multiplier) only
    # when rescore is on and the index's scalar kind is not f32, otherwise warns. Warns if queries are
    # already quantized while rescore is on.
    # Searches with the exact flag, takes distances and keys, promoting 1-D results to 2-D. When
    # rescoring, fetches the stored vectors by key, unpacks bits for binary precisions, casts to int,
    # computes dot products with the float queries, keeps top_k by rescored score.
    # Returns (results, elapsed seconds) or with the index appended if output_index.
    # KNOWN ISSUE: without rescoring the reported "score" is a usearch distance (lower is better, e.g. cosine distance or Hamming), not a similarity, contrary to how the faiss inner-product path behaves.
    # KNOWN ISSUE: unlike the faiss variant, missing or padded results (fewer than k hits) are not filtered or handled during rescoring.
    """
    Performs semantic search using the usearch library.

    Rescoring will be performed if:
    1. `rescore` is True
    2. The query embeddings are not quantized
    3. The corpus is quantized, i.e. the corpus precision is not float32
    Only if these conditions are true, will we search for `top_k * rescore_multiplier` samples and then rescore to only
    keep `top_k`.

    Args:
        query_embeddings: Embeddings of the query sentences. Ideally not
            quantized to allow for rescoring.
        corpus_embeddings: Embeddings of the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both. The embeddings can be quantized to "int8", "ubinary",
            or "binary" for more efficient search.
        corpus_index: usearch index for the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both.
        corpus_precision: Precision of the corpus embeddings. The
            options are "float32", "int8", "ubinary", or "binary".
            Default is "float32".
        top_k: Number of top results to retrieve. Default is 10.
        ranges: Ranges for quantization of embeddings. This is only used
            for int8 quantization, where the ranges refer to the
            minimum and maximum values for each dimension. So, it's a 2D
            array with shape (2, embedding_dim). Default is None, which
            means that the ranges will be calculated from the
            calibration embeddings.
        calibration_embeddings: Embeddings used for calibration during
            quantization. This is only used for int8 quantization, where
            the calibration embeddings can be used to compute ranges,
            i.e. the minimum and maximum values for each dimension.
            Default is None, which means that the ranges will be
            calculated from the query embeddings. This is not
            recommended.
        rescore: Whether to perform rescoring. Note that rescoring still
            will only be used if the query embeddings are not quantized
            and the corpus is quantized, i.e. the corpus precision is
            not "float32". Default is True.
        rescore_multiplier: Oversampling factor for rescoring. The code
            will now search `top_k * rescore_multiplier` samples and
            then rescore to only keep `top_k`. Default is 2.
        exact: Whether to use exact search or approximate search.
            Default is True.
        output_index: Whether to output the usearch index used for the
            search. Default is False.

    Returns:
        A tuple containing a list of search results and the time taken
        for the search. If `output_index` is True, the tuple will also
        contain the usearch index used for the search.

    Raises:
        ValueError: If both `corpus_embeddings` and `corpus_index` are
            provided or if neither is provided.
        ValueError: If `corpus_precision` is not "float32", "int8",
            "ubinary", or "binary".

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    The time taken for the search is a float value.
    """
    from usearch.compiled import ScalarKind
    from usearch.index import Index

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")
    if corpus_precision not in ["float32", "int8", "ubinary", "binary"]:
        raise ValueError('corpus_precision must be "float32", "int8", "ubinary", or "binary" for usearch')

    # If corpus_index is not provided, create a new index
    if corpus_index is None:
        if corpus_precision == "float32":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="cos",
                dtype="f32",
            )
        elif corpus_precision == "int8":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="ip",
                dtype="i8",
            )
        elif corpus_precision == "binary":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="hamming",
                dtype="i8",
            )
        elif corpus_precision == "ubinary":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1] * 8,
                metric="hamming",
                dtype="b1",
            )
        corpus_index.add(np.arange(len(corpus_embeddings)), corpus_embeddings)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # rescore_multiplier
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if rescore:
            if corpus_index.dtype != ScalarKind.F32:
                rescore_embeddings = query_embeddings
                k *= rescore_multiplier
            else:
                logger.warning(
                    "Rescoring is enabled but the corpus is not quantized. Either pass `rescore=False` or "
                    'quantize the corpus embeddings with `quantize_embeddings(embeddings, precision="...") `'
                    'and pass `corpus_precision="..."` to `semantic_search_usearch`.'
                )

        query_embeddings = quantize_embeddings(
            query_embeddings,
            precision=corpus_precision,
            ranges=ranges,
            calibration_embeddings=calibration_embeddings,
        )
    elif rescore:
        logger.warning(
            "Rescoring is enabled but the query embeddings are quantized. Either pass `rescore=False` or don't quantize the query embeddings."
        )

    # Perform the search using the usearch index
    start_t = time.time()
    matches = corpus_index.search(query_embeddings, count=k, exact=exact)
    scores = matches.distances
    indices = matches.keys

    if scores.ndim < 2:
        scores = np.atleast_2d(scores)
    if indices.ndim < 2:
        indices = np.atleast_2d(indices)

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None:
        top_k_embeddings = np.array([corpus_index.get(query_indices) for query_indices in indices])
        # If the corpus precision is binary, we need to unpack the bits
        if corpus_precision in ("ubinary", "binary"):
            top_k_embeddings = np.unpackbits(top_k_embeddings.astype(np.uint8), axis=-1)
        top_k_embeddings = top_k_embeddings.astype(int)

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings)
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(query_embeddings))[:, None], rescored_indices]
        scores = rescored_scores[np.arange(len(query_embeddings))[:, None], rescored_indices]

    delta_t = time.time() - start_t

    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def quantize_embeddings(
    embeddings: Tensor | np.ndarray | list[Tensor] | list[np.ndarray],
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"],
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
) -> np.ndarray | list[np.ndarray]:
    # [FUNCTION DOC]
    # Quantizes embeddings to the requested precision.
    # Inputs: a tensor, numpy array, or list of tensors/arrays; precision ("float32", "int8", "uint8",
    # "binary", "ubinary"); optional ranges (2 x dim array of per-dimension min and max); optional
    # calibration embeddings used to compute ranges.
    # Tensors are moved to CPU numpy. An empty list returns an empty list. A list of 2-D matrices
    # (multi-vector, possibly ragged) is quantized matrix by matrix, recursively; for int8/uint8 with
    # no ranges or calibration, all matrices concatenated serve as calibration so buckets are shared.
    # Other lists are stacked into an array.
    # Raises a generic Exception if the input is already int8/uint8, and ValueError for unknown
    # precisions. A (0, dim) matrix yields an empty array of the right width and dtype (dim for int
    # precisions, ceil(dim/8) bytes for binary ones).
    # float32: cast. int8/uint8: ranges from argument, else min/max of calibration embeddings, else of
    # the embeddings themselves (warning if fewer than 100 rows); step is (max - min) / 255, with zero
    # steps replaced by 1; value is floor((x - min) / step) clipped to 0..255, returned as uint8, or
    # minus 128 as int8. ubinary: positive-sign bits packed along the last axis into uint8 (rows kept);
    # binary: the same packed bytes minus 128 as int8.
    """
    Quantizes embeddings to a lower precision. This can be used to reduce the memory footprint and increase the
    speed of similarity search. The supported precisions are "float32", "int8", "uint8", "binary", and "ubinary".

    Args:
        embeddings: Unquantized (e.g. float) embeddings with to quantize
            to a given precision
        precision: The precision to convert to. Options are "float32",
            "int8", "uint8", "binary", "ubinary".
        ranges (Optional[np.ndarray]): Ranges for quantization of
            embeddings. This is only used for int8 and uint8 quantization,
            where the ranges refer to the minimum and maximum values for each
            dimension. So, it's a 2D array with shape (2,
            embedding_dim). Default is None, which means that the ranges
            will be calculated from the calibration embeddings.
        calibration_embeddings (Optional[np.ndarray]): Embeddings used
            for calibration during quantization. This is only used for
            int8 and uint8 quantization, where the calibration embeddings
            can be used to compute ranges, i.e. the minimum and maximum
            values for each dimension. Default is None, which means that
            the ranges will be calculated from the embeddings. This is
            not recommended.

    Returns:
        Quantized embeddings with the specified precision. For a list of multi-vector matrices (variable-length
        ``(num_tokens, dim)`` arrays), returns a list of quantized matrices with shared per-dimension buckets.
    """
    if isinstance(embeddings, Tensor):
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, list):
        if not embeddings:
            # Nothing to quantize: preserve the (empty) list shape.
            return []
        if isinstance(embeddings[0], Tensor):
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        # Multi-vector input: a list of (num_tokens, dim) matrices (possibly ragged, one per text). Quantize
        # each matrix separately to preserve the variable-length structure, but share the per-dimension
        # buckets across all matrices (from `ranges` / `calibration_embeddings`, else all tokens together) so
        # values quantize consistently across matrices.
        if isinstance(embeddings[0], np.ndarray) and embeddings[0].ndim == 2:
            if precision.endswith("int8") and ranges is None and calibration_embeddings is None:
                calibration_embeddings = np.concatenate(embeddings, axis=0)
            return [
                quantize_embeddings(
                    matrix, precision=precision, ranges=ranges, calibration_embeddings=calibration_embeddings
                )
                for matrix in embeddings
            ]
        embeddings = np.array(embeddings)
    if embeddings.dtype in (np.uint8, np.int8):
        raise Exception("Embeddings to quantize must be float rather than int8 or uint8.")

    if embeddings.ndim == 2 and embeddings.shape[0] == 0:
        # A (0, dim) matrix (e.g. a fully-masked multi-vector document) has nothing to calibrate or
        # pack: return the correctly-shaped empty output for the precision.
        dim = embeddings.shape[1]
        if precision.endswith("int8"):
            return np.zeros((0, dim), dtype=np.int8 if precision == "int8" else np.uint8)
        if precision.endswith("binary"):
            return np.zeros((0, (dim + 7) // 8), dtype=np.int8 if precision == "binary" else np.uint8)

    if precision == "float32":
        return embeddings.astype(np.float32)

    if precision.endswith("int8"):
        # Either use the 1. provided ranges, 2. the calibration dataset or 3. the provided embeddings
        if ranges is None:
            if calibration_embeddings is not None:
                ranges = np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))
            else:
                if embeddings.shape[0] < 100:
                    logger.warning(
                        f"Computing {precision} quantization buckets based on {len(embeddings)} embedding{'s' if len(embeddings) != 1 else ''}."
                        f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
                        "or a `calibration_embeddings` that can be used to calculate the buckets."
                    )
                ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / 255
        steps = np.where(steps == 0, 1, steps)

        q_vals = np.clip(np.floor((embeddings - starts) / steps), 0, 255)
        if precision == "uint8":
            return q_vals.astype(np.uint8)
        elif precision == "int8":
            return (q_vals - 128).astype(np.int8)

    if precision == "binary":
        return (np.packbits(embeddings > 0, axis=-1).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)

    if precision == "ubinary":
        return np.packbits(embeddings > 0, axis=-1).reshape(embeddings.shape[0], -1)

    raise ValueError(f"Precision {precision!r} is not supported")
