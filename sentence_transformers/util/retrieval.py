# [MODULE DOC]
# Embedding-based retrieval utilities on torch tensors: paraphrase mining (top scoring distinct
# pairs within one set), chunked exact semantic search of queries against a corpus (top-k per
# query), and fast community detection (threshold-based clustering on cosine similarity).
# Default scoring is cosine similarity from the similarity module. Chunking bounds memory use.
from __future__ import annotations

import heapq
import logging
import queue
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from tqdm.autonotebook import tqdm

from .similarity import cos_sim
from .tensor import normalize_embeddings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer


def paraphrase_mining(
    model: SentenceTransformer,
    sentences: list[str],
    show_progress_bar: bool = False,
    batch_size: int = 32,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
    truncate_dim: int | None = None,
    prompt_name: str | None = None,
    prompt: str | None = None,
) -> list[list[float | int]]:
    # [FUNCTION DOC]
    # Encodes the sentences with the model (as a tensor, honoring batch size, progress bar,
    # truncate_dim, prompt_name and prompt) and delegates to paraphrase_mining_embeddings with the
    # chunk sizes, max_pairs, top_k and score function. Returns its list of [score, id1, id2].
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    Args:
        model (SentenceTransformer): SentenceTransformer model for embedding computation
        sentences (List[str]): A list of strings (texts or sentences)
        show_progress_bar (bool, optional): Plotting of a progress bar. Defaults to False.
        batch_size (int, optional): Number of texts that are encoded simultaneously by the model. Defaults to 32.
        query_chunk_size (int, optional): Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time). Defaults to 5000.
        corpus_chunk_size (int, optional): Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time). Defaults to 100000.
        max_pairs (int, optional): Maximal number of text pairs returned. Defaults to 500000.
        top_k (int, optional): For each sentence, we retrieve up to top_k other sentences. Defaults to 100.
        score_function (Callable[[Tensor, Tensor], Tensor], optional): Function for computing scores. By default, cosine similarity. Defaults to cos_sim.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. If None, uses the model's ones. Defaults to None.
        prompt_name (Optional[str], optional): The name of a predefined prompt to use when encoding the sentence.
            It must match a key in the model `prompts` dictionary, which can be set during model initialization
            or loaded from the model configuration.

            Ignored if `prompt` is provided. Defaults to None.

        prompt (Optional[str], optional): A raw prompt string to prepend directly to the input sentence during encoding.

            For instance, `prompt="query: "` transforms the sentence "What is the capital of France?" into:
            "query: What is the capital of France?". Use this to override the prompt logic entirely and supply your own prefix.
            This takes precedence over `prompt_name`. Defaults to None.

    Returns:
        List[List[Union[float, int]]]: Returns a list of triplets with the format [score, id1, id2]
    """

    # Compute embedding for the sentences
    embeddings = model.encode(
        sentences,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
        convert_to_tensor=True,
        truncate_dim=truncate_dim,
        prompt_name=prompt_name,
        prompt=prompt,
    )

    return paraphrase_mining_embeddings(
        embeddings,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
        max_pairs=max_pairs,
        top_k=top_k,
        score_function=score_function,
    )


def paraphrase_mining_embeddings(
    embeddings: Tensor,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> list[list[float | int]]:
    # [FUNCTION DOC]
    # Finds the highest-scoring pairs of distinct items in an embedding matrix.
    # Increments top_k by one (each item matches itself). Iterates over corpus chunks and query chunks,
    # scores each block with score_function, takes the top min(top_k, block width) per query row, and
    # for each hit with i != j and score above the running minimum pushes (score, i, j) onto a min
    # priority queue. Once max_pairs pushes have happened, each further push also pops the lowest
    # entry and raises the minimum threshold to its score.
    # Afterwards drains the queue, normalizes each pair to (smaller id, larger id), drops duplicates,
    # and returns a list of [score, id1, id2] sorted by descending score.
    # KNOWN ISSUE: the initial minimum score is -1, so with non-cosine score functions, pairs scoring -1 or below are never returned.
    # KNOWN ISSUE: both (i, j) and (j, i) occupy queue slots before de-duplication, so fewer than max_pairs unique pairs may be returned even when more exist.
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    Args:
        embeddings (Tensor): A tensor with the embeddings
        query_chunk_size (int): Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
        corpus_chunk_size (int): Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
        max_pairs (int): Maximal number of text pairs returned.
        top_k (int): For each sentence, we retrieve up to top_k other sentences
        score_function (Callable[[Tensor, Tensor], Tensor]): Function for computing scores. By default, cosine similarity.

    Returns:
        List[List[Union[float, int]]]: Returns a list of triplets with the format [score, id1, id2]
    """

    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = score_function(
                embeddings[query_start_idx : query_start_idx + query_chunk_size],
                embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
            )

            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores, min(top_k, len(scores[0])), dim=1, largest=True, sorted=False
            )
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, sorted_i, sorted_j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def information_retrieval(*args, **kwargs) -> list[list[dict[str, int | float]]]:
    # [FUNCTION DOC]
    # HELPER: Deprecated alias that forwards all arguments to semantic_search and returns its result.
    """This function is deprecated. Use semantic_search instead"""
    return semantic_search(*args, **kwargs)


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> list[list[dict[str, int | float]]]:
    # [FUNCTION DOC]
    # Exact top-k semantic search.
    # Inputs: query and corpus embeddings (tensor, numpy array or list of tensors, converted to
    # tensors; a 1-D query becomes a single row; dense or sparse COO), chunk sizes, top_k, score
    # function. Queries are moved to the corpus device.
    # For each query chunk and corpus chunk (sparse tensors sliced via index selection), computes the
    # score matrix, takes the per-row top min(top_k, chunk width), and merges hits into a size-top_k
    # min-heap per query keyed by score (pushing until full, then push-pop).
    # Returns, per query, a list of dicts with "corpus_id" (global corpus index) and "score", sorted by
    # decreasing score.
    """
    This function performs by default a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    Args:
        query_embeddings (:class:`~torch.Tensor`): A 2 dimensional tensor with the query embeddings. Can be a sparse COO tensor.
        corpus_embeddings (:class:`~torch.Tensor`): A 2 dimensional tensor with the corpus embeddings. Can be a sparse COO tensor.
        query_chunk_size (int, optional): Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory. Defaults to 100.
        corpus_chunk_size (int, optional): Scans the corpus 500k entries at a time. Increasing that value increases the speed, but requires more memory. Defaults to 500000.
        top_k (int, optional): Retrieve top k matching entries. Defaults to 10.
        score_function (Callable[[:class:`~torch.Tensor`, :class:`~torch.Tensor`], :class:`~torch.Tensor`], optional): Function for computing scores. By default, cosine similarity.

    Returns:
        List[List[Dict[str, Union[int, float]]]]: A list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        query_end_idx = min(query_start_idx + query_chunk_size, len(query_embeddings))
        if query_embeddings.is_sparse:
            indices = torch.arange(query_start_idx, query_end_idx, device=query_embeddings.device)
            query_chunk = query_embeddings.index_select(0, indices)
        else:
            query_chunk = query_embeddings[query_start_idx:query_end_idx]

        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus_embeddings))
            if corpus_embeddings.is_sparse:
                indices = torch.arange(corpus_start_idx, corpus_end_idx, device=corpus_embeddings.device)
                corpus_chunk = corpus_embeddings.index_select(0, indices)
            else:
                corpus_chunk = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarities
            cos_scores = score_function(query_chunk, corpus_chunk)

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heap tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {"corpus_id": corpus_id, "score": score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x["score"], reverse=True)

    return queries_result_list


def community_detection(
    embeddings: torch.Tensor | np.ndarray,
    threshold: float = 0.75,
    min_community_size: int = 10,
    batch_size: int = 1024,
    show_progress_bar: bool = False,
) -> list[list[int]]:
    # [FUNCTION DOC]
    # Fast community detection over embeddings.
    # Inputs: embeddings (tensor or array), similarity threshold, minimum community size, batch size,
    # progress bar flag. Normalizes embeddings to unit length and processes rows in batches, computing
    # cosine similarity of each batch against all embeddings.
    # On CUDA/NPU: keeps rows whose count of scores at or above threshold reaches min size, and for
    # each such row takes the top "count" indices (sorted by score, so the row itself is first).
    # Otherwise: for rows whose min_community_size-th best score reaches threshold, takes the top
    # sort_max_size (initially max(2 * min size, 50), capped at corpus size, doubled while the last
    # value still meets the threshold; this growth persists across rows) and keeps indices at or
    # above threshold.
    # Candidate communities are sorted largest first; then greedily each keeps only indices not yet
    # claimed and is accepted if still at least min size, marking its members as claimed.
    # Returns the accepted communities as lists of ints, largest first, first element being the
    # central point. min_community_size is capped at the number of embeddings.
    """
    Function for Fast Community Detection.

    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): The input embeddings.
        threshold (float): The threshold for determining if two embeddings are close. Defaults to 0.75.
        min_community_size (int): The minimum size of a community to be considered. Defaults to 10.
        batch_size (int): The batch size for computing cosine similarity scores. Defaults to 1024.
        show_progress_bar (bool): Whether to show a progress bar during computation. Defaults to False.

    Returns:
        List[List[int]]: A list of communities, where each community is represented as a list of indices.
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)
    embeddings = normalize_embeddings(embeddings)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in tqdm(
        range(0, len(embeddings), batch_size), desc="Finding clusters", disable=not show_progress_bar
    ):
        # Compute cosine similarity scores
        cos_scores = embeddings[start_idx : start_idx + batch_size] @ embeddings.T

        # Use a torch-heavy approach if the embeddings are on CUDA, otherwise a loop-heavy one
        if embeddings.device.type in ["cuda", "npu"]:
            # Threshold the cos scores and determine how many close embeddings exist per embedding
            threshold_mask = cos_scores >= threshold
            row_wise_count = threshold_mask.sum(1)

            # Only consider embeddings with enough close other embeddings
            large_enough_mask = row_wise_count >= min_community_size
            if not large_enough_mask.any():
                continue

            row_wise_count = row_wise_count[large_enough_mask]
            cos_scores = cos_scores[large_enough_mask]

            # The max is the largest potential community, so we use that in topk
            k = row_wise_count.max()
            _, top_k_indices = cos_scores.topk(k=k, largest=True)

            # Move the batch indices to CPU once, then slice per community
            top_k_indices = top_k_indices.cpu()
            for count, indices in zip(row_wise_count.tolist(), top_k_indices):
                extracted_communities.append(indices[:count].numpy().astype(np.uint32))
        else:
            # Minimum size for a community
            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

            # Filter for rows >= min_threshold
            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    # Only check top k most similar entries
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    # Check if we need to increase sort_max_size
                    while top_val_large[-1] >= threshold and sort_max_size < len(embeddings):
                        sort_max_size = min(2 * sort_max_size, len(embeddings))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    extracted_communities.append(
                        top_idx_large[top_val_large >= threshold].cpu().numpy().astype(np.uint32)
                    )

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities using a dense boolean mask of already claimed indices
    unique_communities = []
    used = np.zeros(len(embeddings), dtype=bool)

    for community in extracted_communities:
        # community keeps its topk order, so the central point stays first
        non_overlapped_community = community[~used[community]]
        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            used[non_overlapped_community] = True

    unique_communities = [
        community.tolist() for community in sorted(unique_communities, key=lambda x: len(x), reverse=True)
    ]

    return unique_communities
