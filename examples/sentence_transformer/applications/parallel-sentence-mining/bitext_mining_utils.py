# [MODULE DOC]
# HELPER: small utility module (adapted from LASER) for margin-based bitext mining: margin scoring of candidates,
# FAISS k-nearest-neighbour search and opening optionally compressed text files.
"""
This file contains some utilities functions used to find parallel sentences
in two monolingual corpora.

Code in this file has been adapted from the LASER repository:
https://github.com/facebookresearch/LASER
"""

import gzip
import lzma
import time

import faiss
import numpy as np


########  Functions to find and score candidates
def score(x, y, fwd_mean, bwd_mean, margin):
    # [FUNCTION DOC]
    # HELPER: computes the margin function applied to the dot product of vectors x and y and the average of the forward and
    # backward neighbourhood means. Returns the resulting score.
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin):
    # [FUNCTION DOC]
    # HELPER: inputs are source matrix x, target matrix y, a candidate index matrix (one row of target indices per source),
    # per-source forward means, per-target backward means and a margin function. Returns a float matrix of the same shape where
    # each entry is the margin score of the source row against the given candidate. Pure loops, no side effects.
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores


def kNN(x, y, k, use_ann_search=False, ann_num_clusters=32768, ann_num_cluster_probe=3):
    # [FUNCTION DOC]
    # HELPER: finds the k most similar rows of y for every row of x by inner product. With approximate search, builds an IVF flat
    # index with a flat inner-product quantizer, using the smaller of ann_num_clusters and one cluster per 1,000 rows of y, sets
    # the probe count, trains it on y and adds y; otherwise uses an exact flat inner-product index. Prints which mode is used and
    # the elapsed time. Returns the similarity matrix and the index matrix.
    # KNOWN ISSUE: with approximate search and fewer than 1,000 target vectors the cluster count becomes 0 and FAISS fails.
    start_time = time.time()
    if use_ann_search:
        print("Perform approx. kNN search")
        n_cluster = min(ann_num_clusters, int(y.shape[0] / 1000))
        quantizer = faiss.IndexFlatIP(y.shape[1])
        index = faiss.IndexIVFFlat(quantizer, y.shape[1], n_cluster, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = ann_num_cluster_probe
        index.train(y)
        index.add(y)
        sim, ind = index.search(x, k)
    else:
        print("Perform exact search")
        idx = faiss.IndexFlatIP(y.shape[1])
        idx.add(y)
        sim, ind = idx.search(x, k)

    print(f"Done: {time.time() - start_time:.2f} sec")
    return sim, ind


def file_open(filepath):
    # [FUNCTION DOC]
    # HELPER: opens a file for UTF-8 text reading, transparently decompressing ".gz" files with gzip and names ending in "xz"
    # with lzma. Returns the open file object. Raises the usual I/O errors for missing files.
    # Function to allowing opening files based on file extension
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf8")
    elif filepath.endswith("xz"):
        return lzma.open(filepath, "rt", encoding="utf8")
    else:
        return open(filepath, encoding="utf8")
