# [MODULE DOC]
# HELPER: small utility module implementing LexRank sentence centrality (adapted from the lexrank package): builds a Markov
# transition matrix from a sentence similarity matrix, splits it into connected components and computes each component's
# stationary distribution with the power method. Uses a module logger for convergence warnings.
"""
LexRank implementation
Source: https://github.com/crabcamp/lexrank/tree/dev
"""

import logging

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax

logger = logging.getLogger(__name__)


def degree_centrality_scores(similarity_matrix, threshold=None, increase_power=True):
    # [FUNCTION DOC]
    # Computes LexRank centrality scores. Inputs: a square similarity matrix, an optional threshold (None for continuous LexRank,
    # otherwise a float in [0, 1) that binarizes the matrix) and increase_power (whether the power method squares the transition
    # matrix each iteration). Raises ValueError for an invalid threshold. Builds the continuous or discrete Markov matrix and
    # returns the unnormalized stationary distribution as a numpy array (one score per sentence).
    if not (threshold is None or isinstance(threshold, float) and 0 <= threshold < 1):
        raise ValueError(
            "'threshold' should be a floating-point number from the interval [0, 1) or None",
        )

    if threshold is None:
        markov_matrix = create_markov_matrix(similarity_matrix)

    else:
        markov_matrix = create_markov_matrix_discrete(
            similarity_matrix,
            threshold,
        )

    scores = stationary_distribution(
        markov_matrix,
        increase_power=increase_power,
        normalized=False,
    )

    return scores


def _power_method(transition_matrix, increase_power=True, max_iter=10000):
    # [FUNCTION DOC]
    # HELPER: iterates the power method for the stationary vector of a transition matrix, starting from a vector of ones. A
    # single-element matrix returns the ones vector immediately. Each iteration multiplies the transposed transition matrix by
    # the current vector and stops when successive vectors are all close; with increase_power the matrix is squared after every
    # iteration for faster convergence. After max_iter iterations (default 10,000) without convergence it logs a warning and
    # returns the last vector.
    eigenvector = np.ones(len(transition_matrix))

    if len(eigenvector) == 1:
        return eigenvector

    transition = transition_matrix.transpose()

    for _ in range(max_iter):
        eigenvector_next = np.dot(transition, eigenvector)

        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next

        eigenvector = eigenvector_next

        if increase_power:
            transition = np.dot(transition, transition)

    logger.warning("Maximum number of iterations for power method exceeded without convergence!")
    return eigenvector_next


def connected_nodes(matrix):
    # [FUNCTION DOC]
    # HELPER: finds connected components of the graph given by a matrix using scipy and returns a list of index arrays, one per
    # component, in label order.
    _, labels = connected_components(matrix)

    groups = []

    for tag in np.unique(labels):
        group = np.where(labels == tag)[0]
        groups.append(group)

    return groups


def create_markov_matrix(weights_matrix):
    # [FUNCTION DOC]
    # HELPER: turns a square weights matrix into a row-stochastic matrix. Raises ValueError if not square. If any weight is zero
    # or negative, returns a row-wise softmax of the weights; otherwise divides each row by its sum.
    # KNOWN ISSUE: a matrix with zeros (such as the binarized matrix from create_markov_matrix_discrete) is passed through softmax, so zero weights become non-zero transition probabilities and the discrete variant is not a true thresholded graph.
    n_1, n_2 = weights_matrix.shape
    if n_1 != n_2:
        raise ValueError("'weights_matrix' should be square")

    row_sum = weights_matrix.sum(axis=1, keepdims=True)

    # normalize probability distribution differently if we have negative transition values
    if np.min(weights_matrix) <= 0:
        return softmax(weights_matrix, axis=1)

    return weights_matrix / row_sum


def create_markov_matrix_discrete(weights_matrix, threshold):
    # [FUNCTION DOC]
    # HELPER: binarizes the weights matrix (1 where the weight is at least the threshold, 0 elsewhere) and converts it with
    # create_markov_matrix. Returns the resulting matrix.
    discrete_weights_matrix = np.zeros(weights_matrix.shape)
    ixs = np.where(weights_matrix >= threshold)
    discrete_weights_matrix[ixs] = 1

    return create_markov_matrix(discrete_weights_matrix)


def stationary_distribution(transition_matrix, increase_power=True, normalized=True):
    # [FUNCTION DOC]
    # HELPER: computes a stationary distribution of a square transition matrix (ValueError if not square) by running the power
    # method separately on the sub-matrix of every connected component and writing each result into the matching positions.
    # If normalized, divides the whole vector by the number of nodes. Returns a numpy vector.
    n_1, n_2 = transition_matrix.shape
    if n_1 != n_2:
        raise ValueError("'transition_matrix' should be square")

    distribution = np.zeros(n_1)

    grouped_indices = connected_nodes(transition_matrix)

    for group in grouped_indices:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        distribution[group] = eigenvector

    if normalized:
        distribution /= n_1

    return distribution
