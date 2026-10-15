# [MODULE DOC]
# Tests for the sparse encoder search engine adapters semantic_search_seismic and semantic_search_qdrant,
# using stub modules injected into the global module registry so the real Seismic and Qdrant libraries are not
# needed. Checks result ordering per query, empty results, and the query vectors sent to Qdrant.
from __future__ import annotations

import sys
import types

import pytest
import torch

from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant, semantic_search_seismic


class StubSeismicIndex:
    # [CLASS DOC]
    # HELPER: Stub Seismic index. Holds a fixed list of result rows (one list per query of tuples of query id,
    # score and document id, rows possibly out of order and empty for queries with no match).
    """Returns what Seismic returns: one row per query, in arbitrary order, empty rows for misses."""

    def __init__(self, rows):
        # [FUNCTION DOC]
        # HELPER: Stores the rows to return.
        self.rows = rows

    def batch_search(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Ignores all keyword arguments and returns the stored rows.
        return self.rows


@pytest.fixture
def stub_seismic_module(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Fixture that builds a fake "seismic" module (dataset class placeholder, the stub index class, and a
    # version string function returning "U30") and installs it in the module registry via monkeypatch.
    """Seismic is not a test dependency, so stand in for the import that the function performs."""
    module = types.ModuleType("seismic")
    module.SeismicDataset = object
    module.SeismicIndex = StubSeismicIndex
    module.get_seismic_string = lambda: "U30"
    monkeypatch.setitem(sys.modules, "seismic", module)


def test_seismic_query_without_matches(stub_seismic_module) -> None:
    # [FUNCTION DOC]
    # Feeds rows for queries 2, none and 0 (out of order, one empty) and asserts results are reordered per query
    # index with integer corpus ids and scores, and the unmatched query yields an empty list.
    """A query that matches no document comes back as an empty row that carries no query id."""
    # Seismic hands back the rows for queries 2, <none> and 0, in that order.
    rows = [
        [("2", 0.9, "2"), ("2", 0.6, "3")],
        [],
        [("0", 0.8, "0"), ("0", 0.7, "1")],
    ]
    queries = [[("fruit", 1.0)], [("nomatch", 1.0)], [("veg", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex(rows), top_k=2)

    assert results == [
        [{"corpus_id": 0, "score": 0.8}, {"corpus_id": 1, "score": 0.7}],
        [],
        [{"corpus_id": 2, "score": 0.9}, {"corpus_id": 3, "score": 0.6}],
    ]


def test_seismic_orders_results_by_query(stub_seismic_module) -> None:
    # [FUNCTION DOC]
    # Feeds two single-hit rows in reverse query order and asserts each lands on its own query.
    """Every row carries its query id, so out-of-order rows still land on the right query."""
    rows = [
        [("1", 0.5, "7")],
        [("0", 0.4, "9")],
    ]
    queries = [[("a", 1.0)], [("b", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex(rows), top_k=1)

    assert results == [[{"corpus_id": 9, "score": 0.4}], [{"corpus_id": 7, "score": 0.5}]]


def test_seismic_all_queries_without_matches(stub_seismic_module) -> None:
    # [FUNCTION DOC]
    # Feeds only empty rows and asserts every query gets an empty result list.
    """With no query matching anything there is no id to order by at all."""
    queries = [[("a", 1.0)], [("b", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex([[], []]), top_k=3)

    assert results == [[], []]


VOCAB_SIZE = 100


class StubSparseVector:
    # [CLASS DOC]
    # HELPER: Stub Qdrant sparse vector holding indices and values attributes.
    """Stands in for the Qdrant sparse vector, so a test can read back what was searched for."""

    def __init__(self, indices=None, values=None):
        # [FUNCTION DOC]
        # HELPER: Stores the given indices and values (both default None).
        self.indices = indices
        self.values = values


class StubHit:
    # [CLASS DOC]
    # HELPER: Stub Qdrant search hit with an id and a score.
    def __init__(self, corpus_id, score):
        # [FUNCTION DOC]
        # HELPER: Stores the corpus id as the id attribute and the score.
        self.id = corpus_id
        self.score = score


class StubResponse:
    # [CLASS DOC]
    # HELPER: Stub Qdrant query response holding a list of points.
    def __init__(self, points):
        # [FUNCTION DOC]
        # HELPER: Stores the points list.
        self.points = points


class StubQdrantClient:
    # [CLASS DOC]
    # HELPER: Stub Qdrant client recording every searched query vector as an (indices, values) pair in a list.
    """Records the query vector of every search, and answers each one with the same single hit."""

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Accepts and ignores any arguments; initialises an empty searched list.
        self.searched = []

    def query_points(self, query, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Appends the query's indices and values to the searched list and returns a response with one hit of
        # id 0 and score 1.0.
        self.searched.append((query.indices, query.values))
        return StubResponse([StubHit(0, 1.0)])


@pytest.fixture
def stub_qdrant_module(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Fixture that builds fake qdrant_client, qdrant_client.http and qdrant_client.http.models modules
    # (client class and sparse vector class from the stubs) and installs them in the module registry.
    """Qdrant is not a test dependency, so stand in for the imports that the function performs."""
    module = types.ModuleType("qdrant_client")
    module.QdrantClient = StubQdrantClient
    http = types.ModuleType("qdrant_client.http")
    models = types.ModuleType("qdrant_client.http.models")
    models.SparseVector = StubSparseVector
    http.models = models
    module.http = http
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", models)


def test_qdrant_single_query_embedding(stub_qdrant_module) -> None:
    # [FUNCTION DOC]
    # Searches with a 1-D sparse query of vocabulary size 100 and asserts one result list and that the searched
    # vector has token indices 3 and 8 with their values.
    """Encoding a single query gives a 1-dimensional tensor, whose size(0) is the vocabulary."""
    query = torch.sparse_coo_tensor(torch.tensor([[3, 8]]), torch.tensor([0.5, 0.25]), (VOCAB_SIZE,)).coalesce()
    client = StubQdrantClient()

    results, _ = semantic_search_qdrant(query, corpus_index=(client, "collection"), top_k=1)

    assert results == [[{"corpus_id": 0, "score": 1.0}]]
    assert client.searched == [([3, 8], [0.5, 0.25])]


def test_qdrant_batch_of_queries(stub_qdrant_module) -> None:
    # [FUNCTION DOC]
    # Searches with a 2-D sparse batch of two queries and asserts one search per query containing only that
    # query's tokens.
    """A batch is searched once per query, each with only the tokens of that query."""
    query = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [3, 8, 5]]), torch.tensor([0.5, 0.25, 0.75]), (2, VOCAB_SIZE)
    ).coalesce()
    client = StubQdrantClient()

    results, _ = semantic_search_qdrant(query, corpus_index=(client, "collection"), top_k=1)

    assert results == [[{"corpus_id": 0, "score": 1.0}], [{"corpus_id": 0, "score": 1.0}]]
    assert client.searched == [([3, 8], [0.5, 0.25]), ([5], [0.75])]


def test_qdrant_rejects_3d_query_embeddings(stub_qdrant_module) -> None:
    # [FUNCTION DOC]
    # Asserts a 3-D sparse tensor (stacked batches) raises a ValueError saying it must be 1D or 2D.
    """A 3D tensor, like two stacked batches, must fail loudly rather than search with row ids as token ids."""
    batch = torch.sparse_coo_tensor(torch.tensor([[0, 1], [3, 8]]), torch.tensor([0.5, 0.25]), (2, VOCAB_SIZE))
    query = torch.stack([batch, batch])

    with pytest.raises(ValueError, match="must be a 1D or 2D sparse tensor"):
        semantic_search_qdrant(query, corpus_index=(StubQdrantClient(), "collection"), top_k=1)


def test_qdrant_single_query_searches_same_vector_as_batch_of_one(stub_qdrant_module) -> None:
    # [FUNCTION DOC]
    # Asserts a 1-D query and the equivalent 2-D batch of one send identical vectors to the client.
    """The promoted 1-dimensional tensor must search for the same vector as an explicit batch of one."""
    indices = torch.tensor([3, 8])
    values = torch.tensor([0.5, 0.25])
    single = torch.sparse_coo_tensor(indices.unsqueeze(0), values, (VOCAB_SIZE,)).coalesce()
    batched = torch.sparse_coo_tensor(
        torch.stack([torch.zeros_like(indices), indices]), values, (1, VOCAB_SIZE)
    ).coalesce()

    single_client = StubQdrantClient()
    batched_client = StubQdrantClient()
    semantic_search_qdrant(single, corpus_index=(single_client, "collection"), top_k=1)
    semantic_search_qdrant(batched, corpus_index=(batched_client, "collection"), top_k=1)

    assert single_client.searched == batched_client.searched
