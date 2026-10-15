# [MODULE DOC]
# Tests token pooling modules for MultiVectorEncoder, which reduce the number of per-token document vectors. HierarchicalTokenPooling clusters
# non-protected tokens (agglomerative clustering via scipy on cosine distances) into roughly tokens divided by pool_factor clusters and replaces
# each by its mean, keeping the first num_protected_tokens verbatim; LambdaTokenPooling applies a user function per sample. Both act only for
# tasks listed in "tasks" (documents by default), accept lists of 2-D tensors or padded 3-D batches (with mask or zero-row detection, left or right
# padding), and work as pipeline modules. Covers shapes, math, dtypes, pickling, encode integration, gradient flow, validation and edge cases.
from __future__ import annotations

import pickle
from itertools import combinations

import pytest
import torch
from torch import Tensor

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import (
    HierarchicalTokenPooling,
    LambdaTokenPooling,
)
from sentence_transformers.util import batch_to_device


def _normed(shape: tuple[int, int]) -> Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns a random tensor of the given shape with each row L2-normalized.
    return torch.nn.functional.normalize(torch.randn(*shape), p=2, dim=-1)


@pytest.fixture(autouse=True)
def _seed() -> None:
    # [FUNCTION DOC]
    # HELPER: Autouse fixture seeding torch's RNG with 0 before every test.
    torch.manual_seed(0)


class TestPoolShapeInvariants:
    # [CLASS DOC]
    # Tests input/output container and shape invariants of pool().
    def test_list_input_returns_list(self) -> None:
        # [FUNCTION DOC]
        # Asserts a list of 12- and 20-token inputs returns a list with 6 and 10 tokens.
        docs = [_normed((12, 8)), _normed((20, 8))]
        pooling = HierarchicalTokenPooling(pool_factor=2)
        out = pooling.pool(docs)
        assert isinstance(out, list)
        assert [e.shape for e in out] == [(6, 8), (10, 8)]

    def test_3d_input_returns_3d(self) -> None:
        # [FUNCTION DOC]
        # Asserts a right-padded 3-D batch with mask returns a 3-D tensor padded to the longest pooled length (2, 10, 8).
        docs = [_normed((12, 8)), _normed((20, 8))]
        padded = torch.nn.utils.rnn.pad_sequence(docs, batch_first=True)
        attn = padded.abs().sum(-1) > 0
        pooling = HierarchicalTokenPooling(pool_factor=2)
        out = pooling.pool(padded, attention_mask=attn)
        assert isinstance(out, Tensor)
        # 20 // 2 = 10 tokens is the max. Batch is (2, 10, 8).
        assert out.shape == (2, 10, 8)

    def test_3d_input_left_padding_round_trip(self) -> None:
        # [FUNCTION DOC]
        # Left-pads two docs by hand and pools without a mask (zero-row detection, left side); asserts a tensor whose shorter row is zero on the left and
        # whose longer row has real values at the start.
        docs = [_normed((6, 4)), _normed((10, 4))]
        # Left-pad by hand.
        max_len = max(len(d) for d in docs)
        padded = torch.stack([torch.cat([torch.zeros(max_len - len(d), 4), d], dim=0) for d in docs])
        pooling = HierarchicalTokenPooling(pool_factor=2)
        # No mask: rely on zero-detection with padding_side="left".
        out = pooling.pool(padded, padding_side="left")
        assert isinstance(out, Tensor)
        # Padding preserved on the left after the round-trip.
        assert torch.all(out[0, : out.shape[1] - 3] == 0)
        assert (out[1, :5] != 0).any(dim=-1).any()

    def test_empty_list_input(self) -> None:
        # [FUNCTION DOC]
        # Asserts pooling an empty list returns an empty list.
        pooling = HierarchicalTokenPooling(pool_factor=2)
        assert pooling.pool([]) == []

    def test_task_outside_tasks_returns_input_unchanged(self) -> None:
        # [FUNCTION DOC]
        # Asserts pool with task "query" (not in the default tasks) returns the very same input object, while "document" returns a pooled list.
        # encode() delegates its per-call gating to this: with the default tasks (documents only),
        # pool(task="query") must be an identity.
        docs = [_normed((12, 8))]
        pooling = HierarchicalTokenPooling(pool_factor=2)
        assert pooling.pool(docs, task="query") is docs
        assert isinstance(pooling.pool(docs, task="document"), list)

    def test_bad_input_type_raises(self) -> None:
        # [FUNCTION DOC]
        # Asserts a bare 2-D tensor raises ValueError requiring 3-D tensor input.
        pooling = HierarchicalTokenPooling(pool_factor=2)
        with pytest.raises(ValueError, match="Tensor input must be 3D"):
            pooling.pool(_normed((5, 8)))  # 2D bare tensor rejected

    def test_3d_unbind_preserves_middle_zero_row(self) -> None:
        # [FUNCTION DOC]
        # With identity lambda pooling and right padding without a mask, asserts a real all-zero middle row is kept and only the trailing zero row is
        # dropped (output 1x3x2).
        # A real token with an all-zero embedding must not be clipped off the end of its sequence.
        # (Regression guard for the vectorized fallback path that uses row-is-real detection.)
        emb = torch.stack(
            [
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 0.0]),  # zero-valued but "real"
                torch.tensor([0.5, 0.5]),
                torch.tensor([0.0, 0.0]),  # padding
            ]
        )
        padded = emb.unsqueeze(0)  # (1, 4, 2)
        # Round-trip through LambdaTokenPooling with an identity func, no attention_mask.
        out = LambdaTokenPooling(pool_fn=lambda x: x).pool(padded, padding_side="right")
        # Expected: real rows [0..2] preserved (including middle zero). Trailing pad row dropped.
        assert out.shape == (1, 3, 2)


class TestHierarchicalTokenPooling:
    # [CLASS DOC]
    # Tests HierarchicalTokenPooling's clustering behaviour.
    def test_class_pool_matches_module_helper_directly(self) -> None:
        # [FUNCTION DOC]
        # Asserts pool() equals the internal single-sequence helper for the same input and settings.
        # The pool() class method and the internal _hierarchical_pool_one produce identical output
        # when given the same input (this pins the wiring between the public API and the math).
        from sentence_transformers.multi_vector_encoder.modules.token_pooling import _hierarchical_pool_one

        emb = _normed((15, 8))
        direct = _hierarchical_pool_one(emb, pool_factor=2, num_protected_tokens=1)
        via_pool = HierarchicalTokenPooling(pool_factor=2, num_protected_tokens=1).pool([emb])[0]
        assert torch.allclose(direct, via_pool)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_pooling_supports_low_precision_dtypes(self, dtype: torch.dtype) -> None:
        # [FUNCTION DOC]
        # Parametrized over float32, float16 and bfloat16. Asserts the output keeps the input dtype and matches the float32 result in shape and within 1e-2.
        # The ColPali/ColQwen2 family runs bf16 by default, and numpy has no bfloat16: the scipy
        # distance step must run in fp32 while the pooled output keeps the input dtype.
        emb = _normed((15, 8)).to(dtype)
        out = HierarchicalTokenPooling(pool_factor=2, num_protected_tokens=1).pool([emb])[0]
        assert out.dtype == dtype
        # Same values in fp32 must yield the same clustering: only the mean accumulation dtype differs.
        reference = HierarchicalTokenPooling(pool_factor=2, num_protected_tokens=1).pool([emb.float()])[0]
        assert out.shape == reference.shape
        assert torch.allclose(out.float(), reference, atol=1e-2)

    def test_no_op_when_pool_factor_1(self) -> None:
        # [FUNCTION DOC]
        # Asserts forward with pool_factor 1 returns the same features dict object (no-op fast path).
        docs = [_normed((10, 4))]
        pooling = HierarchicalTokenPooling(pool_factor=1)
        # Pipeline forward path is a fast no-op.
        features = {
            "token_embeddings": docs[0].unsqueeze(0),
            "attention_mask": torch.ones(1, 10, dtype=torch.bool),
        }
        out = pooling.forward(features, task="document")
        assert out is features  # returned unchanged (fast path)

    def test_num_protected_tokens_untouched(self) -> None:
        # [FUNCTION DOC]
        # Asserts the first two output rows equal the two protected input rows.
        emb = _normed((12, 8))
        out = HierarchicalTokenPooling(pool_factor=2, num_protected_tokens=2).pool([emb])[0]
        # First 2 rows are the protected tokens verbatim.
        assert torch.allclose(out[:2], emb[:2])

    def test_pooled_rows_are_cluster_means(self) -> None:
        # [FUNCTION DOC]
        # With 7 rows, one protected and pool factor 3, brute-forces subsets to assert every pooled row equals the mean of a disjoint subset of
        # non-protected input rows and that the subsets cover all of them.
        # Every non-protected row of the output must equal the mean of some non-empty subset of the
        # non-protected input rows, and the subsets must partition the non-protected input rows.
        # Verifies the scatter_add + dense-remap math.
        torch.manual_seed(1)
        emb = _normed((7, 4))  # 6 non-protected rows -> brute-force over 2^6 subsets is trivial.
        num_protected_tokens = 1
        pool_factor = 3
        pooling = HierarchicalTokenPooling(pool_factor=pool_factor, num_protected_tokens=num_protected_tokens)
        out = pooling.pool([emb])[0]
        pooled_rows = out[num_protected_tokens:]
        source_rows = emb[num_protected_tokens:]
        n = source_rows.size(0)
        used = [False] * n
        for pooled_row in pooled_rows:
            match: tuple[int, ...] | None = None
            for size in range(1, n + 1):
                for combo in combinations([i for i in range(n) if not used[i]], size):
                    if torch.allclose(source_rows[list(combo)].mean(dim=0), pooled_row, atol=1e-5):
                        match = combo
                        break
                if match is not None:
                    break
            assert match is not None, "pooled row is not the mean of any unused subset of input rows"
            for i in match:
                used[i] = True
        assert all(used), "some input rows never contributed to a pooled row"


class TestLambdaTokenPooling:
    # [CLASS DOC]
    # Tests LambdaTokenPooling.
    def test_applies_pool_fn_per_sample(self) -> None:
        # [FUNCTION DOC]
        # Asserts a pairwise-averaging function is applied per sample (10 and 6 tokens become 5 and 3).
        def halve(emb: Tensor) -> Tensor:
            # [FUNCTION DOC]
            # HELPER: Drops a trailing odd row and averages consecutive pairs of rows.
            n = emb.size(0)
            return emb[: n - n % 2].view(-1, 2, emb.size(-1)).mean(dim=1)

        docs = [_normed((10, 4)), _normed((6, 4))]
        out = LambdaTokenPooling(pool_fn=halve).pool(docs)
        assert [e.shape for e in out] == [(5, 4), (3, 4)]

    def test_save_raises_not_implemented(self, tmp_path) -> None:
        # [FUNCTION DOC]
        # Asserts saving a lambda pooling raises NotImplementedError saying it cannot be saved.
        pooling = LambdaTokenPooling(pool_fn=lambda x: x)
        with pytest.raises(NotImplementedError, match="cannot be saved"):
            pooling.save(str(tmp_path))


def _keep_first_two(emb: Tensor) -> Tensor:
    # [FUNCTION DOC]
    # HELPER: Module-level picklable pool function returning the first two rows.
    """Module-level (pickle-friendly) pool function used by the pickle smoke test."""
    return emb[:2]


class TestPicklability:
    # [CLASS DOC]
    # Tests pickling of pooling modules (needed for multi-process encoding).
    def test_hierarchical_pooling_round_trips_through_pickle(self) -> None:
        # [FUNCTION DOC]
        # Asserts a pickled and restored HierarchicalTokenPooling keeps its settings and produces the same output.
        # Ensures `pool={...}` (multi-process encode) works with a saveable pooling.
        pooling = HierarchicalTokenPooling(pool_factor=3, num_protected_tokens=1)
        restored = pickle.loads(pickle.dumps(pooling))
        assert isinstance(restored, HierarchicalTokenPooling)
        assert restored.pool_factor == 3 and restored.num_protected_tokens == 1
        emb = _normed((12, 8))
        assert torch.allclose(pooling.pool([emb])[0], restored.pool([emb])[0])

    def test_lambda_pooling_with_top_level_func_pickles(self) -> None:
        # [FUNCTION DOC]
        # Asserts LambdaTokenPooling with a module-level function pickles and restores with identical output.
        # LambdaTokenPooling survives pickle as long as `pool_fn` is picklable itself.
        pooling = LambdaTokenPooling(pool_fn=_keep_first_two)
        restored = pickle.loads(pickle.dumps(pooling))
        assert isinstance(restored, LambdaTokenPooling)
        emb = _normed((10, 4))
        assert torch.equal(pooling.pool([emb])[0], restored.pool([emb])[0])

    def test_lambda_pooling_with_lambda_does_not_pickle(self) -> None:
        # [FUNCTION DOC]
        # Asserts pickling LambdaTokenPooling with an anonymous lambda raises PicklingError or AttributeError.
        # Anonymous lambdas / nested functions won't pickle. Confirms the docstring caveat.
        pooling = LambdaTokenPooling(pool_fn=lambda x: x)
        with pytest.raises((pickle.PicklingError, AttributeError)):
            pickle.dumps(pooling)


class TestPipelineModuleForward:
    # [CLASS DOC]
    # Tests the pipeline forward method's task gating.
    def test_forward_pools_documents(self) -> None:
        # [FUNCTION DOC]
        # Asserts forward with task document halves the token dimension of a 2x12x8 batch.
        pooling = HierarchicalTokenPooling(pool_factor=2)
        features = {
            "token_embeddings": _normed((2, 12, 8)),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling.forward(features, task="document")
        assert out["token_embeddings"].shape[1] == 6

    def test_forward_skips_queries(self) -> None:
        # [FUNCTION DOC]
        # Asserts forward with task query leaves the token embeddings as the same tensor object.
        pooling = HierarchicalTokenPooling(pool_factor=2)
        embeddings = _normed((2, 12, 8))
        features = {
            "token_embeddings": embeddings,
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling.forward(features, task="query")
        assert out["token_embeddings"] is embeddings  # unchanged reference

    def test_forward_pools_queries_when_opted_in(self) -> None:
        # [FUNCTION DOC]
        # Asserts listing query in tasks makes forward pool queries too.
        # CRISP-style strategies compress queries too: listing "query" in tasks lifts the gate.
        pooling = HierarchicalTokenPooling(pool_factor=2, tasks=["query", "document"])
        features = {
            "token_embeddings": _normed((2, 12, 8)),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling.forward(features, task="query")
        assert out["token_embeddings"].shape[1] == 6


class TestEncodeWithPooling:
    # [CLASS DOC]
    # Tests encode-time pooling via the token_pooling argument on a tiny MultiVectorEncoder (downloads the model in each test).
    def test_encode_document_with_pooling_kwarg(self) -> None:
        # [FUNCTION DOC]
        # Asserts hierarchical pooling yields fewer document tokens than no pooling.
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        text = "a fairly long document with plenty of distinct tokens for clustering here"
        without = model.encode_document([text])[0]
        with_pooling = model.encode_document(
            [text],
            token_pooling=HierarchicalTokenPooling(pool_factor=2),
        )[0]
        assert with_pooling.shape[0] < without.shape[0]

    def test_encode_document_with_lambda_pooling(self) -> None:
        # [FUNCTION DOC]
        # Asserts a lambda keeping the first two rows yields exactly two document tokens.
        def keep_first_two(emb: Tensor) -> Tensor:
            # [FUNCTION DOC]
            # HELPER: Returns the first two rows.
            return emb[:2]

        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        text = "some longer text with multiple tokens"
        out = model.encode_document(
            [text],
            token_pooling=LambdaTokenPooling(pool_fn=keep_first_two),
        )[0]
        assert out.shape[0] == 2

    def test_pooling_skips_queries(self) -> None:
        # [FUNCTION DOC]
        # Asserts query encoding with default-task pooling is unchanged in shape and values.
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        text = "a fairly long query text with many tokens"
        without = model.encode_query([text])[0]
        with_pooling = model.encode_query(
            [text],
            token_pooling=HierarchicalTokenPooling(pool_factor=2),
        )[0]
        # Queries pass through unchanged: shape AND values must match (guards against silent mutation).
        assert with_pooling.shape == without.shape
        assert torch.equal(with_pooling, without)

    def test_pooling_pools_queries_when_opted_in(self) -> None:
        # [FUNCTION DOC]
        # Asserts query encoding with pooling opted in for queries yields fewer tokens.
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        text = "a fairly long query text with many tokens"
        without = model.encode_query([text])[0]
        pooled = model.encode_query(
            [text],
            token_pooling=HierarchicalTokenPooling(pool_factor=2, tasks=["query", "document"]),
        )[0]
        assert pooled.shape[0] < without.shape[0]

    def test_encode_document_with_pooling_shrinks_token_counts(self) -> None:
        # [FUNCTION DOC]
        # Asserts every document in a batch of two gets strictly fewer tokens with pooling.
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        texts = [
            "a fairly long document with plenty of distinct tokens for clustering here",
            "a much shorter text",
        ]
        without = model.encode_document(texts)
        with_pooling = model.encode_document(
            texts,
            token_pooling=HierarchicalTokenPooling(pool_factor=2),
        )
        # Pooled output has strictly fewer tokens per doc than the un-pooled version.
        assert all(pooled.shape[0] < plain.shape[0] for pooled, plain in zip(with_pooling, without))


class TestTrainingGradientFlow:
    # [CLASS DOC]
    # Tests that pooling inside the model pipeline is differentiable.
    def test_forward_backward_with_pooling_in_pipeline(self) -> None:
        # [FUNCTION DOC]
        # Appends hierarchical pooling to a tiny model in train mode, runs a forward on a preprocessed document, asserts the token count shrank, then
        # backpropagates the sum of the non-protected pooled rows and asserts at least one transformer parameter got a non-zero gradient. Network.
        # A checkpoint with baked-in pooling must remain finetunable: with grad enabled, the
        # clustering used to raise "Can't call numpy() on Tensor that requires grad" in forward.
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        model.append(HierarchicalTokenPooling(pool_factor=2))
        model.train()
        features = model.preprocess(
            ["a fairly long document with plenty of distinct tokens to cluster together here"],
            task="document",
        )
        features = batch_to_device(features, model.device)
        num_tokens = features["input_ids"].shape[1]
        out = model(features, task="document")
        assert out["token_embeddings"].shape[1] < num_tokens, "pooling did not trigger"
        # Row 0 is the protected passthrough and carries gradient on its own, so backward through
        # the pooled rows only. This fails too if the cluster means are detached alongside the
        # clustering.
        out["token_embeddings"][:, 1:].sum().backward()
        grads = [p.grad for p in model[0].parameters() if p.grad is not None]
        assert grads, "no transformer parameter received a gradient"
        assert any(g.abs().sum() > 0 for g in grads), "no gradient reached the transformer through the cluster means"


class TestConstructorValidation:
    # [CLASS DOC]
    # Tests HierarchicalTokenPooling constructor validation and config persistence.
    @pytest.mark.parametrize("bad", [0, -1])
    def test_pool_factor_must_be_positive(self, bad: int) -> None:
        # [FUNCTION DOC]
        # Parametrized over 0 and -1: asserts pool_factor below 1 raises ValueError.
        with pytest.raises(ValueError, match="pool_factor must be >= 1"):
            HierarchicalTokenPooling(pool_factor=bad)

    def test_num_protected_tokens_must_be_non_negative(self) -> None:
        # [FUNCTION DOC]
        # Asserts a negative num_protected_tokens raises ValueError.
        with pytest.raises(ValueError, match="num_protected_tokens must be >= 0"):
            HierarchicalTokenPooling(num_protected_tokens=-1)

    def test_tasks_accepts_a_bare_string(self) -> None:
        # [FUNCTION DOC]
        # Asserts a bare string for tasks is coerced to a one-element list.
        # Without the coercion, list("query") would silently split into characters.
        assert HierarchicalTokenPooling(tasks="query").tasks == ["query"]

    def test_tasks_round_trip_through_config(self, tmp_path) -> None:
        # [FUNCTION DOC]
        # Saves pooling with tasks and pool_factor to tmp_path, loads it, and asserts both survive.
        HierarchicalTokenPooling(pool_factor=2, tasks=["query", "document"]).save(str(tmp_path))
        restored = HierarchicalTokenPooling.load(str(tmp_path))
        assert restored.tasks == ["query", "document"]
        assert restored.pool_factor == 2


class TestForwardEdgeCases:
    # [CLASS DOC]
    # Tests forward on degenerate batches.
    def test_forward_empty_batch(self) -> None:
        # [FUNCTION DOC]
        # For hierarchical and identity lambda pooling, asserts an empty batch keeps 3-D token embeddings (0, 5, 4) and a 2-D mask (0, 5).
        # A features dict with a 0-batch must round-trip without collapsing to 1D. Cover both a
        # subclass that routes through super().forward (Hierarchical at pool_factor>1) and one that
        # reaches BaseTokenPooling.forward directly (Lambda).
        for pooling in (HierarchicalTokenPooling(pool_factor=2), LambdaTokenPooling(pool_fn=lambda x: x)):
            features = {
                "token_embeddings": torch.empty((0, 5, 4)),
                "attention_mask": torch.empty((0, 5), dtype=torch.bool),
            }
            out = pooling.forward(features, task="document")
            assert out["token_embeddings"].shape == (0, 5, 4)
            assert out["attention_mask"].shape == (0, 5)

    def test_forward_fully_padded_row_in_batch(self) -> None:
        # [FUNCTION DOC]
        # With a halving lambda, asserts a fully masked row pools to zero tokens while other rows pool normally, and the output mask is right-padded to the
        # longest pooled length.
        # A fully-padded (all-masked) row pools to zero tokens. The other rows pool normally and the
        # batch right-pads to the max pooled length.
        pooling = LambdaTokenPooling(pool_fn=lambda emb: emb[: max(emb.size(0) // 2, 1)] if emb.size(0) else emb)
        embeddings = torch.randn(3, 10, 4)
        attention_mask = torch.zeros(3, 10, dtype=torch.bool)
        # Row 0: fully padded (all False). Row 1 and 2: right-padded.
        attention_mask[1, :6] = True
        attention_mask[2, :4] = True
        out = pooling.forward({"token_embeddings": embeddings, "attention_mask": attention_mask}, task="document")
        # Row 0 → 0 tokens (identity of empty), row 1 → 3, row 2 → 2. Max len = 3, right-padded.
        assert out["attention_mask"].tolist() == [
            [False, False, False],
            [True, True, True],
            [True, True, False],
        ]


class TestBaseForwardVariableLength:
    # [CLASS DOC]
    # Tests the base class's re-padding and mask reconstruction with variable lengths.
    def test_base_forward_via_lambda_pooling_variable_length(self) -> None:
        # [FUNCTION DOC]
        # With a halving lambda over rows with 8 and 6 real tokens, asserts output shape (2, 4, 4) and a right-padded mask of 4 and 3 trues.
        # Exercise BaseTokenPooling.forward via LambdaTokenPooling (HierarchicalTokenPooling.forward
        # short-circuits with a fast path, so the base's re-pad + mask reconstruction is untested
        # unless we route through Lambda).
        pooling = LambdaTokenPooling(pool_fn=lambda emb: emb[: max(emb.size(0) // 2, 1)])
        # Batch of two documents with different real lengths (mask=1 for the first 8 / 6 rows).
        embeddings = torch.randn(2, 10, 4)
        attention_mask = torch.zeros(2, 10, dtype=torch.bool)
        attention_mask[0, :8] = True
        attention_mask[1, :6] = True
        out = pooling.forward({"token_embeddings": embeddings, "attention_mask": attention_mask}, task="document")
        # Halving lengths: doc 0 -> 4 tokens, doc 1 -> 3 tokens. Padded to max=4.
        assert out["token_embeddings"].shape == (2, 4, 4)
        assert out["attention_mask"].tolist() == [
            [True, True, True, True],
            [True, True, True, False],
        ]


class TestUnbindAndPadEdge:
    # [CLASS DOC]
    # Tests unbinding and re-padding edge cases of pool().
    def test_3d_input_left_padded_with_mask(self) -> None:
        # [FUNCTION DOC]
        # Pools a left-padded batch with an explicit mask and left padding side; asserts shape (2, 5, 4) and zeros in the shorter row's first two positions.
        # Left-padded 3D input WITH mask (mask should short-circuit padding_side detection).
        docs = [_normed((6, 4)), _normed((10, 4))]
        max_len = max(len(d) for d in docs)
        padded = torch.stack([torch.cat([torch.zeros(max_len - len(d), 4), d], dim=0) for d in docs])
        mask = torch.zeros(2, max_len, dtype=torch.bool)
        mask[0, -6:] = True
        mask[1, -10:] = True
        pooling = HierarchicalTokenPooling(pool_factor=2)
        out = pooling.pool(padded, attention_mask=mask, padding_side="left")
        assert isinstance(out, Tensor)
        # Output re-padded on the specified side (left). Each doc pooled to ceil(t / 2).
        # Doc 0: 6 tokens -> 3 pooled. Doc 1: 10 tokens -> 5 pooled. Padded to max=5.
        assert out.shape == (2, 5, 4)
        # Left-padding preserved on the shorter row.
        assert torch.all(out[0, :2] == 0)

    def test_3d_input_empty_batch(self) -> None:
        # [FUNCTION DOC]
        # Asserts an empty 3-D batch returns a 3-D tensor of shape (0, 0, 4).
        # Batch=0 3D input must round-trip as a 3D empty tensor (not collapse to 1D).
        empty = torch.empty((0, 5, 4))
        pooling = HierarchicalTokenPooling(pool_factor=2)
        out = pooling.pool(empty)
        assert isinstance(out, Tensor)
        assert out.shape == (0, 0, 4)

    def test_3d_bad_padding_side_raises(self) -> None:
        # [FUNCTION DOC]
        # Asserts an invalid padding_side raises ValueError.
        padded = torch.randn(2, 6, 4)
        pooling = HierarchicalTokenPooling(pool_factor=2)
        with pytest.raises(ValueError, match="padding_side must be"):
            pooling.pool(padded, padding_side="middle")

    def test_3d_bad_padding_side_raises_even_with_mask(self) -> None:
        # [FUNCTION DOC]
        # Asserts an invalid padding_side raises ValueError even when a mask is given.
        # `_unbind_padded` short-circuits its own padding_side validation when a mask is passed,
        # so the pool() entry point must own the validation.
        padded = torch.randn(2, 6, 4)
        mask = torch.ones(2, 6, dtype=torch.bool)
        pooling = HierarchicalTokenPooling(pool_factor=2)
        with pytest.raises(ValueError, match="padding_side must be"):
            pooling.pool(padded, attention_mask=mask, padding_side="middle")

    def test_left_pad_middle_zero_preserved_no_mask(self) -> None:
        # [FUNCTION DOC]
        # With identity lambda pooling and left padding without a mask, asserts the leading zero row is dropped and the real middle zero row kept (1x3x2).
        # KNOWN ISSUE: The trailing comment says the output stays left-padded with a zero leading position, but with a single row of 3 real tokens the
        # asserted shape has no padding at all.
        # Symmetric to test_3d_unbind_preserves_middle_zero_row but for left padding: the leading
        # zero row is padding (must drop). The middle zero row is a real all-zero embedding (must keep).
        emb = torch.stack(
            [
                torch.tensor([0.0, 0.0]),  # padding at the left
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 0.0]),  # zero-valued but "real"
                torch.tensor([0.5, 0.5]),
            ]
        )
        padded = emb.unsqueeze(0)  # (1, 4, 2)
        out = LambdaTokenPooling(pool_fn=lambda x: x).pool(padded, padding_side="left")
        # Expected: 3 real rows kept, left-padded so leading position stays zero.
        assert out.shape == (1, 3, 2)
