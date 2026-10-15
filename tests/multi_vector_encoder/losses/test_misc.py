# [MODULE DOC]
# Tests for MultiVector* losses (late-interaction MaxSim ranking, cached/GradCache variants, distillation and margin losses) using stub models that
# pass precomputed per-token embeddings through, so no real encoder is needed. Main focus: document columns with different token counts across
# and within columns (as produced by native-resolution VLMs) are padded to a common length with masks so padded positions never score, flattened
# VLM inputs are sliced grid-aware in mini-batches, gradient equivalence between cached and non-cached losses, task propagation from collated
# features, and merged forward passes of homogeneous columns.
"""Cross-column varlen support in MultiVector* losses.

Native-resolution VLMs (Qwen2-VL family, ColIdefics3, ...) emit different per-column token counts:
the same batch can have the positive column at T=10 and the negative column at T=14, and there is
no way to fix this with collator padding. The ``MultipleNegativesRankingLoss`` / ``CachedMNRL``
``torch.stack(..., dim=1)`` would then fail with "stack expects each tensor to be equal size".

This file pins the regression: each loss pads cross-column to the per-batch max token count and
threads the resulting mask so MaxSim never scores against padded positions.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

import sentence_transformers.multi_vector_encoder.losses.cached_multiple_negatives_ranking as cached_mnrl_module
import sentence_transformers.multi_vector_encoder.losses.multiple_negatives_ranking as mnrl_module
import sentence_transformers.util.distributed as distributed_module
from sentence_transformers.base.losses.gradcache import _minibatch_ranges
from sentence_transformers.base.losses.merged_forward import column_merging_disabled, merge_feature_batches
from sentence_transformers.multi_vector_encoder import losses as mve_losses


class _PassthroughModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Stub model whose call returns the given feature dict unchanged (it must already contain token_embeddings and attention_mask), ignoring task.
    """Stub model that returns the ``token_embeddings`` already placed in the feature dict.

    The losses call ``self.model(sf, task=...)["token_embeddings"]``. This stub bypasses tokenisation
    so we can construct ragged document columns explicitly and verify the loss handles them.
    """

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # [FUNCTION DOC]
        # HELPER: Returns the input features unchanged.
        return features


def _make_feature(t_tokens: int, batch: int, dim: int, seed: int) -> dict[str, Tensor]:
    # [FUNCTION DOC]
    # HELPER: Returns a feature dict with seeded random L2-normalized token embeddings of shape batch by tokens by dim that require gradients, and an
    # all-true boolean attention mask.
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.functional.normalize(torch.randn(batch, t_tokens, dim, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    return {
        "token_embeddings": emb,
        "attention_mask": torch.ones(batch, t_tokens, dtype=torch.bool),
    }


@pytest.fixture
def varlen_features() -> list[dict[str, Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Fixture returning a triplet batch of 4 rows and dim 8: query with 6 tokens, positive with 10 and negative with 14 (cross-column ragged).
    """Triplet batch where the positive column has 10 tokens and the negative has 14. This is the
    case that fails the pre-fix ``torch.stack(embeddings[1:], dim=1)``."""
    batch, dim = 4, 8
    return [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=1),  # query
        _make_feature(t_tokens=10, batch=batch, dim=dim, seed=2),  # positive (T=10)
        _make_feature(t_tokens=14, batch=batch, dim=dim, seed=3),  # negative (T=14)
    ]


def test_mnr_handles_varlen_document_columns(varlen_features) -> None:
    # [FUNCTION DOC]
    # Asserts the non-cached multi-vector MNRL yields a finite loss on the ragged triplet and can backpropagate.
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


def test_cached_mnr_handles_varlen_document_columns(varlen_features) -> None:
    # [FUNCTION DOC]
    # Asserts the cached multi-vector MNRL (mini-batch 2) yields a finite loss on the ragged triplet and can backpropagate.
    """Cross-column varlen through the GradCache path. Exercises the same surface as the non-cached
    MNRL test but with chunked embedding forward + cached gradients."""
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), mini_batch_size=2, show_progress_bar=False
    )
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


class _GridCheckingModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Stub model that, when pixel_values are present, asserts their row count equals the total patches implied by the sliced image_grid_thw and
    # that num_images_per_sample has one entry per sample; then returns seeded random normalized embeddings with 5 tokens per row and a full mask.
    """Asserts that flattened VLM tensors arrive consistently sliced in every mini-batch call:
    ``pixel_values`` holds patch rows (not sample rows), so a naive ``v[begin:end]`` slice hands the
    vision tower patch rows from the wrong samples while the sliced grid demands a different count."""

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # [FUNCTION DOC]
        # HELPER: Performs the grid consistency assertions (raising AssertionError on mismatch) and returns random per-token embeddings seeded by batch size.
        batch = features["input_ids"].shape[0]
        if "pixel_values" in features:
            expected_patches = int(features["image_grid_thw"].prod(dim=1).sum().item())
            assert features["pixel_values"].shape[0] == expected_patches, (
                f"pixel rows ({features['pixel_values'].shape[0]}) must match the sliced grid ({expected_patches})"
            )
            assert features["num_images_per_sample"].shape[0] == batch
        g = torch.Generator().manual_seed(batch)
        emb = torch.nn.functional.normalize(torch.randn(batch, 5, 8, generator=g), p=2, dim=-1)
        return {"token_embeddings": emb, "attention_mask": torch.ones(batch, 5, dtype=torch.bool)}


def test_cached_mnr_slices_flattened_vlm_inputs_grid_aware() -> None:
    # [FUNCTION DOC]
    # Builds a 4-row query column of input ids and a document column with flattened pixel_values for one image per sample with patch counts 4, 8, 4, 12.
    # Runs the cached loss with mini-batch 2 under no-grad through the grid-checking model and asserts a finite loss (slicing follows grid offsets).
    """Native-resolution VLM columns (Qwen2-VL family) flatten per-sample visual tokens into one
    ``pixel_values`` tensor with an ``image_grid_thw`` row per image. The GradCache mini-batching
    must slice those by grid offsets, not by sample index."""
    batch = 4
    query_features = {"input_ids": torch.ones(batch, 4, dtype=torch.long)}
    # Per-image patch counts 4 / 8 / 4 / 12: sample slicing would pass 2 patch rows where 12+ are needed.
    grid = torch.tensor([[1, 2, 2], [1, 2, 4], [1, 2, 2], [1, 4, 3]])
    doc_features = {
        "input_ids": torch.ones(batch, 5, dtype=torch.long),
        "pixel_values": torch.randn(int(grid.prod(dim=1).sum()), 6),
        "image_grid_thw": grid,
        "num_images_per_sample": torch.ones(batch, dtype=torch.long),
    }

    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_GridCheckingModel(), mini_batch_size=2, show_progress_bar=False
    )
    with torch.no_grad():
        value = loss([query_features, doc_features], labels=None)
    assert torch.isfinite(value)


def test_cached_mnr_gradients_match_non_cached() -> None:
    # [FUNCTION DOC]
    # Builds identical ragged triplets twice (6 rows), backpropagates the plain and cached (mini-batch 2) losses, and asserts equal loss values and
    # equal gradients on every column's token embeddings within 1e-6.
    """GradCache's contract is gradient equivalence with the non-cached loss. Regression pin for
    the per-chunk backward normalization: it must run on the batch-averaged loss inside the graph,
    or every gradient is silently scaled by ``batch_size`` while the mean is reported."""
    batch, dim = 6, 8

    def build_features() -> list[dict[str, Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Returns a fresh triplet of features with 6, 10 and 14 tokens using fixed seeds.
        return [
            _make_feature(t_tokens=6, batch=batch, dim=dim, seed=1),
            _make_feature(t_tokens=10, batch=batch, dim=dim, seed=2),
            _make_feature(t_tokens=14, batch=batch, dim=dim, seed=3),
        ]

    plain_features = build_features()
    plain_loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    plain_value = plain_loss(plain_features, labels=None)
    plain_value.backward()

    cached_features = build_features()
    cached_loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), mini_batch_size=2, show_progress_bar=False
    )
    cached_value = cached_loss(cached_features, labels=None)
    cached_value.backward()

    assert torch.allclose(plain_value, cached_value, atol=1e-6)
    for plain_sf, cached_sf in zip(plain_features, cached_features):
        assert torch.allclose(plain_sf["token_embeddings"].grad, cached_sf["token_embeddings"].grad, atol=1e-6)


def _make_ragged_feature(lengths: list[int], dim: int, seed: int) -> dict[str, Tensor]:
    # [FUNCTION DOC]
    # HELPER: Returns seeded random normalized embeddings padded to the longest given length with a right-padded mask marking each row's real length.
    # Requires gradients.
    # KNOWN ISSUE: This helper is defined twice in the module under the same name; the later definition replaces this one at import time, so this copy
    # is dead code (both behave the same).
    """Like ``_make_feature``, but each sample has its own real token count. ``mini_batch_num_tokens``
    packs by real tokens, so it only produces uneven mini-batches when the mask is ragged."""
    g = torch.Generator().manual_seed(seed)
    batch, width = len(lengths), max(lengths)
    emb = torch.nn.functional.normalize(torch.randn(batch, width, dim, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    mask = torch.zeros(batch, width, dtype=torch.bool)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    return {"token_embeddings": emb, "attention_mask": mask}


def test_cached_mnr_token_budget_matches_non_cached() -> None:
    # [FUNCTION DOC]
    # With ragged query and positive columns, asserts a 10-token budget really splits the query column into several uneven ranges, then asserts the cached
    # loss with that budget matches the plain loss in value and gradients (gradients must be non-zero).
    """``mini_batch_num_tokens`` packs mini-batches by real token count rather than by sequence count,
    which must not change the loss or the gradients. Padded positions dropped by the mini-batch pad
    trimming are masked out anyway, so both paths have to agree exactly."""
    dim = 8
    query_lengths = [3, 6, 2, 5, 4, 7]
    positive_lengths = [8, 3, 9, 4, 6, 2]

    def build_features() -> list[dict[str, Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Returns fresh ragged query and positive features with fixed seeds.
        return [
            _make_ragged_feature(query_lengths, dim=dim, seed=1),
            _make_ragged_feature(positive_lengths, dim=dim, seed=2),
        ]

    # Guard against a vacuous comparison: the budget must genuinely split the column into
    # uneven mini-batches, rather than falling back to one chunk for the whole batch.
    ranges = _minibatch_ranges(build_features()[0], mini_batch_size=len(query_lengths), mini_batch_num_tokens=10)
    assert len(ranges) > 1, f"token budget did not split the batch: {ranges}"
    assert len({end - begin for begin, end in ranges}) > 1, f"expected uneven mini-batches, got {ranges}"

    plain_features = build_features()
    plain_loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    plain_value = plain_loss(plain_features, labels=None)
    plain_value.backward()

    cached_features = build_features()
    cached_loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(),
        mini_batch_num_tokens=10,
        show_progress_bar=False,
    )
    cached_value = cached_loss(cached_features, labels=None)
    cached_value.backward()

    assert torch.allclose(plain_value, cached_value, atol=1e-6)
    for plain_sf, cached_sf in zip(plain_features, cached_features):
        # Guard the comparison against a loss that silently produces no gradient at all.
        assert cached_sf["token_embeddings"].grad.abs().sum() > 0
        assert torch.allclose(plain_sf["token_embeddings"].grad, cached_sf["token_embeddings"].grad, atol=1e-6)


def test_cached_mnr_rejects_non_positive_token_budget() -> None:
    # [FUNCTION DOC]
    # Asserts the cached loss constructor rejects a token budget of 0 with ValueError.
    """The constructor runs the shared validation, so a non-positive budget fails at build time
    rather than producing a degenerate mini-batch split later."""
    with pytest.raises(ValueError, match="mini_batch_num_tokens must be a positive integer or None."):
        mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel(), mini_batch_num_tokens=0)


class _RaggedTrimPassthroughModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Stub model that trims each call's features to the longest real length in that mini-batch, so successive mini-batches of one column return
    # different token counts.
    """Trims trailing pad off each mini-batch to the per-batch real max, simulating a
    native-resolution VLM whose vision encoder emits ``T`` proportional to the batch's longest
    real sequence. Successive mini-batches in the GradCache path therefore see different ``T``s."""

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # [FUNCTION DOC]
        # HELPER: Returns token embeddings and boolean mask cut to the maximum real token count in the batch.
        emb = features["token_embeddings"]
        mask = features["attention_mask"].bool()
        T_real = int(mask.sum(dim=1).max().item())
        return {
            "token_embeddings": emb[:, :T_real].contiguous(),
            "attention_mask": mask[:, :T_real],
        }


def _make_ragged_feature(real_lengths: list[int], dim: int, seed: int) -> dict[str, Tensor]:
    # [FUNCTION DOC]
    # HELPER: Returns seeded random normalized embeddings (requiring gradients) padded to the longest real length, with a right-padded boolean mask.
    T_max = max(real_lengths)
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.functional.normalize(torch.randn(len(real_lengths), T_max, dim, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    mask = torch.zeros(len(real_lengths), T_max, dtype=torch.bool)
    for i, real_t in enumerate(real_lengths):
        mask[i, :real_t] = True
    return {"token_embeddings": emb, "attention_mask": mask}


def test_cached_mnr_handles_varlen_mini_batches_within_a_column() -> None:
    # [FUNCTION DOC]
    # With the trimming stub and mini-batch 2, so mini-batches of the same column return different token counts, asserts the cached loss is finite and
    # backpropagates.
    """The GradCache cat path pads each mini-batch chunk to a common ``T`` before concatenating,
    so native-resolution VLMs that emit different ``T`` per mini-batch within the same column flow
    through cleanly. Distinct from the cross-column ragged-``T`` case: that is solved by
    ``stack_padded_token_embeddings``. This one is solved by ``cat_padded_token_embeddings``."""
    dim = 8
    # mini_batch_size=2 splits each column into [rows 0-1, rows 2-3]. The trim passthrough returns
    # T=max-real per mini-batch, so the positive column emits T=10 then T=7, and the negative
    # column emits T=12 then T=11. The pre-fix ``torch.cat`` failed on these mismatched ``T``s.
    features = [
        _make_ragged_feature([5, 5, 4, 5], dim=dim, seed=1),
        _make_ragged_feature([10, 9, 6, 7], dim=dim, seed=2),
        _make_ragged_feature([8, 12, 9, 11], dim=dim, seed=3),
    ]

    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_RaggedTrimPassthroughModel(), mini_batch_size=2, show_progress_bar=False
    )
    value = loss(features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


def test_mnr_pad_positions_do_not_influence_loss(varlen_features) -> None:
    # [FUNCTION DOC]
    # Computes the loss on the ragged triplet, then extends the positive column with 4 extra positions filled with 1000 and masked out, and asserts the
    # loss is unchanged within 1e-5.
    """Sanity: corrupting the (would-be-)padded positions of the shorter column to a value that
    would dominate MaxSim must NOT change the loss, because the mask excludes them."""
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    baseline = loss(varlen_features, labels=None).item()

    # Reshape: make the positive column LONGER but stuff garbage past its real length, mask covers
    # the real range. This mirrors what the loss does internally. If the mask flows correctly, the
    # loss is identical to the baseline above.
    short_pos = varlen_features[1]["token_embeddings"]
    batch, real_t, dim = short_pos.shape
    extended = torch.cat([short_pos, torch.full((batch, 4, dim), 1e3, device=short_pos.device)], dim=1)
    extended.requires_grad_(True)
    corrupted = [
        varlen_features[0],
        {
            "token_embeddings": extended,
            "attention_mask": torch.cat(
                [
                    torch.ones(batch, real_t, dtype=torch.bool),
                    torch.zeros(batch, 4, dtype=torch.bool),
                ],
                dim=1,
            ),
        },
        varlen_features[2],
    ]
    corrupted_value = loss(corrupted, labels=None).item()
    assert abs(baseline - corrupted_value) < 1e-5, (
        f"masked-out positions must not influence the loss: baseline={baseline}, corrupted={corrupted_value}"
    )


def test_mnr_fixed_length_regression() -> None:
    # [FUNCTION DOC]
    # Asserts the loss is finite when both document columns share the same token count.
    """When all document columns share the same T (text / fixed-resolution VLM case), the new
    pad logic is a no-op: same batch, same loss as before."""
    batch, dim = 3, 8
    same_t_features = [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=11),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=12),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=13),
    ]
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    value = loss(same_t_features, labels=None)
    assert torch.isfinite(value)


class _TaskRecordingModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Stub model recording the task argument of every call and returning features unchanged.
    """Records the ``task`` passed on each call and passes features through."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Initializes the module with an empty list of seen tasks.
        super().__init__()
        self.seen_tasks: list[str | None] = []

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # [FUNCTION DOC]
        # HELPER: Appends the task to seen_tasks and returns the features.
        self.seen_tasks.append(task)
        return features


def test_losses_use_collator_stamped_task_over_positional(varlen_features) -> None:
    # [FUNCTION DOC]
    # Stamps task "document" on the first column and "query" on the second, runs the loss, and asserts the model saw document, query, then document for
    # the unstamped third column (positional fallback). Mutates the fixture's dicts.
    """Losses read the ``task`` the collator stamped into each column's features (positional
    query/document only as fallback), so ``router_mapping`` overrides reach the model forward."""
    varlen_features[0]["task"] = "document"
    varlen_features[1]["task"] = "query"
    model = _TaskRecordingModel()
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=model)
    loss(varlen_features, labels=None)
    # Third column carries no stamp and falls back to the positional default.
    assert model.seen_tasks == ["document", "query", "document"]


def test_cached_mnr_stamped_task_reaches_backward_hook(varlen_features) -> None:
    # [FUNCTION DOC]
    # Stamps task "document" on the query column and runs the cached loss (mini-batch 2) forward and backward with a task-recording model; asserts 12
    # calls (two mini-batches per column, three columns, forward and backward re-embedding) all with task document.
    """The GradCache second pass re-embeds from the same feature dicts: the stamped task must
    govern both the no-grad forward and the with-grad re-run."""
    varlen_features[0]["task"] = "document"  # override the positional "query" for column 0
    model = _TaskRecordingModel()
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=model, mini_batch_size=2, show_progress_bar=False
    )
    value = loss(varlen_features, labels=None)
    value.backward()
    # 4 rows / mini_batch_size=2 -> 2 calls per column per phase, 3 columns, 2 phases = 12 calls.
    assert len(model.seen_tasks) == 12
    assert all(task == "document" for task in model.seen_tasks), model.seen_tasks


class _MaskRewritingModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Stub model returning a new dict whose attention mask excludes the last two positions for non-query tasks, leaving the input dict's mask
    # untouched (simulating a mask module such as a skiplist).
    """Returns a FRESH output dict whose attention_mask excludes the last two document positions,
    mimicking a mask module (e.g. skiplist) rewriting the scoring mask. The input dict keeps its
    original mask, so a loss reading the input mask scores excluded positions."""

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # [FUNCTION DOC]
        # HELPER: Copies the input mask as booleans, masks the last two positions unless the task is query, and returns the input embeddings with that mask.
        mask = features["attention_mask"].bool().clone()
        if task != "query":
            mask[:, -2:] = False
        return {"token_embeddings": features["token_embeddings"], "attention_mask": mask}


def test_losses_read_scoring_mask_from_model_output() -> None:
    # [FUNCTION DOC]
    # For MNRL, MarginMSE (zero labels) and DistillKLDiv (three ragged candidate columns, 2x3 teacher scores), compares the loss on clean features with
    # the loss after writing 1000 into the last two document positions, and asserts no difference (the losses use the output mask).
    """The scoring mask comes from the model OUTPUT dict (where the mask module rewrote it), not
    from the input dict: garbage at positions the output mask excludes must not influence any loss."""
    batch, dim = 3, 8

    def build(corrupt: bool, n_docs: int, batch_size: int = batch) -> list[dict[str, Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Returns a query feature plus n_docs 10-token document features; when corrupt, overwrites their last two positions with 1000 without gradients.
        features = [_make_feature(t_tokens=6, batch=batch_size, dim=dim, seed=1)]
        for idx in range(n_docs):
            doc = _make_feature(t_tokens=10, batch=batch_size, dim=dim, seed=2 + idx)
            if corrupt:
                with torch.no_grad():
                    doc["token_embeddings"][:, -2:] = 1e3
            features.append(doc)
        return features

    model = _MaskRewritingModel()

    mnrl = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=model)
    baseline = mnrl(build(corrupt=False, n_docs=2), labels=None).item()
    corrupted = mnrl(build(corrupt=True, n_docs=2), labels=None).item()
    assert abs(baseline - corrupted) < 1e-5, f"MNRL read the input mask: {baseline} vs {corrupted}"

    margin_labels = torch.zeros(batch)
    margin_mse = mve_losses.MultiVectorMarginMSELoss(model=model)
    baseline = margin_mse(build(corrupt=False, n_docs=2), margin_labels).item()
    corrupted = margin_mse(build(corrupt=True, n_docs=2), margin_labels).item()
    assert abs(baseline - corrupted) < 1e-5, f"MarginMSE read the input mask: {baseline} vs {corrupted}"

    # KD shape: 2 queries with 3 candidate document columns and (2, 3) teacher scores.
    kd_labels = torch.tensor([[5.0, 1.0, 0.1], [4.0, 0.5, 0.2]])
    kd = mve_losses.MultiVectorDistillKLDivLoss(model=model)

    def build_kd(corrupt: bool) -> list[dict[str, Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Returns a 2-row query feature plus three document features with 10, 12 and 14 tokens; when corrupt, overwrites the last two positions of
        # each document with 1000.
        query = _make_feature(t_tokens=6, batch=2, dim=dim, seed=7)
        # Ragged token counts across columns exercise the pad branch of stack_padded_token_embeddings,
        # which is the production path (columns are padded independently by the collator).
        docs = [_make_feature(t_tokens=10 + 2 * way, batch=2, dim=dim, seed=8 + way) for way in range(3)]
        if corrupt:
            with torch.no_grad():
                for doc in docs:
                    doc["token_embeddings"][:, -2:] = 1e3
        return [query, *docs]

    baseline = kd(build_kd(corrupt=False), kd_labels).item()
    corrupted = kd(build_kd(corrupt=True), kd_labels).item()
    assert abs(baseline - corrupted) < 1e-5, f"DistillKLDiv read the input mask: {baseline} vs {corrupted}"


def test_similarity_fct_config_includes_hyperparameters() -> None:
    # [FUNCTION DOC]
    # Asserts get_config_dict renders XTRScores and XTRKDScores objects with their top_k and chunk_elements values for MNRL, cached MNRL and DistillKLDiv,
    # and the default scorer as the bare name "colbert_scores".
    """Configured metric objects serialize with their hyperparameters, so ``XTRScores(top_k=16)`` and
    ``XTRScores(top_k=256)`` are distinguishable in model cards and logs. Plain functions keep
    serializing as their bare name."""
    from sentence_transformers.multi_vector_encoder.scoring import XTRKDScores, XTRScores

    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), similarity_fct=XTRScores(top_k=16)
    )
    assert loss.get_config_dict()["similarity_fct"] == "XTRScores(top_k=16, chunk_elements=None)"

    cached = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), similarity_fct=XTRScores(top_k=16, chunk_elements=4)
    )
    assert cached.get_config_dict()["similarity_fct"] == "XTRScores(top_k=16, chunk_elements=4)"

    kd = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel(), similarity_fct=XTRKDScores(top_k=8))
    assert kd.get_config_dict()["similarity_fct"] == "XTRKDScores(top_k=8, chunk_elements=None)"

    default = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    assert default.get_config_dict()["similarity_fct"] == "colbert_scores"


def test_mnr_gather_across_devices_single_process_matches_no_gather(varlen_features) -> None:
    # [FUNCTION DOC]
    # Asserts that on a single process gather_across_devices gives the same finite loss as without gathering, and that it backpropagates.
    """On a single process ``gather_across_devices=True`` must be a no-op: ``all_gather`` falls back to
    the local tensor and ``all_gather_padded`` skips the cross-rank pad, so the loss equals the
    non-gathered loss (world_size=1, rank=0). Guards the gather wiring without a real process group.
    The cross-rank pad itself is only reachable under multi-GPU DDP, which this cannot exercise."""
    base = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    gathered = mve_losses.MultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), gather_across_devices=True
    )
    baseline = base(varlen_features, labels=None)
    value = gathered(varlen_features, labels=None)
    assert torch.isfinite(value)
    assert torch.allclose(value, baseline), f"gather no-op mismatch: {value.item()} vs {baseline.item()}"
    value.backward()


def test_cached_mnr_gather_across_devices_single_process(varlen_features) -> None:
    # [FUNCTION DOC]
    # Asserts the cached loss with gathering enabled on a single process gives a finite loss and backpropagates.
    """Single-process gather guard for the GradCache loss: the gather block must run (all_gather
    no-ops without a process group) and produce a finite, differentiable loss."""
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), mini_batch_size=2, show_progress_bar=False, gather_across_devices=True
    )
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


def test_distill_kl_div_validates_input_columns_and_label_shape() -> None:
    # [FUNCTION DOC]
    # Asserts DistillKLDiv raises ValueError for only a query column or a query plus one document column (needs at least 3 features), and for teacher
    # scores whose width does not match the number of document columns.
    """Too few input columns, or teacher scores whose width does not match the number of document
    columns, must fail loud instead of producing an opaque shape error. A single document column is
    rejected too: a softmax over one candidate is constant, so it would train nothing at a perfect
    reported loss."""
    loss = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel())
    queries = _make_feature(t_tokens=4, batch=3, dim=8, seed=1)
    with pytest.raises(ValueError, match="at least 3 sentence features"):
        loss([queries], torch.randn(3, 2))

    one_document = _make_feature(t_tokens=6, batch=3, dim=8, seed=2)
    with pytest.raises(ValueError, match="at least 3 sentence features"):
        loss([queries, one_document], torch.randn(3, 1))

    documents = [_make_feature(t_tokens=6, batch=3, dim=8, seed=2 + way) for way in range(2)]
    with pytest.raises(ValueError, match="teacher scores"):
        loss([queries, *documents], torch.randn(3, 3))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), False])
def test_distill_kl_div_rejects_unusable_temperatures(bad: float) -> None:
    # [FUNCTION DOC]
    # Parametrized over zero, negative, NaN, infinity and False: asserts each temperature argument rejects the value with ValueError.
    """A non-positive temperature silently inverts the objective and NaN poisons every weight on the
    first backward, so both are rejected at construction like the evaluator already does."""
    for kwargs in ({"temperature": bad}, {"student_temperature": bad}, {"teacher_temperature": bad}):
        with pytest.raises(ValueError, match="must be a positive finite number"):
            mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel(), **kwargs)


def test_distill_kl_div_warns_once_when_the_teacher_collapses(caplog) -> None:
    # [FUNCTION DOC]
    # With teacher temperature 0.1 and a large score spread, asserts the first forward logs a warning naming the temperature and that candidates carry
    # no gradient, the second forward logs nothing, and a default-temperature loss logs nothing. Uses caplog.
    """This loss works in log space, so it hands the check ``teacher_log_probs.exp()``, which is the
    target ``KLDivLoss(log_target=True)`` consumes rather than a separately computed softmax."""

    def build_features() -> list[dict[str, Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Returns fresh query and two document features with 4, 5 and 6 tokens.
        # Rebuilt per call: the loss forward rewrites the attention masks of its features.
        return [_make_feature(t_tokens=4 + way, batch=2, dim=8, seed=1 + way) for way in range(3)]

    labels = torch.tensor([[8.5, -10.0], [7.0, -12.0]])
    loss = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel(), teacher_temperature=0.1)
    with caplog.at_level("WARNING"):
        loss(build_features(), labels)
    assert "teacher_temperature=0.1" in caplog.text
    assert "carry no gradient" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        loss(build_features(), labels)
    assert caplog.text == "", "the check is latched to the first forward"

    quiet = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel())
    with caplog.at_level("WARNING"):
        quiet(build_features(), labels)
    assert caplog.text == "", "a temperature matched to the spread must not warn"


@pytest.mark.parametrize(
    ("module", "build_loss"),
    [
        (
            mnrl_module,
            lambda **kwargs: mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel(), **kwargs),
        ),
        (
            cached_mnrl_module,
            lambda **kwargs: mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
                model=_PassthroughModel(), mini_batch_size=2, show_progress_bar=False, **kwargs
            ),
        ),
    ],
    ids=["mnrl", "cached_mnrl"],
)
def test_mnr_loss_does_not_scale_with_world_size(monkeypatch, module, build_loss) -> None:
    # [FUNCTION DOC]
    # Parametrized over MNRL and cached MNRL with their modules. Computes a baseline, then monkeypatches the distributed helpers to report an initialized
    # group of world size 2 at rank 0 and the module's padded gather to duplicate the local tensors, and asserts the gathered loss equals the baseline
    # plus log 2 (duplicated candidates), not double the baseline.
    """A 2-rank gather returning ``[local; local]`` doubles every query's candidate pool with an exact
    copy, which costs exactly ``log(2)`` of cross-entropy. The ``loss * get_world_size()`` multiplier
    that ``8c4331ec`` removed would double the loss instead. The single-process gather guards above
    cannot see that, because the multiplier is 1 at world size 1, so this reports world size 2 as
    well as faking the gather."""

    def build_features() -> list[dict[str, Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Returns fresh query and two document features with 6, 10 and 14 tokens and batch 4.
        return [_make_feature(t_tokens=6 + 4 * way, batch=4, dim=8, seed=1 + way) for way in range(3)]

    baseline = build_loss()(build_features(), labels=None).item()

    def two_rank_gather(tensor: Tensor, mask: Tensor, with_grad: bool = False) -> tuple[Tensor, Tensor]:
        # [FUNCTION DOC]
        # HELPER: Fake padded gather returning the tensor and mask each concatenated with themselves along the batch dimension.
        return torch.cat([tensor, tensor]), torch.cat([mask, mask])

    # get_world_size / get_rank read these from their own module globals at call time, so patching
    # here reaches the losses whichever way they imported the helpers.
    monkeypatch.setattr(distributed_module, "is_dist_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_world_size", lambda *args, **kwargs: 2)
    monkeypatch.setattr(distributed_module.dist, "get_rank", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "all_gather_padded", two_rank_gather)

    gathered = build_loss(gather_across_devices=True)(build_features(), labels=None).item()
    assert gathered == pytest.approx(baseline + math.log(2), abs=1e-5)


def test_pairwise_scorers_match_the_diagonal_of_their_matrix_form() -> None:
    # [FUNCTION DOC]
    # With seeded normalized queries (3x4x8) and documents (3x5x8), asserts colbert_scores_pairwise equals the diagonal of colbert_scores on documents
    # grouped one per query, and that xtr_scores_pairwise equals the diagonal of xtr_scores when top_k is large enough to include every token.
    """Both pairwise scorers are only covered by plumbing and naming assertions elsewhere. Their
    contract is that ``fn(q, d)[i]`` equals the ``(i, i)`` entry the matrix form produces, so pin
    that against the matrix scorers rather than against a hand-rolled reference."""
    from sentence_transformers.multi_vector_encoder.scoring import (
        colbert_scores,
        colbert_scores_pairwise,
        xtr_scores,
        xtr_scores_pairwise,
    )

    generator = torch.Generator().manual_seed(6)
    queries = torch.nn.functional.normalize(torch.randn(3, 4, 8, generator=generator), p=2, dim=-1)
    documents = torch.nn.functional.normalize(torch.randn(3, 5, 8, generator=generator), p=2, dim=-1)

    # The matrix forms take (Q, N, d_tokens, dim); N=1 makes column i document i.
    grouped = documents.unsqueeze(1)
    colbert_diagonal = colbert_scores(queries, grouped).diagonal()
    assert torch.allclose(colbert_scores_pairwise(queries, documents), colbert_diagonal, atol=1e-5)

    # XTR's top-k spans the whole in-batch pool, so it only agrees when nothing is left out.
    xtr_diagonal = xtr_scores(queries, grouped, top_k=queries.shape[1] * documents.numel()).diagonal()
    pairwise = xtr_scores_pairwise(queries, documents, top_k=queries.shape[1] * documents.numel())
    assert torch.allclose(pairwise, xtr_diagonal, atol=1e-5)


class _TokenizingModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Stub model turning integer input ids into 17-class one-hot float embeddings and passing the attention mask through.
    """Embeds integer ``input_ids`` as one-hot vectors, mirroring how a real model turns padded
    token batches into embeddings while the attention mask marks the real positions."""

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # [FUNCTION DOC]
        # HELPER: Returns one-hot token embeddings and the unchanged attention mask.
        return {
            "token_embeddings": torch.nn.functional.one_hot(features["input_ids"], num_classes=17).float(),
            "attention_mask": features["attention_mask"],
        }


def _int_column(t_tokens: int, seed: int, batch: int = 2, left_pad: int = 0) -> dict[str, Tensor]:
    # [FUNCTION DOC]
    # HELPER: Returns seeded random integer input ids (1 to 16) of shape batch by tokens and an integer mask of ones, with the first left_pad positions zeroed.
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, 17, (batch, t_tokens), generator=generator)
    attention_mask = torch.ones(batch, t_tokens, dtype=torch.long)
    if left_pad:
        attention_mask[:, :left_pad] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _int_features() -> list[dict[str, Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Returns four integer columns of widths 6, 10, 12 (left-padded by 3) and 14.
    # One left-padded column so the chunk trim window must respect padding on either side.
    return [
        _int_column(6, seed=1),
        _int_column(10, seed=2),
        _int_column(12, seed=3, left_pad=3),
        _int_column(14, seed=4),
    ]


def test_distill_kl_div_merged_forward_matches_per_column() -> None:
    # [FUNCTION DOC]
    # Asserts DistillKLDiv gives the same value with merged and per-column forwards for equal-width float columns and for ragged integer token columns
    # (zero-padded on merge). Also asserts ragged float columns and VLM-like features with row-misaligned pixel values refuse to merge.
    """The merged (batch_size * n_ways) document forward must score identically to the per-column
    fallback: for equal-width float columns (merge without padding) and for ragged integer token
    columns (the zero-pad branch, whose padded positions carry attention 0 and never reach scoring)."""
    batch = 2
    kd_labels = torch.tensor([[5.0, 1.0, 0.1], [4.0, 0.5, 0.2]])

    def float_features() -> list[dict[str, Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Returns a 6-token query plus three 10-token float document features.
        return [_make_feature(6, batch, 8, seed=7)] + [_make_feature(10, batch, 8, seed=8 + way) for way in range(3)]

    loss = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel())
    assert merge_feature_batches(float_features()[1:]) is not None
    merged_value = loss(float_features(), kd_labels).item()
    with column_merging_disabled():
        fallback_value = loss(float_features(), kd_labels).item()
    assert abs(merged_value - fallback_value) < 1e-6

    loss = mve_losses.MultiVectorDistillKLDivLoss(model=_TokenizingModel())
    assert merge_feature_batches(_int_features()[1:]) is not None
    merged_value = loss(_int_features(), kd_labels).item()
    with column_merging_disabled():
        fallback_value = loss(_int_features(), kd_labels).item()
    assert abs(merged_value - fallback_value) < 1e-6

    # Ragged float columns cannot be zero-padded safely: the merge must refuse and fall back.
    ragged_float = [_make_feature(10, batch, 8, seed=1), _make_feature(12, batch, 8, seed=2)]
    assert merge_feature_batches(ragged_float) is None
    # Row-misaligned VLM features must refuse as well, even when homogeneous.
    vlm_like = [
        {"input_ids": torch.ones(batch, 4, dtype=torch.long), "pixel_values": torch.randn(7, 3)} for _ in range(2)
    ]
    assert merge_feature_batches(vlm_like) is None


def test_distill_kl_div_mini_batch_matches_full() -> None:
    # [FUNCTION DOC]
    # With the one-hot tokenizing stub, asserts DistillKLDiv with mini_batch_size 2 matches the unchunked loss on the integer columns (one left-padded),
    # both merged and with merging disabled, and that the config reports mini_batch_size None and 2 respectively.
    """mini_batch_size splits the merged forward into row chunks, each embedded at its own width
    (leading padding kept, so the left-padded column keeps its positions): the loss must not change."""
    kd_labels = torch.tensor([[5.0, 1.0, 0.1], [4.0, 0.5, 0.2]])

    full = mve_losses.MultiVectorDistillKLDivLoss(model=_TokenizingModel())
    chunked = mve_losses.MultiVectorDistillKLDivLoss(model=_TokenizingModel(), mini_batch_size=2)
    full_value = full(_int_features(), kd_labels).item()
    chunked_value = chunked(_int_features(), kd_labels).item()
    assert abs(full_value - chunked_value) < 1e-6

    # The fallback path chunks each column forward too.
    with column_merging_disabled():
        chunked_fallback_value = chunked(_int_features(), kd_labels).item()
    assert abs(full_value - chunked_fallback_value) < 1e-6

    assert full.get_config_dict()["mini_batch_size"] is None
    assert chunked.get_config_dict()["mini_batch_size"] == 2


def test_margin_mse_merged_forward_matches_per_column() -> None:
    # [FUNCTION DOC]
    # Asserts MarginMSE with 2x2 margins gives the same value merged, chunked (mini-batch 2) and with merging disabled.
    """MarginMSE splits the merged forward straight back into per-column tensors: merged, chunked,
    and per-column fallback paths must all produce the same loss."""
    margins = torch.tensor([[0.4, 1.2], [0.1, 0.6]])

    merged = mve_losses.MultiVectorMarginMSELoss(model=_TokenizingModel())
    chunked = mve_losses.MultiVectorMarginMSELoss(model=_TokenizingModel(), mini_batch_size=2)
    merged_value = merged(_int_features(), margins).item()
    chunked_value = chunked(_int_features(), margins).item()
    with column_merging_disabled():
        fallback_value = merged(_int_features(), margins).item()
    assert abs(merged_value - fallback_value) < 1e-6
    assert abs(chunked_value - fallback_value) < 1e-6


def test_mnrl_merged_forward_matches_per_column() -> None:
    # [FUNCTION DOC]
    # Asserts MNRL (query kept separate, documents merged) gives the same value merged, chunked and per-column, and reports mini_batch_size in its config.
    """MNRL merges only the document columns, keeping the query column on its own forward: merged,
    chunked, and per-column fallback paths must all produce the same loss."""
    merged = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_TokenizingModel())
    chunked = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_TokenizingModel(), mini_batch_size=2)
    merged_value = merged(_int_features()).item()
    chunked_value = chunked(_int_features()).item()
    with column_merging_disabled():
        fallback_value = merged(_int_features()).item()
    assert abs(merged_value - fallback_value) < 1e-6
    assert abs(chunked_value - fallback_value) < 1e-6

    assert merged.get_config_dict()["mini_batch_size"] is None
    assert chunked.get_config_dict()["mini_batch_size"] == 2


def test_merge_refuses_inside_column_merging_disabled() -> None:
    # [FUNCTION DOC]
    # Asserts two equal-width columns merge normally, refuse to merge inside the column_merging_disabled context, and merge again after leaving it.
    """Wrapper losses that call an inner loss repeatedly with the same feature dicts (AdaptiveLayerLoss)
    pass state through those dicts, which fresh merged dicts would strand."""
    columns = [_int_column(6, seed=1), _int_column(6, seed=2)]
    assert merge_feature_batches(columns) is not None
    with column_merging_disabled():
        assert merge_feature_batches(columns) is None
    assert merge_feature_batches(columns) is not None


def test_merge_refuses_non_tensor_values_that_do_not_compare() -> None:
    # [FUNCTION DOC]
    # Skipped without numpy. Asserts columns carrying numpy arrays as extra metadata (whose inequality is not a boolean) refuse to merge instead of raising.
    """A custom preprocess_fn can put values in the batch whose ``!=`` returns an array rather than a
    bool. Those cannot be shown equal across columns, so the merge must refuse instead of raising."""
    numpy = pytest.importorskip("numpy")
    columns = [
        {**_int_column(6, seed=1), "custom": numpy.array([1, 2, 3])},
        {**_int_column(6, seed=2), "custom": numpy.array([1, 2, 3])},
    ]
    assert merge_feature_batches(columns) is None


def test_merge_refuses_flattened_features() -> None:
    # [FUNCTION DOC]
    # Builds two FlashAttention-2 flattened columns with transformers' DataCollatorWithFlattening (labels removed), asserts they carry cumulative sequence
    # lengths and equal max lengths, and asserts merge refuses them.
    """FA2-flattened columns pack their rows into one sequence described by ``cu_seq_lens_q``.
    Concatenating those offsets end to end describes nothing, so the merge must refuse and let the
    caller fall back to per-column forwards."""
    from transformers import DataCollatorWithFlattening

    collator = DataCollatorWithFlattening(return_tensors="pt", return_flash_attn_kwargs=True)
    columns = [
        {key: value for key, value in collator(rows).items() if key != "labels"}
        for rows in (
            [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}],
            [{"input_ids": [6, 7]}, {"input_ids": [8, 9, 10]}],
        )
    ]
    assert "cu_seq_lens_q" in columns[0]
    # Equal per-column longest documents, so the max_length_q metadata match cannot mask the refusal.
    assert columns[0]["max_length_q"] == columns[1]["max_length_q"]
    assert merge_feature_batches(columns) is None


@pytest.mark.parametrize("num_negatives", [1, 2])
def test_margin_mse_accepts_raw_teacher_scores(num_negatives: int) -> None:
    # [FUNCTION DOC]
    # Parametrized over 1 and 2 negatives. Asserts MarginMSE gives exactly the same loss for raw teacher scores (positive then negatives) as for the
    # precomputed margins (positive score minus each negative score).
    """Raw teacher scores of shape (batch, num_negatives + 1) must yield the same loss as the
    equivalent pre-computed margins, matching the dense MarginMSELoss."""
    batch, dim = 4, 8
    features = [_make_feature(t_tokens=6, batch=batch, dim=dim, seed=1)] + [
        _make_feature(t_tokens=10 + 2 * way, batch=batch, dim=dim, seed=2 + way) for way in range(1 + num_negatives)
    ]
    generator = torch.Generator().manual_seed(42)
    raw_scores = torch.randn(batch, num_negatives + 1, generator=generator)
    margins = raw_scores[:, 0:1] - raw_scores[:, 1:]

    loss = mve_losses.MultiVectorMarginMSELoss(model=_PassthroughModel())
    raw_value = loss(features, raw_scores).item()
    margin_value = loss(features, margins).item()
    assert raw_value == margin_value, f"raw-score labels diverged from margins: {raw_value} vs {margin_value}"


def test_margin_mse_accepts_1d_labels_for_single_negative() -> None:
    # [FUNCTION DOC]
    # Asserts 1-D margins with one negative give exactly the same loss as the equivalent (batch, 1) column.
    """With one negative the margin may arrive as one scalar per row. That 1D form must match the
    equivalent (batch, 1) column vector."""
    batch, dim = 4, 8
    features = [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=1),
        _make_feature(t_tokens=10, batch=batch, dim=dim, seed=2),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=3),
    ]
    generator = torch.Generator().manual_seed(7)
    margins = torch.randn(batch, generator=generator)

    loss = mve_losses.MultiVectorMarginMSELoss(model=_PassthroughModel())
    flat_value = loss(features, margins).item()
    column_value = loss(features, margins.unsqueeze(1)).item()
    assert flat_value == column_value, f"1D labels diverged from (batch, 1): {flat_value} vs {column_value}"


@pytest.mark.parametrize("label_shape", [(2, 3), (1, 1)])
def test_margin_mse_rejects_mismatched_label_shape(label_shape: tuple[int, int]) -> None:
    # [FUNCTION DOC]
    # Parametrized over label shapes with wrong column count or wrong batch size; asserts ValueError mentioning negative columns.
    """Labels must match (batch, num_negatives) exactly. A wrong column count is neither margins nor
    raw scores, and a wrong batch would otherwise broadcast through MSELoss with only a warning."""
    batch, dim = 2, 8
    features = [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=1),
        _make_feature(t_tokens=10, batch=batch, dim=dim, seed=2),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=3),
    ]
    loss = mve_losses.MultiVectorMarginMSELoss(model=_PassthroughModel())
    with pytest.raises(ValueError, match="negative columns"):
        loss(features, torch.randn(*label_shape))


@pytest.mark.parametrize(
    "loss_class",
    [mve_losses.MultiVectorMultipleNegativesRankingLoss, mve_losses.CachedMultiVectorMultipleNegativesRankingLoss],
)
@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_mnr_rejects_non_positive_scale(loss_class: type[nn.Module], scale: float) -> None:
    # [FUNCTION DOC]
    # Parametrized over MNRL and cached MNRL and scales 0 and -1; asserts construction raises ValueError requiring a positive scale.
    """scale=0.0 zeroes every gradient and a negative scale trains backwards, so the constructor
    must reject both, matching the dense MultipleNegativesRankingLoss."""
    with pytest.raises(ValueError, match="Scale must be a positive value."):
        loss_class(model=_PassthroughModel(), scale=scale)


def test_margin_mse_accepts_xtr_pairwise_metric() -> None:
    # [FUNCTION DOC]
    # Asserts MarginMSE accepts xtr_scores_pairwise as similarity (finite loss, config name rendered) and that the default config name is
    # colbert_scores_pairwise.
    """The unified similarity_fct convention makes xtr_scores_pairwise a drop-in MarginMSE scorer,
    previously a TypeError from the a_mask/b_mask call."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores_pairwise

    batch, dim = 2, 8
    features = [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=1),
        _make_feature(t_tokens=10, batch=batch, dim=dim, seed=2),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=3),
    ]
    generator = torch.Generator().manual_seed(11)
    margins = torch.randn(batch, generator=generator)

    loss = mve_losses.MultiVectorMarginMSELoss(model=_PassthroughModel(), similarity_fct=xtr_scores_pairwise)
    assert torch.isfinite(loss(features, margins))
    assert loss.get_config_dict()["similarity_fct"] == "xtr_scores_pairwise"

    default = mve_losses.MultiVectorMarginMSELoss(model=_PassthroughModel())
    assert default.get_config_dict()["similarity_fct"] == "colbert_scores_pairwise"


def test_similarity_fct_config_renders_partial_bindings() -> None:
    # [FUNCTION DOC]
    # Asserts a functools.partial similarity renders as the function name with its bound keyword arguments in the config.
    """A functools.partial similarity_fct (e.g. MeanMaxSim via length_normalize) serializes with its
    bound settings, like configured metric objects do."""
    from functools import partial

    from sentence_transformers.multi_vector_encoder.scoring import colbert_scores

    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), similarity_fct=partial(colbert_scores, length_normalize=True)
    )
    assert loss.get_config_dict()["similarity_fct"] == "colbert_scores(length_normalize=True)"


def test_distill_kl_div_separate_temperatures() -> None:
    # [FUNCTION DOC]
    # Asserts DistillKLDiv with student and teacher temperatures both 0.5 equals a shared temperature of 0.5, differs with 0.5 and 2.0, and that the
    # config dict lists similarity, default shared temperature, both explicit temperatures and mini_batch_size None.
    """student_temperature / teacher_temperature default to the shared temperature and split the two
    softmaxes when set, with the loss scaled by the student temperature squared."""
    batch, dim = 2, 8
    labels = torch.tensor([[4.0, 1.0], [3.5, 0.5]])

    def build_features() -> list[dict[str, Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Returns fresh query and two document features with 4, 6 and 5 tokens.
        return [
            _make_feature(t_tokens=4, batch=batch, dim=dim, seed=1),
            _make_feature(t_tokens=6, batch=batch, dim=dim, seed=2),
            _make_feature(t_tokens=5, batch=batch, dim=dim, seed=3),
        ]

    shared = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel(), temperature=0.5)
    shared_value = shared(build_features(), labels).item()
    aliased = mve_losses.MultiVectorDistillKLDivLoss(
        model=_PassthroughModel(), student_temperature=0.5, teacher_temperature=0.5
    )
    assert aliased(build_features(), labels).item() == pytest.approx(shared_value)

    split = mve_losses.MultiVectorDistillKLDivLoss(
        model=_PassthroughModel(), student_temperature=0.5, teacher_temperature=2.0
    )
    assert split(build_features(), labels).item() != pytest.approx(shared_value)
    assert split.get_config_dict() == {
        "similarity_fct": "colbert_kd_scores",
        "temperature": 1.0,
        "student_temperature": 0.5,
        "teacher_temperature": 2.0,
        "mini_batch_size": None,
    }


def test_mean_colbert_scores_divide_by_query_tokens() -> None:
    # [FUNCTION DOC]
    # For the matrix, KD and pairwise ColBERT scorers, asserts the mean_ variant multiplied by the number of query tokens (4) equals the plain variant.
    """The mean_colbert_* callables are the length-normalized form of their colbert_* counterparts,
    so a loss can name MeanMaxSim scoring instead of binding a functools.partial."""
    from sentence_transformers.multi_vector_encoder.scoring import (
        colbert_kd_scores,
        colbert_scores,
        colbert_scores_pairwise,
        mean_colbert_kd_scores,
        mean_colbert_scores,
        mean_colbert_scores_pairwise,
    )

    generator = torch.Generator().manual_seed(4)
    queries = torch.nn.functional.normalize(torch.randn(2, 4, 8, generator=generator), dim=-1)
    documents = torch.nn.functional.normalize(torch.randn(2, 2, 6, 8, generator=generator), dim=-1)
    pairwise_documents = torch.nn.functional.normalize(torch.randn(2, 6, 8, generator=generator), dim=-1)

    for plain, mean, args in (
        (colbert_scores, mean_colbert_scores, (queries, documents)),
        (colbert_kd_scores, mean_colbert_kd_scores, (queries, documents)),
        (colbert_scores_pairwise, mean_colbert_scores_pairwise, (queries, pairwise_documents)),
    ):
        assert torch.allclose(mean(*args) * 4, plain(*args), atol=1e-5), plain.__name__
