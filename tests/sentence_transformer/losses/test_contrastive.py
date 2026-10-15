# [MODULE DOC]
# Tests the binary-label sanity warning of ContrastiveLoss and OnlineContrastiveLoss. The losses must emit a UserWarning when
# labels are not 0/1, only once (checked on the first forward only), and stay silent for binary labels. Uses a fake model that
# echoes precomputed embeddings. The two module constants are regex patterns matching each loss's warning message.
from __future__ import annotations

import warnings

import pytest
import torch

from sentence_transformers.sentence_transformer.losses.contrastive import ContrastiveLoss
from sentence_transformers.sentence_transformer.losses.online_contrastive import OnlineContrastiveLoss

CONTRASTIVE_WARNING = "ContrastiveLoss expects binary labels.*"
ONLINE_WARNING = "OnlineContrastiveLoss expects binary labels.*"


class _EchoModel:
    # [CLASS DOC]
    # HELPER: Fake model whose call returns the input feature dict's "sentence_embedding" tensor unchanged under the same key,
    # so the losses consume hand-crafted embeddings.
    def __call__(self, sentence_feature: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Returns a new dict containing only the "sentence_embedding" entry of the given features.
        return {"sentence_embedding": sentence_feature["sentence_embedding"]}


def _features() -> list[dict[str, torch.Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Returns two feature dicts, each with a 2x2 "sentence_embedding" tensor, representing two pairs of 2-D embeddings.
    return [
        {"sentence_embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
        {"sentence_embedding": torch.tensor([[0.9, 0.1], [1.0, 0.0]])},
    ]


def test_contrastive_loss_warns_once_for_non_binary_labels() -> None:
    # [FUNCTION DOC]
    # With labels 1.0 and 0.25, asserts the first ContrastiveLoss call warns about binary labels, and a second call (with that warning
    # forced to "always" and everything else ignored) records no warnings. Asserts both loss values are finite.
    loss = ContrastiveLoss(_EchoModel())
    labels = torch.tensor([1.0, 0.25])

    with pytest.warns(UserWarning, match="ContrastiveLoss expects binary labels"):
        first_value = loss(_features(), labels)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=CONTRASTIVE_WARNING, category=UserWarning)
        second_value = loss(_features(), labels)

    assert not caught
    assert torch.isfinite(first_value)
    assert torch.isfinite(second_value)


def test_contrastive_loss_does_not_warn_for_binary_labels() -> None:
    # [FUNCTION DOC]
    # With binary labels, asserts ContrastiveLoss emits no binary-label warning and returns a finite value.
    loss = ContrastiveLoss(_EchoModel())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=CONTRASTIVE_WARNING, category=UserWarning)
        value = loss(_features(), torch.tensor([1.0, 0.0]))

    assert not caught
    assert torch.isfinite(value)


def test_contrastive_loss_only_checks_the_first_batch() -> None:
    # [FUNCTION DOC]
    # Asserts that the label check runs only on the first forward: a first call with binary labels then a second with non-binary
    # labels produces no warning at all.
    # The label check runs once, on the first forward, to avoid a per-step device sync. Non-binary
    # labels that first appear in a later batch are therefore not flagged (labels are fixed per run).
    loss = ContrastiveLoss(_EchoModel())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=CONTRASTIVE_WARNING, category=UserWarning)
        loss(_features(), torch.tensor([1.0, 0.0]))
        loss(_features(), torch.tensor([1.0, 0.25]))

    assert not caught


def test_online_contrastive_loss_warns_once_for_non_binary_labels() -> None:
    # [FUNCTION DOC]
    # Same as the ContrastiveLoss warn-once test but for OnlineContrastiveLoss: warns on the first non-binary call, silent on the
    # second, both values finite.
    loss = OnlineContrastiveLoss(_EchoModel())
    labels = torch.tensor([1.0, 0.25])

    with pytest.warns(UserWarning, match="OnlineContrastiveLoss expects binary labels"):
        first_value = loss(_features(), labels)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=ONLINE_WARNING, category=UserWarning)
        second_value = loss(_features(), labels)

    assert not caught
    assert torch.isfinite(first_value)
    assert torch.isfinite(second_value)


def test_online_contrastive_loss_does_not_warn_for_binary_labels() -> None:
    # [FUNCTION DOC]
    # With binary labels, asserts OnlineContrastiveLoss emits no binary-label warning and returns a finite value.
    loss = OnlineContrastiveLoss(_EchoModel())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", message=ONLINE_WARNING, category=UserWarning)
        value = loss(_features(), torch.tensor([1.0, 0.0]))

    assert not caught
    assert torch.isfinite(value)
