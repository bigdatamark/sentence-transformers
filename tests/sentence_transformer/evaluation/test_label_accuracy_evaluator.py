# [MODULE DOC]
# Tests for LabelAccuracyEvaluator, which measures classification accuracy of a SoftmaxLoss head on top of a SentenceTransformer.
# Uses the all-nli "pair-class" dataset from the Hugging Face Hub (network access) and writes evaluator output to a temp directory.
# KNOWN ISSUE: The module docstring says it tests BinaryClassificationEvaluator, but the module actually tests LabelAccuracyEvaluator.
"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import load_dataset
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import LabelAccuracyEvaluator
from sentence_transformers.sentence_transformer.losses import SoftmaxLoss
from sentence_transformers.sentence_transformer.readers import InputExample


@pytest.mark.skip(reason="This test is rather slow, and the LabelAccuracyEvaluator is not commonly used.")
def test_LabelAccuracyEvaluator(paraphrase_distilroberta_base_v1_model: SentenceTransformer, tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Skipped by default (slow). Loads the first 100 rows of the all-nli pair-class train split, wraps each row as an InputExample
    # with the two sentences and the integer label, builds a 3-label SoftmaxLoss using the model's embedding dimension, and runs
    # LabelAccuracyEvaluator over a non-shuffled DataLoader with batch size 16, writing results to tmp_path.
    # Asserts the returned metrics contain "accuracy" and that it exceeds 0.2 (better than a degenerate untrained head).
    # Inputs: paraphrase_distilroberta_base_v1_model fixture and pytest's tmp_path.
    """Tests that the LabelAccuracyEvaluator can be loaded correctly"""
    model = paraphrase_distilroberta_base_v1_model

    max_dev_samples = 100
    nli_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train").select(
        range(max_dev_samples)
    )

    dev_samples = []
    for row in nli_dataset:
        label_id = int(row["label"])

        dev_samples.append(
            InputExample(
                texts=[row["sentence1"], row["sentence2"]],
                label=label_id,
            )
        )

    train_loss = SoftmaxLoss(
        model=model,
        embedding_dimension=model.get_embedding_dimension(),
        num_labels=3,
    )

    dev_dataloader = DataLoader(dev_samples, shuffle=False, batch_size=16)
    evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)
    metrics = evaluator(model, output_path=str(tmp_path))
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.2
