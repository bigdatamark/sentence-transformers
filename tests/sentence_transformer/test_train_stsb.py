# [MODULE DOC]
# Training tests for SentenceTransformer: fine-tunes a DistilBERT model on STS-Benchmark with
# CosineSimilarityLoss or on AllNLI with SoftmaxLoss, then checks the STSB test Spearman score. Skipped at
# module level when training extras are missing.
# KNOWN ISSUE: The module docstring says it tests pretrained model scores, but this module actually trains
# models and then evaluates them (it was copied from the pretrained STSB test module).
"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator

import pytest
from datasets import Dataset, DatasetDict, load_dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss, SoftmaxLoss
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture()
def nli_resource() -> Generator[Dataset, None, None]:
    # [FUNCTION DOC]
    # Fixture: downloads the "pair-class" train split of the AllNLI dataset from the Hub (network), keeps at most
    # the first 10000 rows, renames premise/hypothesis to sentence1/sentence2 and yields the dataset.
    max_train_samples = 10000
    nli_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
    nli_dataset = nli_dataset.select(range(min(max_train_samples, len(nli_dataset))))
    nli_dataset = nli_dataset.rename_columns({"premise": "sentence1", "hypothesis": "sentence2"})
    yield nli_dataset


def evaluate_stsb_test(model: SentenceTransformer, expected_score: float, test_dataset: Dataset) -> None:
    # [FUNCTION DOC]
    # HELPER: Evaluates the model with an EmbeddingSimilarityEvaluator named "sts-test" on the dataset's
    # sentence1/sentence2/score columns, prints the primary metric times 100 and asserts it exceeds the expected
    # score or is within 0.1 of it.
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        scores=test_dataset["score"],
        name="sts-test",
    )
    scores = model.evaluate(evaluator)
    score = scores[evaluator.primary_metric] * 100
    print(f"STS-Test Performance: {score:.2f} vs. exp: {expected_score:.2f}")
    assert score > expected_score or abs(score - expected_score) < 0.1


@pytest.mark.slow
@pytest.mark.skipif(
    not is_training_available(), reason='Sentence Transformers was not installed with the `["train"]` extra.'
)
def test_train_stsb_slow(distilbert_base_uncased_model: SentenceTransformer, stsb_dataset_dict: DatasetDict) -> None:
    # [FUNCTION DOC]
    # Slow: trains for one epoch on the full STSB train split with CosineSimilarityLoss (batch 16, warmup ratio
    # 0.1, output in a temporary directory) and expects a test score of about 80.
    model = distilbert_base_uncased_model
    train_dataset = stsb_dataset_dict["train"]
    test_dataset = stsb_dataset_dict["test"]
    train_loss = CosineSimilarityLoss(model=model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 80.0, test_dataset)


@pytest.mark.skipif("CI" in os.environ, reason="This test is too slow for the CI (~8 minutes)")
@pytest.mark.skipif(
    not is_training_available(), reason='Sentence Transformers was not installed with the `["train"]` extra.'
)
def test_train_stsb(distilbert_base_uncased_model: SentenceTransformer, stsb_dataset_dict: DatasetDict) -> None:
    # [FUNCTION DOC]
    # Skipped in CI: trains on the first 100 STSB training rows and expects a test score of about 60.
    model = distilbert_base_uncased_model
    train_dataset = stsb_dataset_dict["train"].select(range(100))
    test_dataset = stsb_dataset_dict["test"]
    train_loss = CosineSimilarityLoss(model=model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 60.0, test_dataset)


@pytest.mark.slow
@pytest.mark.skipif(
    not is_training_available(), reason='Sentence Transformers was not installed with the `["train"]` extra.'
)
def test_train_nli_slow(
    distilbert_base_uncased_model: SentenceTransformer, nli_resource: Dataset, stsb_dataset_dict: DatasetDict
) -> None:
    # [FUNCTION DOC]
    # Slow: trains on the NLI fixture with a 3-label SoftmaxLoss sized to the model's embedding dimension and
    # expects an STSB test score of about 50.
    model = distilbert_base_uncased_model
    test_dataset = stsb_dataset_dict["test"]
    train_loss = SoftmaxLoss(
        model=model,
        embedding_dimension=model.get_embedding_dimension(),
        num_labels=3,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=nli_resource,
            loss=train_loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 50.0, test_dataset)


@pytest.mark.skipif("CI" in os.environ, reason="This test is too slow for the CI (~25 minutes)")
@pytest.mark.skipif(
    not is_training_available(), reason='Sentence Transformers was not installed with the `["train"]` extra.'
)
def test_train_nli(
    distilbert_base_uncased_model: SentenceTransformer, nli_resource: Dataset, stsb_dataset_dict: DatasetDict
) -> None:
    # [FUNCTION DOC]
    # Skipped in CI: trains with SoftmaxLoss on the first 100 NLI rows and expects an STSB test score of about 50.
    model = distilbert_base_uncased_model
    test_dataset = stsb_dataset_dict["test"]
    train_dataset = nli_resource.select(range(100))
    train_loss = SoftmaxLoss(
        model=model,
        embedding_dimension=model.get_embedding_dimension(),
        num_labels=3,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 50.0, test_dataset)
