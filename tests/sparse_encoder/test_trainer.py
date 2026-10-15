# [MODULE DOC]
# Tests for SparseEncoderTrainer: model card update on trainer construction, a short training run (with
# regular or streaming datasets) that changes parameters, and an extensive slow matrix checking how prompts
# (none, a single string, per-dataset, per-column, or nested per-dataset-per-column) are applied to the texts
# the model preprocesses for single datasets versus dataset dicts, with optional transforms, streaming and
# per-dataset losses. Skipped at module level without the training extras.
from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder import losses
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDatasetDict

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture
def dummy_train_eval_datasets_for_trainer() -> tuple[Dataset, Dataset]:
    # [FUNCTION DOC]
    # Fixture returning a 20-row train and 10-row eval in-memory dataset with sentence1, sentence2 and a
    # float score column alternating 0 and 1.
    # Create minimal datasets for trainer tests
    train_data = {
        "sentence1": [f"train_s1_{i}" for i in range(20)],
        "sentence2": [f"train_s2_{i}" for i in range(20)],
        "score": [float(i % 2) for i in range(20)],
    }
    eval_data = {
        "sentence1": [f"eval_s1_{i}" for i in range(10)],
        "sentence2": [f"eval_s2_{i}" for i in range(10)],
        "score": [float(i % 2) for i in range(10)],
    }
    train_dataset = Dataset.from_dict(train_data)
    eval_dataset = Dataset.from_dict(eval_data)
    return train_dataset, eval_dataset


def test_model_card_reuse(splade_bert_tiny_model: SparseEncoder):
    # [FUNCTION DOC]
    # Records the model's initial card text, constructs a trainer with a SpladeLoss, saves the model to a temp
    # directory and asserts the written README differs from the initial card text (or exists when there was none).
    model = splade_bert_tiny_model

    initial_card_text = model._model_card_text

    SparseEncoderTrainer(
        model=model,
        loss=losses.SpladeLoss(
            model=model,
            loss=losses.SparseMultipleNegativesRankingLoss(model=model),
            document_regularizer_weight=3e-5,
            query_regularizer_weight=5e-5,
        ),
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "sparse_model_local"
        model.save_pretrained(str(model_path))

        with open(model_path / "README.md", encoding="utf8") as f:
            trained_model_card_text = f.read()

        if initial_card_text:
            assert trained_model_card_text != initial_card_text
        else:
            assert trained_model_card_text is not None  # Should have created one


@pytest.mark.parametrize("streaming", [False, True])
def test_trainer(
    splade_bert_tiny_model: SparseEncoder,
    dummy_train_eval_datasets_for_trainer: tuple[Dataset, Dataset],
    streaming: bool,
) -> None:
    # [FUNCTION DOC]
    # Parametrized over streaming. Optionally converts the fixture datasets to iterable datasets, snapshots the
    # parameters, trains two steps (batch 2, eval every 2 steps, no reporting) in a temp directory with SpladeLoss,
    # and asserts some parameter changed and that encoding still works afterwards.
    model = splade_bert_tiny_model
    train_dataset, eval_dataset = dummy_train_eval_datasets_for_trainer

    context = nullcontext()
    if streaming:
        train_dataset = train_dataset.to_iterable_dataset()
        eval_dataset = eval_dataset.to_iterable_dataset()

    original_model_params = [p.clone() for p in model.parameters()]

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        args = SparseEncoderTrainingArguments(
            output_dir=str(temp_dir),
            max_steps=2,
            eval_strategy="steps",  # Changed from eval_steps to eval_strategy
            eval_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=1,
            remove_unused_columns=False,  # Important for custom dict datasets
            report_to=["none"],
        )
        with context:  # context is nullcontext unless streaming causes issues not caught here
            trainer = SparseEncoderTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=loss,
            )
            trainer.train()

    if isinstance(context, nullcontext):
        # Check if model parameters have changed after training
        model_changed = False
        for p_orig, p_new in zip(original_model_params, model.parameters()):
            if not torch.equal(p_orig, p_new):
                model_changed = True
                break
        assert model_changed, "Model parameters should have changed after training."

        # Simple check to ensure prediction works after training
        try:
            model.encode(["Test sentence after training."])
        except Exception as e:
            pytest.fail(f"Encoding failed after training: {e}")


@pytest.mark.slow
@pytest.mark.parametrize("train_dict", [False, True])
@pytest.mark.parametrize("eval_dict", [False, True])
@pytest.mark.parametrize("loss_dict", [False, True])
@pytest.mark.parametrize("add_transform", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "prompts",
    [
        None,  # No prompt
        "Prompt: ",  # Single prompt to all columns and all datasets
        {"stsb-1": "Prompt 1: ", "stsb-2": "Prompt 2: "},  # Different prompts for different datasets
        {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "},  # Different prompts for different columns
        {
            "stsb-1": {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "},
            "stsb-2": {"sentence1": "Prompt 3: ", "sentence2": "Prompt 4: "},
        },  # Different prompts for different datasets and columns
    ],
)
def test_trainer_prompts(
    splade_bert_tiny_model: SparseEncoder,
    train_dict: bool,
    eval_dict: bool,
    loss_dict: bool,
    add_transform: bool,
    streaming: bool,
    prompts: dict[str, dict[str, str]] | dict[str, str] | str | None,
):
    # [FUNCTION DOC]
    # Slow; parametrized over train/eval dataset dicts, a per-dataset loss dict, a transform, streaming and prompt
    # configurations. Skips loss dicts unless both train and eval are dicts. Builds two tiny train and eval
    # datasets, wraps model preprocessing to record each text with its prompt prepended, assembles regular or
    # iterable datasets or dataset dicts, optionally applies an uppercase transform (via map for streaming, set
    # transform otherwise), and optionally maps each dataset name to the loss. Expects a ValueError about nested
    # prompts when nested prompts are used without both dicts. Trains briefly while wrapping compute_loss to record
    # collated batch keys. Then asserts "dataset_name" is in the batch keys exactly when a dict dataset is used
    # together with a loss dict or dict prompts, and that the recorded texts equal the expected set of (prompted,
    # possibly uppercased) texts for the configuration. Per-dataset prompts with plain datasets are treated as
    # column names and thus not applied.
    if loss_dict and (not train_dict or not eval_dict):
        pytest.skip(
            "Skipping test because loss_dict=True requires train_dict=True and eval_dict=True; already tested via test_trainer."
        )

    model = splade_bert_tiny_model

    train_dataset_1 = Dataset.from_dict(
        {
            "sentence1": ["train 1 sentence 1a", "train 1 sentence 1b"],
            "sentence2": ["train 1 sentence 2a", "train 1 sentence 2b"],
        }
    )
    train_dataset_2 = Dataset.from_dict(
        {
            "sentence1": ["train 2 sentence 1a", "train 2 sentence 1b"],
            "sentence2": ["train 2 sentence 2a", "train 2 sentence 2b"],
        }
    )
    eval_dataset_1 = Dataset.from_dict(
        {
            "sentence1": ["eval 1 sentence 1a", "eval 1 sentence 1b"],
            "sentence2": ["eval 1 sentence 2a", "eval 1 sentence 2b"],
        }
    )
    eval_dataset_2 = Dataset.from_dict(
        {
            "sentence1": ["eval 2 sentence 1a", "eval 2 sentence 1b"],
            "sentence2": ["eval 2 sentence 2a", "eval 2 sentence 2b"],
        }
    )

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
    )

    tracked_texts = []
    old_preprocess = model.preprocess

    def preprocess_tracker(texts, prompt=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the texts (with the prompt prepended when given) into the tracked list, then calls the
        # original preprocess.
        if prompt:
            tracked_texts.extend([prompt + text for text in texts])
        else:
            tracked_texts.extend(texts)
        return old_preprocess(texts, prompt=prompt, **kwargs)

    model.preprocess = preprocess_tracker

    if train_dict:
        if streaming:
            train_dataset = IterableDatasetDict({"stsb-1": train_dataset_1, "stsb-2": train_dataset_2})
        else:
            train_dataset = DatasetDict({"stsb-1": train_dataset_1, "stsb-2": train_dataset_2})
    else:
        if streaming:
            train_dataset = train_dataset_1.to_iterable_dataset()
        else:
            train_dataset = train_dataset_1

    if eval_dict:
        if streaming:
            eval_dataset = IterableDatasetDict({"stsb-1": eval_dataset_1, "stsb-2": eval_dataset_2})
        else:
            eval_dataset = DatasetDict({"stsb-1": eval_dataset_1, "stsb-2": eval_dataset_2})
    else:
        if streaming:
            eval_dataset = eval_dataset_1.to_iterable_dataset()
        else:
            eval_dataset = eval_dataset_1

    def upper_transform(batch):
        # [FUNCTION DOC]
        # HELPER: Uppercases every text in every column of the batch in place and returns the batch.
        for column_name, column in batch.items():
            batch[column_name] = [text.upper() for text in column]
        return batch

    if add_transform:
        if streaming:
            if train_dict:
                train_dataset = IterableDatasetDict(
                    {
                        dataset_name: dataset.map(upper_transform, batched=True, features=dataset.features)
                        for dataset_name, dataset in train_dataset.items()
                    }
                )
            else:
                train_dataset = train_dataset.map(upper_transform, batched=True, features=train_dataset.features)
            if eval_dict:
                eval_dataset = IterableDatasetDict(
                    {
                        dataset_name: dataset.map(upper_transform, batched=True, features=dataset.features)
                        for dataset_name, dataset in eval_dataset.items()
                    }
                )
            else:
                eval_dataset = eval_dataset.map(upper_transform, batched=True, features=eval_dataset.features)
        else:
            train_dataset.set_transform(upper_transform)
            eval_dataset.set_transform(upper_transform)

    if loss_dict:
        loss = {
            "stsb-1": loss,
            "stsb-2": loss,
        }

    # Variables to more easily track the expected outputs. Uppercased if add_transform is True as we expect
    # the transform to be applied to the data.
    all_train_1_1 = {s.upper() if add_transform else s for s in train_dataset_1["sentence1"]}
    all_train_1_2 = {s.upper() if add_transform else s for s in train_dataset_1["sentence2"]}
    all_train_2_1 = {s.upper() if add_transform else s for s in train_dataset_2["sentence1"]}
    all_train_2_2 = {s.upper() if add_transform else s for s in train_dataset_2["sentence2"]}
    all_eval_1_1 = {s.upper() if add_transform else s for s in eval_dataset_1["sentence1"]}
    all_eval_1_2 = {s.upper() if add_transform else s for s in eval_dataset_1["sentence2"]}
    all_eval_2_1 = {s.upper() if add_transform else s for s in eval_dataset_2["sentence1"]}
    all_eval_2_2 = {s.upper() if add_transform else s for s in eval_dataset_2["sentence2"]}
    all_train_1 = all_train_1_1 | all_train_1_2
    all_train_2 = all_train_2_1 | all_train_2_2
    all_eval_1 = all_eval_1_1 | all_eval_1_2
    all_eval_2 = all_eval_2_1 | all_eval_2_2
    all_train = all_train_1 | all_train_2
    all_eval = all_eval_1 | all_eval_2

    if prompts == {
        "stsb-1": {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "},
        "stsb-2": {"sentence1": "Prompt 3: ", "sentence2": "Prompt 4: "},
    } and (train_dict, eval_dict) != (True, True):
        context = pytest.raises(
            ValueError,
            match="The prompts provided to the trainer are a nested dictionary. In this setting, the first "
            "level of the dictionary should map to dataset names and the second level to column names. "
            "However, as the provided dataset is a not a DatasetDict, no dataset names can be inferred. "
            "The keys to the provided prompts dictionary are .*",
        )
    else:
        context = nullcontext()

    with tempfile.TemporaryDirectory() as temp_dir:
        args = SparseEncoderTrainingArguments(
            output_dir=str(temp_dir),
            prompts=prompts,
            max_steps=4 if train_dict else 2,
            eval_steps=4 if train_dict else 2,
            eval_strategy="steps",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            report_to=["none"],
        )
        trainer = SparseEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )

        tracked_texts.clear()

        datacollator_keys = set()
        old_compute_loss = trainer.compute_loss

        def compute_loss_tracker(model, inputs, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Adds the batch's keys to the tracked key set and delegates to the original compute_loss.
            datacollator_keys.update(set(inputs.keys()))
            return old_compute_loss(model, inputs, **kwargs)

        trainer.compute_loss = compute_loss_tracker
        with context:
            trainer.train()

        if not isinstance(context, nullcontext):
            return

    # prompt_length keys may appear in the batch when prompts are provided (Transformer.preprocess always
    # computes them), but SpladePooling simply ignores them. Only Pooling uses them when include_prompt=False.

    # We only need the dataset_name if the loss requires it, or the prompts are a nested dictionary
    if (train_dict or eval_dict) and (loss_dict or (isinstance(prompts, dict))):
        assert "dataset_name" in datacollator_keys
    else:
        assert "dataset_name" not in datacollator_keys

    if prompts is None:
        if (train_dict, eval_dict) == (False, False):
            expected = all_train_1 | all_eval_1
        elif (train_dict, eval_dict) == (True, False):
            expected = all_train | all_eval_1
        elif (train_dict, eval_dict) == (False, True):
            expected = all_train_1 | all_eval
        elif (train_dict, eval_dict) == (True, True):
            expected = all_train | all_eval

    elif prompts == "Prompt: ":
        if (train_dict, eval_dict) == (False, False):
            expected = {prompts + sample for sample in all_train_1} | {prompts + sample for sample in all_eval_1}
        elif (train_dict, eval_dict) == (True, False):
            expected = {prompts + sample for sample in all_train} | {prompts + sample for sample in all_eval_1}
        elif (train_dict, eval_dict) == (False, True):
            expected = {prompts + sample for sample in all_train_1} | {prompts + sample for sample in all_eval}
        elif (train_dict, eval_dict) == (True, True):
            expected = {prompts + sample for sample in all_train} | {prompts + sample for sample in all_eval}

    elif prompts == {"stsb-1": "Prompt 1: ", "stsb-2": "Prompt 2: "}:
        # If we don't have dataset dictionaries, the prompts will be seen as column names
        if (train_dict, eval_dict) == (False, False):
            expected = all_train_1 | all_eval_1
        elif (train_dict, eval_dict) == (True, False):
            expected = (
                {prompts["stsb-1"] + sample for sample in all_train_1}
                | {prompts["stsb-2"] + sample for sample in all_train_2}
                | all_eval_1
            )
        elif (train_dict, eval_dict) == (False, True):
            expected = (
                all_train_1
                | {prompts["stsb-1"] + sample for sample in all_eval_1}
                | {prompts["stsb-2"] + sample for sample in all_eval_2}
            )
        elif (train_dict, eval_dict) == (True, True):
            expected = (
                {prompts["stsb-1"] + sample for sample in all_train_1}
                | {prompts["stsb-2"] + sample for sample in all_train_2}
                | {prompts["stsb-1"] + sample for sample in all_eval_1}
                | {prompts["stsb-2"] + sample for sample in all_eval_2}
            )

    elif prompts == {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "}:
        if (train_dict, eval_dict) == (False, False):
            expected = (
                {prompts["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["sentence2"] + sample for sample in all_eval_1_2}
            )
        elif (train_dict, eval_dict) == (True, False):
            expected = (
                {prompts["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["sentence1"] + sample for sample in all_train_2_1}
                | {prompts["sentence2"] + sample for sample in all_train_2_2}
                | {prompts["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["sentence2"] + sample for sample in all_eval_1_2}
            )
        elif (train_dict, eval_dict) == (False, True):
            expected = (
                {prompts["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["sentence2"] + sample for sample in all_eval_1_2}
                | {prompts["sentence1"] + sample for sample in all_eval_2_1}
                | {prompts["sentence2"] + sample for sample in all_eval_2_2}
            )
        elif (train_dict, eval_dict) == (True, True):
            expected = (
                {prompts["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["sentence1"] + sample for sample in all_train_2_1}
                | {prompts["sentence2"] + sample for sample in all_train_2_2}
                | {prompts["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["sentence2"] + sample for sample in all_eval_1_2}
                | {prompts["sentence1"] + sample for sample in all_eval_2_1}
                | {prompts["sentence2"] + sample for sample in all_eval_2_2}
            )

    elif prompts == {
        "stsb-1": {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "},
        "stsb-2": {"sentence1": "Prompt 3: ", "sentence2": "Prompt 4: "},
    }:
        # All other cases are tested above with the ValueError context
        if (train_dict, eval_dict) == (True, True):
            expected = (
                {prompts["stsb-1"]["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["stsb-1"]["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["stsb-2"]["sentence1"] + sample for sample in all_train_2_1}
                | {prompts["stsb-2"]["sentence2"] + sample for sample in all_train_2_2}
                | {prompts["stsb-1"]["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["stsb-1"]["sentence2"] + sample for sample in all_eval_1_2}
                | {prompts["stsb-2"]["sentence1"] + sample for sample in all_eval_2_1}
                | {prompts["stsb-2"]["sentence2"] + sample for sample in all_eval_2_2}
            )

    assert set(tracked_texts) == expected
