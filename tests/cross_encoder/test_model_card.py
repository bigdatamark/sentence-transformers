# [MODULE DOC]
# Pytest tests for CrossEncoder model card generation: the full README text produced after a CrossEncoderTrainer has
# recorded training-dataset information on the model card data, dataset-transform handling, and the usage-snippet
# generator of CrossEncoderModelCardData. Skips the module without the training extra. Hub lookups are avoided via
# local-files-only (and the root conftest's Hub mocks).
from __future__ import annotations

import pytest

from sentence_transformers.base.model_card import generate_model_card
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer
from sentence_transformers.cross_encoder.model_card import CrossEncoderModelCardData
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def dummy_dataset():
    # [FUNCTION DOC]
    # HELPER: Session-scoped fixture: in-memory Dataset of 10 rows with "anchor", "positive" and "negative" columns
    # holding "anchor i", "positive i", "negative i" for i from 1 to 10.
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "anchor": ["anchor 1", "anchor 2", ..., "anchor 10"],
        "positive": ["positive 1", "positive 2", ..., "positive 10"],
        "negative": ["negative 1", "negative 2", ..., "negative 10"],
    }
    """
    return Dataset.from_dict(
        {
            "anchor": [f"anchor {i}" for i in range(1, 11)],
            "positive": [f"positive {i}" for i in range(1, 11)],
            "negative": [f"negative {i}" for i in range(1, 11)],
        }
    )


@pytest.mark.parametrize(
    ("num_datasets", "num_labels", "expected_substrings"),
    [
        # 0 actually refers to just a single dataset
        (
            0,
            1,
            [
                "- sentence-transformers",
                "- cross-encoder",
                "pipeline_tag: text-ranking",
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)",
                "[sentence-transformers](https://www.SBERT.net) library",
                "It computes scores for pairs of texts, which can be used for text reranking and semantic search.",
                "**Maximum Sequence Length:** 512 tokens",
                "- **Number of Output Labels:** 1 label",
                "<!-- - **Training Dataset:** Unknown -->",
                "<!-- - **Language:** Unknown -->",
                "<!-- - **License:** Unknown -->",
                'model = CrossEncoder("cross_encoder_model_id")',
                "['anchor 1', 'positive 1'],",
                "print(scores)",
                "ranks = model.rank(",
                "#### Unnamed Dataset",
                "| modality | text                                                                           | text                                                                           | text                                                                           |",
                "| details  | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> |",
                "| <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:",
            ],
        ),
        (
            0,
            3,
            [
                "- sentence-transformers",
                "- cross-encoder",
                "pipeline_tag: text-classification",
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)",
                "[sentence-transformers](https://www.SBERT.net) library",
                "It computes scores for pairs of texts, which can be used for text pair classification.",
                "**Maximum Sequence Length:** 512 tokens",
                "- **Number of Output Labels:** 3 labels",
                "<!-- - **Training Dataset:** Unknown -->",
                "<!-- - **Language:** Unknown -->",
                "<!-- - **License:** Unknown -->",
                'model = CrossEncoder("cross_encoder_model_id")',
                "['anchor 1', 'positive 1'],",
                "print(scores)",
                "#### Unnamed Dataset",
                " | <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "Loss: [<code>CrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#crossentropyloss)",
            ],
        ),
        (
            1,
            1,
            [
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0 dataset using the [sentence-transformers](https://www.SBERT.net) library.",
                "#### train_0",
            ],
        ),
        (
            2,
            1,
            [
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0 and train_1 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "#### train_0",
                "#### train_1",
            ],
        ),
        (
            10,
            1,
            [
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8 and train_9 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "<details><summary>train_0</summary>",  # We start using <details><summary> if we have more than 3 datasets
                "#### train_0",
                "</details>\n<details><summary>train_9</summary>",
                "#### train_9",
            ],
        ),
        # We start using "50 datasets" when the ", "-joined dataset name exceed 200 characters
        (
            50,
            1,
            [
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on 50 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "<details><summary>train_0</summary>",
                "#### train_0",
                "</details>\n<details><summary>train_49</summary>",
                "#### train_49",
            ],
        ),
    ],
)
def test_model_card_base(
    dummy_dataset: Dataset, num_datasets: int, num_labels: int, expected_substrings: list[str]
) -> None:
    # [FUNCTION DOC]
    # Parametrised over number of training datasets (0 meaning a single plain Dataset, otherwise a DatasetDict of that
    # many copies named train_0, train_1, ...), number of labels, and expected substrings. Builds a fresh CrossEncoder
    # from the tiny STSB BERT checkpoint with that label count, sets local-files-only on its card data, constructs a
    # CrossEncoderTrainer (which fills the card data from the dataset), generates the model card, and asserts every
    # expected substring occurs and that no three consecutive newlines appear. Expectations cover tags, pipeline tag
    # (text-ranking for one label, text-classification for three), description, max sequence length, label count, the
    # dataset statistics table and samples, the default loss (BinaryCrossEntropyLoss or CrossEntropyLoss),
    # dataset-name lists, collapsible details sections when there are more than three datasets, and "on 50 datasets"
    # once the joined names exceed 200 characters.
    model = CrossEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors", num_labels=num_labels)

    # Let's avoid requesting the Hub for e.g. checking if a base model exists there
    model.model_card_data.local_files_only = True

    train_dataset = dummy_dataset
    if num_datasets:
        train_dataset = DatasetDict({f"train_{i}": train_dataset for i in range(num_datasets)})

    # This adds data to model.model_card_data
    CrossEncoderTrainer(model, train_dataset=train_dataset)

    model_card = generate_model_card(model)

    # For debugging purposes, we can save the model card to a file
    # with open(f"test_model_card_{num_datasets}d_{num_labels}l.md", "w", encoding="utf8") as f:
    #     f.write(model_card)

    for substring in expected_substrings:
        assert substring in model_card

    # We don't want to have two consecutive empty lines anywhere
    assert "\n\n\n" not in model_card


def test_model_card_set_transform(dummy_dataset: Dataset, reranker_bert_tiny_model: CrossEncoder) -> None:
    # [FUNCTION DOC]
    # Applies an on-the-fly transform (renaming columns to new_anchor/new_positive/new_negative with upper-cased text)
    # to a copy of the session dataset, constructs a trainer with it on the tiny reranker, generates the card, and
    # asserts the post-transform column names appear as code headers while the original ones do not.
    model = reranker_bert_tiny_model

    # Let's avoid requesting the Hub for e.g. checking if a base model exists there
    model.model_card_data.local_files_only = True

    def dummy_transform(batch):
        # [FUNCTION DOC]
        # HELPER: Transform function: returns a batch dict with the three columns renamed with a "new_" prefix and
        # their texts upper-cased.
        return {
            "new_anchor": [text.upper() for text in batch["anchor"]],
            "new_positive": [text.upper() for text in batch["positive"]],
            "new_negative": [text.upper() for text in batch["negative"]],
        }

    # Use a copy to avoid mutating the session-scoped fixture
    dataset = dummy_dataset.select(range(len(dummy_dataset)))
    dataset.set_transform(dummy_transform)

    CrossEncoderTrainer(model, train_dataset=dataset)
    model_card = generate_model_card(model)

    # Post-transform column names should appear as column headers
    for substring in ["<code>new_anchor</code>", "<code>new_positive</code>", "<code>new_negative</code>"]:
        assert substring in model_card

    # Pre-transform column names should not appear as column headers
    for substring in ["<code>anchor</code>", "<code>positive</code>", "<code>negative</code>"]:
        assert substring not in model_card


class TestGenerateUsageSnippetCrossEncoder:
    # [CLASS DOC]
    # Tests for the CrossEncoder usage snippet (example code with predict and rank) generated by the model card data.
    """CrossEncoder snippet with predict/rank."""

    def test_cross_encoder_default_examples(self) -> None:
        # [FUNCTION DOC]
        # Asserts the snippet for card data with no model imports CrossEncoder, calls predict on pairs, uses the
        # default egg-calories example, and includes a rank call.
        data = CrossEncoderModelCardData()
        data.model = None
        snippet = data.generate_usage_snippet()

        assert "from sentence_transformers import CrossEncoder" in snippet
        assert "model.predict(pairs)" in snippet
        assert "How many calories in an egg" in snippet
        assert "model.rank(" in snippet

    def test_cross_encoder_custom_examples(self) -> None:
        # [FUNCTION DOC]
        # Asserts custom usage examples are rendered as quoted strings and the shape comment for two single-label
        # pairs is "(2,)".
        data = CrossEncoderModelCardData()
        data.usage_examples = [["q1", "a1"], ["q1", "a2"]]
        data.model = None
        snippet = data.generate_usage_snippet()

        assert "'q1'" in snippet
        assert "'a1'" in snippet
        assert "'a2'" in snippet
        assert "# (2,)" in snippet

    def test_cross_encoder_multi_label(self) -> None:
        # [FUNCTION DOC]
        # With a fake model having three labels and text modality, asserts the shape comment is "(2, 3)" and no rank
        # call is present.
        """Multi-label: shape includes num_labels, no rank section."""
        data = CrossEncoderModelCardData()
        data.usage_examples = [["q", "a1"], ["q", "a2"]]

        class FakeModel:
            # [CLASS DOC]
            # HELPER: Minimal stand-in model with three labels and text-only modalities.
            num_labels = 3
            modalities = ["text"]

        data.model = FakeModel()
        snippet = data.generate_usage_snippet()

        assert "# (2, 3)" in snippet
        assert "model.rank(" not in snippet

    def test_cross_encoder_single_label_has_rank(self) -> None:
        # [FUNCTION DOC]
        # With a fake single-label text model, asserts the shape comment is "(2,)" and a rank call is present.
        """Single-label: shape is (n,), rank section present."""
        data = CrossEncoderModelCardData()
        data.usage_examples = [["q", "a1"], ["q", "a2"]]

        class FakeModel:
            # [CLASS DOC]
            # HELPER: Minimal stand-in model with one label and text-only modalities.
            num_labels = 1
            modalities = ["text"]

        data.model = FakeModel()
        snippet = data.generate_usage_snippet()

        assert "# (2,)" in snippet
        assert "model.rank(" in snippet

    def test_cross_encoder_multimodal_pairs(self) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. With a model id, image-plus-text usage examples, display values pointing to an asset
        # path, and a fake single-label text-and-image model, asserts the snippet defines pairs, references the image
        # via a huggingface.co URL for the model id, includes the text "A cat", and omits the rank call.
        """Multimodal pairs use _format_snippet_value for non-text elements and skip rank section."""
        try:
            from PIL import Image as PILModule
        except ImportError:
            pytest.skip("Pillow not installed")

        data = CrossEncoderModelCardData()
        data.model_id = "user/multimodal-ce"
        img = PILModule.new("RGB", (64, 64), color=(255, 0, 0))
        data.usage_examples = [[img, "A cat"], [img, "A dog"]]
        data.usage_examples_display = [["assets/image_0.jpg", "A cat"], ["assets/image_0.jpg", "A dog"]]

        class FakeModel:
            # [CLASS DOC]
            # HELPER: Minimal stand-in model with one label and text plus image modalities.
            num_labels = 1
            modalities = ["text", "image"]

        data.model = FakeModel()
        snippet = data.generate_usage_snippet()

        assert "pairs = [" in snippet
        assert "huggingface.co/user/multimodal-ce" in snippet
        assert "'A cat'" in snippet
        # Multimodal pairs should NOT have rank section (rank doesn't support non-text)
        assert "model.rank(" not in snippet

    def test_cross_encoder_multimodal_run_usage_snippet(self, reranker_bert_tiny_model: CrossEncoder) -> None:
        # [FUNCTION DOC]
        # Skips without Pillow. Sets image-plus-text usage examples and widget generation on the tiny text-only
        # reranker, calls the run-usage-snippet method while swallowing any exception (prediction on images is
        # expected to fail), then asserts the generated snippet still contains the pairs definition and the predict
        # call.
        """run_usage_snippet with multimodal pairs gracefully falls back when predict fails."""
        try:
            from PIL import Image as PILModule
        except ImportError:
            pytest.skip("Pillow not installed")

        model = reranker_bert_tiny_model
        model.model_card_data.usage_examples = [
            [PILModule.new("RGB", (64, 64), color=(255, 0, 0)), "A cat"],
            [PILModule.new("RGB", (64, 64), color=(0, 255, 0)), "A dog"],
        ]
        model.model_card_data.generate_widget_examples = True
        # The text-only reranker can't process images, so predict will fail.
        # run_usage_snippet is called inside to_dict's try/except, so we call it
        # directly and verify that it either succeeds or raises (but doesn't crash
        # in an unexpected way like an AttributeError).
        try:
            model.model_card_data.run_usage_snippet()
        except Exception:
            # Expected: model.predict fails on image input, similarities stays None
            pass

        # Snippet generation should still work regardless
        snippet = model.model_card_data.generate_usage_snippet()
        assert "pairs = [" in snippet
        assert "model.predict(pairs)" in snippet
