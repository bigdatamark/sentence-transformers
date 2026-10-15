# [MODULE DOC]
# Pytest unit tests for BaseDataCollator, the trainer data collator that turns a list of row dicts into a batch dict.
# Behaviour exercised: each non-label column's values are passed to a preprocess function and its output keys are
# prefixed with the column name and an underscore; "label" or "score" columns become a "label" tensor; prompts and
# router mappings (column to task) may be flat or nested per dataset name; per-column max length may be a scalar or a
# per-task dict; warnings are logged for suspicious column orders and ID-like columns. Tests use a stub preprocess
# function, so no model is loaded.
from __future__ import annotations

import logging

import pytest
import torch

from sentence_transformers.base.data_collator import BaseDataCollator


def make_collator(**kwargs) -> BaseDataCollator:
    # [FUNCTION DOC]
    # HELPER: Creates a BaseDataCollator whose default preprocess function returns, for n texts, input ids and
    # attention mask tensors of ones with shape n by 3 (long dtype). Keyword arguments override or extend the
    # constructor arguments. Returns the collator.
    """Create a BaseDataCollator with a simple preprocess_fn that returns fixed tensors."""

    def preprocess_fn(texts, prompt=None, task=None):
        # [FUNCTION DOC]
        # HELPER: Stub preprocess function accepting texts plus optional prompt and task (no other keyword arguments);
        # returns the fixed ones tensors described above.
        return {
            "input_ids": torch.ones(len(texts), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }

    defaults = {"preprocess_fn": preprocess_fn}
    defaults.update(kwargs)
    return BaseDataCollator(**defaults)


class TestBasicCollation:
    # [CLASS DOC]
    # Tests the basic batch layout produced by the collator.
    def test_output_keys_are_prefixed(self):
        # [FUNCTION DOC]
        # Asserts that collating one row with two text columns yields prefixed input-ids and attention-mask keys for
        # both columns.
        collator = make_collator()
        features = [{"sentence1": "hello", "sentence2": "world"}]
        batch = collator(features)
        assert "sentence1_input_ids" in batch
        assert "sentence1_attention_mask" in batch
        assert "sentence2_input_ids" in batch
        assert "sentence2_attention_mask" in batch

    def test_empty_features(self):
        # [FUNCTION DOC]
        # Asserts collating an empty feature list returns an empty dict.
        collator = make_collator()
        assert collator([]) == {}


class TestLabelExtraction:
    # [CLASS DOC]
    # Tests extraction of the label column.
    def test_label_column_extracted(self):
        # [FUNCTION DOC]
        # Asserts a "label" column becomes a float tensor of the row labels under "label" and is not passed through
        # preprocessing.
        collator = make_collator()
        features = [
            {"sentence1": "a", "label": 1.0},
            {"sentence1": "b", "label": 0.0},
        ]
        batch = collator(features)
        assert "label" in batch
        assert torch.equal(batch["label"], torch.tensor([1.0, 0.0]))
        # label should not be passed to preprocess_fn (no "label_input_ids")
        assert "label_input_ids" not in batch

    def test_score_label_column(self):
        # [FUNCTION DOC]
        # Asserts a "score" column is also treated as the label and returned under "label".
        collator = make_collator()
        features = [{"sentence1": "a", "score": 0.5}]
        batch = collator(features)
        assert "label" in batch
        assert torch.equal(batch["label"], torch.tensor([0.5]))


class TestResolveRouterMapping:
    # [CLASS DOC]
    # Tests the private router-mapping resolution, which picks the column-to-task mapping for a batch.
    def test_flat_mapping_returned_as_is(self):
        # [FUNCTION DOC]
        # Asserts a flat mapping is returned unchanged.
        mapping = {"col1": "task_a"}
        collator = make_collator(router_mapping=mapping)
        result = collator._resolve_router_mapping({})
        assert result == mapping

    def test_nested_mapping_resolves_by_dataset_name(self):
        # [FUNCTION DOC]
        # Asserts a nested mapping keyed by dataset name returns the inner mapping for the batch's dataset name.
        mapping = {"ds1": {"col1": "task_a"}, "ds2": {"col1": "task_b"}}
        collator = make_collator(router_mapping=mapping)
        result = collator._resolve_router_mapping({"dataset_name": "ds1"})
        assert result == {"col1": "task_a"}

    def test_nested_mapping_missing_dataset_returns_empty(self):
        # [FUNCTION DOC]
        # Asserts a nested mapping returns an empty dict when the dataset name is not a key.
        mapping = {"ds1": {"col1": "task_a"}}
        collator = make_collator(router_mapping=mapping)
        result = collator._resolve_router_mapping({"dataset_name": "unknown"})
        assert result == {}


class TestResolvePrompts:
    # [CLASS DOC]
    # Tests the private prompt resolution for a batch.
    def test_string_prompt(self):
        # [FUNCTION DOC]
        # Asserts a string prompt is returned as is.
        collator = make_collator(prompts="Search: ")
        assert collator._resolve_prompts({}) == "Search: "

    def test_flat_dict_prompt(self):
        # [FUNCTION DOC]
        # Asserts a flat column-to-prompt dict is returned as is.
        prompts = {"query": "Search: ", "document": ""}
        collator = make_collator(prompts=prompts)
        assert collator._resolve_prompts({}) == prompts

    def test_nested_dict_resolves_by_dataset_name(self):
        # [FUNCTION DOC]
        # Asserts a nested dict keyed by dataset name returns the inner dict for the batch's dataset.
        prompts = {"ds1": {"query": "Q: "}, "ds2": {"query": "S: "}}
        collator = make_collator(prompts=prompts)
        assert collator._resolve_prompts({"dataset_name": "ds1"}) == {"query": "Q: "}

    def test_nested_dict_without_dataset_name_raises(self):
        # [FUNCTION DOC]
        # Asserts a nested prompt dict raises ValueError mentioning "nested dictionary" when the batch has no dataset
        # name.
        prompts = {"ds1": {"query": "Q: "}}
        collator = make_collator(prompts=prompts)
        with pytest.raises(ValueError, match="nested dictionary"):
            collator._resolve_prompts({})

    def test_empty_dict_returns_empty(self):
        # [FUNCTION DOC]
        # Asserts an empty prompt dict resolves to a falsy value.
        collator = make_collator(prompts={})
        result = collator._resolve_prompts({})
        assert not result

    def test_none_returns_none(self):
        # [FUNCTION DOC]
        # Asserts None prompts resolve to None.
        collator = make_collator(prompts=None)
        assert collator._resolve_prompts({}) is None


class TestGetPromptForColumn:
    # [CLASS DOC]
    # Tests the private per-column prompt lookup.
    def test_string_prompt_returns_for_any_column(self):
        # [FUNCTION DOC]
        # Asserts a string prompt applies to any column.
        collator = make_collator()
        assert collator._get_prompt_for_column("Search: ", "anything") == "Search: "

    def test_dict_prompt_returns_matching_column(self):
        # [FUNCTION DOC]
        # Asserts a dict prompt returns the entry for each named column.
        collator = make_collator()
        prompts = {"query": "Q: ", "document": "D: "}
        assert collator._get_prompt_for_column(prompts, "query") == "Q: "
        assert collator._get_prompt_for_column(prompts, "document") == "D: "

    def test_dict_prompt_returns_none_for_missing_column(self):
        # [FUNCTION DOC]
        # Asserts a dict prompt returns None for a column it lacks.
        collator = make_collator()
        assert collator._get_prompt_for_column({"query": "Q: "}, "other") is None


class TestColumnOrderWarning:
    # [CLASS DOC]
    # Tests the warning about column order (e.g. an answer-like column before a question-like column). Uses pytest's
    # log capture.
    def test_warns_on_wrong_order(self, caplog):
        # [FUNCTION DOC]
        # Asserts collating a row whose "answer" column precedes "question" logs a warning mentioning "answer" and
        # "index".
        collator = make_collator()
        features = [{"answer": "a", "question": "q"}]
        with caplog.at_level(logging.WARNING):
            collator(features)
        assert any("answer" in msg and "index" in msg for msg in caplog.messages)

    def test_warns_only_once_per_column_set(self, caplog):
        # [FUNCTION DOC]
        # Asserts the column-order warning is logged once for the first call and not again for a second call with the
        # same columns.
        collator = make_collator()
        features = [{"answer": "a", "question": "q"}]
        with caplog.at_level(logging.WARNING):
            collator(features)
            count_first = sum(1 for msg in caplog.messages if "answer" in msg)
            collator(features)
            count_second = sum(1 for msg in caplog.messages if "answer" in msg)
        assert count_first == 1
        assert count_second == count_first

    def test_no_warning_for_correct_order(self, caplog):
        # [FUNCTION DOC]
        # Asserts no column-order warning is logged when "question" precedes "answer".
        collator = make_collator()
        features = [{"question": "q", "answer": "a"}]
        with caplog.at_level(logging.WARNING):
            collator(features)
        assert not any("question" in msg and "index" in msg for msg in caplog.messages)


class TestIdColumnWarning:
    # [CLASS DOC]
    # Tests the warning for columns that look like IDs.
    def test_warns_on_id_like_columns(self, caplog):
        # [FUNCTION DOC]
        # Asserts a "query_id" column triggers a warning containing "look like ID columns" but is still preprocessed.
        collator = make_collator()
        features = [{"query_id": "q1", "sentence": "hello"}]
        with caplog.at_level(logging.WARNING):
            batch = collator(features)
        assert any("look like ID columns" in msg for msg in caplog.messages)
        # Warn-only: the column is still tokenized.
        assert "query_id_input_ids" in batch

    def test_no_warning_without_id_like_columns(self, caplog):
        # [FUNCTION DOC]
        # Asserts ordinary text columns do not trigger the ID-column warning.
        collator = make_collator()
        features = [{"sentence1": "hello", "sentence2": "world"}]
        with caplog.at_level(logging.WARNING):
            collator(features)
        assert not any("look like ID columns" in msg for msg in caplog.messages)


class TestMaxLength:
    # [CLASS DOC]
    # Tests how the max length setting is forwarded to the preprocess function.
    def make_recording_collator(self, **kwargs) -> tuple[BaseDataCollator, list[dict]]:
        # [FUNCTION DOC]
        # HELPER: Builds a BaseDataCollator with a recording preprocess function and returns it together with the list
        # of recorded calls. Keyword arguments go to the constructor.
        calls = []

        def preprocess_fn(texts, prompt=None, task=None, **extra):
            # [FUNCTION DOC]
            # HELPER: Recording preprocess stub: appends a dict of the task and any extra keyword arguments to the
            # call list and returns input ids of ones (n by 3).
            calls.append({"task": task, **extra})
            return {"input_ids": torch.ones(len(texts), 3, dtype=torch.long)}

        return BaseDataCollator(preprocess_fn=preprocess_fn, **kwargs), calls

    def test_scalar_applies_to_every_column(self):
        # [FUNCTION DOC]
        # Asserts a scalar max length of 64 is passed for both columns.
        collator, calls = self.make_recording_collator(max_length=64)
        collator([{"question": "q", "answer": "a"}])
        assert [call.get("max_length") for call in calls] == [64, 64]

    def test_dict_resolves_by_task(self):
        # [FUNCTION DOC]
        # Asserts a per-task max length dict with a router mapping passes 32 for the query column and 180 for the
        # document column.
        collator, calls = self.make_recording_collator(
            max_length={"query": 32, "document": 180},
            router_mapping={"question": "query", "answer": "document"},
        )
        collator([{"question": "q", "answer": "a"}])
        assert [call.get("max_length") for call in calls] == [32, 180]

    def test_dict_without_task_leaves_uncapped(self):
        # [FUNCTION DOC]
        # Asserts a per-task dict without any router mapping (so no task) passes no max length.
        collator, calls = self.make_recording_collator(max_length={"query": 32})
        collator([{"question": "q", "answer": "a"}])
        assert all("max_length" not in call for call in calls)

    def test_dict_missing_task_entry_uncapped(self):
        # [FUNCTION DOC]
        # Asserts that with a router mapping only for "question", that column gets 32 and the unmapped column gets no
        # max length.
        collator, calls = self.make_recording_collator(max_length={"query": 32}, router_mapping={"question": "query"})
        collator([{"question": "q", "answer": "a"}])
        assert calls[0]["max_length"] == 32
        assert "max_length" not in calls[1]

    def test_unset_never_reaches_legacy_preprocess_fns(self):
        # [FUNCTION DOC]
        # Asserts that with max length unset, the default stub (which accepts no extra keyword arguments) still works,
        # i.e. no max length argument is passed.
        # make_collator's preprocess_fn has no **kwargs: with max_length unset the kwarg must not be passed.
        collator = make_collator()
        batch = collator([{"question": "q"}])
        assert "question_input_ids" in batch


class TestTaskResolution:
    # [CLASS DOC]
    # Tests per-column task resolution and stamping of tasks into the batch.
    def test_base_hook_returns_router_mapping_entry(self):
        # [FUNCTION DOC]
        # Asserts the private task hook returns the router mapping entry for a mapped column and None for an unmapped
        # one.
        collator = make_collator()
        assert collator._get_task_for_column("question", 0, {"question": "query"}) == "query"
        assert collator._get_task_for_column("answer", 1, {"question": "query"}) is None

    def test_task_stamped_when_router_mapping_resolves(self):
        # [FUNCTION DOC]
        # Asserts the batch contains a "column_task" entry with the mapped task for each mapped column.
        collator = make_collator(router_mapping={"question": "query", "answer": "document"})
        batch = collator([{"question": "q", "answer": "a"}])
        assert batch["question_task"] == "query"
        assert batch["answer_task"] == "document"

    def test_no_stamp_without_router_mapping(self):
        # [FUNCTION DOC]
        # Asserts no task entries are added when there is no router mapping.
        collator = make_collator()
        batch = collator([{"question": "q", "answer": "a"}])
        assert "question_task" not in batch
        assert "answer_task" not in batch
