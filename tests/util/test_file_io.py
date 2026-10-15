# [MODULE DOC]
# Tests for sentence_transformers.util.file_io load_file_path and load_dir_path: local short-circuits when a
# path is an existing local directory, Hub fallbacks (hf_hub_download and snapshot_download are mocked), and
# which Hub errors are swallowed (return None) versus propagated, including the retry against the local cache.
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError

from sentence_transformers.util.file_io import load_dir_path, load_file_path


def test_load_file_path_local_missing_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Mocks the Hub download, asks for a missing file in an existing temp directory, and asserts None is
    # returned and the Hub was never called.
    """When the parent is a real local directory but the requested file is missing,
    `load_file_path` should return ``None`` without calling `hf_hub_download`.

    Without the #3370 guard the call would still return ``None`` (the
    `HFValidationError` raised by the Hub for a Windows-style path is in the catch
    list), so a plain return-value assertion would not detect a regression: this
    test explicitly checks that the Hub call is never issued.
    """
    hub_mock = MagicMock()
    monkeypatch.setattr("sentence_transformers.util.file_io.hf_hub_download", hub_mock)

    result = load_file_path(str(tmp_path), "modules.json", local_files_only=True)

    assert result is None
    hub_mock.assert_not_called()


def test_load_file_path_local_exists(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Writes a config file into a temp directory and asserts its path is returned.
    """`load_file_path` returns the local path when the file is present."""
    target = tmp_path / "config.json"
    target.write_text("{}")

    result = load_file_path(str(tmp_path), "config.json", local_files_only=True)
    assert result is not None
    assert Path(result) == target


def test_load_file_path_nonlocal_path_calls_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Mocks the Hub download to return a fake path and asserts a repo id (not a local directory) calls it once
    # and returns its result.
    """When the path is not an existing local directory, the local-dir guard must
    not fire and `hf_hub_download` should still be invoked.
    """
    hub_mock = MagicMock(return_value="/fake/cache/modules.json")
    monkeypatch.setattr("sentence_transformers.util.file_io.hf_hub_download", hub_mock)

    result = load_file_path("some-org/some-model", "modules.json", local_files_only=True)

    assert result == "/fake/cache/modules.json"
    hub_mock.assert_called_once()


def test_load_dir_path_local_missing_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Mocks the snapshot download, asks for a missing subfolder of an existing temp directory, and asserts None
    # is returned without calling the Hub.
    """Regression test for #3370. The local-dir guard short-circuits to ``None`` and
    `snapshot_download` is never called, so callers don't see a confusing
    `HFValidationError` for what's actually just a missing local file.
    """
    snapshot_mock = MagicMock()
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path(str(tmp_path), "1_Pooling", local_files_only=True)

    assert result is None
    snapshot_mock.assert_not_called()


def test_load_dir_path_local_subfolder_exists(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Creates a subfolder in a temp directory and asserts its path is returned.
    """`load_dir_path` returns the resolved local path when the subfolder exists."""
    subfolder = tmp_path / "1_Pooling"
    subfolder.mkdir()

    result = load_dir_path(str(tmp_path), "1_Pooling", local_files_only=True)
    assert result is not None
    assert Path(result) == subfolder


def test_load_dir_path_nonlocal_path_calls_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Mocks the snapshot download to return a fake path and asserts the result is that path joined with the
    # subfolder, with one call.
    """When the path is not an existing local directory, the local-dir guard must
    not fire and `snapshot_download` should still be invoked.
    """
    snapshot_mock = MagicMock(return_value="/fake/cache/path")
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path("some-org/some-model", "1_Pooling", local_files_only=True)

    assert result == str(Path("/fake/cache/path", "1_Pooling"))
    snapshot_mock.assert_called_once()


def test_load_file_path_entry_not_found_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Asserts an entry-not-found Hub error makes load_file_path return None.
    """`EntryNotFoundError` (file genuinely missing from the repo) should be swallowed
    and return ``None`` so callers can fall back to defaults (e.g. wrapping
    `bert-base-uncased` as a SentenceTransformer when `modules.json` doesn't exist).
    """
    monkeypatch.setattr(
        "sentence_transformers.util.file_io.hf_hub_download",
        MagicMock(side_effect=EntryNotFoundError("file not in repo")),
    )

    result = load_file_path("some-org/some-model", "modules.json")
    assert result is None


def test_load_file_path_transient_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Asserts any other error (a RuntimeError) from the Hub download propagates out of load_file_path.
    """Transient/auth errors (rate limit, 401, network) must propagate. Returning
    ``None`` silently would let `_load_modules` fall back to a vanilla transformer
    when the user actually has a SentenceTransformer model behind the failed call.
    """
    monkeypatch.setattr(
        "sentence_transformers.util.file_io.hf_hub_download",
        MagicMock(side_effect=RuntimeError("simulated rate limit")),
    )

    with pytest.raises(RuntimeError, match="simulated rate limit"):
        load_file_path("some-org/some-model", "modules.json")


def test_load_dir_path_local_entry_not_found_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Asserts a local-entry-not-found error from the snapshot download returns None after exactly one call.
    """`LocalEntryNotFoundError` from the first `snapshot_download` (e.g. offline +
    nothing cached) should short-circuit to ``None`` without retrying the cache,
    since the retry would do the same thing.
    """
    snapshot_mock = MagicMock(side_effect=LocalEntryNotFoundError("not cached"))
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path("some-org/some-model", "1_Pooling", local_files_only=True)
    assert result is None
    snapshot_mock.assert_called_once()


def test_load_dir_path_hf_validation_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Asserts a Hub validation error (malformed repo id) returns None after exactly one call.
    """`HFValidationError` (malformed repo id) is unambiguously "not on the Hub".
    Short-circuit to ``None`` rather than retrying.
    """
    snapshot_mock = MagicMock(side_effect=HFValidationError("bad repo id"))
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path("not a repo id", "1_Pooling")
    assert result is None
    snapshot_mock.assert_called_once()


def test_load_dir_path_transient_with_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Makes the first snapshot call fail with a RuntimeError and the second return a cached path; asserts the
    # joined cached path is returned, two calls were made, and the second used local files only.
    """When the first `snapshot_download` fails with a transient error but the cache
    has the model, `load_dir_path` should return the cached path.
    """
    snapshot_mock = MagicMock(side_effect=[RuntimeError("simulated network error"), "/fake/cache/path"])
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    result = load_dir_path("some-org/some-model", "1_Pooling")
    assert result == str(Path("/fake/cache/path", "1_Pooling"))
    assert snapshot_mock.call_count == 2
    assert snapshot_mock.call_args_list[1].kwargs["local_files_only"] is True


def test_load_dir_path_transient_with_cache_miss_reraises_original(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Makes the first snapshot call fail with a RuntimeError and the cache retry fail with local-entry-not-found;
    # asserts the original RuntimeError is raised after two calls.
    """When the first call fails with a transient error and the cache also lacks the
    model, the original transient error is re-raised (not the cache miss). The cache
    miss would mask the real cause. Users see the rate-limit/auth/network error instead.
    """
    snapshot_mock = MagicMock(
        side_effect=[RuntimeError("simulated network error"), LocalEntryNotFoundError("not cached")],
    )
    monkeypatch.setattr("sentence_transformers.util.file_io.snapshot_download", snapshot_mock)

    with pytest.raises(RuntimeError, match="simulated network error"):
        load_dir_path("some-org/some-model", "1_Pooling")
    assert snapshot_mock.call_count == 2
