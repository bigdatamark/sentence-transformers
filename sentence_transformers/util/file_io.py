# [MODULE DOC]
# File resolution and download helpers for models stored locally or on the Hugging Face Hub.
# Functions first look in the local filesystem, then fall back to the Hub (via hf_hub_download /
# snapshot_download, tagged with library name "sentence-transformers"). Only unambiguous
# "not found" Hub errors become a None result; other errors (auth, rate-limit, network) propagate.
# Also provides a silent tqdm subclass and a streaming HTTP download helper. Logs at debug level.
from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


class disabled_tqdm(tqdm):
    # [CLASS DOC]
    # HELPER: tqdm subclass whose progress bar is always disabled, used to silence snapshot downloads.
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Forces the "disable" keyword to True and forwards everything to tqdm's constructor.
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Deletes an attribute, ignoring the AttributeError raised for a missing "_lock"
        # attribute (a known huggingface_hub interaction); other AttributeErrors are re-raised.
        """Fix for https://github.com/huggingface/huggingface_hub/issues/1603"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


def is_sentence_transformer_model(
    model_name_or_path: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> bool:
    # [FUNCTION DOC]
    # Returns True if the model (local dir or Hub repo id) contains a "modules.json" file, as resolved
    # by load_file_path with the given token, cache folder, revision and local-only flag; False if not.
    # Non-"not found" Hub errors propagate. May download the file (network, cache writes).
    """
    Checks if the given model name or path corresponds to a SentenceTransformer model.

    Args:
        model_name_or_path (str): The name or path of the model.
        token (Optional[Union[bool, str]]): The token to be used for authentication. Defaults to None.
        cache_folder (Optional[str]): The folder to cache the model files. Defaults to None.
        revision (Optional[str]): The revision of the model. Defaults to None.
        local_files_only (bool): Whether to only use local files for the model. Defaults to False.

    Raises:
        Exception: Propagates errors from the Hub that are not "file not found" (e.g.
            authentication, rate-limit, or network errors). The probe deliberately surfaces
            these instead of returning ``False``, so callers don't mistake a flaky Hub for
            "this isn't a SentenceTransformer".

    Returns:
        bool: True if the model is a SentenceTransformer model, False if it exists but
        does not contain a ``modules.json``.
    """
    return bool(
        load_file_path(
            model_name_or_path,
            "modules.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def load_file_path(
    model_name_or_path: str,
    filename: str | Path,
    subfolder: str = "",
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str | None:
    # [FUNCTION DOC]
    # Resolves a single file of a model to a local path.
    # Inputs: model name/path, filename (string or Path), optional subfolder, token, cache folder,
    # revision, local_files_only.
    # 1) If model_name_or_path/subfolder/filename exists locally, returns that path as a string.
    # 2) If model_name_or_path is an existing local directory (but lacks the file), returns None.
    # 3) Otherwise downloads from the Hub, splitting subfolder+filename into the final file name and
    # its parent directory (posix) as the Hub subfolder, and returns the cached path.
    # EntryNotFoundError, HFValidationError and LocalEntryNotFoundError are logged at debug and give
    # None; all other exceptions propagate. Side effects: network, cache directory writes.
    """
    Loads a file from a local or remote location.

    Args:
        model_name_or_path (str): The model name or path.
        filename (str): The name of the file to load.
        subfolder (str): The subfolder within the model subfolder (if applicable).
        token (Optional[Union[bool, str]]): The token to access the remote file (if applicable).
        cache_folder (Optional[str]): The folder to cache the downloaded file (if applicable).
        revision (Optional[str], optional): The revision of the file (if applicable). Defaults to None.
        local_files_only (bool, optional): Whether to only consider local files. Defaults to False.

    Raises:
        Exception: Errors that are not unambiguously "this file is not on the Hub"
            propagate to the caller (e.g. authentication, rate-limit, network). Only
            ``EntryNotFoundError``, ``HFValidationError``, and ``LocalEntryNotFoundError``
            are converted to a ``None`` return.

    Returns:
        Optional[str]: The path to the loaded file, or ``None`` if the file is not present
        locally and the Hub confirms it is not in the repo (or no valid repo id was given).
    """
    # If file is local
    file_path = Path(model_name_or_path, subfolder, filename)
    if file_path.exists():
        return str(file_path)

    # Skip the Hub fallback when the parent is a real local dir. Avoids a wasted call.
    if Path(model_name_or_path).is_dir():
        return None

    # If file is remote
    file_path = Path(subfolder, filename)
    try:
        return hf_hub_download(
            model_name_or_path,
            filename=file_path.name,
            subfolder=file_path.parent.as_posix(),
            revision=revision,
            library_name="sentence-transformers",
            token=token,
            cache_dir=cache_folder,
            local_files_only=local_files_only,
        )
    except (EntryNotFoundError, HFValidationError, LocalEntryNotFoundError) as exc:
        # Unambiguous "not found" cases. Other errors (auth, rate limit, network)
        # propagate so callers don't silently fall back to a different model.
        logger.debug(f"Could not load {filename!r} from {model_name_or_path!r}: {exc}")
        return None


def load_dir_path(
    model_name_or_path: str,
    subfolder: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str | None:
    # [FUNCTION DOC]
    # Resolves a model subfolder to a local directory path.
    # Converts a Path subfolder to a posix string. Returns the local path if model/subfolder exists;
    # returns None if the model path is a local directory without it. Otherwise snapshot-downloads the
    # repo restricted to files under the subfolder (no pattern restriction for "" or "."), with a
    # silent progress bar. HFValidationError / LocalEntryNotFoundError are logged at debug and give
    # None. Any other error triggers one retry with local_files_only enabled; if that retry finds
    # nothing in the cache, the original error is re-raised (cache-miss suppressed from the traceback).
    # Returns the snapshot path joined with the subfolder as a string.
    # KNOWN ISSUE: if the repo exists on the Hub but has no such subfolder, the snapshot download succeeds and the function returns a path that does not exist, rather than None as the docstring suggests.
    """
    Loads the subfolder path for a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        subfolder (str): The subfolder to load.
        token (Optional[Union[bool, str]]): The token for authentication.
        cache_folder (Optional[str]): The folder to cache the downloaded files.
        revision (Optional[str], optional): The revision of the model. Defaults to None.
        local_files_only (bool, optional): Whether to only use local files. Defaults to False.

    Raises:
        Exception: Errors that are not unambiguously "this is not on the Hub" propagate
            to the caller. ``HFValidationError`` and ``LocalEntryNotFoundError`` are
            converted to a ``None`` return. Other failures (auth, rate-limit, network)
            trigger a single cache-only retry. If the cache also lacks the file, the
            original exception is re-raised (a cache miss would mask the real cause).

    Returns:
        Optional[str]: The subfolder path, or ``None`` if the parent is a local directory
        without the subfolder, or no valid repo id was given.
    """
    if isinstance(subfolder, Path):
        subfolder = subfolder.as_posix()

    # If file is local
    dir_path = Path(model_name_or_path, subfolder)
    if dir_path.exists():
        return str(dir_path)

    # Skip the Hub fallback when the parent is a real local dir. Avoids a wasted call.
    if Path(model_name_or_path).is_dir():
        return None

    download_kwargs = {
        "repo_id": model_name_or_path,
        "revision": revision,
        "allow_patterns": f"{subfolder}/**" if subfolder not in ["", "."] else None,
        "library_name": "sentence-transformers",
        "token": token,
        "cache_dir": cache_folder,
        "local_files_only": local_files_only,
        "tqdm_class": disabled_tqdm,
    }
    # Try to download from the remote
    try:
        repo_path = snapshot_download(**download_kwargs)
    except (HFValidationError, LocalEntryNotFoundError) as exc:
        # Unambiguous "not found" / "not cached" cases.
        logger.debug(f"Could not load subfolder {subfolder!r} from {model_name_or_path!r}: {exc}")
        return None
    except Exception as first_error:
        # Transient (auth, rate limit, network), try cache as fallback.
        download_kwargs["local_files_only"] = True
        try:
            repo_path = snapshot_download(**download_kwargs)
        except LocalEntryNotFoundError:
            # Cache miss after a transient first failure: re-raise the original error
            # (with `from None` to suppress the cache miss from the traceback) so the
            # user sees the real cause, e.g. rate limit, not the misleading cache miss.
            raise first_error from None
    return str(Path(repo_path, subfolder))


def http_get(url: str, path: str) -> None:
    # [FUNCTION DOC]
    # Downloads url to the local file path with a byte progress bar.
    # Raises ImportError if httpx is not installed. Creates the parent directory if path has one.
    # Streams a GET request (following redirects), raising httpx's status error for non-success codes.
    # Uses Content-Length as progress total when present. Writes 1024-byte chunks to "<path>_part",
    # then atomically replaces path with it. On any error the partial file is removed and the error
    # re-raised. The progress bar is always closed. Returns None. Side effects: network, file writes.
    """Download a URL to a local file with a progress bar.

    The content is streamed in chunks and first written to a temporary
    ``"<path>_part"`` file, which is atomically moved to ``path`` once the
    download has completed successfully. Parent directories of ``path`` are
    created automatically if they do not exist.

    Args:
        url (str): The HTTP(S) URL to download.
        path (str): Destination file path on the local filesystem.

    Raises:
        ImportError: If the optional ``httpx`` dependency is not installed.
        httpx.HTTPStatusError: If the HTTP request returns a non-success status code.
        OSError: If the file cannot be written to ``path``.

    Returns:
        None
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required to use this function. Please install it via `pip install httpx`.")

    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    download_filepath = path + "_part"
    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(
            unit="B", total=total, unit_scale=True, leave=False, desc=f"Downloading {os.path.basename(path)}"
        )

        try:
            with open(download_filepath, "wb") as file_binary:
                for chunk in response.iter_bytes(chunk_size=1024):
                    if chunk:
                        progress.update(len(chunk))
                        file_binary.write(chunk)
            os.replace(download_filepath, path)
        except Exception:
            if os.path.exists(download_filepath):
                os.remove(download_filepath)
            raise
        finally:
            progress.close()
