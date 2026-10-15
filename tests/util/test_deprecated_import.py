# [MODULE DOC]
# Tests for the deprecated import path redirection in sentence_transformers.util.deprecated_import: each old
# module path in the deprecation mapping must import with a DeprecationWarning and resolve to the same object
# as its new path, and class-named paths must not shadow the class with a module.
from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from sentence_transformers.util.deprecated_import import DEPRECATED_MODULE_PATHS


@pytest.mark.parametrize(("deprecated_path", "new_path"), DEPRECATED_MODULE_PATHS.items())
def test_deprecated_import_paths(deprecated_path: str, new_path: str):
    # [FUNCTION DOC]
    # Parametrized over (deprecated path, new path) pairs. Removes the deprecated path from the global module
    # cache, imports it while recording warnings and asserts at least one DeprecationWarning carries the exact
    # "Importing from ... is deprecated ... Please use ... instead." message; then imports the new path and asserts
    # no DeprecationWarning is raised; finally asserts both imports return the identical module object. Mutates
    # the global module registry.
    """Test that deprecated import paths point to the correct new paths."""
    # Remove from sys.modules so the meta path finder is triggered fresh
    sys.modules.pop(deprecated_path, None)

    with warnings.catch_warnings(record=True) as warnings_list:
        warnings.simplefilter("always", DeprecationWarning)

        # Import the deprecated module
        deprecated_module = importlib.import_module(deprecated_path)
        deprecation_warnings = [
            warning for warning in warnings_list if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) > 0
        assert any(
            (
                f"Importing from '{deprecated_path}' is deprecated and will be removed in a future version. "
                f"Please use '{new_path}' instead."
            )
            in str(warning.message)
            for warning in deprecation_warnings
        )
        assert deprecated_module is not None

    with warnings.catch_warnings(record=True) as warnings_list:
        warnings.simplefilter("always", DeprecationWarning)
        # Import the new module
        new_module = importlib.import_module(new_path)
        deprecation_warnings = [
            warning for warning in warnings_list if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0
        assert new_module is not None

    # Verify that both imports point to the same module
    assert deprecated_module is new_module


@pytest.mark.parametrize("deprecated_path", DEPRECATED_MODULE_PATHS.keys())
def test_imported_from_partial_path_first(deprecated_path: str):
    # [FUNCTION DOC]
    # Parametrized over deprecated paths; skips unless the last path component starts with an uppercase letter
    # (a class name). Imports the parent module first and gets the class attribute, then imports the full path and
    # gets the same-named attribute, and asserts both are non-None and identical.
    """Test that classes from deprecated paths are the same object as from new paths."""
    module_name = ".".join(deprecated_path.split(".")[:-1])
    class_name = deprecated_path.split(".")[-1]
    if not class_name[0].isupper():
        pytest.skip("Only test for module paths that are both modules and classes.")

    module_from_partial_path = importlib.import_module(module_name)
    class_from_partial_path = getattr(module_from_partial_path, class_name, None)

    module_from_full_path = importlib.import_module(deprecated_path)
    class_from_full_path = getattr(module_from_full_path, class_name, None)

    assert class_from_partial_path is not None, f"Could not get {class_name} from new path {module_name}"
    assert class_from_full_path is not None, f"Could not get {class_name} from deprecated path {deprecated_path}"

    assert class_from_full_path is class_from_partial_path, (
        f"{class_name} from deprecated path {deprecated_path} is not the same object as from new path {module_name}"
    )


@pytest.mark.parametrize("deprecated_path", DEPRECATED_MODULE_PATHS.keys())
def test_imported_from_full_path_first(deprecated_path: str):
    # [FUNCTION DOC]
    # Same as the previous test but imports the full deprecated path first and the parent module second, guarding
    # against the parent package attribute being replaced by a submodule object.
    """Test that classes from deprecated paths are the same object as from new paths.

    Some import mechanisms had significant issues with e.g.:
    >>> from sentence_transformers.SentenceTransformer import SentenceTransformer as S1
    >>> from sentence_transformers import SentenceTransformer as S2

    Where S2 would be a `module` instead of the expected `SentenceTransformer` class, as the first import had already
    introduced `sentence_transformers.SentenceTransformer` as a module.
    """
    module_name = ".".join(deprecated_path.split(".")[:-1])
    class_name = deprecated_path.split(".")[-1]
    if not class_name[0].isupper():
        pytest.skip("Only test for module paths that are both modules and classes.")

    module_from_full_path = importlib.import_module(deprecated_path)
    class_from_full_path = getattr(module_from_full_path, class_name, None)

    module_from_partial_path = importlib.import_module(module_name)
    class_from_partial_path = getattr(module_from_partial_path, class_name, None)

    assert class_from_partial_path is not None, f"Could not get {class_name} from new path {module_name}"
    assert class_from_full_path is not None, f"Could not get {class_name} from deprecated path {deprecated_path}"

    assert class_from_full_path is class_from_partial_path, (
        f"{class_name} from deprecated path {deprecated_path} is not the same object as from new path {module_name}"
    )
