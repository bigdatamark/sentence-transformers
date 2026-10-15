# [MODULE DOC]
# HELPER: Decorators that keep deprecated call signatures working for public APIs.
# Each decorator wraps a function, rewrites deprecated keyword (and sometimes positional)
# arguments into their current names, logs a deprecation warning through the transformers
# logger (mostly "warning_once", so each message is shown once per process), then calls
# the wrapped function. A module-level logger and a type variable F (any callable) are defined.
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

from transformers.utils import logging as transformers_logging

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def deprecated_kwargs(**renames: str) -> Callable[[F], F]:
    # [FUNCTION DOC]
    # HELPER: Decorator factory for generic keyword renames.
    # Input: keyword arguments mapping old parameter name to new parameter name.
    # Returns a decorator; the decorated wrapper, for each old name present in the call's kwargs,
    # logs a warning_once naming old and new name, then moves the value to the new name unless
    # the new name was also passed, in which case the old value is discarded. Then calls the
    # original function with the rewritten kwargs and returns its result. Uses functools.wraps.
    """Decorator factory that transparently renames deprecated keyword arguments.

    Emits a deprecation warning when a caller uses the old name. If both old and
    new names are provided, the old name is silently dropped in favor of the new.

    Usage::

        @deprecated_kwargs(sentence_embedding_dimension="embedding_dimension")
        def __init__(self, embedding_dimension: int, ...):
            ...

    Note: for backward-compatible loading of saved configs without warnings,
    use ``config_key_renames`` on the :class:`Module` subclass. This
    decorator is for warning users who explicitly pass the old keyword argument.
    """

    def decorator(func: F) -> F:
        # [FUNCTION DOC]
        # HELPER: Takes the function to decorate and returns the renaming wrapper described above.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Applies the renames to kwargs (mutating the local kwargs dict) and forwards all arguments.
            for old_name, new_name in renames.items():
                if old_name in kwargs:
                    logger.warning_once(
                        f"The `{old_name}` argument was renamed and is now deprecated. "
                        f"Please use `{new_name}` instead."
                    )
                    if new_name not in kwargs:
                        kwargs[new_name] = kwargs.pop(old_name)
                    else:
                        kwargs.pop(old_name)
            return func(*args, **kwargs)

        return cast("F", wrapper)

    return decorator


def transformer_kwargs_decorator(func):
    # [FUNCTION DOC]
    # HELPER: Decorator for Transformer.__init__ handling legacy kwargs.
    # Renames model_args to model_kwargs, tokenizer_args to processor_kwargs, config_args to
    # config_kwargs, logging a warning_once per rename; the old value is used only if the new name
    # is absent. A "cache_dir" kwarg is always removed; if it is not None, a warning_once is logged
    # and it is inserted as a default "cache_dir" key into each of model_kwargs, processor_kwargs
    # and config_kwargs (creating empty dicts if missing, never overriding an existing cache_dir).
    # KNOWN ISSUE: if a caller passes model_kwargs/processor_kwargs/config_kwargs explicitly as None, the cache_dir handling keeps None (setdefault does not replace it) and then fails trying to set a key on None.
    """Decorator for :class:`Transformer.__init__` that handles deprecated keyword arguments.

    Handles the following legacy kwargs:

    * ``model_args`` -> ``model_kwargs``
    * ``tokenizer_args`` -> ``processor_kwargs``
    * ``config_args`` -> ``config_kwargs``
    * ``cache_dir`` -> distributed into ``model_kwargs``, ``processor_kwargs``, and ``config_kwargs``
    """
    _RENAMED_KWARGS = {
        "model_args": "model_kwargs",
        "tokenizer_args": "processor_kwargs",
        "config_args": "config_kwargs",
    }

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Performs the renames and cache_dir distribution on kwargs, then calls the wrapped function.
        for old_name, new_name in _RENAMED_KWARGS.items():
            if old_name in kwargs:
                kwarg_value = kwargs.pop(old_name)
                logger.warning_once(
                    f"The Transformer `{old_name}` argument was renamed and is now deprecated, "
                    f"please use `{new_name}` instead."
                )
                if new_name not in kwargs:
                    kwargs[new_name] = kwarg_value

        if "cache_dir" in kwargs:
            cache_dir = kwargs.pop("cache_dir")
            if cache_dir is not None:
                logger.warning_once(
                    "The Transformer `cache_dir` argument is deprecated. "
                    "Please pass `cache_dir` via `model_kwargs`, `processor_kwargs`, and/or `config_kwargs` instead."
                )
                for dict_name in ("model_kwargs", "processor_kwargs", "config_kwargs"):
                    kwargs.setdefault(dict_name, {})
                    kwargs[dict_name].setdefault("cache_dir", cache_dir)

        return func(*args, **kwargs)

    return wrapper


def cross_encoder_init_args_decorator(func):
    # [FUNCTION DOC]
    # HELPER: Decorator for CrossEncoder.__init__ handling legacy positional and keyword args.
    # Positional: the first positional argument (model_name_or_path) is kept; following positional
    # values are mapped in order to num_labels, max_length, activation_fn, device, each logging a
    # warning (not once-only) and set as kwargs unless already given by keyword.
    # Keyword renames (warning_once each, old value used only when new absent): model_name,
    # automodel_args, tokenizer_args, tokenizer_kwargs, config_args, cache_dir (to cache_folder),
    # default_activation_function (to activation_fn).
    # classifier_dropout is removed, a warning logged, and stored into config_kwargs (creating the
    # dict if absent, otherwise overwriting the key in the given dict, which mutates the caller's dict).
    # KNOWN ISSUE: positional arguments beyond the fifth are silently discarded instead of raising a TypeError.
    # KNOWN ISSUE: if config_kwargs is passed as None together with classifier_dropout, assigning into it raises a TypeError.
    """Decorator for :class:`CrossEncoder.__init__` that handles deprecated keyword arguments.

    Handles the following legacy kwargs:

    * ``model_name`` -> ``model_name_or_path``
    * ``automodel_args`` -> ``model_kwargs``
    * ``tokenizer_args`` -> ``processor_kwargs``
    * ``tokenizer_kwargs`` -> ``processor_kwargs``
    * ``config_args`` -> ``config_kwargs``
    * ``cache_dir`` -> ``cache_folder``
    * ``default_activation_function`` -> ``activation_fn``
    * ``classifier_dropout`` -> ``config_kwargs["classifier_dropout"]``

    Also handles legacy positional arguments (``num_labels``, ``max_length``,
    ``activation_fn``, ``device``), which are now keyword-only.
    """

    # Old positional order after model_name_or_path (which is still positional):
    # num_labels, max_length, activation_fn, device
    _POSITIONAL_ARGS = ("num_labels", "max_length", "activation_fn", "device")

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Receives self plus the original call arguments, applies the conversions above and calls the wrapped init.
        # Handle old-style positional arguments: in v5.3 and earlier, num_labels,
        # max_length, activation_fn, and device could be passed positionally.
        if args:
            # First positional arg is model_name_or_path (still positional), rest are legacy
            new_args = args[:1]
            for i, value in enumerate(args[1:]):
                if i < len(_POSITIONAL_ARGS):
                    name = _POSITIONAL_ARGS[i]
                    logger.warning(
                        f"Passing `{name}` as a positional argument to CrossEncoder is deprecated. "
                        f"Please use `{name}={value!r}` as a keyword argument instead."
                    )
                    if name not in kwargs:
                        kwargs[name] = value
            args = new_args

        kwargs_renamed_mapping = {
            "model_name": "model_name_or_path",
            "automodel_args": "model_kwargs",
            "tokenizer_args": "processor_kwargs",
            "tokenizer_kwargs": "processor_kwargs",
            "config_args": "config_kwargs",
            "cache_dir": "cache_folder",
            "default_activation_function": "activation_fn",
        }
        for old_name, new_name in kwargs_renamed_mapping.items():
            if old_name in kwargs:
                kwarg_value = kwargs.pop(old_name)
                logger.warning_once(
                    f"The CrossEncoder `{old_name}` argument was renamed and is now deprecated. Please use `{new_name}` instead."
                )
                if new_name not in kwargs:
                    kwargs[new_name] = kwarg_value

        if "classifier_dropout" in kwargs:
            classifier_dropout = kwargs.pop("classifier_dropout")
            logger.warning(
                f"The CrossEncoder `classifier_dropout` argument is deprecated. Please use `config_kwargs={{'classifier_dropout': {classifier_dropout}}}` instead."
            )
            if "config_kwargs" not in kwargs:
                kwargs["config_kwargs"] = {"classifier_dropout": classifier_dropout}
            else:
                kwargs["config_kwargs"]["classifier_dropout"] = classifier_dropout

        return func(self, *args, **kwargs)

    return wrapper


def cross_encoder_predict_rank_args_decorator(func: F) -> F:
    # [FUNCTION DOC]
    # HELPER: Decorator for CrossEncoder.predict and CrossEncoder.rank.
    # At decoration time records the method name and which of convert_to_numpy/convert_to_tensor
    # are not in the wrapped function's signature. The returned wrapper is additionally wrapped with
    # deprecated_kwargs renaming sentences to inputs and activation_fct to activation_fn.
    # At call time it removes num_workers (warning_once, no effect) and removes any of the
    # signature-absent convert flags (warning_once each), then calls the function.
    """Decorator for :class:`CrossEncoder.predict` / :class:`CrossEncoder.rank` that handles deprecated keyword arguments.

    Handles the following legacy kwargs:

    * ``sentences`` -> ``inputs`` (via :func:`deprecated_kwargs`)
    * ``activation_fct`` -> ``activation_fn``
    * ``num_workers`` -> removed (no-op)
    * ``convert_to_numpy`` / ``convert_to_tensor`` -> removed (no-op) on methods that no longer
      declare them, which is ``rank`` but not ``predict``
    """
    method = func.__name__
    parameters = inspect.signature(func).parameters
    dropped_convert_kwargs = [name for name in ("convert_to_numpy", "convert_to_tensor") if name not in parameters]

    @deprecated_kwargs(sentences="inputs", activation_fct="activation_fn")
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Drops the obsolete keyword arguments with warnings and forwards to the wrapped method.
        if "num_workers" in kwargs:
            kwargs.pop("num_workers")
            logger.warning_once(
                f"The CrossEncoder.{method} `num_workers` argument is deprecated and has no effect. "
                "It will be removed in a future version."
            )

        for name in dropped_convert_kwargs:
            if name in kwargs:
                kwargs.pop(name)
                logger.warning_once(
                    f"The CrossEncoder.{method} `{name}` argument is deprecated and has no effect, as "
                    f"`{method}` always returns Python floats. It will be removed in a future version."
                )

        return func(self, *args, **kwargs)

    return cast("F", wrapper)


def save_to_hub_args_decorator(func):
    # [FUNCTION DOC]
    # HELPER: Decorator for BaseModel.save_to_hub.
    # Pops a "repo_name" kwarg; if truthy and "repo_id" is not given, logs a warning_once and uses it
    # as repo_id (otherwise it is silently dropped). If two or more positional arguments (after self)
    # are given, inserts None as the third positional argument so legacy positional calls line up
    # with the newly inserted "token" parameter. Returns the wrapped function's result.
    """
    A decorator to update the signature of the :class:`~sentence_transformers.base.model.BaseModel.save_to_hub` method
    to replace the deprecated `repo_name` argument with `repo_id`, and to introduce backwards compatibility for
    positional arguments despite a newly added `token` argument.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Applies the repo_name rename and positional shift, then calls save_to_hub.
        # If repo_id not already set, use repo_name
        repo_name = kwargs.pop("repo_name", None)
        if repo_name and "repo_id" not in kwargs:
            logger.warning_once(
                "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated. Please use `repo_id` instead."
            )
            kwargs["repo_id"] = repo_name

        # If positional args are used, adjust for the new "token" keyword argument
        if len(args) >= 2:
            args = (*args[:2], None, *args[2:])

        return func(self, *args, **kwargs)

    return wrapper
