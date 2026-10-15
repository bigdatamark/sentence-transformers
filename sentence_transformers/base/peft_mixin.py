# [MODULE DOC]
# PEFT adapter support for models. Provides a decorator that turns a stub method into a delegation to the method of the
# same name on the model's underlying transformers model (after checking PEFT compatibility), and PeftAdapterMixin,
# which exposes the transformers PEFT adapter API (load, add, set, enable/disable, query, state dict, delete) on models.
from __future__ import annotations

from functools import wraps

from transformers.integrations.peft import PeftAdapterMixin as PeftAdapterMixinTransformers


def peft_wrapper(func):
    # [FUNCTION DOC]
    # HELPER: decorator. Returns a wrapper (keeping the wrapped function's metadata) that ignores the wrapped body: it checks
    # that the model is PEFT compatible (ValueError otherwise), raises AttributeError if the underlying transformers model
    # lacks a method with the wrapped function's name, and otherwise calls that method with all arguments and returns its
    # result.
    """Wrapper to call the method on the auto_model with a check for PEFT compatibility."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: the delegating wrapper described above; the instance must expose a transformers_model attribute.
        self.check_peft_compatible_model()
        method_name = func.__name__
        if not hasattr(self.transformers_model, method_name):
            raise AttributeError(
                f"The underlying transformers model ({type(self.transformers_model).__name__}) does not have a "
                f"`{method_name}` method. This may indicate an incompatible or outdated version of transformers or peft."
            )
        method = getattr(self.transformers_model, method_name)
        return method(*args, **kwargs)

    return wrapper


class PeftAdapterMixin:
    # [CLASS DOC]
    # Mixin adding PEFT adapter methods to a model that exposes transformers_model. All adapter methods are stubs whose
    # behaviour is delegation to the identically named method of the underlying transformers model.
    """
    Wrapper Mixin that adds the functionality to easily load and use adapters on the model. For
    more details about adapters check out the documentation of PEFT
    library: https://huggingface.co/docs/peft/index

    Currently supported PEFT methods follow those supported by transformers library,
    you can find more information on:
    https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin
    """

    def has_peft_compatible_model(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if the underlying transformers model is an instance of the transformers PEFT adapter mixin.
        return isinstance(self.transformers_model, PeftAdapterMixinTransformers)

    def check_peft_compatible_model(self) -> None:
        # [FUNCTION DOC]
        # Raises ValueError if the model is not PEFT compatible; otherwise does nothing.
        if not self.has_peft_compatible_model():
            raise ValueError(
                "PEFT methods are only supported for Sentence Transformer models that use the Transformer module."
            )

    @peft_wrapper
    def load_adapter(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Delegates to the underlying model's load_adapter (loads adapter weights from a local or Hub folder; requires peft).
        """
        Load adapter weights from file or remote Hub folder." If you are not familiar with adapters and PEFT methods, we
        invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

        Requires peft as a backend to load the adapter weights and the underlying model to be compatible with PEFT.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `load_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.load_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `load_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.load_adapter
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def add_adapter(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Delegates to the underlying model's add_adapter (adds a new trainable adapter, default name "default").
        """
        Adds a fresh new adapter to the current model for training purposes. If no adapter name is passed, a default
        name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use "default" as the
        default adapter name).

        Requires peft as a backend to load the adapter weights and the underlying model to be compatible with PEFT.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `add_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.add_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `add_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.add_adapter

        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def set_adapter(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Delegates to the underlying model's set_adapter (activates one adapter, disabling others).
        """
        Sets a specific adapter by forcing the model to use that adapter and disable the other adapters.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `set_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.set_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `set_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.set_adapter
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def disable_adapters(self) -> None:
        # [FUNCTION DOC]
        # Delegates to the underlying model's disable_adapters (inference with the base model only).
        """
        Disable all adapters that are attached to the model. This leads to inferring with the base model only.
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def enable_adapters(self) -> None:
        # [FUNCTION DOC]
        # Delegates to the underlying model's enable_adapters.
        """
        Enable adapters that are attached to the model. The model will use `self.active_adapter()`
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def active_adapters(self) -> list[str]:
        # [FUNCTION DOC]
        # Delegates to the underlying model's active_adapters and returns the list of active adapter names.
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters
        for inference) returns the list of all active adapters so that users can deal with them accordingly.

        For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return
        a single string.
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def active_adapter(self) -> str: ...  # Implementation handled by the wrapper
        # [FUNCTION DOC]
        # Delegates to the underlying model's active_adapter and returns the active adapter name.

    @peft_wrapper
    def get_adapter_state_dict(self, *args, **kwargs) -> dict:
        # [FUNCTION DOC]
        # Delegates to the underlying model's get_adapter_state_dict and returns the adapter's weight tensors (active adapter
        # if no name is given).
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter.
        If no adapter_name is passed, the active adapter is used.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `get_adapter_state_dict` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.get_adapter_state_dict
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `get_adapter_state_dict` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.get_adapter_state_dict
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def delete_adapter(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Delegates to the underlying model's delete_adapter, removing an adapter's layers.
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Delete an adapter's LoRA layers from the underlying model.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `delete_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.delete_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `delete_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.delete_adapter
        """
