#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Utility functions for xlite."""

import threading
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger
from typing import Any, Literal, TypedDict

import torch
import torch.nn as nn
from vllm.logger import logger
from xlite._C import Model, ModelConfig

from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

_MISSING = object()
"""Unique sentinel for missing attributes in this module."""


class AttributeSetterMixin:
    """A mixin that allows setting attributes safely without raising AttributeError for missing attributes. This is
    useful for handling C++ extension objects that may not have all attributes defined in all versions. The class will
    simply ignore attempts to set attributes that do not exist, while allowing setting existing attributes as usual.

    Additionally, a context manager interface is provided for checking the incoming value before setting the attribute.
    The value is only set if the attribute exists and the optional `match_condition` is satisfied.

    Good for backwards compatibility. For subclasses, :mod:`AttributeSetterMixin` must be the first parent class in the
    inheritance chain to work properly (i.e., the second object in the method resolution order (MRO)).

    Example usage::

        class Model:
            def __init__(self):
                self.some_existing_attr = 0


        class MyModel(AttributeSetterMixin, Model):
            _on_missing_attr = "ignore"  # silently ignore missing attributes


        model = MyModel(...)
        model.some_existing_attr = 42  # sets the attribute as usual
        model.some_missing_attr = "hello"  # does nothing, no error raised

        with model.condition(lambda v: isinstance(v, int) and v > 0):
            model.some_existing_attr = -1  # does not set because condition is not met
            model.some_existing_attr = 100  # sets because condition is met
    """

    _on_missing_attr: Literal["raise", "warn", "ignore"] = "warn"
    """Behavior when attempting to set a missing attribute. If `warn`, a logger must be provided to log a warning."""
    _logger: Logger | None = None
    """Optional logger for warning about missing attributes. If None, no warnings will be logged."""

    def __init_subclass__(cls) -> None:
        if cls.__mro__[1] is not AttributeSetterMixin:
            raise TypeError(
                f"{cls.__name__} inherits from AttributeSetterMixin but does not have AttributeSetterMixin as the first"
                f" parent class. Use `class {cls.__name__}(AttributeSetterMixin, ...)` to define the subclass, instead."
            )

    def _get_thread_local(self) -> threading.local:
        """Lazily initialize a per-instance, per-thread local storage without going through __setattr__."""
        try:
            return object.__getattribute__(self, "_thread_local")
        except AttributeError:
            local = threading.local()
            object.__setattr__(self, "_thread_local", local)
            return local

    def __setattr__(self, name: str, value: Any) -> None:
        if not (hasattr(type(self), name) or name in self.__dict__):
            if self._on_missing_attr == "raise":
                raise AttributeError(f"{type(self).__name__} has no attribute {name}.")
            elif self._on_missing_attr == "warn" and self._logger:
                self._logger.warning(
                    "%s has no attribute %s. Your `xlite` version might be incompatible.", type(self).__name__, name
                )
            return
        match_condition = getattr(self._get_thread_local(), "match_condition", None)
        if match_condition is not None and not match_condition(value):
            return
        super().__setattr__(name, value)

    @contextmanager
    def condition(self, match_condition: Callable[..., bool]) -> Generator["AttributeSetterMixin", None, None]:
        """Context manager that gates attribute setting on `match_condition`.

        Usage::

            with obj.condition(lambda v: v > 0):
                obj.some_attr = 42  # only set if 42 > 0
        """
        local = self._get_thread_local()
        previous = getattr(local, "match_condition", None)  # save for nesting
        local.match_condition = match_condition
        try:
            yield self
        finally:
            local.match_condition = previous  # always restore, even on exception


class XModel(AttributeSetterMixin, Model):
    """:mod:`xlite._C.Model` subclass with safe attribute setting for better backwards compatibility."""

    if torch.distributed.get_rank() == 0:
        _logger = logger


class XModelConfig(AttributeSetterMixin, ModelConfig):
    """:mod:`xlite._C.ModelConfig` subclass with safe attribute setting for better backwards compatibility."""

    if torch.distributed.get_rank() == 0:
        _logger = logger


@dataclass
class AttnMetadataRouter:
    """A router for attention metadata objects of different types. This is used to handle the differences in attention
    metadata across different model architectures and vLLM/vLLM-ascend versions in a more robust way.

    The router provides unified access to commonly used attention metadata attributes (e.g., actual sequence lengths for
    query and block tables) via properties.

    Currently included metadata types:

    - `AscendMetadata`
    - `AscendMLAMetadata`
    - `AscendSFAMetadata`

    Typically, the attention metadata has the following notations::

        |---------- N-1 iteration --------|
        |---------------- N iteration ---------------------|
        |- tokenA -|......................|-- newTokens ---|
        |---------- context_len ----------|
        |-------------------- seq_len ---------------------|
                                          |-- query_len ---|
    """

    attn_metadata: Any
    """The attention metadata object to route, e.g., an instance of `AscendMetadata` or `AscendSFAMetadata`."""
    device: str | torch.device | int | None = "cpu"
    """Device specification for the returned tensors. If None, the tensors will be on the same device as the original
    metadata tensors. The current implementation assumes `cpu` device for minimal data transfer."""

    @contextmanager
    def on_device(self, device: str | torch.device | int | None) -> Generator["AttnMetadataRouter", None, None]:
        """Context manager to temporarily set the device for the router. This is useful for cases where we want to
        access multiple properties on the same device without repeatedly specifying the device.

        Usage::

            with router.on_device("cpu"):
                query_lens = router.cu_query_lens  # on cpu
                block_tables = router.block_tables  # also on cpu
        """
        original_device = self.device
        self.device = device
        try:
            yield self
        finally:
            self.device = original_device

    def __getattr__(self, name: str) -> Any:
        """Route attribute access to the appropriate handler method based on the attribute name."""
        if (value := getattr(self.attn_metadata, name, _MISSING)) is not _MISSING:
            return value

        raise AttributeError(f"{type(self.attn_metadata).__name__} has no attribute {name}.")

    @property
    def cu_query_lens(self) -> torch.Tensor:
        """Get the cumulative query lengths from the attention metadata, if available."""
        if isinstance(self.attn_metadata, (AscendMetadata, AscendMLAMetadata)):
            return torch.as_tensor(self.attn_metadata.query_start_loc, device=self.device)

        if isinstance(self.attn_metadata, AscendSFAMetadata):
            return torch.as_tensor(self.attn_metadata.cum_query_lens, device=self.device)

        for candidate in ["query_start_loc", "cum_query_lens", "actual_seq_lengths_q"]:
            if (lengths := getattr(self.attn_metadata, candidate, None)) is not None:
                return torch.as_tensor(lengths, device=self.device)

        raise ValueError(
            f"Cannot find actual sequence lengths for query in attention metadata of type {type(self.attn_metadata)}."
        )

    @property
    def block_tables(self) -> torch.Tensor:
        """Get the block tables from the attention metadata, if available."""
        if isinstance(self.attn_metadata, AscendMetadata):
            return torch.as_tensor(self.attn_metadata.block_tables, device=self.device)

        if isinstance(self.attn_metadata, AscendSFAMetadata):
            return torch.as_tensor(self.attn_metadata.block_table, device=self.device)

        if isinstance(self.attn_metadata, AscendMLAMetadata):
            # AscendMLAMetadataBuilder.build_decode_metadata breaks `AscendMLAMetadata.block_tables`
            # thus we may need to patch together block tables from prefill and decode metadata if available
            block_tables = []
            if self.attn_metadata.decode is not None:
                block_tables.append(torch.as_tensor(self.attn_metadata.decode.block_table, device=self.device))
            if self.attn_metadata.prefill is not None:
                block_tables.append(torch.as_tensor(self.attn_metadata.prefill.block_table, device=self.device))
            if block_tables:
                return torch.concat(block_tables, dim=0)
            return torch.as_tensor(self.attn_metadata.block_tables, device=self.device)

        for candidate in ["block_tables", "block_table"]:
            if (tables := getattr(self.attn_metadata, candidate)) is not None:
                return torch.as_tensor(tables, device=self.device)

        raise ValueError(f"Cannot find block tables in attention metadata of type {type(self.attn_metadata)}.")

    @property
    def seq_lens(self) -> torch.Tensor:
        """Return the per-sequence `seq_lens` tensor in a device-safe torch.Tensor form."""
        if isinstance(self.attn_metadata, (AscendMetadata, AscendSFAMetadata)):
            return torch.as_tensor(self.attn_metadata.seq_lens_cpu, device=self.device)

        if isinstance(self.attn_metadata, AscendMLAMetadata):
            # AscendMLAMetadataBuilder.build_decode_metadata breaks `AscendMLAMetadata.seq_lens`
            # thus prefill metadata's seq_lens is preferentially used if available
            if self.attn_metadata.prefill is not None:
                return torch.as_tensor(self.attn_metadata.prefill.seq_lens, device=self.device)
            return torch.as_tensor(self.attn_metadata.seq_lens_cpu, device=self.device)

        for candidate in ["seq_lens_cpu", "seq_lens"]:
            if (s := getattr(self.attn_metadata, candidate)) is not None:
                return torch.as_tensor(s, device=self.device)

        raise ValueError(f"Cannot find seq_lens in attention metadata of type {type(self.attn_metadata)}.")

    @property
    def num_prefills(self) -> int:
        for candidate in ["num_prefills"]:
            if (num_prefills := getattr(self.attn_metadata, candidate)) is not None:
                return int(num_prefills)
        return 0

    @property
    def num_decodes(self) -> int:
        for candidate in ["num_decodes"]:
            if (num_decodes := getattr(self.attn_metadata, candidate)) is not None:
                return int(num_decodes)
        return 0

    @property
    def num_decode_tokens(self) -> int:
        for candidate in ["num_decode_tokens"]:
            if (num_decode_tokens := getattr(self.attn_metadata, candidate, None)) is not None:
                return int(num_decode_tokens)
        return 0

    @property
    def num_actual_tokens(self) -> int:
        """Return the number of actual tokens (excluding padding)."""
        for candidate in ["num_actual_tokens"]:
            if (num_actual_tokens := getattr(self.attn_metadata, candidate)) is not None:
                return int(num_actual_tokens)
        raise ValueError(f"Cannot find num_actual_tokens in attention metadata of type {type(self.attn_metadata)}.")


def get_nested_attr(obj: Any, /, *attrs: str, default: Any = None, raises: bool = False) -> Any:
    """Get/collect a nested attribute from an object.

    The attribute path is specified as a sequence of attribute names. If any attribute in the path is missing, the
    function returns the specified default value (which is None by default).

    Args:
        obj (Any): Root object.
        *attrs (str): Sequence of attribute names to traverse.
        default (Any, keyword-only, default=None): Default value to return if any attribute is missing.
        raises (bool, keyword-only, default=False): Whether to raise an error if any attribute is missing.

    Returns:
        Any: The resolved nested attribute.
    """
    current = obj
    for attr in attrs:
        if (current := getattr(current, attr, _MISSING)) is _MISSING:
            if raises:
                raise AttributeError(f"{type(obj).__name__} has no attribute {'.'.join(attrs)} (failed at {attr}).")
            return default
    return current


def get_dotted_attr(obj: Any, dotted_attr: str, /, *, default: Any = None, raises: bool = False) -> Any:
    """Get a nested attribute from an object using a dotted attribute string.

    This is a convenience wrapper around :meth:`_get_nested_attr` that allows specifying the attribute path as a single
    dotted string.

    Args:
        obj (Any): Root object.
        dotted_attr (str): Dotted attribute string, e.g., "foo.bar.baz" to access `obj.foo.bar.baz`.
        default (Any, keyword-only, default=None): Default value to return if any attribute is missing.
        raises (bool, keyword-only, default=False): Whether to raise an error if any attribute is missing.

    Returns:
        Any: The resolved nested attribute.
    """
    return get_nested_attr(obj, *dotted_attr.split("."), default=default, raises=raises)


class WeightGetterConfig(TypedDict):
    """Configuration dictionary for layer weight extraction in `get_layer_weights`.

    This class is written as a TypedDict for better type checking with `mypy` in the `xlite` module.
    """

    secondary_flattening: str | slice | None
    post_processor: Callable[[torch.Tensor], torch.Tensor] | None


def get_layer_weights(
    layers: Sequence[nn.Module],
    layer_attr: str,
    /,
    *,
    secondary_flattening: str | slice | None = None,
    post_processor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Extract specified weights from a sequence of layers with optional secondary flattening and post-processing.

    This function retrieves the specified attribute (e.g., "self_attn.q_proj.weight") from each layer in the provided
    sequence. If `secondary_flattening` is specified, it will further expand the retrieved attribute as a list and
    collect all items from these lists across layers. An optional `post_processor` can be applied to each retrieved
    tensor before returning the final list of weights.

    Args:
        layers (Sequence[nn.Module]): Sequence of layers to retrieve weights from.
        layer_attr (str): Dotted attribute string specifying the layer attribute to retrieve (`layers.[i].[layer_attr]`)
            , e.g., "self_attn.q_norm.weight".
        secondary_flattening (str | slice | None, optional): If specified, indicates that the retrieved layer attribute
            is a list of tensors and we need to further flatten it. The expansion can be specified as:

            - `str`: A dotted attribute string such that `layers.[i].[secondary_flattening]` gives the number of items
              to flatten for that layer.
            - `slice`: A slice specifying how to slice `layers.[i].[layer_attr]` and then flatten the sliced part.
            - `None`: No secondary flattening; `layers.[i].[layer_attr]` is directly collected.
        post_processor (Callable[[torch.Tensor], torch.Tensor] | None, optional): An optional function to apply to
            each retrieved tensor before returning the final list of weights.
        **kwargs: Additional keyword arguments for future extensions.

    Returns:
        list[torch.Tensor]: List of retrieved weights.
    """
    if not secondary_flattening:
        weights = [
            weight for layer in layers if (weight := get_dotted_attr(layer, layer_attr, default=None)) is not None
        ]
    elif isinstance(secondary_flattening, str):
        weights = [
            weight
            for layer in layers
            if (weight_lst := get_dotted_attr(layer, layer_attr, default=[])) is not None
            for weight in weight_lst[: get_dotted_attr(layer, secondary_flattening, default=0)]
        ]
    elif isinstance(secondary_flattening, slice):
        weights = [
            weight
            for layer in layers
            if (weight_lst := get_dotted_attr(layer, layer_attr, default=[])) is not None
            for weight in weight_lst[secondary_flattening]
        ]
    else:
        raise ValueError(
            f"Invalid type for secondary_flattening: {type(secondary_flattening)}. Expected str, slice, or None."
        )

    if not post_processor:
        return weights
    return [post_processor(weight) for weight in weights]
