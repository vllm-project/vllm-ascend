# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-light helpers for vLLM-compatible config dataclasses."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.fields import Field as PydanticField
from typing_extensions import dataclass_transform

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = Any

ConfigT = TypeVar("ConfigT", bound=DataclassInstance)


@overload
@dataclass_transform(field_specifiers=(PydanticField,))
def config(cls: type[ConfigT]) -> type[ConfigT]: ...


@overload
@dataclass_transform(field_specifiers=(PydanticField,))
def config(*, config: ConfigDict | None = None, **kwargs: Any) -> Callable[[type[ConfigT]], type[ConfigT]]: ...


@dataclass_transform(field_specifiers=(PydanticField,))
def config(
    cls: type[ConfigT] | None = None,
    *,
    config: ConfigDict | None = None,
    **kwargs: Any,
) -> type[ConfigT] | Callable[[type[ConfigT]], type[ConfigT]]:
    """Create a vLLM-compatible config dataclass without importing vllm.config.

    This mirrors ``vllm.config.utils.config`` while avoiding that module's
    package-initialization cycle during vLLM platform discovery.
    """
    merged_config = ConfigDict(extra="forbid")
    if config is not None:
        merged_config.update(config)

    def decorator(config_cls: type[ConfigT]) -> type[ConfigT]:
        return dataclass(config_cls, config=merged_config, **kwargs)  # type: ignore[return-value]

    return decorator if cls is None else decorator(cls)
