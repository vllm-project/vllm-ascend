"""Naming binds to the same local object order as memory and object layouts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from ..types import KeysByObject

if TYPE_CHECKING:
    from ..local_layout import ObjectCoordinate


class KeyNamer(Protocol):
    """Fixed naming formulas; each local axis pairs with one LocalLayout object."""

    def keys(self, hashes: Sequence[bytes | str]) -> KeysByObject: ...

    def lookup_keys(self, hashes: Sequence[bytes | str]) -> KeysByObject: ...


class KeyBuilder(Protocol):
    """Bind fixed wire rules directly to numeric coordinates, without local memory details."""

    def bind(self, group_id: int, coordinates: Sequence[ObjectCoordinate], *, lookup_head_count: int) -> KeyNamer: ...
