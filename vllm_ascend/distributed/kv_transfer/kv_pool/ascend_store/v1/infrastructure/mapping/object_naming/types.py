"""Naming binds to the same local object order as memory and object layouts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from ..types import KeysByPartition

if TYPE_CHECKING:
    from ..local_layout import ObjectPartitionCoordinate


class KeyNamer(Protocol):
    """Fixed naming formulas; each local axis pairs with one LocalLayout object."""

    def keys(self, hashes: Sequence[bytes | str]) -> KeysByPartition: ...

    def lookup_keys(self, hashes: Sequence[bytes | str]) -> KeysByPartition: ...


class KeyBuilder(Protocol):
    """Bind fixed wire rules directly to numeric coordinates, without local memory details."""

    def bind(
        self, group_id: int, coordinates: Sequence[ObjectPartitionCoordinate], *, lookup_head_count: int
    ) -> KeyNamer: ...
