"""Fixed prefix/suffix wire names shared by the concrete Backend codecs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..infrastructure.mapping.local_layout import ObjectPartitionCoordinate


@dataclass(frozen=True, slots=True)
class WireKeyBuilder:
    """Own compiled wire strings; bind group coordinates directly, without an adapter."""

    prefixes_by_group_stage: tuple[tuple[str, ...], ...]
    head_suffixes: tuple[str, ...]

    def bind(
        self, group_id: int, coordinates: Sequence[ObjectPartitionCoordinate], *, lookup_head_count: int
    ) -> BoundLayerwiseKeys:
        prefixes = self.prefixes_by_group_stage[group_id]
        return BoundLayerwiseKeys(
            tuple((prefixes[position.pp_rank], self.head_suffixes[position.head_rank]) for position in coordinates),
            tuple((prefix, suffix) for prefix in prefixes for suffix in self.head_suffixes[:lookup_head_count]),
        )


@dataclass(frozen=True, slots=True)
class BoundLayerwiseKeys:
    """Local and Lookup key axes; no topology, configuration or mutable layout remains."""

    local_parts: tuple[tuple[str, str], ...]
    lookup_parts: tuple[tuple[str, str], ...]

    def keys(self, hashes: Sequence[bytes | str]) -> tuple[tuple[str, ...], ...]:
        return self._map(self.local_parts, hashes)

    def lookup_keys(self, hashes: Sequence[bytes | str]) -> tuple[tuple[str, ...], ...]:
        return self._map(self.lookup_parts, hashes)

    @staticmethod
    def _map(parts: tuple[tuple[str, str], ...], hashes: Sequence[bytes | str]) -> tuple[tuple[str, ...], ...]:
        encoded = tuple(value.hex() if isinstance(value, bytes) else value for value in hashes)
        return tuple(tuple(prefix + value + suffix for value in encoded) for prefix, suffix in parts)
