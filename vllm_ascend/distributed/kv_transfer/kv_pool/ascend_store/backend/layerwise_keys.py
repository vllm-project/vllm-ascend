# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound wire-key construction shared by writers and scheduler lookups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerwiseKeyBuilder:
    """Backend wire identity with every fixed character compiled before traffic."""

    prefixes_by_group_stage: tuple[tuple[str, ...], ...]
    head_suffixes: tuple[str, ...]

    @property
    def pp_size(self) -> int:
        return len(self.prefixes_by_group_stage[0])

    def make_full_key(self, group: int, block_hash: str, head: int, stage: int) -> str:
        return self.prefixes_by_group_stage[group][stage] + block_hash + self.head_suffixes[head]

    def bind_group(self, group: int, head: int, stage: int, num_heads: int) -> BoundLayerwiseKeys:
        prefixes = self.prefixes_by_group_stage[group]
        return BoundLayerwiseKeys(
            ((prefixes[stage], self.head_suffixes[head]),),
            tuple((prefix, suffix) for prefix in prefixes for suffix in self.head_suffixes[:num_heads]),
        )

    def make_hit_check_keys(self, group_id: int, block_hash: str, num_ranks: int) -> list[str]:
        """Require every saving head on every stage, in stage-major order."""
        return [
            self.make_full_key(group_id, block_hash, rank, stage)
            for stage in range(self.pp_size)
            for rank in range(num_ranks)
        ]


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
