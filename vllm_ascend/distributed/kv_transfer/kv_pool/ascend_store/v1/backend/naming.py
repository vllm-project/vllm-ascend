"""Fixed prefix/suffix wire names shared by the concrete Backend codecs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...metadata import KeyMetadata


@dataclass(frozen=True, slots=True)
class WireKeyBuilder:
    """Own compiled wire strings; bind group coordinates directly, without an adapter."""

    prefixes_by_group_stage: tuple[tuple[str, ...], ...]
    head_suffixes: tuple[str, ...]

    def __call__(self, metadata: KeyMetadata) -> tuple[str, str]:
        return (
            self.prefixes_by_group_stage[metadata.kv_cache_group_id][metadata.pp_rank],
            self.head_suffixes[metadata.head_or_tp_rank],
        )

    def lookup_parts(self, group_id: int, head_count: int) -> tuple[tuple[str, str], ...]:
        # These wire formats share objects across DCP ranks; do not expand DCP.
        return tuple(
            (prefix, suffix)
            for prefix in self.prefixes_by_group_stage[group_id]
            for suffix in self.head_suffixes[:head_count]
        )
