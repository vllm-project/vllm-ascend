"""Fixed prefix/suffix wire names shared by the concrete Backend codecs."""

from collections.abc import Sequence
from dataclasses import dataclass

from ...backend.layerwise_keys import BoundLayerwiseKeys
from ..infrastructure.mapping.local_layout import ObjectCoordinate


@dataclass(frozen=True, slots=True)
class WireKeyBuilder:
    """Own compiled wire strings; bind group coordinates directly, without an adapter."""

    prefixes_by_group_stage: tuple[tuple[str, ...], ...]
    head_suffixes: tuple[str, ...]

    def bind(
        self, group_id: int, coordinates: Sequence[ObjectCoordinate], *, lookup_head_count: int
    ) -> BoundLayerwiseKeys:
        prefixes = self.prefixes_by_group_stage[group_id]
        return BoundLayerwiseKeys(
            tuple((prefixes[position.pp_rank], self.head_suffixes[position.head_rank]) for position in coordinates),
            tuple((prefix, suffix) for prefix in prefixes for suffix in self.head_suffixes[:lookup_head_count]),
        )
