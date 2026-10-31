# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Map explicitly described cache slices to pool object byte ranges.

Descriptions use global layer/component identities and logical token/head
coordinates. Rank and group numbers are resolved by the caller, not compared
between P and D. A component can represent K, V, an MLA latent, or scale data;
only byte-compatible components may share a format_id.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.executor import PoolReadRange


@dataclass(frozen=True)
class CacheSlice:
    layer: str
    component: str
    format_id: str
    token_start: int
    token_end: int
    head_start: int
    head_end: int
    vector_bytes: int
    token_stride: int
    head_stride: int

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.layer, self.component, self.format_id)):
            raise ValueError("Cache slices require layer, component and byte-format identities")
        for name in (
            "token_start",
            "token_end",
            "head_start",
            "head_end",
            "vector_bytes",
            "token_stride",
            "head_stride",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"Cache slice {name} must be an integer")
        if not 0 <= self.token_start < self.token_end or not 0 <= self.head_start < self.head_end:
            raise ValueError("Cache slice logical intervals must be nonempty and nonnegative")
        if min(self.vector_bytes, self.token_stride, self.head_stride) <= 0:
            raise ValueError("Cache slice byte widths and strides must be positive")
        # Allow HND, NHD, or padded variants; reject overlapping element views.
        tokens = self.token_end - self.token_start
        heads = self.head_end - self.head_start
        hnd = (
            self.token_stride >= self.vector_bytes
            and self.head_stride >= (tokens - 1) * self.token_stride + self.vector_bytes
        )
        nhd = (
            self.head_stride >= self.vector_bytes
            and self.token_stride >= (heads - 1) * self.head_stride + self.vector_bytes
        )
        if not (hnd or nhd):
            raise ValueError("Cache slice strides overlap logical vectors")

    def offset(self, token: int, head: int) -> int:
        return (token - self.token_start) * self.token_stride + (head - self.head_start) * self.head_stride

    @property
    def span_bytes(self) -> int:
        return self.offset(self.token_end - 1, self.head_end - 1) + self.vector_bytes


@dataclass(frozen=True)
class SourceCacheSlice:
    cache: CacheSlice
    key: str
    object_size: int
    object_offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Source slice requires a pool key")
        if type(self.object_offset) is not int or self.object_offset < 0:
            raise ValueError("Source slice offset must be nonnegative")
        if type(self.object_size) is not int or self.object_size < self.object_offset + self.cache.span_bytes:
            raise ValueError("Source cache slice exceeds its pool object")


@dataclass(frozen=True)
class TargetCacheSlice:
    cache: CacheSlice
    address: int

    def __post_init__(self) -> None:
        if type(self.address) is not int or self.address <= 0:
            raise ValueError("Target cache slice requires a positive registered address")


def _subtract(rect: tuple[int, int, int, int], overlap: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    t0, t1, h0, h1 = rect
    a, b, c, d = overlap
    pieces = ((t0, a, h0, h1), (b, t1, h0, h1), (a, b, h0, c), (a, b, d, h1))
    return [piece for piece in pieces if piece[0] < piece[1] and piece[2] < piece[3]]


class _SourceIntervals:
    """Prune unrelated token pages without changing source replica priority."""

    def __init__(self, sources: list[SourceCacheSlice]):
        self.ordered = sorted(enumerate(sources), key=lambda item: item[1].cache.token_start)
        self.starts = [source.cache.token_start for _, source in self.ordered]
        self.prefix_ends = []
        end = 0
        for _, source in self.ordered:
            end = max(end, source.cache.token_end)
            self.prefix_ends.append(end)

    def overlapping(self, start: int, end: int) -> list[SourceCacheSlice]:
        stop = bisect_left(self.starts, end)
        first = bisect_right(self.prefix_ends, start, hi=stop)
        # A long earlier interval can straddle shorter later intervals. Prefix
        # maxima preserve it, and this filter removes intervening expired pages.
        candidates = [item for item in self.ordered[first:stop] if item[1].cache.token_end > start]
        # Sorting by the original ordinal preserves first-available-replica
        # semantics, even when source pages arrived in a different token order.
        return [source for _, source in sorted(candidates)]


class RestorePlanner:
    def build(self, sources: list[SourceCacheSlice], targets: list[TargetCacheSlice]) -> list[PoolReadRange]:
        """Build a fully covered plan, choosing the first available replica.

        A caller describes CP stripes as their actual contiguous token slices.
        Request-lifetime/recurrent state must be validated before it is exposed
        as a slice; this planner does not equate such state with token KV.
        """
        by_component: dict[tuple[str, str, str, int], list[SourceCacheSlice]] = {}
        sizes: dict[str, int] = {}
        for source in sources:
            if sizes.setdefault(source.key, source.object_size) != source.object_size:
                raise ValueError(f"Inconsistent source object size: {source.key}")
            cache = source.cache
            by_component.setdefault((cache.layer, cache.component, cache.format_id, cache.vector_bytes), []).append(
                source
            )
        indices = {key: _SourceIntervals(component) for key, component in by_component.items()}
        plan: list[PoolReadRange] = []
        for target in targets:
            local = target.cache
            remaining = [(local.token_start, local.token_end, local.head_start, local.head_end)]
            index = indices.get((local.layer, local.component, local.format_id, local.vector_bytes))
            candidates = [] if index is None else index.overlapping(local.token_start, local.token_end)
            for source in candidates:
                remote = source.cache
                next_remaining = []
                for rect in remaining:
                    overlap = (
                        max(rect[0], remote.token_start),
                        min(rect[1], remote.token_end),
                        max(rect[2], remote.head_start),
                        min(rect[3], remote.head_end),
                    )
                    if overlap[0] >= overlap[1] or overlap[2] >= overlap[3]:
                        next_remaining.append(rect)
                        continue
                    next_remaining.extend(_subtract(rect, overlap))
                    # Traverse in destination-contiguous order. Coalescing is
                    # allowed only if both source and destination are adjacent.
                    width = local.vector_bytes
                    if local.token_stride == remote.token_stride == width:
                        vectors = (
                            (overlap[0], h, (overlap[1] - overlap[0]) * width) for h in range(overlap[2], overlap[3])
                        )
                    elif local.head_stride == remote.head_stride == width:
                        vectors = (
                            (t, overlap[2], (overlap[3] - overlap[2]) * width) for t in range(overlap[0], overlap[1])
                        )
                    elif local.token_stride <= local.head_stride:
                        vectors = (
                            (t, h, width) for h in range(overlap[2], overlap[3]) for t in range(overlap[0], overlap[1])
                        )
                    else:
                        vectors = (
                            (t, h, width) for t in range(overlap[0], overlap[1]) for h in range(overlap[2], overlap[3])
                        )
                    for token, head, size in vectors:
                        item = PoolReadRange(
                            source.key,
                            source.object_size,
                            source.object_offset + remote.offset(token, head),
                            target.address + local.offset(token, head),
                            size,
                        )
                        if plan and self._adjacent(plan[-1], item):
                            previous = plan.pop()
                            item = PoolReadRange(
                                item.key,
                                item.object_size,
                                previous.source_offset,
                                previous.destination,
                                previous.size + item.size,
                            )
                        plan.append(item)
                remaining = next_remaining
                if not remaining:
                    break
            if remaining:
                raise ValueError(
                    f"Incomplete pool cache coverage for {local.layer}/{local.component}: {remaining}; "
                    f"required format={local.format_id}, vector_bytes={local.vector_bytes}"
                )
        return plan

    @staticmethod
    def _adjacent(left: PoolReadRange, right: PoolReadRange) -> bool:
        return (
            left.key == right.key
            and left.object_size == right.object_size
            and left.source_offset + left.size == right.source_offset
            and left.destination + left.size == right.destination
        )
