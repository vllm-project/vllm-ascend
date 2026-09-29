"""Define the fixed scheduling spec of one KV Pool program."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LoadScheduleKind(str, Enum):
    """Fixed Load scheduling policy compiled into a KV Pool program."""

    SYNC = "synchronous"
    ASYNC = "asynchronous"
    LAYERWISE = "layerwise"


class StoreScheduleKind(str, Enum):
    """Fixed Store scheduling policy compiled into a KV Pool program."""

    ASYNC = "asynchronous"
    LAYERWISE = "layerwise"


@dataclass(frozen=True, slots=True)
class KVPoolSchedule:
    """Time-axis rules compiled into one KV Pool program."""

    load_kind: LoadScheduleKind
    store_kind: StoreScheduleKind | None
    layerwise_prefetch_layers: int

    @property
    def requires_layerwise_backend(self) -> bool:
        return self.load_kind is LoadScheduleKind.LAYERWISE or self.store_kind is StoreScheduleKind.LAYERWISE

    @property
    def store_enabled(self) -> bool:
        return self.store_kind is not None

    @property
    def fences_store_on_finish(self) -> bool:
        return self.store_kind is StoreScheduleKind.LAYERWISE
