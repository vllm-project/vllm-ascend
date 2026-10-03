"""Define the fixed temporal policy selected for one KV Pool Runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LoadScheduleKind(str, Enum):
    """Fixed Load scheduling policy owned by the Timeline."""

    SYNC = "synchronous"
    ASYNC = "asynchronous"
    LAYERWISE = "layerwise"


class StoreScheduleKind(str, Enum):
    """Fixed Store scheduling policy owned by the Timeline."""

    ASYNC = "asynchronous"
    LAYERWISE = "layerwise"


@dataclass(frozen=True, slots=True)
class KVPoolSchedule:
    """Time-axis policy used to construct the Runtime state machines."""

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
