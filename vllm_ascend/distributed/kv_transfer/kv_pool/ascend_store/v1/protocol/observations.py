"""Aggregate published cache events across Worker outputs."""

from vllm.distributed.kv_events import KVCacheEvent, KVConnectorKVEvents, KVEventAggregator


class StoreEvents(KVConnectorKVEvents):
    def __init__(self, events: list[KVCacheEvent], num_workers: int = 1) -> None:
        self._events = KVEventAggregator(num_workers)
        self._events.add_events(events)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._events.add_events(events)

    def increment_workers(self, count: int = 1) -> None:
        self._events.increment_workers(count)

    def clear_events(self) -> None:
        self._events.clear_events()
        self._events.reset_workers()

    def merge(self, other: KVConnectorKVEvents) -> "StoreEvents":
        self._events.add_events(other.get_all_events())
        self._events.increment_workers(other.get_number_of_workers())
        return self

    def aggregate(self) -> "StoreEvents":
        events = self._events.get_common_events()
        self._events.clear_events()
        self._events.add_events(events)
        self._events.reset_workers()
        return self

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._events.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._events.get_number_of_workers()
