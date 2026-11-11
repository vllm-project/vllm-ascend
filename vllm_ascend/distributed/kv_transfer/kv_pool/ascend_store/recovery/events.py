# SPDX-License-Identifier: Apache-2.0
# Protocol port from vLLM v0.33.0, mooncake/store/connector.py.
from collections import Counter
from collections.abc import Sequence

from vllm.distributed.kv_events import BlockStored, KVCacheEvent, KVConnectorKVEvents


class StoreKVEvents(KVConnectorKVEvents):
    """KV event aggregation for MooncakeStoreConnector."""

    def __init__(
        self,
        num_workers: int,
        group_tp_replication_factors: Sequence[int] = (1,),
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than zero.")
        if any(factor <= 0 for factor in group_tp_replication_factors):
            raise ValueError("TP replication factors must be greater than zero.")
        self._event_counter: Counter[KVCacheEvent] = Counter()
        self._num_workers = num_workers
        self._group_tp_replication_factors = tuple(group_tp_replication_factors)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        if not isinstance(events, list):
            raise TypeError("events must be a list of KVCacheEvent.")
        self._event_counter.update(events)

    def _replication_factor(self, event: KVCacheEvent) -> int:
        if not isinstance(event, BlockStored) or event.group_idx is None:
            return 1
        return self._group_tp_replication_factors[event.group_idx]

    def _is_common_event(self, event: KVCacheEvent, count: int) -> bool:
        return count * self._replication_factor(event) >= self._num_workers

    def aggregate(self) -> "StoreKVEvents":
        common_events = self.pop_common_events()
        self._event_counter.clear()
        self._event_counter.update(common_events)
        self._num_workers = 1
        return self

    def pop_common_events(self) -> list[KVCacheEvent]:
        common_events = [event for event, count in self._event_counter.items() if self._is_common_event(event, count)]
        for event in common_events:
            del self._event_counter[event]
        return common_events

    def has_events(self) -> bool:
        return bool(self._event_counter)

    def increment_workers(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive.")
        self._num_workers += count

    def get_all_events(self) -> list[KVCacheEvent]:
        return list(self._event_counter.elements())

    def get_number_of_workers(self) -> int:
        return self._num_workers

    def clear_events(self) -> None:
        self._event_counter.clear()
        self._num_workers = 1

    def __repr__(self) -> str:
        return f"<StoreKVEvents events={self.get_all_events()}>"
