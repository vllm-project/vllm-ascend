import threading
from collections import OrderedDict
from typing import Any

from vllm.logger import logger


class KeyLRUCache:
    def __init__(self, capacity: int, store_scheduler: Any):
        self.capacity = capacity
        self.store_scheduler = store_scheduler
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> int | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, gva: int) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = gva
                return
            self._cache[key] = gva
            self._evict_if_needed()

    def batch_get_and_alloc(
        self,
        keys: list[str],
        page_size_bytes: int,
    ) -> list[int | None]:
        if not keys:
            return []

        hit_gvas: list[int | None] = [None] * len(keys)
        miss_indices: list[int] = []
        miss_keys: list[str] = []

        with self._lock:
            for i, key in enumerate(keys):
                if key in self._cache:
                    self._cache.move_to_end(key)
                    hit_gvas[i] = self._cache[key]
                else:
                    miss_indices.append(i)
                    miss_keys.append(key)

        if miss_keys and self.store_scheduler is not None:
            new_gvas = self.store_scheduler.batch_alloc(
                miss_keys,
                [page_size_bytes for _ in range(len(miss_keys))],
            )
            with self._lock:
                for idx, key in enumerate(miss_keys):
                    self._cache[key] = new_gvas[idx]
                self._evict_if_needed()
            for idx, orig_i in enumerate(miss_indices):
                hit_gvas[orig_i] = new_gvas[idx]

        return hit_gvas

    def remove(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def remove_batch(self, keys: list[str]) -> None:
        with self._lock:
            for key in keys:
                self._cache.pop(key, None)

    def _evict_if_needed(self) -> None:
        evicted_keys: list[str] = []
        while len(self._cache) > self.capacity:
            key, gva = self._cache.popitem(last=False)
            evicted_keys.append(key)
        if evicted_keys and self.store_scheduler is not None:
            self.store_scheduler.remove_batch(evicted_keys)
            logger.debug("LRU evicted %d keys", len(evicted_keys))

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
