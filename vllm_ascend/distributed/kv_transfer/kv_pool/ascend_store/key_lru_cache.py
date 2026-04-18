import threading
from collections import OrderedDict
from typing import Any

from vllm.logger import logger


class KeyLRUCache:
    def __init__(self, capacity: int, store_scheduler: Any):
        self.capacity = capacity
        self.store_scheduler = store_scheduler
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._block_hash_to_keys: OrderedDict[bytes, list[str]] = OrderedDict()
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
        block_hash_groups: dict[bytes, list[str]] | None = None,
    ) -> list[int | None]:
        if not keys:
            return []

        if block_hash_groups:
            return self._batch_get_and_alloc_by_block(
                keys, page_size_bytes, block_hash_groups)

        return self._batch_get_and_alloc_by_key(keys, page_size_bytes)

    def _batch_get_and_alloc_by_block(
        self,
        keys: list[str],
        page_size_bytes: int,
        block_hash_groups: dict[bytes, list[str]],
    ) -> list[int | None]:
        key_to_index: dict[str, int] = {key: i for i, key in enumerate(keys)}
        gvas: list[int | None] = [None] * len(keys)
        miss_keys: list[str] = []

        with self._lock:
            for block_hash, group_keys in block_hash_groups.items():
                if block_hash in self._block_hash_to_keys:
                    self._block_hash_to_keys.move_to_end(block_hash)
                    cached_keys = self._block_hash_to_keys[block_hash]
                    for ck, gk in zip(cached_keys, group_keys):
                        idx = key_to_index[gk]
                        gvas[idx] = self._cache.get(ck)
                        if ck in self._cache:
                            self._cache.move_to_end(ck)
                else:
                    for gk in group_keys:
                        idx = key_to_index[gk]
                        if gk in self._cache:
                            self._cache.move_to_end(gk)
                            gvas[idx] = self._cache[gk]
                        else:
                            miss_keys.append(gk)

        if miss_keys and self.store_scheduler is not None:
            new_gvas = self.store_scheduler.batch_alloc(
                miss_keys,
                [page_size_bytes for _ in range(len(miss_keys))],
            )
            with self._lock:
                for idx, key in enumerate(miss_keys):
                    self._cache[key] = new_gvas[idx]
                for block_hash, group_keys in block_hash_groups.items():
                    if block_hash not in self._block_hash_to_keys:
                        self._block_hash_to_keys[block_hash] = group_keys
                    self._block_hash_to_keys.move_to_end(block_hash)
                self._evict_if_needed()
            for idx, key in enumerate(miss_keys):
                gvas[key_to_index[key]] = new_gvas[idx]
        else:
            with self._lock:
                for block_hash, group_keys in block_hash_groups.items():
                    if block_hash not in self._block_hash_to_keys:
                        self._block_hash_to_keys[block_hash] = group_keys
                    self._block_hash_to_keys.move_to_end(block_hash)

        return gvas

    def _batch_get_and_alloc_by_key(
        self,
        keys: list[str],
        page_size_bytes: int,
    ) -> list[int | None]:
        gvas: list[int | None] = [None] * len(keys)
        miss_indices: list[int] = []
        miss_keys: list[str] = []

        with self._lock:
            for i, key in enumerate(keys):
                if key in self._cache:
                    self._cache.move_to_end(key)
                    gvas[i] = self._cache[key]
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
                gvas[orig_i] = new_gvas[idx]

        return gvas

    def get_block(self, block_hash: bytes) -> list[int | None] | None:
        with self._lock:
            if block_hash not in self._block_hash_to_keys:
                return None
            self._block_hash_to_keys.move_to_end(block_hash)
            keys = self._block_hash_to_keys[block_hash]
            return [self._cache.get(key) for key in keys]

    def has_block(self, block_hash: bytes) -> bool:
        with self._lock:
            return block_hash in self._block_hash_to_keys

    def remove_block(self, block_hash: bytes) -> list[str] | None:
        with self._lock:
            if block_hash not in self._block_hash_to_keys:
                return None
            keys = self._block_hash_to_keys.pop(block_hash)
            for key in keys:
                self._cache.pop(key, None)
            return keys

    def remove_blocks(self, block_hashes: list[bytes]) -> list[str]:
        all_removed_keys: list[str] = []
        with self._lock:
            for block_hash in block_hashes:
                if block_hash in self._block_hash_to_keys:
                    keys = self._block_hash_to_keys.pop(block_hash)
                    for key in keys:
                        self._cache.pop(key, None)
                    all_removed_keys.extend(keys)
        return all_removed_keys

    def remove(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def remove_batch(self, keys: list[str]) -> None:
        with self._lock:
            for key in keys:
                self._cache.pop(key, None)

    def _evict_if_needed(self) -> None:
        evicted_keys: list[str] = []
        while len(self._block_hash_to_keys) > self.capacity:
            block_hash, keys = self._block_hash_to_keys.popitem(last=False)
            for key in keys:
                self._cache.pop(key, None)
            evicted_keys.extend(keys)
        if evicted_keys and self.store_scheduler is not None:
            self.store_scheduler.remove_batch(evicted_keys)
            logger.debug("LRU evicted %d keys from block hash entries", len(evicted_keys))

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
