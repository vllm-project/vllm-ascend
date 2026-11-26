# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Atomic similarity metadata operations, without tensor or device resources."""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol

from vllm_ascend.distributed.ec_transfer.cache_statistics import (
    CacheStatistics,
    CacheStatisticsAccumulator,
    CacheStatisticsSnapshot,
)
from vllm_ascend.distributed.ec_transfer.phash_index import PHASH_BITS, bands, hamming
from vllm_ascend.distributed.ec_transfer.similarity_cache_config import SimilarityCacheConfig


def sort_similarity_scores(scores: dict[str, float]) -> OrderedDict[str, float]:
    return OrderedDict(
        sorted(scores.items(), key=lambda item: (-item[1] if math.isfinite(item[1]) else math.inf, item[0]))
    )


@dataclass
class SimilarityCacheItem:
    phash: int
    ssim_similar_items: OrderedDict[str, float] = field(default_factory=OrderedDict)
    # Ownership information permits atomic eviction without parsing tensor keys.
    resized_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ssim_similar_items = sort_similarity_scores(self.ssim_similar_items)

    def snapshot(self) -> SimilarityCacheItem:
        result = SimilarityCacheItem(self.phash)
        # Scores are ranked on writes, not on every scheduler/RPC read.
        result.ssim_similar_items = self.ssim_similar_items.copy()
        result.resized_keys = list(self.resized_keys)
        return result


class SimilarityIndex(Protocol):
    def batch_get_entries(self, mm_hashes: list[str]) -> list[SimilarityCacheItem | None]: ...

    def batch_register_phashes(self, entries: list[tuple[str, int, str]]) -> None: ...

    def batch_recall(self, phashes: list[int]) -> list[list[str]]: ...

    def record_similarity_scores(self, mm_hash: str, scores: list[tuple[float, str]]) -> None: ...

    def report_cache_statistics(self, scheduler_id: str, counts: CacheStatistics) -> CacheStatisticsSnapshot: ...

    def close(self) -> None: ...


class LocalSimilarityIndex:
    """One owner for the maps and lock; callers receive detached snapshots.

    Used directly for local matching, or owned by the standalone shared service.
    Registration only publishes metadata; data availability is checked separately.
    """

    def __init__(self, config: SimilarityCacheConfig) -> None:
        self._config = config
        self.similarity_cache: dict[str, SimilarityCacheItem] = {}
        self.phash_to_resized_mm_hashes: dict[int, list[str]] = {}
        self._phash_band_index: dict[tuple[int, int], set[int]] = {}
        self._lock = Lock()
        self._statistics = CacheStatisticsAccumulator()

    def batch_get_entries(self, mm_hashes: list[str]) -> list[SimilarityCacheItem | None]:
        with self._lock:
            return [
                None if (item := self.similarity_cache.get(mm_hash)) is None else item.snapshot()
                for mm_hash in mm_hashes
            ]

    def batch_register_phashes(self, entries: list[tuple[str, int, str]]) -> None:
        for mm_hash, phash, resized_key in entries:
            if not isinstance(mm_hash, str) or not mm_hash or not isinstance(resized_key, str) or not resized_key:
                raise ValueError("pHash registration requires nonempty image and resized keys")
            if isinstance(phash, bool) or not isinstance(phash, int) or not 0 <= phash < 1 << PHASH_BITS:
                raise ValueError("pHash must be an unsigned 64-bit integer")
        with self._lock:
            for mm_hash, phash, resized_key in entries:
                item = self.similarity_cache.get(mm_hash)
                if item is None:
                    item = SimilarityCacheItem(phash)
                    self.similarity_cache[mm_hash] = item
                # Concurrent computations of the same image use the first hash.
                phash = item.phash
                if resized_key not in item.resized_keys:
                    item.resized_keys.append(resized_key)
                keys = self.phash_to_resized_mm_hashes.get(phash)
                if keys is None:
                    keys = []
                    self.phash_to_resized_mm_hashes[phash] = keys
                    if self._config.max_hamming > 0:
                        for band in bands(phash):
                            self._phash_band_index.setdefault(band, set()).add(phash)
                if resized_key not in keys:
                    keys.append(resized_key)
                self._evict_if_needed()

    def batch_recall(self, phashes: list[int]) -> list[list[str]]:
        with self._lock:
            recalled = {phash: self._recall(phash) for phash in dict.fromkeys(phashes)}
            return [list(recalled[phash]) for phash in phashes]

    def _recall(self, query: int) -> list[str]:
        if self._config.max_hamming == 0:
            return list(self.phash_to_resized_mm_hashes.get(query, ()))
        candidates: set[int] = set()
        for band in bands(query):
            candidates.update(self._phash_band_index.get(band, ()))
        return [
            key
            for phash in sorted(candidates)
            if hamming(query, phash) <= self._config.max_hamming
            for key in self.phash_to_resized_mm_hashes[phash]
        ]

    def record_similarity_scores(self, mm_hash: str, scores: list[tuple[float, str]]) -> None:
        with self._lock:
            current = self.similarity_cache.get(mm_hash)
            if current is None:
                return
            current_scores = dict(current.ssim_similar_items)
            for score, candidate in scores:
                if candidate == mm_hash:
                    continue
                current_scores[candidate] = score
                other = self.similarity_cache.get(candidate)
                if other is not None:
                    other_scores = dict(other.ssim_similar_items)
                    other_scores[mm_hash] = score
                    other.ssim_similar_items = sort_similarity_scores(other_scores)
            current.ssim_similar_items = sort_similarity_scores(current_scores)

    def _evict_if_needed(self) -> None:
        if len(self.similarity_cache) < self._config.max_capacity * self._config.eviction_threshold:
            return
        target = int(self._config.max_capacity * self._config.lower_watermark)
        victims = random.sample(list(self.similarity_cache), len(self.similarity_cache) - target)
        for mm_hash in victims:
            item = self.similarity_cache.pop(mm_hash)
            removed = set(item.resized_keys)
            keys = [key for key in self.phash_to_resized_mm_hashes[item.phash] if key not in removed]
            if keys:
                self.phash_to_resized_mm_hashes[item.phash] = keys
                continue
            del self.phash_to_resized_mm_hashes[item.phash]
            if self._config.max_hamming > 0:
                for band in bands(item.phash):
                    bucket = self._phash_band_index[band]
                    bucket.discard(item.phash)
                    if not bucket:
                        del self._phash_band_index[band]
        # Scores may still reference evicted images. Their embeddings are
        # checked against Memcache before use; metadata eviction deletes no data.

    def report_cache_statistics(self, scheduler_id: str, counts: CacheStatistics) -> CacheStatisticsSnapshot:
        return self._statistics.report(scheduler_id, counts)

    def close(self) -> None:
        """No external resources are owned by the local implementation."""
