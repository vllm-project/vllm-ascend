# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""pHash recall and SSIM matching for the multimodal encoder cache."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch
from vllm.logger import logger
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_ascend.distributed.ec_transfer.memcache_resized_tensor_store import (
    MemcacheResizedTensorStore,
)
from vllm_ascend.distributed.ec_transfer.phash import compute_phash, hamming
from vllm_ascend.distributed.ec_transfer.similarity_cache_config import (
    SimilarityCacheConfig,
    _MAX_SIMILARITY_COMPUTE_WORKERS,
    _PARALLEL_SIMILARITY_COMPUTE_THRESHOLD,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)


@dataclass
class SimilarityCacheItem:
    """Matcher-lifetime pHash and ranked SSIM matches for one image."""

    # Perceptual hash calculated from the image pixel tensor.
    phash: int
    # candidate mm_hash -> SSIM score, kept in descending score order.
    ssim_similar_items: OrderedDict[str, float] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        self.ssim_similar_items = OrderedDict(
            sorted(
                self.ssim_similar_items.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )


class PhashSSIMCacheMatcher:
    """Recall and verify reusable encoder-cache entries by pixel similarity."""

    def __init__(
        self,
        backend: MemcacheBackend,
        config: SimilarityCacheConfig,
        *,
        merge_size: int,
        model_id: str,
    ) -> None:
        if not config.enabled:
            raise ValueError("PhashSSIMCacheMatcher requires similarity_cache.enabled=true")

        # Scheduler-side Memcache client used for candidate availability checks
        # and shared with the resized-tensor store for data reads and writes.
        self._backend = backend
        # Validated similarity-cache settings, including pHash recall distance
        # and the SSIM acceptance threshold.
        self._config = config
        # Vision patch merge factor used when converting resized patch tensors
        # into the gray planes consumed by pHash and SSIM.
        self._merge_size = merge_size
        # Reuse CPU workers across pHash recall and SSIM scoring. The two
        # phases are awaited sequentially, so one pool avoids idle duplicates.
        self._similarity_compute_executor = ThreadPoolExecutor(
            max_workers=_MAX_SIMILARITY_COMPUTE_WORKERS,
            thread_name_prefix="ec-similarity-compute",
        )
        # Matcher-lifetime cache keyed by the original image mm_hash. SSIM
        # matches are ranked once when written, so the first qualified entry is
        # always the best known match for the image.
        self.similarity_cache: dict[str, SimilarityCacheItem] = {}
        # Reverse pHash lookup containing complete resized-tensor keys. Each
        # key carries the original mm_hash and the per-image grid, so recalled
        # tensors no longer require a separate metadata map.
        self.phash_to_resized_mm_hashes: dict[int, list[str]] = {}
        # Protects matcher-lifetime similarity state shared by pHash worker
        # threads, including similarity_cache and phash_to_resized_mm_hashes.
        self._similarity_state_lock = Lock()
        # Persistence adapter for model-, shape-, and dtype-specific resized
        # tensor keys.
        self._tensor_store = MemcacheResizedTensorStore(
            backend,
            merge_size,
            model_id,
            config.resized_image_recv_buffer_size,
        )

    def get_cached_similarity_match(self, mm_hash: str) -> str | None:
        """Return the highest-ranked cached SSIM match above the threshold."""
        with self._similarity_state_lock:
            similarity_item = self.similarity_cache.get(mm_hash)
            if similarity_item is None:
                return None
            for candidate_mm_hash, score in similarity_item.ssim_similar_items.items():
                if score > self._config.ssim_threshold:
                    return candidate_mm_hash
            return None

    def get_cached_phash(self, mm_hash: str) -> int | None:
        """Return the pHash prepared by batch prewarming, if available."""
        with self._similarity_state_lock:
            cache_item = self.similarity_cache.get(mm_hash)
            return None if cache_item is None else cache_item.phash

    def find_resized_mm_hashes_by_hamming(
        self,
        query_phash: int,
    ) -> list[str] | None:
        candidate_mm_hashes_by_hamming: list[str] = []
        with self._similarity_state_lock:
            phash_items = list(self.phash_to_resized_mm_hashes.items())
        for candidate_phash, candidate_mm_hashes in sorted(phash_items):
            if hamming(query_phash, candidate_phash) <= self._config.max_hamming:
                candidate_mm_hashes_by_hamming.extend(candidate_mm_hashes)
        return candidate_mm_hashes_by_hamming or None

    def find_and_cache_matched_ssim(
        self,
        mm_hash: str,
        current_feature: Any,
        phash_matched_mm_hashes: list[str],
    ) -> str | None:
        current_resized_tensor = extract_resized_tensor(current_feature)
        current_image_grid = extract_image_grid(current_feature)
        if current_resized_tensor is None or current_image_grid is None:
            logger.debug(
                "EC similarity match skipped: current=%s reason=query_data_unavailable",
                mm_hash,
            )
            return None

        ssim_score_list = self._get_ssim_score_list(
            mm_hash,
            phash_matched_mm_hashes,
            current_image_grid,
            current_resized_tensor,
        )

        if not ssim_score_list:
            logger.debug(
                "EC similarity match: current=%s result=no_ssim_hit threshold=%.6f",
                mm_hash,
                self._config.ssim_threshold,
            )
            return None
        self._record_similarity_score(mm_hash, ssim_score_list)
        qualified_match = ssim_score_list[0]
        score, candidate_mm_hash = qualified_match
        if score >= self._config.ssim_threshold:
            logger.debug(
                "EC similarity match success: current=%s candidate=%s ssim=%.6f",
                mm_hash,
                candidate_mm_hash,
                score,
            )
            return candidate_mm_hash
        return None

    def batch_ensure_similarity_cache_available(
        self,
        exact_match_missing_mm_hashes: list[str],
        exact_match_missing_features: list[Any],
    ) -> None:
        """Prewarm all keys needed by cached SSIM hits and pHash candidates."""
        if len(exact_match_missing_mm_hashes) != len(exact_match_missing_features):
            raise ValueError(
                "batch_ensure_similarity_cache_available requires the same number of mm_hashes "
                "and missing_features, got "
                f"{len(exact_match_missing_mm_hashes)} and "
                f"{len(exact_match_missing_features)}"
            )

        (
            ssim_exist_mm_hash_list,
            mm_hash_to_phash_list,
            phash_match_failed_mm_hash_list,
        ) = self._classify_similarity_cache_entries(exact_match_missing_mm_hashes)

        mm_hash_to_feature_map = dict(
            zip(exact_match_missing_mm_hashes, exact_match_missing_features)
        )
        # Phash match failed, calculate phash and save to memcache and cache
        if phash_match_failed_mm_hash_list:
            self._compute_and_store_missing_phashes(
                phash_match_failed_mm_hash_list,
                mm_hash_to_feature_map,
            )

        prewarm_keys = list(dict.fromkeys(ssim_exist_mm_hash_list))
        # If Phash found, process by hamming search result
        if mm_hash_to_phash_list:
            (
                hamming_not_match_mm_hashes,
                candidate_resized_keys_by_hamming,
            ) = self._collect_candidate_resized_mm_hashes_by_hamming(
                mm_hash_to_phash_list
            )
            prewarm_keys.extend(dict.fromkeys(candidate_resized_keys_by_hamming))
            prewarm_keys.extend(
                dict.fromkeys(
                    candidate_mm_hash
                    for resized_key in candidate_resized_keys_by_hamming
                    if (
                        candidate_mm_hash
                        := self._tensor_store.get_mm_hash_from_resized_key(
                            resized_key
                        )
                    )
                    is not None
                )
            )
            if hamming_not_match_mm_hashes:
                logger.debug("EC similarity batch not prewarm for they are not exist: hamming_not_match_mm_hashes=%r",
                             hamming_not_match_mm_hashes)

        if prewarm_keys:
            self._backend.exists(prewarm_keys)
        logger.debug("EC similarity batch prewarm: ssim_hits=%r prewarm_keys=%r", ssim_exist_mm_hash_list, prewarm_keys)



    def _classify_similarity_cache_entries(
        self,
        exact_match_missing_mm_hashes: list[str],
    ) -> tuple[list[str], list[tuple[str, int]], list[str]]:
        ssim_exist_list: list[str] = []
        ssim_match_failed_list: list[str] = []
        with self._similarity_state_lock:
            for mm_hash in exact_match_missing_mm_hashes:
                cache_item = self.similarity_cache.get(mm_hash)
                ssim_matched_mm_hash = None
                if cache_item is not None:
                    # Find the First SSIM match Item
                    ssim_matched_mm_hash = next(
                        (
                            ssim_similar_mm_hash
                            for ssim_similar_mm_hash, score in cache_item.ssim_similar_items.items()
                            if score > self._config.ssim_threshold
                        ),
                        None,
                    )
                if ssim_matched_mm_hash is None:
                    ssim_match_failed_list.append(mm_hash)
                else:
                    ssim_exist_list.append(ssim_matched_mm_hash)

            mm_hash_to_phash_list: list[tuple[str, int]] = []
            phash_match_failed_list: list[str] = []
            for mm_hash in ssim_match_failed_list:
                cache_item = self.similarity_cache.get(mm_hash)
                if cache_item is not None and cache_item.phash is not None:
                    mm_hash_to_phash_list.append((mm_hash, cache_item.phash))
                else:
                    phash_match_failed_list.append(mm_hash)
        return ssim_exist_list, mm_hash_to_phash_list, phash_match_failed_list

    def _compute_and_store_missing_phashes(
        self,
        phash_match_failed_list: list[str],
        mm_hash_to_feature: dict[str, Any],
    ) -> None:
        """Compute missing pHashes, then persist their resized pixel tensors."""
        phash_match_failed_features = [mm_hash_to_feature[phash_match_failed_mm_hash].data for phash_match_failed_mm_hash in phash_match_failed_list]
        if len(phash_match_failed_list) <= _PARALLEL_SIMILARITY_COMPUTE_THRESHOLD:
            logger.debug(
                "EC missing pHash compute: size=%d mode=serial",
                len(phash_match_failed_list),
            )
            for mm_hash, feature in zip(phash_match_failed_list, phash_match_failed_features):
                self._compute_phash_and_save_resized_tensors(mm_hash, feature)
        else:
            logger.debug(
                "EC missing pHash compute: size=%d mode=parallel workers=%d",
                len(phash_match_failed_list),
                min(
                    len(phash_match_failed_list),
                    _MAX_SIMILARITY_COMPUTE_WORKERS,
                ),
            )
            # Parallelized for CPU-bound pHash computation. Wrapped by list to make sure all task completed.
            list(
                self._similarity_compute_executor.map(
                    self._compute_phash_and_save_resized_tensors,
                    phash_match_failed_list,
                    phash_match_failed_features,
                )
            )


    def _compute_phash_and_save_resized_tensors(
        self,
        mm_hash: str,
        feature: Any,
    ):
        resized_tensor = extract_resized_tensor(feature)
        grid = extract_image_grid(feature)
        if resized_tensor is None or grid is None:
            logger.debug(
                "EC pHash prewarm skipped: current=%s reason=no_resized_or_grid",
                mm_hash,
            )
            return
        image_phash = compute_phash(resized_tensor, grid, merge_size=self._merge_size)
        if image_phash is None:
            return
        if not self._tensor_store.put(mm_hash, resized_tensor, grid):
            return
        resized_key = self._tensor_store.get_resized_key(
            mm_hash,
            grid,
            resized_tensor.shape,
            resized_tensor.dtype,
        )
        with self._similarity_state_lock:
            candidate_resized_keys = self.phash_to_resized_mm_hashes.setdefault(
                image_phash, []
            )
            if resized_key not in candidate_resized_keys:
                candidate_resized_keys.append(resized_key)
            self.similarity_cache.setdefault(
                mm_hash, SimilarityCacheItem(phash=image_phash, ssim_similar_items=OrderedDict())
            )

    def _collect_candidate_resized_mm_hashes_by_hamming(
        self,
        mm_hash_to_phash_list: list[tuple[str, int]],
    ) -> tuple[list[str], list[str]]:
        hamming_not_match_mm_hashes: list[str] = []
        candidate_resized_keys_by_hamming: list[str] = []
        with self._similarity_state_lock:
            phash_items = list(self.phash_to_resized_mm_hashes.items())
        for mm_hash_to_phash_tuple in mm_hash_to_phash_list:
            current_mm_hash, current_phash = mm_hash_to_phash_tuple
            has_hamming_match = False
            for candidate_phash, candidate_resized_keys in sorted(phash_items):
                if hamming(current_phash, candidate_phash) <= self._config.max_hamming:
                    candidate_resized_keys_by_hamming.extend(
                        candidate_resized_keys
                    )
                    if candidate_resized_keys:
                        has_hamming_match = True
            if not has_hamming_match:
                hamming_not_match_mm_hashes.append(current_mm_hash)
        return hamming_not_match_mm_hashes, candidate_resized_keys_by_hamming


    def shutdown(self) -> None:
        self._similarity_compute_executor.shutdown(wait=True)
        self._tensor_store.shutdown()


    def _get_ssim_score_list(
        self,
        mm_hash: str,
        phash_matched_resized_keys: list[str],
        current_grid: tuple[int, ...],
        current_resized_tensor: torch.Tensor,
    ) -> list[tuple[float, str]]:
        compatible_resized_keys = [
            key
            for key in phash_matched_resized_keys
            if self._tensor_store.is_compatible_resized_key(
                key,
                current_grid,
                current_resized_tensor,
            )
        ]
        if not compatible_resized_keys:
            return []
        _, exist_keys, key_infos = self._tensor_store.batch_get_resized_key_infos(
            compatible_resized_keys,
            current_resized_tensor,
        )
        if not exist_keys:
            return []
        regular_requests, oversized_requests = (
            self._tensor_store.clarify_exist_requests(exist_keys, key_infos)
        )
        regular_requests_batches, oversized_requests_batches = (
            self._tensor_store.build_request_batches(
                regular_requests,
                oversized_requests,
            )
        )
        calculated_scores: list[tuple[float, str]] = []
        if regular_requests_batches:
            regular_scores = self._tensor_store.select_ssim_from_regular_batches(
                regular_requests_batches,
                current_resized_tensor,
                self._config.ssim_threshold,
            )
            calculated_scores.extend(regular_scores)
            if regular_scores and regular_scores[0][0] > self._config.ssim_threshold:
                # Stop here: an accepted regular-buffer candidate means no
                # oversized tensor needs to be loaded or compared.
                return calculated_scores

        if oversized_requests_batches:
            oversized_scores = self._tensor_store.select_ssim_from_oversized_batches(
                oversized_requests_batches,
                current_resized_tensor,
                self._config.ssim_threshold,
            )
            calculated_scores.extend(oversized_scores)
        return sorted(
            calculated_scores,
            key=lambda item: (-item[0], item[1]),
        )

    def _record_similarity_score(
        self,
        mm_hash: str,
        ssim_score_list: list[tuple[float, str]],
    ) -> None:
        """Record every calculated SSIM result in both cache directions."""
        if not ssim_score_list:
            return
        with self._similarity_state_lock:
            current_similarity = self.similarity_cache.get(mm_hash)
            if current_similarity is None:
                logger.warning(
                    "EC similarity score skipped: current=%s reason=no_phash",
                    mm_hash,
                )
                return
            current_scores = dict(current_similarity.ssim_similar_items)
            for score, candidate_mm_hash in ssim_score_list:
                current_scores[candidate_mm_hash] = score
            current_similarity.ssim_similar_items = self._sort_ssim_similar_items(
                current_scores
            )

            for score, candidate_mm_hash in ssim_score_list:
                candidate_similarity = self.similarity_cache.get(candidate_mm_hash)
                if candidate_similarity is None:
                    continue
                candidate_scores = dict(candidate_similarity.ssim_similar_items)
                candidate_scores[mm_hash] = score
                candidate_similarity.ssim_similar_items = (
                    self._sort_ssim_similar_items(candidate_scores)
                )

    @staticmethod
    def _sort_ssim_similar_items(
        scores: dict[str, float],
    ) -> OrderedDict[str, float]:
        return OrderedDict(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        )

def extract_resized_tensor(item: MultiModalKwargsItem) -> torch.Tensor | None:
    for field in ("pixel_values", "pixel_values_videos"):
        if field in item:
            data = item[field].data
            if isinstance(data, torch.Tensor):
                return data
    return None


def extract_image_grid(item: MultiModalKwargsItem) -> tuple[int, ...] | None:
    for field in ("image_grid_thw", "video_grid_thw"):
        if field in item:
            data = item[field].data
            if isinstance(data, torch.Tensor) and data.numel() >= 3:
                return tuple(int(x) for x in data.flatten()[:3].tolist())
    return None
