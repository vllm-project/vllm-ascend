# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""pHash recall and SSIM matching for the multimodal encoder cache."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

import torch
from vllm.logger import logger
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_ascend.distributed.ec_transfer.memcache_resized_tensor_store import (
    MemcacheResizedTensorStore,
)
from vllm_ascend.distributed.ec_transfer.phash import compute_phash
from vllm_ascend.distributed.ec_transfer.similarity_cache_config import (
    _MAX_SIMILARITY_COMPUTE_WORKERS,
    _PARALLEL_SIMILARITY_COMPUTE_THRESHOLD,
    SimilarityCacheConfig,
)
from vllm_ascend.distributed.ec_transfer.similarity_index import (
    LocalSimilarityIndex,
    SimilarityIndex,
    sort_similarity_scores,
)
from vllm_ascend.distributed.ec_transfer.similarity_index import (
    SimilarityCacheItem as SimilarityCacheItem,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
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
        index: SimilarityIndex | None = None,
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
        # Pixel tensors and grids are CPU inputs; reuse workers across
        # preprocessing batches independently of NPU encoder embeddings.
        self._similarity_compute_executor = ThreadPoolExecutor(
            max_workers=_MAX_SIMILARITY_COMPUTE_WORKERS,
            thread_name_prefix="ec-similarity-compute",
        )
        self._index = LocalSimilarityIndex(config) if index is None else index
        # Persistence adapter for model-, shape-, and dtype-specific resized
        # tensor keys.
        self._tensor_store = MemcacheResizedTensorStore(
            backend,
            merge_size,
            model_id,
            config.resized_image_recv_buffer_size,
        )

    def get_cached_similarity_match(self, mm_hash: str) -> str | None:
        """Return the best qualified cached match with an available embedding."""
        item = self._index.batch_get_entries([mm_hash])[0]
        if item is None:
            return None
        candidates = [
            candidate
            for candidate, score in item.ssim_similar_items.items()
            if candidate != mm_hash and score > self._config.ssim_threshold
        ]
        if not candidates:
            return None
        exists = self._backend.exists(candidates)
        if not exists or len(exists) != len(candidates):
            return None
        return next((candidate for candidate, available in zip(candidates, exists) if available == 1), None)

    def get_cached_phash(self, mm_hash: str) -> int | None:
        item = self._index.batch_get_entries([mm_hash])[0]
        return None if item is None else item.phash

    def find_resized_mm_hashes_by_hamming(self, query_phash: int) -> list[str] | None:
        return self._index.batch_recall([query_phash])[0] or None

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
            [key for key in phash_matched_mm_hashes if self._tensor_store.get_mm_hash_from_resized_key(key) != mm_hash],
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
        # A completed batch may contain only low scores. Hash equality is
        # still only recall; reject equality with the threshold and NaN too.
        if not score > self._config.ssim_threshold:
            return None
        logger.debug(
            "EC similarity match success: current=%s candidate=%s ssim=%.6f",
            mm_hash,
            candidate_mm_hash,
            score,
        )
        return candidate_mm_hash

    def batch_ensure_similarity_cache_available(
        self,
        exact_match_missing_mm_hashes: list[str],
        exact_match_missing_features: list[Any],
    ) -> None:
        """Prewarm keys for cached matches without persisting query tensors.

        A query with no pHash is registered in the selected shared/local index,
        but its resized tensor is intentionally not persisted yet. Memcache
        persistence happens after final matching, and only for a final save
        miss.
        """
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

        mm_hash_to_feature_map = dict(zip(exact_match_missing_mm_hashes, exact_match_missing_features))
        # Phash match failed: calculate and register the pHash metadata, but
        # do not put the query tensor into Memcache before final matching.
        if phash_match_failed_mm_hash_list:
            self._compute_and_cache_missing_phashes(
                phash_match_failed_mm_hash_list,
                mm_hash_to_feature_map,
            )

        prewarm_keys = list(dict.fromkeys(ssim_exist_mm_hash_list))
        # If Phash found, process by hamming search result
        if mm_hash_to_phash_list:
            (
                hamming_not_match_mm_hashes,
                candidate_resized_keys_by_hamming,
            ) = self._collect_candidate_resized_mm_hashes_by_hamming(mm_hash_to_phash_list)
            prewarm_keys.extend(dict.fromkeys(candidate_resized_keys_by_hamming))
            prewarm_keys.extend(
                dict.fromkeys(
                    candidate_mm_hash
                    for resized_key in candidate_resized_keys_by_hamming
                    if (candidate_mm_hash := self._tensor_store.get_mm_hash_from_resized_key(resized_key)) is not None
                )
            )
            if hamming_not_match_mm_hashes:
                logger.debug(
                    "EC similarity batch not prewarm for they are not exist: hamming_not_match_mm_hashes=%r",
                    hamming_not_match_mm_hashes,
                )

        if prewarm_keys:
            self._backend.exists(prewarm_keys)
        logger.debug("EC similarity batch prewarm: ssim_hits=%r prewarm_keys=%r", ssim_exist_mm_hash_list, prewarm_keys)

    def _classify_similarity_cache_entries(
        self,
        exact_match_missing_mm_hashes: list[str],
    ) -> tuple[list[str], list[tuple[str, int]], list[str]]:
        ssim_exist_list: list[str] = []
        phashes: list[tuple[str, int]] = []
        missing: list[str] = []
        entries = self._index.batch_get_entries(exact_match_missing_mm_hashes)
        for mm_hash, item in zip(exact_match_missing_mm_hashes, entries):
            if item is None:
                missing.append(mm_hash)
                continue
            match = next(
                (
                    candidate
                    for candidate, score in item.ssim_similar_items.items()
                    if candidate != mm_hash and score > self._config.ssim_threshold
                ),
                None,
            )
            if match is not None:
                ssim_exist_list.append(match)
            else:
                phashes.append((mm_hash, item.phash))
        return ssim_exist_list, phashes, missing

    def _compute_and_cache_missing_phashes(
        self,
        phash_match_failed_list: list[str],
        mm_hash_to_feature: dict[str, Any],
    ) -> None:
        """Compute on local workers, then publish one atomic metadata batch."""
        features = [mm_hash_to_feature[mm_hash].data for mm_hash in phash_match_failed_list]
        if len(features) <= _PARALLEL_SIMILARITY_COMPUTE_THRESHOLD:
            entries = [
                self._compute_phash_metadata(mm_hash, feature)
                for mm_hash, feature in zip(phash_match_failed_list, features)
            ]
        else:
            futures = [
                self._similarity_compute_executor.submit(self._compute_phash_metadata, mm_hash, feature)
                for mm_hash, feature in zip(phash_match_failed_list, features)
            ]
            # No index RPC runs on the compute pool. Drain every computation
            # before propagating errors or publishing the batch.
            wait(futures)
            entries = [future.result() for future in futures]
        self._index.batch_register_phashes([entry for entry in entries if entry is not None])

    def _compute_phash_metadata(
        self,
        mm_hash: str,
        feature: Any,
    ) -> tuple[str, int, str] | None:
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
        resized_key = self._tensor_store.get_resized_key(
            mm_hash,
            grid,
            resized_tensor.shape,
            resized_tensor.dtype,
        )
        return mm_hash, image_phash, resized_key

    def persist_missed_resized_tensor(
        self,
        mm_hash: str,
        current_feature: Any,
    ) -> None:
        """Persist a resized tensor for an image that missed L2.

        Called only after matching has completed. Its local key may already be
        registered from ensure-time pHash prewarming; persistence now supplies
        the Memcache value for future requests.
        """
        image_phash = self.get_cached_phash(mm_hash)
        resized_tensor = extract_resized_tensor(current_feature)
        grid = extract_image_grid(current_feature)
        if resized_tensor is None or grid is None:
            logger.debug(
                "EC missed resized tensor skipped: current=%s reason=no_resized_or_grid",
                mm_hash,
            )
            return
        if image_phash is None:
            image_phash = compute_phash(resized_tensor, grid, merge_size=self._merge_size)
            if image_phash is None:
                logger.debug(
                    "EC missed resized tensor skipped: current=%s reason=phash_unavailable",
                    mm_hash,
                )
                return
        if not self._tensor_store.put(mm_hash, resized_tensor, grid):
            logger.warning("EC missed resized tensor put failed: mm_hash=%s", mm_hash)
            return
        resized_key = self._tensor_store.get_resized_key(
            mm_hash,
            grid,
            resized_tensor.shape,
            resized_tensor.dtype,
        )
        self._index.batch_register_phashes([(mm_hash, image_phash, resized_key)])

    def _collect_candidate_resized_mm_hashes_by_hamming(
        self,
        mm_hash_to_phash_list: list[tuple[str, int]],
    ) -> tuple[list[str], list[str]]:
        hamming_not_match_mm_hashes: list[str] = []
        candidate_resized_keys_by_hamming: list[str] = []
        phashes = list(dict.fromkeys(phash for _, phash in mm_hash_to_phash_list))
        matches_by_phash = dict(zip(phashes, self._index.batch_recall(phashes)))
        for current_mm_hash, current_phash in mm_hash_to_phash_list:
            candidate_resized_keys = matches_by_phash[current_phash]
            if candidate_resized_keys:
                candidate_resized_keys_by_hamming.extend(candidate_resized_keys)
            else:
                hamming_not_match_mm_hashes.append(current_mm_hash)
        return hamming_not_match_mm_hashes, candidate_resized_keys_by_hamming

    def shutdown(self) -> None:
        self._similarity_compute_executor.shutdown(wait=True)
        try:
            self._tensor_store.shutdown()
        finally:
            self._index.close()

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
        regular_requests, oversized_requests = self._tensor_store.clarify_exist_requests(exist_keys, key_infos)
        regular_requests_batches, oversized_requests_batches = self._tensor_store.build_request_batches(
            regular_requests,
            oversized_requests,
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
        self._index.record_similarity_scores(mm_hash, ssim_score_list)

    @staticmethod
    def _sort_ssim_similar_items(scores: dict[str, float]) -> OrderedDict[str, float]:
        return sort_similarity_scores(scores)


def extract_resized_tensor(item: MultiModalKwargsItem) -> torch.Tensor | None:
    for field_name in ("pixel_values", "pixel_values_videos"):
        if field_name in item:
            data = item[field_name].data
            if isinstance(data, torch.Tensor):
                return data
    return None


def extract_image_grid(item: MultiModalKwargsItem) -> tuple[int, ...] | None:
    for field_name in ("image_grid_thw", "video_grid_thw"):
        if field_name in item:
            data = item[field_name].data
            if isinstance(data, torch.Tensor) and data.numel() >= 3:
                return tuple(int(x) for x in data.flatten()[:3].tolist())
    return None
