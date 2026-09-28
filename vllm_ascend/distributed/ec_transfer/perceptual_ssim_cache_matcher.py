# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Perceptual matching support for the Memcache encoder cache connector."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import logger
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_ascend.distributed.ec_transfer.phash import bands, compute_phash, hamming
from vllm_ascend.distributed.ec_transfer.ssim import DEFAULT_DATA_RANGE, build_patch_gray_plane, ssim_score
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_RESIZED_KEY_PREFIX = "resized_"
_SUPPORTED_MATCHERS = frozenset({"phash_ssim"})
_PARALLEL_SIMILARITY_COMPUTE_THRESHOLD = 3
_MAX_SIMILARITY_COMPUTE_WORKERS = 8


@dataclass(frozen=True)
class SimilarityCacheConfig:
    """Validated ``ec_connector_extra_config.similarity_cache`` settings."""

    enabled: bool = False
    matcher: str = "phash_ssim"
    max_hamming: int = 5
    ssim_threshold: float = 0.99
    ssim_data_range: float = DEFAULT_DATA_RANGE

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "SimilarityCacheConfig":
        ec_config = vllm_config.ec_transfer_config
        if ec_config is None:
            raise ValueError("ECMemcacheConnector requires ec_transfer_config")
        raw = ec_config.get_from_extra_config("similarity_cache", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("ec_connector_extra_config.similarity_cache must be a mapping")

        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("similarity_cache.enabled must be a boolean")

        matcher = raw.get("matcher", "phash_ssim")
        if not isinstance(matcher, str) or matcher not in _SUPPORTED_MATCHERS:
            raise ValueError(
                "similarity_cache.matcher must be one of "
                f"{sorted(_SUPPORTED_MATCHERS)}, got {matcher!r}"
            )

        max_hamming = raw.get("max_hamming", 5)
        if isinstance(max_hamming, bool) or not isinstance(max_hamming, int):
            raise ValueError("similarity_cache.max_hamming must be an integer")
        if not 0 <= max_hamming < 8:
            raise ValueError("similarity_cache.max_hamming must be between 0 and 7")

        ssim_threshold = raw.get("ssim_threshold", 0.99)
        if isinstance(ssim_threshold, bool) or not isinstance(ssim_threshold, (int, float)):
            raise ValueError("similarity_cache.ssim_threshold must be a number")
        ssim_threshold = float(ssim_threshold)
        if not 0.0 <= ssim_threshold <= 1.0:
            raise ValueError("similarity_cache.ssim_threshold must be between 0 and 1")

        ssim_data_range = raw.get("ssim_data_range", DEFAULT_DATA_RANGE)
        if isinstance(ssim_data_range, bool) or not isinstance(ssim_data_range, (int, float)):
            raise ValueError("similarity_cache.ssim_data_range must be a number")
        ssim_data_range = float(ssim_data_range)
        if ssim_data_range <= 0:
            raise ValueError("similarity_cache.ssim_data_range must be greater than 0")

        return cls(
            enabled=enabled,
            matcher=matcher,
            max_hamming=max_hamming,
            ssim_threshold=ssim_threshold,
            ssim_data_range=ssim_data_range,
        )


@dataclass(frozen=True)
class PerceptualCacheHit:
    store_key: str
    matcher: str
    phash_distance: int | None = None
    ssim_score: float | None = None


@dataclass(frozen=True)
class ImagePixelTensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype
    grid: tuple[int, ...]


@dataclass(frozen=True)
class PhashMatchItem:
    """Carry query data and candidates through pHash recall and prewarming."""

    # Shape, dtype, and image_grid_thw needed to interpret this mm_hash's image
    # pixel tensor. The metadata does not imply that the tensor is persisted;
    # membership in _phash_to_mm_hashes identifies registered candidates.
    image_pixel_tensor_metadata: ImagePixelTensorMetadata
    # pHash of the current query image, retained for diagnostics while the
    # candidate list is filtered during prewarming.
    current_phash: int
    # Candidate mm_hashes grouped by their Hamming distance from current_phash.
    # Each candidate maps to its SSIM score against this item, or None when it
    # has only been recalled/prewarmed and has not yet been compared by SSIM.
    # Example: {2: {"hash-a": 0.995, "hash-b": None}, 4: {"hash-c": None}}.
    candidate_mm_hash_group_by_hamming: dict[int, dict[str, float | None]]


class MemcacheResizedTensorStore:
    """Persist resized CPU tensors through a data-capable scheduler backend."""

    def __init__(self, backend: MemcacheBackend) -> None:
        self._backend = backend

    @staticmethod
    def get_resized_key(mm_hash: str) -> str:
        return _RESIZED_KEY_PREFIX + mm_hash

    def put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        key = self.get_resized_key(mm_hash)
        if self._backend.exists([key]) == [1]:
            return
        tensor = tensor.contiguous()
        result = self._backend.put_from(
            key,
            tensor.data_ptr(),
            tensor.nbytes,
            MmcDirect.COPY_H2G.value,
        )
        if result != 0:
            logger.warning("EC resized tensor put failed: key=%s result=%s", key, result)

    def get(self, mm_hash: str, metadata: ImagePixelTensorMetadata) -> torch.Tensor | None:
        return self.batch_get([mm_hash], [metadata])[0]

    def batch_get(
        self,
        mm_hashes: list[str],
        metadatas: list[ImagePixelTensorMetadata],
    ) -> list[torch.Tensor | None]:
        """Load resized tensors in one backend call while preserving input order."""
        if len(mm_hashes) != len(metadatas):
            raise ValueError(
                "MemcacheResizedTensorStore.batch_get requires the same number of "
                f"mm_hashes and metadatas, got {len(mm_hashes)} and {len(metadatas)}"
            )
        if not mm_hashes:
            return []

        keys = [self.get_resized_key(mm_hash) for mm_hash in mm_hashes]
        key_infos = self._backend.batch_get_key_info(keys)
        outputs: list[torch.Tensor | None] = [None] * len(keys)
        if not key_infos or len(key_infos) != len(keys):
            logger.warning(
                "EC resized tensor batch key info mismatch: keys=%d key_infos=%d",
                len(keys),
                len(key_infos) if key_infos else 0,
            )
            return outputs

        valid_indexes: list[int] = []
        valid_keys: list[str] = []
        valid_tensors: list[torch.Tensor] = []
        valid_sizes: list[int] = []
        for index, (key, key_info, metadata) in enumerate(zip(keys, key_infos, metadatas)):
            nbytes = int(key_info.size())
            if nbytes <= 0:
                continue
            output = torch.empty(metadata.shape, dtype=metadata.dtype, device="cpu")
            if output.nbytes != nbytes:
                logger.warning(
                    "EC resized tensor size mismatch: key=%s expected=%d actual=%d",
                    key,
                    output.nbytes,
                    nbytes,
                )
                continue
            valid_indexes.append(index)
            valid_keys.append(key)
            valid_tensors.append(output)
            valid_sizes.append(nbytes)

        if not valid_keys:
            return outputs
        results = self._backend.batch_get_into_buffers(
            valid_keys,
            [tensor.data_ptr() for tensor in valid_tensors],
            valid_sizes,
            MmcDirect.COPY_G2H.value,
        )
        if results is None or len(results) != len(valid_keys):
            logger.warning(
                "EC resized tensor batch get result mismatch: keys=%d results=%d",
                len(valid_keys),
                len(results) if results is not None else 0,
            )
            return outputs
        for index, key, tensor, result in zip(valid_indexes, valid_keys, valid_tensors, results):
            if result == 0:
                outputs[index] = tensor
            else:
                logger.warning("EC resized tensor get failed: key=%s result=%s", key, result)
        return outputs


class PhashSSIMCacheMatcher:
    """Recall and verify reusable encoder-cache entries by pixel similarity."""

    def __init__(
        self,
        backend: MemcacheBackend,
        config: SimilarityCacheConfig,
        *,
        merge_size: int,
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
        # Keep resized-tensor persistence off the scheduling path. A single
        # worker preserves submission order and avoids concurrent store writes.
        self._put_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ec-resized-put")
        # Reuse CPU workers across pHash recall and SSIM scoring. The two
        # phases are awaited sequentially, so one pool avoids idle duplicates.
        self._similarity_compute_executor = ThreadPoolExecutor(
            max_workers=_MAX_SIMILARITY_COMPUTE_WORKERS,
            thread_name_prefix="ec-similarity-compute",
        )
        # Content example:
        # {
        #     "current-image-hash": PhashMatchItem(
        #         image_pixel_tensor_metadata=ImagePixelTensorMetadata(
        #             shape=(1600, 1176), dtype=torch.float32, grid=(1, 40, 40)
        #         ),
        #         current_phash=123456789,
        #         candidate_mm_hash_group_by_hamming={
        #             2: {"candidate-hash-A": 0.995},
        #             4: {"candidate-hash-B": None},
        #         },
        #     )
        # }
        # The tensor metadata and pHash are matcher-lifetime data and survive
        # clear_step_state. Candidate groups are step-local recall results and
        # are reset before the next prewarm without discarding the cached pHash.
        self.phash_match_results: dict[str, PhashMatchItem] = {}
        # Protects phash_match_results because large batches compute pHashes
        # and update candidate groups from multiple recall threads.
        self._phash_cache_lock = Lock()




        # Persistence adapter for resized tensors stored under resized_<mm_hash>.
        self._tensor_store = MemcacheResizedTensorStore(backend)

        # Reverse lookup from an exact pHash value to all original mm_hashes
        # that share that value.
        self._phash_to_mm_hashes: dict[int, set[str]] = {}
        # LSH-style band lookup: (band position, band value) -> pHash values.
        # It narrows the hashes considered before the full Hamming check.
        self._band_index: dict[tuple[int, int], set[int]] = {}







        # Lifetime metrics for completed SSIM comparisons and their aggregate
        # wall-clock latency in milliseconds.
        self.ssim_comparison_count = 0
        self.ssim_comparison_total_ms = 0.0
        # Lifetime metrics for resized-tensor Memcache batches and aggregate
        # batch loading latency in milliseconds.
        self.resized_tensor_batch_get_count = 0
        self.resized_tensor_batch_get_total_ms = 0.0

    def get_or_update_similarity_match(self, mm_hash: str, current_feature: Any) -> str | None:
        """Return the best SSIM hit from the nearest prewarmed pHash group."""
        with self._phash_cache_lock:
            match_item = self.phash_match_results.get(mm_hash)
        if match_item is None:
            query_resized = extract_resized_tensor(current_feature)
            query_grid = extract_image_grid(current_feature)
            if query_resized is not None and query_grid is not None:
                self._cache_or_compute_phash(mm_hash, query_resized, query_grid)
            logger.debug("EC similarity match skipped: current=%s reason=not_prewarmed", mm_hash)
            return None

        candidate_groups = match_item.candidate_mm_hash_group_by_hamming
        if not candidate_groups:
            logger.debug("EC similarity match: current=%s result=no_candidate", mm_hash)
            return None

        min_hamming = min(candidate_groups)
        candidate_mm_hashes = list(candidate_groups[min_hamming])
        if not candidate_mm_hashes:
            logger.debug(
                "EC similarity match: current=%s hamming=%d result=no_candidate",
                mm_hash,
                min_hamming,
            )
            return None
        logger.debug(
            "EC similarity match candidates selected: current=%s hamming=%d candidates=%r",
            mm_hash,
            min_hamming,
            candidate_mm_hashes,
        )

        current_resized_tensor = extract_resized_tensor(current_feature)
        current_image_grid = extract_image_grid(current_feature)
        if (
            current_resized_tensor is None
            or current_image_grid != match_item.image_pixel_tensor_metadata.grid
        ):
            logger.debug(
                "EC similarity match skipped: current=%s reason=query_data_unavailable_or_grid_changed",
                mm_hash,
            )
            return None

        gray_plane = build_patch_gray_plane(current_resized_tensor, current_image_grid, merge_size=self._merge_size)
        if gray_plane is None:
            logger.debug("EC similarity match skipped: current=%s reason=gray_plane_unavailable", mm_hash)
            return None

        ssim_score_list = self._get_ssim_score_list(mm_hash, candidate_mm_hashes, gray_plane, current_image_grid)
        self._record_ssim_scores(mm_hash, match_item, min_hamming, ssim_score_list)

        qualified_scores = [result for result in ssim_score_list if result[0] >= self._config.ssim_threshold]
        hit = self._best_existing(qualified_scores)
        if hit is None:
            logger.debug(
                "EC similarity match: current=%s hamming=%d result=no_ssim_hit threshold=%.6f",
                mm_hash,
                min_hamming,
                self._config.ssim_threshold,
            )
            return None
        score, candidate_mm_hash = hit
        logger.debug(
            "EC similarity match success: current=%s candidate=%s hamming=%d ssim=%.6f",
            mm_hash,
            candidate_mm_hash,
            min_hamming,
            score,
        )
        return candidate_mm_hash

    def batch_ensure_similarity_cache_available(
        self,
        mm_hashes: list[str],
        missing_features: list[Any],
    ) -> list[list[str]]:
        if len(mm_hashes) != len(missing_features):
            raise ValueError(
                "batch_ensure_similarity_cache_available requires the same number of mm_hashes "
                f"and missing_features, got {len(mm_hashes)} and {len(missing_features)}"
            )

        candidate_phash_items = self._batch_get_candidate_phash_items(mm_hashes, missing_features)

        resized_keys: set[str] = set()
        for candidate_phash_item in candidate_phash_items:
            if candidate_phash_item is None:
                continue
            for candidate_mm_hash in self._flatten_candidate_mm_hashes(
                    candidate_phash_item.candidate_mm_hash_group_by_hamming
            ):
                resized_key = self._tensor_store.get_resized_key(candidate_mm_hash)
                resized_keys.add(resized_key)

        # Do warmup
        existing_resized_keys = self._warmup_existing_resized_keys(resized_keys)
        results: list[list[str]] = []
        for current_image_mm_hash, match_item in zip(mm_hashes, candidate_phash_items):
            if match_item is None:
                results.append([])
                continue
            available_groups = {
                distance: {
                    candidate_mm_hash: score
                    for candidate_mm_hash, score in candidate_scores.items()
                    if self._tensor_store.get_resized_key(candidate_mm_hash) in existing_resized_keys
                }
                for distance, candidate_scores in match_item.candidate_mm_hash_group_by_hamming.items()
            }
            available_groups = {
                distance: candidate_scores
                for distance, candidate_scores in available_groups.items()
                if candidate_scores
            }
            available_item = PhashMatchItem(
                image_pixel_tensor_metadata=match_item.image_pixel_tensor_metadata,
                current_phash=match_item.current_phash,
                candidate_mm_hash_group_by_hamming=available_groups,
            )
            with self._phash_cache_lock:
                self.phash_match_results[current_image_mm_hash] = available_item
            candidates = self._flatten_candidate_mm_hashes(available_groups)
            logger.debug(
                "EC similarity candidates prewarmed: current=%s prewarmed candidates=%r",
                current_image_mm_hash,
                candidates,
            )
            results.append(candidates)
        return results

    def _batch_get_candidate_phash_items(
        self,
        mm_hashes: list[str],
        missing_features: list[Any],
    ) -> list[PhashMatchItem | None]:
        self._clear_candidate_groups(mm_hashes)
        resized_tensors = [feature.data for feature in missing_features]
        if len(mm_hashes) <= _PARALLEL_SIMILARITY_COMPUTE_THRESHOLD:
            logger.debug("EC similarity batch recall: size=%d mode=serial", len(mm_hashes))
            candidate_phash_items = [
                self._get_candidate_phash_item(mm_hash, resized_tensor)
                for mm_hash, resized_tensor in zip(mm_hashes, resized_tensors)
            ]
        else:
            logger.debug(
                "EC similarity batch recall: size=%d mode=parallel workers=%d",
                len(mm_hashes),
                min(len(mm_hashes), _MAX_SIMILARITY_COMPUTE_WORKERS),
            )
            # executor.map preserves input order, so each recall remains
            # aligned with the corresponding mm_hash and item.
            candidate_phash_items = list(
                self._similarity_compute_executor.map(
                    self._get_candidate_phash_item,
                    mm_hashes,
                    resized_tensors,
                )
            )
        return candidate_phash_items

    def _warmup_existing_resized_keys(self, resized_keys: set[str]) -> set[str]:
        if not resized_keys:
            return set()
        exists = self._backend.exists(resized_keys)
        if not exists:
            logger.debug(
                "EC similarity batch prewarm exists returned no results: resized_keys=%r",
                resized_keys,
            )
            return set()
        if len(exists) != len(resized_keys):
            logger.debug(
                "EC similarity batch prewarm exists returned %d results for %d resized keys: keys=%r",
                len(exists),
                len(resized_keys),
                resized_keys,
            )
            return set()
        return {resized_key for resized_key, is_available in zip(resized_keys, exists) if is_available == 1}

    def _get_candidate_phash_item(self, mm_hash: str, resized_tensor: Any) -> PhashMatchItem | None:
        if resized_tensor is None:
            logger.debug("EC candidate phash search skipped: current=%s reason=no_feature_data", mm_hash)
            return None
        resized = extract_resized_tensor(resized_tensor)
        grid = extract_image_grid(resized_tensor)
        if resized is None or grid is None:
            logger.debug("EC candidate phash search skipped: current=%s reason=no_resized_or_grid", mm_hash)
            return None

        phash_result = self._cache_or_compute_phash(mm_hash, resized, grid)
        if phash_result is None:
            logger.debug("EC candidate phash search skipped: current=%s reason=phash_unavailable", mm_hash)
            return None
        phash, grid = phash_result
        candidate_mm_hash_group_by_hamming = self.find_candidate_phash_by_hamming(
            phash, grid, exclude=mm_hash
        )
        if not candidate_mm_hash_group_by_hamming:
            logger.debug("EC similarity prewarm: current=%s candidates=[] resized_keys=[]", mm_hash)
            return None
        match_item = PhashMatchItem(
            image_pixel_tensor_metadata=ImagePixelTensorMetadata(
                tuple(resized.shape), resized.dtype, grid
            ),
            current_phash=phash,
            candidate_mm_hash_group_by_hamming=candidate_mm_hash_group_by_hamming,
        )
        with self._phash_cache_lock:
            self.phash_match_results[mm_hash] = match_item
        return match_item

    def register(self, mm_hash: str, item: Any) -> None:
        if item is None:
            return
        resized = extract_resized_tensor(item)
        grid = extract_image_grid(item)
        if resized is None or grid is None:
            return

        phash_result = self._cache_or_compute_phash(mm_hash, resized, grid)
        if phash_result is None:
            return
        image_phash, grid = phash_result
        if mm_hash in self._phash_to_mm_hashes.get(image_phash, ()):
            return
        self._phash_to_mm_hashes.setdefault(image_phash, set()).add(mm_hash)
        for band_key in bands(image_phash):
            self._band_index.setdefault(band_key, set()).add(image_phash)

        persistent_tensor = resized.detach().to(device="cpu").contiguous()
        self._put_executor.submit(self._put_resized_safely, mm_hash, persistent_tensor)

    def clear_step_state(self) -> None:
        self._clear_candidate_groups()

    def shutdown(self) -> None:
        self._similarity_compute_executor.shutdown(wait=True)
        self._put_executor.shutdown(wait=True)

    def _put_resized_safely(self, mm_hash: str, tensor: torch.Tensor) -> None:
        try:
            self._tensor_store.put(mm_hash, tensor)
        except Exception:
            logger.exception("EC resized tensor async put failed: mm_hash=%s", mm_hash)

    def _cache_or_compute_phash(
        self,
        mm_hash: str,
        resized: torch.Tensor,
        grid: tuple[int, ...],
    ) -> tuple[int, tuple[int, ...]] | None:
        current_metadata = ImagePixelTensorMetadata(tuple(resized.shape), resized.dtype, grid)
        with self._phash_cache_lock:
            cached = self.phash_match_results.get(mm_hash)
        if cached is not None:
            if cached.image_pixel_tensor_metadata != current_metadata:
                logger.warning(
                    "EC pHash identity conflict: mm_hash=%s cached_metadata=%r current_metadata=%r",
                    mm_hash,
                    cached.image_pixel_tensor_metadata,
                    current_metadata,
                )
                return None
            return cached.current_phash, cached.image_pixel_tensor_metadata.grid
        image_phash = compute_phash(resized, grid, merge_size=self._merge_size)
        if image_phash is None:
            return None
        result = PhashMatchItem(
            image_pixel_tensor_metadata=current_metadata,
            current_phash=image_phash,
            candidate_mm_hash_group_by_hamming={},
        )
        with self._phash_cache_lock:
            cached_or_result = self.phash_match_results.setdefault(mm_hash, result)
        if cached_or_result.image_pixel_tensor_metadata != current_metadata:
            logger.warning(
                "EC pHash identity conflict: mm_hash=%s cached_metadata=%r current_metadata=%r",
                mm_hash,
                cached_or_result.image_pixel_tensor_metadata,
                current_metadata,
            )
            return None
        return cached_or_result.current_phash, cached_or_result.image_pixel_tensor_metadata.grid

    def _clear_candidate_groups(self, mm_hashes: list[str] | None = None) -> None:
        """Drop step-local candidates while retaining matcher-lifetime pHashes."""
        with self._phash_cache_lock:
            keys = list(self.phash_match_results) if mm_hashes is None else mm_hashes
            for mm_hash in keys:
                cached = self.phash_match_results.get(mm_hash)
                if cached is None or not cached.candidate_mm_hash_group_by_hamming:
                    continue
                self.phash_match_results[mm_hash] = PhashMatchItem(
                    image_pixel_tensor_metadata=cached.image_pixel_tensor_metadata,
                    current_phash=cached.current_phash,
                    candidate_mm_hash_group_by_hamming={},
                )

    def find_candidate_phash_by_hamming(
        self,
        query_phash: int,
        grid: tuple[int, ...],
        *,
        exclude: str,
    ) -> dict[int, dict[str, float | None]]:
        candidate_phashes = {
            candidate_phash
            for band_key in bands(query_phash)
            for candidate_phash in self._band_index.get(band_key, ())
        }
        recalled_mm_hashes_by_hamming: dict[int, list[str]] = {}
        for candidate_phash in candidate_phashes:
            # Apply the exact 64-bit Hamming threshold after the band lookup.
            distance = hamming(query_phash, candidate_phash)
            if distance > self._config.max_hamming:
                continue
            # One pHash may belong to multiple original images. Expand it back
            # to mm_hashes, excluding the query image itself.
            for candidate_mm_hash in self._phash_to_mm_hashes.get(candidate_phash, ()):
                if candidate_mm_hash == exclude:
                    continue
                # Matching grids ensure the persisted resized tensor has a
                # layout compatible with the current query before SSIM.
                candidate_item = self.phash_match_results.get(candidate_mm_hash)
                if (
                    candidate_item is not None
                    and candidate_item.image_pixel_tensor_metadata.grid == grid
                ):
                    recalled_mm_hashes_by_hamming.setdefault(distance, []).append(candidate_mm_hash)
        # Set-backed indexes have no stable traversal order. Sort both the
        # Hamming groups and each group's mm_hashes for deterministic matching.
        return {
            distance: {
                candidate_mm_hash: None
                for candidate_mm_hash in sorted(recalled_mm_hashes_by_hamming[distance])
            }
            for distance in sorted(recalled_mm_hashes_by_hamming)
        }

    @staticmethod
    def _flatten_candidate_mm_hashes(
        candidate_mm_hash_group_by_hamming: dict[int, dict[str, float | None]],
    ) -> list[str]:
        """Flatten Hamming groups in ascending distance and mm_hash order."""
        return [
            candidate_mm_hash
            for distance in sorted(candidate_mm_hash_group_by_hamming)
            for candidate_mm_hash in candidate_mm_hash_group_by_hamming[distance]
        ]

    def _get_ssim_score_list(
        self,
        mm_hash: str,
        candidate_mm_hashes: list[str],
        gray_plane: torch.Tensor,
        grid: tuple[int, ...],
    ) -> list[tuple[float, str]]:
        valid_mm_hashes: list[str] = []
        metadatas: list[ImagePixelTensorMetadata] = []
        for candidate_mm_hash in candidate_mm_hashes:
            candidate_item = self.phash_match_results.get(candidate_mm_hash)
            if candidate_item is None:
                continue
            metadata = candidate_item.image_pixel_tensor_metadata
            if metadata.grid != grid:
                continue
            valid_mm_hashes.append(candidate_mm_hash)
            metadatas.append(metadata)
        if not valid_mm_hashes:
            return []

        batch_get_start = time.perf_counter()
        candidate_tensors = self._tensor_store.batch_get(valid_mm_hashes, metadatas)
        batch_get_elapsed_ms = (time.perf_counter() - batch_get_start) * 1e3
        self.resized_tensor_batch_get_count += 1
        self.resized_tensor_batch_get_total_ms += batch_get_elapsed_ms
        logger.debug(
            "EC resized tensor batch get: current=%s candidates=%d elapsed_ms=%.3f",
            mm_hash,
            len(valid_mm_hashes),
            batch_get_elapsed_ms,
        )

        score_mm_hashes: list[str] = []
        score_tensors: list[torch.Tensor] = []
        score_metadatas: list[ImagePixelTensorMetadata] = []
        for candidate_mm_hash, candidate_tensor, metadata in zip(
            valid_mm_hashes, candidate_tensors, metadatas
        ):
            if candidate_tensor is None:
                continue
            score_mm_hashes.append(candidate_mm_hash)
            score_tensors.append(candidate_tensor)
            score_metadatas.append(metadata)
        if not score_mm_hashes:
            return []

        if len(score_mm_hashes) <= _PARALLEL_SIMILARITY_COMPUTE_THRESHOLD:
            logger.debug("EC similarity SSIM scoring: size=%d mode=serial", len(score_mm_hashes))
            score_results = [
                self._score_candidate(candidate_mm_hash, candidate_tensor, metadata, gray_plane)
                for candidate_mm_hash, candidate_tensor, metadata in zip(
                    score_mm_hashes, score_tensors, score_metadatas
                )
            ]
        else:
            logger.debug(
                "EC similarity SSIM scoring: size=%d mode=parallel workers=%d",
                len(score_mm_hashes),
                min(len(score_mm_hashes), _MAX_SIMILARITY_COMPUTE_WORKERS),
            )
            score_results = list(
                self._similarity_compute_executor.map(
                    self._score_candidate,
                    score_mm_hashes,
                    score_tensors,
                    score_metadatas,
                    [gray_plane] * len(score_mm_hashes),
                )
            )

        ssim_score_list: list[tuple[float, str]] = []
        for result in score_results:
            if result is None:
                continue
            score, candidate_mm_hash, elapsed_ms = result
            logger.debug(
                "EC similarity comparison: current=%s candidate=%s ssim=%.6f elapsed_ms=%.3f",
                mm_hash,
                candidate_mm_hash,
                score,
                elapsed_ms,
            )
            ssim_score_list.append((score, candidate_mm_hash))
            self.ssim_comparison_count += 1
            self.ssim_comparison_total_ms += elapsed_ms
        return ssim_score_list

    def _score_candidate(
        self,
        candidate_mm_hash: str,
        candidate_resized: torch.Tensor,
        metadata: ImagePixelTensorMetadata,
        gray_plane: torch.Tensor,
    ) -> tuple[float, str, float] | None:
        """Build one candidate plane and calculate SSIM without shared writes."""
        start = time.perf_counter()
        candidate_gray = build_patch_gray_plane(
            candidate_resized,
            metadata.grid,
            merge_size=self._merge_size,
        )
        if candidate_gray is None:
            return None
        score = ssim_score(gray_plane, candidate_gray, data_range=self._config.ssim_data_range)
        if score is None:
            return None
        return score, candidate_mm_hash, (time.perf_counter() - start) * 1e3

    def _record_ssim_scores(
        self,
        mm_hash: str,
        match_item: PhashMatchItem,
        hamming_distance: int,
        scored: list[tuple[float, str]],
    ) -> None:
        """Record calculated SSIM scores in both directions of the match graph."""
        if not scored:
            return
        with self._phash_cache_lock:
            current = self.phash_match_results.get(mm_hash)
            if current is None:
                current = match_item
            current_groups = self._copy_candidate_groups(current)
            current_group = current_groups.setdefault(hamming_distance, {})
            for score, candidate_mm_hash in scored:
                current_group[candidate_mm_hash] = score
            self.phash_match_results[mm_hash] = PhashMatchItem(
                image_pixel_tensor_metadata=current.image_pixel_tensor_metadata,
                current_phash=current.current_phash,
                candidate_mm_hash_group_by_hamming=current_groups,
            )

            for score, candidate_mm_hash in scored:
                candidate_item = self.phash_match_results.get(candidate_mm_hash)
                if candidate_item is None:
                    logger.warning(
                        "EC similarity reverse score skipped: current=%s candidate=%s reason=no_match_item",
                        mm_hash,
                        candidate_mm_hash,
                    )
                    continue
                candidate_groups = self._copy_candidate_groups(candidate_item)
                candidate_groups.setdefault(hamming_distance, {})[mm_hash] = score
                self.phash_match_results[candidate_mm_hash] = PhashMatchItem(
                    image_pixel_tensor_metadata=candidate_item.image_pixel_tensor_metadata,
                    current_phash=candidate_item.current_phash,
                    candidate_mm_hash_group_by_hamming=candidate_groups,
                )

    @staticmethod
    def _copy_candidate_groups(
        match_item: PhashMatchItem,
    ) -> dict[int, dict[str, float | None]]:
        return {
            distance: dict(candidate_scores)
            for distance, candidate_scores in match_item.candidate_mm_hash_group_by_hamming.items()
        }

    def _first_existing(self, candidate_mm_hashes: list[str]) -> str | None:
        if not candidate_mm_hashes:
            return None
        exists = self._backend.exists(candidate_mm_hashes)
        if len(exists) != len(candidate_mm_hashes):
            logger.warning(
                "EC similarity exists returned %d results for %d candidates",
                len(exists),
                len(candidate_mm_hashes),
            )
            return None
        for candidate_mm_hash, is_available in zip(candidate_mm_hashes, exists):
            if is_available == 1:
                return candidate_mm_hash
        return None

    def _best_existing(self, scored: list[tuple[float, str]]) -> tuple[float, str] | None:
        ranked = sorted(scored, key=lambda candidate: (-candidate[0], candidate[1]))
        candidate = self._first_existing([candidate_mm_hash for _, candidate_mm_hash in ranked])
        if candidate is None:
            return None
        return next(item for item in ranked if item[1] == candidate)


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
