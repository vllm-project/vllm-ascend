# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Memcache persistence and SSIM scoring for resized image tensors."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.parse import quote

import torch
from vllm.logger import logger

from vllm_ascend.distributed.ec_transfer.similarity_cache_config import (
    DEFAULT_RESIZED_IMAGE_RECV_BUFFER_SIZE,
    MAX_RESIZED_IMAGE_RECV_BUFFER_SIZE,
    MAX_RESIZED_IMAGE_TRANSFER_SIZE,
    MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE,
    _MAX_SIMILARITY_COMPUTE_WORKERS,
    _PARALLEL_SIMILARITY_COMPUTE_THRESHOLD,
)
from vllm_ascend.distributed.ec_transfer.ssim import (
    build_patch_gray_plane,
    ssim_score,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)

_RESIZED_KEY_PREFIX = "resized_"


@dataclass(frozen=True)
class ResizedTensorLoadRequest:
    """One resized-tensor transfer into a batch receive buffer."""

    # Position where the loaded tensor is written in the batch output list.
    output_index: int
    # Complete model-, grid-, shape-, and dtype-specific Memcache key.
    resized_key: str
    # Number of bytes occupied by the serialized tensor value.
    nbytes: int


class MemcacheResizedTensorStore:
    """Persist resized CPU pixel tensors independently of NPU embeddings."""

    def __init__(
        self,
        backend: MemcacheBackend,
        merge_size: int,
        model_id: str,
        recv_buffer_size: int = DEFAULT_RESIZED_IMAGE_RECV_BUFFER_SIZE,
    ) -> None:
        if isinstance(recv_buffer_size, bool) or not isinstance(recv_buffer_size, int):
            raise ValueError("recv_buffer_size must be an integer")
        if not (MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE <= recv_buffer_size <= MAX_RESIZED_IMAGE_RECV_BUFFER_SIZE):
            raise ValueError(
                "recv_buffer_size must be between "
                f"{MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE} and "
                f"{MAX_RESIZED_IMAGE_RECV_BUFFER_SIZE} bytes, got "
                f"{recv_buffer_size}"
            )
        self._backend = backend
        # Keep the model namespace human-readable while escaping separators
        # and other characters that may occur in Hugging Face IDs or paths.
        self._model_identifier = quote(model_id, safe=".-_")
        self._recv_buffer_size = recv_buffer_size
        # Allocate and register lazily so a process that never has a similarity
        # candidate does not reserve the configured host-memory capacity.
        self._recv_buffer: torch.Tensor | None = None
        self._recv_buffer_registered = False
        self._merge_size = merge_size
        # register_buffer uses shared pending state in MemcacheBackend.
        # Keep registration alive for the entire PUT, including duplicate ptrs.
        self._put_lock = Lock()
        # Reuse CPU workers across SSIM batches. Device work stays on the
        # calling thread so it retains the producer's current NPU stream.
        self._similarity_compute_executor = ThreadPoolExecutor(
            max_workers=_MAX_SIMILARITY_COMPUTE_WORKERS,
            thread_name_prefix="ec-similarity-compute",
        )

    def get_resized_key(
        self,
        mm_hash: str,
        grid: tuple[int, ...],
        shape: tuple[int, ...] | torch.Size,
        dtype: torch.dtype,
    ) -> str:
        """Build a model-, grid-, shape-, and dtype-specific tensor key."""
        if len(grid) < 3:
            raise ValueError(f"grid must contain T, H, and W, got {grid!r}")
        grid_suffix = "x".join(str(int(dimension)) for dimension in grid[:3])
        shape_suffix = "x".join(str(int(dimension)) for dimension in shape)
        dtype_suffix = str(dtype).removeprefix("torch.")
        return (
            f"{_RESIZED_KEY_PREFIX}{mm_hash}_grid:{grid_suffix}_"
            f"shape:{shape_suffix}_"
            f"{dtype_suffix}_{self._model_identifier}"
        )

    @staticmethod
    def get_mm_hash_from_resized_key(key: str) -> str | None:
        """Extract the original mm_hash from a versioned resized key."""
        if not key.startswith(_RESIZED_KEY_PREFIX):
            return None
        mm_hash, separator, _ = key[len(_RESIZED_KEY_PREFIX) :].partition("_grid:")
        return mm_hash if separator and mm_hash else None

    @staticmethod
    def get_grid_from_resized_key(key: str) -> tuple[int, int, int] | None:
        """Extract the per-image ``(T, H, W)`` grid from a resized key."""
        _, separator, suffix = key.partition("_grid:")
        if not separator:
            return None
        grid_text, separator, _ = suffix.partition("_shape:")
        if not separator:
            return None
        dimensions = grid_text.split("x")
        if len(dimensions) != 3:
            return None
        try:
            grid = tuple(int(dimension) for dimension in dimensions)
        except ValueError:
            return None
        if any(dimension <= 0 for dimension in grid):
            return None
        return grid

    def is_compatible_resized_key(
        self,
        key: str,
        grid: tuple[int, ...],
        tensor: torch.Tensor,
    ) -> bool:
        """Check whether ``key`` exactly matches the current tensor layout."""
        mm_hash = self.get_mm_hash_from_resized_key(key)
        if mm_hash is None:
            return False
        return key == self.get_resized_key(
            mm_hash,
            grid,
            tensor.shape,
            tensor.dtype,
        )

    def put(
        self,
        mm_hash: str,
        tensor: torch.Tensor,
        grid: tuple[int, ...],
    ) -> bool:
        key = self.get_resized_key(mm_hash, grid, tensor.shape, tensor.dtype)
        if self._backend.exists([key]) == [1]:
            return True
        tensor = tensor.detach().to(device="cpu").contiguous()
        if tensor.nbytes > MAX_RESIZED_IMAGE_TRANSFER_SIZE:
            logger.warning(
                "EC resized tensor put skipped: key=%s bytes=%d max_bytes=%d",
                key,
                tensor.nbytes,
                MAX_RESIZED_IMAGE_TRANSFER_SIZE,
            )
            return False

        # The external Memcache SDK cannot wait on a PyTorch producer stream.
        self._synchronize_npu(tensor)
        tensor_ptr = tensor.data_ptr()
        with self._put_lock:
            registered = False
            try:
                self._backend.register_buffer([tensor_ptr], [tensor.nbytes])
                registered = True
                result = self._backend.put_from(
                    key,
                    tensor_ptr,
                    tensor.nbytes,
                    MmcDirect.COPY_H2G.value,
                )
            finally:
                if registered:
                    self._backend.unregister_buffer([tensor_ptr], [tensor.nbytes])
        if result != 0:
            logger.warning("EC resized tensor put failed: key=%s result=%s", key, result)
            return False
        return True

    def batch_get_resized_key_infos(
        self,
        resized_keys: list[str],
        current_resized_tensor: torch.Tensor,
    ) -> tuple[list[str], list[str], list[Any]]:
        if not resized_keys:
            return [], [], []

        key_infos = self._backend.batch_get_key_info(resized_keys)

        if not key_infos or len(key_infos) != len(resized_keys):
            logger.warning(
                "EC resized tensor batch key info mismatch: keys=%d key_infos=%d",
                len(resized_keys),
                len(key_infos) if key_infos else 0,
            )
            return resized_keys, [], key_infos
        expected_nbytes = current_resized_tensor.nbytes
        not_exist_keys: list[str] = []
        exist_keys: list[str] = []
        exist_key_infos: list[Any] = []
        for key, key_info in zip(resized_keys, key_infos):
            nbytes = int(key_info.size())
            if nbytes <= 0:
                not_exist_keys.append(key)
                continue
            if nbytes != expected_nbytes:
                logger.warning(
                    "EC resized tensor size mismatch: key=%s expected=%d actual=%d",
                    key,
                    expected_nbytes,
                    nbytes,
                )
                not_exist_keys.append(key)
                continue
            if nbytes > MAX_RESIZED_IMAGE_TRANSFER_SIZE:
                logger.warning(
                    "EC resized tensor get skipped: key=%s bytes=%d max_bytes=%d",
                    key,
                    nbytes,
                    MAX_RESIZED_IMAGE_TRANSFER_SIZE,
                )
                not_exist_keys.append(key)
                continue
            exist_keys.append(key)
            exist_key_infos.append(key_info)

        return not_exist_keys, exist_keys, exist_key_infos

    def clarify_exist_requests(
        self,
        exist_keys: list[str],
        key_infos: list[Any],
    ) -> tuple[list[ResizedTensorLoadRequest], list[ResizedTensorLoadRequest]]:

        regular_requests: list[ResizedTensorLoadRequest] = []
        oversized_requests: list[ResizedTensorLoadRequest] = []
        for index, (key, key_info) in enumerate(zip(exist_keys, key_infos)):
            nbytes = int(key_info.size())
            request = ResizedTensorLoadRequest(
                output_index=index,
                resized_key=key,
                nbytes=nbytes,
            )
            if nbytes <= self._recv_buffer_size:
                regular_requests.append(request)
            else:
                oversized_requests.append(request)
        return regular_requests, oversized_requests

    def select_ssim_from_regular_batches(
        self,
        regular_requests_batches: list[list[ResizedTensorLoadRequest]],
        current_resized_tensor: torch.Tensor,
        ssim_threshold: float,
    ) -> list[tuple[float, str]]:
        """Return the first batch containing a qualified SSIM result."""
        recv_buffer = self._ensure_recv_buffer()
        calculated_scores: list[tuple[float, str]] = []
        for regular_requests_batch in regular_requests_batches:
            local_request_batch = [
                ResizedTensorLoadRequest(
                    output_index=local_index,
                    resized_key=request.resized_key,
                    nbytes=request.nbytes,
                )
                for local_index, request in enumerate(regular_requests_batch)
            ]
            regular_batch_outputs: list[torch.Tensor | None] = [None] * len(local_request_batch)
            self._load_batch_into_buffer(
                local_request_batch,
                regular_batch_outputs,
                recv_buffer,
                current_resized_tensor,
            )

            batch_scores = self.calculate_ssim_results(
                current_resized_tensor,
                regular_batch_outputs,
                local_request_batch,
            )
            calculated_scores.extend(batch_scores)
            calculated_scores.sort(key=lambda item: (-item[0], item[1]))
            if batch_scores and max(score for score, _ in batch_scores) > ssim_threshold:
                logger.debug(
                    "EC similarity regular batch hit: candidate=%s ssim=%.6f threshold=%.6f",
                    calculated_scores[0][1],
                    calculated_scores[0][0],
                    ssim_threshold,
                )
                return calculated_scores
        return calculated_scores

    def select_ssim_from_oversized_batches(
        self,
        oversized_requests_batches: list[list[ResizedTensorLoadRequest]],
        current_resized_tensor: torch.Tensor,
        ssim_threshold: float,
    ) -> list[tuple[float, str]]:
        """Score oversized batches and stop after the first qualified batch."""
        if not oversized_requests_batches:
            return []

        temporary_buffer_size = max(
            request.nbytes for request_batch in oversized_requests_batches for request in request_batch
        )
        logger.warning(
            "EC resized tensors exceed the configured persistent receive "
            "buffer; using one temporary buffer: temporary_bytes=%d "
            "persistent_bytes=%d",
            temporary_buffer_size,
            self._recv_buffer_size,
        )
        temporary_buffer = torch.empty(
            temporary_buffer_size,
            dtype=torch.uint8,
            device="cpu",
        )
        temporary_buffer_ptr = temporary_buffer.data_ptr()
        registered = False
        calculated_scores: list[tuple[float, str]] = []
        try:
            self._backend.register_buffer(
                [temporary_buffer_ptr],
                [temporary_buffer.nbytes],
            )
            registered = True
            for oversized_requests_batch in oversized_requests_batches:
                local_request_batch = [
                    ResizedTensorLoadRequest(
                        output_index=local_index,
                        resized_key=request.resized_key,
                        nbytes=request.nbytes,
                    )
                    for local_index, request in enumerate(oversized_requests_batch)
                ]
                oversized_batch_outputs: list[torch.Tensor | None] = [None] * len(local_request_batch)
                self._load_batch_into_buffer(
                    local_request_batch,
                    oversized_batch_outputs,
                    temporary_buffer,
                    current_resized_tensor,
                )
                batch_scores = self.calculate_ssim_results(
                    current_resized_tensor,
                    oversized_batch_outputs,
                    local_request_batch,
                )
                calculated_scores.extend(batch_scores)
                calculated_scores.sort(key=lambda item: (-item[0], item[1]))
                if batch_scores and max(score for score, _ in batch_scores) > ssim_threshold:
                    logger.debug(
                        "EC similarity oversized batch hit: candidate=%s ssim=%.6f threshold=%.6f",
                        calculated_scores[0][1],
                        calculated_scores[0][0],
                        ssim_threshold,
                    )
                    return calculated_scores
            return calculated_scores
        finally:
            if registered:
                self._backend.unregister_buffer(
                    [temporary_buffer_ptr],
                    [temporary_buffer.nbytes],
                )

    def calculate_ssim_results(
        self,
        current_resized_tensor: torch.Tensor,
        candidate_tensors: list[torch.Tensor | None],
        request_batch: list[ResizedTensorLoadRequest],
    ) -> list[tuple[float, str]]:
        """Calculate SSIM for one loaded batch while preserving key order."""
        if len(candidate_tensors) != len(request_batch):
            raise ValueError(
                "candidate_tensors and request_batch must have the same "
                f"length, got {len(candidate_tensors)} and "
                f"{len(request_batch)}"
            )

        score_resized_keys: list[str] = []
        score_tensors: list[torch.Tensor] = []
        # Failed Memcache reads leave a None at the corresponding position and
        # are not scored.
        for candidate_tensor, request in zip(candidate_tensors, request_batch):
            if candidate_tensor is None:
                continue
            score_resized_keys.append(request.resized_key)
            score_tensors.append(candidate_tensor)

        candidate_count = len(score_tensors)
        if candidate_count == 0:
            return []

        current_grid = self.get_grid_from_resized_key(score_resized_keys[0])
        if current_grid is None:
            return []
        current_gray_plane = build_patch_gray_plane(
            current_resized_tensor,
            current_grid,
            merge_size=self._merge_size,
        )
        if current_gray_plane is None:
            return []

        cpu_only = current_gray_plane.device.type == "cpu" and all(
            tensor.device.type == "cpu" for tensor in score_tensors
        )
        if candidate_count <= _PARALLEL_SIMILARITY_COMPUTE_THRESHOLD or not cpu_only:
            logger.debug(
                "EC similarity SSIM scoring: size=%d mode=serial",
                candidate_count,
            )
            score_results = [
                self._score_candidate(
                    current_gray_plane,
                    resized_key,
                    candidate_tensor,
                )
                for resized_key, candidate_tensor in zip(score_resized_keys, score_tensors)
            ]
        else:
            logger.debug(
                "EC similarity SSIM scoring: size=%d mode=parallel workers=%d",
                candidate_count,
                min(candidate_count, _MAX_SIMILARITY_COMPUTE_WORKERS),
            )
            futures = [
                self._similarity_compute_executor.submit(
                    self._score_candidate,
                    current_gray_plane,
                    resized_key,
                    candidate_tensor,
                )
                for resized_key, candidate_tensor in zip(score_resized_keys, score_tensors)
            ]
            # Finish all readers even if a task fails before releasing tensors.
            wait(futures)
            score_results = [future.result() for future in futures]

        return [result for result in score_results if result is not None]

    def _score_candidate(
        self,
        current_gray_plane: torch.Tensor,
        candidate_resized_key: str,
        candidate_tensor: torch.Tensor,
    ) -> tuple[float, str] | None:

        candidate_grid = self.get_grid_from_resized_key(candidate_resized_key)
        candidate_mm_hash = self.get_mm_hash_from_resized_key(candidate_resized_key)
        if candidate_grid is None or candidate_mm_hash is None:
            return None
        candidate_gray = build_patch_gray_plane(
            candidate_tensor,
            candidate_grid,
            merge_size=self._merge_size,
        )
        if candidate_gray is None:
            return None
        score = ssim_score(current_gray_plane, candidate_gray)
        if score is None:
            return None
        return score, candidate_mm_hash

    @staticmethod
    def get_transfer_direction(device: torch.device, *, to_store: bool) -> int:
        """Return an SDK direction; current pixel transfers use fixed host directions."""
        if device.type == "cpu":
            return MmcDirect.COPY_H2G.value if to_store else MmcDirect.COPY_G2H.value
        if device.type == "npu":
            return MmcDirect.COPY_L2G.value if to_store else MmcDirect.COPY_G2L.value
        raise ValueError(f"Unsupported EC resized tensor device: {device}")

    @staticmethod
    def _synchronize_npu(tensor: torch.Tensor) -> None:
        if tensor.device.type == "npu":
            torch.npu.current_stream(tensor.device).synchronize()

    def _ensure_recv_buffer(self) -> torch.Tensor:
        recv_buffer = self._recv_buffer
        if recv_buffer is not None:
            return recv_buffer
        recv_buffer = torch.empty(
            self._recv_buffer_size,
            dtype=torch.uint8,
            device="cpu",
        )
        self._backend.register_buffer(
            [recv_buffer.data_ptr()],
            [recv_buffer.nbytes],
        )
        self._recv_buffer = recv_buffer
        self._recv_buffer_registered = True
        logger.info(
            "Registered EC resized-image receive buffer: bytes=%d",
            recv_buffer.nbytes,
        )
        return recv_buffer

    def _load_batch_into_buffer(
        self,
        request_batch: list[ResizedTensorLoadRequest],
        outputs: list[torch.Tensor | None],
        recv_buffer: torch.Tensor,
        current_resized_tensor: torch.Tensor,
    ) -> None:
        offsets: list[int] = []
        offset_bytes = 0
        for request in request_batch:
            offsets.append(offset_bytes)
            offset_bytes += request.nbytes

        base_addr = recv_buffer.data_ptr()
        # Allocation/reuse may follow device work on the caller stream.
        self._synchronize_npu(recv_buffer)
        results = self._backend.batch_get_into_buffers(
            [request.resized_key for request in request_batch],
            [base_addr + offset for offset in offsets],
            [request.nbytes for request in request_batch],
            MmcDirect.COPY_G2H.value,
        )
        if results is None or len(results) != len(request_batch):
            logger.warning(
                "EC resized tensor batch get result mismatch: keys=%d results=%d",
                len(request_batch),
                len(results) if results is not None else 0,
            )
            return

        output_shape = tuple(current_resized_tensor.shape)
        output_dtype = current_resized_tensor.dtype
        try:
            for request, offset, result in zip(
                request_batch,
                offsets,
                results,
            ):
                if result == 0:
                    outputs[request.output_index] = (
                        recv_buffer[offset : offset + request.nbytes]
                        .view(output_dtype)
                        .reshape(output_shape)
                        .clone()
                    )
                else:
                    logger.warning(
                        "EC resized tensor get failed: key=%s result=%s",
                        request.resized_key,
                        result,
                    )
        finally:
            # Device clones are asynchronous. Finish every queued read before
            # the next SDK GET overwrites or unregisters this shared buffer.
            self._synchronize_npu(recv_buffer)
        return

    @staticmethod
    def _build_capacity_batches(
        requests: list[ResizedTensorLoadRequest],
        capacity_bytes: int,
    ) -> list[list[ResizedTensorLoadRequest]]:
        batches: list[list[ResizedTensorLoadRequest]] = []
        current_batch: list[ResizedTensorLoadRequest] = []
        current_bytes = 0
        for request in requests:
            if request.nbytes > capacity_bytes:
                raise ValueError(
                    "Resized tensor exceeds receive buffer capacity: "
                    f"key={request.resized_key} bytes={request.nbytes} "
                    f"capacity_bytes={capacity_bytes}"
                )
            if current_batch and current_bytes + request.nbytes > capacity_bytes:
                batches.append(current_batch)
                current_batch = []
                current_bytes = 0
            current_batch.append(request)
            current_bytes += request.nbytes
        if current_batch:
            batches.append(current_batch)
        return batches

    def build_request_batches(
        self,
        regular_requests: list[ResizedTensorLoadRequest],
        oversized_requests: list[ResizedTensorLoadRequest],
    ) -> tuple[
        list[list[ResizedTensorLoadRequest]],
        list[list[ResizedTensorLoadRequest]],
    ]:
        regular_requests_batches: list[list[ResizedTensorLoadRequest]] = []
        oversized_requests_batches: list[list[ResizedTensorLoadRequest]] = []
        if regular_requests:
            regular_requests_batches = self._build_capacity_batches(
                regular_requests,
                self._recv_buffer_size,
            )

        if oversized_requests:
            temporary_buffer_size = max(request.nbytes for request in oversized_requests)
            oversized_requests_batches = self._build_capacity_batches(
                oversized_requests,
                temporary_buffer_size,
            )
        return regular_requests_batches, oversized_requests_batches

    def shutdown(self) -> None:
        self._similarity_compute_executor.shutdown(wait=True)
        recv_buffer = self._recv_buffer
        if recv_buffer is None:
            return
        self._synchronize_npu(recv_buffer)
        if self._recv_buffer_registered:
            self._backend.unregister_buffer(
                [recv_buffer.data_ptr()],
                [recv_buffer.nbytes],
            )
            self._recv_buffer_registered = False
        self._recv_buffer = None
