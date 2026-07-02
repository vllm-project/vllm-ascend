"""Ascend host-side KV cache offload handler.

The backend stores offloaded KV blocks in ACL host memory registered with
``aclrtHostRegister``.  The registered device pointer is wrapped as NPU storage
so regular tensor copies can move blocks between NPU HBM and host DRAM through
Ascend SDMA without routing through CPU tensors or per-job event polling.
"""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCacheRef, CanonicalKVCaches
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult, TransferSpec

DEFAULT_ALLOC_CHUNK_BLOCKS = 256

# Load libascendcl.so once at module level for efficiency
_libacl = ctypes.CDLL("libascendcl.so")


@dataclass(frozen=True)
class HostAllocation:
    host_ptr: int
    dev_ptr: int
    size_bytes: int


@dataclass(frozen=True)
class HostTensorChunk:
    start: int
    end: int
    tensor: torch.Tensor


def _acl_malloc_host(size_bytes: int) -> tuple[int, int]:
    host_ptr = ctypes.c_void_p()
    ret = _libacl.aclrtMallocHost(ctypes.byref(host_ptr), ctypes.c_size_t(size_bytes))
    if ret != 0:
        raise RuntimeError(f"aclrtMallocHost failed: error {ret} (size={size_bytes})")

    dev_ptr = ctypes.c_void_p()
    ret = _libacl.aclrtHostRegister(host_ptr, ctypes.c_uint64(size_bytes), ctypes.c_int(0), ctypes.byref(dev_ptr))
    if ret != 0:
        _libacl.aclrtFreeHost(host_ptr)
        raise RuntimeError(f"aclrtHostRegister failed: error {ret} (size={size_bytes})")

    return host_ptr.value, dev_ptr.value


def _compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for idx in range(len(shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * shape[idx + 1]
    return tuple(strides)


def _make_host_tensor(
    dev_ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    size_bytes: int,
) -> torch.Tensor:
    storage = torch_npu._C._construct_storage_from_data_pointer(dev_ptr, device, size_bytes)
    return torch.empty(0, dtype=dtype, device=device).set_(storage, 0, shape, _compact_strides(shape))


def _get_block_ids(spec) -> np.ndarray:
    if hasattr(spec, "block_ids"):
        return spec.block_ids
    if hasattr(spec, "block_indices"):
        return spec.block_indices
    raise AttributeError(f"{type(spec).__name__} has neither block_ids nor block_indices")


def _expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
) -> None:
    assert skip_count < block_size_factor
    bases = block_ids * block_size_factor
    offsets = np.arange(block_size_factor)
    expanded = (bases[:, None] + offsets[None, :]).ravel()
    if skip_count:
        expanded = expanded[skip_count:]
    output[: len(expanded)] = expanded


class HostSideOffloadingHandler(OffloadingHandler):
    """Host-DRAM offloading handler for Ascend NPU KV cache tensors.

    The handler consumes vLLM's canonical KV cache representation, where every
    backing tensor has the block dimension first and group membership is carried
    by CanonicalKVCacheRef entries.
    """

    def __init__(
        self,
        block_size_factor: int,
        num_cpu_blocks: int,
        gpu_caches: CanonicalKVCaches,
        max_host_memory_bytes: int | None = None,
        alloc_chunk_blocks: int = DEFAULT_ALLOC_CHUNK_BLOCKS,
    ):
        self.block_size_factor = block_size_factor
        self.num_cpu_blocks = num_cpu_blocks
        self.alloc_chunk_blocks = max(1, alloc_chunk_blocks)

        # Guard against empty gpu_caches
        if not gpu_caches.tensors:
            raise RuntimeError("gpu_caches.tensors is empty, cannot derive block shape for host allocation")

        self.d2h_stream = torch.npu.Stream()
        self.h2d_stream = torch.npu.Stream()
        self._finished: list[TransferResult] = []
        self._done_jobs: set[int] = set()
        self._allocations: list[HostAllocation] = []

        self.npu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[tuple[HostTensorChunk, ...]] = []
        self.kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = (
            gpu_caches.group_data_refs
        )
        self._transfer_count = 0
        self._transfer_blocks_by_type: dict[str, int] = {"NPU->CPU": 0, "CPU->NPU": 0}

        planned_bytes = self._estimate_host_memory_bytes(gpu_caches)
        if max_host_memory_bytes is not None and planned_bytes > max_host_memory_bytes:
            raise RuntimeError(
                f"Host-side KV offload requested {planned_bytes / float(2**30):.3f} GiB, "
                f"exceeding the configured limit {max_host_memory_bytes / float(2**30):.3f} GiB. "
                "Reduce num_cpu_blocks or increase VLLM_ASCEND_KV_HOST_SIDE_MAX_MEMORY_BYTES."
            )

        logger.warning(
            "HostSideOffloadingHandler allocating %d host blocks, block_size_factor=%d, planned=%.3f GiB",
            num_cpu_blocks,
            self.block_size_factor,
            planned_bytes / float(2**30),
        )

        total_bytes = 0
        for kv_cache_tensor in gpu_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            chunks = self._alloc_host_tensor_chunks(
                tensor.shape, tensor.dtype, tensor.device
            )
            total_bytes += sum(
                chunk.tensor.numel() * chunk.tensor.element_size()
                for chunk in chunks
            )
            self.npu_tensors.append(tensor)
            self.cpu_tensors.append(chunks)

        logger.warning("HostSideOffloadingHandler allocated %.3f GiB host-mapped DRAM", total_bytes / float(2**30))

    def _estimate_host_memory_bytes(self, gpu_caches: CanonicalKVCaches) -> int:
        total = 0
        for kv_cache_tensor in gpu_caches.tensors:
            tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, kv_cache_tensor.page_size_bytes)
            )
            shape = list(tensor.shape)
            shape[0] = self.num_cpu_blocks * self.block_size_factor
            total += int(np.prod(shape)) * tensor.element_size()
        return total

    def _alloc_host_tensor(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        size_bytes = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
        host_ptr, dev_ptr = _acl_malloc_host(size_bytes)
        self._allocations.append(HostAllocation(host_ptr, dev_ptr, size_bytes))
        return _make_host_tensor(dev_ptr, shape, dtype, device, size_bytes)

    def _alloc_host_tensor_chunks(
        self,
        gpu_shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[HostTensorChunk, ...]:
        chunks: list[HostTensorChunk] = []
        for block_start in range(0, self.num_cpu_blocks, self.alloc_chunk_blocks):
            block_count = min(self.alloc_chunk_blocks, self.num_cpu_blocks - block_start)
            subblock_start = block_start * self.block_size_factor
            subblock_end = (block_start + block_count) * self.block_size_factor
            shape = list(gpu_shape)
            shape[0] = block_count * self.block_size_factor
            chunks.append(
                HostTensorChunk(
                    start=subblock_start,
                    end=subblock_end,
                    tensor=self._alloc_host_tensor(tuple(shape), dtype, device),
                )
            )
        return tuple(chunks)

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        started = time.perf_counter()
        src_spec, dst_spec = spec
        if isinstance(src_spec, CPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            transfer_type = "CPU->NPU"
            stream = self.h2d_stream
            src_tensors = self.cpu_tensors
            dst_tensors = self.npu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, CPULoadStoreSpec)
            transfer_type = "NPU->CPU"
            stream = self.d2h_stream
            src_tensors = self.npu_tensors
            dst_tensors = self.cpu_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor

        src_blocks = _get_block_ids(src_spec)
        dst_blocks = _get_block_ids(dst_spec)
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        block_count = 0
        if isinstance(src_spec, GPULoadStoreSpec):
            stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for group_id, group_size, block_idx, src_group_blocks, dst_group_blocks in self._grouped_specs(
                src_spec, dst_spec, src_block_size_factor, dst_block_size_factor
            ):
                if group_size == 0:
                    continue
                block_count += int(group_size)

                src_sub_blocks_to_skip = block_idx % src_block_size_factor
                dst_sub_blocks_to_skip = block_idx % dst_block_size_factor
                src_logical = np.empty(
                    src_group_blocks.size * src_block_size_factor
                    - src_sub_blocks_to_skip,
                    dtype=np.int64,
                )
                dst_logical = np.empty(
                    dst_group_blocks.size * dst_block_size_factor
                    - dst_sub_blocks_to_skip,
                    dtype=np.int64,
                )
                _expand_block_ids(
                    src_group_blocks,
                    src_block_size_factor,
                    src_logical,
                    skip_count=src_sub_blocks_to_skip,
                )
                _expand_block_ids(
                    dst_group_blocks,
                    dst_block_size_factor,
                    dst_logical,
                    skip_count=dst_sub_blocks_to_skip,
                )
                assert src_logical.size >= group_size
                assert dst_logical.size >= group_size
                src_to_dst = np.stack(
                    (src_logical[:group_size], dst_logical[:group_size]),
                    axis=1,
                ).astype(np.int64, copy=False)

                for data_ref in self.kv_cache_groups_data_refs[group_id]:
                    tensor_idx = data_ref.tensor_idx
                    self._copy_blocks(
                        src_tensors[tensor_idx],
                        dst_tensors[tensor_idx],
                        src_to_dst,
                    )
            stream.synchronize()

        try:
            result = TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=None,
                transfer_time=time.perf_counter() - started,
                transfer_type=(src_spec.medium(), dst_spec.medium()),
            )
        except TypeError:
            result = (job_id, True)
        self._finished.append(result)
        self._done_jobs.add(job_id)
        self._record_transfer(transfer_type, block_count, time.perf_counter() - started)
        return True

    def _record_transfer(self, transfer_type: str, block_count: int, elapsed_s: float) -> None:
        self._transfer_count += 1
        self._transfer_blocks_by_type[transfer_type] += block_count
        if self._transfer_count <= 16 or self._transfer_count % 32 == 0:
            logger.warning(
                "HOST_SIDE_KV_TRANSFER job=%d type=%s blocks=%d elapsed_ms=%.3f totals=%s",
                self._transfer_count,
                transfer_type,
                block_count,
                elapsed_s * 1000.0,
                dict(self._transfer_blocks_by_type),
            )

    @staticmethod
    def _is_chunked(value) -> bool:
        return isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], HostTensorChunk)

    @staticmethod
    def _copy_block_rows(src_tensor: torch.Tensor, dst_tensor: torch.Tensor, src_to_dst: np.ndarray) -> None:
        for src_idx, dst_idx in src_to_dst:
            dst_tensor[int(dst_idx)].copy_(src_tensor[int(src_idx)], non_blocking=True)

    def _copy_blocks(self, src_tensor, dst_tensor, src_to_dst: np.ndarray) -> None:
        if self._is_chunked(dst_tensor):
            for chunk in dst_tensor:
                mask = (src_to_dst[:, 1] >= chunk.start) & (src_to_dst[:, 1] < chunk.end)
                if not np.any(mask):
                    continue
                chunk_map = src_to_dst[mask].copy()
                chunk_map[:, 1] -= chunk.start
                self._copy_block_rows(src_tensor, chunk.tensor, chunk_map)
            return
        if self._is_chunked(src_tensor):
            for chunk in src_tensor:
                mask = (src_to_dst[:, 0] >= chunk.start) & (src_to_dst[:, 0] < chunk.end)
                if not np.any(mask):
                    continue
                chunk_map = src_to_dst[mask].copy()
                chunk_map[:, 0] -= chunk.start
                self._copy_block_rows(chunk.tensor, dst_tensor, chunk_map)
            return
        self._copy_block_rows(src_tensor, dst_tensor, src_to_dst)

    @staticmethod
    def _grouped_specs(
        src_spec,
        dst_spec,
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ):
        src_block_ids = _get_block_ids(src_spec)
        dst_block_ids = _get_block_ids(dst_spec)
        gpu_spec = src_spec if isinstance(src_spec, GPULoadStoreSpec) else dst_spec
        group_sizes = getattr(gpu_spec, "group_sizes", None)
        block_indices = getattr(gpu_spec, "block_indices", None)
        if group_sizes is None:
            group_size = len(gpu_spec.block_ids)
            yield 0, group_size, 0, src_block_ids, dst_block_ids
            return

        if block_indices is None:
            block_indices = (0,) * len(group_sizes)
        assert len(block_indices) == len(group_sizes)
        src_offset = 0
        dst_offset = 0
        for group_id, (group_size, block_idx) in enumerate(zip(group_sizes, block_indices)):
            src_logical_blocks_to_skip = block_idx % src_block_size_factor
            dst_logical_blocks_to_skip = block_idx % dst_block_size_factor
            src_size = math.ceil(
                (group_size + src_logical_blocks_to_skip) / src_block_size_factor
            )
            dst_size = math.ceil(
                (group_size + dst_logical_blocks_to_skip) / dst_block_size_factor
            )
            yield (
                group_id,
                group_size,
                block_idx,
                src_block_ids[src_offset : src_offset + src_size],
                dst_block_ids[dst_offset : dst_offset + dst_size],
            )
            src_offset += src_size
            dst_offset += dst_size

    def get_finished(self) -> list[TransferResult]:
        finished = self._finished
        self._finished = []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        return

    def __del__(self):
        for alloc in self._allocations:
            try:
                _libacl.aclrtHostUnregister(ctypes.c_void_p(alloc.host_ptr))
                _libacl.aclrtFreeHost(ctypes.c_void_p(alloc.host_ptr))
            except Exception:
                pass
