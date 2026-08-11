# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend worker for vLLM's native CPU KV-cache offloading."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import torch
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import (
    CPUOffloadingWorker,
    compute_sub_block_ptrs,
)

# Direction codes shared with csrc/torch_binding.cpp::swap_blocks_batch.
DIRECTION_H2D = 0
DIRECTION_D2H = 1


@dataclass
class Transfer:
    job_id: int
    stream: torch.npu.Stream
    start_event: torch.npu.Event
    end_event: torch.npu.Event
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin_memory = is_pin_memory_available()
    return (
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin_memory),
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin_memory),
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin_memory),
    )


class SingleDirectionNPUOffloadingHandler:
    """Transfer blocks in one direction using an ordered NPU DMA stream."""

    def __init__(
        self,
        npu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        npu_to_cpu: bool,
    ) -> None:
        assert len(npu_tensors) == len(cpu_tensors)
        assert npu_tensors

        for npu_tensor, cpu_tensor in zip(npu_tensors, cpu_tensors):
            assert npu_tensor.dtype == torch.int8
            assert npu_tensor.ndim == 2
            assert npu_tensor.device.type == "npu"
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            assert cpu_tensor.shape[1] == (npu_tensor.shape[1] * blocks_per_chunk)

        # Keep independent lists in each handler so one direction can be shut
        # down without releasing tensors still referenced by the other.
        self.src_tensors = list(npu_tensors if npu_to_cpu else cpu_tensors)
        self.dst_tensors = list(cpu_tensors if npu_to_cpu else npu_tensors)
        self.npu_to_cpu = npu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs
        self.src_blocks_per_chunk = 1 if npu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if npu_to_cpu else 1
        self.direction = DIRECTION_D2H if npu_to_cpu else DIRECTION_H2D

        self._transfer_events: dict[int, torch.npu.Event] = {}
        self._transfers: deque[Transfer] = deque()
        self._stream_pool: list[torch.npu.Stream] = []
        self._event_pool: list[torch.npu.Event] = []
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def transfer_async(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        npu_spec = src_spec if self.npu_to_cpu else dst_spec
        assert isinstance(npu_spec, GPULoadStoreSpec)
        group_sizes = npu_spec.group_sizes
        block_indices = npu_spec.block_indices
        assert len(group_sizes) == len(self.kv_cache_groups_data_refs)
        assert len(block_indices) == len(self.kv_cache_groups_data_refs)

        num_copy_ops = sum(
            group_size * len(group_data_refs)
            for group_size, group_data_refs in zip(group_sizes, self.kv_cache_groups_data_refs)
        )
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop() if self._buffer_pool else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
        all_src = src.numpy()
        all_dst = dst.numpy()
        all_sizes = sizes.numpy()

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        num_transfer_bytes = 0
        for group_size, block_idx, group_data_refs in zip(
            group_sizes,
            block_indices,
            self.kv_cache_groups_data_refs,
        ):
            if group_size == 0:
                continue

            src_blocks_to_skip = block_idx % self.src_blocks_per_chunk
            dst_blocks_to_skip = block_idx % self.dst_blocks_per_chunk
            src_logical_block_count = group_size + src_blocks_to_skip
            dst_logical_block_count = group_size + dst_blocks_to_skip

            src_block_count = cdiv(src_logical_block_count, self.src_blocks_per_chunk)
            dst_block_count = cdiv(dst_logical_block_count, self.dst_blocks_per_chunk)
            src_end_offset = src_offset + src_block_count
            dst_end_offset = dst_offset + dst_block_count
            assert src_end_offset <= len(src_blocks)
            assert dst_end_offset <= len(dst_blocks)

            group_src = src_blocks[src_offset:src_end_offset]
            group_dst = dst_blocks[dst_offset:dst_end_offset]
            for data_ref in group_data_refs:
                tensor_idx = data_ref.tensor_idx
                end_idx = op_idx + group_size
                compute_sub_block_ptrs(
                    group_src,
                    self.src_blocks_per_chunk,
                    all_src[op_idx:end_idx],
                    self.src_tensors[tensor_idx],
                    skip_count=src_blocks_to_skip,
                )
                compute_sub_block_ptrs(
                    group_dst,
                    self.dst_blocks_per_chunk,
                    all_dst[op_idx:end_idx],
                    self.dst_tensors[tensor_idx],
                    skip_count=dst_blocks_to_skip,
                )
                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == len(src_blocks)
        assert dst_offset == len(dst_blocks)
        assert op_idx == num_copy_ops

        stream = self._stream_pool.pop() if self._stream_pool else torch.npu.Stream()
        start_event = self._event_pool.pop() if self._event_pool else torch.npu.Event(enable_timing=True)
        end_event = self._event_pool.pop() if self._event_pool else torch.npu.Event(enable_timing=True)

        if self.npu_to_cpu:
            # KV writes happen on the model stream. D2H must not read a block
            # until those writes are complete.
            stream.wait_stream(torch.npu.current_stream())
        if self._transfers:
            stream.wait_event(self._transfers[-1].end_event)

        with torch.npu.stream(stream):
            start_event.record(stream)
            if num_copy_ops > 0:
                torch.ops._C_ascend.swap_blocks_batch(
                    src,
                    dst,
                    sizes,
                    self.direction,
                )
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_sizes=batch_sizes,
            )
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.num_bytes,
                    transfer_time=transfer_time,
                )
            )
            self._stream_pool.append(transfer.stream)
            self._event_pool.extend((transfer.end_event, transfer.start_event))
            self._buffer_pool.append(
                (
                    transfer.batch_src,
                    transfer.batch_dst,
                    transfer.batch_sizes,
                )
            )
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        while self._transfers:
            transfer = self._transfers.popleft()
            transfer.end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()


class NPUOffloadingWorker(CPUOffloadingWorker):
    """Native CPU offloading worker backed by Ascend batched DMA."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
    ) -> None:
        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))

        npu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            npu_page_size_bytes = kv_cache_tensor.page_size_bytes
            npu_tensor = kv_cache_tensor.tensor
            if npu_tensor.dtype != torch.int8 or npu_tensor.ndim != 2:
                raise ValueError(
                    "Canonical NPU KV cache tensors must be two-dimensional "
                    f"int8 views, got shape={tuple(npu_tensor.shape)}, "
                    f"dtype={npu_tensor.dtype}"
                )
            if npu_tensor.shape[1] != npu_page_size_bytes:
                raise ValueError(
                    "Canonical NPU KV cache page size mismatch: "
                    f"shape[1]={npu_tensor.shape[1]}, "
                    f"page_size_bytes={npu_page_size_bytes}"
                )
            cpu_page_size_bytes = npu_page_size_bytes * blocks_per_chunk

            start_time = time.monotonic()
            cpu_tensor = torch.zeros(
                (num_cpu_blocks, cpu_page_size_bytes),
                dtype=torch.int8,
                device="cpu",
                pin_memory=pin_memory,
            )
            logger.debug(
                "Allocated pinned CPU tensor %d x %d (%.2f GB): %.3f s",
                num_cpu_blocks,
                cpu_page_size_bytes,
                num_cpu_blocks * cpu_page_size_bytes / 1e9,
                time.monotonic() - start_time,
            )
            npu_tensors.append(npu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionNPUOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=True,
        )
        self._load_handler = SingleDirectionNPUOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=False,
        )
