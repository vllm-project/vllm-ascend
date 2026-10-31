# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Paired Host-main/NPU-indexer transfers for sparse KV prefix caching."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import CanonicalKVCaches, GPULoadStoreSpec, OffloadingWorker, TransferResult

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import NPUOffloadingWorker


def _select_tensors(caches: CanonicalKVCaches, device: str) -> CanonicalKVCaches:
    indices = [i for i, item in enumerate(caches.tensors) if item.tensor.device.type == device]
    remap = {old: new for new, old in enumerate(indices)}
    return CanonicalKVCaches(
        tensors=[caches.tensors[i] for i in indices],
        group_data_refs=[
            [replace(ref, tensor_idx=remap[ref.tensor_idx]) for ref in refs if ref.tensor_idx in remap]
            for refs in caches.group_data_refs
        ],
    )


class SparsePrefixOffloadingWorker(OffloadingWorker):
    """Join CPU main-KV copies with the existing asynchronous NPU offloader.

    Main KV is shared within TP. Only TP0 saves/restores it; every rank still
    acknowledges the complete job, so upstream never publishes a paired cache
    key or releases a load before all indexer and main-KV writes complete.
    The upstream scheduler fences source blocks and owns key publication/LRU.
    """

    def __init__(self, caches: CanonicalKVCaches, blocks_per_chunk: int, num_chunks: int, *, tp_rank: int) -> None:
        self._host = _select_tensors(caches, "cpu")
        device = _select_tensors(caches, "npu")
        if not self._host.tensors or not device.tensors:
            raise ValueError("Sparse prefix offload requires both Host main KV and NPU indexer tensors")
        self._blocks_per_chunk = blocks_per_chunk
        self._tp_rank = tp_rank
        self._cpu = (
            [
                torch.empty((num_chunks, item.page_size_bytes * blocks_per_chunk), dtype=torch.int8)
                for item in self._host.tensors
            ]
            if tp_rank == 0
            else []
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="sparse-prefix",
            # NPU event synchronization needs this worker's device context
            # in the background thread as well as in the model thread.
            initializer=torch.npu.set_device,
            initargs=(torch.npu.current_device(),),
        )
        try:
            self._device = NPUOffloadingWorker(device, blocks_per_chunk, num_chunks)
        except BaseException:
            self._executor.shutdown(wait=True)
            raise
        self._host_jobs: dict[int, Future[tuple[int, float]]] = {}
        self._device_results: dict[int, TransferResult] = {}

    def _copy_host(self, src_spec, dst_spec, storing: bool, ready) -> tuple[int, float]:
        started = time.monotonic()
        if ready is not None:
            # D2H of newly generated main KV runs on TP0's model stream.
            # Waiting in this background thread keeps the scheduler thread free.
            ready.synchronize()
        src_factor, dst_factor = (1, self._blocks_per_chunk) if storing else (self._blocks_per_chunk, 1)
        npu_spec = src_spec if storing else dst_spec
        if not isinstance(npu_spec, GPULoadStoreSpec):
            raise TypeError("Sparse prefix transfer requires a GPU block specification")
        if len(npu_spec.group_sizes) != len(self._host.group_data_refs):
            raise ValueError("Sparse prefix transfer group counts do not match the caches")
        src_offset = dst_offset = copied = 0
        for refs, count, start in zip(self._host.group_data_refs, npu_spec.group_sizes, npu_spec.block_indices):
            src_skip, dst_skip = start % src_factor, start % dst_factor
            for ref in refs:
                live = self._host.tensors[ref.tensor_idx].tensor
                backing = self._cpu[ref.tensor_idx]
                source, destination = (live, backing) if storing else (backing, live)
                page_bytes = live.shape[1]
                for index in range(count):
                    src_block = int(src_spec.block_ids[src_offset + (src_skip + index) // src_factor])
                    dst_block = int(dst_spec.block_ids[dst_offset + (dst_skip + index) // dst_factor])
                    src_begin = (src_skip + index) % src_factor * page_bytes
                    dst_begin = (dst_skip + index) % dst_factor * page_bytes
                    destination[dst_block, dst_begin : dst_begin + ref.page_size_bytes].copy_(
                        source[src_block, src_begin : src_begin + ref.page_size_bytes]
                    )
                    copied += ref.page_size_bytes
            if count:
                src_offset += cdiv(count + src_skip, src_factor)
                dst_offset += cdiv(count + dst_skip, dst_factor)
        if src_offset != len(src_spec.block_ids) or dst_offset != len(dst_spec.block_ids):
            raise ValueError("Sparse prefix transfer block counts do not match its groups")
        return copied, time.monotonic() - started

    def _submit(self, job_id, src_spec, dst_spec, storing: bool) -> bool:
        if job_id in self._host_jobs:
            raise ValueError(f"Duplicate sparse prefix transfer job {job_id}")
        submit = self._device.submit_store if storing else self._device.submit_load
        if not submit(job_id, src_spec, dst_spec):
            return False
        if self._tp_rank == 0:
            ready = torch.npu.current_stream().record_event() if storing else None
            self._host_jobs[job_id] = self._executor.submit(self._copy_host, src_spec, dst_spec, storing, ready)
        else:
            done: Future[tuple[int, float]] = Future()
            done.set_result((0, 0.0))
            self._host_jobs[job_id] = done
        return True

    def submit_store(self, job_id, src_spec, dst_spec) -> bool:
        return self._submit(job_id, src_spec, dst_spec, True)

    def submit_load(self, job_id, src_spec, dst_spec) -> bool:
        return self._submit(job_id, src_spec, dst_spec, False)

    def get_finished(self) -> list[TransferResult]:
        for result in self._device.get_finished():
            self._device_results[result.job_id] = result
        finished = []
        for job_id, future in list(self._host_jobs.items()):
            if not future.done() or job_id not in self._device_results:
                continue
            # Fail closed on a CPU copy failure; never publish its prefix key.
            host_bytes, host_seconds = future.result()
            result = self._device_results.pop(job_id)
            finished.append(
                replace(
                    result,
                    transfer_size=(result.transfer_size or 0) + host_bytes,
                    transfer_time=max(result.transfer_time or 0, host_seconds),
                )
            )
            del self._host_jobs[job_id]
        return finished

    def wait(self, job_ids: set[int]) -> None:
        self._device.wait(job_ids)
        for job_id in job_ids:
            if job_id in self._host_jobs:
                self._host_jobs[job_id].result()

    def shutdown(self) -> None:
        try:
            self._device.shutdown()
        finally:
            self._executor.shutdown(wait=True)
