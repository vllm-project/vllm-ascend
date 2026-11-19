"""Queued Ascend DMA submission with completion events and error propagation."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

# Direction codes used by _C_ascend.swap_blocks_batch.
DIRECTION_H2D = 0
DIRECTION_D2H = 1


class BatchMemcpyParams(NamedTuple):
    src_bases: np.ndarray
    dst_bases: np.ndarray
    block_bytes: np.ndarray
    direction: int


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    direction: int,
) -> BatchMemcpyParams:
    """Cache the base pointers and byte strides of corresponding block buffers."""
    assert list(src_caches) == list(dst_caches), "Source and destination cache keys must match in order"
    src_bases, dst_bases, block_bytes = [], [], []
    for src, dst in zip(src_caches.values(), dst_caches.values()):
        src_bytes = src.stride(0) * src.element_size()
        assert src_bytes == dst.stride(0) * dst.element_size(), "Per-block byte sizes must match"
        src_bases.append(src.data_ptr())
        dst_bases.append(dst.data_ptr())
        block_bytes.append(src_bytes)
    return BatchMemcpyParams(
        np.asarray(src_bases, dtype=np.int64),
        np.asarray(dst_bases, dtype=np.int64),
        np.asarray(block_bytes, dtype=np.int64),
        direction,
    )


def copy_blocks(src_ids: list[int], dst_ids: list[int], params: BatchMemcpyParams) -> None:
    """Submit DMA on the current NPU stream."""
    assert len(src_ids) == len(dst_ids), "Source and destination block counts must match"
    if not src_ids:
        return
    block_bytes = params.block_bytes[:, None]
    src = params.src_bases[:, None] + block_bytes * np.asarray(src_ids, dtype=np.int64)
    dst = params.dst_bases[:, None] + block_bytes * np.asarray(dst_ids, dtype=np.int64)
    sizes = np.broadcast_to(block_bytes, src.shape).ravel().copy()
    torch.ops._C_ascend.swap_blocks_batch(
        torch.from_numpy(src.ravel()), torch.from_numpy(dst.ravel()), torch.from_numpy(sizes), params.direction
    )


@dataclass
class _CopyJob:
    src: list[int]
    dst: list[int]
    params: BatchMemcpyParams
    is_store: bool
    event_idx: int | None
    events: list | None
    submitted: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None


class NPUDmaCopyBackend:
    def __init__(self) -> None:
        self._store_params = self._load_params = None
        self._load_stream = self._store_stream = None
        self._device = None
        self._queue = None
        self._thread = None
        self._shutdown = False
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def init(self, npu_caches, cpu_caches, device, load_stream, store_stream) -> None:
        if self._thread is not None or self._shutdown:
            raise RuntimeError("DMA backend cannot be initialized twice or after shutdown")
        self._device = device
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._store_params = build_params(npu_caches, cpu_caches, DIRECTION_D2H)
        self._load_params = build_params(cpu_caches, npu_caches, DIRECTION_H2D)
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._copy_loop, name="npu-kv-offload-copy", daemon=True)
        self._thread.start()
        self._ready.wait()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Ascend DMA submission failed") from error

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int | None = None,
        events_list: list | None = None,
        *,
        params: BatchMemcpyParams | None = None,
        wait_for_submit: bool = False,
    ) -> None:
        if len(src_blocks) != len(dst_blocks):
            raise ValueError("Source and destination block counts differ")
        if (event_idx is None) != (events_list is None):
            raise ValueError("event_idx and events_list must be supplied together")
        descriptor = params if params is not None else (self._store_params if is_store else self._load_params)
        if descriptor is None or self._queue is None:
            raise RuntimeError("DMA backend is not initialized")
        job = _CopyJob(
            list(src_blocks),
            list(dst_blocks),
            descriptor,
            is_store,
            event_idx,
            events_list,
        )
        with self._lock:
            if self._shutdown:
                raise RuntimeError("DMA backend has shut down")
            if self._error is not None:
                raise RuntimeError("Ascend DMA submission failed") from self._error
            self._queue.put(job)
        if wait_for_submit:
            # Wait for host submission, NOT for DMA completion on the device.
            job.submitted.wait()
            if job.error is not None:
                raise RuntimeError("Ascend DMA submission failed") from job.error
            self.raise_if_failed()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            if self._queue is not None:
                self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _copy_loop(self) -> None:
        try:
            torch.npu.set_device(self._device)
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self.raise_if_failed()
                stream = self._store_stream if job.is_store else self._load_stream
                with torch.npu.stream(stream):
                    copy_blocks(job.src, job.dst, job.params)
                    if job.events is not None:
                        event = torch.npu.Event()
                        event.record(stream)
                        job.events.append((job.event_idx, event))
            except Exception as exc:
                job.error = exc
                with self._lock:
                    if self._error is None:
                        self._error = exc
            finally:
                # Also release queued submitters after any earlier failure.
                job.submitted.set()
