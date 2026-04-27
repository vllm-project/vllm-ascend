# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""DMA copy backend for NPU<->CPU block transfers.

Mirrors :class:`vllm.v1.simple_kv_offload.copy_backend.DmaCopyBackend`
but routes batched memcpy through ``torch.ops._C_ascend.swap_blocks_batch``
and uses ``torch.npu`` streams/events.
"""

from __future__ import annotations

import queue
import threading

import torch

from vllm.logger import init_logger

from vllm_ascend.simple_kv_offload.npu_mem_ops import (
    DIRECTION_D2H,
    DIRECTION_H2D,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class NPUDmaCopyBackend:
    """``aclrtMemcpyBatchAsync`` copy backend running on a worker thread.

    Two pre-built ``BatchMemcpyParams`` are cached (load=H2D, store=D2H).
    Submitted jobs are dispatched in FIFO order to a single worker
    thread; each job issues its copies on a dedicated NPU stream and
    records an Event the main thread can poll without synchronizing
    the device.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.npu.Stream | None = None
        self._store_stream: torch.npu.Stream | None = None
        self._device: torch.device | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        npu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.npu.Stream,
        store_stream: torch.npu.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._device = device
        # Stores go NPU->CPU (D2H), loads go CPU->NPU (H2D).
        self._store_params = build_params(
            npu_caches, cpu_caches, DIRECTION_D2H
        )
        self._load_params = build_params(
            cpu_caches, npu_caches, DIRECTION_H2D
        )

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            name="npu-kv-offload-copy",
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.npu.Event]],
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (src_blocks, dst_blocks, params, is_store, event_idx, events_list)
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Worker thread main loop
    # ------------------------------------------------------------------
    def _copy_loop(self) -> None:
        # NOTE: matches upstream cuda backend semantics — no cross-stream
        # sync. The scheduler manager only schedules stores for blocks
        # whose KV data is **confirmed computed** (see
        # ``confirmed_tokens`` in ``SimpleCPUOffloadScheduler``), so
        # those blocks have long been written and visible across streams
        # by the time we read them here. Loads target GPU blocks held
        # by ``BlockPool.touch`` until load completes, so they are also
        # safe to write without a barrier.
        assert self._device is not None
        assert self._queue is not None
        assert self._load_stream is not None
        assert self._store_stream is not None
        torch.npu.set_device(self._device)

        while True:
            item = self._queue.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
            ) = item

            stream = self._store_stream if is_store else self._load_stream
            with torch.npu.stream(stream):
                copy_blocks(src_blocks, dst_blocks, params)
                event = torch.npu.Event()
                event.record(stream)
            events_list.append((event_idx, event))
